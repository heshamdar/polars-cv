# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**polars-cv** is a Polars plugin for high-performance vision and array operations. It lets users build image-processing pipelines that run as Polars expressions over DataFrame columns — zero-copy by default, lazy evaluation, explicit over implicit.

```python
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
df.with_columns(processed=pl.col("image").cv.pipe(pipe).sink("numpy"))
```

The project is a Rust/Python hybrid built with [Maturin](https://github.com/PyO3/maturin) (PyO3). The Python package lives under `polars-cv/`, and `view-buffer/` is a separate Rust crate that is the core tensor engine.

---

## Commands

All commands should be run from the `polars-cv/` subdirectory unless noted otherwise.

### Build

```bash
uv sync --group dev              # Install Python dev dependencies
maturin develop --release        # Compile Rust plugin and install into .venv
maturin build --release          # Build distributable wheels
```

### Test

```bash
uv run pytest tests/                            # Full test suite (requires plugin built first)
uv run pytest tests/ -k "not plugin_required"  # Schema/builder tests only (no Rust needed)
uv run pytest tests/test_pipeline_builder.py   # Single test file
uv run pytest tests/ -k "test_resize"          # Single test by name
python scripts/test_multiple_python.py --all   # Test across Python 3.10–3.13
```

Rust unit tests (run from the workspace root or with `-p` flag):
```bash
cargo test -p view-buffer --all-features   # view-buffer engine tests
cargo test -p polars-cv                    # Rust plugin tests
```

### Lint & Format

```bash
uv run ruff check python/        # Python lint
uv run ruff format python/       # Python format
cargo fmt --all -- --check       # Rust format check
cargo clippy --all-targets --all-features -- -D warnings  # Rust lint
```

---

## Architecture

The project has three layers:

```
Python: polars_cv
  Pipeline builder, expression namespaces, DAG construction,
  schema inference, lazy composition, geometry/metrics APIs
        ↕ JSON graph serialization
Rust: polars-cv (the plugin)
  vb_graph expression entry point, graph execution, source
  decoding, sink encoding, per-row parameter resolution, cloud I/O
        ↕ Rust crate dependency
Rust: view-buffer (the engine)
  ViewBuffer, ViewExpr, stride-aware operations, kernel fusion,
  zero-copy interop with Arrow/ndarray
```

### Data Flow

1. User builds a `Pipeline` in Python → internally creates a `PipelineGraph` (DAG of `GraphNode`s).
2. `.sink(...)` on a `LazyPipelineExpr` serializes the graph to JSON and calls `register_plugin_function("vb_graph", ...)`.
3. Polars calls the Rust `vb_graph` expression function with the JSON and any per-row expression parameters.
4. Rust deserializes into a `UnifiedGraph` and compiles it once into a process-wide cache (`graph/compiled.rs`: parsed spec, topological order, slot-bound params); repeat calls (e.g. per streaming morsel) pay only a hash lookup. It then executes topologically per-row: decode source → apply operations → encode sink.
5. Returns a Polars `Series` (dtype depends on sink: Binary, Float64, Struct, List, Array).

### Key Python Modules (`polars-cv/python/polars_cv/`)

| File | Role |
|------|------|
| `pipeline.py` | `Pipeline` builder — all image/array operations as chainable methods |
| `lazy.py` | `LazyPipelineExpr` — lazy `.pipe()`, `.merge_pipe()`, `.sink()`, binary ops |
| `expressions.py` | `CvNamespace` — the `.cv` accessor registered on Polars Series/Expr |
| `_types.py` | Core type definitions: `OpSpec`, `ParamValue`, `SourceSpec`, `SinkSpec`, `Domain`, `DType` |
| `_graph.py` | `PipelineGraph` / `GraphNode` — DAG construction, JSON serialization, CSE, plugin registration |
| `geometry/` | Point/contour/bbox schemas and Polars expression namespaces |
| `metrics/` | Detection metrics (PR curves, AP, FROC, LROC, bootstrap, AUC) |

### Key Rust Modules

**polars-cv/src/**
- `lib.rs` — PyO3 module entry, `vb_graph` polars expression function, dtype inference, and the `op_contract`/`op_output_dtype`/`enum_variants`/`known_ops` FFI the Python planner reads
- `execute.rs` — `resolve_op()` dispatcher mapping `OpSpec` variants to view-buffer calls; owns the `KNOWN_OPS` registry
- `graph/` — `UnifiedGraph` execution engine: `types.rs` (`UnifiedGraph`, `GraphNode`, `OutputSpec`, `RowErrorPolicy`), `compiled.rs` (process-wide compiled-graph cache), source decoding (`decode.rs`), sink encoding (`encode.rs`)
- `params.rs` — `ParamValue` resolving literals vs per-row Polars column values
- `pipeline.rs` — serde types for the JSON graph spec crossing the plugin boundary
- `cloud.rs` — remote/cloud source I/O (`file_path` decode, `cloud_options`, concurrent prefetch)
- `image_metadata.rs` — header-only metadata plugin functions (`.cv.width()`/`height()`/`channels()`/`image_dtype()`)
- `output.rs` — zero-copy numpy/torch struct output encoding
- `contour.rs`, `point.rs` — standalone plugin functions for geometry namespaces

**view-buffer/src/**
- `core/` — `ViewBuffer` (strided N-D array), `DType`, `Layout`
- `ops/` — operation definitions organized by category (`image.rs`, `color.rs`, `compute.rs`, `filter.rs`, `view.rs`, `binary.rs`, `reduction.rs`, `histogram.rs`)
- `ops/dto.rs` — `ViewDto` enum: the serializable bridge between JSON and Rust op code
- `expr.rs` — `ViewExpr` lazy builder with `.plan()` / `.execute()`

---

## Key Conventions

### Domain System

Every `Pipeline` tracks a **domain** through operations:

| Domain | What it holds | Produced by |
|--------|---------------|-------------|
| `buffer` | Multi-dimensional array | `source("image_bytes")` |
| `contour` | Geometry vectors | `extract_contours()` |
| `scalar` | Single numeric value | `reduce_sum()` |
| `vector` | 1-D numeric array (incl. histogram buckets) | `perceptual_hash()`, `histogram()` |

Domain constraints are enforced at pipeline-build time. Operations that don't match the current domain raise immediately in Python, not at execution time.

### Parameter Values

Most numeric operation parameters accept either a literal (`224`) or a Polars expression (`pl.col("target_height")`). This is typed as `ParamValue` in `_types.py`. Per-row expression params are resolved in Rust via `params.rs`. Structural parameters (enum tags, kernel shapes, axis lists) are literals only.

### `Pipeline` Is Immutable

Every operation on `Pipeline` returns a new clone. Do not mutate an existing pipeline in place.

### Alpha Channel Handling

Image sources always preserve alpha. How each operation treats channels (and
therefore alpha) is declared by its `OutputChannelRule` in
`view-buffer/src/ops/shape_rule.rs`, the single authority the Python planner
reads via `channel_rule`:
- `PreserveChannels` — channel count is unchanged (alpha passes through).
- `StripProcessRestore { color_channels }` — alpha is split off, the op runs on
  the color channels, then alpha is re-attached (e.g. `RGBA`→gray yields `GrayA`).
- `Fixed(n)` — output has exactly `n` channels regardless of input (e.g.
  `grayscale`/`canny` → 1), dropping any alpha.
- `NotApplicable` / `Unknown` — no `[H, W, C]` image result, or not knowable at
  plan time.

### Test Structure

- Tests requiring the compiled Rust plugin are decorated with `@plugin_required` (class decorator) or use the `plugin_required` fixture from `conftest.py`.
- Unit tests for pure Python (schema inference, builder validation) live in files like `test_pipeline_builder.py` and `test_lazy_schema.py` and require no compiled plugin.
- Integration tests and reference tests (comparing output against NumPy/OpenCV ground truth) are in separate files under `tests/reference/`.
- Reuse `conftest.py` fixtures (`create_test_png`, `sample_image_bytes`, etc.) rather than redefining helpers per file.

### Adding a New Operation

1. Implement in **view-buffer** (`view-buffer/src/ops/`) — add to appropriate module and register in `ViewDto`.
2. Add a dispatch arm in **polars-cv** `src/execute.rs` (`resolve_op()`).
3. Add a method to `Pipeline` in `python/polars_cv/pipeline.py`.
4. Mirror the method on `LazyPipelineExpr` in `python/polars_cv/lazy.py`.
5. Write tests covering both unit (builder validation) and integration (actual execution) cases.

---

## Detailed Context

Subsystem-specific AGENTS.md files provide deeper guidance:

- `AGENTS.md` — root navigation, cross-cutting architecture decisions
- `polars-cv/python/polars_cv/AGENTS.md` — Python API internals
- `polars-cv/src/AGENTS.md` — Rust plugin internals
- `view-buffer/AGENTS.md` — view-buffer engine internals
- `polars-cv/tests/AGENTS.md` — test conventions and fixture patterns
- `polars-cv/python/polars_cv/geometry/AGENTS.md` — geometry subsystem
- `polars-cv/python/polars_cv/metrics/AGENTS.md` — metrics subsystem
- `polars-cv/benchmarks/AGENTS.md` — benchmark framework

## Known Limitations

- f64 inputs through the float-promoting scalar ops execute correctly (in f64) but are excluded from kernel fusion, which computes in f32.
