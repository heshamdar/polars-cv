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

This project installs **editable**: `.venv` carries a `.pth` pointing at
`python/`, and `maturin develop` writes `_lib.abi3.so` into `python/polars_cv/`.
So Python edits take effect immediately, but **the compiled extension does not** —
after pulling commits that touch Rust, the `.so` stays at its build-time version
until you re-run `maturin develop`, and you are testing old Rust against new
Python. `polars_cv.build_info()` reports the three versions that must agree
(`__version__`, the compiled plugin, the installed distribution), and
`tests/test_version_consistency.py` fails when they do not.

### Test

```bash
uv run pytest tests/                            # Full suite; plugin tests self-skip if unbuilt
uv run pytest tests/test_pipeline_builder.py   # Single test file
uv run pytest tests/ -k "test_resize"          # Single test by name
python scripts/test_multiple_python.py --all   # Test across Python 3.10–3.13
```

Tests are marked with `network` (needs network access) and `slow` (excluded from
the default lane). CI runs `pytest -m "not network and not slow"` on every push
and a separate `-m "slow and not network"` lane on a schedule.

Rust unit tests (run from the workspace root or with `-p` flag):
```bash
cargo test -p view-buffer --all-features   # view-buffer engine tests
cargo test -p polars-cv                    # Rust plugin tests
```

### Lint & Format

```bash
uvx ruff check python tests benchmarks         # Python lint (matches CI)
uvx ruff format python tests benchmarks        # Python format
cargo fmt --all -- --check       # Rust format check
cargo clippy --all-targets --all-features -- -D warnings  # Rust lint
```

A [pre-commit](https://pre-commit.com/) config (`.pre-commit-config.yaml`) wires
these up; `pre-commit>=4.5.1` is in the dev group. Install hooks with
`uv run pre-commit install`.

### Docs

```bash
uv sync --group docs             # Install docs dependencies
uv run mkdocs serve              # Live-preview the MkDocs site locally
uv run mkdocs build --strict     # Build the site (fails on broken links/nav)
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
| `expressions.py` | `CvNamespace` — the `.cv` accessor registered on Polars expressions (`.pipe()`, `.read_bytes()`, header-only metadata) |
| `_types.py` | Core type definitions: `OpSpec`, `ParamValue`, `SourceSpec`, `SinkSpec`, `Domain`, `DType` |
| `_graph.py` | `PipelineGraph` / `GraphNode` — DAG construction, JSON serialization, CSE, plugin registration |
| `_namespace.py` | Shared base for the `.cv`/`.point`/`.contour`/`.bbox` expression namespaces (plugin-registration boilerplate) |
| `display.py` | `show_images()` — notebook rendering of image columns |
| `_graph_viz.py` | Graph visualization (networkx/graphviz/pydot) |
| `geometry/` | Point/contour/bbox schemas and Polars expression namespaces |
| `metrics/` | Detection metrics (PR curves, AP, FROC, LROC, bootstrap, AUC) |

### Key Rust Modules

**polars-cv/src/**
- `lib.rs` — PyO3 module entry, `vb_graph` polars expression function, dtype inference, and the `op_schema`/`op_contract`/`op_output_dtype`/`enum_variants`/`known_ops` FFI the Python planner reads
- `execute.rs` — `resolve_op()` dispatcher mapping `OpSpec`s to `GraphStep`s (`graph/step.rs`: buffer ops wrap view-buffer's `ViewDto`; graph-only steps are their own variants); owns the `KNOWN_OPS` registry
- `graph/` — `UnifiedGraph` execution engine: `types.rs` (`UnifiedGraph`, `GraphNode`, `OutputSpec`, `RowErrorPolicy`), `compiled.rs` (process-wide compiled-graph cache), `step.rs` (`GraphStep` — the plugin-level step vocabulary), source decoding (`decode.rs`), sink encoding (`encode.rs`)
- `params.rs` — `ParamValue` resolving literals vs per-row Polars column values
- `pipeline.rs` — serde types for the JSON graph spec crossing the plugin boundary
- `cloud.rs` — remote/cloud transport (`object_store` backends, `cloud_options`, bounded-concurrency reads)
- `fetch.rs` — stage one of every path-based read: path column → bytes (`prefetch`, `row_bytes`, `parse_on_error`), shared by the `file_path` source and `read_bytes.rs`; owns the path-sandboxing TODO
- `read_bytes.rs` — `read_file_bytes` plugin function (`.cv.read_bytes()`) — `fetch.rs` with the decode omitted, for byte-identical passthrough
- `image_metadata.rs` — header-only metadata plugin functions (`.cv.width()`/`height()`/`channels()`/`image_dtype()`)
- `output.rs` — zero-copy numpy/torch struct output encoding
- `engine_warning.rs` — one-time single-threaded-batch warning (points users to `engine="streaming"`)
- `contour.rs`, `point.rs` — standalone plugin functions for geometry namespaces
- `geom_params.rs` — `GeomParams`: per-row parameter resolution for those standalone functions, reading expression params off the extra inputs the Python `_ArgBinder` appends and names in `input_slots`

**view-buffer/src/** (see `view-buffer/AGENTS.md` for the full module tree)
- `core/` — `ViewBuffer` (strided N-D array), `DType`, `Layout`
- `ops/` — operation definitions by category (`image.rs`, `color.rs`, `compute.rs`, `scalar.rs`, `filter.rs`, `affine.rs`, `view.rs`, `binary.rs`, `reduction.rs`, `histogram.rs`, `phash.rs`, `pad.rs`, `mask.rs`), plus `shape_rule.rs` (the plan-time rank/channel authority), `validation.rs`, `traits.rs`, `util.rs`
- `ops/dto.rs` — `ViewDto` enum: the serializable bridge between JSON and Rust op code
- `expr.rs` — `ViewExpr` lazy builder with `.plan()` / `.execute()`
- `execution/` — `ExecutionPlan`, runner, kernel fusion
- `geometry/` — contour extraction, rasterization, measures, pairwise matching, transforms
- `interop/` — zero-copy Arrow, ndarray, `image`, and Polars-arrow integration
- `protocol.rs` — VIEW binary protocol (header + data serialization)

---

## Key Conventions

### Domain System

Every `Pipeline` tracks a **domain** through operations:

| Domain | What it holds | Produced by |
|--------|---------------|-------------|
| `buffer` | Multi-dimensional array | `source()` (defaults to `"auto"`; also `"image_bytes"`, `"file_path"`, …) |
| `contour` | Geometry vectors | `extract_contours()` |
| `scalar` | Single numeric value | `reduce_sum()` |
| `vector` | 1-D numeric array (incl. histogram buckets) | `perceptual_hash()`, `histogram()` |

Domain constraints are enforced at pipeline-build time. Operations that don't match the current domain raise immediately in Python, not at execution time.

### Parameter Values

Most operation parameters accept either a literal (`224`) or a Polars expression (`pl.col("target_height")`). This is typed as `ParamValue` in `_types.py`. Per-row expression params are resolved in Rust via `params.rs` (`resolve_*` for numbers, `resolve_str`/`resolve_bool` for enums and flags), and by `geom_params.rs` for the `.contour`/`.point`/`.bbox` namespaces, which bypass `vb_graph` and carry their expression params as extra plugin inputs recorded in an `input_slots` name→index map.

The rule for whether a parameter may be per-row is *not* its type: **a parameter is eligible iff its value has no effect on the output shape, rank or dtype**, because the lazy schema is computed at plan time and must match what executes. So non-structural enums and flags (`filter`, `interpolation`, `pad(mode=)`, `convolve2d(border=, normalize=)`, …) are per-row, while structural parameters are literal-only: `cast(dtype=)`, `normalize(method=, out_dtype=)`, reduction `axis`, `perceptual_hash(hash_size=, algorithm=)`, `rotate(expand=)`, `histogram(closed=, output=)`, the `transpose`/`flip` axis lists and `reshape`'s element count. For a list-valued parameter the *length* is structural while the elements are not — a `convolve2d` kernel keeps a literal element count but each coefficient may be an expression. Plan-time shape probing binds every expression param to an integer placeholder, so the enum/flag accessors substitute their default under `ParamCtx::probe`; that substitution is sound only because of the eligibility rule.

A parameter column may contain **nulls**. `Pipeline.on_null_param("raise"|"null")` (and `on_null(...)` on the geometry accessors) chooses between failing the query and nulling just the affected rows. This is one shared mechanism, never per-op handling: a `NullParamPolicy` rides on `ParamCtx` and every null reaches `ParamCol::on_null`, which flags the context so `graph/compiled.rs` skips the node for that row — reusing the same null-propagation path a null input image already takes, so nulling is node-scoped rather than row-scoped. Do not add per-op or per-parameter null keywords: a fallback value is already `pl.col("h").fill_null(224)`, and a per-parameter policy would have to enter the `ParamValue` wire format and its `__eq__`/`__hash__` (or CSE would merge ops differing only in policy).

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
- `test_contour_raster_crosscheck.py` checks the analytic contour measures
  (`area`, `centroid`, `iou`, `dice`, `contains_point`) against pixel counts on a
  rasterized mask — two independent implementations of the same quantity, so a
  fault in either shows up as a mismatch. Contours whose vertices are all
  integers on axis-aligned edges put no pixel centre on an edge, so those cases
  assert *exact* equality; diagonal and curved shapes assert a tolerance scaled
  by perimeter, since discretization error tracks boundary length, not area.
  Extend the `RECTILINEAR` / `CURVED` shape tables rather than adding one-off
  tests, and keep new rectilinear shapes on integer coordinates so they stay
  exact.

### Adding a New Operation

1. Implement in **view-buffer** (`view-buffer/src/ops/`) if it is a buffer→buffer
   engine op — add to the appropriate module, give it truthful `Op` contracts
   (shape/dtype/domain/channel rules), and register it in `ViewDto`
   (`tests/apply_op_coverage.rs` requires a probe per variant). Graph-level
   steps (node references, non-buffer outputs) become `GraphStep` variants in
   `polars-cv/src/graph/step.rs` instead.
2. Add a dispatch arm in **polars-cv** `src/execute.rs` (`resolve_op()`),
   returning the `GraphStep`. The Python planner picks up the op's schema
   effect automatically through the `op_schema` FFI — no Python-side schema
   special cases.
3. Add a method to `Pipeline` in `python/polars_cv/pipeline.py`. The matching
   `LazyPipelineExpr` method is generated automatically from `Pipeline` at import
   time (`python/polars_cv/lazy.py`) — do **not** hand-mirror it. If the op needs
   bespoke lazy behaviour (e.g. it takes another `LazyPipelineExpr` operand),
   define it explicitly on `LazyPipelineExpr` and the generator will skip it.
4. Regenerate the type stub: `python scripts/gen_lazy_stub.py` (CI guards it via
   `test_lazy_stub_is_current`).
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
