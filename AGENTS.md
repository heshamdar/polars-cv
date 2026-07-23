# AGENTS.md — polars-cv

> **Read this file first when starting any task in this repository.**
> After making changes, update any relevant AGENTS.md file.

## Quick Navigation

| File | Scope |
|------|-------|
| [`polars-cv/python/polars_cv/AGENTS.md`](polars-cv/python/polars_cv/AGENTS.md) | Python API — pipeline builder, lazy composition, graph serialization, schema inference |
| [`polars-cv/python/polars_cv/geometry/AGENTS.md`](polars-cv/python/polars_cv/geometry/AGENTS.md) | Geometry subsystem — point/contour/bbox schemas and namespaces |
| [`polars-cv/python/polars_cv/metrics/AGENTS.md`](polars-cv/python/polars_cv/metrics/AGENTS.md) | Detection metrics — matchers, FROC/LROC/PR curves, bootstrap |
| [`polars-cv/src/AGENTS.md`](polars-cv/src/AGENTS.md) | Rust plugin — graph execution, source/sink encoding, op dispatch |
| [`view-buffer/AGENTS.md`](view-buffer/AGENTS.md) | Core tensor engine — ViewBuffer, ViewExpr, operations, execution |
| [`polars-cv/tests/AGENTS.md`](polars-cv/tests/AGENTS.md) | Testing conventions and fixtures |
| [`polars-cv/benchmarks/AGENTS.md`](polars-cv/benchmarks/AGENTS.md) | Benchmark framework |

## What Is polars-cv?

A **Polars plugin for high-performance vision and array operations**. Users define pipelines that operate directly on DataFrame columns instead of orchestrating OpenCV, PIL, NumPy, etc.

```python
import polars as pl
from polars_cv import Pipeline

pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
pipe = Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3).dilate(ksize=3)
df.with_columns(processed=pl.col("image").cv.pipe(pipe).sink("numpy"))
```

This is a **pre-release, largely AI-developed project**. Fix inconsistencies when you encounter them, or document them in the relevant AGENTS.md.

## Guiding Principles

1. **Zero-copy by default.** Prefer view operations (transpose, crop, flip) that only modify metadata (strides/offsets) over operations that allocate new memory. Materialization should be explicit and deliberate.

2. **Lazy evaluation.** Follow Polars' paradigm: build the plan at planning time, execute at execution time. The Python layer builds pipeline specifications; the Rust layer executes them. No computation should happen in Python at pipeline construction time.

3. **Explicit over implicit.** No hidden assumptions. If a dtype is needed, the user must specify it or the system must infer it deterministically. Planning-time schema (`collect_schema()`) must match execution-time schema — never assume at planning time something that could differ at execution time.

4. **Strong contracts at planning time.** Catch errors early. Types, domains, shapes should be validated when the pipeline is built, not when it runs. When something is unknowable at planning time (e.g., dtype from decoded image bytes), mark it as `auto` and surface the implications to the user.

5. **Composition with low coupling.** Pipeline operations are independent, composable units. `LazyPipelineExpr` allows chaining, merging, and multi-output pipelines without operations knowing about each other.

6. **Python for planning, Rust for execution.** Keep Python focused on pipeline specification, validation, and graph construction. Reserve Rust for performance-critical execution-time work. If something can be done purely in Python (schema inference, validation, utilities), do it there.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Python: polars_cv                                      │
│  Pipeline builder, expression namespaces, graph         │
│  construction, schema inference, validation             │
├─────────────────────────────────────────────────────────┤
│  Rust: polars-cv (the plugin)                           │
│  Graph execution, source decoding, sink encoding,       │
│  parameter resolution, cloud I/O                        │
├─────────────────────────────────────────────────────────┤
│  Rust: view-buffer (the engine)                         │
│  ViewBuffer, ViewExpr, stride-aware ops, kernel fusion, │
│  zero-copy interop (Arrow, ndarray, image)              │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

```
Python Pipeline spec
  → JSON graph serialization (PipelineGraph)
  → register_plugin_function("vb_graph", graph_json, expr_column_names)
  → Polars calls Rust vb_graph(inputs, kwargs)
  → UnifiedGraph::from_json() → topological execution
  → Per-row: decode source → apply ops (ViewExpr/ViewBuffer) → encode sink
  → Returns Series (Binary, Float64, Struct, List, Array)
```

### Domain System

Pipelines track data domain through operations:

| Domain | Description | Example |
|--------|-------------|---------|
| `buffer` | Multi-dimensional arrays (images) | After `source("image_bytes")` |
| `contour` | Geometry (vectors of points) | After `extract_contours()` |
| `scalar` | Single numeric values | After `reduce_sum()` |
| `vector` | Fixed-length numeric arrays, incl. histogram buckets | After `perceptual_hash()`, `bounding_box()`, `histogram()` |

### Alpha Channel Handling

Alpha channels are **always preserved** during image decoding (RGBA → 4ch, GrayA → 2ch). How each operation treats channels (and therefore alpha) is declared by its `OutputChannelRule` in view-buffer (`view-buffer/src/ops/shape_rule.rs`), which the Python planner reads via `channel_rule`:

- **`PreserveChannels`** — all channels processed uniformly (resize, normalize, flip, etc.)
- **`StripProcessRestore { color_channels }`** — alpha separated, op applied to color channels, alpha restored (blur, cvt_color, sobel)
- **`Fixed(n)`** — alpha discarded, output channels fixed by the op (grayscale → 1, canny → 1)
- **`NotApplicable` / `Unknown`** — non-image-buffer ops (reductions, geometry) or not knowable at plan time

Image sources have unknown channel count at planning time. Users can assert known channels with `.assert_shape(channels=4)`.

### Expression Namespaces

| Namespace | Purpose |
|-----------|---------|
| `.cv` | Image/array pipelines via `.pipe()` → `.sink()`, metadata (`.width()`, `.height()`, `.channels()`, `.image_dtype()`) |
| `.point` | Point geometry ops (normalize, distance, etc.) |
| `.contour` | Contour geometry ops (area, perimeter, IoU, matching) |
| `.bbox` | Bounding box ops (pairwise IoU, match detections) |

## Directory Structure

```
polars-cv/                          # Workspace root
├── AGENTS.md                       # ← You are here
├── Cargo.toml                      # Workspace Cargo.toml
├── CONTRIBUTING.md                 # Release process, CI
├── view-buffer/                    # Rust tensor engine
│   ├── AGENTS.md
│   └── src/
├── polars-cv/                      # Main package (Rust + Python)
│   ├── Cargo.toml
│   ├── pyproject.toml
│   ├── src/                        # Rust plugin source
│   │   └── AGENTS.md
│   ├── python/polars_cv/           # Python package
│   │   ├── AGENTS.md
│   │   ├── geometry/AGENTS.md
│   │   └── metrics/AGENTS.md
│   ├── tests/                      # Python tests (pytest)
│   │   └── AGENTS.md
│   ├── benchmarks/                 # Benchmark suite
│   │   └── AGENTS.md
│   ├── examples/                   # Runnable demos (13 numbered 01–13 + detection_data.py)
│   ├── docs/                       # MkDocs user-guide documentation
│   └── scripts/                    # Utility scripts (gen_lazy_stub.py, test_multiple_python.py)
└── CHANGELOG.md                    # Keep-a-Changelog release history
```

See root [`CONTRIBUTING.md`](CONTRIBUTING.md) for the release process and CI, and
[`README.docker.md`](README.docker.md) for the Docker build environment.

## Task Routing

| If you're... | Read these AGENTS.md files |
|---|---|
| Adding a new image operation | Root → Python API → Rust Plugin → view-buffer |
| Working on pipeline builder or lazy composition | Root → Python API |
| Working on detection metrics | Root → Python API → Metrics |
| Working on geometry (points, contours) | Root → Geometry |
| Working on graph execution or sources/sinks | Root → Rust Plugin |
| Working on view-buffer ops or ViewExpr | Root → view-buffer |
| Writing or fixing tests | Root → Tests |
| Working on benchmarks | Root → Benchmarks |
| Fixing schema inference or dtype contracts | Root → Python API → Rust Plugin |

## Build and Development

```bash
cd polars-cv
maturin develop --release          # Build Rust plugin into .venv
uv run pytest tests/               # Run tests
uv run ruff check python/ tests/   # Lint Python
cargo clippy --workspace           # Lint Rust
```

## Known Issues

- **f64 chains stay unfused:** the FusedKernel computes in f32, so the float-promoting scalar family is correct-but-unfused for f64 inputs (`view-buffer/src/expr.rs::extract_ops`).

## Release History

Per-release changes (added/changed/fixed/performance) are tracked in
[`CHANGELOG.md`](CHANGELOG.md) — consult it rather than duplicating a running log
here.

## Durable Architecture Notes

These are the load-bearing design decisions worth internalizing before making
changes; they explain *why* the code is shaped the way it is.

- **Single schema authority (view-buffer).** Each op's schema effect — output
  domain, dtype, rank, and channel count — is declared once, on the op itself in
  Rust (`OutputRankRule`/`OutputChannelRule` in
  `view-buffer/src/ops/shape_rule.rs`), and read by the Python planner through the
  `op_schema`/`op_contract`/`op_output_dtype` FFI. The planner contains no per-op
  special cases and no parallel contract table. Planning-time schema must equal
  execution-time schema; guarded by `tests/test_sanitation.py`.
- **Graph steps vs engine ops.** Graph-level steps (`GraphStep` in
  `polars-cv/src/graph/step.rs`: binary ops, masks, geometry, reductions,
  histograms, perceptual hash) are separate from engine-executable buffer ops
  (`ViewDto` in view-buffer). Anything that changes the data domain lives in
  `GraphStep`.
- **Compiled-graph cache.** The `vb_graph` plugin compiles each graph once into a
  process-wide cache (`graph/compiled.rs`: parsed spec, topo order, slot-bound
  params, pre-resolved static ops). The streaming engine invokes the plugin per
  morsel, so repeat calls pay only a hash lookup, and graph structure is validated
  at compile time instead of failing late.
- **Kernel fusion.** Consecutive scalar compute ops (scale/relu/clamp/gamma/invert)
  plus casts fold into one `FusedKernel` pass (any-numeric read → f32 ops →
  out-dtype write). `out_dtype` is pinned to what the unfused chain would produce,
  so fusion never changes the planned schema. f64 promote-family inputs stay
  unfused (see Known Issues).
- **Rotation/affine unification.** `rotate()` with arbitrary angles routes through
  `ComputeOp::RotateAffine` → `AffineParams::from_rotation()` → `apply_affine_warp()`,
  sharing the affine code path; 90/180/270 stay zero-copy via `ViewOp`. Consecutive
  affine ops fuse into a single matrix at planning time.
- **Lazy parity.** `LazyPipelineExpr` generates a forwarder for every chainable
  `Pipeline` method at import time (drift-guarded by
  `test_lazy_pipeline_method_parity`); the type stub is regenerated via
  `scripts/gen_lazy_stub.py` and guarded by `test_lazy_stub_is_current`.
