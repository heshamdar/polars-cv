# AGENTS.md — polars-cv

> **Read this file first when starting any task in this repository.**
> After making changes, update any relevant AGENTS.md file (this one or subdirectory-specific ones) to reflect what changed.

## What Is polars-cv?

polars-cv is a **Polars plugin for high-performance vision and array operations**. It gives developers and ML practitioners who work with both structured (tabular) data and image/image-like data a single, cohesive tool — instead of orchestrating multiple libraries (OpenCV, PIL, NumPy, etc.), users define pipelines that operate directly on Polars DataFrame columns.

```python
import polars as pl
from polars_cv import Pipeline

pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
df.with_columns(processed=pl.col("image").cv.pipe(pipe).sink("numpy"))
```

This is a **largely AI-developed, pre-release project**. Expect inconsistencies and areas that need consolidation. When you encounter them, fix them or document them in the relevant AGENTS.md.

## Guiding Principles

These principles should inform every decision when working in this codebase:

1. **Zero-copy by default.** Prefer view operations (transpose, crop, flip) that only modify metadata (strides/offsets) over operations that allocate new memory. Materialization should be explicit and deliberate.

2. **Lazy evaluation.** Follow Polars' paradigm: build the plan at planning time, execute at execution time. The Python layer builds pipeline specifications; the Rust layer executes them. No computation should happen in Python at pipeline construction time.

3. **Explicit over implicit.** No hidden assumptions. If a dtype is needed, the user must specify it or the system must be able to infer it deterministically. Planning-time schema (`collect_schema()`) must match execution-time schema — never assume at planning time something that could differ at execution time.

4. **Strong contracts at planning time.** Use contract-based assertions to catch errors early. Types, domains, shapes should be validated when the pipeline is built, not when it runs. When something is unknowable at planning time (e.g., dtype from decoded image bytes), it must be explicitly marked as `auto` and the implications surfaced to the user (e.g., "you must specify dtype for list/array sinks").

5. **Composition with low coupling.** Pipeline operations should be independent, composable units. The `LazyPipelineExpr` allows chaining, merging, and multi-output pipelines without operations knowing about each other.

6. **Python for planning, Rust for execution.** Keep the Python layer focused on pipeline specification, validation, and graph construction. Reserve Rust for performance-critical execution-time work. If something can be done purely in Python (e.g., schema inference, validation, new utility functions), do it there.

## Architecture Overview

### Two-Tier Rust Architecture

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

- **view-buffer** (`view-buffer/`): Low-level zero-copy tensor framework. Defines `ViewBuffer`, `ViewExpr`, `ExecutionPlan`. Handles the actual compute. While originally intended to be a fully independent crate, it is currently tightly coupled to polars-cv in practice. Agents working on either layer typically need context from both. See [`view-buffer/AGENTS.md`](view-buffer/AGENTS.md).

- **polars-cv Rust plugin** (`polars-cv/src/`): Bridges view-buffer with Polars via pyo3/pyo3-polars. Handles graph JSON parsing, source decoding, sink encoding, parameter resolution, cloud storage. See [`polars-cv/src/AGENTS.md`](polars-cv/src/AGENTS.md).

- **polars-cv Python package** (`polars-cv/python/polars_cv/`): The user-facing API. Pipeline builder, expression namespaces (`.cv`, `.point`, `.contour`), lazy composition, graph serialization. See [`polars-cv/python/polars_cv/AGENTS.md`](polars-cv/python/polars_cv/AGENTS.md).

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
| `vector` | Multiple numeric values | After `perceptual_hash()`, `bounding_box()`, `histogram(output="counts")` |
| `histogram` | Histogram buckets (extents and counts) | After `histogram(output="buckets")` |

### Expression Namespaces

| Namespace | Registration | Purpose |
|-----------|-------------|---------|
| `.cv` | `CvNamespace` | Image/array pipelines via `.pipe()` → `.sink()` (includes buffer-space `label_reduce(contours=...)`) |
| `.point` | `PointNamespace` | Point geometry ops (normalize, distance, etc.) |
| `.contour` | `ContourNamespace` | Contour geometry ops (area, perimeter, IoU, etc.) |
| `.bbox` | `BBoxNamespace` | Bounding box ops (pairwise IoU, match detections) |

### Metrics Subsystem

The `metrics/` subpackage provides detection evaluation built from polars-cv
primitives and Polars lazy expressions:

```
Input Data → Matcher → DetectionTable → Metric Function → MetricResult
```

- **Matchers**: `ContourMatcher` (heatmap/mask), `BBoxMatcher` (bounding boxes), `PreMatchedAdapter` (pre-computed TP/FP)
- **Metrics**: `precision_recall_curve`, `average_precision`, `mean_average_precision`, `froc_curve`, `lroc_curve`, `confusion_at_threshold`, `precision_at_threshold`, `recall_at_threshold`, `f1_at_threshold`
- **Bootstrap**: `bootstrap_metric_sequential` (general), `bootstrap_pr_auc` (vectorized)

See `polars-cv/python/polars_cv/metrics/AGENTS.md` for full architecture details.

## Directory Structure

```
polars-cv/                          # Workspace root (this file)
├── AGENTS.md                       # ← You are here
├── Cargo.toml                      # Workspace Cargo.toml (members: view-buffer, polars-cv)
├── CONTRIBUTING.md                 # Release process, CI, multi-Python testing
├── view-buffer/                    # Low-level Rust tensor engine
│   ├── AGENTS.md                   # Agent guide for view-buffer
│   └── src/                        # Rust source
├── polars-cv/                      # Main package (Rust + Python)
│   ├── Cargo.toml                  # Crate config (cdylib, pyo3, polars)
│   ├── pyproject.toml              # Python build config (maturin)
│   ├── src/                        # Rust plugin source
│   │   └── AGENTS.md               # Agent guide for Rust plugin
│   ├── python/polars_cv/           # Python package
│   │   ├── AGENTS.md               # Agent guide for Python API
│   │   └── geometry/
│   │       └── AGENTS.md           # Agent guide for geometry subsystem
│   ├── tests/                      # Python tests (pytest)
│   │   └── AGENTS.md               # Agent guide for tests
│   ├── benchmarks/                 # Benchmark suite
│   │   └── AGENTS.md               # Agent guide for benchmarks
│   ├── examples/                   # Runnable Python demos (focused feature examples, each <500 lines)
│   ├── notebooks/                  # Legacy notebook area (examples now live in `examples/`)
│   ├── docs/                       # MkDocs user-guide documentation
│   └── scripts/                    # Utility scripts (multi-Python testing)
└── .cursor/
    └── polars-cv-contribution-guide.md  # Detailed op-addition walkthrough
```

## How to Route Your Task

| If you're... | Read these AGENTS.md files |
|---|---|
| Adding a new image operation | Root → `polars-cv/python/polars_cv/AGENTS.md` → `polars-cv/src/AGENTS.md` → `view-buffer/AGENTS.md` → `.cursor/polars-cv-contribution-guide.md` |
| Working on pipeline builder or lazy composition | Root → `polars-cv/python/polars_cv/AGENTS.md` |
| Working on detection metrics (FROC, LROC, mAP, PR, matchers) | Root → `polars-cv/python/polars_cv/AGENTS.md` → `polars-cv/python/polars_cv/metrics/AGENTS.md` |
| Working on geometry (points, contours) | Root → `polars-cv/python/polars_cv/geometry/AGENTS.md` |
| Working on graph execution or sources/sinks | Root → `polars-cv/src/AGENTS.md` |
| Working on view-buffer ops or ViewExpr | Root → `view-buffer/AGENTS.md` |
| Writing or fixing tests | Root → `polars-cv/tests/AGENTS.md` |
| Working on benchmarks | Root → `polars-cv/benchmarks/AGENTS.md` |
| Fixing schema inference or dtype contracts | Root → `polars-cv/python/polars_cv/AGENTS.md` → `polars-cv/src/AGENTS.md` |

## Build and Development

```bash
# Build the Rust plugin (required for plugin tests)
cd polars-cv
maturin develop --release

# Run tests (from polars-cv/ directory)
cd polars-cv
uv run pytest tests/

# Use the local .venv
source polars-cv/.venv/bin/activate

# Lint
uv run ruff check polars-cv/python/ polars-cv/tests/
cargo clippy --workspace
```

## Known Issues and Technical Debt

These are known problems. When you encounter them, fix them if in scope, or note them here if not.

### Legacy Pipeline Path (Partially Removed)

`CvNamespace.pipeline()`, `apply_pipeline()`, and `Pipeline.sink()` have been removed from Python.
The legacy Rust row-by-row executor in `execute.rs` was also removed; the module now only exposes shared graph helpers (`resolve_op`, decode helpers, sink encoding).

Remaining cleanup:
- Consolidate `pipeline.rs` serde types (`PipelineSpec`, `SourceSpec`, `SinkSpec`, `OpSpec`) into graph-owned types if desired.

### Tiling (Currently No-Op)

Tiling was implemented to improve cache efficiency for large images but didn't deliver expected performance gains. It is currently a no-op. The `configure_tiling` / `get_tiling_config` API is exposed but non-functional. This should either be removed or investigated further for SIMD improvements.

### `vb_graph_multi` Ghost Reference

Fixed: Python expression docs now reference only `vb_graph`.

### Inconsistent Test Fixtures

Many test files redefine their own `_plugin_available()`, `plugin_required` marker, and PNG creation fixtures instead of using the shared ones from `conftest.py`. These should be consolidated.

### Temporary Files

- `notebooks/delete_this.ipynb` and `notebooks/delete_this_too.ipynb` should be removed.
- `docs/` at the workspace root (not inside `polars-cv/`) appears to be a remnant and can be removed.
- `tmp_metrics/` directory at workspace root.

### Examples Folder Organization

The canonical runnable demos are now the numbered scripts under
`polars-cv/examples/`:

- `01_getting_started.py`
- `02_image_transforms.py`
- `03_pipeline_composition.py`
- `04_geometry.py`
- `05_reductions_and_features.py`
- `06_detection_metrics.py`
- `07_perceptual_hashing.py`
- `08_ml_integration.py`

Keep these scripts focused (single topic), independent, and under ~500 lines.

The examples directory also includes `detection_data.py`, a shared synthetic
dataset generator used by `06_detection_metrics.py`. It creates one coherent
dataset consumed by contour, bbox, and pre-matched metric paths, with CLI
controls for both data-generation difficulty and matcher thresholds.

### Test File Naming

Several test files have `_gaps` in their name (e.g., `test_binary_ops_gaps.py`, `test_resize_gaps.py`). These were named when they covered gaps in functionality. The gaps have since been filled, making the naming misleading. Consider renaming.

### Outer tests/ Directory

`tests/test_tiff_integration.py` lives at the workspace root level while all other tests are in `polars-cv/tests/`. This should be consolidated.

### `cloud_options` Not Serialized

`SourceSpec.to_dict()` in `_types.py` does not include `cloud_options`, which means cloud configuration may not propagate through graph-based pipelines.

### `ANNOTATED_POINT_SCHEMA` Not Re-exported

Defined in `geometry/schemas.py` and exported by `geometry/__init__.py`, but not included in the main `polars_cv/__init__.py` `__all__`.

### Resize Rank Preservation (Fixed)

The `resize` operation now preserves input rank: 2D `[H, W]` input produces 2D `[H_new, W_new]` output, and 3D input stays 3D. The Rust runner (`runner.rs`) conditionally reshapes after resize based on the input rank. The `grayscale` planning-time `infer_shape` also no longer promotes 2D inputs to 3D.

### Detection Metric Fixes (Applied)

Several correctness fixes have been applied to the detection metrics:
- **FROC**: `iou_threshold` expression fixed, unweighted `total_gts` overcounting fixed, bootstrap now recomputes `total_targets` from sampled images.
- **LROC**: Lower-right endpoint `(fpf=1, sens=max)` added to the curve for complete AUC. Two variants supported: `"best_tp"` (any TP above threshold) and `"top_scoring"` (classical single-commitment).
- **PR AUC**: `auc()` now uses monotone-envelope precision interpolation (standard AP). Raw trapezoidal integration available via `raw_auc()`. The old `ap()` method was removed.
- **ContourMatcher**: Default `min_contour_area` changed from 0.0 to 1.0. Zero-score detections are now filtered *before* matching (prevents false GT claims). `label_reduce` has centroid fallback for sub-pixel contours and a new `"boundary"` region mode that includes boundary pixels.
- **Pipeline efficiency**: ContourMatcher resize path fuses resize + contour extraction into a single `vb_graph` pass using multi-output dict sink. Intermediate format uses `"blob"` (VIEW protocol) instead of `"list"`.
- **Mann-Whitney U**: `mann_whitney_auc(level="detection"|"image")` added to `FROCResult` and `LROCResult`.
