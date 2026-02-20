# AGENTS.md — Python API (`polars_cv`)

> Read the [root AGENTS.md](../../../AGENTS.md) first for project-wide context.
> Update this file when you change the Python API surface, pipeline builder, lazy composition, types, or graph serialization.

## Purpose

This is the **user-facing Python layer**. It is responsible for:

- **Pipeline specification** — building operation sequences with `Pipeline`
- **Lazy composition** — composing pipelines via `LazyPipelineExpr` (`.cv.pipe()`)
- **Graph construction** — serializing pipelines into JSON graphs for Rust execution via `PipelineGraph`
- **Schema inference** — determining output Polars dtypes at planning time
- **Validation** — enforcing domain, dtype, and operation contracts before execution
- **Expression namespaces** — registering `.cv`, `.point`, `.contour` on `pl.Expr`

**No computation happens here.** This layer only builds specs and validates contracts. All execution is in Rust.

## Key Files

| File | Responsibility | Lines |
|------|---------------|-------|
| `__init__.py` | Public API surface, `numpy_from_struct`, mask/hash comparison helpers, tiling config re-export | ~430 |
| `pipeline.py` | `Pipeline` builder — source, operations, sink, domain tracking, dtype tracking, shape inference | ~2600 |
| `lazy.py` | `LazyPipelineExpr` — lazy composition, `.pipe()`, `.merge_pipe()`, `.alias()`, `.sink()`, binary ops | ~900 |
| `metrics/` | Detection metrics — matchers, DetectionTable, metric functions, bootstrap CI, AUC helpers — see [`metrics/AGENTS.md`](metrics/AGENTS.md) | ~1500 |
| `expressions.py` | `CvNamespace` (`.cv.pipe()`, ~~`.cv.pipeline()`~~), `apply_pipeline()` | ~150 |
| `_types.py` | `OpSpec`, `ParamValue`, `SourceSpec`, `SinkSpec`, `SourceFormat`, `SinkFormat`, `DType`, `OPERATION_CONTRACTS` | ~850 |
| `_graph.py` | `PipelineGraph`, `GraphNode` — DAG construction, JSON serialization, CSE optimization, `register_plugin_function` call | ~680 |
| `_graph_viz.py` | Graph visualization (networkx/graphviz) | ~200 |
| `geometry/` | Point and contour namespaces, schemas, validation — see [`geometry/AGENTS.md`](geometry/AGENTS.md) |

## Core Concepts

### Pipeline Builder Pattern

`Pipeline` uses an **immutable clone-on-modify** pattern. Every operation returns a new `Pipeline` instance:

```python
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
# Each method call clones the pipeline and appends an OpSpec
```

Key internal state tracked on each Pipeline:
- `_source: SourceSpec | None` — how to decode input data
- `_ops: list[OpSpec]` — ordered list of operations
- `_sink: SinkSpec | None` — how to encode output (only set via `.sink()`)
- `_domain: str` — current domain (buffer/contour/scalar/vector)
- `_output_dtype: str` — current dtype tracking (u8/f32/auto/etc.)
- `_ndim: int | None` — current dimensionality
- `_expr_columns: dict[str, pl.Expr]` — expression parameters to pass to Rust

### LazyPipelineExpr Composition

This is the **primary API path** (replacing the old `CvNamespace.pipeline()` method):

```python
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes").resize(224, 224))
expr = img.sink("numpy")            # Single output
expr = img.alias("resized").sink({"resized": "numpy"})  # Multi-output
```

`LazyPipelineExpr` enables:
- **Chaining:** `.pipe(Pipeline().blur(sigma=2.0))` adds ops to the graph
- **Branching:** `.alias("name")` creates named reference points
- **Merging:** `.merge_pipe(other_expr)` combines independent pipelines into one graph
- **Binary ops:** `.bitwise_and(other)`, `.blend(other, alpha=0.5)`, etc.
- **Multi-output:** `.sink({"alias1": "format1", "alias2": "format2"})`

### Graph Serialization

When `.sink()` is called on a `LazyPipelineExpr`, a `PipelineGraph` is built:

1. All `LazyPipelineExpr` nodes are traversed
2. Each becomes a `GraphNode` with its pipeline spec, upstream dependencies, and optional alias
3. Output specs are attached to terminal nodes
4. Common subexpression elimination (CSE) shares common prefixes
5. The graph is serialized to JSON
6. `register_plugin_function(function_name="vb_graph", kwargs={"graph_json": ..., "expr_column_names": ...})` is called

### Operation Contracts (`OPERATION_CONTRACTS` in `_types.py`)

Every operation has a contract defining:
- `DTypeEffect` — how it changes the dtype (preserve, promote to f32, explicit, etc.)
- `NdimEffect` — how it changes dimensionality (preserve, set to 0, set to 1, etc.)

These contracts drive schema inference at planning time. **Planning-time schema must match execution-time schema.** If an operation's effect on dtype cannot be determined at planning time (e.g., `auto` from image_bytes source), it must be flagged.

### ParamValue — Literal vs Expression Parameters

Operations accept either literal values or Polars expressions:

```python
pipe.resize(height=224, width=pl.col("target_w"))
#           ^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^
#           literal           expression (resolved per-row at execution time)
```

`ParamValue` wraps this distinction. Expression params are tracked in `_expr_columns` and passed to Rust as additional input columns.

### Metrics Subsystem (`metrics/`)

The `metrics/` subpackage provides a three-layer detection metrics pipeline:

1. **Matchers** (`_matching/`) — convert raw data into a canonical `DetectionTable`:
   - `ContourMatcher` — heatmap + binary mask → contour extraction/matching
   - `BBoxMatcher` — bounding box lists → Rust `bbox_match_detections` plugin
   - `PreMatchedAdapter` — pre-computed TP/FP per detection
2. **Metric functions** (`_metrics/`) — operate on `DetectionTable`:
   - `precision_recall_curve`, `average_precision`, `mean_average_precision`
   - `froc_curve`, `lroc_curve`
   - `confusion_at_threshold`, `precision_at_threshold`, `recall_at_threshold`, `f1_at_threshold`
3. **Result objects** — `MetricResult` base with `auc()`, `partial_auc()`, `interpolate()`, `summary_table()`

All curve aggregation uses Polars expressions — no Python loops over rows.
See [`metrics/AGENTS.md`](metrics/AGENTS.md) for full details.

## What the Canonical API Path Looks Like

```python
# Build pipeline (no source or sink yet)
preprocess = Pipeline().source("image_bytes").resize(height=224, width=224).normalize()

# Apply to column and sink
result = df.with_columns(
    processed=pl.col("image").cv.pipe(preprocess).sink("numpy")
)
```

The flow is: `Pipeline()` → `.cv.pipe()` → `LazyPipelineExpr` → `.sink()` → `PipelineGraph` → `vb_graph` Polars expression.

## Legacy Code (To Be Removed)

### `CvNamespace.pipeline()` in `expressions.py`

This is the **old API path**. It takes a fully-formed Pipeline (with sink already attached) and executes it directly. It should be removed in favor of `.cv.pipe(pipe).sink(format)`.

### `apply_pipeline()` in `expressions.py`

This function is used by the old `CvNamespace.pipeline()`. It constructs a graph from a finalized pipeline. It can be removed once `CvNamespace.pipeline()` is removed, as `.sink()` on `LazyPipelineExpr` handles this directly.

## Adding a New Operation (Python Side)

1. **`pipeline.py`**: Add a method to `Pipeline` class:
   - Validate domain with `_validate_domain()`
   - Clone with `_clone()`
   - Append `OpSpec` with params wrapped in `ParamValue`
   - Update dtype tracking with `_update_output_dtype()`
   - Return new Pipeline

2. **`lazy.py`**: Add a corresponding method to `LazyPipelineExpr` that delegates to `Pipeline`:
   ```python
   def my_op(self, ...) -> "LazyPipelineExpr":
       return self._chain(Pipeline().my_op(...))
   ```

3. **`_types.py`**: Add the operation's contract to `OPERATION_CONTRACTS`

4. **Rust side**: Map the operation name in `resolve_op` — see [`polars-cv/src/AGENTS.md`](../../src/AGENTS.md)

## Common Pitfalls

- **Don't mutate Pipeline in place.** Always use `_clone()` then modify the clone. The immutable pattern is intentional.
- **Don't forget to update `OPERATION_CONTRACTS`** when adding ops. Schema inference will fail silently or produce wrong types.
- **Expression params must be tracked.** If an op accepts `pl.Expr` parameters, they must go through `_track_expr()` to be serialized to Rust.
- **The `auto` dtype.** Sources like `image_bytes` and `file_path` have dtype `auto` because the actual dtype is only known at execution time (after decoding). Operations that need a known dtype (like `sink("list")` or `sink("array")`) must have it resolved before the sink, either via `source(..., dtype="f32")`, `.cast(...)`, or a dtype-fixing operation.
