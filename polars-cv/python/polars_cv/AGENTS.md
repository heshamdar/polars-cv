# AGENTS.md — Python API (`polars_cv`)

> Read the [root AGENTS.md](../../../AGENTS.md) first for project-wide context.
> Update this file when you change the Python API surface, pipeline builder, lazy composition, types, or graph serialization.

## Purpose

The **user-facing Python layer**. Responsible for:

- **Pipeline specification** — building operation sequences with `Pipeline`
- **Lazy composition** — composing pipelines via `LazyPipelineExpr` (`.cv.pipe()`)
- **Graph construction** — serializing pipelines into JSON graphs for Rust execution via `PipelineGraph`
- **Schema inference** — determining output Polars dtypes at planning time
- **Validation** — enforcing domain, dtype, and operation contracts before execution
- **Expression namespaces** — registering `.cv`, `.point`, `.contour`, `.bbox` on `pl.Expr`

**No computation happens here.** This layer builds specs and validates contracts. All execution is in Rust.

## Key Files

| File | Responsibility |
|------|---------------|
| `__init__.py` | Public API surface, `numpy_from_struct`, `show_images`, mask/hash helpers, tiling config |
| `pipeline.py` | `Pipeline` builder — source, operations, domain/dtype/shape tracking |
| `lazy.py` | `LazyPipelineExpr` — lazy composition, `.pipe()`, `.merge_pipe()`, `.alias()`, `.sink()`, binary ops |
| `expressions.py` | `CvNamespace` — `.cv.pipe()`, `.cv.width()`, `.cv.height()`, `.cv.channels()`, `.cv.image_dtype()` |
| `_types.py` | `OpSpec`, `ParamValue`, `SourceSpec`, `SinkSpec`, `DType`, `ColorSpace`, `Domain` |
| `_graph.py` | `PipelineGraph`, `GraphNode` — DAG construction, JSON serialization, CSE, plugin registration |
| `_graph_viz.py` | Graph visualization (networkx/graphviz) |
| `display.py` | `show_images()` — notebook image rendering, format detection, VIEW/numpy to PNG |
| `metrics/` | Detection metrics — see [`metrics/AGENTS.md`](metrics/AGENTS.md) |
| `geometry/` | Point/contour namespaces, schemas — see [`geometry/AGENTS.md`](geometry/AGENTS.md) |

## Core Concepts

### Pipeline Builder Pattern

`Pipeline` uses an **immutable clone-on-modify** pattern. Every operation returns a new `Pipeline` instance:

```python
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
pipe = Pipeline().source("image_bytes").channel_select(index=0)
pipe = Pipeline().source("image_bytes").cvt_color("rgb", "hsv")
pipe = Pipeline().source("image_bytes").sobel(axis="x")
pipe = Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3)
pipe = Pipeline().source("image_bytes", on_error="null")  # graceful error handling
```

Key internal state tracked on each Pipeline:
- `_source: SourceSpec | None` — how to decode input data
- `_ops: list[OpSpec]` — ordered list of operations
- `_domain: str` — current domain (buffer/contour/scalar/vector)
- `_output_dtype: str` — current dtype tracking (u8/f32/auto/etc.)
- `_ndim: int | None` — current dimensionality
- `_expr_columns: dict[str, pl.Expr]` — expression parameters to pass to Rust

### LazyPipelineExpr Composition

The **primary API path**: `Pipeline()` -> `.cv.pipe()` -> `LazyPipelineExpr.sink()` -> `PipelineGraph` -> `vb_graph`.

```python
img = pl.col("image").cv.pipe(Pipeline().source("image_bytes").resize(height=224, width=224))
expr = img.sink("numpy")                                    # single output
expr = img.alias("resized").sink({"resized": "numpy"})      # multi-output
```

`LazyPipelineExpr` enables:
- **Chaining:** `.pipe(Pipeline().blur(sigma=2.0))` adds ops to the graph
- **Branching:** `.alias("name")` creates named reference points
- **Merging:** `.merge_pipe(other_expr)` combines independent pipelines into one graph
- **Binary ops:** `.bitwise_and(other)`, `.blend(other, alpha=0.5)`, etc.
- **Multi-output:** `.sink({"alias1": "format1", "alias2": "format2"})`

### Graph Serialization

When `.sink()` is called, a `PipelineGraph` is built:

1. All `LazyPipelineExpr` nodes are traversed
2. Each becomes a `GraphNode` with its pipeline spec, upstream dependencies, and optional alias
3. Output specs are attached to terminal nodes
4. Common subexpression elimination (CSE) shares common prefixes
5. The graph is serialized to JSON
6. `register_plugin_function(function_name="vb_graph", ...)` is called

### Operation Contracts (view-buffer is the authority)

Every operation's schema effect — output domain, dtype, rank (ndim) and channel
count — comes from view-buffer's per-op `ViewDto` contract, surfaced to Python
through `_lib.op_contract(op_json)` and `_lib.op_output_dtype(op_json, in_dtype)`.
The Python planner (`_compute_output_domain_dtype_ndim` / `_update_channels_from_rule`)
**reads** these rules; it does not re-declare them. There is no Python contract
table to keep in sync.

The contract fields read by the planner are:
- `output_domain` — buffer / scalar / vector / contour (`any` = identity, leaves
  the domain unchanged)
- `dtype_rule` — resolved to a concrete dtype by `op_output_dtype`
- `rank_rule` — `fixed:N`, `reduce_one`, `preserve`, or `unknown`
- `channel_rule` — drives planning-time channel inference

These drive schema inference at planning time. **Planning-time schema must match
execution-time schema.** If an op's dtype cannot be determined at planning time
(e.g. `auto` from an `image_bytes` source), it stays `auto`.

### Alpha Channel Handling

Alpha channels are **always preserved** during image decoding. Image sources
(`image_bytes`, `file_path`) produce unknown channel count at planning time
(`_shape_hints.channels = None`). Users can assert known channels via
`.assert_shape(channels=4)`.

Each op's alpha/channel behaviour is described by its view-buffer `channel_rule`
(e.g. passthrough, drop-to-fixed, color-conversion). Channel inference is
implemented in `Pipeline._update_channels_from_rule()`, called at the end of
`_update_shape_hints()`, which reads `op_contract(...)["channel_rule"]` and
applies it to the tracked channel count. Rust implements the matching behaviour
based on the buffer's actual channel count.

### ParamValue — Literal vs Expression Parameters

Operations accept either literal values or Polars expressions:

```python
pipe.resize(height=224, width=pl.col("target_w"))
#           ^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^
#           literal           expression (resolved per-row at execution time)
```

`ParamValue` wraps this distinction. Expression params are tracked in `_expr_columns` and passed to Rust as additional input columns.

**Dynamic parameter coverage:** Most numeric parameters accept `IntOrExpr` / `FloatOrExpr`. This includes: resize dimensions, crop offsets, pad values, rotate angle, blur sigma, threshold value, canny thresholds, contrast/gamma/brightness factors, morphology ksize/iterations, channel_select index, convolve2d ksize, warp_affine output_size, rasterize fill_value/background, reduce_percentile q, reduce_std ddof. Structural parameters (matrix, kernel, axes, enum values) remain static only.

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

3. **Schema inference**: nothing to add in `_types.py`. The op's domain, dtype,
   rank and channel effects are read at planning time from its view-buffer
   `ViewDto` contract via `_lib.op_contract` / `_lib.op_output_dtype`, so make
   sure the op exposes the right contract on the Rust side (next step).

4. **Rust side**: Map the operation name in `resolve_op` and give it a `ViewDto`
   contract — see [`polars-cv/src/AGENTS.md`](../../src/AGENTS.md)

### Affine Pipeline Fusion

Consecutive affine-family operations are fused at serialization time via `_fuse_affine_ops()` (called in `_to_spec_dict()`). Matrix composition uses `_compose_affine_ops()` which performs standard 2×3 matrix multiplication. The fused operation uses the output dimensions from the **last** affine in the chain.

Fusible operations:
- `warp_affine()` — always fusible
- `shear()` and `rotate_and_scale()` — construct matrices and delegate to `warp_affine()`, so they fuse automatically
- `rotate()` with a **static, non-fast-path angle** (not 90/180/270) and **known input dimensions** — converted to `warp_affine` via `_try_convert_rotate_to_affine()` at planning time

**Not fusible:** `rotate()` with an expression-based angle, or with a fast-path angle (90/180/270 use zero-copy `ViewOp` and cannot be represented as an affine matrix).

### Shape Hints for Rotation

`_update_shape_hints()` handles rotation dimensions:
- **Static 90/270 with expand=False:** H and W are swapped
- **Static angle with expand=True:** Output dimensions computed by `_compute_rotate_expand_shape()` using the rotated bounding box formula
- **Expression-based angle:** Shape hints set to `None` (unknown at planning time)

## Common Pitfalls

- **Don't mutate Pipeline in place.** Always use `_clone()` then modify the clone. The immutable pattern is intentional.
- **Give new ops a correct view-buffer `ViewDto` contract.** Schema inference
  reads dtype/domain/rank/channel from it at planning time; a wrong or missing
  contract makes planned and executed schemas diverge (caught by the
  plan==exec tests in `test_sanitation.py`).
- **Expression params must be tracked.** If an op accepts `pl.Expr` parameters, they must go through `_track_expr()` to be serialized to Rust.
- **The `auto` dtype.** Sources like `image_bytes` and `file_path` have dtype `auto` because the actual dtype is only known at execution time (after decoding). Operations that need a known dtype (like `sink("list")` or `sink("array")`) must have it resolved before the sink, either via `source(..., dtype="f32")`, `.cast(...)`, or a dtype-fixing operation.
- **Continuation nodes must inherit upstream typing context.** In `LazyPipelineExpr.pipe()` for op-only continuation pipelines (`source is None`), compute node domain/dtype/ndim using upstream state + new ops. Copying op-only pipeline typing state can cause contract drift (planned dtype mismatch at execution).
