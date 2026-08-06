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
| `expressions.py` | `CvNamespace` — `.cv.pipe()`, `.cv.read_bytes()`, `.cv.width()`, `.cv.height()`, `.cv.channels()`, `.cv.image_dtype()` |
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
pipe = Pipeline().source("image_bytes").convert_color("rgb", "hsv")
pipe = Pipeline().source("image_bytes").sobel(axis="x")
pipe = Pipeline().source("image_bytes").grayscale().threshold(128).erode(ksize=3)
pipe = Pipeline().source(
    "image_bytes", on_error="null"
)  # null this source's decode errors
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).on_error("null")
# ^ graph-level per-row policy: "raise" (default) | "null" | "null_with_message".
#   "null" nulls all outputs of a failing row (decode, op, or encode errors);
#   "null_with_message" additionally adds a reserved `_error` string field to
#   the output struct. Composed pipelines must agree on the policy
#   (PipelineGraph._to_dict raises on conflicts).
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
img = pl.col("image").cv.pipe(
    Pipeline().source("image_bytes").resize(height=224, width=224)
)
expr = img.sink("numpy")  # single output
expr = img.alias("resized").sink({"resized": "numpy"})  # multi-output
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

**A node reference is not a dependency until it is an upstream edge.** An op or
source that points at another `LazyPipelineExpr` by node id — `rasterize(shape=)`,
`source("contour", shape=)` — must *also* append it to `Pipeline._shape_refs`,
because `_shape_refs` is what `cv.pipe` / `LazyPipelineExpr.pipe` turn into the
`upstream` list, and only an upstream edge puts the referenced node into the
graph at all. Record the id without the edge and the reference dangles at
execution — invisibly, for as long as some other consumer happens to pull the
same node in (masking with the same image, which every example does). The
appended edge does not disturb the referenced node's own input:
`_build_column_bindings` keys on the node having a column, and the executor
picks the decode path from `has_column_binding`, using upstream only for
ordering.

### Operation Contracts (view-buffer is the authority)

Every operation's schema effect — output domain, dtype, rank (ndim) and channel
count — comes from view-buffer's per-op `ViewDto` contract, surfaced to Python
through `_lib.op_contract(op_json)` and `_lib.op_schema(op_json, domain, dtype, ndim)`.
The Python planner (`_compute_output_domain_dtype_ndim` / `_update_channels_from_rule`)
**reads** these rules; it does not re-declare them. There is no Python contract
table to keep in sync.

The contract fields read by the planner are:
- `output_domain` — buffer / scalar / vector / contour (`any` = identity, leaves
  the domain unchanged)
- `dtype_rule` — resolved to a concrete dtype by `op_schema`
- `rank_rule` — `fixed:N`, `reduce_one`, `preserve`, or `unknown`
- `channel_rule` — drives planning-time channel inference

These drive schema inference at planning time. **Planning-time schema must match
execution-time schema.** If an op's dtype cannot be determined at planning time
(e.g. `auto` from an `image_bytes` source), it stays `auto`.

An `auto` **source** (the `source()` default) is treated like `blob` here: its
decode path is chosen from the column dtype in Rust at execution time, so
`_expected_ndim` is `None` and the dtype stays `auto` unless the caller asserts
one. The `list`/`array` sink guards in `lazy.py` let `auto` through alongside
`list`/`array` because Rust's `resolved_output_specs` resolves a `List`/`Array`
column's leaf dtype and rank when the plan sees the input; a Binary/image column
under `auto` then surfaces the error there instead.

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

**The rule:** a parameter may be per-row **iff it has no effect on output shape,
rank, or dtype**. Everything else follows from that one invariant.

**Dynamic parameter coverage.** Three kinds of parameter are expression-capable:

1. *Scalars* (`IntOrExpr` / `FloatOrExpr`), via `_track_expr`: resize dimensions,
   crop offsets, pad amounts and values, rotate angle and `border_value`,
   warp_affine `output_size`/`border_value`, blur sigma, threshold value, canny
   thresholds, contrast/gamma/brightness/sharpen factors, morphology
   ksize/iterations, channel_select index, convolve2d ksize, rasterize and
   contour-source `width`/`height`/`fill_value`/`background`, histogram
   `range_min`/`range_max`, extract_contours `min_area`, reduce_percentile q,
   reduce_std ddof.
2. *Per-element lists*, via `_param_list` — the list **length** stays structural
   while each element may be an expression: warp_affine `matrix`, `reshape`
   shape, convolve2d `kernel` (which is what makes `sharpen(strength)` dynamic),
   normalize `mean`/`std`, channel_swap `order`. Rust reads these with
   `resolve_f32_list` / `resolve_usize_list`.
3. *Non-structural enums and flags*, via `_enum_param` and `get::opt_bool_dyn`:
   resize/letterbox `filter`, rotate/warp_affine `interpolation`, `pad(mode)`,
   `pad_to_size(position)`, `convolve2d(border)`, extract_contours
   `mode`/`method`, label_reduce `reduction`/`region_mode`,
   `apply_mask(invert)`, `area(signed)`, `convolve2d(normalize)`.

**Plan-time probing is why enums need care.** `op_infer_shape` (`lib.rs`) runs
each op four times with every expression param bound to an *integer* probe. A
dynamic enum cannot read an integer, so `ParamCtx::probe` marks the context and
the enum/bool accessors substitute their default. That is sound only because of
the rule above — the variant probing picks cannot change the inferred schema.
Signalling it explicitly (rather than sniffing the column's dtype) keeps real
execution strict: routing an integer column into an enum param still errors.

**Structural parameters are literal-only and enforced on both sides.** Axis
lists, reduction `axis`, `perceptual_hash(hash_size)`, `reshape` arity,
`rotate(expand)`, and the dtype-bearing enums `cast(dtype)`,
`normalize(method`/`out_dtype)`, `histogram(closed`/`output)` fix the plan-time
schema, so they must be literals. A literal `ParamValue` can never hold a
`pl.Expr` — `ParamValue.__post_init__` (`_types.py`) rejects it with a clear
"structural" error, and the Rust resolvers (`params::get::maybe_usize_literal` /
`opt_u32_literal` / `req_enum_literal`, and `ParamValue::resolve_string`) reject
a bound expression slot as defense-in-depth. Guarded by
`TestStructuralParamsRejectExpressions` / `TestFillRangeParamsAcceptExpressions`
in `test_param_strictness.py` and `test_structural_literal_resolvers_reject_bound_slots`
in `params.rs`.

**The geometry namespaces use a different mechanism.** `.contour`/`.point`/`.bbox`
bypass `vb_graph`, so they have no `ParamValue`. Their per-row channel is the
plugin's *input series*: `_ArgBinder` (`_namespace.py`) appends an
expression-valued parameter as an extra argument and records it in an
`input_slots` name→index map, which Rust reads via `GeomParams`
(`src/geom_params.rs`). Names, not positions, because these functions also take
*optional* data operands (`scores`, `origin`) whose position would otherwise be
ambiguous.

**Null parameter values are a shared policy, not per-op handling.** A parameter
column may contain nulls; `Pipeline.on_null_param("raise"|"null")` says whether
that fails the query or nulls the affected rows. It is stored as
`Pipeline._on_null_param`, hoisted by the same loop as `_on_error` in
`PipelineGraph._to_dict()`, and emitted as a top-level `"on_null_param"` key
**only when non-default**, so unaffected graphs serialize byte-identically and
keep their compiled-graph cache entry. Rust applies it at one place —
`ParamCol::on_null` — so no operation declares anything.

That shared loop also carries a conflict check, but it can only ever fire for
`_on_error`: the hoist collects non-default values, and with `"raise"` and
`"null"` as the only null-param policies the collected set can never hold two.
So an explicit `.on_null_param("raise")` composed with a `"null"` pipeline gives
the graph `"null"` rather than an error — that is intended, not an oversight
(`TestComposition::test_one_pipeline_setting_the_policy_applies_to_the_graph`).
Adding a third policy would make the branch reachable and change that.

Do **not** add a per-op or per-parameter null keyword. Deliberately absent, for
two reasons: a fallback value is already expressible as
`pl.col("h").fill_null(224)`, and a per-parameter policy would have to enter the
`ParamValue` wire format, which would mean `__eq__`/`__hash__` must include it
or CSE will merge ops that differ only in policy.

The geometry namespaces have no `Pipeline` to hang a graph-level setting on, so
the policy lives on the accessor: `on_null(policy)` returns a copy with
`_on_null` set, and `_ArgBinder.call` injects it into kwargs beside
`input_slots`. That keeps it out of all 15 geometry method signatures.

It lives on `_GeomNullPolicy`, a mixin the three geometry namespaces add
alongside `_PluginNamespace` — **not** on `_PluginNamespace` itself, which `.cv`
also inherits. `.cv` routes its per-row parameters through `vb_graph`, where
only `Pipeline.on_null_param` is read, so inheriting `on_null` there would let
`pl.col("x").cv.on_null("null")` chain and read as effective while doing
nothing. On the mixin, that call is an `AttributeError`
(`test_cv_does_not_expose_on_null`). Keep any future accessor-level policy on
the same mixin unless `.cv` genuinely honours it.

## Adding a New Operation (Python Side)

1. **`pipeline.py`**: Add a method to `Pipeline` that returns
   `self._append_op("<op_name>", lambda p: {...params...})`. That is the whole
   builder — there is no sequence to get right:

   ```python
   def erode(self, *, ksize: IntOrExpr = 3, iterations: IntOrExpr = 1) -> "Pipeline":
       """..."""
       return self._append_op(
           "erode",
           lambda p: {
               "ksize": p._track_expr(ksize),
               "iterations": p._track_expr(iterations),
           },
       )
   ```

   The callback receives the *cloned* pipeline, so `p._track_expr` registers
   per-row expressions on the clone rather than the receiver. `_append_op`
   then validates the input domain against `op_contract(...)["input_domains"]`
   and hands off to `_push_op`, which appends and applies **both** halves of
   the plan-time effect: the `op_schema` fold (domain/dtype/ndim) and the
   shape hints (`op_infer_shape` for H/W, the channel rule for C).

   **Do not touch `_ops` directly.** `_push_op` is the only function permitted
   to mutate it, enforced by `test_op_append_is_structurally_exclusive` in
   `tests/test_append_contract.py`. That guard exists because the previous
   convention — each builder calling the update methods by hand — let 41 of 60
   builders skip the shape-hint half and publish a planned schema execution
   could not produce. Never assign `_current_domain` / `_output_dtype` /
   `_shape_hints` by hand either; they follow from the op's Rust contract.

   Validation that must happen before the op is built (a kernel-size check, an
   enum parse) goes in the method body before the `return`; work that must
   happen *after* the append (e.g. `scale`'s `preserve_dtype` cast-back) reads
   the returned pipeline. Both compose without bypassing the append path.

2. **`lazy.py`**: Nothing to add for an ordinary op. `LazyPipelineExpr` generates a
   forwarder for every chainable `Pipeline` method at import time
   (`_install_pipeline_forwarders`), copying the signature so `inspect`/IDEs/the
   parity test see the real parameters. Only define a method explicitly here if it
   needs bespoke lazy behaviour (e.g. a binary op taking another
   `LazyPipelineExpr`); the generator skips names already defined. After changing
   `Pipeline`, regenerate the type stub with `python scripts/gen_lazy_stub.py`.

3. **Schema inference**: nothing to add in `_types.py` or the planner. The
   op's domain, dtype, rank and channel effects are read at planning time from
   its Rust contract via `_lib.op_schema` (and `_lib.op_contract` for
   channels/rank detail), so make sure the op declares the right contract on
   the Rust side (next step). Do not add per-op special cases in Python —
   `test_op_schema_authority` and the batch-fold conformance tests in
   `test_sanitation.py` guard this.

4. **Rust side**: Map the operation name in `resolve_op` to a `GraphStep`
   (buffer ops wrap a view-buffer `ViewDto`; graph-level steps get their own
   variant) — see [`polars-cv/src/AGENTS.md`](../../src/AGENTS.md)

### Affine Pipeline Fusion

Consecutive affine-family operations are fused at serialization time via `_fuse_affine_ops()` (called in `_to_spec_dict()`). Matrix composition uses `_compose_affine_ops()` which performs standard 2×3 matrix multiplication. The fused operation uses the output dimensions from the **last** affine in the chain.

Fusible operations:
- `warp_affine()` — always fusible
- `shear()` and `rotate_and_scale()` — construct matrices and delegate to `warp_affine()`, so they fuse automatically
- `rotate()` with a **static, non-fast-path angle** (not 90/180/270) and **known input dimensions** — converted to `warp_affine` via `_try_convert_rotate_to_affine()` at planning time

**Not fusible:** `rotate()` with an expression-based angle, or with a fast-path angle (90/180/270 use zero-copy `ViewOp` and cannot be represented as an affine matrix).

### Shape Hints (single authority: view-buffer `infer_shape`)

`_update_shape_hints()` no longer re-derives any per-dimension geometry in
Python. It reads the op's view-buffer `infer_shape` through the `op_infer_shape`
FFI (`_update_hw_from_infer_shape`), the same authority execution uses, so the
tracked H/W cannot disagree with what the op produces.

Not every step *has* an inferable shape: axis reductions, histograms, channel
merge and the binary ops are graph-level steps `op_infer_shape` rejects. For
those the H/W hints are **invalidated**, not carried forward — several of them
do change H/W, and keeping the pre-op values is how a pipeline came to publish
`[100, 200, 2]` for data that executes as `[200, 3, 2]`. Unknown is always safe:
`expected_shape` reports `None` and a typed sink asks for an explicit shape. Unknowns
propagate automatically: an unknown input dim or a per-row expression param
yields a `None` output dim. This covers every op uniformly — including rotation
(static 90/270 swap, static-angle expand bounding box, and expression-angle
"unknown", all computed by the Rust `RotateAffine`/`Rotate90` `infer_shape`).
Channels stay with `_update_channels_from_rule` (the channel rule); rank stays
with `op_schema`.

## Common Pitfalls

- **Don't mutate Pipeline in place.** Always use `_clone()` then modify the clone. The immutable pattern is intentional.
- **Give new ops a correct Rust contract (`ViewDto` op / `GraphStep`).** Schema inference
  reads dtype/domain/rank/channel from it at planning time; a wrong or missing
  contract makes planned and executed schemas diverge (caught by the
  plan==exec tests in `test_sanitation.py`).
- **Expression params must be tracked.** If an op accepts `pl.Expr` parameters, they must go through `_track_expr()` to be serialized to Rust.
- **The `auto` dtype.** Sources like `image_bytes` and `file_path` have dtype `auto` because the actual dtype is only known at execution time (after decoding). Operations that need a known dtype (like `sink("list")` or `sink("array")`) must have it resolved before the sink, either via `source(..., dtype="f32")`, `.cast(...)`, or a dtype-fixing operation.
- **Continuation nodes must inherit upstream typing context.** In `LazyPipelineExpr.pipe()` for op-only continuation pipelines (`source is None`), compute node domain/dtype/ndim using upstream state + new ops. Copying op-only pipeline typing state can cause contract drift (planned dtype mismatch at execution).
