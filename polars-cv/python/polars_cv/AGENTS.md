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
| `_types.py` | `SlotTable` (a graph's plugin inputs), `CloudOptions`, the `IntOrExpr`-style aliases; the enums (`DType`, `Domain`, …) come from `_ops_generated.py` |
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
pipe = Pipeline().source("image_bytes").convert_color(from_space="rgb", to_space="hsv")
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

A Pipeline's whole internal state:
- `_plan: Plan` — the Rust plan (`src/plan.rs`): the source, the typed ops and
  the state at every op boundary. Immutable; `_state` is `_plan.state`.
- `_exprs: list[pl.Expr]` — the expressions the plan's `{"$slot": i}`
  parameters name, by `i` (each distinct expression once, by `meta.eq`)
- `_node_refs` — the `LazyPipelineExpr` nodes the ops or source read by id
- `_on_error`, `_on_null_param` — the graph-level per-row policies

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
6. `_plugin.call("vb_graph", ...)` is called (the package's only route to `register_plugin_function`)

**A node reference is not a dependency until it is an upstream edge.** An op
that points at another `LazyPipelineExpr` by node id — `rasterize(shape=)` (and
so `source("contour", shape=)`), a binary operand — is recorded in `Pipeline._node_refs`
(by `_encode_field`, for every node-typed field), because `_node_refs` is both
what plans the op (the node's state, by id) and what `cv.pipe` /
`LazyPipelineExpr.pipe` turn into the
`upstream` list, and only an upstream edge puts the referenced node into the
graph at all. Record the id without the edge and the reference dangles at
execution — invisibly, for as long as some other consumer happens to pull the
same node in (masking with the same image, which every example does). The
appended edge does not disturb the referenced node's own input:
`PipelineGraph._to_dict`'s column bindings key on the node having a column, and the executor
picks the decode path from `has_column_binding`, using upstream only for
ordering.

### Operation Contracts (view-buffer is the authority)

Every operation's schema effect — output domain, dtype, rank (ndim), H/W and
channel count — comes from the op's Rust contract, applied in Rust by one
`plan::step` per appended op (`src/plan.rs`). The ops themselves live in the
pipeline's Rust `Plan`, which keeps the state entering each op; Python holds
the plan object and never an op list. Every change is a `Plan` method that
plans each op it keeps: `push` (an append, with `refs` — the states of the
nodes the op reads by id), `with_source` (which plans the ops again from the
source's state), `select` (a slice, reorder or deletion), `rebased` (a
continuation onto an upstream node's state) and `run_pass`. So no per-position
fact is ever re-keyed by hand, and an `assert_shape` is planned like any op.
The tracked state is one `PlanState` (`Pipeline._state`: domain, dtype, rank,
known sizes), computed only in Rust and never edited in Python.
Python **reads** these rules; it
does not re-declare them. There is no Python contract
table to keep in sync.

The contract fields read by the planner are:
- `output_domain` — buffer / scalar / vector / contour (`any` = identity, leaves
  the domain unchanged)
- `dtype_rule` — resolved to a concrete dtype by `plan::step`
- `shape` (`OpShape`) — the output shape over the input's; the rank is its
  length and the channel count its axis 2

These drive schema inference at planning time. **Planning-time schema must match
execution-time schema.** If an op's dtype cannot be determined at planning time
(e.g. `auto` from an `image_bytes` source), it stays `auto`.

An `auto` **source** (the `source()` default) is treated like `blob` here: its
decode path is chosen from the column dtype in Rust at execution time, so
the rank is `None` and the dtype stays `auto` unless the caller asserts
one. The `list`/`array` sink guards in `lazy.py` let `auto` through alongside
`list`/`array` because Rust's `resolved_output_specs` resolves a `List`/`Array`
column's leaf dtype and rank when the plan sees the input; a Binary/image column
under `auto` then surfaces the error there instead.

### Alpha Channel Handling

Alpha channels are **always preserved** during image decoding. Image sources
(`image_bytes`, `file_path`) produce unknown channel count at planning time
(`PlanState.dims[2]` is `None`). Users can assert known channels via
`.assert_shape(channels=4)`.

Each op's alpha/channel behaviour is its view-buffer `OpShape` (passthrough,
`SingleChannel`, `ColorChannels`); `plan::step` reads the channel count off the
shape's axis 2. Rust implements the matching behaviour
based on the buffer's actual channel count.

### Literal vs Expression Parameters

Operations accept either literal values or Polars expressions:

```python
pipe.resize(height=224, width=pl.col("target_w"))
#           ^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^
#           literal           expression (resolved per-row at execution time)
```

An expression crosses as `{"$slot": i}` over the pipeline's expression table
(`Pipeline._slot`); a graph renumbers every pipeline's slots onto its one
`SlotTable` (`Plan.to_spec`) and passes the expressions to Rust as additional
input columns.

**The rule:** a parameter may be per-row **iff it has no effect on output shape,
rank, or dtype**. Everything else follows from that one invariant.

**Dynamic parameter coverage.** Three kinds of parameter are expression-capable:

1. *Scalars* (`IntOrExpr` / `FloatOrExpr`), via `_wire`: resize dimensions,
   crop offsets, pad amounts and values, rotate angle and `border_value`,
   warp_affine `output_size`/`border_value`, blur sigma, threshold value, canny
   thresholds, contrast/gamma/brightness/sharpen factors, morphology
   ksize/iterations, channel_select index, rasterize
   `width`/`height`/`fill_value`/`background`, histogram
   `range` (both ends), extract_contours `min_area`, reduce_percentile q,
   reduce_std ddof.
2. *Per-element lists* — the list **length** stays structural while each
   element may be an expression: warp_affine `matrix`, `reshape` shape,
   convolve2d `kernel` (which is what makes `sharpen(strength)` dynamic),
   normalize `mean`/`std`, channel_swap `order`. On a typed op these are
   `Vec<Param<T>>` (or `[Param<T>; N]`) fields, encoded element by element by
   `_encode_field`.
3. *Non-structural enums and flags*, typed `Param<Enum>`/`Param<bool>` fields:
   resize/letterbox `filter`, rotate/warp_affine `interpolation`, `pad(mode)`,
   `pad_to_size(position)`, `convolve2d(border)`, extract_contours
   `mode`/`method`, label_reduce `reduction`/`region_mode`,
   `apply_mask(invert)`, `area(signed)`, `convolve2d(normalize)`.

**The plan reads the op as written, which is why the rule matters.** The
planner reads every rule (domain, dtype, rank, channels, identity) from the
`Wire` op, where a per-row value is unknown: never a placeholder. It can
publish a schema only because of the rule above — no per-row-eligible value
changes those rules; a rule that does read a value (a neighbourhood radius, a
`validate` check) says nothing about one it does not know, and the row checks
it.

**Structural parameters are literal-only and enforced on both sides.** Axis
lists, reduction `axis`, `perceptual_hash(hash_size)`, `reshape` arity,
`rotate(expand)`, and the dtype-bearing enums `cast(dtype)`,
`normalize(method`/`out_dtype)`, `histogram(closed`/`output)` fix the plan-time
schema, so they must be literals. On a typed op the field is a `Literal<T>`: `_encode_field` refuses a
`pl.Expr` for a field the catalogue does not type per-row (the "structural"
`TypeError`), and `{"$slot": n}` in a `Literal` is
a serde error in Rust (`ops::tests::a_slot_in_a_structural_field_is_rejected`).
Guarded by `TestStructuralParamsRejectExpressions` in
`test_param_strictness.py`.

**The geometry namespaces use the same wire form.** `.contour`/`.point`/`.bbox`
bypass `vb_graph`, but each accessor method is generated from its Rust
definition (`src/geom_fns.rs`) as one `_GeomNamespace._call` (`_namespace.py`),
which appends an expression-valued parameter or data operand as an extra
argument and writes `{"$slot": n}` into that field. Rust parses the arguments
as the function's own definition (`GeomParams::parse`, `src/geom_params.rs`).
Each field names its own position, so the optional data operands (`order`,
`origin`) cannot be confused with an appended parameter.

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
`pl.col("h").fill_null(224)`, and a per-parameter policy would have to enter every op's wire form, which CSE
compares, or CSE will merge ops that differ only in policy.

The geometry namespaces have no `Pipeline` to hang a graph-level setting on, so
the policy lives on the accessor: `on_null(policy)` returns a copy with
`_on_null` set, and `_GeomNamespace._call` injects it into the kwargs. That
keeps it out of every geometry method signature.

It lives on `_GeomNamespace`, the three geometry namespaces' base — **not** on
`_PluginNamespace` itself, which `.cv` also inherits. `.cv` routes its per-row parameters through `vb_graph`, where
only `Pipeline.on_null_param` is read, so inheriting `on_null` there would let
`pl.col("x").cv.on_null("null")` chain and read as effective while doing
nothing. On the geometry base, that call is an `AttributeError`
(`test_cv_does_not_expose_on_null`). Keep any future accessor-level policy on
the same mixin unless `.cv` genuinely honours it.

## Adding a New Operation (Python Side)

1. **Nothing, for an ordinary op.** The builder is generated from the op's Rust
   definition (`src/ops/`, see the root `CLAUDE.md`): `scripts/gen_ops.py`
   writes it into `_ops_generated.py` — signature (the positional rule is
   derived, `gen_ops.positional`), defaults, docstring — and `Pipeline`
   inherits it from `_OpsMixin`. Every generated method appends through `Pipeline._append_typed` → `_push` →
`Plan.push`, which checks the input domain and applies the whole plan-time
effect in one `plan::step`: the schema fold (domain/dtype/ndim) and the shape hints (the op's
   symbolic `shape` for H/W, the channel rule for C).

   Hand-write a `Pipeline` method only as *sugar* over a generated one: an
   `internal` op (`#[op(visibility = "internal")]`) generates `_<name>`, and
   the sugar (`scale`'s `out_dtype`, `rasterize`'s `shape=`, `flip_h`) calls
   it. Validation that must precede the op goes before that call; work after
   the append (a `preserve_dtype` cast-back) reads the returned pipeline.

   **There is no op list to touch.** The ops are the Rust `Plan`, a frozen
object with no setter; the only way to add one is `Plan.push`, which plans
it whole. (The convention it replaced — each builder calling the update
methods by hand — let 41 of 60 builders skip the shape-hint half and publish a
planned schema execution could not produce.) Never build or edit a
`PlanState` by hand either; it follows from the op's Rust contract.

2. **`lazy.py`**: Nothing to add. `LazyPipelineExpr` generates a forwarder for
   every chainable `Pipeline` method at import time
   (`_install_pipeline_forwarders`), copying the signature so `inspect`/IDEs/the
   parity test see the real parameters, and a binary `lazy_only` op's method is
   generated into `_LazyOpsMixin`. Only the multi-operand `lazy_only` ops
   (`apply_mask`, `channel_merge`) are hand-written here. After changing the
   surface, regenerate the type stub with `python scripts/gen_lazy_stub.py`.

3. **Schema inference**: nothing to add in `_types.py` or the planner. The
   op's domain, dtype, rank and channel effects are read at planning time from
   its Rust contract via `Plan.push`, and the optimisation passes read
   its spatial and identity rules in Rust (`passes.rs`), so make sure the op declares the right contract on
   the Rust side (next step). Do not add per-op special cases in Python —
   `test_op_schema_authority` and the batch-fold conformance tests in
   `test_sanitation.py` guard this.

4. **Rust side**: Map the operation name in `resolve_op` to a `GraphStep`
   (buffer ops wrap a view-buffer `ViewDto`; graph-level steps get their own
   variant) — see [`polars-cv/src/AGENTS.md`](../../src/AGENTS.md)

### Affine Operations (not fused)

`warp_affine()`, `shear()`, `rotate_and_scale()` and `rotate()` each execute as
their own op: there is **no plan-time fusion** of adjacent affine ops. An
`affine_fusion` Tier-1 pass used to compose runs of them into one warp, and was
removed because folding several interpolation passes into one (and dropping the
intermediate clip of an `expand=False` rotate) changed pixels by up to ~185/255,
breaking the byte-for-byte on/off guarantee every optimization carries (see
[`_optimize.py`](_optimize.py) and the CHANGELOG). `test_removed_surfaces.py`
pins the deleted `_fuse_affine_inplace`/`_compose_affine_ops`, so do not restore
them. `shear()` and `rotate_and_scale()` build their matrix (the literal
rotation matrix via the `rotation_matrix_2d` FFI) and delegate to
`warp_affine()`; `_to_spec_dict()` emits ops verbatim.

### Shape Hints (single authority: view-buffer `OpShape`)

No per-dimension geometry is derived in Python. Every op's shape arithmetic is
one view-buffer `OpShape`, which execution evaluates on known sizes and `plan::step` evaluates symbolically: each typed op builds its `OpShape` from its
own fields (its family's generic `shape()` on the `Wire` op), a per-row field as `Sym::PerRow` and an unknown
input size as `Dim::Input(k)`. So the tracked H/W cannot disagree with what
the op produces, and no placeholder value stands in for a per-row one
(`a_lowered_op_keeps_its_shape` holds it to the executed step's where an op lowers).

Not every step *has* an inferable shape: axis reductions, histograms, channel
merge and the binary ops are graph-level steps with no `OpShape`. For those
the H/W hints are **invalidated**, not carried forward — several of them do
change H/W, and keeping the pre-op values is how a pipeline came to publish
`[100, 200, 2]` for data that executes as `[200, 3, 2]`. Unknown is always
safe: the output publishes no shape and a typed sink asks for an explicit
shape. A per-row parameter leaves unknown exactly the axes it decides
(`resize(height=pl.col("h"), width=100)` plans `[?, 100]`); a per-row rotation
angle is known only for a square input.
Channels come from the channel rule and rank from `plan::fold`; `plan::step` applies all three and clips the hints to the output rank.

An unknown input rank normally means "do not ask": there is no shape to reason
about, and a fabricated one publishes a fabricated result. The exception is
a step that *builds* a buffer out of a non-buffer domain — `input_domains`
excludes buffer, `output_domain` is buffer — whose output geometry comes from
its own params and reads no input at all. `rasterize` is the case, and
`plan::input_dims` recognises it from the contract rather than by name. Without
it a fully determined mask published no shape, and `sink("array")` demanded an
explicit one.

A `source()` or `.sink()` parameter that the chosen format never reads is
rejected. Each source and sink format is a typed Rust struct carrying exactly
the fields its decode or encode reads (`src/formats/`, each
`deny_unknown_fields`), and the builder validates what the caller passed
against that definition (`Plan.with_source`, and `check_graph` for the whole graph at
`.sink()`) — the deserializer the graph itself uses — so an unknown, misspelled or inapplicable keyword is refused while the
pipeline is built, naming the formats it does apply to. `source()` sends
exactly the keywords the caller passed (read from its own `locals()`; every
keyword defaults to `None`, so passed means not `None`), and
`thumbnail()` validates the spec it writes the same way. Do not add a per-parameter check beside it — that is what
produced one raise, one warning and five silent drops on the source side, and an
open keyword surface on the sink side.

`source("contour")` decodes the column to the contour domain and nothing
more. Its canvas keywords (`width`/`height`/`shape`/`fill_value`/`background`)
append the `rasterize` op, so there is one route to a mask and it is planned
and executed as that op (`TestContourSourcePlanTimeContract`).

## Common Pitfalls

- **Don't mutate Pipeline in place.** Always use `_clone()` then modify the clone. The immutable pattern is intentional.
- **Give new ops a correct Rust contract (`ViewDto` op / `GraphStep`).** Schema inference
  reads dtype/domain/rank/channel from it at planning time; a wrong or missing
  contract makes planned and executed schemas diverge (caught by the
  plan==exec tests in `test_sanitation.py`).
- **Expression params must be slotted.** An expression reaches Rust only as a slot in the pipeline's table (`_slot()`, which `_encode_field` and `_wire()` call); a typed op's fields get this from their catalogue type.
- **The `auto` dtype.** Sources like `image_bytes` and `file_path` have dtype `auto` because the actual dtype is only known at execution time (after decoding). Operations that need a known dtype (like `sink("list")` or `sink("array")`) must have it resolved before the sink, either via `source(..., dtype="f32")`, `.cast(...)`, or a dtype-fixing operation. `contour` is *not* one of them — it decodes f64 coordinates and `rasterize` fixes u8, so it rejects a `dtype=` assertion rather than accepting one it never reads.
- **Continuation nodes must inherit upstream typing context.** In `LazyPipelineExpr.pipe()` for op-only continuation pipelines (a plan with no source), compute node domain/dtype/ndim using upstream state + new ops. Copying op-only pipeline typing state can cause contract drift (planned dtype mismatch at execution).
