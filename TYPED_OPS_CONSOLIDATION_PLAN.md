# Plan: consolidate the typed-op migration (follow-up to PR #99)

> **Status ledger — keep current.** Continues `TYPED_OPS_PLAN.md` (P0–P10) on
> branch `claude/pr99-refactor-review-htrbfy`, which starts at the PR #99 head
> (`0f22dd4`). Each phase **deletes first**: the old or duplicate mechanism is
> removed, the build or suite goes red for exactly the reasons the ledger
> predicts, and the desired mechanism is then wired in until everything is
> green again. A phase's commit contains both halves; the ledger rows are
> ticked in the same commit.
>
> **Handover:** `HANDOVER.md` (repo root) summarises the state after C5 and the
> next steps for a new agent.
>
> | Phase | Status |
> |---|---|
> | C0 — Correctness fixes (test-first) | **done** — C0.1, C0.2, C0.4–C0.7 fixed test-first (`tests/test_plan_claims.py`, docstring binding guard); C0.3 moved to C2 |
> | C1 — Typed planner state | **done** — `PlanState` is a frozen Rust pyclass (`Domain`, `PlannedDType`); Python dataclass, `HINT_DIMS`, `Domain` mirror, `Domain::Any`, string dtype helpers deleted; binary ops plan over both states |
> | C2 — Declarations are ops; Rust plans the graph | **done** — `assert_shape` op checked per row; `planned` wire field, `plan_assert`/`plan_sink`/`check_sink`, `fold_output_*`, `asserted`/`declared` flags, Python assertion machinery deleted; `.sink()` runs `check_graph`; one refs mechanism for node reads |
> | C3 — Rust owns the op list (`Plan`) | **done** — `Plan` frozen pyclass (`push`/`select`/`with_source`/`rebased`/`continuing`/`run_pass`/`to_spec`); `OpSpec`/`ParamValue`/`SourceSpec`/`planning_slots`, `_push_op`/`_append_op`/`_replay`/`_state_at`/`_entering`/`_Position`/`_STATE_COPIERS`/`_copy_state_from`/`_create_sub_pipeline`/`_track_expr`, `plan_step`/`plan_source`/`node_pass` FFI and the AST/copy-table guards deleted. Deviations: one `select(positions, start)` serves slice, reorder and deletion; CSE compares the ops' wire form over the graph's slot table in Python (the slot table is Python's), not a Rust `common_prefix_len`; `_to_python` stays (numpy scalars at encode) |
> | C4 — One op definition (mode-generic ops) | **done** — C4a: rank/channel rules deleted, read off `OpShape`. C4b: every wire op is a variant of a mode-generic engine family (`ImageOpKind<M>`, `ComputeOp<M>`, `ViewOp<M>`, `ColorConvertOp<M>`, `ConvolveOp<M>`, `GeometryOp<M>`, `ReductionOp<M>`, `HistogramOp<M>`, `PerceptualHashOp<M>`) or of `GraphOp<M>`; `OpDef` and every per-op typed struct deleted. C4c: `TypedOp = GraphStep<Wire>`, every `Op` rule generic over the mode; `ParamCtx::planning`, `planning_value`, `planning_step`, `Family` deleted. Deviations: rotate keeps its lowering, now inside `ViewExpr::apply_op` (`a_lowered_op_keeps_its_shape`); `check_rank` still passes 1 for an unknown size (filtered to rank-only verdicts) |
> | C5 — Geometry namespaces on the typed ops | **done** — `ContourFn`/`PointFn`/`BBoxFn` families (`src/geom_fns.rs`) and the `GeometryOp` variants for op-backed accessors (`OP_ACCESSORS`); `ContourKwargs`/`PointKwargs`, call-site defaults, `_ArgBinder`, `_enum_or_expr`, hand-written accessor methods deleted; methods generated from `geom_catalog.json`; literals checked at build by `check_geom_call`. Decision: one `origin` default, `"centroid"` (migration page). Also: `t` in `point.interpolate` keyword-only |
> | C6 — One registry, one default convention | **done** — C6a: the wire applies `#[param(default)]` (an optional field with a default is a compile error); `Visibility` enum, named as a variant in the attribute. C6c: a contour source only decodes (to the contour domain); `source("contour", width=, …)` appends `rasterize()`; the source's canvas fields, `fill`, `decode_contour_source*`, its node-canvas read and the source-level null handling deleted. C6b: `Source`/`Sink` are `#[derive(Ops)]` enums (a family with no per-row value has no mode); `formats!`, `#[derive(Op)]`, `OpFields`, `op_desc`, the per-parse catalogue, `DEFAULT_QUALITY` and the sink `_ =>` arms deleted. C6d: `SourceFormat` deleted; `Source::route` turns `auto` into the concrete source per batch and the row loop matches on `Source`. Deviations (user decisions): the contour source does not *carry* a Rasterize — it stops rasterizing, and a bare `source("contour")` is the contour domain; no `registry!` macro — the derive is the one registry mechanism and `typed_ops!` only places families in `GraphStep` |
> | C7 — Python surface fully generated | not started |
> | C8 — Sweep, guards and docs | not started |

## Why

The review of PR #99 (see "Findings" at the end) found that the migration moved
facts from Python into Rust, but kept or re-created a second copy of most of
them: each op is declared twice in Rust (typed struct and engine variant), its
shape is built twice, plan-time rules come from resolving ops against placeholder
values, the planner state and its bookkeeping are mirrored in Python, execution
re-folds the schema with a second planner, and several vocabularies survive as
Python mirrors. Hand-written code fell by 320 lines (−0.6%); tests grew by 1,192.

## Target end state

1. **One definition per op.** A typed op struct carries the wire form, the
   catalogue (docs, defaults, per-row eligibility) and — through a mode-generic
   build — both the executed engine op and its plan-time symbolic form. Rules
   (domain, dtype, rank, channels, shape, identity, spatial) are implemented once,
   generically over the mode. No placeholder values, no second shape constructor.
2. **One planner, in Rust.** Plan state is a typed Rust value (`Domain`,
   `PlannedDType`, `Vec<Dim>`), exposed to Python as a frozen object. Declarations
   (`assert_shape`, a canvas from another node) are ops. The graph loader re-plans
   the graph from its ops; no plan state crosses the wire; execution has no second
   schema fold.
3. **Rust owns the op list.** Python's `Pipeline` holds a Rust `Plan` plus its
   expression table and graph references; there is no Python `OpSpec`,
   `ParamValue`, `PlanState`, replay or state-copy machinery.
4. **One registry, one default convention, one generator.** Ops, sources and
   sinks register through one macro; a default is declared once and both the
   Python signature and serde use it; every Python method (ops, sources,
   geometry namespaces, lazy forwarders, stubs) is generated.

## Rules for every phase

- **Delete first, then wire.** Start each phase by deleting every row of its
  ledger section. Let the compiler, the type checker and the suite report what
  depended on it; wire the replacement only for those call sites.
- **Test-first for behaviour.** A bug fix or a new rule starts with a failing test
  at the user-facing entry point, watched failing for the stated reason.
- **Each deletion is guarded.** Rust removals are guarded by the compiler. Python
  and prose removals are added to `polars-cv/scripts/check_removed_symbols.py`
  with the reason. A test deleted because its property became structural says so
  in the commit message.
- **Gate:** `scripts/verify.sh` (full) green at every phase exit, golden corpus
  unchanged unless the phase says which entries change and why.

---

## C0 — Correctness fixes (test-first, no structural change)

| # | Failing test first (user entry point) | Fix | Deleted / changed |
|---|---|---|---|
| C0.1 | `source("blob", dtype="u8").cast("u8")` over an f32 blob returns u8 with optimizations on (today: fails with a planner-blame error) | The blob decoder checks the declared dtype against the blob header and refuses a mismatch naming both, so the claim is a fact identity elimination may rely on | `BlobSource.dtype` doc "for the planner" → "checked at decode"; `graph/compiled.rs` blob branch passes the declared dtype |
| C0.2 | `source("contour", shape=<node with assert_shape>)…pad_to_size(...)` gives the same output with optimizations on and off, and a wrong upstream assertion is reported as the user's assertion | `source(contour, shape=)` records its canvas through the same path as `rasterize(shape=)` (a `by_user=False` declaration), so `declared` is set | **Delete** the `dataclasses.replace(new._state, dims=…)` block in `Pipeline.source` (`pipeline.py:931-937`) |
| C0.3 | *moved to C2 (C2.sink)*: the histogram-bucket encoding outranks the (domain, sink) pair and is known only from the op list, which `plan_sink` does not see | — | — |
| C0.4 | `plan_assert` with `{"size": -5}` or `0` is refused | `Declared::Size(u32)` + non-zero check in Rust | **Delete** the positivity branch of Python `_asserted_rank` (Rust is the authority) |
| C0.5 | `grayscale().channel_select(2)` raises at build | `check_rank` passes known sizes; when the input shape is fully known every `validate` failure is raised at build, not only rank-level ones | **Delete** the "unknown sizes are passed as 1" placeholder for a fully known shape |
| C0.6 | `repr()`/`explain()` of `source(...).resize(4,4)` does not print a fictitious `assert_shape(...)` | `__repr__` renders the assertions that were written, where they were written | **Delete** the `known = [...]` block in `Pipeline.__repr__` that renders inferred dims as `assert_shape` |
| C0.7 | Every `>>>` example in the generated docstrings (`_ops_generated.py`) and hand-written `Pipeline` docstrings binds against the real signatures | Extend `test_documented_pipeline_calls_bind` to docstrings; fix `src/ops/color.rs:19` (`convert_color("rgb", "hsv")`) | the stale example |

---

## C1 — Typed planner state

**Delete first:**

| What | Where | Replaced by |
|---|---|---|
| `plan::State { domain: String, dtype: String, dims: [Option<i64>; 3] }` string fields | `polars-cv/src/plan.rs:31` | `State { domain: Domain, dtype: PlannedDType, ndim, dims, … }` as a `#[pyclass(frozen, get_all)]` (Python sees the wire spellings via getters) |
| Python `PlanState` dataclass and `PlanState.of` | `polars_cv/pipeline.py:181-226` | the Rust `State` object itself |
| `plan.rs` `"auto"`/`"scalar"`/`"vector"` string matches (`fold`, `single_input_dtype`, `binary_dtype`, `check_sink`) | `plan.rs:245-275, 380, 475` | matches on `Domain` / `PlannedDType` |
| `lib.rs::output_dtype_for`, `dtype_short_name`, `parse_dtype` (string-typed dtype lattice) | `polars-cv/src/lib.rs:70-145` | `OutputDTypeRule::resolve_planned` (the one lattice, already in view-buffer) |
| `OutputSpec.expected_domain: String`, `expected_dtype: String` | `graph/types.rs:33-35` | `Domain`, `PlannedDType` |
| `SinkKind::resolve` string match `("buffer", …)` | `graph/sink_kind.rs:89-110` | match on `Domain` |
| Python `Domain` enum | `polars_cv/_types.py:167` | nothing (`current_domain()` returns the string); `gen_ops.NOT_GENERATED["Domain"]` deleted |
| `test_enum_parity_domain`, `_RUST_INTERNAL_DOMAINS` | `tests/test_sanitation.py:722-745` | structural (no Python copy) |
| `test_domain_vocabulary_declared_once` | `tests/test_append_contract.py:352` | structural |
| `HINT_DIMS` (Python) | `polars_cv/_types.py:654` | `State.DIM_NAMES` class attribute from Rust `DIM_NAMES` |
| `Domain::Any` (a wildcard variant no pipeline can be in; the reason `Domain` could not be generated) | `view-buffer/src/ops/mod.rs:88` | `GraphStep::output_domain(input)`; `Domain::accepts` deleted |
| `decode::parse_dtype_str`, `dtype_str_to_polars` | `graph/decode.rs:191, 565` | `list_array_inner_dtype(PlannedDType)` |
| Python `dataclasses.replace` edits of the state (`LazyPipelineExpr.pipe` reset `asserted`; lazy `_continuation`/`_binary_op` dropped the sizes) | `lazy.py:210, 670, 690` | the upstream state as Rust planned it |
| `test_the_ffi_agrees_with_the_planner` (compared `plan_step` with itself through a fabricated state) | `tests/test_alpha_channel.py:541` | structural |

**Wire:** `plan_step`/`plan_source`/`plan_assert` take and return `State`;
`tests/_plan_view.py` reads attributes (the seam absorbs the change).

---

## C2 — Declarations are ops; Rust plans the graph

**Delete first:**

| What | Where | Replaced by |
|---|---|---|
| `planned` wire field, `WireOutput`, `OutputSpec`'s `Deserialize` | `graph/types.rs:20-99` | `OutputSpec` computed by `UnifiedGraph::from_json` from the planned graph |
| `OPEN_STRUCT_EXEMPT["types.rs::OutputSpec"]` | `tests/_kwargs_scan.py:52` | nothing (no open struct) |
| `plan_assert` FFI, `plan::Assertion`, `plan::Declared` (as a side channel) | `plan.rs:55-150, 340` | an `assert_shape` typed op (`AssertShape { ndim, dims }`, internal) planned by `plan::step`; identity at execution, where it checks the declaration |
| Python `_assertions`, `Assertion`, `_new_assertion`, `_assertion_window`, `_apply_assertions_at`, `_asserted_rank`, `_canvas_of` | `polars_cv/pipeline.py` | `assert_shape()` appends the op; `rasterize(shape=)` / `source("contour", shape=)` read the referenced node's state in Rust |
| `passes::Node.assertions` and the "assertion boundary" special cases | `passes.rs:92-200` | the `assert_shape` op is `IdentityRule::Never` and not `Pointwise`, so both passes treat it like any other op |
| `fold_output_rank`, `fold_output_dtype`, `op_json`, `resolve_one_output_spec`'s folds, the `input_dtypes.first()` fallback | `graph/compiled.rs:1566-1745` | `plan::step` replayed from a source state derived from the column dtype (`plan::source_state_for_column`) — the builder's planner, not a second one |
| `histogram_buckets` post-construction mutation on `OutputSpec` | `graph/types.rs:53, 381` | read off the planned node's final step |
| `plan_sink` FFI and `plan::check_sink` (a second copy of the sink checks in `graph/decode.rs:565-745`, without the (domain, sink) table) | `plan.rs:356-430` | `.sink()` validates by loading its graph through `UnifiedGraph::from_json` + `SinkKind::resolve` — the code the plugin runs (C2.sink: `…perceptual_hash().sink("numpy")` raises at `.sink()`) |
| `resolve_op_from_json` op → JSON → op round trips | `lib.rs:99`, `compiled.rs:1686/1723`, `types.rs:381`, `passes.rs:181/374` | typed `TypedOp` values throughout |

**Wire:** `UnifiedGraph::from_json` plans every node in topological order
(source state, then each op via `plan::step`, with `NodeRef` fields reading the
referenced node's planned state). `validate_output_schema` compares against that
plan. The migration page's "Hand-built graph JSON" section loses `planned`.

---

## C3 — Rust owns the op list (`Plan`)

**Delete first:**

| What | Where | Replaced by |
|---|---|---|
| `OpSpec`, `ParamValue`, `SourceSpec`, `planning_slots`, `_to_python` | `polars_cv/_types.py` | `Plan` (frozen pyclass) holding the typed `Source`, the typed ops (local slots) and the entering `State` per op |
| `Pipeline._ops`, `_entering`, `_Position`, `_state_at`, `_push_op`, `_append_op`, `_replay`, `_STATE_COPIERS`, `_copy_state_from`, `_same` | `polars_cv/pipeline.py` | `Plan.push(op_json, refs)`, `Plan.slice(a, b)`, `Plan.reorder(order)`, `Plan.final_state`; a pipeline is `(plan, exprs, graph refs, policies)` copied by constructing a new one |
| `node_pass`'s JSON-list + state-list FFI | `passes.rs:92-130` | `Plan.run_pass(name)` |
| `plan_step` double parse (`plan.rs:161-162`) | — | ops parsed once, at `Plan.push` |
| Hand-built `PlanState(...)` in `LazyPipelineExpr._continuation` / `_binary_op` (drop dims) | `polars_cv/lazy.py:670, 690` | `Plan.rebase(state)` / a binary push reading both operands' states |
| CSE's Python `OpSpec ==` prefix compare | `polars_cv/_graph.py:355-383` | `Plan.common_prefix_len(others, slot maps)` over canonical typed ops |
| Tests whose property becomes structural: `test_op_append_is_structurally_exclusive`, `test_pipeline_state_copy_is_complete`, `test_every_pipeline_field_survives_a_copy`, `test_replay_takes_its_assertions_explicitly`, `test_a_slice_replays_the_states_it_keeps`, `test_push_op_applies_the_whole_plan_step_unconditionally`, `test_python_holds_no_copy_of_the_channel_rule_arithmetic` | `tests/test_append_contract.py` | the `Plan` API (there is nothing in Python left to scan) |

---

## C4 — One op definition (mode-generic ops)

**Decision (C4b): combine.** The wire struct is folded into the engine struct:
the typed op *is* the engine op. Recorded here as answered.

Two stages, each deleting first and each green on its own.

### C4a — Rank and channels are read off `OpShape`

Every step has an `OpShape` (the graph-level ones too), and the planner reads
the output rank as the shape's length and the channel count as its axis 2.
The two declared rules that restated those facts go.

**Delete first:**

| What | Where | Replaced by |
|---|---|---|
| `OutputRankRule`, `OutputChannelRule` (enums, `apply`) | `view-buffer/src/ops/shape_rule.rs` | `OpShape::rank(inputs)` (the length of `dims`) and axis 2 of `OpShape::dims` |
| `Op::output_rank_rule`, `Op::output_channel_rule` and every impl | `view-buffer/src/ops/{traits,image,compute,color,filter,view,binary,reduction,histogram,phash}.rs`, `geometry/ops.rs` | — |
| `ViewDto::output_rank_rule`/`output_channel_rule`, `GraphStep::output_rank_rule`/`output_channel_rule` | `ops/dto.rs`, `graph/step.rs` | `GraphStep::shape` (total: every variant has a shape) |
| The rule/shape parity tests (`parity_tests::check` and its probes) | `shape_rule.rs` | nothing to bind: one authority |
| `plan::fold`'s rank match, `dims[2] = channel_rule.apply(..)` | `plan.rs` | the shape |
| "graph-level steps have no `OpShape`, H/W invalidated" | `plan.rs`, AGENTS | reductions, histograms, hashes, merges and masks are sized by their shape |

### C4b — The typed op is the engine op, generic over a mode

`view_buffer::mode`: `trait Mode { type V<T>; type L<T>; fn sym(..) -> Sym<T>; }`
with two modes — `Exec` (`V<T> = L<T> = T`, the default type parameter, so the
kernels read plain values unchanged) and `Wire` (`V<T> = Param<T>`,
`L<T> = Literal<T>`). A `Wire` op already says, per field, "known" or "per
row", so it is the plan-time view: no third mode and no placeholder values.
Each op is one struct in view-buffer, `Resize<M: Mode = Exec>`, carried by its
engine enum's variant; its rules (shape, dtype, domains, identity, spatial)
are implemented once, generically. `#[derive(Op)]` (moved to view-buffer with
`Param`, `Literal`, `FieldType`, the catalogue types) also derives
`Resize<Wire>::resolve(values) -> Resize<Exec>`. `GraphStep<M = Exec>`;
`TypedOp` holds `Wire` ops and wraps each into a `GraphStep<Wire>` (total, no
values), which the planner reads and execution resolves per row.

**Delete first:**

| What | Where | Replaced by |
|---|---|---|
| `OpDef::shape` (the typed second constructor) on all ops, `TypedOp::shape` | `polars-cv/src/ops/*.rs` | the engine op's generic `shape()` on the `Wire` op |
| `OpDef::resolve`'s hand-written field-by-field resolution on all ops | `polars-cv/src/ops/*.rs` | derived `resolve` (Wire → Exec) + one `step()` wrap per op |
| The duplicate per-op structs in polars-cv (`ops::image::Resize`, …) | `polars-cv/src/ops/*.rs` | the engine structs in view-buffer |
| `typed_shape_is_the_resolved_steps` | `ops/mod.rs` | structural (one struct, one shape) |
| `ParamCtx::planning`, `is_planning`, `WireScalar::planning_value` and every `if ctx.is_planning()` | `params.rs`, `ops/param.rs`, `ops/{filter,affine,geometry}.rs`, `view-buffer/src/naming.rs` | the `Wire` step (no values are invented) |
| `planning_step` | `lib.rs` | `TypedOp::step()` |
| `check_rank`'s placeholder sizes | `plan.rs` | `validate` over the `Wire` op where it reads only the rank and parameters |
| `rotate`'s `Rotation` split into three engine steps chosen from a value | `ops/affine.rs` | one engine `Rotate<M>` whose shape is `MaybeSwapHw`/`RotateExpand` from its own fields |
| `ImageOpKind::shape`-style per-family shape functions that restate the typed shape | `view-buffer/src/ops/*.rs` | the generic `shape()` of each op struct |

**Shape of the combine (settled at the start of C4b).** The engine enums are
the catalogue. Each wire op is one variant of its engine family enum
(`ImageOpKind<M>`, `ComputeOp<M>`, `ViewOp<M>`, …, and a polars-cv `GraphOp<M>`
for the graph-only ops: binary, mask, merge, declare, label), with the wire
field names and types, its doc comment as the Python docstring, and
`#[op(name = "resize", sample = {...})]`. `#[derive(Ops)]` on the enum
generates, per variant, the strict wire (de)serialization (unknown key and
missing required field refused, `Param`/`Literal` per field), the catalogue
entry, the slot visitor and `Resolve` (`Wire` → `Exec`). Kernels keep matching
`ImageOpKind::Resize { width, .. }` unchanged: `M` defaults to `Exec`, where
`M::V<u32>` *is* `u32`. `TypedOp` becomes `GraphStep<Wire>`: the planner reads
its rules directly and execution resolves it per row (static ops once), so no
per-op `OpDef` remains.

Additional deletions this adds:

| What | Where | Replaced by |
|---|---|---|
| The engine op enums' serde derives (a second, unused wire spelling of every op, by Rust variant name) and `test_fused_op_serialization` | `view-buffer/src/ops/*.rs`, `geometry/ops.rs`, `tests/integration_plan.rs` | the one wire (`#[derive(Ops)]`) |
| `polars-cv/src/ops/{image,compute,view,color,filter,affine,channel,geometry,reduce,histogram,phash}.rs` typed structs and their `OpDef` impls | polars-cv | the engine variants |
| `OpDef`, `OpFields`-per-struct, `TypedOp`'s per-op variants and `typed_ops!` lines | `ops/mod.rs` | the families' derives; `typed_ops!` lists families |
| Tuple/derived engine fields that are not the wire's (`Threshold(f64)`, precomputed histogram `edges`, …) | view-buffer | the wire fields, computed from at execution |

Transition: one family at a time; `TypedOp` holds a migrated family as one
variant (`Image(ImageOpKind<Wire>)`) beside the remaining per-op variants, so
the build is green between families. `GraphStep<M>` and the planner switch
come after the last family.

---

## C5 — Geometry namespaces on the typed ops

| What | Where | Replaced by |
|---|---|---|
| `ContourKwargs`, `PointKwargs` (bags of `Option` fields shared by every function) | `polars-cv/src/contour.rs:124`, `point.rs` | each plugin function deserializes its own typed struct (the op's, where one exists: `ContourArea`, `ContourScale`, …) |
| Per-call-site Rust defaults (`params.get(&kwargs.origin, ScaleOrigin::Origin, row)`, `signed` false, `dx` 0.0, …) | `contour.rs:821-1102`, `point.rs` | the struct's declared defaults |
| Hand-written `.contour`/`.point`/`.bbox` methods and `_ArgBinder` | `polars_cv/geometry/*.py`, `_namespace.py` | generated from the catalogue |
| Two defaults for one op: `.contour.scale(origin="origin")` vs `Pipeline.scale_contour(origin="centroid")` | `geometry/contours.py`, `ops/geometry.rs` | one default (**decision point**, recorded in the migration page) |

---

## C6 — One registry, one default convention

| What | Where | Replaced by |
|---|---|---|
| `formats!` (near copy of `typed_ops!`) | `polars-cv/src/formats/mod.rs` | one `registry!` macro used for ops, sources and sinks |
| `Option` + `unwrap_or` defaults on formats (`JpegSink::DEFAULT_QUALITY`, `ContourSource::fill` 255/0, `plan.rs:324` 255) | `formats/*.rs`, `plan.rs` | `#[param(default = …)]` applied by serde **and** the catalogue, for ops and formats alike |
| `ContourSource`'s copy of `Rasterize`'s fields | `formats/source.rs:96-121` | the contour source carries a `Rasterize` |
| `Sink::quality()/shape()/as_f16()` `_ =>` arms | `formats/sink.rs:102-131` | data on each typed sink |
| `SourceFormat` enum mirroring `Source` | `graph/compiled.rs:1276` | dispatch on `Source` |
| `check_applies` rebuilding the catalogue per parse | `formats/mod.rs:28` | a static catalogue |
| Stringly typed op `visibility` | `polars-cv-macros`, `ops/mod.rs` | a `Visibility` enum |

---

## C7 — Python surface fully generated

| What | Where | Replaced by |
|---|---|---|
| Hand-written `Pipeline.source()` body (per-field dispatch, `_validate_enum` of format/dtype, `decode_max_size` check, contour width/height/shape exclusivity) | `pipeline.py:706-940` | generated from `io_catalog.json`; `RasterSize` expresses the exclusivity |
| `_validate_enum`, `_enum_or_expr`, `_reject_expr` (once unused) | `_types.py:182-246` | Rust validation |
| Runtime lazy forwarders `_install_pipeline_forwarders`, `_make_forwarder`, `_chainable_pipeline_ops` | `lazy.py:809-880` | forwarders emitted by `gen_ops.py` |
| `scripts/gen_lazy_stub.py` and its separate stub pass | `polars-cv/scripts/` | `gen_ops.py` writes `lazy.pyi` |
| Hand-written "Domain: a → b" docstring prose | `polars-cv/src/ops/*.rs` | generated from each op's domain contract |
| `convolve2d(ksize=)` (determined by the kernel length) | `ops/filter.rs` | derived (**decision point**, breaking) |

---

## C8 — Sweep, guards and docs

- Remove every absence scan whose subject no longer exists; list each in the
  commit message with the structural mechanism that replaced it.
- `check_removed_symbols.py`: every symbol deleted above.
- `CLAUDE.md`, `AGENTS.md` (root, `src/`, `python/`), `docs/user-guide/migration.md`:
  the per-row rule restated as "no effect on rank or dtype; a size may be per-row
  and is then unknown at plan time"; module tables; the `planned` wire field gone.
- Line-count ledger per phase (hand-written plugin / engine / Python, tests,
  generated) against the PR #99 head.

---

## Line-count ledger

| At | plugin | engine | macros | py-hand | py-gen | tests |
|---|---:|---:|---:|---:|---:|---:|
| `main` before the migration `ac2e95b` | 16,986 | 20,732 | 0 | 16,078 | 0 | 53,972 |
| PR #99 head `0f22dd4` | 19,097 | 21,233 | 197 | 12,022 | 2,127 | 55,164 |
| after C5 `ac45f93` | 17,448 | 22,474 | 767 | 10,496 | 2,715 | 54,992 |
| after C6 `ded469c` | 17,254 | 22,501 | 653 | 10,465 | 2,715 | 55,111 |

---

## Findings this plan answers (review of PR #99)

A1 blob dtype claim + identity elimination → C0.1 · A2 contour canvas claim →
C0.2 · A3 (domain, sink) checked late → C0.3 · A4 negative assertion sizes →
C0.4 · A5 build-time knowledge unused → C0.5 · B1/B2 each op declared twice,
shape built twice → C4 · B3 planning placeholder → C4 · B4 rank/channel rules
duplicate `OpShape` → C4 · B5 stringly typed state → C1 · B6 two planners → C2 ·
B7 `planned` on the wire → C2 · B8/B9 Python bookkeeping and planning → C3 ·
C1–C11 mirrors → C1, C5, C6, C7 · D1–D4 catch-alls → C5, C6 · E1–E4 guards →
C0.7, C3, C4, C8 · F docs → C8.
