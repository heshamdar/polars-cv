# AGENTS.md — polars-cv

> **Read this file first when starting any task in this repository.**
> After making changes, update any relevant AGENTS.md file.

## Quick Navigation

| File | Scope |
|------|-------|
| [`GLOSSARY.md`](GLOSSARY.md) | One meaning per overloaded word (view, plan, node, step, graph, …) and the type names that use a word another way |
| [`polars-cv/python/polars_cv/AGENTS.md`](polars-cv/python/polars_cv/AGENTS.md) | Python API — pipeline builder, lazy composition, graph serialization, schema inference |
| [`polars-cv/python/polars_cv/geometry/AGENTS.md`](polars-cv/python/polars_cv/geometry/AGENTS.md) | Geometry subsystem — point/contour/bbox schemas and namespaces |
| [`polars-cv/python/polars_cv/metrics/AGENTS.md`](polars-cv/python/polars_cv/metrics/AGENTS.md) | Detection metrics — matchers, FROC/LROC/PR curves, bootstrap |
| [`polars-cv/src/AGENTS.md`](polars-cv/src/AGENTS.md) | Rust plugin — graph execution, source/sink encoding, op dispatch |
| [`view-buffer/AGENTS.md`](view-buffer/AGENTS.md) | Kernel library (the engine) — ViewBuffer, ViewExpr, operations, execution |
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

7. **Priorities, in order: correctness, ergonomics, maintainability, performance.** Performance is last on purpose — the plugin runs over frames the engine already carries, so a clear, obviously-correct formula beats a fast one. Reject a design on performance grounds only when it is clearly suboptimal (an unnecessary per-row allocation, an O(n) pass where O(1) exists), never to shave constants at the cost of the three above it.

## Architecture

The project has three layers:

```
┌─────────────────────────────────────────────────────────┐
│  Python: polars_cv                                      │
│  Pipeline builder, expression namespaces, DAG           │
│  construction, schema inference, validation, lazy       │
│  composition, geometry/metrics APIs                     │
├────────────── JSON graph serialization ─────────────────┤
│  Rust: polars-cv (the plugin)                           │
│  vb_graph expression entry point, graph execution,      │
│  source decoding, sink encoding, per-row parameter      │
│  resolution, cloud I/O                                  │
├────────────── Rust crate dependency ────────────────────┤
│  Rust: view-buffer (the engine)                         │
│  ViewBuffer, ViewExpr, stride-aware ops, kernel fusion, │
│  zero-copy interop (Arrow, ndarray, image)              │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. The user builds a `Pipeline` in Python, which internally creates a `PipelineGraph` (a DAG of `GraphNode`s).
2. `.sink(...)` on a `LazyPipelineExpr` serializes the graph to JSON and calls `_plugin.call("vb_graph", graph_json, input columns)`. That is the only route to `register_plugin_function`: it pins the exact `.so` and hands the plugin storage rather than extension types.
3. Polars calls the Rust `vb_graph` expression function with the JSON and any per-row expression parameters.
4. Rust deserializes the JSON into a `UnifiedGraph` and compiles it once into a process-wide cache (`graph/compiled/`: parsed spec, topological order, slot-bound params). Repeat calls (e.g. per streaming morsel) pay only a hash lookup. It then executes topologically, per row: decode source → apply ops (`ViewExpr`/`ViewBuffer`) → encode sink.
5. It returns a Polars `Series` (dtype depends on the sink: Binary, Float64, Struct, List, Array).

### Expression Namespaces

| Namespace | Purpose |
|-----------|---------|
| `.cv` | Image/array pipelines via `.pipe()` → `.sink()`, byte access (`.read_bytes()`), metadata (`.width()`, `.height()`, `.channels()`, `.image_dtype()`) |
| `.point` | Point geometry ops (normalize, distance, etc.) |
| `.contour` | Contour geometry ops (area, perimeter, IoU, matching) |
| `.bbox` | Bounding box ops (pairwise IoU, match detections) |

The three geometry namespaces also carry `on_null("raise"|"null")`, which says
what a null in a per-row expression parameter means. `.cv` deliberately has no
`on_null` — its parameters belong to a `Pipeline`, so the control there is
`Pipeline.on_null_param()`.

## Directory Structure

```
./                                  # Repository root (the Cargo workspace)
├── AGENTS.md                       # ← You are here
├── Cargo.toml                      # Workspace Cargo.toml
├── CONTRIBUTING.md                 # Release process, CI
├── view-buffer/                    # Rust kernel library (the engine)
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
│   ├── benchmarks/                 # Regression suite over the checkout
│   │   └── AGENTS.md
│   ├── examples/                   # Runnable demos (13 numbered 01–13 + detection_data.py)
│   ├── docs/                       # MkDocs user-guide documentation
│   └── scripts/                    # Generators (gen_ops.py, ...) and utilities
├── scripts/                        # Repo tooling: verify.sh (CI checks), with-pyo3-env.sh (PyO3 env for cargo), dev-clean.sh (reclaim disk)
├── benchmarks/                     # Published-wheel comparison against PIL, OpenCV and others
└── CHANGELOG.md                    # Keep-a-Changelog release history
```

See root [`CONTRIBUTING.md`](CONTRIBUTING.md) for the release process and CI, and
[`README.docker.md`](README.docker.md) for the Docker build environment. The two
`benchmarks/` directories (repository root and `polars-cv/`) are explained in
[`benchmarks/README.md`](benchmarks/README.md).

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
maturin develop                    # Build Rust plugin (debug) into .venv
uv run pytest tests/               # Run tests
uv run ruff check python/ tests/   # Lint Python
../scripts/with-pyo3-env.sh cargo clippy --workspace   # Lint Rust
```

Run cargo through `scripts/with-pyo3-env.sh` (or `source` it): it supplies the
PyO3 environment `maturin develop` sets, without which cargo and maturin
invalidate each other's builds and every switch rebuilds the polars stack. On
the web, the SessionStart hook exports it into every session shell.

Local x86_64 builds pick up `target-cpu=x86-64-v3` from [`.cargo/config.toml`](.cargo/config.toml)
(per-triple, not `[build].rustflags` — a global flag breaks aarch64 Darwin/`ring`).
CI and wheel jobs clear it with `RUSTFLAGS=""`.

## Known Issues

- **f64-accumulated chains stay unfused:** the FusedKernel computes in f32, so the float-promoting scalar family is correct-but-unfused for inputs whose `DType::accumulator` is f64 — f64 and the 32/64-bit integers. They execute correctly (in f64) but are excluded from kernel fusion (`view-buffer/src/kernels/elementwise/mod.rs::lower_to_scalars`).

Deferred work and verified defects are tracked in the root `ISSUES.md`, not here.

## Release History

Per-release changes (added/changed/fixed/performance) are tracked in
[`CHANGELOG.md`](CHANGELOG.md) — consult it rather than duplicating a running log
here.

## Durable Architecture Notes

These are the load-bearing design decisions worth internalizing before making
changes; they explain *why* the code is shaped the way it is.

- **Single schema authority (view-buffer).** Each op's schema effect — output
  domain, dtype, rank, and channel count — is declared once, on the op itself in
  Rust (`OpShape` in `view-buffer/src/ops/shape_rule.rs`: rank is its length,
  channels its axis 2; `OutputDTypeRule` for dtype), and applied in Rust by `plan::step` (once per appended op,
  `polars-cv/src/plan.rs`). A pipeline's ops live in its Rust `Plan`, with the
  state at every op boundary; the node-scope passes run on it in Rust too
  (`Plan.run_pass`, `polars-cv/src/passes.rs`). The planner contains no per-op
  special cases and no parallel contract table. Planning-time schema must equal
  execution-time schema; guarded by `tests/test_sanitation.py`.
- **Graph steps vs engine ops.** Graph-level steps (`GraphStep` in
  `polars-cv/src/graph/step.rs`: binary ops, masks, geometry, reductions,
  histograms, perceptual hash, lookup tables) are separate from
  engine-executable buffer ops (`ViewDto` in view-buffer). Anything that
  changes the data domain lives in `GraphStep`, and so does a buffer op that
  can fail on its *values* (`LutOp`: a value past the table): a `ViewDto` op
  cannot report a data error (its `validate` sees shapes and dtypes only, and
  a kernel panic is an engine bug), so such an op is its own family whose
  `execute` returns `Result`, as `HistogramOp`'s does.
- **Compiled-graph cache.** The `vb_graph` plugin compiles each graph once into a
  process-wide cache (`graph/compiled/`: parsed spec, topo order, slot-bound
  params, pre-resolved static ops). The streaming engine invokes the plugin per
  morsel, so repeat calls pay only a hash lookup, and graph structure is validated
  at compile time instead of failing late.
- **Kernel fusion.** Consecutive scalar compute ops (scale/relu/clamp/gamma/invert)
  plus casts fold into one `FusedKernel` pass (any-numeric read → f32 ops →
  out-dtype write). `out_dtype` is pinned to what the unfused chain would produce,
  so fusion never changes the planned schema. Promote-family inputs whose
  accumulator is f64 stay unfused (see Known Issues).
- **Rotation/affine unification.** `rotate()` with arbitrary angles routes through
  `ComputeOp::RotateAffine` → `AffineParams::from_rotation()` → `apply_affine_warp()`,
  sharing the affine code path; 90/180/270 stay zero-copy via `ViewOp`. Consecutive
  affine ops are not fused: each resamples in turn (the `affine_fusion` pass was
  removed because it changed pixels).
- **Lazy parity.** `LazyPipelineExpr` inherits a forwarder for every chainable
  `Pipeline` method, written as real methods into `_lazy_forwarders.py` by
  `scripts/gen_ops.py` from the built `Pipeline` (freshness:
  `test_the_committed_catalog_is_the_built_one`; drift:
  `test_lazy_pipeline_method_parity`). There is no stub: type checkers read the
  code.
- **One mandatory append path.** A `Pipeline`'s ops are its Rust `Plan`
  (`polars_cv._lib.Plan`), an immutable object Python cannot edit. The only
  ways to change it are its methods — `push` (an append), `select` (a slice,
  reorder or deletion), `with_source`, `rebased` (a continuation onto an
  upstream node) and `run_pass` — and each plans every op it keeps with
  `plan::step`: the input-domain check, the schema fold (domain/dtype/ndim) and
  the shape, before anything changes. Builders reach `push` through
  `Pipeline._push`; the lazy continuation is a `rebased` plan, which is what
  makes `.pipe(p.op())` and `.pipe(p).op()` agree by construction. Guarded by
  the frozen pyclass itself, the Rust `plan::tests`, and an eager/lazy parity
  sweep in `tests/test_append_contract.py` whose op table is
  completeness-asserted against the real chainable-op list.

  This replaced a convention where each builder made the update calls by hand.
  It failed the way hand-maintained sequences do: every builder ran the schema
  fold, only 19 of 60 also updated the shape hints, and the ratchet guarding it
  enumerated one of the two calls. Prefer a mechanism callers cannot step
  around over a test that lists what they must remember.

## Canonical Paths

The concrete list behind the "canonical paths are mandatory" principle in the
root [`CLAUDE.md`](CLAUDE.md#canonical-paths). Each row is a fact with exactly
one authority, the mechanism that owns it, and the guard that rejects a second
declaration. **Read from the authority; never restate it.** If you need
something the authority cannot express, extend the authority — do not open a
side channel.

| Fact | Single authority | Rejection mechanism |
|------|------------------|---------------------|
| Appending an op to a `Pipeline` (domain check + schema fold + shape, one `plan::step`) | The Rust `Plan` (`Plan.push`; every other rewrite is a `Plan` method that plans each op again) | Structural: `Plan` is a frozen pyclass with no setter, so Python holds no op list to edit (`test_the_plan_cannot_be_edited_from_python`); `select_plans_the_kept_ops_again_from_the_state_entering_them` |
| An op's shape / rank / channel / dtype / memory / spatial / identity contract | `Op` trait methods (generic over the mode; a graph op's rules are exhaustive over its `Role`), **no defaults**; identity rules never depend on parameter values (`OpShape::preserves` decides) | Compile error: a new op that omits one does not build; `test_op_schema_rules_are_required_not_defaulted` pins the no-default form |
| An op's accepted input domains | Rust `GraphStep::input_domains` (exhaustive — no catch-all arm), checked by `plan::step` | Structural: the domain vocabulary is `Domain::NAMED` (no wildcard variant), generated into Python like every registered enum; execution reads the same contract (`input_domains`, `operand_domains`) in `graph/row.rs`'s one buffer gather rather than restating it per arm |
| A graph node's input (decode a column through a source, or read a node) | `NodeInput` (`graph/types.rs`), written only by `Plan::node_spec` (`Plan.to_node`, through `Pipeline._to_node_dict`) | Structural: an enum, so a node cannot both decode and read a node, or do neither; `deny_unknown_fields` and `GRAPH_VERSION` refuse the version-1 fields by name (`the_version_1_node_form_is_refused`); `node_spec` refuses a source its input contradicts (`a_plan_writes_its_node_and_refuses_an_input_its_source_contradicts`) |
| A node's dependencies | Its input plus its ops' `GraphStep::operands` (Rust `GraphNode::dependencies`; Python `LazyPipelineExpr._dependencies` / `PipelineGraph._dependencies` read the same `operands()` through `Plan.operands`, plus `merge_pipe`'s `_merged` branches) | There is no list to restate them in (`an_operand_named_only_in_its_op_is_a_dependency`, `test_an_operand_is_a_dependency_once`, `test_a_node_depends_on_what_its_ops_read_not_on_stale_references`); every reference is checked once at parse (`check_references`, `a_dangling_node_reference_is_one_compile_error_for_every_reading_op`) |
| A step's reads of other nodes' outputs | `GraphStep::operands`, compiled once to plan positions and read per row by `RowOutputs::operands` (`graph/row.rs`) | Compile error: `RowOutputs`'s storage is private to its module and `run_step` receives only the operands already gathered; `a_null_operand_nulls_the_node_for_every_reading_op` sweeps every operand-reading sample op |
| An op's H/W effect | view-buffer `OpShape` (`Op::shape`, generic over the mode), read from the `Wire` op by `plan::step` and identity elimination | one generic definition per family; `a_lowered_op_keeps_its_shape` for an op that executes as another |
| Which ops exist | Rust `ops::TypedOp` (`typed_ops!` registry) → generated Python `TYPED_OPS` | `catalog_matches_the_committed_file` (the reviewed op set), `test_every_op_is_emitted_by_a_builder` (works with no `.so`); an unregistered name fails deserialization |
| A typed op's fields, types, defaults, docs and Python signature | Its variant of a `#[derive(Ops)]` family (an engine enum in view-buffer, or `GraphOp`), via `tests/golden/op_catalog.json` → `scripts/gen_ops.py` → `_ops_generated.py` | the derived wire rejects an unknown/missing/mistyped field and applies a declared default; `catalog_matches_the_committed_file` (Rust) and `test_the_committed_catalog_is_the_built_one` (built `.so` + generated module) |
| A kernel's AVX2 build (runtime CPU dispatch) | `view-buffer/src/core/dispatch.rs`: `SimdKernel` + `dispatch()` (whole body compiled with AVX2, never FMA) | In every debug build, `dispatch` runs both builds and asserts byte-identical output, so any test reaching a kernel checks it (`the_parity_check_rejects_outputs_that_differ`); CI builds for the wheels' baseline, where the two builds differ |
| Converting an element between dtypes (round/saturate rule) | `view-buffer/src/core/convert.rs`: `CastFrom`, the store of every kernel that computes a value (the element-wise engine, the warp, the blur, typed grayscale), applied in bulk as the `Convert` map (`convert_slice`, `convert_view`, so `cast_to`); `ViewBuffer::to_f16_bits` is the half-precision sink's conversion on top of it | `convert::tests` (ties, NaN, ±inf, saturation, every loop remainder; f16 bits against per-element `f16::from_f32` over binary16's edges, dtypes and layouts); `cast_round_tests` |
| How a per-value compute op runs (scalar ops, scale, relu, clamp, invert, gamma, contrast, normalize, fused chains) | `view-buffer/src/kernels/elementwise/`: `apply` lowers the op, then `strategy` picks, once: integer arithmetic (an integer affine kernel over 8/16-bit input into the same dtype, e.g. `invert`), a lookup table (costly or per-channel kernels over 8/16-bit input) or blocked f32 passes; each is an element map the one traversal (`core::map`) runs | `elementwise::tests` compares every op × dtype × layout (a channel-first view and images of several blocks included), shared and sole-owned, bit for bit with the pre-engine code kept verbatim as `legacy.rs`; `copy_counts.rs` pins which cases allocate |
| Lowering a compute op to scalar ops | `kernels::elementwise::lower_to_scalars`, read by both fusion (`expr.rs::try_fuse`) and standalone execution | There is no second lowering to disagree with; the fusion tests and the oracle parity tests both run through it |
| Running a per-value or per-pixel kernel over a buffer (in place or into a new one; how a view is read) | `view-buffer/src/core/map.rs`: a kernel is an `ElementMap` (value → value, told its run's position for per-channel maps) or a `PixelMap` (`C` channels → one value); `map_owned` writes in place when `unique_contiguous_mut` allows, else `map_new`/`map_pixels` write into spare capacity, reading any view in `Walk` runs (whole pixels for a pixel map), all dispatched. The engine's strategies, `cast`, `threshold`, `grayscale` and the f16 conversion are maps; statistics read `map::for_each_run` | `copy_counts.rs` (`per_value_ops_on_a_view_allocate_only_their_output`, `threshold_of_any_view_allocates_only_its_output`, `grayscale_of_any_view_allocates_only_its_output`, each watched failing at 2 allocations); `a_run_holds_whole_grains_in_every_layout` and the parity tests' channel-first layouts, watched failing with the grain ignored |
| Reading a view's elements in logical order (packing, `append_to`/`write_to`, every map over a view) | `view-buffer/src/core/strided.rs`: `Walk::of` coalesces the layout once into packed units, evenly spaced rows and outer axes (inline: no allocation); `copy_to` packs (`copy_elements_into`, so `to_contiguous` and every list/array sink), `for_each_run` hands out runs, each a whole number of a given grain (a pixel's channels), calling its consumer directly, never through `dyn`, so it inlines into a dispatched build | `strided::tests`: the coalescing fixture table, a random-view property test (rank 1–4 permutes/flips/slices × u8/u16/f32/f64) against a per-element reference, and the grain test, all watched failing against mutations; a debug-build bound check on every walk |
| Handing a buffer to fast_image_resize (every resize op) | `view-buffer/src/interop/fir.rs`: `FirViewAdapter<P>` (`ExternalLayout::FastImageResize` = `LayoutFacts::is_dense_rows`), rows from `ViewBuffer::dense_rows`; `kernels/resize.rs::resize_pixels::<P>` packs only what the adapter refuses | `tests/it/resize_views.rs`: crop / flip_v / both × u8/u16/f32 × rank 2 and 1–4 channels × three filters against fir's own `ImageRef` over the packed pixels, watched failing against mutated row reads; `copy_counts.rs` (`resizing_a_view_with_packed_rows_allocates_only_its_output`); the layout fixture `fast_image_resize_takes_any_view_with_packed_rows` |
| Why an `unsafe` block is sound | A `// SAFETY:` comment on the block, stating the invariant it relies on and the code that establishes it (for FFI and allocator sites, the polars / pyo3 / arrow contract) | `clippy::undocumented_unsafe_blocks = "deny"` in `[workspace.lints.clippy]`, inherited by every member (`[lints] workspace = true`); `test_every_unsafe_block_needs_a_safety_comment_in_every_member` holds the inheritance |
| Whether a buffer may be written in place | `ViewBuffer::unique_contiguous_mut` (contiguous, own Rust allocation, sole owner) | `copy_counts.rs` in-place cases (`a_shared_u8_invert_allocates_its_output_once` pins the refusal) |
| The runtime `DType` → element-type match | `with_dtype!` in `view-buffer/src/core/dtype.rs` (exhaustive, no default arm) | Compile error when a dtype is added; `test_no_second_dtype_spelling_table` |
| Every spelling of a dtype (short / VIEW wire code / numpy), and its cast-only source names (the last column of a row: Polars `Boolean` -> `u8`) | `dtype_table!` in `view-buffer/src/core/dtype.rs` | `dtype_single_authority.rs` + `test_no_second_dtype_spelling_table` (a partial dispatch is reported) |
| Enum variant names crossing the FFI | `named_variants!` + `naming::REGISTRY` (engine) chained with `naming::PLUGIN_REGISTRY` (plugin-owned enums: `RowErrorPolicy`, `NullParamPolicy`, `FetchErrorPolicy`) | `every_named_enum_is_registered` (a `NAMED` table not in the registry fails), `registered_enums_have_unique_names`, `plugin_enums_have_unique_names`, `plugin_enums_do_not_shadow_engine_enums`; the Python classes are generated from the registries (`enum_catalog.json` → `gen_ops.py`, bar the stated `NOT_GENERATED` exceptions), pinned by `enum_catalog_matches_the_committed_file` and `test_the_committed_catalog_is_the_built_one` |
| A graph policy's wire spelling (`on_error`, `on_null_param`) | Its `NAMED` table, read by `ops::param::literal_field` (no serde derive on the enum) | `graph_policies_parse_through_their_named_tables`; there is no second spelling to compare |
| An op's domain transitions (the "Domain:" docstring line) | Its Rust `input_domains`/`output_domain`, read by `catalog_json` over the op's structural choices → `domains` in `op_catalog.json` → the generated line; sugar composes its declared ops' (`@_sugar`, `polars_cv/_domains.py` renders both) | `the_domain_contract_is_read_off_the_op`, `test_sugar_appends_exactly_the_ops_it_declares`, `test_every_public_pipeline_method_has_one_kind` |
| A parameter's default | `#[param(default = ...)]` on the field: the wire applies it when the field is absent, and the catalogue gives it to the generated signature (an `Option` field declares none) | `an_absent_field_takes_its_declared_default`; the derive refuses an `Option` field with a default |
| Source/sink formats, and which parameters each reads | One variant per format of the `Source`/`Sink` families in `polars-cv/src/formats/` (`#[derive(Ops)]` → `tests/golden/io_catalog.json` → generated `SourceFormat`/`SinkFormat`); the builder checks what the caller passed with `Plan.with_source` (which also plans the source's state), and `.sink()` checks the whole graph with `check_graph` (the plugin's own compile, planning and `decode::output_schema`) | Deserialization: an unknown format or field is refused, and a field the format does not read names where it applies (`formats::from_wire`); `io_catalog_matches_the_committed_file` and `test_the_committed_catalog_is_the_built_one`; `test_param_applicability.py` sweeps parameter × format grids from the catalogue and checks the `quality` claim against the encoders |
| `LazyPipelineExpr`'s method surface | generated by `gen_ops.py`: forwarders from the built `Pipeline` into `_lazy_forwarders.py`, and every `lazy_only` op into `_LazyOpsMixin` | `test_lazy_pipeline_method_parity`, `test_every_lazy_only_op_is_a_generated_lazy_method`, `test_the_committed_catalog_is_the_built_one` |
| Which builder parameters are positional | Derived, never declared: `gen_ops.positional` (an op's only required field is positional-or-keyword, the rest keyword-only) | `tests/golden/signatures.json` (`test_the_call_surface_matches_the_snapshot`); `test_documented_pipeline_calls_bind` binds every documented `Pipeline()` call; the derive refuses a `#[param(positional)]` key |
| An output's planned schema | Planned by Rust from the graph itself (`resolved_output_specs`: each node's source state, then `plan::step` per op); the wire output is only `OutputRequest { node, sink }` | `an_output_carries_only_its_node_and_sink` (a `planned`/`expected_*` field is refused), `output_facts_are_planned_from_the_ops` |
| A shape declaration (`assert_shape`) | The internal `assert_shape` typed op: planned by `plan::declare`, checked per row by `GraphStep::AssertShape` | `a_declaration_the_state_contradicts_is_refused`, the `AssertShape` case of `every_graph_step_variant_executes` |
| The graph wire format's node fields | `GraphNode` with `#[serde(deny_unknown_fields)]` | Deserialization error — a stale or misspelled key fails the query |
| What a path reader (`file_path` source, `read_bytes`, header metadata) may read, including what an absent `allowed_roots` means | `fetch::PathPolicy::resolve` (the only constructor; no `Default`), required by `Fetcher::new` / `HeaderFetcher::new`; the three readers resolve before reading any row. Absent denies local paths and remote URLs alike (#180) | Compile error for a reader without a policy; `tests/test_path_sandbox.py` sweeps all three surfaces for absence (local, and `s3://`/`gs://`/`https://`), `["*"]`, `[]`, the mixed list and the denial (each watched failing against the old open default); `an_absent_allowed_roots_is_refused_naming_the_option_and_both_ways_out` |
| How `token_command` runs | `cloud_auth::run_token_command`: `Command::new(argv[0]).args(rest)`, no shell on any platform; the wire is a JSON argument vector parsed by `cloud::parse_token_command` | `token_command_arguments_reach_the_program_literally` (`;` and `$(...)` stay text), `token_command_refuses_everything_but_an_argument_vector` (the old string is refused), Python `CloudOptions` raising on a `str` |
| Null parameter handling | `NullParamPolicy` on `ParamCtx`, via `ParamCol::on_null` | Reviewed by hand: never add per-op or per-parameter null keywords |
| What a `(domain, sink format)` pair produces | `SinkKind::resolve` in `src/graph/sink_kind.rs` | Compile error: the four halves of the sink contract (`dtype_for_output`, `encode_node_output`, `null_row_result_for_spec`, `build_series_from_spec`) match on the enum, so a new kind is non-exhaustive in all four at once; `every_kind_is_produced_by_some_pair` rejects a kind no pair names |
| Which files a source-scanning guard reads | `tests/_discovery.py` — every accessor raises rather than returning empty | `test_scans_go_through_discovery` (AST walk: a direct `glob`/`rglob` in `tests/` fails unless the file is in `_DISCOVERY_EXEMPT` with a reason), `test_discovery_fixtures.py` |
| Which guards exist, and which rule each holds | This table: its third column is the index of guards (`tests/_guard_index.py` parses it) | `test_every_guard_the_index_names_exists` (a cited test, test file or Rust guard that resolves to nothing fails); `test_every_source_scanning_test_file_is_in_the_index` (a `test_*.py` that reads source must be cited by file or by one of its tests); parser fixtures in `test_guard_index.py` |
| The `rotate_and_scale` matrix | `AffineParams::rotation_matrix_2d` (view-buffer), evaluated per row when `ComputeOp::RotateAndScale` lowers | Structural: `rotate_and_scale` is a typed op, so its angle, centre and scale reach the engine as parameters and Python holds no trig to drift (a polars `.radians()` copy rounded differently: `test_per_row_arguments_match_literals`) |
| A `Pipeline`'s state, when copied | `Pipeline._clone` — copies every field (lists copied, the immutable plan shared); `to_graph` and CSE derive through it, then replace the plan | `test_a_derived_pipeline_keeps_every_setting_and_shares_no_list` |
| Whether the compiled extension matches the sources | `POLARS_CV_SOURCE_HASH` from `build.rs`, recomputed by `build_info()` | `test_compiled_plugin_matches_the_rust_sources` — the version comparison cannot fire within a release cycle |
| Dtype spellings on the Python side | `python/polars_cv/_dtype_names.py`, generated from `dtype_table!` by `scripts/gen_dtype_names.py` | `test_dtype_names_module_is_current` (regenerate-and-diff; the generator also rejects a cast-only name that is a dtype spelling), `test_engine_dtype_names_match_the_generated_table` pins `_types.DType` to it without the plugin |
| Which plan-time optimizations exist | Each where it runs: the graph-scope passes (CSE, which compares the expression identities only Python holds) in `polars_cv/_graph_passes.py`; the node-scope passes in Rust `LogicalPass` (`polars-cv/src/passes.rs`) and the engine's in `engine_passes!` (view-buffer, which also declares `OptConfig`), via `tests/golden/pass_catalog.json`. `scripts/gen_ops.py` concatenates them → generated `PASS_CATALOG` / `OptFlags` fields → `OPTIMIZATION_PASSES` | `pass_catalog_matches_the_committed_file` and `test_the_committed_catalog_is_the_built_one`; `test_the_graph_tier_is_declared_by_python_alone` (Rust never also names a graph pass); `_graph.py` refuses to import with a declared graph pass it does not implement (`test_a_graph_pass_without_an_implementation_refuses_the_import`); `OptConfig` refuses an unknown engine key (`an_unknown_engine_toggle_is_refused`), and `Plan.run_pass` an unknown pass. Optimization is one explicit phase (`PipelineGraph.optimize`); construction and serialization never optimize (`TestStaging`), and toggling a pass changes only the physical graph, never the output (`test_optimize_equivalence.py`) |
| A polars-cv extension type's name and storage | Rust `ext_types::ExtType` (storage read from `geometry::schema` / `output::numpy_output_dtype`), mirrored by `polars_cv.extension_types.EXTENSION_TYPES` so `import polars_cv` can register without the `.so` | `test_python_types_match_the_rust_declaration` (names, order and full storage dtype over the `extension_types` FFI, both directions); `all_lists_every_variant_once` holds `ExtType::ALL` to the enum; `ext_from_params` returns polars' generic `Extension` for our name over any other storage, so an instance of our class *is* the canonical layout |
| How Python reaches the compiled plugin | `polars_cv._plugin.call` — pins polars to the file the import system loads and passes every argument as `.ext.storage()`, so Rust never receives an extension dtype and only builds tagged outputs (`ExtType::tag`) | `test_only_the_plugin_module_registers_plugin_functions` (AST scan of the package, fixtures in `test_plugin_entry_point.py`); `test_accessors_accept_tagged_inputs` sweeps every accessor case table with tagged inputs; `test_plugin_path_is_the_file_python_imports` |
| The compiled module's Python-visible signatures (`polars_cv._lib`) | The `#[pyfunction]`/`#[pyclass]` definitions registered in `polars-cv/src/lib.rs`; `python/polars_cv/_lib.pyi` is a hand-written restatement for `ty`, not an authority | `test_lib_stub_parity.py` (via `tests/_stub_parity.py`, fixtures in `test_stub_parity_fixtures.py`): every stub entry must exist in the built `_lib` with the same signature and every export must be stubbed, bar the reasoned `UNSTUBBED` table; self-skips without the `.so` |
| Whether a channel count carries alpha (2 = gray + alpha, 4 = RGBA) | `view_buffer::ops::color::has_alpha` / `color_channels`, read by the resamplers' premultiply, the shape rule's carried alpha, grayscale and the colour conversions | Source scan: `test_only_the_declaration_tests_for_alpha` (fixtures in `test_alpha_authority.py`) reports a channel-count comparison restating the rule |
| The package's import layering (`__init__` re-exports; nothing imports the root) | Named modules hold implementations; `__init__.py` re-exports them | `test_no_import_cycle_and_no_root_import` (scanner `tests/_import_layering.py`, fixtures in `test_import_layering_fixtures.py`); `test_each_module_imports_alone` imports each module in a fresh process |
| Which engine a query collects on | polars' default (`engine="auto"`) or the user's affinity, forwarded where a caller chose one (`DetectionTable.collect(engine=...)`); the package never passes a literal engine | `test_the_package_never_chooses_an_engine` (`test_engine_choice.py`, with fixtures); in the suite, `test_no_test_pins_the_in_memory_engine` (`test_engine_pin_guard.py`): only the lane's engine, or both side by side |
| Per-row work inside a query | The Rust plugin; the package runs no Python UDF (`map_batches`, `map_elements`, `map_groups`) | `test_the_package_has_no_python_udfs` (AST scan, fixtures in `test_no_python_udfs.py`) |
| How a keyed join in the metrics treats a null key | As a key (`nulls_equal=True`) | `test_every_metrics_join_matches_null_keys` (`tests/_join_ratchet.py`, fixtures in `test_join_ratchet.py`) |
| How tests read planner state (a pipeline's source, ops, planned states, expressions) | `tests/_plan_view.py`, the one seam over `Plan` and `_exprs` | `test_tests_read_planner_state_only_through_the_seam` (textual scan, fixtures in `test_plan_view_seam.py`); `test_every_exemption_names_a_real_file` |
| How an error names its row | `polars-cv/src/row_ref.rs`, attached by the code that owns each row loop | `test_rows_are_named_only_through_row_ref` (`tests/_row_ref_scan.py`, fixtures in `test_row_ref_guard.py`) rejects a hand-rolled `(row {})` |
| The uv version CI installs | `.tool-versions` at the repo root, read by every `setup-uv` step (and no `python` entry, which would override each job's matrix) | `test_every_setup_uv_step_installs_the_pinned_uv`, `test_tool_versions_pins_an_exact_uv` (fixtures in `test_ci_tool_pins.py`) |
| Which license files the published sdist carries | `scripts/check_sdist.py`, run by `publish.yml` on the sdist before any upload | `test_publish_checks_the_sdist_before_uploading` and the archive fixtures in `test_sdist_metadata.py`; `test_sdist_build.py` builds one and runs the check |
| That every example script still runs | `polars-cv/examples/`, run by `test_examples.py` (slow lane); a helper imported rather than run is exempted with its reason | `test_every_example_is_either_run_or_exempted`, `test_non_script_exemptions_are_real_and_needed`, `test_examples_run` |
| A detection metric's estimator (one value per group, per bootstrap replicate) | `metrics/_statistics.py`: a `Statistic` (`by_group(table, keys)`); the function form (`average_precision`, `froc_auc`, …) is its ungrouped reading and `bootstrap_ci` its per-replicate one | `tests/test_statistics.py::test_every_public_metric_is_classified` (a public function taking a `DetectionTable` must be a `Statistic`'s reading or declared otherwise, and agrees with it); grouped-equals-filtered for every statistic |
| A per-group ordered scan (running sum/max, lag, row index, first/last) or a reproducible sum in the metrics | `metrics/_grouped_scan.py`: `grouped_scan` (one sort, then `.over(keys)` windows on it), `bracket` (the nearest knot at or below / above, per group: the as-of join) and `exact_sums` / `exact_mean` (Int128 fixed point) | `tests/test_streaming_plans.py`: every public metrics plan, and every plan an eager metric collects, is read from its physical streaming graph, and any node polars marks as an in-memory engine fallback not in `KNOWN_FALLBACKS` fails; a window is accepted only straight after the sort on its keys, so a `.over()` scan elsewhere or a sort inside `agg` fails |
| AP integration (all-points step rule, 11/101-point grids) | `metrics/_metrics/_precision_recall.py::ap_from_points`, which the scalar curve AP, the grouped AP and every bootstrap reach | `TestAllPointsAPStepRule`/`TestNPointAP` against independent references (pycocotools' accumulate); `tests/reference/test_coco_parity_ref.py` end to end |
| Matching output: per-detection rows, class/threshold carried, metadata, sweeps | `metrics/_matching/_table.py` (`matched_table`, `explode_matches`), the one tail of every matcher | `tests/test_iou_sweep.py` (multi-class rows keep their class; a sweep's slices equal single matchings) |
| Which groupings of a `DetectionTable` are meaningful (a sweep's thresholds kept apart) | `DetectionTable.frames(keys)`, the one read every metric makes (`.detections`/`.image_metadata` are `frames()`) | Runtime: a sweep read without `iou_threshold` among the keys raises; `TestSweepTable::test_pooling_the_thresholds_is_refused` |

The former exception — the op spec riding its params on `#[serde(flatten)]`, which
cannot refuse an unknown key — is gone: every op is a variant of a `#[derive(Ops)]` family, whose wire
refuses an unknown field, and the untyped legacy spec was deleted in 0.29.0.

An enum that belongs to the plugin rather than the engine declares itself with
the same exported `named_variants!` and lands in `PLUGIN_REGISTRY`, which the
enum catalogue chains onto the engine's `REGISTRY`. Registering is what
generates its Python class; there is no FFI arm or Python mirror to add.

## The Single-Authority Refactor: What Was Done, What Is Left

This is written down because it was not. The phased plan behind the
single-authority work existed only in the agent session that produced it, so
recovering it meant reading commit messages and the sections of
`tests/test_sanitation.py`, each named by the invariant it checks (Plan == Exec,
registry parity, enum parity).
A plan that lives in a session is a plan that ends with it — keep this section
current instead.

**Done.**

1. *The mandatory op-append contract.* One way to append an op, applying the
   whole plan-time effect (now the Rust `Plan.push`). Fixed the `transpose` /
   `channel_select` / `channel_merge` shape desyncs and the lazy continuation's
   dead shape replay. (A1/A2/A3 shape half, A10.)
2. *Deleting what nothing reached.* view-buffer's pipeline-composition layer
   (`ops/io.rs`) and cost-reporting subsystem (`ops/cost.rs`),
   `rasterize(anti_alias=)`, node-level `shape_hints`, the geometry validation
   module.
3. *One declaration per fact.* `dtype_table!`, the `naming::REGISTRY`,
   the op registry (now the typed catalogue), input domains read from the
   Rust contract.
   Two planned items were examined and dropped as not real (a table-driven
   `resolve_op`, a `node_outputs` newtype) — recorded here so they are not
   re-proposed.
4. *Guards that fail closed.* The dtype ratchet under its own fixtures (the
   `resolve_op` arm scan went with `resolve_op` itself), `scripts/verify.sh` as
   one verification entry point.
5. *The sink contract.* `encode_node_output` keyed on the planned domain rather
   than the runtime `NodeOutput` variant, so the dtype the planner publishes and
   the value execution produces come from one key. (A1/A2/A3 sink half.)
6. *Path sandboxing (default-deny, #171).* `fetch::PathPolicy` and the
   `allowed_roots` option on `source("file_path", ...)` (and an `auto` source
   over a path column), `.cv.read_bytes(...)` and the header-metadata methods.
   It began as an opt-in whose absence meant "allow everything"
   (`PathPolicy::default()`, substituted at three readers); that open default is
   gone, a deliberate breaking change. A path column is data, so a reader that
   is not told where it may read reads nothing: an absent `allowed_roots` is an
   error naming the option and both ways out (a list of roots, or
   `allowed_roots=["*"]`, the explicit unrestricted opt-in), whatever
   `on_error` says, and it is raised before any row is read. `[]` and `"*"`
   beside real roots are refused too. Every reader turns its `allowed_roots`
   into a policy through `PathPolicy::resolve`, the only constructor (no
   `Default`), and both fetch functions take the policy as a *required*
   argument, so a new reader cannot reach a path by omitting it, nor
   reintroduce the open default without the compiler objecting. Remote URLs
   keep their semantics: a remote root is a URI prefix, a remote path must
   start with one and contain no `..`, local and remote roots never admit each
   other's paths, and `["*"]` admits both. An absent `allowed_roots` denies
   remote URLs as it does local paths, decided in #180: ambient credentials
   are rarely scoped to one prefix, and `http(s)://` has no credential boundary
   at all, so "the credentials bound it" is not a sandbox. The error names
   remote URLs and shows a remote root, so a cloud pipeline sees its fix.

7. *The typed op protocol (0.29.0, CR-45…CR-49).* One typed Rust definition per op, source and sink; a
   generated Python builder, enums and pass flags; a Rust planner
   (`Plan`, `check_graph`) with
   symbolic shapes (`OpShape`). It replaces the name + untyped-param-map
   protocol. It is not the "table-driven `resolve_op`" dropped above: that
   kept the untyped map and moved the arms into a table; this removes the
   untyped map, so the registries, parity tests and read-tracker that guard it
   are deleted rather than re-tabulated. The 0.29.0 CHANGELOG entry records
   what changed for users.

**Where deferred work is tracked.** Verified *defects* — a behaviour the code
should have but does not — are pinned executably in
`polars-cv/tests/test_known_gaps.py`, one `xfail(strict=True)` each, so a fix
turns the suite red rather than passing unnoticed; prefer adding an entry there
to extending a prose list. Everything else open (cleanups, coverage holes,
performance gaps, pending decisions and upstream drafts, most of which are not
xfail-able wrong behaviour) lives in the root `ISSUES.md` ledger with a stable
`CR-NN` id per item; a pinned defect is listed there too.

(An earlier version of this section claimed the two items below were each pinned
in `test_known_gaps.py`. They were not — one was a missing feature, the other a
perf limitation, and neither is a defect the ledger is for. The claim is removed
rather than back-filled with pins that would misuse the file.)

**Known non-defect gaps.**

- **`shear` / `rotate_and_scale` take `output_size` (and `center`) as required
  arguments.** This is by design, not a missing feature: the output shape is
  part of the plan-time schema, and an image source's height/width are unknown
  until execution, so there is no plan-time value to auto-compute from. The
  signatures enforce it (a required keyword-only argument) instead of raising
  late. Auto-sizing could be offered *only* for sources whose shape is already
  known at plan time (e.g. a `list` source with explicit dims, or post-`resize`);
  making it a silent conditional default would violate "explicit over implicit",
  so it is intentionally not done.
- **f64 and the 32/64-bit integers through the float-promoting scalar ops are excluded from kernel fusion**,
  which computes in f32. Correct, but slower than it needs to be. This is a perf
  limitation, tracked in the root **Known Issues** section, not a defect.
