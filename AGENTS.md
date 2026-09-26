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

7. **Priorities, in order: correctness, ergonomics, maintainability, performance.** Performance is last on purpose — the plugin runs over frames the engine already carries, so a clear, obviously-correct formula beats a fast one. Reject a design on performance grounds only when it is clearly suboptimal (an unnecessary per-row allocation, an O(n) pass where O(1) exists), never to shave constants at the cost of the three above it.

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
  → _plugin.call("vb_graph", graph_json, input columns)   # the one route to register_plugin_function
  → Polars calls Rust vb_graph(inputs, kwargs)
  → UnifiedGraph::from_json() → topological execution
  → Per-row: decode source → apply ops (ViewExpr/ViewBuffer) → encode sink
  → Returns Series (Binary, Float64, Struct, List, Array)
```

### Domain System

Pipelines track data domain through operations:

| Domain | Description | Example |
|--------|-------------|---------|
| `buffer` | Multi-dimensional arrays (images) | After `source()` (defaults to `"auto"`; also `"image_bytes"`, `"file_path"`, …) |
| `contour` | Geometry (vectors of points) | After `extract_contours()` |
| `scalar` | Single numeric values | After `reduce_sum()` |
| `vector` | Fixed-length numeric arrays, incl. histogram buckets | After `perceptual_hash()`, `bounding_box()`, `histogram()` |

### Alpha Channel Handling

Alpha channels are **always preserved** during image decoding (RGBA → 4ch, GrayA → 2ch). How each operation treats channels (and therefore alpha) is its `OpShape` (`view-buffer/src/ops/shape_rule.rs`); the planner reads the channel count as the shape's axis 2 and the rank as its length:

- **`Preserve`** and the H/W-only shapes — all channels processed uniformly (resize, normalize, flip, etc.)
- **`ColorChannels { channels, .. }`** — alpha separated, op applied to color channels, alpha restored (cvt_color)
- **`SingleChannel`** — alpha discarded, one output channel (grayscale, canny)
- **`Dynamic`**, or an unknown input channel count — not knowable at plan time

Image sources have unknown channel count at planning time. Users can assert known channels with `.assert_shape(channels=4)`.

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
│   └── scripts/                    # Generators (gen_ops.py, ...) and utilities
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
  Rust (`OpShape` in `view-buffer/src/ops/shape_rule.rs`: rank is its length,
  channels its axis 2; `OutputDTypeRule` for dtype), and applied in Rust by `plan::step` (once per appended op,
  `polars-cv/src/plan.rs`). A pipeline's ops live in its Rust `Plan`, with the
  state at every op boundary; the node-scope passes run on it in Rust too
  (`Plan.run_pass`, `polars-cv/src/passes.rs`). The planner contains no per-op
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
| An op's accepted input domains | Rust `GraphStep::input_domains` (exhaustive — no catch-all arm), checked by `plan::step` | Structural: the domain vocabulary is `Domain::NAMED` (no wildcard variant), generated into Python like every registered enum; execution reads the same contract via `step_buffer_operand` rather than restating it per arm |
| An op's H/W effect | view-buffer `OpShape` (`Op::shape`, generic over the mode), read from the `Wire` op by `plan::step` and identity elimination | one generic definition per family; `a_lowered_op_keeps_its_shape` for an op that executes as another |
| Which ops exist | Rust `ops::TypedOp` (`typed_ops!` registry) → generated Python `TYPED_OPS` | `catalog_matches_the_committed_file` (the reviewed op set), `test_every_op_is_emitted_by_a_builder` (works with no `.so`); an unregistered name fails deserialization |
| A typed op's fields, types, defaults, docs and Python signature | Its variant of a `#[derive(Ops)]` family (an engine enum in view-buffer, or `GraphOp`), via `tests/golden/op_catalog.json` → `scripts/gen_ops.py` → `_ops_generated.py` | the derived wire rejects an unknown/missing/mistyped field and applies a declared default; `catalog_matches_the_committed_file` (Rust) and `test_the_committed_catalog_is_the_built_one` (built `.so` + generated module) |
| Every spelling of a dtype (short / VIEW wire code / numpy) | `dtype_table!` in `view-buffer/src/core/dtype.rs` | `dtype_single_authority.rs` + `test_no_second_dtype_spelling_table` (a partial dispatch is reported) |
| Enum variant names crossing the FFI | `named_variants!` + `naming::REGISTRY` (engine) chained with `naming::PLUGIN_REGISTRY` (plugin-owned enums: `RowErrorPolicy`, `NullParamPolicy`, `FetchErrorPolicy`) | `every_named_enum_is_registered` (a `NAMED` table not in the registry fails), `registered_enums_have_unique_names`, `plugin_enums_have_unique_names`, `plugin_enums_do_not_shadow_engine_enums`; the Python classes are generated from the registries (`enum_catalog.json` → `gen_ops.py`, bar the stated `NOT_GENERATED` exceptions), pinned by `enum_catalog_matches_the_committed_file` and `test_the_committed_catalog_is_the_built_one` |
| A graph policy's wire spelling (`on_error`, `on_null_param`) | Its `NAMED` table, read by `ops::param::literal_field` (no serde derive on the enum) | `graph_policies_parse_through_their_named_tables`; there is no second spelling to compare |
| An op's domain transitions (the "Domain:" docstring line) | Its Rust `input_domains`/`output_domain`, read by `catalog_json` over the op's structural choices → `domains` in `op_catalog.json` → the generated line; sugar composes its declared ops' (`@_sugar`, `polars_cv/_domains.py` renders both) | `the_domain_contract_is_read_off_the_op`, `test_sugar_appends_exactly_the_ops_it_declares`, `test_every_public_pipeline_method_has_one_kind` |
| A parameter's default | `#[param(default = ...)]` on the field: the wire applies it when the field is absent, and the catalogue gives it to the generated signature (an `Option` field declares none) | `an_absent_field_takes_its_declared_default`; the derive refuses an `Option` field with a default |
| Source/sink formats, and which parameters each reads | One variant per format of the `Source`/`Sink` families in `polars-cv/src/formats/` (`#[derive(Ops)]` → `tests/golden/io_catalog.json` → generated `SourceFormat`/`SinkFormat`); the builder checks what the caller passed with `Plan.with_source` (which also plans the source's state), and `.sink()` checks the whole graph with `check_graph` (the plugin's own compile, planning and `decode::output_schema`) | Deserialization: an unknown format or field is refused, and a field the format does not read names where it applies (`formats::from_wire`); `io_catalog_matches_the_committed_file` and `test_the_committed_catalog_is_the_built_one`; `test_param_applicability.py` sweeps parameter × format grids from the catalogue and checks the `quality` claim against the encoders |
| `LazyPipelineExpr`'s method surface | generated by `gen_ops.py`: forwarders from the built `Pipeline` into `_lazy_forwarders.py`, and the binary `lazy_only` ops into `_LazyOpsMixin` | `test_lazy_pipeline_method_parity`, `test_every_lazy_only_op_is_a_lazy_method_with_its_fields`, `test_the_committed_catalog_is_the_built_one` |
| Which builder parameters are positional | Derived, never declared: `gen_ops.positional` (an op's only required field is positional-or-keyword, the rest keyword-only) | `tests/golden/signatures.json` (`test_the_call_surface_matches_the_snapshot`); `test_documented_pipeline_calls_bind` binds every documented `Pipeline()` call; the derive refuses a `#[param(positional)]` key |
| An output's planned schema | Planned by Rust from the graph itself (`resolved_output_specs`: each node's source state, then `plan::step` per op); the wire output is only `OutputRequest { node, sink }` | `an_output_carries_only_its_node_and_sink` (a `planned`/`expected_*` field is refused), `output_facts_are_planned_from_the_ops` |
| A shape declaration (`assert_shape`) | The internal `assert_shape` typed op: planned by `plan::declare`, checked per row by `GraphStep::AssertShape` | `a_declaration_the_state_contradicts_is_refused`, the `AssertShape` case of `every_graph_step_variant_executes` |
| The graph wire format's node fields | `GraphNode` with `#[serde(deny_unknown_fields)]` | Deserialization error — a stale or misspelled key fails the query |
| Null parameter handling | `NullParamPolicy` on `ParamCtx`, via `ParamCol::on_null` | Reviewed by hand: never add per-op or per-parameter null keywords |
| What a `(domain, sink format)` pair produces | `SinkKind::resolve` in `src/graph/sink_kind.rs` | Compile error: the four halves of the sink contract (`dtype_for_output`, `encode_node_output`, `null_row_result_for_spec`, `build_series_from_spec`) match on the enum, so a new kind is non-exhaustive in all four at once; `every_kind_is_produced_by_some_pair` rejects a kind no pair names |
| Which files a source-scanning guard reads | `tests/_discovery.py` — every accessor raises rather than returning empty | `test_scans_go_through_discovery` (AST walk: a direct `glob`/`rglob` in `tests/` fails unless the file is in `_DISCOVERY_EXEMPT` with a reason), `test_discovery_fixtures.py` |
| The `rotate_and_scale` matrix | `AffineParams::rotation_matrix_2d` (view-buffer), read via the `rotation_matrix_2d` FFI | `test_the_rotate_and_scale_builder_reads_the_matrix_ffi` — `_rotation_matrix`'s literal path must call the FFI, not recompute the trig (its `pl.Expr` branch is the one sanctioned copy) |
| A `Pipeline`'s state, when copied | `Pipeline._clone` — copies every field (lists copied, the immutable plan shared); `to_graph` and CSE derive through it, then replace the plan | `test_a_derived_pipeline_keeps_every_setting_and_shares_no_list` |
| Whether the compiled extension matches the sources | `POLARS_CV_SOURCE_HASH` from `build.rs`, recomputed by `build_info()` | `test_compiled_plugin_matches_the_rust_sources` — the version comparison cannot fire within a release cycle |
| Dtype spellings on the Python side | `python/polars_cv/_dtype_names.py`, generated from `dtype_table!` by `scripts/gen_dtype_names.py` | `test_dtype_names_module_is_current` (regenerate-and-diff), `test_engine_dtype_names_match_the_generated_table` pins `_types.DType` to it without the plugin |
| Which plan-time optimizations exist | Rust: `LogicalPass` (`polars-cv/src/passes.rs`, the node-scope passes run there) and `engine_passes!` (view-buffer, which also declares `OptConfig`), via `tests/golden/pass_catalog.json` → generated `PASS_CATALOG` / `OptFlags` fields → `OPTIMIZATION_PASSES` | `pass_catalog_matches_the_committed_file` and `test_the_committed_catalog_is_the_built_one`; `OptConfig` refuses an unknown engine key (`an_unknown_engine_toggle_is_refused`), and `Plan.run_pass` an unknown or graph-scope pass. Optimization is one explicit phase (`PipelineGraph.optimize`); construction and serialization never optimize (`TestStaging`), and toggling a pass changes only the physical graph, never the output (`test_optimize_equivalence.py`) |
| A polars-cv extension type's name and storage | Rust `ext_types::ExtType` (storage read from `geom_schema` / `output::numpy_output_dtype`), mirrored by `polars_cv.extension_types.EXTENSION_TYPES` so `import polars_cv` can register without the `.so` | `test_python_types_match_the_rust_declaration` (names, order and full storage dtype over the `extension_types` FFI, both directions); `all_lists_every_variant_once` holds `ExtType::ALL` to the enum; `ext_from_params` returns polars' generic `Extension` for our name over any other storage, so an instance of our class *is* the canonical layout |
| How Python reaches the compiled plugin | `polars_cv._plugin.call` — pins polars to the file the import system loads and passes every argument as `.ext.storage()`, so Rust never receives an extension dtype and only builds tagged outputs (`ExtType::tag`) | `test_only_the_plugin_module_registers_plugin_functions` (AST scan of the package, fixtures in `test_plugin_entry_point.py`); `test_accessors_accept_tagged_inputs` sweeps every accessor case table with tagged inputs; `test_no_module_carries_its_own_plugin_path` |

The former exception — the op spec riding its params on `#[serde(flatten)]`, which
cannot refuse an unknown key — is gone: every op is a variant of a `#[derive(Ops)]` family, whose wire
refuses an unknown field, and the untyped legacy spec was deleted in typed-op
P6.

An enum that belongs to the plugin rather than the engine declares itself with
the same exported `named_variants!` and lands in `PLUGIN_REGISTRY`, which the
enum catalogue chains onto the engine's `REGISTRY`. Registering is what
generates its Python class; there is no FFI arm or Python mirror to add.

## The Single-Authority Refactor: What Was Done, What Is Left

This is written down because it was not. The phased plan behind the
single-authority work existed only in the agent session that produced it, so
recovering it meant reading commit messages and the assessment markers still
embedded in `tests/test_sanitation.py` (findings A1/A2/A3, A4, A10, B1/B2).
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
4. *Guards that fail closed.* The dtype ratchet under its own fixtures, the
   `resolve_op` arm scan, `scripts/verify.sh` as one verification entry point.
5. *The sink contract.* `encode_node_output` keyed on the planned domain rather
   than the runtime `NodeOutput` variant, so the dtype the planner publishes and
   the value execution produces come from one key. (A1/A2/A3 sink half.)
6. *Path sandboxing.* `fetch::PathPolicy` and the `allowed_roots` option on
   `source("file_path", ...)` and `.cv.read_bytes(...)`. Opt-in, so the default
   is unchanged; once asked for, it denies by default. Both fetch functions
   take the policy as a *required* argument, so a new caller cannot reach a
   path by omitting it.

7. *The typed op protocol ([`TYPED_OPS_PLAN.md`](TYPED_OPS_PLAN.md),
   CR-45…CR-49).* One typed Rust definition per op, source and sink; a
   generated Python builder, enums and pass flags; a Rust planner
   (`Plan`, `check_graph`) with
   symbolic shapes (`OpShape`). It replaces the name + untyped-param-map
   protocol. It is not the "table-driven `resolve_op`" dropped above: that
   kept the untyped map and moved the arms into a table; this removes the
   untyped map, so the registries, parity tests and read-tracker that guard it
   are deleted rather than re-tabulated. The plan file holds the phase record,
   the deviations and the deletion matrix.

**Where deferred work is tracked.** Verified *defects* — a behaviour the code
should have but does not — are pinned executably in
`polars-cv/tests/test_known_gaps.py`, one `xfail(strict=True)` each, so a fix
turns the suite red rather than passing unnoticed; prefer adding an entry there
to extending a prose list. No gap is open (the planned-size defects closed in
`PLANNER_SIZES_PLAN.md` S1 and S2). The
broader structural-review backlog — dead code, duplicate declarations, coverage
holes — lives in the root `CODE_REVIEW_FINDINGS.md` ledger with a stable id per
item, since most of those are cleanups rather than xfail-able wrong-behaviour
defects.

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
- **f64 through the float-promoting scalar ops is excluded from kernel fusion**,
  which computes in f32. Correct, but slower than it needs to be. This is a perf
  limitation, tracked in the root **Known Issues** section, not a defect.
