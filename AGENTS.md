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
| `buffer` | Multi-dimensional arrays (images) | After `source()` (defaults to `"auto"`; also `"image_bytes"`, `"file_path"`, …) |
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
- **One mandatory append path.** `Pipeline._push_op()` is the only code allowed
  to *append* to `_ops` (`_set_ops_slice` replaces the list wholesale for CSE
  and re-keys the position-keyed side tables; `_clone` copies everything), and
  it applies an operation's *entire* plan-time effect:
  the input-domain check, the `op_schema` fold (domain/dtype/ndim) and the
  shape hints. Builders call it through `_append_op`; the lazy continuation
  replays through it too, which is what makes `.pipe(p.op())` and
  `.pipe(p).op()` agree by construction. Guarded structurally by
  `tests/test_append_contract.py` — an AST check that nothing else touches
  `_ops`, plus an eager/lazy parity sweep whose op table is
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
| Appending an op to a `Pipeline` (domain check + `op_schema` fold + shape hints) | `Pipeline._push_op()` | `test_op_append_is_structurally_exclusive` — AST walk failing if anything but `_push_op`/`_set_ops_slice`/`_clone` touches `_ops` |
| An op's rank / channel / dtype / memory / spatial / identity contract | `Op` trait methods, **no defaults** | Compile error: a new op that omits one does not build |
| An op's accepted input domains | `op_contract(...)["input_domains"]` (Rust `GraphStep::input_domains`, exhaustive — no catch-all arm) | `test_domain_vocabulary_declared_once` — `Pipeline` may not carry `DOMAIN_*` constants or a `_validate_domain`; execution reads the same contract via `step_buffer_operand` rather than restating it per arm |
| An op's H/W effect | view-buffer `infer_shape`, read via `op_infer_shape` | No inferable shape ⇒ hints invalidated, never carried forward |
| Which ops exist | Rust `KNOWN_OPS` ↔ Python `OP_NAMES` | `known_ops_all_resolve`, `resolve_op_arms_are_all_known_ops`, `test_op_names_matches_rust_known_ops_without_the_plugin` (works with no `.so`); guard arms in `resolve_op` must be listed in `KNOWN_GUARD_ARMS` |
| Every spelling of a dtype (short / VIEW wire code / numpy) | `dtype_table!` in `view-buffer/src/core/dtype.rs` | `dtype_single_authority.rs` + `test_no_second_dtype_spelling_table` (a partial dispatch is reported) |
| Enum variant names crossing the FFI | `named_variants!` + `naming::REGISTRY` (engine) chained with `naming::PLUGIN_REGISTRY` (plugin-owned enums: `RowErrorPolicy`, `NullParamPolicy`, `FetchErrorPolicy`) | `every_named_enum_is_registered` (a `NAMED` table not in the registry fails), `registered_enums_have_unique_names`, `plugin_enums_have_unique_names`, `plugin_enums_do_not_shadow_engine_enums`, `test_every_rust_enum_is_parity_checked` (iterates `enum_names()`, both directions) |
| A policy enum's *wire* spelling vs its published one | serde `rename_all` reads the wire, `NAMED` publishes it | `row_error_policy_names_match_serde`, `null_param_policy_names_match_serde` — nothing else compares the two, and a rename on one side alone lets Python send a value the graph cannot parse |
| Source format vocabulary | Python `SourceFormat` ↔ Rust `KNOWN_SOURCE_FORMATS` | `test_source_formats_match_the_rust_vocabulary` (runs without the plugin); the graph validator rejects an unlisted format |
| Which formats a `source()` / `.sink()` parameter applies to | `SOURCE_PARAM_APPLIES` / `SINK_PARAM_APPLIES` in `_types.py`, read by `reject_inapplicable_params` | `test_param_applicability.py`: the source table's keys must equal `source()`'s keywords and the sink table's must equal `SinkSpec`'s wire fields; the check must read `locals()`; swept parameter × format grids; and the `quality` claim is checked against the encoders. Rust `SinkSpec` is `deny_unknown_fields` |
| `LazyPipelineExpr`'s method surface | generated from `Pipeline` at import | `test_lazy_pipeline_method_parity`, `test_lazy_stub_is_current` |
| The graph wire format's node fields | `GraphNode` with `#[serde(deny_unknown_fields)]` | Deserialization error — a stale or misspelled key fails the query |
| Null parameter handling | `NullParamPolicy` on `ParamCtx`, via `ParamCol::on_null` | Reviewed by hand: never add per-op or per-parameter null keywords |
| What a `(domain, sink format)` pair produces | `SinkKind::resolve` in `src/graph/sink_kind.rs` | Compile error: the four halves of the sink contract (`dtype_for_output`, `encode_node_output`, `null_row_result_for_spec`, `build_series_from_spec`) match on the enum, so a new kind is non-exhaustive in all four at once; `every_kind_is_produced_by_some_pair` rejects a kind no pair names |
| Which files a source-scanning guard reads | `tests/_discovery.py` — every accessor raises rather than returning empty | `test_scans_go_through_discovery` (AST walk: a direct `glob`/`rglob` in `tests/` fails unless the file is in `_DISCOVERY_EXEMPT` with a reason), `test_discovery_fixtures.py` |
| The `rotate_and_scale` matrix | `AffineParams::rotation_matrix_2d` (view-buffer), read via the `rotation_matrix_2d` FFI | `test_the_rotate_and_scale_builder_reads_the_matrix_ffi` — `_rotation_matrix`'s literal path must call the FFI, not recompute the trig (its `pl.Expr` branch is the one sanctioned copy) |
| A `Pipeline`'s state, when copied | `_STATE_COPIERS` + `Pipeline._copy_state_from` — `_clone`, `_create_sub_pipeline` and CSE all inherit everything, then override | `test_pipeline_state_copy_is_complete` (table ↔ `__init__`, both directions) and `test_every_pipeline_field_survives_a_copy` |
| Whether the compiled extension matches the sources | `POLARS_CV_SOURCE_HASH` from `build.rs`, recomputed by `build_info()` | `test_compiled_plugin_matches_the_rust_sources` — the version comparison cannot fire within a release cycle |
| Dtype spellings on the Python side | `python/polars_cv/_dtype_names.py`, generated from `dtype_table!` by `scripts/gen_dtype_names.py` | `test_dtype_names_module_is_current` (regenerate-and-diff), `test_engine_dtype_names_match_the_generated_table` pins `_types.DType` to it without the plugin |
| Which plan-time optimizations exist | `OPTIMIZATION_PASSES` in `_optimize.py` (one `PassSpec` per pass, both tiers) ↔ the `OptFlags` fields | `test_optimize.py::test_flags_match_registry_both_directions` — a pass without a flag or a flag without a pass fails. Optimization is one explicit phase (`PipelineGraph.optimize`); construction and serialization never optimize (`TestStaging`), and toggling a pass changes only the physical graph, never the output (`test_optimize_equivalence.py`) |

One deliberate exception, documented at the site: `OpSpec` is *not*
`deny_unknown_fields`, because its params ride on `#[serde(flatten)]`, which
serde documents as incompatible. It is not a precedent.

`BinaryOp` used to be a second exception — its name table sat in the plugin
crate, so it needed a hand-written arm in `enum_variants` and a by-name
exemption from the parity test. The table moved next to the enum in
view-buffer, and the exception went with it. An enum that genuinely belongs to
the plugin now declares itself with the same exported `named_variants!` and
lands in `PLUGIN_REGISTRY`, which the FFI chains onto the engine's. **Do not
add an arm to `enum_variants`**: registering is what surfaces an enum to Python
*and* what makes the parity test demand a mirror for it, and an arm gets you
the first without the second.

## The Single-Authority Refactor: What Was Done, What Is Left

This is written down because it was not. The phased plan behind the
single-authority work existed only in the agent session that produced it, so
recovering it meant reading commit messages and the assessment markers still
embedded in `tests/test_sanitation.py` (findings A1/A2/A3, A4, A10, B1/B2).
A plan that lives in a session is a plan that ends with it — keep this section
current instead.

**Done.**

1. *The mandatory op-append contract.* `Pipeline._push_op` as the only way to
   append an op, applying the whole plan-time effect. Fixed the `transpose` /
   `channel_select` / `channel_merge` shape desyncs and the lazy continuation's
   dead shape replay. (A1/A2/A3 shape half, A10.)
2. *Deleting what nothing reached.* view-buffer's pipeline-composition layer
   (`ops/io.rs`) and cost-reporting subsystem (`ops/cost.rs`),
   `rasterize(anti_alias=)`, node-level `shape_hints`, the geometry validation
   module. Guarded by `tests/test_removed_surfaces.py`.
3. *One declaration per fact.* `dtype_table!`, the `naming::REGISTRY`,
   `KNOWN_OPS` ↔ `OP_NAMES` parity, input domains read from the Rust contract.
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

**Where deferred work is tracked.** Verified *defects* — a behaviour the code
should have but does not — are pinned executably in
`polars-cv/tests/test_known_gaps.py`, one `xfail(strict=True)` each, so a fix
turns the suite red rather than passing unnoticed; prefer adding an entry there
to extending a prose list. That file currently holds exactly one such defect
(the two `scale`-contour surfaces defaulting their origin differently). The
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
