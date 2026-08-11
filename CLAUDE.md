# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

**polars-cv** is a Polars plugin for high-performance vision and array operations. It lets users build image-processing pipelines that run as Polars expressions over DataFrame columns — zero-copy by default, lazy evaluation, explicit over implicit.

```python
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
df.with_columns(processed=pl.col("image").cv.pipe(pipe).sink("numpy"))
```

The project is a Rust/Python hybrid built with [Maturin](https://github.com/PyO3/maturin) (PyO3). The Python package lives under `polars-cv/`, and `view-buffer/` is a separate Rust crate that is the core tensor engine.

---

## Working Agreements

How to tackle problems here. These apply to every task in this repository; the
sections further down describe *what* the code is, this one describes *how* to
change it.

### Canonical paths are mandatory, and bypasses are rejected

This codebase's defining rule: **for anything with a shared mechanism, the
shared mechanism is the only way in.** A second implementation is not a
shortcut, it is a divergence waiting to be discovered by a user.

The enforcement standard is stricter than "prefer the shared path":

- **A bypass must fail, not degrade.** Code that sidesteps a canonical
  mechanism must be *actively rejected* — a compile error, a raised exception,
  or a failing guard test — never silently accepted with reduced behaviour. An
  op that declines to declare its dtype rule must not simply be treated as
  `PreserveInput`; it must not compile. A dtype string nothing recognises must
  not fall back to `u8`; it must error. Read the recent `CHANGELOG.md` entries
  for what silent acceptance actually costs — a fallback that turned an
  unmappable column into a claimed buffer of bytes, an unread wire field that
  went on being emitted for releases, a parameter plumbed six layers deep and
  discarded at the bottom.
- **Prefer a mechanism callers cannot step around to a test that lists what
  they must remember.** A ratchet enumerating "you must also call X" fails the
  day someone adds Y. Make the sequence unskippable instead: one entry point
  that does the whole thing.
- **No defaulted contract methods on op traits.** `Op::output_rank_rule`,
  `output_channel_rule`, `output_dtype_rule` and `memory_effect` are required
  with no default so a new op cannot inherit a lie. Adding a default to any of
  them is a regression, however convenient.
- **One authority per fact, named once.** A dtype's spellings live in
  `dtype_table!`; enum variant names live in `named_variants!` + the
  `naming::REGISTRY`; op names live in `KNOWN_OPS`; an op's input domain lives
  in its Rust contract. If you find yourself writing a `match` that
  re-enumerates one of those, you are creating the second copy — read from the
  authority instead.
- **Registering is the same act as being checked.** Adding an enum to the
  registry is what surfaces it over FFI *and* what gets it parity-checked;
  adding an op to `KNOWN_OPS` is what makes it resolvable *and* what pins it to
  a `resolve_op` arm. Never add a hand-written arm alongside the registry.

See [Canonical Paths](#canonical-paths) below for the concrete list of
mechanisms and the guard that enforces each one.

### Deleting is part of the work

- A parameter that is accepted and ignored, a field nothing reads, a subsystem
  no caller reaches — delete it, do not document it as "not yet implemented".
  Dead paths are not free: they enter op identity (breaking CSE and the
  compiled-graph cache), they enter every `match`, and they read as coverage.
- Deletions get a guard too. `tests/test_removed_surfaces.py` pins each removed
  surface with the reason, so the next author does not "restore" it. Rust-side
  removals are guarded by the compiler.
- When a fallback arm exists only to hide the case it cannot handle, remove the
  arm and raise.

### Guards must be watched failing

- A new guard is not done until you have watched it fail **for the reason it
  claims**. A checker that silently matches nothing reads as green forever;
  this repo has shipped that failure mode repeatedly.
- A guard with non-trivial logic gets committed fixtures (see
  `tests/_dtype_ratchet.py` and `tests/test_dtype_ratchet_fixtures.py`): both
  known-bad snippets it must reject and known-good ones it must not.
- Prefer compiler exhaustiveness > runtime assertion > source scanning, in that
  order. Reach for a source scan only when the first two cannot express the
  property, and state its limits in the docstring.
- Verify at the user-facing entry point, not the helper. Confirming a planner
  behaviour and inferring the caller is how a working input was broken while
  claiming to fix a silent lie.

### Verification

- Run `scripts/verify.sh` (add `--fast` to skip the slow lane). It runs every
  check CI runs, captures each exit code directly, and prints one PASS/FAIL
  computed from those codes.
- **Never read a filtered view of a check and call it green.** `grep | head`
  cuts the failing suite below the fold; `maturin ... | tail` reports tail's
  exit code, not maturin's. Both have produced false "all green" reports here.
- The install is editable: Python edits take effect immediately, the compiled
  `.so` does not. After touching Rust, re-run `maturin develop` or you are
  testing old Rust against new Python — plugin tests self-skip rather than
  fail, so the window is silent. `polars_cv.build_info()` reports the three
  versions that must agree. Build **debug** (`maturin develop`, no `--release`):
  it is what CI and `verify.sh` use, the whole suite passes against it, and
  `--release` costs several minutes re-optimising the polars stack for nothing
  outside the benchmarks.
- Never edit or weaken an existing test to make it pass without saying so
  explicitly and getting agreement. Updating a test because the behaviour it
  pins was *deliberately* removed is fine — and the removal gets its own guard.

### Dependencies and documentation

- Assume your internal knowledge of any dependency, library, framework or tool
  is outdated. This applies to all dependencies, not just external APIs.
- Fetch current documentation before writing code against one — use the
  `context7` MCP tool if available, otherwise web search/fetch. Do not write
  from memory and hope.

### Communication

- Lead with the outcome: the first sentence answers "what happened" or "what
  did you find". Supporting detail follows.
- Keep it brief and direct. Match written deliverables to substance — no filler
  sections, redundant summaries or boilerplate.
- Before the first tool call, say in one sentence what you are about to do.
  While working, give an update only when you find something important or
  change direction.
- Correct an earlier statement only when the error changes the user's code,
  conclusions or decisions. State the fix plainly and move on.
- If the request seems mistaken or a better approach exists, say so in one
  sentence prefixed with `💡 [SUGGESTION]` and continue with the task as asked.
  Deliver the scope requested — do not quietly narrow, widen or transform it.
- If required information is missing or no available tool fits, say so directly
  rather than guessing. Never use placeholders or invented parameters.

### Action defaults

- Implement changes directly rather than only proposing them. Infer intent and
  use tools to discover details rather than guessing.
- Issue independent tool calls in parallel.
- Editing files, running linters and running the test suite need no approval.
  Ask first for destructive or hard-to-reverse commands: `rm -rf`, dropping
  tables, `git push --force`, `git reset --hard`, rewriting published history,
  `--no-verify`, and anything touching shared infrastructure or external APIs
  with side effects.
- Delegate to subagents only for genuinely independent, parallelizable
  investigations. Do not delegate work you can finish in a handful of tool
  calls, and do not use a subagent to double-check your own work.
- Do not stop early over token budget — context is compacted automatically.
  Before a context refresh, save progress to a file or to Git history; after
  one, read that state back before acting.
- Clean up scratchpad scripts and temporary test files at the end of a session.

---

## Commands

All commands should be run from the `polars-cv/` subdirectory unless noted otherwise.

### Build

```bash
uv sync --group dev              # Install Python dev dependencies
maturin develop                  # Compile Rust plugin (debug) and install into .venv
maturin build --release          # Build distributable wheels
```

**Use the debug build for the develop/test loop.** `maturin develop` with no
`--release` is what `scripts/verify.sh` and both CI workflows run, and it is
several minutes faster per iteration — the release build re-optimises the whole
polars stack. Every test in `tests/` passes against the debug extension; the
only things that need `--release` are the benchmarks (see
`benchmarks/regression/README.md`), where an unoptimised build measures nothing
useful, and the wheels you distribute.

This project installs **editable**: `.venv` carries a `.pth` pointing at
`python/`, and `maturin develop` writes `_lib.abi3.so` into `python/polars_cv/`.
So Python edits take effect immediately, but **the compiled extension does not** —
after pulling commits that touch Rust, the `.so` stays at its build-time version
until you re-run `maturin develop`, and you are testing old Rust against new
Python. `polars_cv.build_info()` reports the three versions that must agree
(`__version__`, the compiled plugin, the installed distribution), and
`tests/test_version_consistency.py` fails when they do not.

### Test

```bash
uv run pytest tests/                            # Full suite; plugin tests self-skip if unbuilt
uv run pytest tests/test_pipeline_builder.py   # Single test file
uv run pytest tests/ -k "test_resize"          # Single test by name
python scripts/test_multiple_python.py --all   # Test across Python 3.10–3.13
```

Tests are marked with `network` (needs network access) and `slow` (excluded from
the default lane). CI runs `pytest -m "not network and not slow"` on every push
and a separate `-m "slow and not network"` lane on a schedule.

Rust unit tests (run from the workspace root or with `-p` flag):
```bash
cargo test -p view-buffer --all-features   # view-buffer engine tests
cargo test -p polars-cv                    # Rust plugin tests
```

### Lint & Format

```bash
uvx ruff check python tests benchmarks         # Python lint (matches CI)
uvx ruff format python tests benchmarks        # Python format
cargo fmt --all -- --check       # Rust format check
cargo clippy --all-targets --all-features -- -D warnings  # Rust lint
```

A [pre-commit](https://pre-commit.com/) config (`.pre-commit-config.yaml`) wires
these up; `pre-commit>=4.5.1` is in the dev group. Install hooks with
`uv run pre-commit install`.

### Docs

```bash
uv sync --group docs             # Install docs dependencies
uv run mkdocs serve              # Live-preview the MkDocs site locally
uv run mkdocs build --strict     # Build the site (fails on broken links/nav)
```

---

## Architecture

The project has three layers:

```
Python: polars_cv
  Pipeline builder, expression namespaces, DAG construction,
  schema inference, lazy composition, geometry/metrics APIs
        ↕ JSON graph serialization
Rust: polars-cv (the plugin)
  vb_graph expression entry point, graph execution, source
  decoding, sink encoding, per-row parameter resolution, cloud I/O
        ↕ Rust crate dependency
Rust: view-buffer (the engine)
  ViewBuffer, ViewExpr, stride-aware operations, kernel fusion,
  zero-copy interop with Arrow/ndarray
```

### Data Flow

1. User builds a `Pipeline` in Python → internally creates a `PipelineGraph` (DAG of `GraphNode`s).
2. `.sink(...)` on a `LazyPipelineExpr` serializes the graph to JSON and calls `register_plugin_function("vb_graph", ...)`.
3. Polars calls the Rust `vb_graph` expression function with the JSON and any per-row expression parameters.
4. Rust deserializes into a `UnifiedGraph` and compiles it once into a process-wide cache (`graph/compiled.rs`: parsed spec, topological order, slot-bound params); repeat calls (e.g. per streaming morsel) pay only a hash lookup. It then executes topologically per-row: decode source → apply operations → encode sink.
5. Returns a Polars `Series` (dtype depends on sink: Binary, Float64, Struct, List, Array).

### Key Python Modules (`polars-cv/python/polars_cv/`)

| File | Role |
|------|------|
| `pipeline.py` | `Pipeline` builder — all image/array operations as chainable methods |
| `lazy.py` | `LazyPipelineExpr` — lazy `.pipe()`, `.merge_pipe()`, `.sink()`, binary ops |
| `expressions.py` | `CvNamespace` — the `.cv` accessor registered on Polars expressions (`.pipe()`, `.read_bytes()`, header-only metadata) |
| `_types.py` | Core type definitions: `OpSpec`, `ParamValue`, `SourceSpec`, `Domain`, `DType`, and the source/sink parameter-applicability tables |
| `_graph.py` | `PipelineGraph` / `GraphNode` — DAG construction, JSON serialization, CSE, plugin registration |
| `_namespace.py` | Shared base for the `.cv`/`.point`/`.contour`/`.bbox` expression namespaces (plugin-registration boilerplate) |
| `display.py` | `show_images()` — notebook rendering of image columns |
| `_graph_viz.py` | Graph visualization (networkx/graphviz/pydot) |
| `geometry/` | Point/contour/bbox schemas and Polars expression namespaces |
| `metrics/` | Detection metrics (PR curves, AP, FROC, LROC, bootstrap, AUC) |

### Key Rust Modules

**polars-cv/src/**
- `lib.rs` — PyO3 module entry, `vb_graph` polars expression function, dtype inference, and the `op_schema`/`op_contract`/`op_output_dtype`/`enum_variants`/`known_ops` FFI the Python planner reads
- `execute.rs` — `resolve_op()` dispatcher mapping `OpSpec`s to `GraphStep`s (`graph/step.rs`: buffer ops wrap view-buffer's `ViewDto`; graph-only steps are their own variants); owns the `KNOWN_OPS` registry
- `graph/` — `UnifiedGraph` execution engine: `types.rs` (`UnifiedGraph`, `GraphNode`, `OutputSpec`, `RowErrorPolicy`), `compiled.rs` (process-wide compiled-graph cache), `step.rs` (`GraphStep` — the plugin-level step vocabulary), source decoding (`decode.rs`), sink encoding (`encode.rs`)
- `params.rs` — `ParamValue` resolving literals vs per-row Polars column values
- `pipeline.rs` — serde types for the JSON graph spec crossing the plugin boundary
- `cloud.rs` — remote/cloud transport (`object_store` backends, `cloud_options`, bounded-concurrency reads)
- `fetch.rs` — stage one of every path-based read: path column → bytes (`prefetch`, `row_bytes`, `parse_on_error`), shared by the `file_path` source and `read_bytes.rs`; owns `PathPolicy` (the `allowed_roots` sandbox)
- `read_bytes.rs` — `read_file_bytes` plugin function (`.cv.read_bytes()`) — `fetch.rs` with the decode omitted, for byte-identical passthrough
- `image_metadata.rs` — header-only metadata plugin functions (`.cv.width()`/`height()`/`channels()`/`image_dtype()`)
- `output.rs` — zero-copy numpy/torch struct output encoding
- `engine_warning.rs` — one-time single-threaded-batch warning (points users to `engine="streaming"`)
- `contour.rs`, `point.rs` — standalone plugin functions for geometry namespaces
- `geom_params.rs` — `GeomParams`: per-row parameter resolution for those standalone functions, reading expression params off the extra inputs the Python `_ArgBinder` appends and names in `input_slots`

**view-buffer/src/** (see `view-buffer/AGENTS.md` for the full module tree)
- `core/` — `ViewBuffer` (strided N-D array), `DType`, `Layout`
- `ops/` — operation definitions by category (`image.rs`, `color.rs`, `compute.rs`, `scalar.rs`, `filter.rs`, `affine.rs`, `view.rs`, `binary.rs`, `reduction.rs`, `histogram.rs`, `phash.rs`, `pad.rs`, `mask.rs`), plus `shape_rule.rs` (the plan-time rank/channel authority), `validation.rs`, `traits.rs`, `util.rs`
- `ops/dto.rs` — `ViewDto` enum: the serializable bridge between JSON and Rust op code
- `expr.rs` — `ViewExpr` lazy builder with `.plan()` / `.execute()`
- `execution/` — `ExecutionPlan`, runner, kernel fusion
- `geometry/` — contour extraction, rasterization, measures, pairwise matching, transforms
- `interop/` — zero-copy Arrow, ndarray, `image`, and Polars-arrow integration
- `protocol.rs` — VIEW binary protocol (header + data serialization)

---

## Key Conventions

### Domain System

Every `Pipeline` tracks a **domain** through operations:

| Domain | What it holds | Produced by |
|--------|---------------|-------------|
| `buffer` | Multi-dimensional array | `source()` (defaults to `"auto"`; also `"image_bytes"`, `"file_path"`, …) |
| `contour` | Geometry vectors | `extract_contours()` |
| `scalar` | Single numeric value | `reduce_sum()` |
| `vector` | 1-D numeric array (incl. histogram buckets) | `perceptual_hash()`, `histogram()` |

Domain constraints are enforced at pipeline-build time. Operations that don't match the current domain raise immediately in Python, not at execution time.

### Parameter Values

Most operation parameters accept either a literal (`224`) or a Polars expression (`pl.col("target_height")`). This is typed as `ParamValue` in `_types.py`. Per-row expression params are resolved in Rust via `params.rs` (`resolve_*` for numbers, `resolve_str`/`resolve_bool` for enums and flags), and by `geom_params.rs` for the `.contour`/`.point`/`.bbox` namespaces, which bypass `vb_graph` and carry their expression params as extra plugin inputs recorded in an `input_slots` name→index map.

The rule for whether a parameter may be per-row is *not* its type: **a parameter is eligible iff its value has no effect on the output shape, rank or dtype**, because the lazy schema is computed at plan time and must match what executes. So non-structural enums and flags (`filter`, `interpolation`, `pad(mode=)`, `convolve2d(border=, normalize=)`, …) are per-row, while structural parameters are literal-only: `cast(dtype=)`, `normalize(method=, out_dtype=)`, reduction `axis`, `perceptual_hash(hash_size=, algorithm=)`, `rotate(expand=)`, `histogram(closed=, output=)`, the `transpose`/`flip` axis lists and `reshape`'s element count. For a list-valued parameter the *length* is structural while the elements are not — a `convolve2d` kernel keeps a literal element count but each coefficient may be an expression. Plan-time shape probing binds every expression param to an integer placeholder, so the enum/flag accessors substitute their default under `ParamCtx::probe`; that substitution is sound only because of the eligibility rule.

A parameter column may contain **nulls**. `Pipeline.on_null_param("raise"|"null")` (and `on_null(...)` on the geometry accessors) chooses between failing the query and nulling just the affected rows. This is one shared mechanism, never per-op handling: a `NullParamPolicy` rides on `ParamCtx` and every null reaches `ParamCol::on_null`, which flags the context so `graph/compiled.rs` skips the node for that row — reusing the same null-propagation path a null input image already takes, so nulling is node-scoped rather than row-scoped. Do not add per-op or per-parameter null keywords: a fallback value is already `pl.col("h").fill_null(224)`, and a per-parameter policy would have to enter the `ParamValue` wire format and its `__eq__`/`__hash__` (or CSE would merge ops differing only in policy).

### `Pipeline` Is Immutable

Every operation on `Pipeline` returns a new clone. Do not mutate an existing pipeline in place.

### Alpha Channel Handling

Image sources always preserve alpha. How each operation treats channels (and
therefore alpha) is declared by its `OutputChannelRule` in
`view-buffer/src/ops/shape_rule.rs`, the single authority the Python planner
reads via `channel_rule`:
- `PreserveChannels` — channel count is unchanged (alpha passes through).
- `StripProcessRestore { color_channels }` — alpha is split off, the op runs on
  the color channels, then alpha is re-attached (e.g. `RGBA`→gray yields `GrayA`).
- `Fixed(n)` — output has exactly `n` channels regardless of input (e.g.
  `grayscale`/`canny` → 1), dropping any alpha.
- `NotApplicable` / `Unknown` — no `[H, W, C]` image result, or not knowable at
  plan time.

### Canonical Paths

Each row is a fact with exactly one authority, the mechanism that owns it, and
the guard that rejects a second declaration. **Read from the authority; never
restate it.** If you need something the authority cannot express, extend the
authority — do not open a side channel.

| Fact | Single authority | Rejection mechanism |
|------|------------------|---------------------|
| Appending an op to a `Pipeline` (domain check + `op_schema` fold + shape hints) | `Pipeline._push_op()` | `test_op_append_is_structurally_exclusive` — AST walk failing if anything but `_push_op`/`_set_ops_slice`/`_clone` touches `_ops` |
| An op's rank / channel / dtype / memory contract | `Op` trait methods, **no defaults** | Compile error: a new op that omits one does not build |
| An op's accepted input domains | `op_contract(...)["input_domains"]` (Rust `GraphStep::input_domains`) | `test_domain_vocabulary_declared_once` — `Pipeline` may not carry `DOMAIN_*` constants or a `_validate_domain` |
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
| Dtype spellings on the Python side | `python/polars_cv/_dtype_names.py`, generated from `dtype_table!` by `scripts/gen_dtype_names.py` | `test_dtype_names_module_is_current` (regenerate-and-diff), `test_engine_dtype_names_match_the_generated_table` pins `_types.DType` to it without the plugin |

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

### Test Structure

- Tests requiring the compiled Rust plugin are decorated with `@plugin_required` (class decorator) or use the `plugin_required` fixture from `conftest.py`.
- Unit tests for pure Python (schema inference, builder validation) live in files like `test_pipeline_builder.py` and `test_lazy_schema.py` and require no compiled plugin.
- Integration tests and reference tests (comparing output against NumPy/OpenCV ground truth) are in separate files under `tests/reference/`.
- Reuse `conftest.py` fixtures (`create_test_png`, `sample_image_bytes`, etc.) rather than redefining helpers per file.
- `test_contour_raster_crosscheck.py` checks the analytic contour measures
  (`area`, `centroid`, `iou`, `dice`, `contains_point`) against pixel counts on a
  rasterized mask — two independent implementations of the same quantity, so a
  fault in either shows up as a mismatch. Contours whose vertices are all
  integers on axis-aligned edges put no pixel centre on an edge, so those cases
  assert *exact* equality; diagonal and curved shapes assert a tolerance scaled
  by perimeter, since discretization error tracks boundary length, not area.
  Extend the `RECTILINEAR` / `CURVED` shape tables rather than adding one-off
  tests, and keep new rectilinear shapes on integer coordinates so they stay
  exact.

### Adding a New Operation

1. Implement in **view-buffer** (`view-buffer/src/ops/`) if it is a buffer→buffer
   engine op — add to the appropriate module, give it truthful `Op` contracts
   (shape/dtype/domain/channel rules), and register it in `ViewDto`
   (`tests/apply_op_coverage.rs` requires a probe per variant). Graph-level
   steps (node references, non-buffer outputs) become `GraphStep` variants in
   `polars-cv/src/graph/step.rs` instead.
2. Add a dispatch arm in **polars-cv** `src/execute.rs` (`resolve_op()`),
   returning the `GraphStep`. The Python planner picks up the op's schema
   effect automatically through the `op_schema` FFI — no Python-side schema
   special cases.
3. Add a method to `Pipeline` in `python/polars_cv/pipeline.py`, appending
   through `self._append_op(name, build_params)`. That is the only way in: it
   clones, builds the params, and routes to `_push_op`, which applies the
   input-domain check, the schema fold and the shape hints together. A builder
   that assigns `_current_domain`, `_output_dtype`, `_expected_ndim` or
   `_shape_hints` itself is doing the planner's job by hand and will fail
   `test_append_contract.py`. The matching `LazyPipelineExpr` method is
   generated automatically from `Pipeline` at import time
   (`python/polars_cv/lazy.py`) — do **not** hand-mirror it. If the op needs
   bespoke lazy behaviour (e.g. it takes another `LazyPipelineExpr` operand),
   define it explicitly on `LazyPipelineExpr` and the generator will skip it.
4. Regenerate the type stub: `python scripts/gen_lazy_stub.py` (CI guards it via
   `test_lazy_stub_is_current`).
5. Write tests covering both unit (builder validation) and integration (actual execution) cases.

**What makes an op "not implemented properly" here is not style, it is
reachability.** The registries above are what make an op resolvable, planned,
and surfaced to Python. An op that skips one of them does not get a degraded
experience — it gets rejected: no `Op` contract means no compile, no
`KNOWN_OPS` entry means `resolve_op` returns "Unknown operation" *and* the
parity tests fail, no `OP_NAMES` entry means the Python builder cannot name it,
and a shape effect the contract does not describe invalidates the hints rather
than publishing a schema execution cannot produce. If you are tempted to add a
Python-side special case for an op's schema, that is the signal the op's Rust
contract is incomplete — fix the contract.

---

## Detailed Context

Subsystem-specific AGENTS.md files provide deeper guidance:

- `AGENTS.md` — root navigation, cross-cutting architecture decisions
- `polars-cv/python/polars_cv/AGENTS.md` — Python API internals
- `polars-cv/src/AGENTS.md` — Rust plugin internals
- `view-buffer/AGENTS.md` — view-buffer engine internals
- `polars-cv/tests/AGENTS.md` — test conventions and fixture patterns
- `polars-cv/python/polars_cv/geometry/AGENTS.md` — geometry subsystem
- `polars-cv/python/polars_cv/metrics/AGENTS.md` — metrics subsystem
- `polars-cv/benchmarks/AGENTS.md` — benchmark framework

## Known Limitations

- f64 inputs through the float-promoting scalar ops execute correctly (in f64) but are excluded from kernel fusion, which computes in f32.
