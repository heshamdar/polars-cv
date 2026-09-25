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
  `output_channel_rule`, `output_dtype_rule`, `memory_effect`,
  `spatial_dependency` and `identity_rule` are required with no default so a new
  op cannot inherit a lie. Adding a default to any of them is a regression,
  however convenient.
- **One authority per fact, named once.** A dtype's spellings live in
  `dtype_table!`; enum variant names live in `named_variants!` + the
  `naming::REGISTRY`; op names live in the typed catalogue (`ops::TypedOp`),
  which Python's `TYPED_OPS` is generated from; an op's input domain lives
  in its Rust contract. If you find yourself writing a `match` that
  re-enumerates one of those, you are creating the second copy — read from the
  authority instead.
- **Registering is the same act as being checked.** Adding an enum to the
  registry is what surfaces it over FFI *and* what generates its Python class;
  adding an op's line to `typed_ops!` is what makes it deserializable,
  resolvable, described to Python *and* covered by the registry-driven tests.
  Never add a hand-written arm alongside the registry.

See [Canonical Paths](#canonical-paths) below for the principle and a pointer
to the concrete list of mechanisms (in [`AGENTS.md`](AGENTS.md#canonical-paths))
and the guard that enforces each one.

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
  computed from those codes. **The script lives at the repo root, not the
  `polars-cv/` subdirectory** — invoke it as `scripts/verify.sh` from the root
  (or by absolute path from anywhere; it `cd`s to the root itself). From the
  `polars-cv/` working directory the path is `../scripts/verify.sh`.
- Run ad-hoc cargo commands as `scripts/with-pyo3-env.sh cargo …` (or
  `source` it once per shell). A bare `cargo clippy`/`cargo test` lacks the PyO3
  environment `maturin develop` sets, so the two keep invalidating each other
  and the next `maturin develop` rebuilds the polars stack (~80s) for nothing.
  `verify.sh` and the pre-commit clippy hook already go through it, and on the
  web the SessionStart hook exports it into every session shell.
- **Never read a filtered view of a check and call it green.** `grep | head`
  cuts the failing suite below the fold; `maturin ... | tail` reports tail's
  exit code, not maturin's. Both have produced false "all green" reports here.
- The install is editable: Python edits take effect immediately, the compiled
  `.so` does not. After touching Rust, re-run `maturin develop` or you are
  testing old Rust against new Python — plugin tests self-skip rather than
  fail, so the window is silent. `polars_cv.build_info()` reports the versions
  that must agree and the source hash that detects a stale `.so` within a
  release cycle. Build **debug** (`maturin develop`, no `--release`):
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
- Inspect large files surgically. Several test files exceed 1,000 lines
  (`test_sanitation.py` is ~2,900); reading one whole to find one symbol burns
  tens of thousands of tokens for nothing. Grep for the symbol first, then read
  with `offset`/`limit` around the hit. Read a file whole only when you genuinely
  need the whole file.

---

## Commands

All commands should be run from the `polars-cv/` subdirectory unless noted otherwise.

### Build

```bash
uv sync --group dev --no-install-project   # Python dev deps only (see below)
maturin develop                            # Compile Rust plugin (debug) — THE dev build
maturin build --release                    # Build distributable wheels (rarely; CI does this)
scripts/dev-clean.sh                        # Reclaim disk: drop target/release + wheels, keep debug
```

**There is exactly one build for development: `maturin develop` (debug).**
Everything else is either CI/wheels (`--release`) or a mistake. Two traps put a
second, slow build into the loop, and both are now closed — keep them closed:

- **`uv sync` without `--no-install-project` builds the project at release LTO.**
  Because polars-cv uses the maturin backend, a plain `uv sync` compiles and
  installs the whole polars stack under `[profile.release]` (fat LTO,
  `codegen-units = 1`) — minutes of work that `maturin develop` then overwrites.
  Always pass `--no-install-project` and let `maturin develop` be the only
  extension build. CI and the SessionStart hook do this; `test_build_efficiency.py`
  guards it. `uv run` used to hit the same build on every call; `[tool.uv]
  package = false` in `pyproject.toml` now stops uv from ever building the
  project (CR-43).
- **Release artifacts fill the container.** A full `--release` tree is ~2 GB the
  dev loop never uses. The SessionStart hook clears a stale one on entry, and
  `scripts/dev-clean.sh` reclaims it on demand (`--all` also drops `target/debug`
  for a cold rebuild). Dev builds are further shrunk by root `Cargo.toml`'s
  `[profile.dev]` (`debug = 1`, `split-debuginfo = "unpacked"`), and linked with
  `lld` on x86_64-linux via `.cargo/config.toml` — the single biggest per-build
  speedup.

**Web sessions are provisioned by the SessionStart hook**
(`.claude/hooks/session-start.sh`, async): it updates the Rust toolchain, syncs
the `dev` and `docs` groups (`--no-install-project`), exports the PyO3 env to the
session, runs `maturin develop`, installs the pre-commit hook and, if missing,
`cargo install`s `cargo-deny` — everything `scripts/verify.sh` needs.
`.claude/hooks/.session-start.done` appears once it has finished; until then
plugin tests self-skip. `test_build_efficiency.py` pins what the hook must do.
Note that any `uv sync --no-install-project` uninstalls the editable project, so
re-run `maturin develop` after one.

**Check the Rust toolchain first — it is the common reason a build fails.** Both
crates set `rust-version = "1.96"` (MSRV) in their `Cargo.toml`, so cargo
*refuses to compile* on anything older, erroring before it starts:
`error: rustc <old> is not supported ... requires rustc 1.96`. A fresh
web/remote-session container often ships a stale `stable` toolchain — run
`rustc --version` and, if it is below 1.96, `rustup update stable` (or install a
newer toolchain) before `maturin develop`/`uv sync`. If you skip this the `.so`
is never built, and the whole plugin-dependent suite **self-skips silently**
(`@plugin_required`) rather than failing — so a green-looking run can be one that
tested nothing. Bump `rust-version` in both `Cargo.toml`s only deliberately: it
is the pinned MSRV, not a free knob.

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
(`__version__`, the compiled plugin, the installed distribution) plus the Rust
source hash the extension was built from against the working tree's, and
`tests/test_version_consistency.py` fails when either disagrees.

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

Rust unit tests (run from the workspace root or with `-p` flag), through
`scripts/with-pyo3-env.sh` so they do not invalidate the `maturin develop` build
(see [Verification](#verification); web sessions already have the env exported):
```bash
scripts/with-pyo3-env.sh cargo test -p view-buffer --all-features   # view-buffer engine tests
scripts/with-pyo3-env.sh cargo test -p polars-cv                    # Rust plugin tests
```

### Lint & Format

```bash
uvx ruff check python tests benchmarks         # Python lint (matches CI)
uvx ruff format python tests benchmarks        # Python format
cargo fmt --all -- --check       # Rust format check
../scripts/with-pyo3-env.sh cargo clippy --all-targets --all-features -- -D warnings  # Rust lint
```

A [pre-commit](https://pre-commit.com/) config wires these up;
`pre-commit>=4.5.1` is in the dev group. The config lives at the **repo root**
(`.pre-commit-config.yaml`), which is where pre-commit resolves it from — it sat
in `polars-cv/` for a while, where pre-commit never loaded it, so the structural
lane the hook exists to run was silently not running. Install the hooks from the
repo root:

```bash
cd "$(git rev-parse --show-toplevel)" && uv run --directory polars-cv pre-commit install
```

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
2. `.sink(...)` on a `LazyPipelineExpr` serializes the graph to JSON and calls `_plugin.call("vb_graph", ...)` — the only route to `register_plugin_function`, which pins the exact `.so` and hands the plugin storage rather than extension types.
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
| `_plugin.py` | `call()` — the one way into the compiled plugin: pins polars to the imported `.so`, passes every argument as `.ext.storage()` |
| `extension_types.py` | `NdArrayType`/`PointType`/`ContourType`/`BBoxType` Arrow extension types, registered at import; `NUMPY_OUTPUT_SCHEMA` |
| `display.py` | `show_images()` — notebook rendering of image columns |
| `_graph_viz.py` | Graph visualization (networkx/graphviz/pydot) |
| `geometry/` | Point/contour/bbox schemas and Polars expression namespaces |
| `metrics/` | Detection metrics (PR curves, AP, FROC, LROC, bootstrap, AUC) |

### Key Rust Modules

**polars-cv/src/**
- `lib.rs` — PyO3 module entry, `vb_graph` polars expression function, dtype inference, and the `plan_step`/`node_pass`/`enum_catalog`/`op_catalog`/`io_catalog`/`plan_source`/`plan_sink` FFI the Python planner reads (`plan.rs` holds `plan_step`, one call per appended op; `passes.rs` the node-scope optimisation passes)
- `ops/` — the typed op catalogue: one `#[derive(Op)]` struct per op, registered in `typed_ops!`; `TypedOp` is the wire op and `OpDef::resolve` maps it to a `GraphStep` (`graph/step.rs`: buffer ops wrap view-buffer's `ViewDto`; graph-only steps are their own variants)
- `formats/` — the typed sources and sinks, one struct per format in a `formats!` registry
- `execute.rs` — source decoding helpers (image bytes, contours) and byte-sink encoding
- `ops/` — the typed op catalogue (typed-op migration, `TYPED_OPS_PLAN.md`): one `#[derive(Op)]` struct + `OpDef` impl per op, `Param<T>`/`Literal<T>` fields, the `typed_ops!` registry and `catalog_json()`
- `graph/` — `UnifiedGraph` execution engine: `types.rs` (`UnifiedGraph`, `GraphNode`, `OutputSpec`, `RowErrorPolicy`), `compiled.rs` (process-wide compiled-graph cache), `step.rs` (`GraphStep` — the plugin-level step vocabulary), source decoding (`decode.rs`), sink encoding (`encode.rs`)
- `params.rs` — `ParamValue` resolving literals vs per-row Polars column values
- `pipeline.rs` — serde types for the JSON graph spec crossing the plugin boundary
- `cloud.rs` — remote/cloud transport (`object_store` backends, `cloud_options`, bounded-concurrency reads)
- `fetch.rs` — stage one of every path-based read: path column → bytes (`prefetch`, `row_bytes`, `parse_on_error`), shared by the `file_path` source and `read_bytes.rs`; owns `PathPolicy` (the `allowed_roots` sandbox)
- `read_bytes.rs` — `read_file_bytes` plugin function (`.cv.read_bytes()`) — `fetch.rs` with the decode omitted, for byte-identical passthrough
- `image_metadata.rs` — header-only metadata plugin functions (`.cv.width()`/`height()`/`channels()`/`image_dtype()`)
- `output.rs` — zero-copy numpy/torch struct output encoding
- `ext_types.rs` — `ExtType`, the polars-cv extension types (`polars_cv.ndarray`/`point`/`contour`/`bbox`); builds tagged outputs such as `sink("ndarray")`, published over FFI by `extension_types`
- `contour.rs`, `point.rs` — standalone plugin functions for geometry namespaces
- `geom_params.rs` — `GeomParams`: per-row resolution of those standalone functions' typed kwargs (`Param<T>` fields and `ColumnRef` operands, `{"$slot": n}` on the wire as for ops), reading the extra inputs the Python `_ArgBinder` appends

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

Most operation parameters accept either a literal (`224`) or a Polars expression (`pl.col("target_height")`). This is typed as `ParamValue` in `_types.py`. Per-row expression params are resolved in Rust via `params.rs` (each op through its typed `Param<T>` fields), and by `geom_params.rs` for the `.contour`/`.point`/`.bbox` namespaces, which bypass `vb_graph` but carry their expression params the same way: extra plugin inputs, each named by its kwarg's `{"$slot": n}`.

The rule for whether a parameter may be per-row is *not* its type: **a parameter is eligible iff its value has no effect on the output shape, rank or dtype**, because the lazy schema is computed at plan time and must match what executes. So non-structural enums and flags (`filter`, `interpolation`, `pad(mode=)`, `convolve2d(border=, normalize=)`, …) are per-row, while structural parameters are literal-only: `cast(dtype=)`, `normalize(method=, out_dtype=)`, reduction `axis`, `perceptual_hash(hash_size=, algorithm=)`, `rotate(expand=)`, `histogram(closed=, output=)`, the `transpose`/`flip` axis lists and `reshape`'s element count. For a list-valued parameter the *length* is structural while the elements are not — a `convolve2d` kernel keeps a literal element count but each coefficient may be an expression. Plan-time shapes are symbolic (a per-row param is `Sym::PerRow` in the op's `OpShape`), but the planner still resolves each op once without a row to read its rules, where every per-row param takes a placeholder (`ParamCtx::planning`); that is sound only because of the eligibility rule.

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

The concrete table — every fact, its single authority, and the exact guard that
rejects a second declaration — lives in
[`AGENTS.md`](AGENTS.md#canonical-paths). It is reference material you reach for
when adding an op, enum, dtype spelling, source/sink parameter or optimization
pass, so it loads on demand there rather than in every session's context. The
former exceptions (`OpSpec`'s `#[serde(flatten)]` params, and the `BinaryOp`
arm, both since removed) are documented alongside it.

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
2. Define it in the **typed catalogue**, `polars-cv/src/ops/<family>.rs`: a
   struct deriving `Op` (plus `Serialize`/`Deserialize` and
   `#[serde(deny_unknown_fields)]` — the derive refuses to compile without it)
   whose fields are `Param<T>` (may be per-row) or `Literal<T>` (structural),
   each with a doc comment (the generated `Args:` entry) and, where Python has
   one, `#[param(default = ...)]` (an op's only required field is generated
   positional-or-keyword, every other keyword-only); an `OpDef` impl
   that opens with an exhaustive destructure and returns the `GraphStep`; and
   one line with a valid sample in `typed_ops!` (`ops/mod.rs`). A parameter
   read only under some branch becomes an enum variant, never an optional
   field that can be ignored. The Python planner picks up the op's schema
   effect through `plan_step` — no Python-side schema special cases.
3. Re-bless the catalogue (`POLARS_CV_BLESS=1 scripts/with-pyo3-env.sh cargo
   test -p polars-cv catalog_matches`), regenerate the builder (`python
   scripts/gen_ops.py`) and `maturin develop`. The generated method appends
   through `Pipeline._append_typed` → `_append_op` → `_push_op`, the only way
   in. The matching `LazyPipelineExpr` method is generated automatically from
   `Pipeline` at import time (`python/polars_cv/lazy.py`) — do **not**
   hand-mirror it. Hand-written `Pipeline` methods are for sugar over generated
   ones (`flip_h`, `thumbnail`, …) and the `lazy_only` ops on
   `LazyPipelineExpr`.
4. Regenerate the type stub: `python scripts/gen_lazy_stub.py` (CI guards it via
   `test_lazy_stub_is_current`).
5. Write tests covering both unit (builder validation) and integration (actual execution) cases.

**What makes an op "not implemented properly" here is not style, it is
reachability.** The registries above are what make an op resolvable, planned,
and surfaced to Python. An op that skips one of them does not get a degraded
experience — it gets rejected: no `Op` contract means no compile, no
`typed_ops!` line means the wire rejects it as "Unknown operation" and no
builder is generated for it, and a shape effect the contract does not describe invalidates the hints rather
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
