# Handover: typed-op consolidation (C0–C8)

For the next agent picking up branch `claude/pr99-refactor-review-htrbfy`.
Read this, then `CLAUDE.md` (working agreements — binding), then
`TYPED_OPS_CONSOLIDATION_PLAN.md` (the plan and its status ledger).

## Where things are

- **Branch:** `claude/pr99-refactor-review-htrbfy`, based on the PR #99 head
  (`0f22dd4`). Push only to this branch.
- **Open PR:** #100 (`heshamdar/polars-cv`, base `main`) is from this branch.
  Its title/description ("Resolve P0 quality-review findings (CR-41..CR-44)")
  predate this work and do not describe C0–C5; update them before review.
- **The plan:** `TYPED_OPS_CONSOLIDATION_PLAN.md` — goal, rules for every
  phase, one ledger table per phase (what to delete, where, replaced by), and
  the status ledger at the top. Its predecessor `TYPED_OPS_PLAN.md` (P0–P10)
  is history.
- **User-facing record:** `CHANGELOG.md` (Unreleased) and
  `polars-cv/docs/user-guide/migration.md` (breaking changes and decisions).

## The goal, in one paragraph

Every fact has one authority. Each op is defined once — a variant of a
mode-generic family whose `Wire` form (`Param<T>`/`Literal<T>` fields: a
literal or a per-row slot) is what a plan holds and reads its rules from, and
whose `Exec` form (plain values) is what runs. The planner lives in Rust and
never invents a value for a per-row parameter. Ops, sources, sinks and the
geometry accessors register through one mechanism with one default convention,
and every Python method is generated.

## State: C0–C5 done

| Phase | Commit(s) | What it did |
|---|---|---|
| plan | `67f7d39` | wrote the consolidation plan |
| C0 | `86d31fd` | correctness fixes, test-first |
| C1 | `ac249e6` | planner state is one typed Rust object (`PlanState`) |
| C2 | `a8f18cb` | `assert_shape` is a checked op; the plugin plans the graph |
| C3 | `51034c9` | Rust `Plan` owns a pipeline's op list |
| C4a | `4657dbc` | rank/channels read off `OpShape` |
| C4b | `0187bac` `54dc6f2` `69f631b` `9cba435` `ef374ec` | every wire op is a variant of a mode-generic family; `OpDef` and all per-op typed structs deleted |
| C4c | `aacb938` | `TypedOp = GraphStep<Wire>`; every `Op` rule generic over the mode; placeholder planning deleted |
| C5 | `ac45f93` | geometry accessors are typed definitions; Python methods generated |

### The architecture now

- **`view_buffer::mode`** — `Mode` (`Exec`, `Wire`), `Param<T>`/`Literal<T>`,
  `Resolve` (Wire → Exec, per row), `Values`/`Literals`, `WireOps` (what
  `#[derive(Ops)]` gives a family's `Wire` form), the catalogue types.
- **Families** (derive `Ops` + `Resolve`, in `polars-cv-macros/src/modal.rs`):
  view-buffer's `ImageOpKind`, `ComputeOp`, `ViewOp`, `ColorConvertOp`,
  `ConvolveOp`, `GeometryOp`, `ReductionOp`, `HistogramOp`, `PerceptualHashOp`;
  polars-cv's `GraphOp` (`src/ops/graph.rs`, graph-only ops, dispatched through
  its exhaustive `Role`). Each variant carries its wire name, doc (the Python
  docstring), field docs, `#[param(default = …)]` and a sample; each family has
  one generic `shape()` and `check()`, and its `Op` impl is generic over `M`
  (a rule reading a per-row value reads it through `M::sym`/`mode::known` and
  says nothing about one it does not know).
- **`GraphStep<M>`** (`src/graph/step.rs`): `Buffer(ViewDto<M>)`, `Geometry`,
  `Reduction`, `Histogram`, `PerceptualHash`, `Graph(GraphOp<M>)`.
  `TypedOp = GraphStep<Wire>` (`src/ops/mod.rs`, `typed_ops!` lists each family
  with its position). `TypedOp::resolve(row, ctx)` = derived `Resolve` of the
  whole step, then `check()`. The planner (`src/plan.rs`) and passes
  (`src/passes.rs`) read rules from the `Wire` op directly.
- **Geometry accessors** (C5): `src/geom_fns.rs` — `ContourFn`/`PointFn`/
  `BBoxFn` families; op-backed accessors *are* `GeometryOp` variants
  (`OP_ACCESSORS`). Each `#[polars_expr]` parses its definition with
  `GeomParams::parse` (`src/geom_params.rs`); Python methods are generated into
  `_ContourOpsMixin`/`_PointOpsMixin`/`_BBoxOpsMixin` and call
  `_GeomNamespace._call` (`python/polars_cv/_namespace.py`), which checks
  literals at build time via the `check_geom_call` FFI.

### Decisions the user made (recorded; do not relitigate)

- **Contour scale `origin` default: `"centroid"`** for both
  `Pipeline.scale_contour` and `.contour.scale` (done in C5).
- **`convolve2d(ksize=)`: drop it** — the side comes from the kernel length
  (an odd square). This is C7's row; not yet done.

### Deviations from the plan, and open items

- `rotate` keeps its value-chosen lowering (lattice angles → zero-copy views,
  else `RotateAffine`; `warp_affine` → `Affine`), now only inside
  `ViewExpr::apply_op` (`ComputeOp::lowered`). `a_lowered_op_keeps_its_shape`
  (view-buffer) pins the lowered shape to the op's.
- `plan::check_rank` still passes `1` for an unknown size to `validate`,
  filtered to rank-only verdicts. Removing it needs `validate` over symbolic
  shapes (`&[Dim]`) across every `Op` impl — optional follow-up.
- `#[derive(Ops)]` refuses a *missing* field even when it declares a default
  (the generated Python always sends every field). C6 changes that.
- `tests/test_known_gaps.py` has no open gap left (its mechanism is kept).

## Next steps

Follow the plan's rules for each: **delete first** (every row of the phase's
ledger), let the compiler/suite report dependents, wire the replacement;
behaviour changes test-first; each Python/prose removal added to
`polars-cv/scripts/check_removed_symbols.py`; tick the status ledger in the
phase commit; full `scripts/verify.sh` green at exit; golden corpus unchanged
unless the commit says which entries change and why.

### C6 — One registry, one default convention

- `formats!` (`polars-cv/src/formats/mod.rs:71`) is a near copy of the old
  per-op registry: make sources and sinks families (derive `Ops` on an enum or
  on each struct, as ops do) or one `registry!` shared with `typed_ops!`.
- Defaults applied by the derive when a field is missing: change
  `polars-cv-macros/src/modal.rs` (`derive_ops`, the `takes` for a non-`Option`
  field with `param_default`) to substitute the declared default, then delete
  the `Option` + `unwrap_or` defaults: `JpegSink::DEFAULT_QUALITY`
  (`formats/sink.rs:30,84-89`), `ContourSource::fill` 255/0
  (`formats/source.rs`), `plan.rs:414` (`Param::Lit(255)`).
- `ContourSource` carries a copy of rasterize's fields: make it carry a
  `GeometryOp::Rasterize` (or its fields' struct).
- `Sink::quality()/shape()/as_f16()` `_ =>` arms (`formats/sink.rs:84-110`):
  make them data on each typed sink.
- `SourceFormat` mirrors `Source` (`graph/compiled.rs:1338`): dispatch on
  `Source` instead.
- `check_applies` rebuilds the catalogue per parse (`formats/mod.rs:29`): a
  static catalogue.
- Stringly typed `visibility` (`"public"`/`"internal"`/`"lazy_only"`) in the
  macros, `OpDesc` and `gen_ops.py`: a `Visibility` enum.

### C7 — Python surface fully generated

- `Pipeline.source()` body (`python/polars_cv/pipeline.py`, search
  `def source`): generate from `io_catalog.json`; `RasterSize` already
  expresses the width/height vs shape exclusivity.
- Then delete `_validate_enum` and `_reject_expr` (`_types.py`) once unused
  (`_enum_or_expr` went in C5).
- Runtime lazy forwarders (`lazy.py`: `_install_pipeline_forwarders`,
  `_make_forwarder`, `_chainable_pipeline_ops`) → emitted by `gen_ops.py`;
  fold `scripts/gen_lazy_stub.py` into `gen_ops.py` (writes `lazy.pyi`).
- "Domain: a → b" docstring prose in op docs → generated from each op's domain
  contract (`GraphStep::input_domains`/`output_domain`); the op-backed geometry
  accessors currently show that prose too.
- `convolve2d(ksize=)`: **drop** (decision above). `ConvolveOp` in
  `view-buffer/src/ops/filter.rs` derives the side from `kernel.len()`
  already (`side()`, `check()`); remove the field, update the migration page,
  CHANGELOG, `signatures.json` (`scripts/gen_signature_snapshot.py`), and pin
  the removal in `tests/test_removed_surfaces.py`.

### C8 — Sweep, guards and docs

- Remove absence scans whose subject no longer exists (list each in the commit
  with the structural mechanism that replaced it).
- `check_removed_symbols.py` complete for every phase.
- `CLAUDE.md`, `AGENTS.md` (root, `polars-cv/src/`, `polars-cv/python/…`),
  `docs/user-guide/migration.md`: restate the per-row rule as "no effect on rank
  or dtype; a size may be per-row and is then unknown at plan time"; module
  tables; the `planned` wire field gone.
- Fill the plan's line-count ledger per phase (method below).

## Commands (run from the repo root unless noted)

```bash
scripts/verify.sh              # the gate (full); --fast skips the slow lane
scripts/with-pyo3-env.sh cargo test -p view-buffer --all-features
scripts/with-pyo3-env.sh cargo test -p polars-cv --lib
POLARS_CV_BLESS=1 scripts/with-pyo3-env.sh cargo test -p polars-cv --lib catalog_matches
cd polars-cv
.venv/bin/maturin develop                       # after any Rust change (debug)
.venv/bin/python scripts/gen_ops.py             # regenerate _ops_generated.py
.venv/bin/python scripts/gen_lazy_stub.py       # regenerate lazy.pyi
.venv/bin/python scripts/gen_signature_snapshot.py   # deliberate API change only
.venv/bin/python scripts/gen_golden_corpus.py   # deliberate plan/output change only
.venv/bin/python scripts/check_removed_symbols.py
.venv/bin/python -m pytest tests -m "not network and not slow" -q
```

Golden files (`polars-cv/tests/golden/`): `op_catalog.json`,
`geom_catalog.json`, `io_catalog.json`, `enum_catalog.json`,
`pass_catalog.json` (Rust `catalog_matches` tests; bless), `op_corpus.json`
(recorded plan/schema/output per op), `signatures.json` (public call surface).

## Environment gotchas (learned the hard way)

- **Run `scripts/…` from the repo root.** The Bash working directory drifts to
  `polars-cv/` after commands that `cd`; a `scripts/with-pyo3-env.sh` there
  fails with "No such file", and a filtered pipe (`| grep`) hides it — one
  "clean" compile in this session was a command that never ran.
- **Never read a filtered view as green** (`CLAUDE.md`): check the exit code
  or the unfiltered tail.
- **Disk is a fixed allowance** (~6 GB free now). `target/debug/deps` grows
  with stale test binaries; delete the extensionless executables there (they
  relink on the next run) before deleting the polars rlibs (minutes to rebuild).
- **After Rust changes, `maturin develop`** or Python tests run the old `.so`.
- **`cargo fmt` may need two passes** after large edits (verify's fmt check
  failed once right after one pass).
- A new FFI must be added to `_REQUIRED_LIB_HOOKS` in
  `polars-cv/tests/test_sanitation.py` with its reason.

## Line-count ledger (method)

`plugin` = `polars-cv/src/**/*.rs`; `engine` = `view-buffer/src/**`;
`macros` = `polars-cv-macros/src/**`; `py-gen` =
`polars-cv/python/polars_cv/_ops_generated.py`; `py-hand` = the other `.py`
under `polars-cv/python`; `tests` = `.py` under `polars-cv/tests`. Counted
with `git ls-tree`/`git show … | wc -l` at a revision (this reproduces the
plan's PR #99 row exactly).

| At | plugin | engine | macros | py-hand | py-gen | tests |
|---|---:|---:|---:|---:|---:|---:|
| PR #99 head `0f22dd4` | 19,097 | 21,233 | 197 | 12,022 | 2,127 | 55,164 |
| after C5 `ac45f93` | 17,448 | 22,474 | 767 | 10,496 | 2,715 | 54,992 |

Net since PR #99: hand-written plugin −1,649 and Python −1,526; the engine
(+1,241) and macros (+570) grew because the op definitions, their modes and
the derives moved there from the plugin; generated Python +588 (the accessor
methods).
