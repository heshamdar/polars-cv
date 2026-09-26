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

## State: C0–C6 done

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
| C6 | `634bd36` `5e72589` `820995b` `ded469c` | wire applies declared defaults; `Visibility` enum; contour source only decodes (canvas keywords are `rasterize()`); `Source`/`Sink` are `#[derive(Ops)]` families; `SourceFormat` gone, `auto` routed to a concrete `Source` |

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
- **A contour source only decodes** (C6c): `source("contour", width=, …)` is
  `source("contour").rasterize(...)`, and a bare `source("contour")` is the
  contour domain.
- **Families with no per-row value have no mode parameter** (C6b).
- **Elegance over the plan's letter is welcome, but ask first** when it
  changes a public surface.

### Deviations from the plan, and open items

- `rotate` keeps its value-chosen lowering (lattice angles → zero-copy views,
  else `RotateAffine`; `warp_affine` → `Affine`), now only inside
  `ViewExpr::apply_op` (`ComputeOp::lowered`). `a_lowered_op_keeps_its_shape`
  (view-buffer) pins the lowered shape to the op's.
- `plan::check_rank` still passes `1` for an unknown size to `validate`,
  filtered to rank-only verdicts. Removing it needs `validate` over symbolic
  shapes (`&[Dim]`) across every `Op` impl — optional follow-up.
- A family with no per-row value (`Source`, `Sink`) derives `Ops` with no
  mode parameter: it is its own wire form and has no `Resolve`.
- `tests/test_known_gaps.py` has no open gap left (its mechanism is kept).

## Next steps

Follow the plan's rules for each: **delete first** (every row of the phase's
ledger), let the compiler/suite report dependents, wire the replacement;
behaviour changes test-first; each Python/prose removal added to
`polars-cv/scripts/check_removed_symbols.py`; tick the status ledger in the
phase commit; full `scripts/verify.sh` green at exit; golden corpus unchanged
unless the commit says which entries change and why.

### C6 — done

See the status ledger. Two user decisions shaped it: a contour source only
decodes (the canvas keywords append `rasterize()`, and a bare
`source("contour")` is the contour domain), and the derive accepts a family
with no mode. There is no `registry!` macro: `#[derive(Ops)]` is the one
registry mechanism for ops, geometry functions, sources and sinks, and
`typed_ops!` only places op families in `GraphStep`.

### C7 — Python surface fully generated

- `Pipeline.source()` body (`python/polars_cv/pipeline.py`, search
  `def source`): generate from `io_catalog.json`. Its canvas keywords are
  `rasterize()`'s (C6c): the generated method appends `rasterize(**canvas)`
  when any is given, reading the keyword set from `rasterize`'s signature as
  the hand-written body does now.
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
| `main` before the migration `ac2e95b` | 16,986 | 20,732 | 0 | 16,078 | 0 | 53,972 |
| PR #99 head `0f22dd4` | 19,097 | 21,233 | 197 | 12,022 | 2,127 | 55,164 |
| after C5 `ac45f93` | 17,448 | 22,474 | 767 | 10,496 | 2,715 | 54,992 |
| after C6 `ded469c` | 17,254 | 22,501 | 653 | 10,465 | 2,715 | 55,111 |

Net since PR #99: hand-written plugin −1,649 and Python −1,526; the engine
(+1,241) and macros (+570) grew because the op definitions, their modes and
the derives moved there from the plugin; generated Python +588 (the accessor
methods).
