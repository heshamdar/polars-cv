# AGENTS.md — Tests (`polars-cv/tests/`)

> Read the [root AGENTS.md](../../AGENTS.md) first for project-wide context.
> Update this file when you add new test patterns, fixtures, or change testing conventions.

## Purpose

All Python tests for polars-cv. Tests use **pytest** exclusively. Coverage includes the Python API, pipeline builder, schema inference, plugin execution, and reference correctness against NumPy/OpenCV.

## Running Tests

To verify a change, run **`scripts/verify.sh`** from the repo root. It runs
every check CI runs, prints each one's exit code, and ends in a single
`PASS`/`FAIL` line computed from those codes.

Use it in preference to running the checks by hand and reading the output.
Reading a *filtered view* of a check has produced false "all green" reports
here more than once: a `grep | head` that cut the failing suite off below the
fold, and a `maturin … | tail` whose reported exit code belonged to `tail`
rather than to `maturin`. Both looked exactly like success. If you do run a
check by hand, take its exit code directly — `cmd | tail` returns `tail`'s
status, so use `${PIPESTATUS[0]}` or `set -o pipefail`.

`test_verify_script_covers_every_ci_check` pins the script to CI, so a check
added to one and not the other fails rather than leaving a local `PASS` that
does not mean CI passes.

The individual lanes:

```bash
cd polars-cv
uv run pytest tests/ -m "structural and not slow"   # the pre-commit lane; ~5s, no build needed
uv run pytest tests/ -m "not network and not slow"  # what CI runs on every push
uv run pytest tests/                                # everything; plugin tests self-skip if unbuilt
uv run pytest tests/reference/ -v                   # reference tests
python scripts/test_multiple_python.py --all        # multi-Python (3.10-3.13)
```

The compiled plugin (`.so`/`.pyd`) must exist at `python/polars_cv/_lib.abi3.so`.
Build it with `maturin develop` — **debug**, not `--release`. That is what CI and
`scripts/verify.sh` build, the whole suite passes against it, and a release
build costs several minutes per iteration for no test-visible difference.

### Markers and CI lanes

Three markers are declared in `pyproject.toml` (`[tool.pytest.ini_options]`), and
**CI never runs the bare `pytest tests/` you probably run locally**:

| Marker | Meaning | Where it runs |
|--------|---------|---------------|
| `network` | Needs network access | Never in CI |
| `slow` | Long-running | `slow-tests` job only — weekly schedule or manual dispatch, `-m "slow and not network"` (`.github/workflows/ci.yml`) |
| `structural` | Checks codebase *shape*, not runtime behaviour | The pre-commit hook, its own step in the `test` job (after `maturin develop`), and its own `verify.sh` line — as well as the fast lane, which it is a subset of |

The per-push lane is `-m "not network and not slow"`. Mark a new test `network`
if it fetches anything and `slow` if it is not sub-second — an unmarked slow test
lands in the lane that gates every merge.

#### The `structural` lane

`-m "structural and not slow"` is ~650 tests in about five seconds. They check
the *shape* of the codebase — registries, single-authority ratchets, removed
surfaces, documented vocabularies — rather than the numerical behaviour of a
pipeline, and five seconds is what makes them affordable in a pre-commit hook.

**The lane needs the compiled extension.** 427 of the 651 are
`@plugin_required`, because a great many structural facts are only observable
through the FFI: the enum-parity sweep reads `enum_variants`, and
`test_param_applicability` sweeps real `source()`/`sink()` calls. Without a
`.so` they do not politely skip — sixteen fail outright, and
`test_schema_parity_chains.py` aborts *collection*, because its `parametrize`
reads `_lib` at import time, which takes the whole run down before any
`-m` filtering happens.

So the hook and `scripts/verify.sh` both assume a built working tree, and the
CI step runs in the `test` job after `maturin develop` rather than in `lint`.
An earlier version of this ran in `lint` with `--no-install-project` on the
belief that these tests self-skip; it went red on the first push. If you are
tempted to move it somewhere cheaper, that is the reason not to.

Applied per module, not per test — a module is either a guard suite or it is not:

```python
pytestmark = pytest.mark.structural
```

Two guards keep the lane honest, both in `test_sanitation.py`:

- `test_every_source_scanning_module_declares_its_lane` — any module importing
  `_discovery` (the mandated way to find files to scan) must declare
  `structural` or `slow`. The set is *derived*, so a new source-scanning guard
  cannot join without choosing. Its limit: a guard reading one named file has no
  reason to import `_discovery` and is marked by hand.
- `test_the_structural_lane_is_not_empty` — a renamed marker or a dropped
  `pyproject.toml` declaration would make the hook select nothing and pass in
  milliseconds, reading as a working guard forever.

Keep the lane fast. A guard that decodes images, runs subprocesses or builds
anything belongs in `slow` — `test_examples.py` is the worked example, and it
answers `slow` to the lane-declaration guard for exactly that reason.

## How the Suite Is Organised

- `tests/*.py` — unit tests (pure Python: builder, schema inference,
  serialization) and integration tests (execute a graph through the compiled
  plugin). The `@plugin_required` decorator, not the filename, is what separates
  them.
- `tests/reference/*_ref.py` — output compared against NumPy / OpenCV / PIL /
  imagehash ground truth. These are the **correctness guarantees**: if a change
  moves one of these expectations, say why in the commit rather than retuning
  the tolerance.
- `_gaps` in a name (e.g. `test_binary_ops_gaps.py`) is historical — they are
  ordinary tests whose gaps have since been filled.

There is deliberately no per-file index here: it goes stale the moment a file is
added. Find the right file by name or content instead:

```bash
ls tests/                                     # names are topic-shaped
uv run pytest --collect-only -q -k <topic>    # locate existing coverage
rg "<op_name>" tests/                         # find who already exercises an op
```

Useful entry points when you need a pattern to copy:

| File | Pattern it demonstrates |
|------|-------------------------|
| `test_pipeline_builder.py` | Builder validation and domain tracking, no plugin |
| `test_lazy_schema.py` | Plan-time schema assertions on `LazyPipelineExpr` |
| `test_integration.py` | End-to-end execution through Polars |
| `test_schema_inference.py` | Planning-time vs execution-time schema agreement |
| `test_source_types.py` | Exercising each source format |
| `test_sanitation.py` | Meta-tests that police the conventions below |
| `test_contour_raster_crosscheck.py` | Differential testing: one quantity, two implementations |
| `test_expression_op_params.py` | Table-driven sweep with a coverage ratchet |

### Per-row (expression) parameters

Nearly every `Pipeline` parameter accepts a `pl.Expr` resolved per row, and
they are swept as one table rather than one test per op:

- `_expr_param_cases.py` — the table. One `ExprCase` per
  `method.parameter`, carrying a pipeline factory and the per-row values.
- `_expr_param_runner.py` — the harness, shared with
  `test_param_strictness.py` so the differential comparison exists once.
- `test_expression_op_params.py` — the sweep and the ratchet.
- `test_expression_workflows.py` — multi-stage workflows (measure the batch
  with one pipeline, parameterise the next with an aggregate over it).

Three claims are checked per case, and **none implies the others**:

1. each row equals the pipeline built with that row's value as a *literal*
   (an off-by-one in `row_idx` produces differing rows that still fail here);
2. distinct values produce distinct outputs (the only check that can see a
   parameter the kernel ignores — a comparison against the literal path
   cannot, since it would be ignored there too);
3. a row's result is the same alone as inside a batch (morsel boundaries,
   broadcasting, the compiled-graph cache).

`test_every_expression_parameter_has_a_case` reads the eligible parameters
off `Pipeline`'s live signatures — an annotation admitting `pl.Expr` — so a
new expression-valued parameter fails the ratchet until it is swept or
exempted in `NOT_SWEPT` with a reason. Add cases to the table rather than
writing one-off tests, and give a case `varies=False` only with a note
saying why the parameter cannot change the output.

### Differential tests against the rasterizer

`test_contour_raster_crosscheck.py` measures the same quantities two ways —
`area`, `centroid`, `iou`, `dice` and `contains_point` from polygon geometry,
versus pixel counts on a mask produced by the scanline filler. The two share no
code, so a fault in either surfaces as a disagreement even when each is
internally consistent. This is what caught the scanline filler painting a
surplus column at every right-hand edge, which every existing rasterize test had
missed.

The file also closes the loop the other way — contour → mask → contour through
`extract_contours`. That leg is lossy in one *predictable* way: the tracer
reports the centres of the boundary pixels, so what comes back is inset half a
pixel and a `w x h` region returns as `(w-1) x (h-1)`. The tests assert that
inset exactly rather than tolerating it; a tolerance wide enough to absorb it
would also absorb a tracer that collapsed, which is exactly the bug these tests
were written to catch.

The shapes live in two tables. `RECTILINEAR` shapes have integer vertices on
axis-aligned edges, so no pixel centre ever lands on an edge and the mask count
equals the area **exactly** — those cases assert equality, and are where the
suite gets its power. `CURVED` shapes have diagonal or curved edges that cut
through pixels, so they assert a tolerance scaled by perimeter (discretization
error tracks boundary length, not area). Add cases to the tables rather than
writing one-off tests, and keep rectilinear additions on integer coordinates so
they stay exact.

## Conventions

### Test Structure

- **Class-based**: `class TestSomething:` with methods
- **Docstrings**: All test classes and methods should have docstrings
- **Type annotations**: All fixtures and test methods should have return type annotations
- **Contour native sink**: `extract_contours().sink("native")` returns `List[Struct]`, not a single struct. Access via `.list.get(0).struct.field("exterior")`.

### Plugin Requirement

```python
from tests.conftest import plugin_required


@plugin_required
class TestMyFeature:
    """Tests that need the compiled Rust plugin."""

    ...
```

The rule: anything that **executes** a graph (`.sink(...)` collected through
Polars, the `.cv`/`.point`/`.contour`/`.bbox` plugin functions, any FFI call into
`_lib`) needs `@plugin_required`. Tests that only build pipelines or assert on
plan-time schema must not use it — they are the lane that still runs when the
plugin is not built.

`plugin_required` is a `pytest.mark.skipif`, not a named marker, so it cannot
be selected on: `-k "not plugin_required"` matches test *names* and deselects
nothing, and `-m` sees no such marker. None is needed — the tests skip
themselves when the extension is absent. Because skips are quiet, a builder
change that should have failed a parity test can look clean against an unbuilt
or stale plugin; the parity guards that can run without one
(`test_op_names_matches_rust_known_ops_without_the_plugin`,
`test_op_names_covers_all_emitted_ops`) exist to cover that window.

### Shared Fixtures (`conftest.py`)

| Fixture | Purpose |
|---------|---------|
| `create_test_png` | Factory: create PNG bytes for given width, height, color |
| `encode_png` | Encode a numpy array as PNG bytes |
| `sample_image_bytes` | Minimal 1x1 red PNG (no PIL dependency) |
| `plugin_required` | Skip if compiled plugin not available |

Reference tests have additional fixtures in `reference/conftest.py` (session-scoped images, contour data).

### Writing a New Test

```python
"""Tests for my plugin feature."""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import polars as pl
import pytest
from polars_cv import Pipeline
from tests.conftest import plugin_required

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest


@plugin_required
class TestMyFeature:
    """Tests that execute through the Rust plugin."""

    def test_end_to_end(self, create_test_png: Callable) -> None:
        """Verify end-to-end execution."""
        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"image": [png_bytes]})
        pipe = Pipeline().source("image_bytes").my_op(param=42)
        result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
        assert result["output"].dtype == pl.Struct(...)
```

### Shared Fixtures Are Mandatory

`plugin_required` and PNG construction live only in `conftest.py`. Import
`plugin_required` from `tests.conftest`; inside fixtures use the
`create_test_png` factory fixture, and for module-level helpers import
`make_test_png` (`from tests.conftest import make_test_png`). Meta-tests in
`test_sanitation.py` (`test_no_local_plugin_available_definitions`,
`test_no_local_png_factories`) fail the suite if a test file redefines
either.

## Writing a Sanitation Guard

The guards in `test_sanitation.py` are the enforcement mechanism for the
single-authority invariants, so a guard that cannot fail is worse than no
guard: it reads as coverage. The dtype-dispatch ratchet was wrong **seven**
times before it settled, and every time it kept passing while covering less
than the version it replaced. Twice the regression was introduced while fixing
the previous one.

Two rules follow from that.

**A new guard is not done until you have watched it fail.** Break the thing it
claims to catch, confirm it reports that specific thing, revert. If you cannot
construct a failure, the guard is decorative. Check the *reason* too — three of
those seven failed for a different reason than they claimed, which is how the
next rewrite lost coverage without anyone noticing.

**A guard with non-trivial logic gets committed fixtures.** Put the logic in a
helper the guard imports (`tests/_dtype_ratchet.py` is the worked example),
then add known-bad inputs it must flag and known-good inputs it must not, as in
`test_dtype_ratchet_fixtures.py`. The fixtures must call the same helper the
real guard calls — a fixture exercising a copy proves nothing about the guard.

Every past blind spot becomes a fixture, so re-introducing one fails the suite.
Keep the good fixtures too: three of the seven rewrites false-positived on
correct, `rustfmt`-clean code, and a guard that fires on valid code gets
weakened or deleted by whoever hits it next.

Beware thresholds. A "six or more names must be all ten" rule made damage
self-concealing — dropping four arms failed and dropping five passed. Prefer a
rule whose sensitivity does not fall off as the defect grows, and test the
whole range rather than one example.

### Discovery goes through `tests/_discovery.py`

Almost every guard here is "find some files, then assert something about
them", and that shape has one failure mode this suite has shipped twice: the
find returns nothing, the assertion holds for free, and the guard reads as
coverage forever. `_test_files()` and `_PACKAGE_MODULES` were both a bare
`rglob` whose empty result was indistinguishable from a clean bill of health,
and the second is what the whole `_push_op` append contract rests on.

So **do not glob in a test module.** Call `rust_src_dir()`, `rust_sources()`,
`package_modules()`, `suite_files()` or `suite_modules()` from
`tests/_discovery.py`; each either returns a non-empty result or raises
`EmptyDiscovery`, which reports as an ordinary failure. Need a set the module
does not offer? Add an accessor there, routed through `discovered()`.

`test_scans_go_through_discovery` enforces this by walking the suite's AST, so
it is a mechanism rather than a reminder. If an empty answer is genuinely the
assertion — `test_streaming_ooc.py` checks that a spill directory *stays*
empty — add the file to `_DISCOVERY_EXEMPT` with the reason.

The skip is keyed on an independent fact for the same reason: `in_checkout()`
asks whether the version manifests are present, and only then does
`rust_src_dir()` insist the crate sources exist. A fact used to excuse a
missing scan must not be the fact the scan is looking for — the previous
`_rust_src_dir()` answered a moved layout with `None`, and five guards read
that as "installed wheel, nothing to check".

## Changing Behaviour

If a change alters what a caller sees — a signature, whether something raises,
a dtype — exercise the **user-facing entry point**, not just the helper you
edited. A change that made `_polars_dtype_to_cv` raise for boolean masks was
verified at the planner and shipped as a bug fix; it had in fact broken the
documented input shape for `ContourMatcher`, which one call through the public
API would have shown.

## Multi-Fix Commits

When a commit applies a list of fixes (a review's findings, say), write a
throwaway script with **one check per item** and run it to produce a
fixed/still-broken table, rather than asserting from memory. Doing this found
three claimed fixes that a batch script had silently skipped when it aborted
partway — and two bugs in the checking script itself, including a check that
could not run at all and so printed a failure it never tested. A batch edit
should collect and report its misses, never `exit` on the first one.
