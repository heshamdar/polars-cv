# AGENTS.md — Tests (`polars-cv/tests/`)

> Read the [root AGENTS.md](../../AGENTS.md) first for project-wide context.
> Update this file when you add new test patterns, fixtures, or change testing conventions.

## Purpose

All Python tests for polars-cv. Tests use **pytest** exclusively. Coverage includes the Python API, pipeline builder, schema inference, plugin execution, and reference correctness against NumPy/OpenCV.

## Running Tests

```bash
cd polars-cv
uv run pytest tests/ -m "not network and not slow"  # what CI runs on every push
uv run pytest tests/                                # everything (plugin must be built)
uv run pytest tests/ -k "not plugin_required"       # builder/schema tests only
uv run pytest tests/reference/ -v                   # reference tests
python scripts/test_multiple_python.py --all        # multi-Python (3.10-3.13)
```

The compiled plugin (`.so`/`.pyd`) must exist at `python/polars_cv/_lib.abi3.so`. Build with `maturin develop --release`.

### Markers and CI lanes

Two markers are declared in `pyproject.toml` (`[tool.pytest.ini_options]`), and
**CI never runs the bare `pytest tests/` you probably run locally**:

| Marker | Meaning | Where it runs |
|--------|---------|---------------|
| `network` | Needs network access | Never in CI |
| `slow` | Long-running | `slow-tests` job only — weekly schedule or manual dispatch, `-m "slow and not network"` (`.github/workflows/ci.yml`) |

The per-push lane is `-m "not network and not slow"`. Mark a new test `network`
if it fetches anything and `slow` if it is not sub-second — an unmarked slow test
lands in the lane that gates every merge.

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

### Differential tests against the rasterizer

`test_contour_raster_crosscheck.py` measures the same quantities two ways —
`area`, `centroid`, `iou`, `dice` and `contains_point` from polygon geometry,
versus pixel counts on a mask produced by the scanline filler. The two share no
code, so a fault in either surfaces as a disagreement even when each is
internally consistent. This is what caught the scanline filler painting a
surplus column at every right-hand edge, which every existing rasterize test had
missed.

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
plugin is not built (`-k "not plugin_required"`).

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
