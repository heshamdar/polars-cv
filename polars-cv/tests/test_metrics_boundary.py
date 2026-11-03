"""The boundary between polars-cv's core and its downstream metrics layer.

``polars_cv.metrics`` is built on polars-cv's public API the way scikit-learn
is built on NumPy, and its generic part (statistics, curves, bootstrap) on
Polars alone. See ``tests/_metrics_boundary.py`` for the rules and their
limits, and ``tests/test_metrics_boundary_fixtures.py`` for the cases the
scanner must reject and accept.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import polars_cv
from tests._discovery import package_modules
from tests._import_layering import read_modules
from tests._metrics_boundary import DOWNSTREAM, violations

pytestmark = pytest.mark.structural

PACKAGE_DIR = Path(polars_cv.__file__).parent
MODULES = read_modules(package_modules(), PACKAGE_DIR)


def test_metrics_boundary_holds() -> None:
    found = violations(MODULES)
    assert not found, "metrics boundary violated:\n" + "\n".join(found)


def test_importing_polars_cv_does_not_load_metrics() -> None:
    """The run-time form of rule (a), at the entry point users import."""
    code = f"import sys, polars_cv; print({DOWNSTREAM!r} in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", "import polars_cv loaded the metrics"
