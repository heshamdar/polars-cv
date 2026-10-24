"""The package's import layering: no cycle, and nothing imports the root.

``__init__`` re-exports; implementations live in named modules. A module that
does ``from polars_cv import X`` reaches back into the package root, which
imports it, which is how an 11-module static cycle (``_namespace``,
``expressions``, ``geometry.*``, ``metrics.*``) formed. See
``tests/_import_layering.py`` for the scanner and its limits, and
``tests/test_import_layering_fixtures.py`` for the cases it must reject/accept.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import polars_cv
from tests._discovery import package_modules
from tests._import_layering import read_modules, violations

pytestmark = pytest.mark.structural

PACKAGE_DIR = Path(polars_cv.__file__).parent
MODULES = read_modules(package_modules(), PACKAGE_DIR)


def test_no_import_cycle_and_no_root_import() -> None:
    found = violations(MODULES)
    assert not found, "import layering violated:\n" + "\n".join(found)


@pytest.mark.parametrize("module", sorted(MODULES))
def test_each_module_imports_alone(module: str) -> None:
    """Importing any one submodule first, in a fresh interpreter, succeeds.

    The dynamic counterpart of the static scan: a cycle that only breaks for
    one entry point shows up here as an ImportError on that entry point.
    """
    name = module.removesuffix(".__init__")
    result = subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"import {name} failed:\n{result.stderr}"
