"""CI must refuse to run without the compiled plugin, rather than skip its tests.

``plugin_required`` is a ``skipif``: when the extension is not built, every test
carrying it drops out and the run stays green. Under CI that is a false pass, so
``conftest.py`` ends the session instead. The decision is
:func:`tests._plugin_gate.ci_requires_plugin`; these tests pin it. The wiring
into ``conftest.py`` (``pytest_sessionstart``) has no unit test of its own: it
is verified by a real run with ``CI=true`` and no built extension.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests._plugin_gate import ci_requires_plugin, is_ci
from tests.conftest import plugin_required


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", " TRUE "])
def test_truthy_ci_values_count_as_ci(value: str) -> None:
    assert is_ci({"CI": value})


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_falsy_ci_values_do_not_count_as_ci(value: str) -> None:
    assert not is_ci({"CI": value})


def test_unset_ci_is_not_ci() -> None:
    assert not is_ci({})


def test_ci_without_plugin_must_fail() -> None:
    assert ci_requires_plugin({"CI": "true"}, plugin_available=False)


def test_ci_with_plugin_runs_normally() -> None:
    assert not ci_requires_plugin({"CI": "true"}, plugin_available=True)


def test_local_run_without_plugin_still_skips() -> None:
    assert not ci_requires_plugin({}, plugin_available=False)


def test_subprocess_env_puts_the_source_tree_first() -> None:
    from tests._plugin_gate import PYTHON_SRC, subprocess_env

    env = subprocess_env({"PYTHONPATH": "/elsewhere"})
    assert env["PYTHONPATH"].split(os.pathsep) == [str(PYTHON_SRC), "/elsewhere"]


def test_subprocess_env_works_without_a_prior_pythonpath() -> None:
    from tests._plugin_gate import PYTHON_SRC, subprocess_env

    assert subprocess_env({}).get("PYTHONPATH") == str(PYTHON_SRC)


def test_subprocess_env_keeps_the_rest_of_the_environment() -> None:
    from tests._plugin_gate import subprocess_env

    assert subprocess_env({"HOME_MARKER": "x"})["HOME_MARKER"] == "x"


@plugin_required
def test_a_fresh_interpreter_imports_the_build_from_source() -> None:
    """The subprocess tests depend on this, not on the editable install."""
    from tests._plugin_gate import subprocess_env

    result = subprocess.run(
        [sys.executable, "-c", "import polars_cv._lib"],
        env=subprocess_env(dict(os.environ)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "a fresh interpreter cannot import polars_cv._lib from python/. Run "
        "`maturin develop` from polars-cv/ to rebuild the extension.\n" + result.stderr
    )
