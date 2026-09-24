"""The package metadata describes what polars-cv actually needs (CR-44).

The declared floors are tested by the ``dependency-floors`` CI job, which
installs ``scripts/dependency_floors.py``'s output; the tests here pin that
script's parsing (with fixtures) and the metadata facts no job exercises.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from .conftest import plugin_required

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: `tomllib` is 3.11+ stdlib; `tomli` is its exact predecessor.
    import tomli as tomllib

pytestmark = pytest.mark.structural

_PKG = Path(__file__).resolve().parents[1]
_PYPROJECT = tomllib.loads((_PKG / "pyproject.toml").read_text())
_VIZ = ("networkx", "graphviz", "pydot")


def _floors_module():
    spec = importlib.util.spec_from_file_location(
        "dependency_floors", _PKG / "scripts" / "dependency_floors.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _project(deps: list[str]) -> str:
    return "[project]\ndependencies = [" + ", ".join(f'"{d}"' for d in deps) + "]\n"


class TestDependencyFloors:
    def test_every_declared_dependency_has_a_testable_floor(self) -> None:
        pinned = _floors_module().floors((_PKG / "pyproject.toml").read_text())
        names = [p.split("==")[0] for p in pinned]
        declared = [
            re.match(r"[A-Za-z0-9._-]+", d)[0]  # ty: ignore[not-subscriptable]
            for d in _PYPROJECT["project"]["dependencies"]
        ]
        assert names == declared

    @pytest.mark.parametrize(
        ("dep", "expected"),
        [
            ("numpy>=2.0.2", "numpy==2.0.2"),
            ("polars>=1.41.1,<2.0", "polars==1.41.1"),
            ("polars<2.0,>=1.41.1", "polars==1.41.1"),
            ("pkg[extra]>=3.1; python_version < '3.11'", "pkg==3.1"),
        ],
    )
    def test_the_floor_is_the_lower_bound(self, dep: str, expected: str) -> None:
        assert _floors_module().floors(_project([dep])) == [expected]

    @pytest.mark.parametrize("dep", ["numpy", "polars<2.0", "numpy==2.0"])
    def test_a_dependency_without_a_floor_is_rejected(self, dep: str) -> None:
        with pytest.raises(ValueError, match="no '>=' floor"):
            _floors_module().floors(_project([dep]))


def test_visualisation_libraries_are_an_extra_not_a_requirement() -> None:
    """They serve only the lazily imported ``show_graph``."""
    core = " ".join(_PYPROJECT["project"]["dependencies"])
    assert not [lib for lib in _VIZ if lib in core]
    viz = " ".join(_PYPROJECT["project"]["optional-dependencies"]["viz"])
    assert all(lib in viz for lib in _VIZ)


def test_the_abi3_floor_matches_requires_python() -> None:
    """The wheel tag and ``requires-python`` state one minimum Python."""
    cargo = (_PKG / "Cargo.toml").read_text()
    abi3 = re.findall(r"abi3-py3(\d+)", cargo)
    requires = re.fullmatch(r">=3\.(\d+)", _PYPROJECT["project"]["requires-python"])
    assert abi3 and requires, "abi3 feature or requires-python not found"
    assert set(abi3) == {requires[1]}, (
        f"Cargo.toml builds abi3-py3{abi3} wheels but requires-python is "
        f"{_PYPROJECT['project']['requires-python']!r}"
    )


@plugin_required
def test_the_package_works_without_the_viz_extra() -> None:
    """Import, execute, and get an actionable error from ``show_graph``."""
    blocked = ", ".join(f"{lib!r}: None" for lib in _VIZ)
    script = textwrap.dedent(
        f"""
        import sys
        sys.modules.update({{{blocked}}})
        import polars as pl
        from polars_cv import Pipeline
        pipe = Pipeline().source("array").scale(2.0)
        df = pl.DataFrame({{"a": [[1.0, 2.0]]}}).cast({{"a": pl.Array(pl.Float32, 2)}})
        assert df.select(pl.col("a").cv.pipe(pipe).sink("list"))["a"][0].to_list() == [2.0, 4.0]
        try:
            pipe.to_graph(pl.col("a")).show_graph()
        except ImportError as e:
            print(e)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert "pip install 'polars-cv[viz]'" in out.stdout, out.stdout + out.stderr
