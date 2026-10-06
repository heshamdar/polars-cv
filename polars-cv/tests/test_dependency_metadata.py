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
import tomllib
from pathlib import Path

import pytest

from .conftest import plugin_required

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


def _ci_jobs() -> dict[str, str]:
    """``ci.yml``'s jobs, each key mapped to its block's text."""
    ci = (_PKG.parent / ".github" / "workflows" / "ci.yml").read_text()
    body = ci[ci.index("\njobs:\n") :]
    return dict(
        re.findall(
            r"^  ([a-z][a-z-]*):\n(.*?)(?=^  [a-z][a-z-]*:\n|\Z)", body, re.M | re.S
        )
    )


def test_every_python_floor_matches_requires_python() -> None:
    """The tooling that restates the minimum Python agrees with ``requires-python``.

    The test matrix's lowest version, the dependency-floors job, ty's and
    ruff's target versions and the multi-Python script each name the floor.
    One left behind tests, type-checks or lints a version the package does
    not support, or never tests the one it does. Read by text, so a renamed
    job or key fails the lookup rather than matching nothing.
    """
    requires = re.fullmatch(r">=3\.(\d+)", _PYPROJECT["project"]["requires-python"])
    assert requires, "requires-python is not a plain >=3.N floor"
    jobs = _ci_jobs()
    matrix = re.search(r"python-version: \[([^\]]*)\]", jobs["test"])
    floors_job = re.findall(r'python-version: "3\.(\d+)"', jobs["dependency-floors"])
    assert matrix and len(floors_job) == 1, "ci.yml's Python versions not found"
    multi = (_PKG / "scripts" / "test_multiple_python.py").read_text()
    script_versions = re.search(r"DEFAULT_VERSIONS = \[(.*?)\]", multi, re.S)
    assert script_versions, "test_multiple_python.py's DEFAULT_VERSIONS not found"
    tool = _PYPROJECT["tool"]
    found = {
        "ci.yml test matrix (lowest)": min(
            re.findall(r'"3\.(\d+)"', matrix[1]), key=int
        ),
        "ci.yml dependency-floors job": floors_job[0],
        "[tool.ty.environment] python-version": re.fullmatch(
            r"3\.(\d+)", tool["ty"]["environment"]["python-version"]
        )[1],  # type: ignore[index]
        "[tool.ruff] target-version": re.fullmatch(
            r"py3(\d+)", tool["ruff"]["target-version"]
        )[1],  # type: ignore[index]
        "test_multiple_python.py DEFAULT_VERSIONS (lowest)": min(
            re.findall(r'"3\.(\d+)"', script_versions[1]), key=int
        ),
        **{
            f"test_multiple_python.py abi3-py3{v}": v
            for v in re.findall(r"abi3-py3(\d+)", multi)
        },
    }
    wrong = {where: f"3.{v}" for where, v in found.items() if v != requires[1]}
    assert not wrong, f"requires-python is >=3.{requires[1]}, but {wrong}"


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
