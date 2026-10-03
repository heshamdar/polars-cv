"""The package never runs Python per row or per batch inside a query.

A Python UDF (``map_batches``, ``map_elements``, ``map_groups``) puts the
interpreter, and the GIL, inside the plan, and unless it is marked elementwise
it is handed its whole input column at once. polars-cv's per-row work belongs
in the Rust plugin; the one UDF the package had, the instance-mask
single-region check, is now ``.contour.single``. Tests may still use UDFs.

An AST scan (a string search would trip on this docstring): it finds a call
to any attribute of those names, so it cannot tell polars' method from an
unrelated one of the same name. None exists in the package; rename one if it
ever does rather than widening this guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._discovery import package_modules

pytestmark = pytest.mark.structural

_UDF_METHODS = frozenset({"map_batches", "map_elements", "map_groups"})


def udf_calls(source: str) -> list[int]:
    """Line numbers of calls to a Python-UDF method in ``source``."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _UDF_METHODS
    ]


def test_the_package_has_no_python_udfs() -> None:
    offenders = [
        f"{path}:{line}"
        for path in package_modules()
        for line in udf_calls(Path(path).read_text())
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("pl.col('x').map_batches(f)", [1]),
        ("e = pl.col('x')\ne.map_elements(f, return_dtype=pl.Int8)", [2]),
        ("lf.group_by('g').map_groups(f, schema=None)", [1]),
        ("'map_batches'  # named, not called", []),
        ("pl.col('x').map_dict({})", []),
    ],
)
def test_the_scan_sees_calls_and_only_calls(source: str, found: list[int]) -> None:
    assert udf_calls(source) == found
