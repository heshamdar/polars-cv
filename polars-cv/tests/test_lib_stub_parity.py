"""``_lib.pyi`` is held to the built ``polars_cv._lib``, in both directions.

The stub is hand-written and ``ty`` reads only the stub, so before this a
``#[pyfunction]`` whose signature changed in Rust left the stub wrong with
nothing failing. The comparison is ``tests/_stub_parity.py`` (fixtures:
``test_stub_parity_fixtures.py``); this file runs it on the real extension.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import polars_cv
from tests._stub_parity import stub_problems
from tests.conftest import plugin_required

pytestmark = pytest.mark.structural

#: Names the extension exports that the stub deliberately omits. The stub's scope
#: is what the Python planner imports by name; each of these is reached only
#: from a test or by pickle, so a signature for it would be a second copy that
#: nothing in the package type-checks against. Each needs a reason, and an entry
#: the stub now declares (or the module lost) fails the test.
UNSTUBBED: dict[str, str] = {
    "_plan_from_json": (
        "the unpickle target `Plan.__reduce__` names; Python never calls it"
    ),
    "_plan_state_from_json": (
        "the unpickle target `PlanState.__reduce__` names; Python never calls it"
    ),
    "point_schema": (
        "read only by test_sanitation's schema parity test, which holds "
        "POINT_SCHEMA to the Rust field list"
    ),
    "contour_schema": (
        "read only by test_sanitation's schema parity test, which holds "
        "CONTOUR_SCHEMA to the Rust field list"
    ),
    "bbox_schema": (
        "read only by test_sanitation's schema parity test, which holds "
        "BBOX_SCHEMA to the Rust field list"
    ),
    "extension_types": (
        "read only by test_extension_types, which holds EXTENSION_TYPES to "
        "the Rust declaration"
    ),
}


def _stub_path() -> Path:
    return Path(polars_cv.__file__).parent / "_lib.pyi"


@plugin_required
def test_the_lib_stub_matches_the_built_extension() -> None:
    """Every stub entry exists in ``_lib`` with the same signature, and vice versa."""
    from polars_cv import _lib

    stub = _stub_path()
    assert stub.is_file(), f"{stub} is missing"
    problems = stub_problems(stub.read_text(), _lib, unstubbed=UNSTUBBED)
    assert not problems, "_lib.pyi disagrees with the built extension:\n  " + (
        "\n  ".join(problems)
    )
