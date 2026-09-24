"""Tests read planner state only through ``tests/_plan_view.py``.

The typed-op migration moves the planner's state out of ``Pipeline``'s Python
fields into a Rust ``Plan`` (``TYPED_OPS_PLAN.md``, P7). Every test that needs
the planned domain, dtype, rank, shape hints, op list or source reads it through
the seam, so that move rewrites one helper instead of thirty test files. This
guard keeps new tests from reaching past it.

Limits: a textual scan (CLAUDE.md orders these last). It matches the private
field names as attribute accesses, so it cannot see a ``getattr`` with a
computed name; the fixtures below pin what it does and does not flag.
"""

from __future__ import annotations

import re

import pytest

from tests._discovery import discovered, suite_files

pytestmark = pytest.mark.structural

#: The planner's private fields. Reading any of them outside the seam couples
#: a test to the Python planner that P7 deletes.
_PRIVATE = re.compile(
    r"\._(shape_hints|ops|source|output_dtype|expected_ndim|current_domain|"
    r"entering|state_at|assertions|asserted_dims|shape_declared)\b"
)

#: Files allowed to read those fields, and why.
EXEMPT: dict[str, str] = {
    "_plan_view.py": "the seam itself",
    "test_append_contract.py": (
        "tests the Python planner's own mechanism; deleted with it in P7"
    ),
    "test_sanitation.py": (
        "its planner-mechanism guards are deleted in P7; its scanners also "
        "name these fields as patterns"
    ),
    "test_removed_surfaces.py": "tombstones name removed internals",
    "test_plan_view_seam.py": "this guard names the fields it bans",
}


def offending_lines(text: str) -> list[int]:
    """1-based numbers of lines that read a private planner field."""
    return [
        n
        for n, line in enumerate(text.splitlines(), start=1)
        if _PRIVATE.search(line) and not line.lstrip().startswith("#")
    ]


class TestFixtures:
    @pytest.mark.parametrize(
        "line",
        [
            "assert pipe._shape_hints.height.value == 7",
            "n = len(p._ops)",
            "fmt = pipe._source.format",
            "assert lazy._pipeline._output_dtype == 'f32'",
            "x = p._entering[0].state.ndim",
        ],
    )
    def test_a_private_read_is_flagged(self, line: str) -> None:
        assert offending_lines(line) == [1]

    @pytest.mark.parametrize(
        "line",
        [
            "assert planned(pipe).height == 7",
            "n = len(ops_of(p))",
            "self._source_bytes = b''",
            "# pipe._ops is private",
            "value = pipe.output_dtype()",
        ],
    )
    def test_a_seam_read_is_not_flagged(self, line: str) -> None:
        assert offending_lines(line) == []


def test_every_exemption_names_a_real_file() -> None:
    names = {p.name for p in suite_files()}
    assert not set(EXEMPT) - names, f"stale exemptions: {set(EXEMPT) - names}"


def test_tests_read_planner_state_only_through_the_seam() -> None:
    scanned = discovered(
        [p for p in suite_files() if p.name not in EXEMPT], "non-exempt test files"
    )
    offenders = {
        str(p.name): lines for p in scanned if (lines := offending_lines(p.read_text()))
    }
    assert not offenders, (
        "read planner state through tests/_plan_view.py (planned, ops_of, "
        f"op_names, source_of, op_json): {offenders}"
    )
