"""Tests read planner state only through ``tests/_plan_view.py``.

A pipeline's source, ops and planned states live in its Rust ``Plan``, and its
expression table in ``_exprs``. Every test that needs the planned domain, dtype,
rank, sizes, op list, source or expressions reads it through the seam, so a
change to how a pipeline holds them rewrites one helper instead of thirty test
files. This guard keeps new tests from reaching past it.

Limits: a textual scan (CLAUDE.md orders these last). It matches the private
field names as attribute accesses, so it cannot see a ``getattr`` with a
computed name; the fixtures below pin what it does and does not flag.
"""

from __future__ import annotations

import re

import pytest

from tests._discovery import discovered, suite_files

pytestmark = pytest.mark.structural

#: A pipeline's private planner fields. Reading any of them outside the seam
#: couples a test to how the pipeline holds its plan.
_PRIVATE = re.compile(r"\._(plan|state|exprs)\b")

#: Files allowed to read those fields, and why.
EXEMPT: dict[str, str] = {
    "_plan_view.py": "the seam itself",
    "test_append_contract.py": "tests the plan object and the copy itself",
    "test_sanitation.py": (
        "plans single ops against fabricated states (``Plan.continuing``)"
    ),
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
            "assert pipe._state.dims[0] == 7",
            "n = len(p._plan)",
            "ops = pipe._plan.ops_json()",
            "assert lazy._pipeline._state.dtype == 'f32'",
            "e = p._exprs[0]",
        ],
    )
    def test_a_private_read_is_flagged(self, line: str) -> None:
        assert offending_lines(line) == [1]

    @pytest.mark.parametrize(
        "line",
        [
            "assert planned(pipe).height == 7",
            "n = len(ops_of(p))",
            "self._plan_bytes = b''",
            "# pipe._plan is private",
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
