"""Every op must reproduce its recorded plan, schema and output.

See ``tests/_golden_cases.py`` for what is recorded and why, and
``scripts/gen_golden_corpus.py`` to re-record deliberately. A failure here
means an operation's behaviour changed: either fix the regression, or — if the
change is intended — re-record and justify the fixture diff in the commit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests._golden_cases import GoldenCase, golden_cases
from tests.conftest import plugin_required

FIXTURE = Path(__file__).resolve().parent / "golden" / "op_corpus.json"

_CASES = golden_cases()
_BY_ID = {c.case_id: c for c in _CASES}


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def _close(a: Any, b: Any, *, integer: bool) -> bool:
    if isinstance(a, str) or isinstance(b, str):  # "nan" / "inf" spellings
        return a == b
    if integer:
        return abs(a - b) <= 1.0 + 1e-3 * abs(b)
    return abs(a - b) <= 1e-6 + 1e-4 * abs(b)


def _row_mismatch(got: dict, want: dict, same_platform: bool) -> str | None:
    for key in ("structure", "n", "array"):
        if got.get(key) != want.get(key):
            return f"{key}: {got.get(key)!r} != {want.get(key)!r}"
    if same_platform:
        if got["sha256"] != want["sha256"]:
            return "exact digest differs"
        return None
    integer = "array" in want and want["array"]["dtype"][0] in "iub"
    for name in ("stats", "samples"):
        g, w = got[name], want[name]
        if len(g) != len(w) or not all(
            _close(x, y, integer=integer) for x, y in zip(g, w)
        ):
            return f"{name} beyond tolerance: {g} vs {w}"
    return None


def _mismatch(
    case: GoldenCase, got: dict, want: dict, same_platform: bool
) -> str | None:
    if "error" in want:
        if got.get("error") != want["error"]:
            return f"error class {got.get('error')!r} != {want['error']!r}"
        if case.expect and case.expect not in got.get("message", ""):
            return f"message {got.get('message')!r} lacks {case.expect!r}"
        return None
    for key in ("error", "plan", "schema"):
        if got.get(key) != want.get(key):
            return f"{key}: {got.get(key)!r} != {want.get(key)!r}"
    if len(got["rows"]) != len(want["rows"]):
        return f"row count {len(got['rows'])} != {len(want['rows'])}"
    for i, (g, w) in enumerate(zip(got["rows"], want["rows"])):
        why = _row_mismatch(g, w, same_platform)
        if why:
            return f"row {i}: {why}"
    return None


def test_the_fixture_covers_exactly_the_cases() -> None:
    """Both directions: no unrecorded case, no orphaned record."""
    recorded = set(_fixture()["cases"])
    defined = set(_BY_ID)
    assert not (defined - recorded), f"unrecorded: {sorted(defined - recorded)}"
    assert not (recorded - defined), f"orphaned: {sorted(recorded - defined)}"


def test_rejection_cases_are_recorded_as_errors() -> None:
    cases = _fixture()["cases"]
    for case in _CASES:
        if case.case_id.startswith("reject/"):
            assert cases[case.case_id].get("error"), (
                f"{case.case_id} is recorded as succeeding"
            )


@plugin_required
@pytest.mark.parametrize("case_id", sorted(_BY_ID))
def test_the_case_reproduces_its_recording(case_id: str) -> None:
    fixture = _fixture()
    case = _BY_ID[case_id]
    why = _mismatch(
        case,
        case.run(),
        fixture["cases"][case_id],
        same_platform=fixture["platform"] == sys.platform,
    )
    assert why is None, f"{case_id}: {why}"
