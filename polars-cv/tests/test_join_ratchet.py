"""Keyed joins in the metrics treat a null key as a key (see ``_join_ratchet``)."""

from __future__ import annotations

import pytest

from tests._discovery import discovered, package_modules
from tests._join_ratchet import unmatched_joins

pytestmark = pytest.mark.structural


def test_every_metrics_join_matches_null_keys() -> None:
    files = discovered(
        [p for p in package_modules() if "metrics" in p.parts], "metrics modules"
    )
    offenders = [
        f"{p.name}:{line}" for p in files for line in unmatched_joins(p.read_text())
    ]
    assert offenders == [], (
        "keyed joins without nulls_equal=True (a null group would match "
        "nothing):\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "snippet",
    [
        'a.join(b, on="k", how="left")',
        "a.join(b, on=keys)",
        'a.join(b, left_on="x", right_on="y", how="inner")',
        'a.join(b, on="k", nulls_equal=False)',
        "a.join(b, on=keys, nulls_equal=flag)",
        'a.join(b, "k")',
        "a.join(b, **opts)",
        'f(a.join(b, on="k", nulls_equal=True).join(c, on="j"))',
    ],
)
def test_ratchet_reports(snippet: str) -> None:
    assert unmatched_joins(snippet) == [1]


@pytest.mark.parametrize(
    "snippet",
    [
        'a.join(b, on="k", how="left", nulls_equal=True)',
        'a.join(b, left_on=["x"], right_on=["y"], nulls_equal=True)',
        'a.join(b, how="cross")',
        '", ".join(parts)',
        '"\\n".join(lines)',
    ],
)
def test_ratchet_accepts(snippet: str) -> None:
    assert unmatched_joins(snippet) == []
