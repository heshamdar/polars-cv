"""Executable specifications for defects that are verified but not yet fixed.

Every test in this module is ``xfail(strict=True)``. Each one:

* describes a defect that has been **confirmed against running code or source**,
  not a suspicion;
* asserts the behaviour the codebase *should* have, so it fails today; and
* names, in its docstring, what "fixed" looks like.

``strict=True`` is the point. When someone lands the fix, the test XPASSes and
the suite goes **red** — which is the signal to delete the marker and let the
test join the suite proper. A backlog that lives in prose is a backlog that
rots; this one cannot silently become stale in either direction. It is the same
reasoning as `AGENTS.md`'s "The Single-Authority Refactor: What Was Done, What
Is Left" section, made executable.

These are the items recorded as deferred in that review: the ones where a fix is
a design change rather than a correction, and so wants its own commit. Adding a
test here is not a way to avoid fixing something — it is a way to stop the
knowledge evaporating between sessions.

Do **not** put a flaky or environment-dependent test here. `xfail` marks "known
broken", never "sometimes fails".
"""

from __future__ import annotations

import polars as pl
import pytest

from .conftest import plugin_required

#: Source-scanning members of this module check codebase shape, so they belong
#: in the lane pre-commit runs. The runtime ones carry `plugin_required`.
pytestmark = pytest.mark.structural


def _gap(reason: str) -> pytest.MarkDecorator:
    """Mark a verified, unfixed defect. Strict, so a fix fails the suite."""
    return pytest.mark.xfail(strict=True, reason=reason)


# ---------------------------------------------------------------------------
# Two surfaces for one operation, with drifted semantics
# ---------------------------------------------------------------------------


@plugin_required
@_gap(
    "the two surfaces still default differently: scale_contour defaults to "
    "centroid (its historic behaviour), .contour.scale to origin"
)
def test_the_two_contour_scale_surfaces_agree_by_default() -> None:
    """`Pipeline.scale_contour` and `.contour.scale` should default alike.

    Both now *expose* `origin=`, so a caller can always be explicit — that half
    is fixed and pinned by `test_affine_builder`-style tests in
    `test_contour_plugin.py`. What remains is the default: `scale_contour`
    keeps `"centroid"` (its behaviour since it shipped) while `.contour.scale`
    keeps `"origin"` (its own). A square at (2,2)-(4,4) scaled by 2 still lands
    at (1,1)-(5,5) one way and (4,4)-(8,8) the other.

    Aligning them moves output for whichever surface loses, so it is a
    deliberate API decision rather than a correction — recorded here until that
    decision is taken.
    """
    from polars_cv.geometry import CONTOUR_SCHEMA

    square = {
        "exterior": [
            {"x": 2.0, "y": 2.0},
            {"x": 4.0, "y": 2.0},
            {"x": 4.0, "y": 4.0},
            {"x": 2.0, "y": 4.0},
        ],
        "holes": [],
    }
    df = pl.DataFrame({"c": [square]}, schema={"c": CONTOUR_SCHEMA})

    namespace = df.select(r=pl.col("c").contour.scale(2.0, 2.0))["r"].to_list()[0]
    namespace_pts = sorted((p["x"], p["y"]) for p in namespace["exterior"])
    centroid = df.select(r=pl.col("c").contour.scale(2.0, 2.0, origin="centroid"))[
        "r"
    ].to_list()[0]
    centroid_pts = sorted((p["x"], p["y"]) for p in centroid["exterior"])

    assert namespace_pts == centroid_pts, (
        f"the namespace default ({namespace_pts}) differs from the graph "
        f"path's default ({centroid_pts}); the same op name means two things"
    )
