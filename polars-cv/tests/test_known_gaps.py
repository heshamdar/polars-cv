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

import re

import polars as pl
import pytest

from ._discovery import (
    requires_checkout,
    rust_sources,
)
from .conftest import plugin_required

#: Source-scanning members of this module check codebase shape, so they belong
#: in the lane pre-commit runs. The runtime ones carry `plugin_required`.
pytestmark = pytest.mark.structural


def _gap(reason: str) -> pytest.MarkDecorator:
    """Mark a verified, unfixed defect. Strict, so a fix fails the suite."""
    return pytest.mark.xfail(strict=True, reason=reason)


# ---------------------------------------------------------------------------
# Wire format: `deny_unknown_fields` has no mechanism behind it
# ---------------------------------------------------------------------------


@requires_checkout
@_gap(
    "the geometry/read_bytes kwargs structs are still open; only the graph "
    "wire-format structs were closed"
)
def test_every_plugin_kwargs_struct_rejects_unknown_fields() -> None:
    """Every `#[polars_expr]` kwargs struct should refuse a field it does not declare.

    `GraphNode`, `UnifiedGraph`, `OutputSpec`, `SourceSpec`, `SinkSpec` and
    `GraphKwargs` carry `#[serde(deny_unknown_fields)]`. `ContourKwargs`,
    `PointKwargs` and `ReadBytesKwargs` do not, so a kwarg Python emits and
    Rust no longer reads is silently dropped — which is exactly how
    `sink("jpeg", qualtiy=50)` encoded at the default quality with nothing
    said, and how `match_detections(strategy=)` survived unread for releases.

    Fixed by: adding the attribute to the three structs, and replacing this
    scan with a mechanism a new struct cannot step around — the scan itself is
    the weakest of the three guard kinds `CLAUDE.md` ranks, and is here only
    because nothing better exists yet.
    """
    open_structs: list[str] = []
    for path in rust_sources():
        if "polars-cv/src" not in path.as_posix():
            continue
        text = path.read_text()
        for match in re.finditer(
            r"#\[derive\([^)]*Deserialize[^)]*\)\]\s*((?:#\[[^\]]*\]\s*)*)"
            r"pub struct (\w+)",
            text,
        ):
            attrs, name = match.group(1), match.group(2)
            if "deny_unknown_fields" in attrs:
                continue
            # `OpSpec` is the one documented, permanent exception: its params
            # ride on `#[serde(flatten)]`, which serde documents as
            # incompatible with `deny_unknown_fields`.
            if name == "OpSpec":
                continue
            open_structs.append(f"{path.name}::{name}")

    assert not open_structs, (
        f"these deserialized structs accept undeclared fields: {sorted(open_structs)}"
    )


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


@requires_checkout
@_gap(
    "the {x, y} point struct is declared in contour.rs, point.rs and "
    "geometry/schemas.py with nothing relating them"
)
def test_the_point_struct_schema_has_one_declaration() -> None:
    """One wire schema, one declaration.

    `contour.rs::point_struct_dtype`, `point.rs::point_output_type` and
    `geometry/schemas.py::POINT_SCHEMA` each spell out `{x: f64, y: f64}`.
    Dtypes, op names and enum variants all have a named authority and a
    bidirectional parity guard; this surface has neither.

    Fixed by: one Rust declaration both plugin modules read, surfaced to Python
    the way `enum_variants` surfaces the naming registry.
    """
    # An `{x, y}` point is an adjacent x/y Float64 pair that is the *whole*
    # field list. A bbox starts `x, y, width, height`, so the third field is
    # what tells the two apart -- matching on `"x"` alone reports bboxes too,
    # and reports nothing useful.
    pair = re.compile(
        r'from_static\("x"\),\s*DataType::Float64\s*\),\s*'
        r'Field::new\(\s*PlSmallStr::from_static\("y"\),\s*DataType::Float64\s*\),'
        r"(?P<after>\s*(?:\]|Field::new))",
        re.S,
    )
    declarations = []
    for path in rust_sources():
        if "polars-cv/src" not in path.as_posix():
            continue
        text = path.read_text()
        for match in pair.finditer(text):
            if "Field::new" in match.group("after"):
                continue  # a wider struct (bbox) that merely begins x, y
            line = text[: match.start()].count("\n") + 1
            declarations.append(f"{path.name}:{line}")

    assert declarations, (
        "no {x, y} point struct construction found in the plugin sources -- "
        "the probe is broken and would report 'one declaration' forever"
    )
    assert len(declarations) == 1, (
        f"the point struct is built in more than one place: {declarations}"
    )


# ---------------------------------------------------------------------------
# Engine-side contract overrides and dead surface
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Guards that are still hand-maintained lists
# ---------------------------------------------------------------------------
