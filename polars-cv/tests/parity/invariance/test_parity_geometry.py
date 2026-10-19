"""The ``.bbox``, ``.point`` and measure-style ``.contour`` namespaces.

``test_parity_accessors.py`` holds the op-backed ``.contour`` accessors to the
pipeline op they are. This file covers what that leaves: the measures and
relations of the geometry columns themselves, over drawn shapes, against an
independent NumPy reference and against laws the answer must obey however the
shape is written down or the query is run.

*Oracle* properties compare with NumPy: the shoelace area and centroid of a
random simple polygon (a star-shaped one: vertices at sorted angles about a
centre, so it is simple by construction), a polygon with a hole, box
overlap, point distance and point-in-polygon.

*Laws* are relations between two answers of the engine, so no tolerance has to
absorb a reference's convention: IoU is symmetric, 1 on itself, in [0, 1] and
agrees with the area of ``intersection``; a translation moves the centroid
and changes no area or IoU; a scale multiplies area by ``sx * sy``; a ring
wound the other way, or started at another vertex, describes the same
region; a row's answer does not depend on its neighbours (nulls included);
the engines agree; an expression argument equals the literal.

Tolerances are two stated models, not tuned per property.

* ``EXACT`` (1e-9): results that differ only in the order floating-point terms
  are summed (a rotated ring, a shifted polygon) — ~7 orders above the error
  of a shoelace sum over <= 12 vertices at the magnitudes drawn here.
* ``OVERLAY`` (1e-7): any answer computed through ``i_overlay``'s boolean
  operations — the area/centroid of a region *with a hole* (``to_geo_region``),
  every IoU/Dice/intersection. That library works on snapped fixed-point
  coordinates, so its error is relative to the shape's extent, not to
  floating-point epsilon: measured over 1200 random holed stars (extent 1 to
  200) the worst relative area error against an exact rational computation was
  3.7e-9, while the same rings without a hole are exact to the last bit. 1e-7
  leaves a 25x margin over that measurement and is still 6 orders tighter than
  any real fault (a missing hole is a 9% error).

A result computed by the same code on the same numbers (a row alone, another
engine) must be *equal*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import polars as pl
from hypothesis import event
from hypothesis import strategies as st

import polars_cv  # noqa: F401  (registers the namespaces)
from polars_cv.geometry.schemas import BBOX_SCHEMA, CONTOUR_SCHEMA, POINT_SCHEMA
from tests.conftest import plugin_required
from tests.parity.framework.budget import property_lanes

pytestmark = plugin_required

EXACT = (1e-9, 1e-9)
OVERLAY = (1e-7, 1e-7)
RTOL, ATOL = EXACT

ENGINES = ("in-memory", "streaming")


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """A simple polygon with at most one hole, star-shaped about ``centre``.

    ``exterior`` is ``(n, 2)``. The hole, when present, is the exterior shrunk
    by 0.3 about the centre, so it lies strictly inside and cannot touch.
    """

    centre: tuple[float, float]
    exterior: np.ndarray
    holes: tuple[np.ndarray, ...] = ()

    def struct(
        self, shift: int = 0, reverse: bool = False, *, holes: bool = True
    ) -> dict:
        """The ``CONTOUR_SCHEMA`` value, optionally with every ring started at
        vertex *shift* and wound the other way when *reverse*."""
        rings = [ring_variant(self.exterior, shift, reverse)]
        hole_rings = [ring_variant(h, shift, reverse) for h in self.holes if holes]
        return {
            "exterior": to_points(rings[0]),
            "holes": [to_points(h) for h in hole_rings],
            "is_closed": True,
        }

    def transformed(self, scale: float, dx: float, dy: float) -> "Shape":
        """Scaled by *scale* about the centre, then moved by ``(dx, dy)``: still
        star-shaped about the moved centre, so still a valid ``Shape``."""
        cx, cy = self.centre

        def move(ring: np.ndarray) -> np.ndarray:
            return np.array([cx + dx, cy + dy]) + scale * (ring - np.array([cx, cy]))

        return Shape(
            (cx + dx, cy + dy),
            move(self.exterior),
            tuple(move(h) for h in self.holes),
        )

    def __repr__(self) -> str:
        return (
            f"Shape(n={len(self.exterior)}, holes={len(self.holes)}, "
            f"centre={self.centre}, exterior={self.exterior.tolist()})"
        )


def ring_variant(ring: np.ndarray, shift: int, reverse: bool) -> np.ndarray:
    out = np.roll(ring, -(shift % len(ring)), axis=0)
    return out[::-1] if reverse else out


def to_points(ring: np.ndarray) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for x, y in ring]


@st.composite
def shapes(draw: st.DrawFn, *, holes: bool = True) -> Shape:
    """A star-shaped polygon: vertex *i* sits in the angular sector
    ``[i, i + 1) * 2pi / n``, so angles strictly increase and no gap reaches
    pi for ``n >= 4`` — the polygon cannot self-intersect."""
    n = draw(st.integers(4, 12), label="vertices")
    jitter = draw(
        st.lists(st.floats(0.05, 0.95), min_size=n, max_size=n), label="angles"
    )
    radii = draw(st.lists(st.floats(1.0, 20.0), min_size=n, max_size=n), label="radii")
    cx = draw(st.floats(-20.0, 20.0), label="cx")
    cy = draw(st.floats(-20.0, 20.0), label="cy")
    theta = 2.0 * math.pi * (np.arange(n) + np.array(jitter)) / n
    r = np.array(radii)
    exterior = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)
    hole = (
        (np.array([cx, cy]) + 0.3 * (exterior - np.array([cx, cy])),)
        if holes and draw(st.integers(0, 2), label="has hole") == 0
        else ()
    )
    return Shape((cx, cy), exterior, hole)


@st.composite
def shape_pairs(draw: st.DrawFn, *, holes: bool = True) -> tuple[Shape, Shape]:
    """Two shapes that usually overlap: independent, or the second a scaled
    and shifted copy of the first."""
    a = draw(shapes(holes=holes), label="a")
    if draw(st.booleans(), label="derived"):
        b = a.transformed(
            draw(st.floats(0.5, 1.5), label="factor"),
            draw(st.floats(-6.0, 6.0), label="dx"),
            draw(st.floats(-6.0, 6.0), label="dy"),
        )
    else:
        b = draw(shapes(holes=holes), label="b")
    return a, b


def nullable(strategy: st.SearchStrategy, *, max_rows: int = 4) -> st.SearchStrategy:
    """1..max_rows rows, each a value from *strategy* or null (null is as likely
    as any one draw of three)."""
    return st.lists(
        st.one_of(st.none(), strategy, strategy, strategy),
        min_size=1,
        max_size=max_rows,
    )


def contour_series(name: str, rows: list[Shape | None], **kw: Any) -> pl.Series:
    return pl.Series(
        name,
        [None if s is None else s.struct(**kw) for s in rows],
        dtype=CONTOUR_SCHEMA,
    )


# ---------------------------------------------------------------------------
# NumPy references
# ---------------------------------------------------------------------------


def shoelace(ring: np.ndarray) -> float:
    """Signed area, positive for a ring that is counter-clockwise in a y-up
    frame (the usual shoelace sum)."""
    x, y = ring.T
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0


def ring_centroid(ring: np.ndarray) -> np.ndarray:
    x, y = ring.T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    return np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / (6.0 * area)


def perimeter(ring: np.ndarray) -> float:
    return float(np.hypot(*(np.roll(ring, -1, axis=0) - ring).T).sum())


def region_area(shape: Shape) -> float:
    return abs(shoelace(shape.exterior)) - sum(abs(shoelace(h)) for h in shape.holes)


def region_centroid(shape: Shape) -> np.ndarray:
    """Centroid of the exterior minus its hole (first moments subtract)."""
    area = abs(shoelace(shape.exterior))
    moment = area * ring_centroid(shape.exterior)
    for hole in shape.holes:
        hole_area = abs(shoelace(hole))
        area -= hole_area
        moment = moment - hole_area * ring_centroid(hole)
    return moment / area


def inside_ring(ring: np.ndarray, px: float, py: float) -> bool:
    """Even-odd ray casting."""
    x, y = ring.T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    straddle = (y > py) != (y1 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_x = x + (py - y) * (x1 - x) / (y1 - y)
    return bool(np.count_nonzero(straddle & (px < cross_x)) % 2)


def inside_region(shape: Shape, px: float, py: float) -> bool:
    return inside_ring(shape.exterior, px, py) and not any(
        inside_ring(h, px, py) for h in shape.holes
    )


def distance_to_ring(ring: np.ndarray, px: float, py: float) -> float:
    a = ring
    b = np.roll(ring, -1, axis=0)
    ab = b - a
    ap = np.array([px, py]) - a
    denom = (ab**2).sum(axis=1)
    t = np.clip((ap * ab).sum(axis=1) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return float(np.hypot(*(np.array([px, py]) - closest).T).min())


def distance_to_region(shape: Shape, px: float, py: float) -> float:
    return min(distance_to_ring(r, px, py) for r in (shape.exterior, *shape.holes))


# ---------------------------------------------------------------------------
# Running and comparing
# ---------------------------------------------------------------------------


def run(frame: pl.DataFrame, exprs: dict[str, pl.Expr], engine: str = "in-memory"):
    """``select`` on a named engine, so no result silently comes from the
    suite's default (streaming) when the property means in-memory."""
    return frame.lazy().select(**exprs).collect(engine=engine)


def close(
    actual: Any,
    expected: Any,
    what: str,
    tol: tuple[float, float] | list[tuple[float, float]] = EXACT,
) -> None:
    """Rows of floats (``None`` for a null row) equal to the reference, within
    *tol* (``(rtol, atol)``, or one per row)."""
    actual, expected = list(actual), list(expected)
    assert len(actual) == len(expected), what
    tols = tol if isinstance(tol, list) else [tol] * len(actual)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert (a is None) == (e is None), f"{what}: row {i} null mismatch {a} vs {e}"
        if a is not None:
            np.testing.assert_allclose(
                a, e, rtol=tols[i][0], atol=tols[i][1], err_msg=f"{what}: row {i}"
            )


def tol_of(shape: Shape | None) -> tuple[float, float]:
    """``OVERLAY`` for a region with a hole, ``EXACT`` otherwise."""
    return OVERLAY if shape is not None and shape.holes else EXACT


def fields(values: list[dict | None], names: tuple[str, ...]) -> list[Any]:
    return [None if v is None else [v[n] for n in names] for v in values]


def xy(values: list[dict | None]) -> list[Any]:
    return fields(values, ("x", "y"))


MEASURES: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    "area": lambda c: c.contour.area(),
    "perimeter": lambda c: c.contour.perimeter(),
    "centroid": lambda c: c.contour.centroid(),
    "bbox": lambda c: c.contour.bounding_box(),
}


# ---------------------------------------------------------------------------
# Contour oracle
# ---------------------------------------------------------------------------


@property_lanes(weight=2, rows=nullable(shapes()))
def test_contour_area_centroid_bbox_match_numpy(rows: list[Shape | None]) -> None:
    """Area, centroid and bounding box of a star polygon (and of a holed one)
    are the shoelace sums, in NumPy, and the perimeter is the length of every
    ring; null rows stay null."""
    frame = pl.DataFrame([contour_series("c", rows)])
    out = run(frame, {k: f(pl.col("c")) for k, f in MEASURES.items()})
    event(f"holes={sum(bool(s and s.holes) for s in rows)}")
    close(
        out["area"],
        [None if s is None else region_area(s) for s in rows],
        "area",
        [tol_of(s) for s in rows],
    )
    close(
        xy(out["centroid"].to_list()),
        [None if s is None else region_centroid(s) for s in rows],
        "centroid",
        [tol_of(s) for s in rows],
    )
    close(
        out["perimeter"],
        [
            None if s is None else sum(perimeter(r) for r in (s.exterior, *s.holes))
            for s in rows
        ],
        "perimeter (every ring)",
    )
    close(
        fields(out["bbox"].to_list(), ("x", "y", "width", "height")),
        [
            None
            if s is None
            else [
                *s.exterior.min(axis=0),
                *(s.exterior.max(axis=0) - s.exterior.min(axis=0)),
            ]
            for s in rows
        ],
        "bounding_box",
    )


@property_lanes(weight=2, rows=nullable(shapes(holes=False)))
def test_contour_perimeter_and_signed_area_match_numpy(
    rows: list[Shape | None],
) -> None:
    """Perimeter is the sum of edge lengths and ``signed=True`` is the shoelace
    sum itself (positive for the ring's own winding), on a hole-free ring."""
    frame = pl.DataFrame([contour_series("c", rows)])
    out = run(
        frame,
        {
            "perimeter": pl.col("c").contour.perimeter(),
            "signed": pl.col("c").contour.area(signed=True),
            "unsigned": pl.col("c").contour.area(),
        },
    )
    close(
        out["perimeter"],
        [None if s is None else perimeter(s.exterior) for s in rows],
        "perimeter",
    )
    close(
        out["signed"],
        [None if s is None else shoelace(s.exterior) for s in rows],
        "signed area",
    )
    close(
        out["unsigned"],
        [None if s is None else abs(shoelace(s.exterior)) for s in rows],
        "unsigned area",
    )


@st.composite
def points_and_shapes(draw: st.DrawFn) -> tuple[list[Shape | None], list[Any]]:
    rows = draw(nullable(shapes(), max_rows=3), label="shapes")
    pts = [
        draw(
            st.one_of(
                st.none(),
                st.tuples(st.floats(-45.0, 45.0), st.floats(-45.0, 45.0)),
            ),
            label="point",
        )
        for _ in rows
    ]
    return rows, pts


@property_lanes(weight=2, case=points_and_shapes())
def test_contains_point_and_distances_match_numpy(case) -> None:
    """Point-in-region is even-odd ray casting over the exterior minus the
    holes; the unsigned distance is the nearest edge of any ring; the signed
    distance has that magnitude and is negative exactly inside the region."""
    rows, pts = case
    frame = pl.DataFrame(
        [
            contour_series("c", rows),
            pl.Series(
                "p",
                [None if p is None else {"x": p[0], "y": p[1]} for p in pts],
                dtype=POINT_SCHEMA,
            ),
        ]
    )
    out = run(
        frame,
        {
            "inside": pl.col("c").contour.contains_point(pl.col("p")),
            "dist": pl.col("p").point.distance_to_contour(pl.col("c")),
            "signed": pl.col("p").point.signed_distance_to_contour(pl.col("c")),
        },
    )
    live = [(s, p) for s, p in zip(rows, pts)]
    for i, (s, p) in enumerate(live):
        if s is None or p is None:
            assert out["inside"][i] is None, f"inside row {i}"
            assert out["dist"][i] is None, f"dist row {i}"
            assert out["signed"][i] is None, f"signed row {i}"
            continue
        d = distance_to_region(s, *p)
        if d < 1e-7:  # on an edge: which side wins is not a claim
            continue
        want_inside = inside_region(s, *p)
        assert out["inside"][i] == want_inside, f"contains_point row {i}"
        np.testing.assert_allclose(out["dist"][i], d, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(abs(out["signed"][i]), d, rtol=RTOL, atol=ATOL)
        assert (out["signed"][i] < 0) == want_inside, f"signed sign row {i}"


# ---------------------------------------------------------------------------
# Contour laws
# ---------------------------------------------------------------------------

PAIR_MEASURES = {
    "iou": lambda a, b: a.contour.iou(b),
    "dice": lambda a, b: a.contour.dice(b),
    "hausdorff": lambda a, b: a.contour.hausdorff_distance(b),
}


@property_lanes(
    weight=2,
    shift=st.integers(0, 11),
    reverse_a=st.booleans(),
    reverse_b=st.booleans(),
    pair=shape_pairs(),
    shift_b=st.integers(0, 11),
)
def test_a_ring_written_differently_is_the_same_region(
    pair, shift, shift_b, reverse_a, reverse_b
) -> None:
    """Starting a ring at another vertex, or winding it the other way, changes
    no measure: area, perimeter, centroid, bounding box, IoU, Dice, Hausdorff
    distance, and the containment of a fixed point."""
    a, b = pair
    probe = [{"x": float(a.centre[0]), "y": float(a.centre[1])}] * 2
    base = pl.DataFrame(
        {
            "a": pl.Series("a", [a.struct(), b.struct()], dtype=CONTOUR_SCHEMA),
            "b": pl.Series("b", [b.struct(), a.struct()], dtype=CONTOUR_SCHEMA),
            "p": pl.Series("p", probe, dtype=POINT_SCHEMA),
        }
    )
    variant = pl.DataFrame(
        {
            "a": pl.Series(
                "a",
                [a.struct(shift, reverse_a), b.struct(shift_b, reverse_b)],
                dtype=CONTOUR_SCHEMA,
            ),
            "b": pl.Series(
                "b",
                [b.struct(shift_b, reverse_b), a.struct(shift, reverse_a)],
                dtype=CONTOUR_SCHEMA,
            ),
            "p": base["p"],
        }
    )
    exprs = {
        **{k: f(pl.col("a")) for k, f in MEASURES.items()},
        **{k: f(pl.col("a"), pl.col("b")) for k, f in PAIR_MEASURES.items()},
        "inside": pl.col("a").contour.contains_point(pl.col("p")),
    }
    want, got = run(base, exprs), run(variant, exprs)
    rows_tol = [tol_of(a), tol_of(b)]
    for name in ("perimeter", "hausdorff"):
        close(got[name], want[name], name)
    close(got["area"], want["area"], "area", rows_tol)
    for name in ("iou", "dice"):
        close(got[name], want[name], name, OVERLAY)
    close(
        xy(got["centroid"].to_list()),
        xy(want["centroid"].to_list()),
        "centroid",
        rows_tol,
    )
    box = ("x", "y", "width", "height")
    close(
        fields(got["bbox"].to_list(), box),
        fields(want["bbox"].to_list(), box),
        "bounding_box",
    )
    # Containment of a probe is a boolean: a probe on an edge could tip, so it
    # is only compared when it is clear of every ring.
    for i, shape in enumerate((a, b)):
        if distance_to_region(shape, *a.centre) > 1e-7:
            assert got["inside"][i] == want["inside"][i], "contains_point"


@property_lanes(weight=2, pair=shape_pairs())
def test_iou_and_dice_laws(pair) -> None:
    """``iou(a, b) == iou(b, a)``, ``iou(a, a) == 1``, ``0 <= iou <= 1``,
    ``dice = 2 iou / (1 + iou)``, and IoU is the ratio of the area of
    ``intersection`` to the area of the union it implies."""
    a, b = pair
    frame = pl.DataFrame([contour_series("a", [a, b]), contour_series("b", [b, a])])
    out = run(
        frame,
        {
            "ab": pl.col("a").contour.iou(pl.col("b")),
            "ba": pl.col("b").contour.iou(pl.col("a")),
            "aa": pl.col("a").contour.iou(pl.col("a")),
            "dice": pl.col("a").contour.dice(pl.col("b")),
            "area_a": pl.col("a").contour.area(),
            "area_b": pl.col("b").contour.area(),
        },
    )
    inter = run(
        frame.select(pl.col("a"), pl.col("b")),
        {"s": pl.col("a").contour.intersection(pl.col("b"))},
    )["s"]
    for i in range(2):
        iou = out["ab"][i]
        assert 0.0 <= iou <= 1.0, f"iou {iou} outside [0, 1]"
        assert 0.0 <= out["dice"][i] <= 1.0, f"dice {out['dice'][i]} outside [0, 1]"
        np.testing.assert_allclose(
            out["ab"][i], out["ba"][i], rtol=OVERLAY[0], atol=OVERLAY[1]
        )
        np.testing.assert_allclose(out["aa"][i], 1.0, rtol=OVERLAY[0], atol=OVERLAY[1])
        np.testing.assert_allclose(
            out["dice"][i], 2 * iou / (1 + iou), rtol=OVERLAY[0], atol=OVERLAY[1]
        )
    # intersection area from the engine's own ``area`` over its result
    areas = run(
        pl.DataFrame({"i": inter}),
        {"a": pl.col("i").contour.area()},
    )["a"]
    for i in range(2):
        inter_area = sum(areas.to_list()[i] or [])
        union = out["area_a"][i] + out["area_b"][i] - inter_area
        np.testing.assert_allclose(
            out["ab"][i], inter_area / union, rtol=OVERLAY[0], atol=OVERLAY[1]
        )


@st.composite
def offsets(draw: st.DrawFn) -> tuple[int, int]:
    return (
        draw(st.integers(-100, 100), label="dx"),
        draw(st.integers(-100, 100), label="dy"),
    )


@property_lanes(weight=2, pair=shape_pairs(), shift=offsets())
def test_translation_moves_position_and_nothing_else(pair, shift) -> None:
    """Translating both contours by an integer offset: area, perimeter, IoU,
    Dice and Hausdorff distance are unchanged, the centroid and the box
    corner move by exactly that offset."""
    a, b = pair
    dx, dy = shift
    frame = pl.DataFrame([contour_series("a", [a]), contour_series("b", [b])])
    moved = {
        "a": pl.col("a").contour.translate(dx, dy),
        "b": pl.col("b").contour.translate(dx, dy),
    }
    base = run(
        frame,
        {
            **{k: f(pl.col("a")) for k, f in MEASURES.items()},
            **{k: f(pl.col("a"), pl.col("b")) for k, f in PAIR_MEASURES.items()},
        },
    )
    after = run(frame, moved)
    got = run(
        after,
        {
            **{k: f(pl.col("a")) for k, f in MEASURES.items()},
            **{k: f(pl.col("a"), pl.col("b")) for k, f in PAIR_MEASURES.items()},
        },
    )
    for name in ("perimeter", "hausdorff"):
        close(got[name], base[name], f"{name} after translate")
    close(got["area"], base["area"], "area after translate", [tol_of(a)])
    for name in ("iou", "dice"):
        close(got[name], base[name], f"{name} after translate", OVERLAY)
    offset = np.array([dx, dy], dtype=float)
    close(
        xy(got["centroid"].to_list()),
        [np.array(c) + offset for c in xy(base["centroid"].to_list())],
        "centroid after translate",
        [tol_of(a)],
    )
    close(
        fields(got["bbox"].to_list(), ("x", "y", "width", "height")),
        [
            [b_[0] + dx, b_[1] + dy, b_[2], b_[3]]
            for b_ in fields(base["bbox"].to_list(), ("x", "y", "width", "height"))
        ],
        "bounding_box after translate",
    )


@st.composite
def factors(draw: st.DrawFn) -> tuple[float, float]:
    sign = st.sampled_from([-1.0, 1.0])
    return (
        draw(sign, label="sign x") * draw(st.floats(0.25, 4.0), label="sx"),
        draw(sign, label="sign y") * draw(st.floats(0.25, 4.0), label="sy"),
    )


@property_lanes(weight=2, pair=shape_pairs(), factor=factors())
def test_scaling_laws(pair, factor) -> None:
    """Scaling both contours about the origin by ``(sx, sy)``: area multiplies
    by ``|sx sy|``, the perimeter of a uniform scale by ``|s|``, IoU and Dice
    do not change (any sign: a mirror is a rigid map of both), and the
    centroid scales with the points. Scaling about the centroid leaves the
    centroid where it was."""
    a, b = pair
    sx, sy = factor
    frame = pl.DataFrame([contour_series("a", [a]), contour_series("b", [b])])
    scaled = run(
        frame,
        {
            "a": pl.col("a").contour.scale(sx, sy, origin="origin"),
            "b": pl.col("b").contour.scale(sx, sy, origin="origin"),
        },
    )
    probes = {
        "area": pl.col("a").contour.area(),
        "centroid": pl.col("a").contour.centroid(),
        "iou": pl.col("a").contour.iou(pl.col("b")),
        "dice": pl.col("a").contour.dice(pl.col("b")),
    }
    base, got = run(frame, probes), run(scaled, probes)
    close(got["area"], [base["area"][0] * abs(sx * sy)], "area", [tol_of(a)])
    close(got["iou"], base["iou"], "iou", OVERLAY)
    close(got["dice"], base["dice"], "dice", OVERLAY)
    c = xy(base["centroid"].to_list())[0]
    close(
        xy(got["centroid"].to_list()),
        [[c[0] * sx, c[1] * sy]],
        "centroid scales",
        [tol_of(a)],
    )
    about = run(
        frame,
        {"c": pl.col("a").contour.scale(sx, sy).contour.centroid()},
    )["c"].to_list()
    close(xy(about), [c], "centroid is the default scale origin", [tol_of(a)])
    # A uniform scale also scales the (hole-free-or-not) perimeter by |s|.
    uniform = run(
        frame,
        {
            "p": pl.col("a").contour.perimeter(),
            "q": pl.col("a").contour.scale(sx, sx, origin="origin").contour.perimeter(),
        },
    )
    close(uniform["q"], [uniform["p"][0] * abs(sx)], "perimeter under uniform scale")


# ---------------------------------------------------------------------------
# Batches, engines, expression arguments
# ---------------------------------------------------------------------------


def contour_exprs() -> dict[str, pl.Expr]:
    c, d, p = pl.col("c"), pl.col("d"), pl.col("p")
    return {
        **{k: f(c) for k, f in MEASURES.items()},
        "signed": c.contour.area(signed=True),
        "iou": c.contour.iou(d),
        "dice": c.contour.dice(d),
        "hausdorff": c.contour.hausdorff_distance(d),
        "inside": c.contour.contains_point(p),
        "moved": c.contour.translate(3, -2),
        "scaled": c.contour.scale(2, 0.5),
        "convex": c.contour.is_convex(),
    }


def point_exprs() -> dict[str, pl.Expr]:
    p, q, c = pl.col("p"), pl.col("q"), pl.col("c")
    return {
        "distance": p.point.distance(q),
        "manhattan": p.point.manhattan_distance(q),
        "angle": p.point.angle_to(q),
        "midpoint": p.point.midpoint(q),
        "lerp": p.point.interpolate(q, t=0.25),
        "moved": p.point.translate(1.5, -2),
        "scaled": p.point.scale(2, 3),
        "rotated": p.point.rotate(0.7),
        "rotated_about": p.point.rotate(0.7, origin=q),
        "normalized": p.point.normalize(40, 50),
        "x": p.point.x(),
        "to_contour": p.point.distance_to_contour(c),
        "signed": p.point.signed_distance_to_contour(c),
        "nearest": p.point.nearest_point_on_contour(c),
    }


@st.composite
def point_values(draw: st.DrawFn, *, integral: bool = False):
    coord = st.integers(-50, 50).map(float) if integral else st.floats(-50.0, 50.0)
    return draw(st.one_of(st.none(), st.tuples(coord, coord)), label="point")


def point_series(name: str, values: list[Any]) -> pl.Series:
    return pl.Series(
        name,
        [None if v is None else {"x": v[0], "y": v[1]} for v in values],
        dtype=POINT_SCHEMA,
    )


@st.composite
def geometry_frames(draw: st.DrawFn) -> pl.DataFrame:
    """Contours ``c``/``d`` and points ``p``/``q``, a few rows, nulls anywhere."""
    n = draw(st.integers(1, 5), label="rows")
    row = nullable(shapes(), max_rows=n)
    cs = draw(row.filter(lambda r: len(r) == n), label="c")
    ds = draw(row.filter(lambda r: len(r) == n), label="d")
    ps = [draw(point_values(), label="p") for _ in range(n)]
    qs = [draw(point_values(), label="q") for _ in range(n)]
    return pl.DataFrame(
        [
            contour_series("c", cs),
            contour_series("d", ds),
            point_series("p", ps),
            point_series("q", qs),
        ]
    )


def rows_of(frame: pl.DataFrame) -> list[dict]:
    return frame.to_dicts()


@property_lanes(weight=2, frame=geometry_frames())
def test_a_row_does_not_depend_on_its_neighbours(frame: pl.DataFrame) -> None:
    """Every row of a batch equals the same row computed alone, and reversing
    the batch reverses the result: null rows stay null and shift nothing."""
    exprs = {**contour_exprs(), **{f"pt_{k}": v for k, v in point_exprs().items()}}
    whole = rows_of(run(frame, exprs))
    for i in range(frame.height):
        alone = rows_of(run(frame.slice(i, 1), exprs))
        assert alone == [whole[i]], f"row {i} differs alone"
    reversed_ = rows_of(run(frame.reverse(), exprs))
    assert reversed_ == whole[::-1], "reversing the batch reordered values"
    nulls = [i for i, v in enumerate(frame["c"].to_list()) if v is None]
    for i in nulls:
        for name in ("area", "perimeter", "centroid", "bbox", "moved", "inside"):
            assert whole[i][name] is None, f"null contour gave {name}={whole[i][name]}"


@property_lanes(
    weight=2,
    frame=geometry_frames(),
    other=geometry_frames(),
    engine=st.sampled_from(ENGINES),
)
def test_every_engine_gives_the_same_geometry(frame, other, engine) -> None:
    """Eager, lazy in-memory and lazy streaming agree exactly, over a frame
    split into two chunks (a morsel boundary mid-column)."""
    exprs = {**contour_exprs(), **{f"pt_{k}": v for k, v in point_exprs().items()}}
    chunked = pl.concat([frame, other], rechunk=False)
    with pl.Config():
        pl.Config.set_engine_affinity("in-memory")
        eager = chunked.select(**exprs)
    got = run(chunked, exprs, engine)
    assert rows_of(got) == rows_of(eager), f"{engine} differs from eager"


# --- expression parameters --------------------------------------------------

STYLES = ("literal", "lit", "column", "derived")


def as_argument(style: str, column: str, value: Any) -> Any:
    """How a parameter reaches the call. ``column`` reads a column holding the
    value (the caller supplies it); ``derived`` is a computed expression over a
    column of nulls."""
    if style == "literal":
        return value
    if style == "lit":
        return pl.lit(value)
    if style == "column":
        return pl.col(column)
    return pl.col(f"{column}_null").fill_null(value)


@property_lanes(
    weight=2,
    frame=geometry_frames(),
    style=st.sampled_from(STYLES),
    values=st.tuples(
        st.floats(-30.0, 30.0),
        st.floats(-30.0, 30.0),
        st.floats(0.25, 3.0),
        st.floats(0.25, 3.0),
        st.floats(-3.0, 3.0),
        st.floats(0.0, 1.0),
    ),
    signed=st.booleans(),
    origin=st.sampled_from(["centroid", "bbox_center", "origin"]),
)
def test_an_expression_argument_equals_the_literal(
    frame, style, values, signed, origin
) -> None:
    """A parameter given as ``pl.lit``, ``pl.col`` or a computed expression
    answers exactly as the same value written as a literal."""
    dx, dy, sx, sy, angle, t = values
    height = frame.height
    columns = {
        name: pl.Series(name, [v] * height, dtype=dtype)
        for name, v, dtype in (
            ("dx", dx, pl.Float64),
            ("dy", dy, pl.Float64),
            ("sx", sx, pl.Float64),
            ("sy", sy, pl.Float64),
            ("angle", angle, pl.Float64),
            ("t", t, pl.Float64),
            ("signed", signed, pl.Boolean),
            ("origin", origin, pl.String),
        )
    }
    nulls = {
        f"{name}_null": pl.Series(
            f"{name}_null", [None] * height, dtype=columns[name].dtype
        )
        for name in columns
    }
    wide = frame.with_columns(*columns.values(), *nulls.values())

    def build(style_: str) -> dict[str, pl.Expr]:
        arg = lambda n, v: as_argument(style_, n, v)  # noqa: E731
        c, p, q = pl.col("c"), pl.col("p"), pl.col("q")
        return {
            "translate": c.contour.translate(arg("dx", dx), arg("dy", dy)),
            "scale": c.contour.scale(
                arg("sx", sx), arg("sy", sy), origin=arg("origin", origin)
            ),
            "area": c.contour.area(signed=arg("signed", signed)),
            "p_translate": p.point.translate(arg("dx", dx), arg("dy", dy)),
            "p_scale": p.point.scale(arg("sx", sx), arg("sy", sy)),
            "p_rotate": p.point.rotate(arg("angle", angle)),
            "p_rotate_about": p.point.rotate(arg("angle", angle), origin=q),
            "p_lerp": p.point.interpolate(q, t=arg("t", t)),
        }

    event(f"style={style}")
    want = rows_of(run(wide, build("literal")))
    got = rows_of(run(wide, build(style)))
    assert got == want, f"{style} differs from the literal"


@property_lanes(
    weight=2,
    frame=geometry_frames(),
    seed=st.integers(0, 2**32 - 1),
)
def test_per_row_arguments_resolve_to_their_row(frame, seed) -> None:
    """With a different value in every row, each row equals the call written
    with *that row's* value as a literal, on that row alone."""
    rng = np.random.default_rng(seed)
    n = frame.height
    dx, dy = rng.uniform(-30, 30, n), rng.uniform(-30, 30, n)
    sx, sy = rng.uniform(0.25, 3.0, n), rng.uniform(0.25, 3.0, n)
    angle = rng.uniform(-3, 3, n)
    signed = rng.random(n) < 0.5
    wide = frame.with_columns(
        dx=pl.Series(dx),
        dy=pl.Series(dy),
        sx=pl.Series(sx),
        sy=pl.Series(sy),
        angle=pl.Series(angle),
        signed=pl.Series(signed),
    )

    def exprs(dx_, dy_, sx_, sy_, angle_, signed_) -> dict[str, pl.Expr]:
        c, p = pl.col("c"), pl.col("p")
        return {
            "translate": c.contour.translate(dx_, dy_),
            "scale": c.contour.scale(sx_, sy_),
            "area": c.contour.area(signed=signed_),
            "p_translate": p.point.translate(dx_, dy_),
            "p_scale": p.point.scale(sx_, sy_),
            "p_rotate": p.point.rotate(angle_),
        }

    whole = rows_of(
        run(
            wide,
            exprs(*(pl.col(n_) for n_ in ("dx", "dy", "sx", "sy", "angle", "signed"))),
        )
    )
    for i in range(n):
        alone = rows_of(
            run(
                frame.slice(i, 1),
                exprs(
                    float(dx[i]),
                    float(dy[i]),
                    float(sx[i]),
                    float(sy[i]),
                    float(angle[i]),
                    bool(signed[i]),
                ),
            )
        )
        assert alone == [whole[i]], f"row {i}: the column value was not its row's"


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


@st.composite
def point_columns(draw: st.DrawFn, *, integral: bool = False, names=("p", "q", "r")):
    n = draw(st.integers(1, 5), label="rows")
    return n, {
        name: [draw(point_values(integral=integral), label=name) for _ in range(n)]
        for name in names
    }


def point_frame(columns: dict[str, list[Any]]) -> pl.DataFrame:
    return pl.DataFrame([point_series(k, v) for k, v in columns.items()])


def both(a: Any, b: Any):
    """Pair two optional points; ``None`` if either is null."""
    return None if a is None or b is None else (np.array(a), np.array(b))


@property_lanes(weight=2, case=point_columns())
def test_point_arithmetic_matches_numpy(case) -> None:
    """Distance, Manhattan distance, angle, midpoint, interpolation, and the
    translate / scale / rotate / normalize maps are the NumPy formulas; a
    null in either operand is null (a null origin included)."""
    n, cols = case
    frame = point_frame(cols)
    out = run(
        frame,
        {
            "distance": pl.col("p").point.distance(pl.col("q")),
            "manhattan": pl.col("p").point.manhattan_distance(pl.col("q")),
            "angle": pl.col("p").point.angle_to(pl.col("q")),
            "midpoint": pl.col("p").point.midpoint(pl.col("q")),
            "lerp": pl.col("p").point.interpolate(pl.col("q"), t=0.25),
            "translate": pl.col("p").point.translate(1.5, -2.0),
            "scale": pl.col("p").point.scale(2.0, 0.5),
            "rotate": pl.col("p").point.rotate(0.7),
            "rotate_about": pl.col("p").point.rotate(0.7, origin=pl.col("q")),
            "normalize": pl.col("p").point.normalize(40.0, 50.0),
            "x": pl.col("p").point.x(),
            "y": pl.col("p").point.y(),
        },
    )
    pq = [both(a, b) for a, b in zip(cols["p"], cols["q"])]
    p = [None if v is None else np.array(v) for v in cols["p"]]

    def over(f: Callable[[np.ndarray, np.ndarray], Any]) -> list[Any]:
        return [None if v is None else f(*v) for v in pq]

    def single(f: Callable[[np.ndarray], Any]) -> list[Any]:
        return [None if v is None else f(v) for v in p]

    def rot(v: np.ndarray, centre: np.ndarray, a: float) -> np.ndarray:
        d = v - centre
        return centre + np.array(
            [
                d[0] * math.cos(a) - d[1] * math.sin(a),
                d[0] * math.sin(a) + d[1] * math.cos(a),
            ]
        )

    close(out["distance"], over(lambda a, b: np.hypot(*(a - b))), "distance")
    close(out["manhattan"], over(lambda a, b: np.abs(a - b).sum()), "manhattan")
    close(
        out["angle"],
        over(lambda a, b: np.arctan2(b[1] - a[1], b[0] - a[0])),
        "angle_to",
    )
    close(xy(out["midpoint"].to_list()), over(lambda a, b: (a + b) / 2), "midpoint")
    close(
        xy(out["lerp"].to_list()), over(lambda a, b: a + 0.25 * (b - a)), "interpolate"
    )
    close(
        xy(out["translate"].to_list()),
        single(lambda a: a + np.array([1.5, -2.0])),
        "translate",
    )
    close(
        xy(out["scale"].to_list()), single(lambda a: a * np.array([2.0, 0.5])), "scale"
    )
    close(
        xy(out["rotate"].to_list()),
        single(lambda a: rot(a, np.zeros(2), 0.7)),
        "rotate",
    )
    # A null origin nulls its row (CHANGELOG: ``.point.rotate(origin=...)``).
    expected_about = over(lambda a, b: rot(a, b, 0.7))
    close(xy(out["rotate_about"].to_list()), expected_about, "rotate about origin")
    close(
        xy(out["normalize"].to_list()),
        single(lambda a: a / np.array([40.0, 50.0])),
        "normalize",
    )
    close(out["x"], single(lambda a: a[0]), "x")
    close(out["y"], single(lambda a: a[1]), "y")


@property_lanes(weight=2, case=point_columns())
def test_distance_is_a_metric_and_rigid_maps_preserve_it(case) -> None:
    """Distance is symmetric, zero on itself and obeys the triangle inequality;
    translating or rotating both points leaves it unchanged, and scaling both
    by ``s`` multiplies it by ``|s|``."""
    n, cols = case
    frame = point_frame(cols)
    p, q, r = (pl.col(k) for k in "pqr")
    out = run(
        frame,
        {
            "pq": p.point.distance(q),
            "qp": q.point.distance(p),
            "pp": p.point.distance(p),
            "qr": q.point.distance(r),
            "pr": p.point.distance(r),
            "moved": p.point.translate(7.0, -3.0).point.distance(
                q.point.translate(7.0, -3.0)
            ),
            "turned": p.point.rotate(1.1).point.distance(q.point.rotate(1.1)),
            "scaled": p.point.scale(-2.5, -2.5).point.distance(
                q.point.scale(-2.5, -2.5)
            ),
        },
    )
    for i in range(n):
        if out["pq"][i] is None:
            assert out["qp"][i] is None and out["moved"][i] is None
            continue
        d = out["pq"][i]
        assert d >= 0.0
        np.testing.assert_allclose(out["qp"][i], d, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(out["moved"][i], d, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(out["turned"][i], d, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(out["scaled"][i], 2.5 * d, rtol=RTOL, atol=ATOL)
        assert out["pp"][i] == 0.0, "distance of a point to itself"
        if out["qr"][i] is not None and out["pr"][i] is not None:
            assert out["pr"][i] <= d + out["qr"][i] + ATOL, "triangle inequality"


@st.composite
def boxes_and_points(draw: st.DrawFn):
    """Integer-valued boxes and points, so a point lands on an edge or corner
    often enough to pin the inclusive rule."""
    n = draw(st.integers(1, 5), label="rows")
    rows = []
    for _ in range(n):
        box = draw(
            st.one_of(
                st.none(),
                st.tuples(
                    st.integers(-5, 5),
                    st.integers(-5, 5),
                    st.integers(0, 6),
                    st.integers(0, 6),
                ),
            ),
            label="box",
        )
        rows.append((box, draw(point_values(integral=True), label="point")))
    return rows


@property_lanes(weight=1, rows=boxes_and_points())
def test_within_bbox_is_an_inclusive_comparison(rows) -> None:
    """``within_bbox`` is ``x0 <= x <= x0 + w`` and likewise for y — edges
    and corners included — and null if the point or the box is."""
    frame = pl.DataFrame(
        [
            pl.Series(
                "b",
                [
                    None
                    if b is None
                    else {
                        "x": float(b[0]),
                        "y": float(b[1]),
                        "width": float(b[2]),
                        "height": float(b[3]),
                    }
                    for b, _ in rows
                ],
                dtype=BBOX_SCHEMA,
            ),
            point_series("p", [p for _, p in rows]),
        ]
    )
    got = run(frame, {"w": pl.col("p").point.within_bbox(pl.col("b"))})["w"].to_list()
    want = [
        None
        if b is None or p is None
        else bool(b[0] <= p[0] <= b[0] + b[2] and b[1] <= p[1] <= b[1] + b[3])
        for b, p in rows
    ]
    assert got == want


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------


@st.composite
def box_values(draw: st.DrawFn, *, degenerate: bool = True):
    size = st.floats(0.1, 30.0)
    if degenerate:
        size = st.one_of(st.just(0.0), size)
    return (
        draw(st.floats(-40.0, 40.0), label="x"),
        draw(st.floats(-40.0, 40.0), label="y"),
        draw(size, label="w"),
        draw(size, label="h"),
    )


def box_sets(*, degenerate: bool = True):
    """A row is a set of 0-4 boxes, or null."""
    return st.one_of(
        st.none(),
        st.lists(box_values(degenerate=degenerate), min_size=0, max_size=4),
    )


def box_series(name: str, rows: list[Any]) -> pl.Series:
    return pl.Series(
        name,
        [
            None
            if row is None
            else [{"x": b[0], "y": b[1], "width": b[2], "height": b[3]} for b in row]
            for row in rows
        ],
        dtype=pl.List(BBOX_SCHEMA),
    )


def box_iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two ``(x, y, w, h)`` boxes; 0 when the union
    has no area (two empty boxes), the convention the engine documents for an
    empty denominator."""
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0.0 else 0.0


def box_matrix(a: list, b: list) -> list[list[float]]:
    return [[box_iou(x, y) for y in b] for x in a]


def matrix_rows(series: pl.Series) -> list[Any]:
    return series.to_list()


def assert_matrices(got: list[Any], want: list[Any], what: str) -> None:
    assert len(got) == len(want)
    for i, (g, w) in enumerate(zip(got, want)):
        assert (g is None) == (w is None), f"{what}: row {i} null mismatch"
        if g is None:
            continue
        assert len(g) == len(w), f"{what}: row {i} has {len(g)} rows, want {len(w)}"
        for gr, wr in zip(g, w):
            assert len(gr) == len(wr), f"{what}: row {i} column count"
            np.testing.assert_allclose(gr, wr, rtol=RTOL, atol=ATOL, err_msg=what)


@property_lanes(
    weight=2,
    left=st.lists(box_sets(), min_size=1, max_size=4),
    right=st.lists(box_sets(), min_size=1, max_size=4),
)
def test_bbox_pairwise_iou_matches_numpy(left, right) -> None:
    """The N x M matrix is the NumPy box IoU (a pair of empty boxes is 0); a
    null on either side is a null row; the rows are ``left``'s boxes."""
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    frame = pl.DataFrame([box_series("a", left), box_series("b", right)])
    got = run(frame, {"m": pl.col("a").bbox.pairwise_iou(pl.col("b"))})["m"].to_list()
    want = [
        None if a is None or b is None else box_matrix(a, b)
        for a, b in zip(left, right)
    ]
    assert_matrices(got, want, "pairwise_iou")


@property_lanes(
    weight=2,
    left=st.lists(box_sets(degenerate=False), min_size=1, max_size=4),
    right=st.lists(box_sets(degenerate=False), min_size=1, max_size=4),
    shift=offsets(),
    scale=st.floats(0.25, 4.0),
)
def test_bbox_iou_laws(left, right, shift, scale) -> None:
    """``iou(a, b)`` is ``iou(b, a)`` transposed, ``iou(a, a)`` has a diagonal
    of 1, every entry is in [0, 1], and translating or scaling both sets
    leaves it unchanged."""
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    dx, dy = shift

    def transformed(rows: list, dxy: tuple[float, float], s: float) -> list:
        return [
            None
            if row is None
            else [
                (b[0] * s + dxy[0], b[1] * s + dxy[1], b[2] * s, b[3] * s) for b in row
            ]
            for row in rows
        ]

    frame = pl.DataFrame([box_series("a", left), box_series("b", right)])
    out = run(
        frame,
        {
            "ab": pl.col("a").bbox.pairwise_iou(pl.col("b")),
            "ba": pl.col("b").bbox.pairwise_iou(pl.col("a")),
            "aa": pl.col("a").bbox.pairwise_iou(pl.col("a")),
        },
    )
    for ab, ba in zip(out["ab"].to_list(), out["ba"].to_list()):
        assert (ab is None) == (ba is None)
        if ab is None:
            continue
        transposed = [[ba[j][i] for j in range(len(ba))] for i in range(len(ab))]
        assert ab == transposed, "iou(a, b) is not iou(b, a) transposed"
        for row in ab:
            assert all(0.0 <= v <= 1.0 for v in row), f"iou outside [0, 1]: {row}"
    for aa, a in zip(out["aa"].to_list(), left):
        if a is None:
            assert aa is None
        else:
            np.testing.assert_allclose(
                [aa[i][i] for i in range(len(a))], 1.0, rtol=RTOL, atol=ATOL
            )
    for what, dxy, s in (("translate", (dx, dy), 1.0), ("scale", (0.0, 0.0), scale)):
        moved = pl.DataFrame(
            [
                box_series("a", transformed(left, dxy, s)),
                box_series("b", transformed(right, dxy, s)),
            ]
        )
        got = run(moved, {"m": pl.col("a").bbox.pairwise_iou(pl.col("b"))})["m"]
        assert_matrices(got.to_list(), out["ab"].to_list(), f"iou under {what}")


@property_lanes(
    weight=1,
    pairs=st.lists(
        st.tuples(box_values(degenerate=False), box_values(degenerate=False)),
        min_size=1,
        max_size=4,
    ),
)
def test_bbox_iou_agrees_with_the_contour_iou_of_the_same_rectangles(pairs) -> None:
    """Two independent implementations of one quantity: ``.bbox`` computes box
    IoU analytically, ``.contour`` clips polygons. A box and the rectangle it
    describes must get the same overlap."""

    def rect(b: tuple) -> dict:
        x, y, w, h = b
        return {
            "exterior": to_points(
                np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            ),
            "holes": [],
            "is_closed": True,
        }

    frame = pl.DataFrame(
        [
            box_series("a", [[a] for a, _ in pairs]),
            box_series("b", [[b] for _, b in pairs]),
            pl.Series("ca", [rect(a) for a, _ in pairs], dtype=CONTOUR_SCHEMA),
            pl.Series("cb", [rect(b) for _, b in pairs], dtype=CONTOUR_SCHEMA),
        ]
    )
    out = run(
        frame,
        {
            "box": pl.col("a").bbox.pairwise_iou(pl.col("b")),
            "poly": pl.col("ca").contour.iou(pl.col("cb")),
        },
    )
    want = [box_iou(a, b) for a, b in pairs]
    boxes = [m[0][0] for m in out["box"].to_list()]
    close(boxes, want, "bbox iou vs numpy")
    # The polygon clipper is an overlay: it carries the OVERLAY error model.
    close(out["poly"].to_list(), boxes, "contour iou vs bbox iou", OVERLAY)


@property_lanes(
    weight=2,
    left=st.lists(box_sets(degenerate=False), min_size=1, max_size=4),
    right=st.lists(box_sets(degenerate=False), min_size=1, max_size=4),
    threshold=st.floats(0.0, 1.0),
)
def test_bbox_correspond_is_consistent_with_pairwise_iou(
    left, right, threshold
) -> None:
    """Every match ``correspond`` reports is an entry of the IoU matrix at or
    above the threshold, used by at most one box, and the overlap it reports
    is that entry. A box left unmatched has no such unused entry above the
    threshold only if matching is greedy-exclusive, so only the positive claim
    is made."""
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    frame = pl.DataFrame([box_series("a", left), box_series("b", right)])
    out = run(
        frame,
        {
            "m": pl.col("a").bbox.pairwise_iou(pl.col("b")),
            "c": pl.col("a").bbox.correspond(pl.col("b"), threshold=threshold),
        },
    )
    for matrix, corr, a in zip(out["m"].to_list(), out["c"].to_list(), left):
        assert (matrix is None) == (corr is None)
        if matrix is None:
            continue
        used = [j for j in corr["right_idx"] if j is not None]
        assert len(used) == len(set(used)), "a right box matched twice"
        assert len(corr["right_idx"]) == len(a)
        for i, (j, overlap) in enumerate(zip(corr["right_idx"], corr["overlap"])):
            if j is None:
                continue
            np.testing.assert_allclose(overlap, matrix[i][j], rtol=RTOL, atol=ATOL)
            assert overlap >= threshold - ATOL, "matched below the threshold"


@property_lanes(
    weight=2,
    left=st.lists(box_sets(), min_size=1, max_size=5),
    right=st.lists(box_sets(), min_size=1, max_size=5),
    engine=st.sampled_from(ENGINES),
)
def test_bbox_rows_are_independent_and_engines_agree(left, right, engine) -> None:
    """Each row of ``pairwise_iou`` / ``correspond`` equals the row alone (null
    and empty rows included), and every engine gives the same answer."""
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    frame = pl.DataFrame([box_series("a", left), box_series("b", right)])
    exprs = {
        "m": pl.col("a").bbox.pairwise_iou(pl.col("b")),
        "c": pl.col("a").bbox.correspond(pl.col("b"), threshold=0.3),
    }
    whole = rows_of(run(frame, exprs, engine))
    for i in range(n):
        assert rows_of(run(frame.slice(i, 1), exprs, engine)) == [whole[i]], (
            f"row {i} differs alone"
        )
    assert rows_of(run(frame.reverse(), exprs, engine)) == whole[::-1]
    assert whole == rows_of(run(frame, exprs, "in-memory")), "engines disagree"
