"""Reference tests for the contour shaping ops and line rendering (CR-98).

* ``offset(distance, join=)`` grows (or, negative, shrinks) the union of a
  set's regions; the result is always a set (a region can split or merge).
* ``union_all()`` is that union as disjoint regions, a set.
* ``smooth(iterations=)`` is Chaikin's corner cutting; ``densify(max_segment_length=)``
  splits each edge into ``ceil(L / max)`` equal parts.
* ``stroke(width=, cap=, join=)`` is the region each member's path covers
  (a line with caps, a region's rings as closed paths), as disjoint regions;
  ``rasterize`` fills it, with ``supersample``.

Oracles, each independent of the engine:

* A round offset is ``{p : signed_distance(p, R) <= d}`` and a round stroke
  ``{p : distance(p, line) <= width / 2}``. Checked on the samples the mask
  takes, the two may disagree only within ``ARC_TOLERANCE`` of the boundary
  (where a polygon stands in for each arc), so each pixel is held between the
  value its surely-inside samples give and the value its possibly-inside ones
  give (``_assert_band``).
* Offset areas of a square against closed forms, per join.
* A pure-Python Chaikin and an exact densify construction, point for point.
* ``rasterize(union_all(s)) == rasterize(s)`` (rasterize already paints the
  union) on integer rectilinear shapes, exactly, with disjoint members.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from matplotlib.path import Path

from polars_cv import Pipeline, numpy_from_column
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA
from tests.conftest import plugin_required

pytestmark = plugin_required

#: The largest gap between an arc and the polygon standing in for it, as the
#: engine documents it (``transforms::ARC_TOLERANCE``), plus snapping slack.
ARC_TOLERANCE = 1.0 / 64.0 + 1e-6


# ---------------------------------------------------------------------------
# Building and running
# ---------------------------------------------------------------------------


def _pts(points) -> list[dict]:
    return [{"x": float(x), "y": float(y)} for x, y in points]


def _region(points, holes=()) -> dict:
    return {
        "exterior": _pts(points),
        "holes": [_pts(h) for h in holes],
        "is_closed": True,
    }


def _line(points) -> dict:
    return {"exterior": _pts(points), "holes": [], "is_closed": False}


def _rect(x0, y0, x1, y1) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _frame(*sets) -> pl.DataFrame:
    return pl.DataFrame({"c": list(sets)}, schema={"c": CONTOUR_SET_SCHEMA})


def _pipe(contours: list[dict], pipe: Pipeline, sink: str = "native"):
    out = _frame(contours).select(pl.col("c").cv.pipe(pipe).sink(sink)).item()
    return out.to_list() if isinstance(out, pl.Series) else out


def _per_row_error(
    contours: list[dict], build, value: float, match: str, sink: str = "native"
) -> None:
    """``build(pl.col("v"))`` refuses ``value`` with its row: the check a
    literal meets at build, reached per row."""
    df = pl.DataFrame(
        {"c": [contours], "v": [value]},
        schema={"c": CONTOUR_SET_SCHEMA, "v": pl.Float64},
    )
    pipe = build(Pipeline().source("contour"), pl.col("v"))
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        df.select(pl.col("c").cv.pipe(pipe).sink(sink))


def _shaped(contours: list[dict], op: str, **kw) -> list[dict]:
    return _pipe(contours, getattr(Pipeline().source("contour"), op)(**kw))


def _mask(contours: list[dict], w: int, h: int, **kw) -> np.ndarray:
    out = _frame(contours).select(
        pl.col("c")
        .cv.pipe(Pipeline().source("contour").rasterize(width=w, height=h, **kw))
        .sink("numpy")
    )
    (mask,) = numpy_from_column(out.to_series(), copy=True)
    assert mask is not None and mask.shape == (h, w, 1)
    return mask[:, :, 0]


def _areas(contours: list[dict]) -> list[float]:
    return _pipe(contours, Pipeline().source("contour").area()) if contours else []


def _xy(points: list[dict]) -> np.ndarray:
    return np.array([(p["x"], p["y"]) for p in points], dtype=float)


# ---------------------------------------------------------------------------
# Distance oracles
# ---------------------------------------------------------------------------


def _samples(w: int, h: int, n: int) -> np.ndarray:
    """Every sub-pixel centre the mask samples, row-major by sample row."""
    offsets = (np.arange(n) + 0.5) / n
    xs = (np.arange(w)[:, None] + offsets[None, :]).ravel()
    ys = (np.arange(h)[:, None] + offsets[None, :]).ravel()
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def _segment_distance(points: np.ndarray, path: np.ndarray, closed: bool) -> np.ndarray:
    """Distance from each point to the polyline ``path`` (one point: to it)."""
    if len(path) == 1:
        return np.hypot(*(points - path[0]).T)
    a = path if closed else path[:-1]
    b = np.roll(path, -1, axis=0) if closed else path[1:]
    best = np.full(len(points), np.inf)
    for pa, pb in zip(a, b, strict=True):
        ab = pb - pa
        denom = float(ab @ ab)
        t = np.zeros(len(points)) if denom == 0 else ((points - pa) @ ab) / denom
        nearest = pa + np.clip(t, 0.0, 1.0)[:, None] * ab
        best = np.minimum(best, np.hypot(*(points - nearest).T))
    return best


def _signed_distance(points: np.ndarray, region: dict) -> np.ndarray:
    """Negative inside the region (exterior minus holes), positive outside."""
    rings = [_xy(region["exterior"]), *(_xy(h) for h in region["holes"])]
    inside = Path(rings[0]).contains_points(points)
    for hole in rings[1:]:
        inside &= ~Path(hole).contains_points(points)
    dist = np.min([_segment_distance(points, r, closed=True) for r in rings], axis=0)
    return np.where(inside, -dist, dist)


def _assert_band(got: np.ndarray, s: np.ndarray, n: int) -> None:
    """``got`` paints ``{s <= 0}`` (``s`` per sample), up to ``ARC_TOLERANCE``.

    A pixel's samples with ``s < -tol`` are surely inside and those with
    ``s <= tol`` possibly inside: its value must lie between the two counts'
    values. The band must leave most pixels decided, or the check says nothing.
    """
    h, w = got.shape
    sure = (s < -ARC_TOLERANCE).reshape(h, n, w, n).sum(axis=(1, 3))
    maybe = (s <= ARC_TOLERANCE).reshape(h, n, w, n).sum(axis=(1, 3))

    def value(k: np.ndarray) -> np.ndarray:
        v = 255 * k / (n * n)
        return np.floor(v + 0.5)

    lo, hi = value(sure), value(maybe)
    bad = (got < lo) | (got > hi)
    assert not bad.any(), (
        f"{int(bad.sum())} pixels outside the band, first at "
        f"{tuple(int(i) for i in np.argwhere(bad)[0])}"
    )
    assert (lo == hi).mean() > 0.9, "the tolerance band decides too few pixels"
    assert (lo == 255).any() and (hi == 0).any(), "a vacuous fixture"


def _star(rng, cx, cy, r, k=9) -> list[tuple[float, float]]:
    angles = np.sort(rng.uniform(0, 2 * math.pi, k))
    radii = rng.uniform(0.45 * r, r, k)
    return list(
        zip(cx + radii * np.cos(angles), cy + radii * np.sin(angles), strict=True)
    )


# ---------------------------------------------------------------------------
# offset
# ---------------------------------------------------------------------------

SQUARE = _rect(5, 5, 15, 15)


class TestOffset:
    @pytest.mark.parametrize(
        ("join", "area"),
        [("miter", 16.0**2), ("bevel", 16.0**2 - 2 * 3.0**2)],
    )
    def test_a_square_grows_by_its_closed_form(self, join: str, area: float) -> None:
        (grown,) = _shaped([_region(SQUARE)], "offset", distance=3.0, join=join)
        assert grown["is_closed"] is True
        assert _areas([grown]) == pytest.approx([area], rel=1e-6)

    def test_a_round_join_adds_quarter_discs(self) -> None:
        (grown,) = _shaped([_region(SQUARE)], "offset", distance=3.0)
        exact = 10.0**2 + 4 * 10.0 * 3.0 + math.pi * 3.0**2
        (area,) = _areas([grown])
        # The arcs are inscribed polygons: never more, and short by at most
        # the arcs' length times the tolerance.
        assert exact - 2 * math.pi * 3.0 * ARC_TOLERANCE <= area <= exact + 1e-6

    @pytest.mark.parametrize("join", ["round", "miter", "bevel"])
    def test_a_square_shrinks_to_a_square_whatever_the_join(self, join: str) -> None:
        (shrunk,) = _shaped([_region(SQUARE)], "offset", distance=-3.0, join=join)
        assert _areas([shrunk]) == pytest.approx([4.0**2], rel=1e-6)

    def test_holes_shrink_as_the_exterior_grows(self) -> None:
        holed = _region(_rect(0, 0, 20, 20), [_rect(5, 5, 15, 15)])
        (grown,) = _shaped([holed], "offset", distance=2.0, join="miter")
        assert len(grown["holes"]) == 1
        assert _areas([grown]) == pytest.approx([24.0**2 - 6.0**2], rel=1e-6)

    def test_shrinking_splits_a_dumbbell(self) -> None:
        # Two 8x8 squares joined by a bar 2 high: shrinking by 1.5 cuts it.
        dumbbell = [
            (0, 0), (8, 0), (8, 3), (12, 3), (12, 0), (20, 0),
            (20, 8), (12, 8), (12, 5), (8, 5), (8, 8), (0, 8),
        ]  # fmt: skip
        parts = _shaped([_region(dumbbell)], "offset", distance=-1.5, join="miter")
        assert sorted(_areas(parts)) == pytest.approx([25.0, 25.0], rel=1e-6)

    def test_growing_merges_a_set_into_one_region(self) -> None:
        pair = [_region(_rect(0, 0, 8, 8)), _region(_rect(10, 0, 18, 8))]
        (merged,) = _shaped(pair, "offset", distance=1.5, join="miter")
        assert _areas([merged]) == pytest.approx([21.0 * 11.0], rel=1e-6)

    def test_shrinking_past_the_middle_leaves_nothing(self) -> None:
        assert _shaped([_region(_rect(0, 0, 4, 4))], "offset", distance=-3.0) == []

    def test_zero_is_the_union(self) -> None:
        pair = [_region(_rect(0, 0, 8, 8)), _region(_rect(4, 4, 12, 12))]
        same = _shaped(pair, "offset", distance=0.0)
        assert same == _shaped(pair, "union_all")

    @pytest.mark.parametrize("distance", [2.5, -1.5])
    @pytest.mark.parametrize("seed", range(3))
    def test_a_round_offset_is_a_distance_band(
        self, distance: float, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        region = _region(_star(rng, 20.3, 19.7, 13.0))
        grown = _shaped([region], "offset", distance=distance)
        n, (w, h) = 2, (41, 40)
        s = _signed_distance(_samples(w, h, n), region) - distance
        _assert_band(_mask(grown, w, h, supersample=n), s, n)

    def test_offset_reads_the_set_as_its_union(self) -> None:
        """Overlapping members shrink as one region, not each on its own."""
        pair = [_region(_rect(0, 0, 6, 10)), _region(_rect(4, 0, 10, 10))]
        (shrunk,) = _shaped(pair, "offset", distance=-2.0, join="miter")
        assert _areas([shrunk]) == pytest.approx([6.0 * 6.0], rel=1e-6)

    def test_a_line_is_refused(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError, match="open contour.*no region"):
            _shaped([_line([(0, 0), (5, 5)])], "offset", distance=1.0)

    @pytest.mark.parametrize("distance", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_distance_is_refused(self, distance: float) -> None:
        _per_row_error(
            [_region(SQUARE)],
            lambda p, v: p.offset(v),
            distance,
            "distance must be finite",
        )

    def test_a_per_row_distance(self) -> None:
        df = pl.DataFrame(
            {"c": [[_region(SQUARE)], [_region(SQUARE)]], "d": [1.0, -1.0]},
            schema={"c": CONTOUR_SET_SCHEMA, "d": pl.Float64},
        )
        pipe = Pipeline().source("contour").offset(pl.col("d"), join="miter").area()
        out = df.select(pl.col("c").cv.pipe(pipe).sink("native")).to_series()
        assert out.to_list() == [pytest.approx([144.0]), pytest.approx([64.0])]


# ---------------------------------------------------------------------------
# union_all
# ---------------------------------------------------------------------------


class TestUnionAll:
    @pytest.mark.parametrize("seed", range(5))
    def test_the_union_paints_what_the_set_paints(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        rects = []
        for _ in range(7):
            x0, y0 = rng.integers(0, 30, 2)
            x1, y1 = x0 + rng.integers(1, 12), y0 + rng.integers(1, 12)
            rects.append(_region(_rect(x0, y0, x1, y1)))
        union = _shaped(rects, "union_all")
        assert all(c["is_closed"] for c in union)
        expected = _mask(rects, 44, 44)
        np.testing.assert_array_equal(_mask(union, 44, 44), expected)
        # Disjoint members: their areas add up to the painted pixels.
        assert sum(_areas(union)) == pytest.approx(int((expected > 0).sum()))

    def test_a_ring_with_an_island_stays_two_regions(self) -> None:
        ring = _region(_rect(0, 0, 20, 20), [_rect(4, 4, 16, 16)])
        island = _region(_rect(8, 8, 12, 12))
        union = _shaped([ring, island], "union_all")
        assert sorted(_areas(union)) == pytest.approx([16.0, 400.0 - 144.0])

    def test_a_member_filling_a_hole_closes_it(self) -> None:
        ring = _region(_rect(0, 0, 20, 20), [_rect(4, 4, 16, 16)])
        plug = _region(_rect(3, 3, 17, 17))
        (whole,) = _shaped([ring, plug], "union_all")
        assert whole["holes"] == []
        assert _areas([whole]) == pytest.approx([400.0])

    def test_an_empty_set_is_empty(self) -> None:
        assert _shaped([], "union_all") == []

    def test_a_line_is_refused(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError, match="open contour.*no region"):
            _shaped([_region(SQUARE), _line([(0, 0), (5, 5)])], "union_all")


# ---------------------------------------------------------------------------
# smooth
# ---------------------------------------------------------------------------


def _cut(a, b):
    return (
        (0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]),
        (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]),
    )


def _chaikin_path(points: list, *, ring: bool) -> list:
    """One pass. A ring cuts its closing edge too and starts at its first
    edge's first cut; a line keeps its ends, unless it returns to its first
    point, when it has none and is cut through it as a loop."""
    if ring:
        points = [*points, points[0]]
    loop = ring or points[0] == points[-1]
    out = [] if loop else [points[0]]
    for a, b in zip(points, points[1:], strict=False):
        out.extend(_cut(a, b))
    if ring:
        return out
    return [*out, out[0]] if loop else [*out, points[-1]]


def _chaikin(contour: dict, iterations: int) -> dict:
    ring = contour["is_closed"]

    def run(path):
        path = [(p["x"], p["y"]) for p in path]
        for _ in range(iterations):
            path = _chaikin_path(path, ring=ring)
        return _pts(path)

    return {
        "exterior": run(contour["exterior"]),
        "holes": [run(h) for h in contour["holes"]],
        "is_closed": ring,
    }


def _assert_same_contour(got: dict, want: dict) -> None:
    assert got["is_closed"] is want["is_closed"]
    np.testing.assert_allclose(_xy(got["exterior"]), _xy(want["exterior"]), atol=1e-9)
    assert len(got["holes"]) == len(want["holes"])
    for g, w in zip(got["holes"], want["holes"], strict=True):
        np.testing.assert_allclose(_xy(g), _xy(w), atol=1e-9)


ZIGZAG = [(1, 1), (5, 2), (3, 7), (9, 8)]


class TestSmooth:
    @pytest.mark.parametrize("iterations", [0, 1, 3])
    @pytest.mark.parametrize(
        "contour",
        [
            _line(ZIGZAG),
            _line([*SQUARE, SQUARE[0]]),
            _region(_rect(0, 0, 20, 20), [_rect(5, 5, 9, 9)]),
            _region([(0, 0), (10, 1), (4, 9)]),
        ],
        ids=["line", "loop-line", "holed-region", "triangle"],
    )
    def test_matches_chaikin(self, contour: dict, iterations: int) -> None:
        (got,) = _shaped([contour], "smooth", iterations=iterations)
        _assert_same_contour(got, _chaikin(contour, iterations))

    def test_the_default_is_one_pass(self) -> None:
        (got,) = _shaped([_line(ZIGZAG)], "smooth")
        _assert_same_contour(got, _chaikin(_line(ZIGZAG), 1))

    @pytest.mark.parametrize("iterations", [40, 2**32 - 1])
    def test_passes_past_the_point_budget_are_refused_before_running(
        self, iterations: int
    ) -> None:
        """A pass doubles the points: 40 would need ~10^13, so the row fails
        by name rather than the process aborting on the allocation."""
        match = rf"smooth: {iterations} passes would make about .* points"
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            _shaped([_line(ZIGZAG)], "smooth", iterations=iterations)
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            _frame([_line(ZIGZAG)]).select(
                pl.col("c").contour.smooth(iterations=iterations)
            )


# ---------------------------------------------------------------------------
# densify
# ---------------------------------------------------------------------------


def _densify_path(points: list, max_len: float, *, ring: bool) -> list:
    """Each edge split into ``ceil(L / max_len)`` equal parts, its ends kept."""
    if ring:
        points = [*points, points[0]]
    out = [points[0]]
    for a, b in zip(points, points[1:], strict=False):
        parts = math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / max_len)
        for k in range(1, parts):
            t = k / parts
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        out.append(b)
    return out[:-1] if ring else out


def _densified(contour: dict, max_len: float) -> dict:
    ring = contour["is_closed"]

    def run(path):
        return _pts(_densify_path([(p["x"], p["y"]) for p in path], max_len, ring=ring))

    return {
        "exterior": run(contour["exterior"]),
        "holes": [run(h) for h in contour["holes"]],
        "is_closed": ring,
    }


class TestDensify:
    @pytest.mark.parametrize("max_len", [0.7, 2.0, 50.0])
    @pytest.mark.parametrize(
        "contour",
        [
            _line(ZIGZAG),
            _region(_rect(0, 0, 20, 20), [_rect(5, 5, 9, 9)]),
            _region([(0, 0), (10, 1), (4, 9)]),
        ],
        ids=["line", "holed-region", "triangle"],
    )
    def test_matches_the_equal_split(self, contour: dict, max_len: float) -> None:
        (got,) = _shaped([contour], "densify", max_segment_length=max_len)
        want = _densified(contour, max_len)
        _assert_same_contour(got, want)
        # The bound itself, independently of the construction.
        for path in [got["exterior"], *got["holes"]]:
            p = _xy(path)
            if contour["is_closed"]:
                p = np.vstack([p, p[:1]])
            assert np.hypot(*np.diff(p, axis=0).T).max() <= max_len + 1e-9

    @pytest.mark.parametrize("max_len", [0.0, -1.0])
    def test_a_length_no_row_can_use_is_refused_at_build(self, max_len: float) -> None:
        with pytest.raises(ValueError, match="max_segment_length must be a finite"):
            Pipeline().source("contour").densify(max_len)

    def test_a_length_past_the_point_budget_is_refused_before_running(self) -> None:
        """1e-9 splits a 40-unit outline into ~4e10 points: refused by name."""
        match = r"densify: max_segment_length=0\.000000001 would make about .* points"
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            _shaped([_region(_rect(0, 0, 10, 10))], "densify", max_segment_length=1e-9)
        with pytest.raises(pl.exceptions.ComputeError, match=match):
            _frame([_region(_rect(0, 0, 10, 10))]).select(
                pl.col("c").contour.densify(1e-9)
            )

    @pytest.mark.parametrize("max_len", [0.0, math.nan, math.inf])
    def test_a_per_row_length_is_checked_with_its_row(self, max_len: float) -> None:
        _per_row_error(
            [_line(ZIGZAG)],
            lambda p, v: p.densify(v),
            max_len,
            "max_segment_length must be a finite",
        )


# ---------------------------------------------------------------------------
# rasterize: lines
# ---------------------------------------------------------------------------


def _stroked(
    contours: list[dict], w: int, h: int, *, supersample: int = 1, **stroke
) -> np.ndarray:
    """``stroke(**stroke)`` of the set, filled by ``rasterize``."""
    pipe = (
        Pipeline()
        .source("contour")
        .stroke(**stroke)
        .rasterize(width=w, height=h, supersample=supersample)
    )
    out = _frame(contours).select(pl.col("c").cv.pipe(pipe).sink("numpy"))
    (mask,) = numpy_from_column(out.to_series(), copy=True)
    assert mask is not None and mask.shape == (h, w, 1)
    return mask[:, :, 0]


class TestStroke:
    @pytest.mark.parametrize("width", [1.0, 2.5, 4.0])
    @pytest.mark.parametrize("n", [1, 3])
    @pytest.mark.parametrize("seed", range(3))
    def test_a_round_stroke_is_a_distance_band(
        self, width: float, n: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        path = rng.uniform(4, 36, (6, 2))
        w, h = 40, 40
        s = _segment_distance(_samples(w, h, n), path, closed=False) - width / 2
        got = _stroked([_line(path)], w, h, width=width, supersample=n)
        _assert_band(got, s, n)

    def test_a_loop_line_is_a_band_round_its_ring(self) -> None:
        loop = [*SQUARE, SQUARE[0]]
        s = _segment_distance(_samples(20, 20, 2), np.array(loop, float), False) - 1.5
        _assert_band(_stroked([_line(loop)], 20, 20, width=3.0, supersample=2), s, 2)

    @pytest.mark.parametrize("cap", ["butt", "round", "square"])
    @pytest.mark.parametrize("join", ["miter", "bevel"])
    def test_a_loop_line_is_joined_at_its_start(self, cap: str, join: str) -> None:
        """A line that returns to its first point has no ends: its start is a
        corner like the others, turned by ``join``, and ``cap`` draws nothing
        (it was stroked as an open path, capping the start corner)."""
        # The square loop starting at each of its corners: every stroke must
        # be the same band, the one starting at a corner's midpoint (no corner
        # at its ends) being the reference.
        x0, y0, x1, y1 = 4, 4, 16, 16
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        reference = [(10, y0), *corners[1:], corners[0], (10, y0)]
        want = _stroked([_line(reference)], 20, 20, width=4.0, cap="butt", join=join)
        for k in range(4):
            loop = [*corners[k:], *corners[:k], corners[k]]
            got = _stroked([_line(loop)], 20, 20, width=4.0, cap=cap, join=join)
            np.testing.assert_array_equal(got, want, err_msg=f"starting at corner {k}")

    def test_a_single_point_with_round_caps_is_a_disc(self) -> None:
        s = np.hypot(*(_samples(12, 12, 2) - (6.2, 5.9)).T) - 2.5
        got = _stroked([_line([(6.2, 5.9)])], 12, 12, width=5.0, supersample=2)
        _assert_band(got, s, 2)

    @pytest.mark.parametrize(("cap", "x0", "x1"), [("butt", 3, 13), ("square", 2, 14)])
    def test_flat_caps_on_the_grid_are_exact(self, cap: str, x0: int, x1: int) -> None:
        got = _stroked([_line([(3, 6), (13, 6)])], 16, 10, width=2.0, cap=cap)
        want = np.zeros((10, 16), np.uint8)
        want[5:7, x0:x1] = 255
        np.testing.assert_array_equal(got, want)

    #: An L of width 4 with butt caps: two 10x4 arms overlapping in a 2x2
    #: square, 76 in all, plus the outer corner's share of the 2x2 square at
    #: [14, 16] x [4, 6] each join fills.
    L_PATH = [(4, 6), (14, 6), (14, 16)]

    @pytest.mark.parametrize(
        ("join", "area"),
        [("miter", 80.0), ("round", 76.0 + math.pi), ("bevel", 78.0)],
    )
    def test_joins_fill_their_closed_form(self, join: str, area: float) -> None:
        got = _stroked(
            [_line(self.L_PATH)], 20, 20,
            width=4.0, cap="butt", join=join, supersample=16,
        )  # fmt: skip
        assert got.sum() / 255 == pytest.approx(area, abs=0.25)

    def test_a_miter_join_on_the_grid_is_exact(self) -> None:
        got = _stroked(
            [_line(self.L_PATH)], 20, 20, width=4.0, cap="butt", join="miter"
        )
        want = np.zeros((20, 20), np.uint8)
        want[4:8, 4:16] = 255
        want[4:16, 12:16] = 255
        np.testing.assert_array_equal(got, want)

    def test_a_regions_rings_are_stroked_as_closed_paths(self) -> None:
        """A region is read as its outline: a band about each ring, exterior
        and hole alike, with no cap anywhere (a ring has no ends)."""
        outer, hole = _rect(3, 3, 17, 17), _rect(7, 7, 12, 12)
        region = _region(outer, [hole])
        n = 2
        samples = _samples(20, 20, n)
        s = (
            np.minimum(
                _segment_distance(samples, np.array(outer, float), closed=True),
                _segment_distance(samples, np.array(hole, float), closed=True),
            )
            - 1.0
        )
        _assert_band(_stroked([region], 20, 20, width=2.0, supersample=n), s, n)

    @pytest.mark.parametrize("join", ["round", "miter", "bevel"])
    def test_a_region_strokes_as_its_boundary(self, join: str) -> None:
        """``boundary()`` draws each ring as a line back to its start, which
        strokes as a closed path: the region's own stroke."""
        region = _region(_star(np.random.default_rng(4), 10.2, 9.8, 7.0))
        via_lines = (
            Pipeline()
            .source("contour")
            .boundary()
            .stroke(width=1.5, join=join)
            .rasterize(width=20, height=20, supersample=4)
        )
        out = _frame([region]).select(pl.col("c").cv.pipe(via_lines).sink("numpy"))
        (want,) = numpy_from_column(out.to_series(), copy=True)
        got = _stroked([region], 20, 20, width=1.5, join=join, supersample=4)
        np.testing.assert_array_equal(got, want[:, :, 0])

    def test_a_mixed_set_strokes_every_member_as_one_union(self) -> None:
        region = _region(_rect(2, 2, 9, 9))
        line = _line([(5, 12), (17, 3)])
        both = _stroked([region, line], 20, 16, width=2.0)
        np.testing.assert_array_equal(
            both,
            np.maximum(
                _stroked([region], 20, 16, width=2.0),
                _stroked([line], 20, 16, width=2.0),
            ),
        )

    def test_the_result_is_disjoint_regions(self) -> None:
        """Two crossing lines stroke to one region: the strokes' union."""
        cross = [_line([(2, 2), (18, 18)]), _line([(2, 18), (18, 2)])]
        regions = _shaped(cross, "stroke", width=2.0)
        assert len(regions) == 1
        assert all(r["is_closed"] for r in regions)

    def test_the_default_draws_a_one_pixel_round_line(self) -> None:
        path = np.array([(2.3, 3.1), (17.6, 11.4)])
        s = _segment_distance(_samples(20, 16, 1), path, closed=False) - 0.5
        _assert_band(_stroked([_line(path)], 20, 16), s, 1)

    def test_rasterize_refuses_a_line_naming_stroke(self) -> None:
        """A line has no region to fill: ``rasterize`` does not guess a width."""
        with pytest.raises(pl.exceptions.ComputeError, match=r"stroke\(\)"):
            _mask([_line(ZIGZAG)], 10, 10)

    @pytest.mark.parametrize("width", [0.0, -1.0])
    def test_a_width_no_row_can_use_is_refused_at_build(self, width: float) -> None:
        with pytest.raises(ValueError, match="width must be a finite"):
            Pipeline().source("contour").stroke(width=width)

    @pytest.mark.parametrize("width", [-2.0, math.nan, math.inf])
    def test_a_per_row_width_is_checked_with_its_row(self, width: float) -> None:
        _per_row_error(
            [_line(ZIGZAG)],
            lambda p, v: p.stroke(width=v),
            width,
            "width must be a finite",
        )


# ---------------------------------------------------------------------------
# One definition, two surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("op", "kwargs"),
    [
        ("offset", {"distance": 1.5, "join": "bevel"}),
        ("union_all", {}),
        ("smooth", {"iterations": 2}),
        ("densify", {"max_segment_length": 1.5}),
        ("stroke", {"width": 2.0, "cap": "square", "join": "bevel"}),
    ],
)
def test_each_accessor_is_its_pipeline_op(op: str, kwargs: dict) -> None:
    regions = [
        _region(_rect(0, 0, 8, 8), [_rect(2, 2, 4, 4)]),
        _region(_rect(6, 6, 12, 12)),
    ]
    df = _frame(regions)
    (pipeline,) = (
        df.select(
            pl.col("c")
            .cv.pipe(getattr(Pipeline().source("contour"), op)(**kwargs))
            .sink("native")
        )
        .to_series()
        .to_list()
    )
    accessor = df.select(getattr(pl.col("c").contour, op)(**kwargs)).item()
    assert pipeline == accessor.to_list()


# ---------------------------------------------------------------------------
# Degenerate rings: every shaping op either answers or refuses, never panics
# ---------------------------------------------------------------------------

#: Rings with too few points to bound anything, as an exterior or as a hole.
DEGENERATE = {
    "empty-exterior": _region([]),
    "empty-hole": _region(_rect(0, 0, 8, 8), [[]]),
    "one-point-exterior": _region([(3, 3)]),
    "two-point-hole": _region(_rect(0, 0, 8, 8), [[(2, 2), (4, 4)]]),
    "empty-line": _line([]),
}

#: Every shaping op, with arguments, as both surfaces spell it.
SHAPING = {
    "offset": {"distance": 1.5},
    "union_all": {},
    "smooth": {"iterations": 2},
    "densify": {"max_segment_length": 1.5},
    "stroke": {"width": 2.0},
}


@pytest.mark.parametrize("shape", sorted(DEGENERATE))
@pytest.mark.parametrize("op", sorted(SHAPING))
def test_a_degenerate_ring_never_panics(op: str, shape: str) -> None:
    """A ring of fewer than three points is accepted by every reader; a
    shaping op on it gives a contour set or the op's own refusal, on both
    surfaces. (``stroke`` panicked inside ``i_overlay`` on an empty ring.)"""
    contour = DEGENERATE[shape]
    if contour["is_closed"] is False and op in ("offset", "union_all"):
        pytest.skip("a line is refused by every region op, tested elsewhere")
    kwargs = SHAPING[op]
    df = _frame([contour])
    pipe = getattr(Pipeline().source("contour"), op)(**kwargs)
    for expr in (
        pl.col("c").cv.pipe(pipe).sink("native"),
        getattr(pl.col("c").contour, op)(**kwargs),
    ):
        try:
            df.select(expr)
        except pl.exceptions.ComputeError as e:
            assert "panicked" not in str(e), e


def test_a_ring_without_an_edge_strokes_to_nothing_a_point_line_to_a_dot() -> None:
    """A closed path has no ends to cap, so with no edge it covers nothing;
    a one-point open line is its round caps, a dot."""
    for ring in ([], [(3, 3)]):
        assert _shaped([_region(ring)], "stroke", width=2.0) == []
    (dot,) = _shaped([_line([(3, 3)])], "stroke", width=2.0)
    assert dot["is_closed"] and len(dot["exterior"]) > 8


def _arc_bound(radius: float) -> float:
    """The documented gap between a round arc and its polygon: 1/64 pixel,
    or ``r * (1 - cos(0.01 pi))`` past ~31.7 px, where ``i_overlay`` caps an
    arc at its finest step (``transforms::arc_tolerance``)."""
    return max(1.0 / 64.0, radius * (1.0 - math.cos(0.01 * math.pi))) + 1e-9


def _square_distance(x: float, y: float, x0: float, x1: float) -> float:
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(x0 - y, 0.0, y - x1)
    return math.hypot(dx, dy)


@pytest.mark.parametrize("distance", [3.0, 20.0, 31.0, 50.0, 200.0, 1000.0])
def test_a_round_offset_keeps_its_documented_arc_bound(distance: float) -> None:
    """Every edge of a round offset of the square [0, 10]^2 lies between the
    true boundary (distance ``d``) and ``d`` less the documented bound: its
    ends sit on the arc, its midpoint is the farthest point inside."""
    (grown,) = _shaped([_region(_rect(0, 0, 10, 10))], "offset", distance=distance)
    ring = [(p["x"], p["y"]) for p in grown["exterior"]]
    worst = 0.0
    for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1], strict=True):
        mid = _square_distance((ax + bx) / 2, (ay + by) / 2, 0.0, 10.0)
        worst = max(worst, distance - mid)
    assert worst <= _arc_bound(distance), f"gap {worst} at r = {distance}"
    # The bound is tight enough to mean something: past the cap it is reached.
    if distance >= 200:
        assert worst > 1.0 / 64.0


@pytest.mark.parametrize("width", [2.0, 64.0, 400.0])
def test_a_round_cap_keeps_its_documented_arc_bound(width: float) -> None:
    (band,) = _shaped([_line([(0, 0), (10, 0)])], "stroke", width=width)
    r = width / 2
    worst = 0.0
    ring = [(p["x"], p["y"]) for p in band["exterior"]]
    for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1], strict=True):
        mx, my = (ax + bx) / 2, (ay + by) / 2
        d = math.hypot(max(-mx, 0.0, mx - 10.0), my)
        worst = max(worst, r - d)
    assert worst <= _arc_bound(r), f"gap {worst} at r = {r}"


#: Many small members, each under the 2^22-point budget, over it together.
_MANY = [_region(_rect(x, 0, x + 10, 10)) for x in range(0, 128 * 12, 12)]


@pytest.mark.parametrize(
    ("op", "kwargs", "asked"),
    [
        ("smooth", {"iterations": 15}, "15 passes"),
        ("densify", {"max_segment_length": 0.001}, "max_segment_length=0.001"),
    ],
)
def test_the_point_budget_holds_over_the_whole_row(op, kwargs, asked) -> None:
    """The budget is the row's: 128 squares at 2^17 or 40 000 points each are
    within it one by one, and over it together (a row of thousands of traced
    contours would otherwise abort on its allocation)."""
    match = rf"{op}: {asked} would make about .* points"
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        _shaped(_MANY, op, **kwargs)
    with pytest.raises(pl.exceptions.ComputeError, match=match):
        _frame(_MANY).select(getattr(pl.col("c").contour, op)(**kwargs))
    # One member alone is within it.
    assert _shaped(_MANY[:1], op, **kwargs)
