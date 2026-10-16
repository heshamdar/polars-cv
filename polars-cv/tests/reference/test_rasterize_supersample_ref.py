"""Reference tests for ``rasterize(supersample=n)`` (CR-96).

Each pixel samples the region at its ``n x n`` sub-pixel centres
``(x + (i + 0.5) / n, y + (j + 0.5) / n)`` and is painted
``background + (fill_value - background) * k / n**2`` for its ``k`` covered
samples, rounded half away from zero. ``n = 1`` is the centre rule
``rasterize`` has always used, byte for byte.

Oracle: matplotlib's ``Path.contains_points`` on the same sample positions,
an independent point-in-polygon test. Random float vertices put no sample on
an edge, where the two could legitimately differ, so the comparison is exact.
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


def _ring(points) -> list[dict]:
    return [{"x": float(x), "y": float(y)} for x, y in points]


def _contour(exterior, holes=()) -> dict:
    return {
        "exterior": _ring(exterior),
        "holes": [_ring(h) for h in holes],
        "is_closed": True,
    }


def _run(contours, w, h, **kw) -> np.ndarray:
    df = pl.DataFrame({"c": [contours]}, schema={"c": CONTOUR_SET_SCHEMA})
    out = df.select(
        pl.col("c")
        .cv.pipe(Pipeline().source("contour").rasterize(width=w, height=h, **kw))
        .sink("numpy")
    )
    (mask,) = numpy_from_column(out.to_series(), copy=True)
    assert mask is not None and mask.shape == (h, w, 1)
    return mask[:, :, 0]


def _oracle(contours, w, h, n, fill=255, background=0) -> np.ndarray:
    """Union of exterior-minus-holes, sampled at every sub-pixel centre."""
    offsets = (np.arange(n) + 0.5) / n
    xs = (np.arange(w)[:, None] + offsets[None, :]).ravel()
    ys = (np.arange(h)[:, None] + offsets[None, :]).ravel()
    gx, gy = np.meshgrid(xs, ys)
    points = np.column_stack([gx.ravel(), gy.ravel()])
    covered = np.zeros(len(points), dtype=bool)
    for c in contours:
        inside = Path([(p["x"], p["y"]) for p in c["exterior"]]).contains_points(points)
        for hole in c["holes"]:
            inside &= ~Path([(p["x"], p["y"]) for p in hole]).contains_points(points)
        covered |= inside
    counts = covered.reshape(h, n, w, n).sum(axis=(1, 3))
    value = background + (fill - background) * counts / (n * n)
    # Round half away from zero, the engine's store rule.
    return (np.sign(value) * np.floor(np.abs(value) + 0.5)).astype(np.uint8)


def _random_polygon(rng, cx, cy, r, k=9) -> list[tuple[float, float]]:
    """A star-shaped polygon with float vertices (no sample on an edge)."""
    angles = np.sort(rng.uniform(0, 2 * math.pi, k))
    radii = rng.uniform(0.35 * r, r, k)
    return list(
        zip(cx + radii * np.cos(angles), cy + radii * np.sin(angles), strict=True)
    )


def _clearance(contours, h, n) -> float:
    """The smallest horizontal distance from a sample to an edge crossing its
    row, over every sample row: where it is ~0 the two tests may disagree
    through rounding alone, so a fixture must keep clear of it."""
    best = math.inf
    rows = (np.arange(h)[:, None] + (np.arange(n) + 0.5) / n).ravel()
    for c in contours:
        for ring in [c["exterior"], *c["holes"]]:
            pts = [(p["x"], p["y"]) for p in ring]
            for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1], strict=True):
                crossing = ((y1 <= rows) & (rows < y2)) | ((y2 <= rows) & (rows < y1))
                sy = rows[crossing]
                cx = x1 + (sy - y1) / (y2 - y1) * (x2 - x1)
                nearest = (np.round(cx * n - 0.5) + 0.5) / n
                if cx.size:
                    best = min(best, float(np.abs(nearest - cx).min()))
                best = min(best, float(np.abs(rows - y1).min()))
    return best


@pytest.mark.parametrize("n", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("seed", range(4))
def test_coverage_matches_matplotlib_sampling(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    w, h = 37, 29
    contours = [
        _contour(_random_polygon(rng, 12.3, 13.1, 10.7)),
        _contour(
            _random_polygon(rng, 25.6, 15.2, 11.9),
            holes=[_random_polygon(rng, 25.6, 15.2, 4.1, k=6)],
        ),
    ]
    assert _clearance(contours, h, n) > 1e-7, "a sample sits on an edge: reseed"
    np.testing.assert_array_equal(
        _run(contours, w, h, supersample=n), _oracle(contours, w, h, n)
    )


def test_one_sample_is_the_centre_rule_byte_for_byte() -> None:
    rng = np.random.default_rng(7)
    contours = [_contour(_random_polygon(rng, 15.5, 15.5, 12.0))]
    np.testing.assert_array_equal(
        _run(contours, 31, 31, supersample=1), _run(contours, 31, 31)
    )


def test_an_axis_aligned_edge_covers_its_exact_fraction() -> None:
    # Left edge at x = 2.25, right at 5.75, rows 1 to 3 whole.
    rect = _contour([(2.25, 1.0), (5.75, 1.0), (5.75, 3.0), (2.25, 3.0)])
    got = _run([rect], 8, 4, supersample=4)
    # Column 2 holds 3 of 4 sample columns, column 5 holds 3 of 4 too.
    row = [0, 0, 191, 255, 255, 191, 0, 0]
    np.testing.assert_array_equal(got, np.array([[0] * 8, row, row, [0] * 8], np.uint8))


def test_inverted_values_paint_the_same_coverage() -> None:
    rect = _contour([(1.5, 0.0), (3.0, 0.0), (3.0, 1.0), (1.5, 1.0)])
    got = _run([rect], 4, 1, supersample=2, fill_value=0, background=200)
    np.testing.assert_array_equal(got, np.array([[200, 100, 0, 200]], np.uint8))


@pytest.mark.parametrize("n", [0, 17])
def test_a_sample_count_no_row_can_use_is_refused(n: int) -> None:
    with pytest.raises(ValueError, match=r"supersample.*between 1 and 16"):
        Pipeline().source("contour").rasterize(width=4, height=4, supersample=n)


def test_a_per_row_sample_count() -> None:
    rect = _contour([(0.5, 0.0), (2.0, 0.0), (2.0, 1.0), (0.5, 1.0)])
    df = pl.DataFrame(
        {"c": [[rect], [rect]], "n": [1, 2]},
        schema={"c": CONTOUR_SET_SCHEMA, "n": pl.UInt32},
    )
    pipe = (
        Pipeline()
        .source("contour")
        .rasterize(width=2, height=1, supersample=pl.col("n"))
    )
    out = df.select(pl.col("c").cv.pipe(pipe).sink("list")).to_series().to_list()
    assert out == [[[[255], [255]]], [[[128], [255]]]]
