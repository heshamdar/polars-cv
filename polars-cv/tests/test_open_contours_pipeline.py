"""Open contours (lines) in the pipeline contour domain; ``boundary()`` and
``close()`` (CR-97).

A contour row with ``is_closed = false`` is a polyline: edges join
consecutive points only and it bounds no region. Pipelines carry lines as the
``.contour`` accessors always have, with one definition per op for both
surfaces, so each op is checked against its accessor on the same column, and
the lengths and areas against NumPy.

* Boundary ops take lines: ``perimeter`` (a line's length, no closing edge),
  ``bounding_box``, ``translate``, ``scale`` (not about a centroid),
  ``simplify``, and ``convex_hull`` (whose hull is a region).
* Region ops refuse them: ``area``, ``centroid``, ``largest``, ``rasterize``.
* ``boundary()`` turns a region's rings into lines (a set: holes add rings);
  ``close()`` joins a line's ends into a region.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA
from tests.conftest import plugin_required

pytestmark = plugin_required


def _pts(points) -> list[dict]:
    return [{"x": float(x), "y": float(y)} for x, y in points]


def _line(points) -> dict:
    return {"exterior": _pts(points), "holes": [], "is_closed": False}


def _region(points, holes=()) -> dict:
    return {
        "exterior": _pts(points),
        "holes": [_pts(h) for h in holes],
        "is_closed": True,
    }


ZIGZAG = [(1, 1), (5, 2), (3, 7), (9, 8)]
SQUARE = [(2, 2), (12, 2), (12, 12), (2, 12)]
HOLE = [(5, 5), (8, 5), (8, 8), (5, 8)]


def _frame(*sets) -> pl.DataFrame:
    return pl.DataFrame({"c": list(sets)}, schema={"c": CONTOUR_SET_SCHEMA})


def _pipe(df: pl.DataFrame, pipe: Pipeline, sink: str = "native"):
    return df.select(pl.col("c").cv.pipe(pipe).sink(sink)).to_series().to_list()


def _length(points) -> float:
    p = np.asarray(points, dtype=float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def _shoelace(points) -> float:
    p = np.asarray(points, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def test_a_line_passes_through_the_pipeline_still_open() -> None:
    out = _pipe(_frame([_line(ZIGZAG), _region(SQUARE)]), Pipeline().source("contour"))
    (row,) = out
    assert [c["is_closed"] for c in row] == [False, True]
    assert [(p["x"], p["y"]) for p in row[0]["exterior"]] == [
        (float(x), float(y)) for x, y in ZIGZAG
    ]


def test_extracted_contours_are_regions() -> None:
    mask = np.zeros((10, 10), np.uint8)
    mask[2:6, 3:8] = 255
    df = pl.DataFrame({"m": [mask]}, schema={"m": pl.Array(pl.UInt8, (10, 10))})
    out = df.select(
        pl.col("m")
        .cv.pipe(Pipeline().source("array").extract_contours())
        .sink("native")
    ).item()
    assert [c["is_closed"] for c in out] == [True]


def test_perimeter_of_a_line_is_its_length() -> None:
    (lengths,) = _pipe(
        _frame([_line(ZIGZAG), _region(SQUARE)]),
        Pipeline().source("contour").perimeter(),
    )
    assert lengths == pytest.approx([_length(ZIGZAG), 40.0])


@pytest.mark.parametrize(
    ("op", "kwargs"),
    [
        ("perimeter", {}),
        ("bounding_box", {}),
        ("translate", {"dx": 1.5, "dy": -2.0}),
        ("simplify", {"tolerance": 1.0}),
        ("convex_hull", {}),
    ],
)
def test_each_boundary_op_agrees_with_its_accessor(op: str, kwargs: dict) -> None:
    """One definition, two surfaces: the pipeline op and ``.contour`` give the
    same answer on a set mixing a line and a region."""
    df = _frame([_line(ZIGZAG), _region(SQUARE, [HOLE])])
    (pipeline,) = _pipe(df, getattr(Pipeline().source("contour"), op)(**kwargs))
    accessor = df.select(getattr(pl.col("c").contour, op)(**kwargs)).item()
    if isinstance(accessor, pl.Series):
        accessor = accessor.to_list()
    if op in ("perimeter",):
        assert pipeline == pytest.approx(accessor)
    elif op == "bounding_box":
        flat = [v for b in accessor for v in (b["x"], b["y"], b["width"], b["height"])]
        assert pipeline == pytest.approx(flat)
    else:
        assert pipeline == accessor


def test_scale_keeps_a_line_open_and_refuses_its_centroid() -> None:
    df = _frame([_line(ZIGZAG)])
    (row,) = _pipe(
        df, Pipeline().source("contour").scale_contour(sx=2.0, sy=2.0, origin="origin")
    )
    assert row[0]["is_closed"] is False
    assert row[0]["exterior"][1] == {"x": 10.0, "y": 4.0}
    with pytest.raises(pl.exceptions.ComputeError, match="centroid"):
        _pipe(df, Pipeline().source("contour").scale_contour(sx=2.0, sy=2.0))


def test_the_convex_hull_of_a_line_is_a_region() -> None:
    (row,) = _pipe(_frame([_line(ZIGZAG)]), Pipeline().source("contour").convex_hull())
    assert row[0]["is_closed"] is True


@pytest.mark.parametrize(
    ("build", "sink"),
    [
        (lambda p: p.area(), "native"),
        (lambda p: p.centroid(), "native"),
        (lambda p: p.largest(), "native"),
        (lambda p: p.rasterize(width=16, height=16), "numpy"),
    ],
    ids=["area", "centroid", "largest", "rasterize"],
)
def test_a_region_op_refuses_a_line(build, sink: str) -> None:
    df = _frame([_region(SQUARE), _line(ZIGZAG)])
    with pytest.raises(
        pl.exceptions.ComputeError, match="open contour.*no region.*close"
    ):
        _pipe(df, build(Pipeline().source("contour")), sink=sink)


def test_boundary_gives_each_ring_as_a_closed_loop_line() -> None:
    (lines,) = _pipe(
        _frame([_region(SQUARE, [HOLE])]), Pipeline().source("contour").boundary()
    )
    assert [c["is_closed"] for c in lines] == [False, False]
    for line, ring in zip(lines, (SQUARE, HOLE), strict=True):
        points = [(p["x"], p["y"]) for p in line["exterior"]]
        assert points == [(float(x), float(y)) for x, y in [*ring, ring[0]]]
    # The lines' lengths add up to the region's perimeter.
    (lengths,) = _pipe(
        _frame([_region(SQUARE, [HOLE])]),
        Pipeline().source("contour").boundary().perimeter(),
    )
    assert sum(lengths) == pytest.approx(40.0 + 12.0)


def test_boundary_refuses_a_line() -> None:
    with pytest.raises(
        pl.exceptions.ComputeError, match="boundary.*already.*line|open"
    ):
        _pipe(_frame([_line(ZIGZAG)]), Pipeline().source("contour").boundary())


def test_close_joins_a_lines_ends_into_a_region() -> None:
    (row,) = _pipe(_frame([_line(ZIGZAG)]), Pipeline().source("contour").close())
    assert row[0]["is_closed"] is True
    (areas,) = _pipe(
        _frame([_line(ZIGZAG)]), Pipeline().source("contour").close().area()
    )
    assert areas == pytest.approx([_shoelace(ZIGZAG)])


def test_close_undoes_boundary() -> None:
    df = _frame([_region(SQUARE)])
    (back,) = _pipe(df, Pipeline().source("contour").boundary().close())
    (original,) = _pipe(df, Pipeline().source("contour"))
    assert back == original


def test_close_refuses_a_region() -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="close.*already closed"):
        _pipe(_frame([_region(SQUARE)]), Pipeline().source("contour").close())


@pytest.mark.parametrize("op", ["boundary", "close"])
def test_the_accessors_are_the_same_ops(op: str) -> None:
    source = _region(SQUARE, [HOLE]) if op == "boundary" else _line(ZIGZAG)
    df = _frame([source])
    (pipeline,) = _pipe(df, getattr(Pipeline().source("contour"), op)())
    accessor = df.select(getattr(pl.col("c").contour, op)()).item()
    assert pipeline == (
        accessor.to_list() if isinstance(accessor, pl.Series) else accessor
    )
