"""Geometry values are read by field name, and a missing value is never 0.0.

Every ``.point``/``.contour``/``.bbox`` function reads its operands through one
reader per geometry type (``src/geom_columns.rs``). These pin, at the user-facing
entry points, what the hand-written parsers they replaced got wrong:
``contains_point`` read a point's first two fields by position and turned a
missing or non-float coordinate into 0.0; a bbox with a null field read it as
0.0; ``rotate`` rotated about ``(0, 0)`` when its ``origin`` was null.
"""

from __future__ import annotations

import polars as pl
import pytest

from tests.conftest import plugin_required

pytestmark = plugin_required

SQUARE = {
    "exterior": [
        {"x": 0.0, "y": 0.0},
        {"x": 10.0, "y": 0.0},
        {"x": 10.0, "y": 10.0},
        {"x": 0.0, "y": 10.0},
    ],
    "holes": [],
    "is_closed": True,
}


def test_contains_point_reads_the_point_by_field_name() -> None:
    # A tall rectangle, so swapping x and y moves a point in or out of it:
    # (x=5, y=50) is inside, (x=50, y=5) is not. Spelled `{y, x}`, the point
    # used to be read by position as (50, 5).
    tall = {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 10.0, "y": 0.0},
            {"x": 10.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    df = pl.DataFrame({"c": [tall], "p": [{"y": 50.0, "x": 5.0}]})
    assert df.schema["p"] == pl.Struct({"y": pl.Float64, "x": pl.Float64})
    out = df.select(pl.col("c").contour.contains_point(pl.col("p")))["c"].to_list()
    assert out == [True]


def test_integer_coordinates_are_refused_not_read_as_the_origin() -> None:
    df = pl.DataFrame({"c": [SQUARE], "p": [{"x": 50, "y": 50}]})
    # Read as (0, 0) this answered True: the origin is on the square.
    with pytest.raises(pl.exceptions.ComputeError, match="f64"):
        df.select(pl.col("c").contour.contains_point(pl.col("p")))


@pytest.mark.parametrize(
    "call",
    [
        lambda: pl.col("p").point.translate(1.0, 1.0),
        lambda: pl.col("p").point.distance(pl.col("q")),
        lambda: pl.col("c").contour.contains_point(pl.col("p")),
    ],
)
def test_a_null_point_coordinate_is_refused(call) -> None:  # noqa: ANN001
    df = pl.DataFrame(
        {
            "c": [SQUARE],
            "p": [{"x": None, "y": 1.0}],
            "q": [{"x": 0.0, "y": 0.0}],
        },
        schema={
            "c": pl.Struct(
                {
                    "exterior": pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64})),
                    "holes": pl.List(
                        pl.List(pl.Struct({"x": pl.Float64, "y": pl.Float64}))
                    ),
                    "is_closed": pl.Boolean,
                }
            ),
            "p": pl.Struct({"x": pl.Float64, "y": pl.Float64}),
            "q": pl.Struct({"x": pl.Float64, "y": pl.Float64}),
        },
    )
    with pytest.raises(pl.exceptions.ComputeError, match="null x"):
        df.select(call())


def test_a_null_point_row_is_a_null_result() -> None:
    df = pl.DataFrame(
        {"p": [{"x": 1.0, "y": 2.0}, None]},
        schema={"p": pl.Struct({"x": pl.Float64, "y": pl.Float64})},
    )
    out = df.select(pl.col("p").point.translate(1.0, 1.0))["p"].to_list()
    assert out == [{"x": 2.0, "y": 3.0}, None]


def test_rotate_with_a_null_origin_is_null() -> None:
    df = pl.DataFrame(
        {
            "p": [{"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
            "o": [{"x": 1.0, "y": 1.0}, None],
        },
        schema={
            "p": pl.Struct({"x": pl.Float64, "y": pl.Float64}),
            "o": pl.Struct({"x": pl.Float64, "y": pl.Float64}),
        },
    )
    out = df.select(pl.col("p").point.rotate(0.0, origin=pl.col("o")))["p"].to_list()
    # A zero rotation about any origin leaves the point; a null origin is no
    # origin at all, not (0, 0).
    assert out == [{"x": 1.0, "y": 0.0}, None]


def test_a_bbox_with_a_null_field_is_refused() -> None:
    bbox = pl.Struct(
        {"x": pl.Float64, "y": pl.Float64, "width": pl.Float64, "height": pl.Float64}
    )
    df = pl.DataFrame(
        {
            "p": [{"x": 1.0, "y": 1.0}],
            "b": [{"x": 0.0, "y": 0.0, "width": None, "height": 5.0}],
        },
        schema={"p": pl.Struct({"x": pl.Float64, "y": pl.Float64}), "b": bbox},
    )
    with pytest.raises(pl.exceptions.ComputeError, match="null width"):
        df.select(pl.col("p").point.within_bbox(pl.col("b")))


# --- A length-1 operand broadcasts --------------------------------------------
#
# A literal operand (or one the optimiser folds back into a literal) reaches
# the plugin as a one-row series. Every reader broadcasts it to every row, as
# `ParamCol` does for a per-row parameter; it used to fail with "geometry row 1
# out of bounds" for every accessor, whatever the arity of the other side.

_POINT = pl.Struct({"x": pl.Float64, "y": pl.Float64})
_LIT_BBOX = pl.struct(
    pl.lit(0.0).alias("x"),
    pl.lit(0.0).alias("y"),
    pl.lit(10.0).alias("width"),
    pl.lit(10.0).alias("height"),
)
_LIT_POINT = pl.struct(pl.lit(5.0).alias("x"), pl.lit(5.0).alias("y"))


def _lit_square() -> pl.Expr:
    from polars_cv.geometry import contour_from_coords

    return contour_from_coords(
        pl.lit([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]])
    )


def _frame() -> pl.DataFrame:
    from polars_cv.geometry import contour_from_coords

    far = [[50.0, 50.0], [50.0, 60.0], [60.0, 60.0], [60.0, 50.0]]
    df = pl.DataFrame(
        {
            "p": [{"x": 5.0, "y": 5.0}, {"x": 50.0, "y": 5.0}],
            "ps": [[{"x": 5.0, "y": 5.0}], [{"x": 50.0, "y": 5.0}]],
            "c": [[[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]], far],
        },
        schema={
            "p": _POINT,
            "ps": pl.List(_POINT),
            "c": pl.List(pl.List(pl.Float64)),
        },
    )
    c = contour_from_coords(pl.col("c"))
    return df.with_columns(c=c, cs=pl.concat_list(c))


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda: pl.col("p").point.within_bbox(_LIT_BBOX), [True, False]),
        (lambda: pl.col("ps").point.within_bbox(_LIT_BBOX), [[True], [False]]),
        (lambda: pl.col("p").point.distance(_LIT_POINT), [0.0, 45.0]),
        (lambda: pl.col("ps").point.distance(_LIT_POINT), [[0.0], [45.0]]),
        (
            lambda: pl.col("ps").point.distance_to_contour(_lit_square()),
            [[5.0], [40.0]],
        ),
        (lambda: pl.col("c").contour.iou(_lit_square()), [1.0, 0.0]),
        (lambda: pl.col("cs").contour.iou(_lit_square()), [[1.0], [0.0]]),
        (lambda: pl.col("cs").contour.contains_point(_LIT_POINT), [[True], [False]]),
    ],
    ids=[
        "point-bbox",
        "point_set-bbox",
        "point-point",
        "point_set-point",
        "point_set-contour",
        "contour-contour",
        "contour_set-contour",
        "contour_set-point",
    ],
)
def test_a_length_one_operand_broadcasts(call, expected) -> None:  # noqa: ANN001
    out = _frame().select(call().alias("r"))["r"].to_list()
    assert out == expected


def test_a_length_one_order_broadcasts() -> None:
    # `correspond`'s `order` is a data operand read outside the geometry
    # readers; a literal one is every row's order, as a full column would be.
    df = _frame()
    gt = pl.concat_list(_lit_square())
    literal = pl.lit([0], dtype=pl.List(pl.UInt32))
    column = pl.col("p").map_batches(
        lambda s: pl.Series([[0]] * len(s), dtype=pl.List(pl.UInt32)),
        return_dtype=pl.List(pl.UInt32),
    )
    got = df.select(pl.col("cs").contour.correspond(gt, order=literal).alias("r"))
    want = df.select(pl.col("cs").contour.correspond(gt, order=column).alias("r"))
    assert got.to_dicts() == want.to_dicts()
