"""``on_error`` on the geometry accessors: an invalid row is nulled, not fatal.

The geometry counterpart of ``source(on_error=...)`` and ``Pipeline.on_error``:
under ``"null"`` a row whose data a function refuses (a line too far from the
frame to close, an open contour given to an area) is null and the other rows
proceed. An error about the **column** — a contour set where one contour is
expected, a dtype no reader understands — is no row's fault and still raises,
since nulling it would null every row of a query that cannot work at all.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.geometry import contour_from_coords
from tests.conftest import plugin_required

LINES = [
    [[0.0, 5.0], [10.0, 5.0], [10.0, 0.0]],  # [y, x]: ends on the top and left edges
    [[30.0, 5.0], [10.0, 5.0], [10.0, 0.0]],  # first point 5 px from the frame
]


def _lines() -> pl.DataFrame:
    return pl.DataFrame({"l": LINES}).select(
        contour_from_coords(pl.col("l"), order="yx", closed=False).alias("c")
    )


def test_on_error_refuses_an_unknown_policy() -> None:
    with pytest.raises(ValueError, match="on_error must be one of"):
        pl.col("c").contour.on_error("skip")


def test_on_error_refuses_null_with_message() -> None:
    # An accessor's output is a plain column with nowhere to carry `_error`.
    with pytest.raises(ValueError, match="null_with_message"):
        pl.col("c").contour.on_error("null_with_message")


@plugin_required
class TestGeometryOnError:
    def test_default_raises_naming_the_row(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError, match=r"max_snap.*row 1"):
            _lines().select(pl.col("c").contour.close_along_border(40, 40))

    def test_null_nulls_only_the_invalid_row(self) -> None:
        out = _lines().select(
            pl.col("c")
            .contour.on_error("null")
            .close_along_border(40, 40)
            .contour.area()
        )["c"]
        assert out.to_list() == [50.0, None]

    def test_point_accessor_nulls_an_invalid_row(self) -> None:
        df = pl.DataFrame(
            {"p": [{"x": 1.0, "y": 2.0}] * 2, "w": [2.0, 0.0], "h": [4.0, 4.0]}
        )
        out = df.select(
            pl.col("p").point.on_error("null").normalize(pl.col("w"), pl.col("h"))
        )["p"]
        assert out.to_list() == [{"x": 0.5, "y": 0.5}, None]

    def test_a_column_error_still_raises_under_null(self) -> None:
        # A contour set where one contour per row is expected is the column's
        # shape, not a row's data: nulling it would null every row.
        sq = [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0]]
        df = pl.DataFrame(
            {"c": [sq, sq], "p": [{"x": 1.0, "y": 1.0}] * 2}
        ).with_columns(cs=pl.concat_list(contour_from_coords(pl.col("c"))))
        with pytest.raises(pl.exceptions.ComputeError, match="contour set"):
            df.select(
                pl.col("p").point.on_error("null").distance_to_contour(pl.col("cs"))
            )

    def test_an_unreadable_column_still_raises_under_null(self) -> None:
        df = pl.DataFrame({"c": [{"exterior": [{"x": 1, "y": 2}]}] * 2})
        with pytest.raises(pl.exceptions.ComputeError, match="f64"):
            df.select(pl.col("c").contour.on_error("null").perimeter())

    def test_on_error_and_on_null_compose(self) -> None:
        # Row 0 is valid, row 1 is invalid data, row 2 has a null parameter.
        df = pl.DataFrame(
            {"l": [LINES[0], LINES[1], LINES[0]], "w": [40.0, 40.0, None]}
        ).with_columns(c=contour_from_coords(pl.col("l"), order="yx", closed=False))
        out = df.select(
            pl.col("c")
            .contour.on_null("null")
            .on_error("null")
            .close_along_border(pl.col("w"), 40)
            .contour.area()
        )["c"]
        assert out.to_list() == [50.0, None, None]
