"""Points and contours from plain coordinate lists, and back.

Geometry is often stored as ``[y, x]`` (row, column) or ``[x, y]`` pairs and
nested lists of rings; getting it into ``POINT_SCHEMA``/``CONTOUR_SCHEMA`` used
to mean hand-building the struct, holes and ``is_closed`` included, each time.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import CONTOUR_SCHEMA, CONTOUR_SET_SCHEMA, POINT_SCHEMA
from polars_cv.geometry import (
    contour_from_coords,
    contour_set_from_coords,
    point_from_coords,
)

from .conftest import plugin_required

_L_YX = [[0, 0], [0, 10], [10, 10]]  # (row, col): x 0 -> 10, then y 0 -> 10
_L_XY = [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 0.0}, {"x": 10.0, "y": 10.0}]


@plugin_required
class TestPointFromCoords:
    def test_yx_and_xy(self) -> None:
        df = pl.DataFrame({"p": [[2.0, 1.0], None]})
        assert df.select(
            point_from_coords(pl.col("p"), order="yx")
        ).to_series().to_list() == [
            {"x": 1.0, "y": 2.0},
            None,
        ]
        out = df.select(point_from_coords(pl.col("p"), order="xy"))
        assert out.schema["p"] == POINT_SCHEMA
        assert out.item(0, 0) == {"x": 2.0, "y": 1.0}

    def test_integer_and_array_coordinates(self) -> None:
        ints = pl.DataFrame({"p": [[3, 4]]})
        assert ints.select(point_from_coords(pl.col("p"))).item() == {
            "x": 3.0,
            "y": 4.0,
        }
        arr = pl.DataFrame({"p": [[3.0, 4.0]]}, schema={"p": pl.Array(pl.Float64, 2)})
        assert arr.select(point_from_coords(pl.col("p"))).item() == {"x": 3.0, "y": 4.0}

    @pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0]])
    def test_a_pair_of_the_wrong_length_is_refused(self, bad: list[float]) -> None:
        # The row the plugin names is its index within the batch polars hands
        # it, which the engine may split differently by version: pin the
        # message, not a row number.
        df = pl.DataFrame({"p": [[1.0, 2.0], bad]})
        with pytest.raises(
            pl.exceptions.ComputeError, match=r"exactly 2 numbers.*\(row \d+\)"
        ):
            df.select(point_from_coords(pl.col("p")))

    def test_a_null_coordinate_is_refused(self) -> None:
        df = pl.DataFrame({"p": [[1.0, None]]})
        with pytest.raises(pl.exceptions.ComputeError, match="null"):
            df.select(point_from_coords(pl.col("p")))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_a_non_finite_coordinate_is_refused(self, bad: float) -> None:
        # Every geometry reader refuses a non-finite coordinate, so building
        # one only moved the error to whichever function read it first.
        df = pl.DataFrame({"p": [[1.0, bad]]})
        with pytest.raises(pl.exceptions.ComputeError, match="non-finite"):
            df.select(point_from_coords(pl.col("p")))
        rings = pl.DataFrame({"c": [[[0.0, 0.0], [bad, 1.0], [1.0, 1.0]]]})
        with pytest.raises(pl.exceptions.ComputeError, match="non-finite"):
            rings.select(contour_from_coords(pl.col("c")))

    def test_an_unknown_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="order"):
            point_from_coords(pl.col("p"), order="zx")


@plugin_required
class TestContourFromCoords:
    def test_a_closed_contour_from_yx_pairs(self) -> None:
        df = pl.DataFrame({"c": [_L_YX]})
        out = df.select(contour_from_coords(pl.col("c"), order="yx"))
        assert out.schema["c"] == CONTOUR_SCHEMA
        assert out.item() == {"exterior": _L_XY, "holes": [], "is_closed": True}

    def test_an_open_polyline(self) -> None:
        df = pl.DataFrame({"c": [_L_YX]})
        line = df.select(contour_from_coords(pl.col("c"), order="yx", closed=False))
        assert line.item()["is_closed"] is False
        assert line.select(pl.col("c").contour.perimeter()).item() == pytest.approx(
            20.0
        )

    def test_a_contour_set(self) -> None:
        df = pl.DataFrame({"c": [[_L_YX, _L_YX[:2]], [], None]})
        out = df.select(contour_set_from_coords(pl.col("c"), order="yx"))
        assert out.schema["c"] == CONTOUR_SET_SCHEMA
        rows = out.to_series().to_list()
        assert [len(r) if r is not None else None for r in rows] == [2, 0, None]
        assert rows[0][1]["exterior"] == _L_XY[:2]


@plugin_required
class TestToCoords:
    def test_a_point_round_trips(self) -> None:
        df = pl.DataFrame(
            {"p": [{"x": 1.0, "y": 2.0}, None]}, schema={"p": POINT_SCHEMA}
        )
        out = df.select(pl.col("p").point.to_coords(order="yx"))
        assert out.schema["p"] == pl.Array(pl.Float64, 2)
        assert out.to_series().to_list() == [[2.0, 1.0], None]
        back = out.select(point_from_coords(pl.col("p"), order="yx"))
        assert back.to_series().to_list() == df.to_series().to_list()

    def test_a_contour_and_a_set_round_trip(self) -> None:
        df = pl.DataFrame({"c": [_L_YX]})
        contour = df.select(contour_from_coords(pl.col("c"), order="yx"))
        coords = contour.select(pl.col("c").contour.to_coords(order="yx"))
        assert coords.schema["c"] == pl.List(pl.Array(pl.Float64, 2))
        assert coords.item().to_list() == [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
        sets = pl.DataFrame({"c": [[_L_YX]]}).select(
            contour_set_from_coords(pl.col("c"), order="xy")
        )
        assert sets.select(pl.col("c").contour.to_coords()).schema["c"] == pl.List(
            pl.List(pl.Array(pl.Float64, 2))
        )

    def test_a_point_set(self) -> None:
        df = pl.DataFrame(
            {"p": [[{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}]]},
            schema={"p": pl.List(POINT_SCHEMA)},
        )
        out = df.select(pl.col("p").point.to_coords())
        assert out.item().to_list() == [[1.0, 2.0], [3.0, 4.0]]

    def test_a_contour_with_holes_is_refused(self) -> None:
        square = [{"x": 0.0, "y": 0.0}, {"x": 9.0, "y": 0.0}, {"x": 9.0, "y": 9.0}]
        df = pl.DataFrame(
            {"c": [{"exterior": square, "holes": [square], "is_closed": True}]},
            schema={"c": CONTOUR_SCHEMA},
        )
        with pytest.raises(pl.exceptions.ComputeError, match="holes"):
            df.select(pl.col("c").contour.to_coords())
