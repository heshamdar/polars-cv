"""Per-row expression parameters in the geometry namespaces.

The `.contour`, `.point` and `.bbox` namespaces bypass the `vb_graph` graph
engine, so they carry per-row parameters as extra plugin *input series* rather
than through `ParamValue` (see `_ArgBinder` in `_namespace.py` and `GeomParams`
in `src/geom_params.rs`).

These operations used to raise `TypeError` on a `pl.Expr` — a deliberate guard
added after a bug where expression arguments were silently *dropped*, i.e.
accepted and then ignored. Now that the capability exists, the important thing
to prove is not that a call succeeds but that the value genuinely varies per
row: every test below feeds two rows with different parameter values and
asserts the two outputs differ. A regression to silent-drop would make them
equal and fail here.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

from tests.conftest import plugin_required

# A CCW unit square, 100x100 at the origin.
SQUARE = {
    "exterior": [
        {"x": 0.0, "y": 0.0},
        {"x": 100.0, "y": 0.0},
        {"x": 100.0, "y": 100.0},
        {"x": 0.0, "y": 100.0},
    ],
    "holes": [],
    "is_closed": True,
}

# Same square wound clockwise, so its *signed* area is negative.
SQUARE_CW = {
    "exterior": [
        {"x": 0.0, "y": 0.0},
        {"x": 0.0, "y": 100.0},
        {"x": 100.0, "y": 100.0},
        {"x": 100.0, "y": 0.0},
    ],
    "holes": [],
    "is_closed": True,
}

# A near-collinear vertex that a large simplify tolerance removes.
JAGGED = {
    "exterior": [
        {"x": 0.0, "y": 0.0},
        {"x": 50.0, "y": 0.4},
        {"x": 100.0, "y": 0.0},
        {"x": 100.0, "y": 100.0},
        {"x": 0.0, "y": 100.0},
    ],
    "holes": [],
    "is_closed": True,
}


def _two_rows(**columns: list) -> pl.DataFrame:
    """Build a two-row frame; every column must have exactly two values."""
    assert all(len(v) == 2 for v in columns.values())
    return pl.DataFrame(columns)


@plugin_required
class TestContourExpressionParams:
    """`.contour` parameters resolve per row."""

    def test_normalize_uses_per_row_dimensions(self) -> None:
        df = _two_rows(c=[SQUARE, SQUARE], w=[100.0, 200.0], h=[100.0, 50.0])
        out = df.with_columns(
            n=pl.col("c").contour.normalize(pl.col("w"), pl.col("h"))
        )["n"].to_list()

        # Corner (100, 100) divided by each row's own (w, h).
        assert out[0]["exterior"][2] == {"x": 1.0, "y": 1.0}
        assert out[1]["exterior"][2] == {"x": 0.5, "y": 2.0}

    def test_normalize_literal_still_works(self) -> None:
        df = _two_rows(c=[SQUARE, SQUARE])
        out = df.with_columns(n=pl.col("c").contour.normalize(100, 100))["n"].to_list()
        assert out[0]["exterior"][2] == {"x": 1.0, "y": 1.0}

    def test_normalize_mixed_literal_and_expression(self) -> None:
        df = _two_rows(c=[SQUARE, SQUARE], h=[100.0, 50.0])
        out = df.with_columns(n=pl.col("c").contour.normalize(100, pl.col("h")))[
            "n"
        ].to_list()
        assert out[0]["exterior"][2]["y"] == 1.0
        assert out[1]["exterior"][2]["y"] == 2.0

    def test_to_absolute_uses_per_row_dimensions(self) -> None:
        df = _two_rows(c=[SQUARE, SQUARE], w=[1.0, 2.0], h=[1.0, 3.0])
        out = df.with_columns(
            a=pl.col("c").contour.to_absolute(pl.col("w"), pl.col("h"))
        )["a"].to_list()
        assert out[0]["exterior"][2] == {"x": 100.0, "y": 100.0}
        assert out[1]["exterior"][2] == {"x": 200.0, "y": 300.0}

    def test_translate_uses_per_row_offsets(self) -> None:
        df = _two_rows(c=[SQUARE, SQUARE], dx=[5.0, -5.0], dy=[1.0, 2.0])
        out = df.with_columns(
            t=pl.col("c").contour.translate(pl.col("dx"), pl.col("dy"))
        )["t"].to_list()
        assert out[0]["exterior"][0] == {"x": 5.0, "y": 1.0}
        assert out[1]["exterior"][0] == {"x": -5.0, "y": 2.0}

    def test_scale_uses_per_row_factors(self) -> None:
        df = _two_rows(c=[SQUARE, SQUARE], sx=[2.0, 3.0], sy=[1.0, 1.0])
        out = df.with_columns(s=pl.col("c").contour.scale(pl.col("sx"), pl.col("sy")))[
            "s"
        ].to_list()
        assert out[0]["exterior"][2]["x"] == 200.0
        assert out[1]["exterior"][2]["x"] == 300.0

    def test_simplify_uses_per_row_tolerance(self) -> None:
        df = _two_rows(c=[JAGGED, JAGGED], tol=[0.1, 3.0])
        out = df.with_columns(s=pl.col("c").contour.simplify(pl.col("tol")))[
            "s"
        ].to_list()
        # The tight tolerance keeps the near-collinear vertex; the loose one drops it.
        assert len(out[0]["exterior"]) == 5
        assert len(out[1]["exterior"]) == 4

    def test_area_signed_flag_is_per_row(self) -> None:
        df = _two_rows(c=[SQUARE_CW, SQUARE_CW], signed=[True, False])
        out = df.with_columns(a=pl.col("c").contour.area(signed=pl.col("signed")))[
            "a"
        ].to_list()
        assert out[0] == -10000.0  # signed: CW winding is negative
        assert out[1] == 10000.0  # unsigned: absolute area

    def test_scale_origin_stays_literal(self) -> None:
        """`origin` selects behaviour rather than carrying a value."""
        df = _two_rows(c=[SQUARE, SQUARE], sx=[2.0, 2.0], sy=[2.0, 2.0])
        out = df.with_columns(
            s=pl.col("c").contour.scale(pl.col("sx"), pl.col("sy"), origin="centroid")
        )["s"].to_list()
        # Scaling 2x about the centroid (50, 50) maps (0, 0) to (-50, -50).
        assert out[0]["exterior"][0] == {"x": -50.0, "y": -50.0}


@plugin_required
class TestPointExpressionParams:
    """`.point` parameters resolve per row."""

    def test_normalize_uses_per_row_dimensions(self) -> None:
        df = _two_rows(p=[{"x": 50.0, "y": 25.0}] * 2, w=[100.0, 50.0], h=[100.0, 25.0])
        out = df.with_columns(n=pl.col("p").point.normalize(pl.col("w"), pl.col("h")))[
            "n"
        ].to_list()
        assert out[0] == {"x": 0.5, "y": 0.25}
        assert out[1] == {"x": 1.0, "y": 1.0}

    def test_to_absolute_uses_per_row_dimensions(self) -> None:
        df = _two_rows(p=[{"x": 0.5, "y": 0.5}] * 2, w=[100.0, 200.0], h=[10.0, 20.0])
        out = df.with_columns(
            a=pl.col("p").point.to_absolute(pl.col("w"), pl.col("h"))
        )["a"].to_list()
        assert out[0] == {"x": 50.0, "y": 5.0}
        assert out[1] == {"x": 100.0, "y": 10.0}

    def test_translate_uses_per_row_offsets(self) -> None:
        df = _two_rows(p=[{"x": 1.0, "y": 1.0}] * 2, dx=[1.0, 10.0], dy=[2.0, 20.0])
        out = df.with_columns(
            t=pl.col("p").point.translate(pl.col("dx"), pl.col("dy"))
        )["t"].to_list()
        assert out[0] == {"x": 2.0, "y": 3.0}
        assert out[1] == {"x": 11.0, "y": 21.0}

    def test_scale_uses_per_row_factors(self) -> None:
        df = _two_rows(p=[{"x": 3.0, "y": 4.0}] * 2, sx=[2.0, 0.5], sy=[1.0, 2.0])
        out = df.with_columns(s=pl.col("p").point.scale(pl.col("sx"), pl.col("sy")))[
            "s"
        ].to_list()
        assert out[0] == {"x": 6.0, "y": 4.0}
        assert out[1] == {"x": 1.5, "y": 8.0}

    def test_rotate_uses_per_row_angle(self) -> None:
        df = _two_rows(p=[{"x": 1.0, "y": 0.0}] * 2, ang=[0.0, math.pi / 2])
        out = df.with_columns(r=pl.col("p").point.rotate(pl.col("ang")))["r"].to_list()
        assert out[0]["x"] == pytest.approx(1.0)
        assert out[1]["x"] == pytest.approx(0.0, abs=1e-9)
        assert out[1]["y"] == pytest.approx(1.0)

    def test_rotate_per_row_angle_with_origin_operand(self) -> None:
        """The optional `origin` operand and a dynamic `angle` coexist.

        Both occupy plugin input slots, which is exactly the collision the
        name-keyed `input_slots` map exists to prevent.
        """
        df = _two_rows(
            p=[{"x": 0.0, "y": 0.0}] * 2,
            o=[{"x": 5.0, "y": 5.0}] * 2,
            ang=[0.0, math.pi],
        )
        out = df.with_columns(
            r=pl.col("p").point.rotate(pl.col("ang"), origin=pl.col("o"))
        )["r"].to_list()
        assert out[0]["x"] == pytest.approx(0.0)
        assert out[1]["x"] == pytest.approx(10.0)
        assert out[1]["y"] == pytest.approx(10.0)

    def test_interpolate_uses_per_row_t(self) -> None:
        df = _two_rows(
            p=[{"x": 0.0, "y": 0.0}] * 2,
            q=[{"x": 10.0, "y": 20.0}] * 2,
            t=[0.25, 0.75],
        )
        out = df.with_columns(
            i=pl.col("p").point.interpolate(pl.col("q"), t=pl.col("t"))
        )["i"].to_list()
        assert out[0] == {"x": 2.5, "y": 5.0}
        assert out[1] == {"x": 7.5, "y": 15.0}


@plugin_required
class TestDetectionMatchingThreshold:
    """`correspond(threshold=)` resolves per row on both namespaces."""

    @staticmethod
    def _boxes() -> tuple[list, list]:
        """One prediction overlapping one ground truth at IoU = 0.25."""
        pred = [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}]
        gt = [{"x": 5.0, "y": 0.0, "width": 10.0, "height": 10.0}]
        return pred, gt

    def test_bbox_threshold_is_per_row(self) -> None:
        pred, gt = self._boxes()
        # IoU here is 1/3; a 0.2 threshold matches, a 0.9 threshold does not.
        df = _two_rows(p=[pred, pred], g=[gt, gt], thr=[0.2, 0.9])
        out = df.with_columns(
            m=pl.col("p").bbox.correspond(pl.col("g"), threshold=pl.col("thr"))
        )["m"].to_list()
        assert out[0]["right_idx"] == [0]
        assert out[1]["right_idx"] == [None]

    def test_bbox_threshold_per_row_with_order_operand(self) -> None:
        """A dynamic threshold and the optional `order` operand coexist."""
        pred, gt = self._boxes()
        df = _two_rows(p=[pred, pred], g=[gt, gt], s=[[0], [0]], thr=[0.2, 0.9])
        out = df.with_columns(
            m=pl.col("p").bbox.correspond(
                pl.col("g"), threshold=pl.col("thr"), order=pl.col("s")
            )
        )["m"].to_list()
        assert out[0]["right_idx"] == [0]
        assert out[1]["right_idx"] == [None]

    def test_out_of_range_threshold_names_the_row(self) -> None:
        pred, gt = self._boxes()
        df = _two_rows(p=[pred, pred], g=[gt, gt], thr=[0.5, 1.5])
        with pytest.raises(Exception, match=r"threshold must be in \[0, 1\].*row 1"):
            df.with_columns(
                m=pl.col("p").bbox.correspond(pl.col("g"), threshold=pl.col("thr"))
            )


@plugin_required
class TestGeometryParamEdgeCases:
    """Broadcasting and null handling match the graph engine's behaviour."""

    def test_aggregation_broadcasts_to_every_row(self) -> None:
        """A length-1 series from an aggregation applies to all rows."""
        df = _two_rows(c=[SQUARE, SQUARE], w=[100.0, 200.0])
        out = df.with_columns(n=pl.col("c").contour.normalize(pl.col("w").max(), 100))[
            "n"
        ].to_list()
        # max() == 200 for both rows, so both normalize identically.
        assert out[0]["exterior"][2]["x"] == 0.5
        assert out[1]["exterior"][2]["x"] == 0.5

    def test_null_parameter_is_an_error(self) -> None:
        df = _two_rows(c=[SQUARE, SQUARE], w=[100.0, None])
        with pytest.raises(Exception, match="null"):
            df.with_columns(n=pl.col("c").contour.normalize(pl.col("w"), 100))

    def test_zero_dimension_names_the_row(self) -> None:
        df = _two_rows(p=[{"x": 1.0, "y": 1.0}] * 2, w=[100.0, 0.0])
        with pytest.raises(Exception, match="non-zero"):
            df.with_columns(n=pl.col("p").point.normalize(pl.col("w"), 100))

    def test_integer_column_is_accepted_for_a_float_parameter(self) -> None:
        """Parameter columns go through `ParamCol`, which spans every numeric dtype."""
        df = pl.DataFrame(
            {"c": [SQUARE, SQUARE], "w": pl.Series([100, 200], dtype=pl.Int32)}
        )
        out = df.with_columns(n=pl.col("c").contour.normalize(pl.col("w"), 100))[
            "n"
        ].to_list()
        assert out[0]["exterior"][2]["x"] == 1.0
        assert out[1]["exterior"][2]["x"] == 0.5
