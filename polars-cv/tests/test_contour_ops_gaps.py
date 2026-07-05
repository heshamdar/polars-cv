"""
Tests filling gaps in contour operation coverage.

Covers: area(signed=True), flip+winding verification, simplify edge cases,
scale with different origins, pairwise metrics with partial/no overlap,
contains_point boundary cases, and is_convex on non-convex shapes.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.geometry import CONTOUR_SCHEMA, POINT_SCHEMA
from tests.conftest import plugin_required

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ccw_square() -> dict:
    """CCW-wound 100×100 square (positive signed area)."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 100.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }


@pytest.fixture
def cw_square() -> dict:
    """CW-wound 100×100 square (negative signed area)."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 0.0, "y": 100.0},
            {"x": 100.0, "y": 100.0},
            {"x": 100.0, "y": 0.0},
        ],
        "holes": [],
        "is_closed": True,
    }


@pytest.fixture
def l_shape() -> dict:
    """L-shaped non-convex contour."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 100.0, "y": 50.0},
            {"x": 50.0, "y": 50.0},
            {"x": 50.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }


@pytest.fixture
def overlapping_squares() -> tuple[dict, dict]:
    """Two overlapping squares: (0,0)→(60,60) and (40,40)→(100,100).

    Intersection: (40,40)→(60,60) = 20×20 = 400
    Union: 2×3600 - 400 = 6800
    IoU = 400/6800 ≈ 0.0588
    """
    c1 = {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 60.0, "y": 0.0},
            {"x": 60.0, "y": 60.0},
            {"x": 0.0, "y": 60.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    c2 = {
        "exterior": [
            {"x": 40.0, "y": 40.0},
            {"x": 100.0, "y": 40.0},
            {"x": 100.0, "y": 100.0},
            {"x": 40.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    return c1, c2


@pytest.fixture
def non_overlapping_squares() -> tuple[dict, dict]:
    """Two non-overlapping squares: (0,0)→(40,40) and (60,60)→(100,100)."""
    c1 = {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 40.0, "y": 0.0},
            {"x": 40.0, "y": 40.0},
            {"x": 0.0, "y": 40.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    c2 = {
        "exterior": [
            {"x": 60.0, "y": 60.0},
            {"x": 100.0, "y": 60.0},
            {"x": 100.0, "y": 100.0},
            {"x": 60.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    return c1, c2


# ---------------------------------------------------------------------------
# area(signed=True)
# ---------------------------------------------------------------------------


@plugin_required
class TestContourAreaSigned:
    """Tests for area(signed=True) which was previously untested."""

    def test_signed_area_ccw_positive(self, ccw_square: dict) -> None:
        """CCW contour should yield positive signed area."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(area=pl.col("contour").contour.area(signed=True))
        assert result["area"][0] > 0
        assert result["area"][0] == pytest.approx(10000.0, rel=0.01)

    def test_signed_area_cw_negative(self, cw_square: dict) -> None:
        """CW contour should yield negative signed area."""
        df = pl.DataFrame({"contour": [cw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(area=pl.col("contour").contour.area(signed=True))
        assert result["area"][0] < 0
        assert result["area"][0] == pytest.approx(-10000.0, rel=0.01)

    def test_unsigned_area_always_positive(self, cw_square: dict) -> None:
        """Default signed=False should always return positive area."""
        df = pl.DataFrame({"contour": [cw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(area=pl.col("contour").contour.area(signed=False))
        assert result["area"][0] > 0

    def test_signed_area_both_windings_same_magnitude(
        self, ccw_square: dict, cw_square: dict
    ) -> None:
        """Signed areas of opposite windings should be equal in magnitude."""
        df = pl.DataFrame(
            {"contour": [ccw_square, cw_square]},
            schema={"contour": CONTOUR_SCHEMA},
        )
        result = df.with_columns(area=pl.col("contour").contour.area(signed=True))
        assert abs(result["area"][0]) == pytest.approx(abs(result["area"][1]), rel=0.01)


# ---------------------------------------------------------------------------
# flip + winding behavioural verification
# ---------------------------------------------------------------------------


@plugin_required
class TestContourFlipWinding:
    """Verify that flip() actually reverses winding direction."""

    def test_flip_reverses_winding(self, ccw_square: dict) -> None:
        """Flipping a CCW contour should produce CW winding."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            original_winding=pl.col("contour").contour.winding(),
            flipped=pl.col("contour").contour.flip(),
        ).with_columns(
            flipped_winding=pl.col("flipped").contour.winding(),
        )
        orig = result["original_winding"][0]
        flipped = result["flipped_winding"][0]
        assert orig != flipped
        assert {orig, flipped} == {"cw", "ccw"}

    def test_double_flip_restores_winding(self, ccw_square: dict) -> None:
        """Flipping twice should restore original winding."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            original_winding=pl.col("contour").contour.winding(),
            double_flipped=pl.col("contour").contour.flip().contour.flip(),
        ).with_columns(
            restored_winding=pl.col("double_flipped").contour.winding(),
        )
        assert result["original_winding"][0] == result["restored_winding"][0]

    def test_flip_preserves_area(self, ccw_square: dict) -> None:
        """Flipping should not change unsigned area."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            orig_area=pl.col("contour").contour.area(),
            flipped_area=pl.col("contour").contour.flip().contour.area(),
        )
        assert result["orig_area"][0] == pytest.approx(
            result["flipped_area"][0], rel=0.01
        )


# ---------------------------------------------------------------------------
# ensure_winding
# ---------------------------------------------------------------------------


@plugin_required
class TestEnsureWinding:
    """Verify ensure_winding() actually corrects winding when needed."""

    def test_ensure_ccw_on_ccw_is_noop(self, ccw_square: dict) -> None:
        """Already-CCW contour should be unchanged by ensure_winding('ccw')."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("ccw"),
        ).with_columns(
            winding=pl.col("ensured").contour.winding(),
        )
        assert result["winding"][0] == "ccw"

    def test_ensure_cw_on_ccw_flips(self, ccw_square: dict) -> None:
        """CCW contour with ensure_winding('cw') should become CW."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("cw"),
        ).with_columns(
            winding=pl.col("ensured").contour.winding(),
        )
        assert result["winding"][0] == "cw"

    def test_ensure_ccw_on_cw_flips(self, cw_square: dict) -> None:
        """CW contour with ensure_winding('ccw') should become CCW."""
        df = pl.DataFrame({"contour": [cw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("ccw"),
        ).with_columns(
            winding=pl.col("ensured").contour.winding(),
        )
        assert result["winding"][0] == "ccw"


# ---------------------------------------------------------------------------
# simplify edge cases
# ---------------------------------------------------------------------------


@plugin_required
class TestContourSimplify:
    """Simplify with various tolerance values."""

    def test_simplify_zero_tolerance_preserves_all_points(
        self, ccw_square: dict
    ) -> None:
        """Zero tolerance should preserve all points."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            simplified=pl.col("contour").contour.simplify(tolerance=0.0),
        )
        # Original has 4 points, simplified with 0 tolerance should keep all
        orig_exterior = result["contour"][0]["exterior"]
        simp_exterior = result["simplified"][0]["exterior"]
        assert len(simp_exterior) == len(orig_exterior)

    def test_simplify_large_tolerance_reduces_points(self, l_shape: dict) -> None:
        """Large tolerance should reduce point count on complex shape."""
        df = pl.DataFrame({"contour": [l_shape]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            simplified=pl.col("contour").contour.simplify(tolerance=100.0),
        )
        simp_exterior = result["simplified"][0]["exterior"]
        # L-shape has 6 points; large tolerance should reduce to fewer
        assert len(simp_exterior) < 6

    def test_simplify_preserves_closure(self, ccw_square: dict) -> None:
        """Simplified contour should remain closed."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            simplified=pl.col("contour").contour.simplify(tolerance=1.0),
        )
        assert result["simplified"][0]["is_closed"] is True


# ---------------------------------------------------------------------------
# scale with different origins
# ---------------------------------------------------------------------------


@plugin_required
class TestContourScaleOrigins:
    """Test scale with all three origin options."""

    def test_scale_origin_default(self, ccw_square: dict) -> None:
        """Scale from origin (0,0) – points should multiply."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            scaled=pl.col("contour").contour.scale(sx=2.0, sy=2.0, origin="origin"),
        )
        ext = result["scaled"][0]["exterior"]
        # Original (100,100) corner should become (200,200)
        xs = [p["x"] for p in ext]
        ys = [p["y"] for p in ext]
        assert max(xs) == pytest.approx(200.0, rel=0.01)
        assert max(ys) == pytest.approx(200.0, rel=0.01)

    def test_scale_origin_centroid(self, ccw_square: dict) -> None:
        """Scale from centroid – centroid should stay fixed."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            orig_centroid=pl.col("contour").contour.centroid(),
            scaled=pl.col("contour").contour.scale(sx=2.0, sy=2.0, origin="centroid"),
        ).with_columns(
            scaled_centroid=pl.col("scaled").contour.centroid(),
        )
        oc = result["orig_centroid"][0]
        sc = result["scaled_centroid"][0]
        assert oc["x"] == pytest.approx(sc["x"], abs=1.0)
        assert oc["y"] == pytest.approx(sc["y"], abs=1.0)

    def test_scale_origin_bbox_center(self, ccw_square: dict) -> None:
        """Scale from bbox_center – bbox center should stay fixed."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            orig_bbox=pl.col("contour").contour.bounding_box(),
            scaled=pl.col("contour").contour.scale(
                sx=2.0, sy=2.0, origin="bbox_center"
            ),
        ).with_columns(
            scaled_bbox=pl.col("scaled").contour.bounding_box(),
        )
        ob = result["orig_bbox"][0]
        sb = result["scaled_bbox"][0]
        orig_cx = ob["x"] + ob["width"] / 2
        orig_cy = ob["y"] + ob["height"] / 2
        scaled_cx = sb["x"] + sb["width"] / 2
        scaled_cy = sb["y"] + sb["height"] / 2
        assert orig_cx == pytest.approx(scaled_cx, abs=1.0)
        assert orig_cy == pytest.approx(scaled_cy, abs=1.0)


# ---------------------------------------------------------------------------
# Pairwise: IoU, Dice, Hausdorff with overlap scenarios
# ---------------------------------------------------------------------------


@plugin_required
class TestPairwiseOverlapScenarios:
    """Test IoU, Dice, Hausdorff with identical, partial, and no overlap."""

    def test_iou_identical_is_one(self, ccw_square: dict) -> None:
        """IoU of identical contours should be 1.0."""
        df = pl.DataFrame(
            {"a": [ccw_square], "b": [ccw_square]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))
        assert result["iou"][0] == pytest.approx(1.0, abs=0.05)

    def test_iou_no_overlap_is_zero(
        self, non_overlapping_squares: tuple[dict, dict]
    ) -> None:
        """IoU of non-overlapping contours should be 0.0."""
        c1, c2 = non_overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))
        assert result["iou"][0] == pytest.approx(0.0, abs=0.05)

    def test_iou_partial_overlap_in_range(
        self, overlapping_squares: tuple[dict, dict]
    ) -> None:
        """IoU of partially overlapping contours should be between 0 and 1."""
        c1, c2 = overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))
        val = result["iou"][0]
        assert 0.0 < val < 1.0

    def test_dice_identical_is_one(self, ccw_square: dict) -> None:
        """Dice of identical contours should be 1.0."""
        df = pl.DataFrame(
            {"a": [ccw_square], "b": [ccw_square]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(dice=pl.col("a").contour.dice(pl.col("b")))
        assert result["dice"][0] == pytest.approx(1.0, abs=0.05)

    def test_dice_no_overlap_is_zero(
        self, non_overlapping_squares: tuple[dict, dict]
    ) -> None:
        """Dice of non-overlapping contours should be 0.0."""
        c1, c2 = non_overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(dice=pl.col("a").contour.dice(pl.col("b")))
        assert result["dice"][0] == pytest.approx(0.0, abs=0.05)

    def test_hausdorff_identical_is_zero(self, ccw_square: dict) -> None:
        """Hausdorff distance of identical contours should be 0.0."""
        df = pl.DataFrame(
            {"a": [ccw_square], "b": [ccw_square]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(
            dist=pl.col("a").contour.hausdorff_distance(pl.col("b"))
        )
        assert result["dist"][0] == pytest.approx(0.0, abs=0.5)

    def test_hausdorff_non_overlapping_positive(
        self, non_overlapping_squares: tuple[dict, dict]
    ) -> None:
        """Hausdorff distance of separated contours should be positive."""
        c1, c2 = non_overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(
            dist=pl.col("a").contour.hausdorff_distance(pl.col("b"))
        )
        assert result["dist"][0] > 0.0

    def test_hausdorff_is_symmetric(
        self, overlapping_squares: tuple[dict, dict]
    ) -> None:
        """Hausdorff distance should be symmetric: d(A,B) == d(B,A)."""
        c1, c2 = overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(
            d_ab=pl.col("a").contour.hausdorff_distance(pl.col("b")),
            d_ba=pl.col("b").contour.hausdorff_distance(pl.col("a")),
        )
        assert result["d_ab"][0] == pytest.approx(result["d_ba"][0], rel=0.01)


# ---------------------------------------------------------------------------
# contains_point
# ---------------------------------------------------------------------------


@plugin_required
class TestContainsPoint:
    """Test point-in-polygon checks."""

    def test_point_inside(self, ccw_square: dict) -> None:
        """A point inside the contour should return True."""
        df = pl.DataFrame(
            {
                "contour": [ccw_square],
                "point": [{"x": 50.0, "y": 50.0}],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            inside=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        assert result["inside"][0] is True

    def test_point_outside(self, ccw_square: dict) -> None:
        """A point outside the contour should return False."""
        df = pl.DataFrame(
            {
                "contour": [ccw_square],
                "point": [{"x": 200.0, "y": 200.0}],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            inside=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        assert result["inside"][0] is False

    def test_point_at_origin_corner(self, ccw_square: dict) -> None:
        """Point exactly at vertex (0,0) – boundary case."""
        df = pl.DataFrame(
            {
                "contour": [ccw_square],
                "point": [{"x": 0.0, "y": 0.0}],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            inside=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        # Boundary behavior is implementation-defined; just verify no crash
        assert result["inside"][0] in (True, False)


# ---------------------------------------------------------------------------
# is_convex
# ---------------------------------------------------------------------------


@plugin_required
class TestIsConvex:
    """Verify is_convex() correctness for convex and non-convex shapes."""

    def test_square_is_convex(self, ccw_square: dict) -> None:
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(convex=pl.col("contour").contour.is_convex())
        assert result["convex"][0] is True

    def test_l_shape_is_not_convex(self, l_shape: dict) -> None:
        df = pl.DataFrame({"contour": [l_shape]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(convex=pl.col("contour").contour.is_convex())
        assert result["convex"][0] is False
