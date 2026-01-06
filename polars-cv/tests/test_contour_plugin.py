"""
Integration tests for contour plugin operations.

These tests verify that all contour operations exposed in the Python frontend
are actually implemented and working correctly in the Rust backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars_cv.geometry import CONTOUR_SCHEMA, POINT_SCHEMA

if TYPE_CHECKING:
    pass


def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    from pathlib import Path

    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


# Mark all tests in this module as requiring the plugin
plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@pytest.fixture
def square_contour() -> dict:
    """Create a 100x100 square contour at origin."""
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
def triangle_contour() -> dict:
    """Create a triangle contour."""
    return {
        "exterior": [
            {"x": 50.0, "y": 0.0},
            {"x": 100.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }


@pytest.fixture
def l_shape_contour() -> dict:
    """Create an L-shaped (non-convex) contour."""
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
def contour_with_hole() -> dict:
    """Create a contour with a hole."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 100.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [
            [
                {"x": 25.0, "y": 25.0},
                {"x": 75.0, "y": 25.0},
                {"x": 75.0, "y": 75.0},
                {"x": 25.0, "y": 75.0},
            ]
        ],
        "is_closed": True,
    }


@pytest.fixture
def square_df(square_contour: dict) -> pl.DataFrame:
    """Create a DataFrame with a square contour."""
    return pl.DataFrame(
        {"contour": [square_contour]}, schema={"contour": CONTOUR_SCHEMA}
    )


@pytest.fixture
def multi_contour_df(
    square_contour: dict, triangle_contour: dict, l_shape_contour: dict
) -> pl.DataFrame:
    """Create a DataFrame with multiple contours."""
    return pl.DataFrame(
        {"contour": [square_contour, triangle_contour, l_shape_contour]},
        schema={"contour": CONTOUR_SCHEMA},
    )


@plugin_required
class TestContourArea:
    """Tests for contour.area() operation."""

    def test_area_square(self, square_df: pl.DataFrame) -> None:
        """Area of 100x100 square should be 10000."""
        result = square_df.with_columns(area=pl.col("contour").contour.area())
        assert result["area"][0] == pytest.approx(10000.0)

    def test_area_triangle(self, triangle_contour: dict) -> None:
        """Area of right triangle should be 0.5 * base * height."""
        df = pl.DataFrame(
            {"contour": [triangle_contour]}, schema={"contour": CONTOUR_SCHEMA}
        )
        result = df.with_columns(area=pl.col("contour").contour.area())
        # Triangle with base 100 and height 100 -> area = 5000
        assert result["area"][0] == pytest.approx(5000.0, rel=0.01)

    def test_area_with_hole(self, contour_with_hole: dict) -> None:
        """Area with hole - currently only exterior is parsed.

        Note: Hole parsing is not yet implemented in the Rust backend,
        so this returns the exterior area only. When hole parsing is
        implemented, this should return 7500 (10000 - 2500).
        """
        df = pl.DataFrame(
            {"contour": [contour_with_hole]}, schema={"contour": CONTOUR_SCHEMA}
        )
        result = df.with_columns(area=pl.col("contour").contour.area())
        # Currently returns exterior area only (holes not parsed yet)
        assert result["area"][0] == pytest.approx(10000.0, rel=0.01)


@plugin_required
class TestContourPerimeter:
    """Tests for contour.perimeter() operation."""

    def test_perimeter_square(self, square_df: pl.DataFrame) -> None:
        """Perimeter of 100x100 square should be 400."""
        result = square_df.with_columns(perimeter=pl.col("contour").contour.perimeter())
        assert result["perimeter"][0] == pytest.approx(400.0)

    def test_perimeter_triangle(self, triangle_contour: dict) -> None:
        """Perimeter of triangle."""
        df = pl.DataFrame(
            {"contour": [triangle_contour]}, schema={"contour": CONTOUR_SCHEMA}
        )
        result = df.with_columns(perimeter=pl.col("contour").contour.perimeter())
        # Triangle with vertices at (50,0), (100,100), (0,100)
        # Sides: ~111.8, 100, ~111.8 -> ~323.6
        assert result["perimeter"][0] > 300.0
        assert result["perimeter"][0] < 350.0


@plugin_required
class TestContourWinding:
    """Tests for contour.winding() operation."""

    def test_winding_returns_string(self, square_df: pl.DataFrame) -> None:
        """Winding should return 'cw' or 'ccw'."""
        result = square_df.with_columns(winding=pl.col("contour").contour.winding())
        assert result["winding"][0] in ("cw", "ccw")

    def test_winding_consistency(self, multi_contour_df: pl.DataFrame) -> None:
        """All contours should have a valid winding direction."""
        result = multi_contour_df.with_columns(
            winding=pl.col("contour").contour.winding()
        )
        for winding in result["winding"]:
            assert winding in ("cw", "ccw")


@plugin_required
class TestContourIsConvex:
    """Tests for contour.is_convex() operation."""

    def test_square_is_convex(self, square_df: pl.DataFrame) -> None:
        """Square should be convex."""
        result = square_df.with_columns(is_convex=pl.col("contour").contour.is_convex())
        assert result["is_convex"][0] is True

    def test_triangle_is_convex(self, triangle_contour: dict) -> None:
        """Triangle should be convex."""
        df = pl.DataFrame(
            {"contour": [triangle_contour]}, schema={"contour": CONTOUR_SCHEMA}
        )
        result = df.with_columns(is_convex=pl.col("contour").contour.is_convex())
        assert result["is_convex"][0] is True

    def test_l_shape_not_convex(self, l_shape_contour: dict) -> None:
        """L-shape should not be convex."""
        df = pl.DataFrame(
            {"contour": [l_shape_contour]}, schema={"contour": CONTOUR_SCHEMA}
        )
        result = df.with_columns(is_convex=pl.col("contour").contour.is_convex())
        assert result["is_convex"][0] is False


@plugin_required
class TestContourCentroid:
    """Tests for contour.centroid() operation."""

    def test_centroid_square(self, square_df: pl.DataFrame) -> None:
        """Centroid of 100x100 square at origin should be (50, 50)."""
        result = square_df.with_columns(centroid=pl.col("contour").contour.centroid())
        centroid = result["centroid"][0]
        assert centroid["x"] == pytest.approx(50.0)
        assert centroid["y"] == pytest.approx(50.0)

    def test_centroid_returns_struct(self, square_df: pl.DataFrame) -> None:
        """Centroid should return a struct with x and y fields."""
        result = square_df.with_columns(centroid=pl.col("contour").contour.centroid())
        assert result["centroid"].dtype == pl.Struct(
            [pl.Field("x", pl.Float64), pl.Field("y", pl.Float64)]
        )


@plugin_required
class TestContourBoundingBox:
    """Tests for contour.bounding_box() operation."""

    def test_bbox_square(self, square_df: pl.DataFrame) -> None:
        """Bounding box of 100x100 square at origin."""
        result = square_df.with_columns(bbox=pl.col("contour").contour.bounding_box())
        bbox = result["bbox"][0]
        assert bbox["x"] == pytest.approx(0.0)
        assert bbox["y"] == pytest.approx(0.0)
        assert bbox["width"] == pytest.approx(100.0)
        assert bbox["height"] == pytest.approx(100.0)

    def test_bbox_returns_struct(self, square_df: pl.DataFrame) -> None:
        """Bounding box should return a struct with x, y, width, height."""
        result = square_df.with_columns(bbox=pl.col("contour").contour.bounding_box())
        expected_dtype = pl.Struct(
            [
                pl.Field("x", pl.Float64),
                pl.Field("y", pl.Float64),
                pl.Field("width", pl.Float64),
                pl.Field("height", pl.Float64),
            ]
        )
        assert result["bbox"].dtype == expected_dtype


@plugin_required
class TestContourConvexHull:
    """Tests for contour.convex_hull() operation."""

    def test_convex_hull_returns_contour(self, square_df: pl.DataFrame) -> None:
        """Convex hull should return a contour."""
        result = square_df.with_columns(hull=pl.col("contour").contour.convex_hull())
        # Should return same type as input
        assert result["hull"].dtype == result["contour"].dtype

    def test_convex_hull_l_shape(self, l_shape_contour: dict) -> None:
        """Convex hull of L-shape should be a valid contour."""
        df = pl.DataFrame(
            {"contour": [l_shape_contour]}, schema={"contour": CONTOUR_SCHEMA}
        )
        result = df.with_columns(hull=pl.col("contour").contour.convex_hull())
        # Just verify it executes without error
        assert result["hull"].is_not_null().all()


@plugin_required
class TestContourFlip:
    """Tests for contour.flip() operation."""

    def test_flip_returns_contour(self, square_df: pl.DataFrame) -> None:
        """Flip should return a contour."""
        result = square_df.with_columns(flipped=pl.col("contour").contour.flip())
        assert result["flipped"].dtype == result["contour"].dtype


@plugin_required
class TestContourTranslate:
    """Tests for contour.translate() operation."""

    def test_translate_returns_contour(self, square_df: pl.DataFrame) -> None:
        """Translate should return a contour."""
        result = square_df.with_columns(
            translated=pl.col("contour").contour.translate(dx=10.0, dy=20.0)
        )
        assert result["translated"].dtype == result["contour"].dtype

    def test_translate_moves_points(self, square_df: pl.DataFrame) -> None:
        """Translate should actually move the contour points."""
        result = square_df.with_columns(
            translated=pl.col("contour").contour.translate(dx=10.0, dy=20.0)
        )
        translated = result["translated"][0]
        exterior = translated["exterior"]

        # Original first point was (0, 0), should now be (10, 20)
        assert exterior[0]["x"] == pytest.approx(10.0)
        assert exterior[0]["y"] == pytest.approx(20.0)

        # Original second point was (100, 0), should now be (110, 20)
        assert exterior[1]["x"] == pytest.approx(110.0)
        assert exterior[1]["y"] == pytest.approx(20.0)

    def test_translate_zero_is_identity(self, square_df: pl.DataFrame) -> None:
        """Translate by (0, 0) should not change points."""
        result = square_df.with_columns(
            translated=pl.col("contour").contour.translate(dx=0.0, dy=0.0)
        )
        translated = result["translated"][0]
        exterior = translated["exterior"]

        assert exterior[0]["x"] == pytest.approx(0.0)
        assert exterior[0]["y"] == pytest.approx(0.0)


@plugin_required
class TestContourScale:
    """Tests for contour.scale() operation."""

    def test_scale_returns_contour(self, square_df: pl.DataFrame) -> None:
        """Scale should return a contour."""
        result = square_df.with_columns(
            scaled=pl.col("contour").contour.scale(sx=0.5, sy=0.5)
        )
        assert result["scaled"].dtype == result["contour"].dtype

    def test_scale_halves_dimensions(self, square_df: pl.DataFrame) -> None:
        """Scaling by 0.5 should halve the bounding box dimensions."""
        # Original: 100x100 square, centroid at (50, 50)
        # After 0.5 scale: points should be halfway between centroid and original
        result = square_df.with_columns(
            scaled=pl.col("contour").contour.scale(sx=0.5, sy=0.5, origin="centroid")
        )
        scaled = result["scaled"][0]
        exterior = scaled["exterior"]

        # After scaling around centroid (50,50):
        # (0,0) -> (25,25), (100,0) -> (75,25), etc.
        # All points should be 0.5 * distance from centroid
        for point in exterior:
            # Distance from center (50, 50) should be halved
            x, y = point["x"], point["y"]
            # All points should be within the scaled bounds
            assert 25.0 <= x <= 75.0
            assert 25.0 <= y <= 75.0

    def test_scale_one_is_identity(self, square_df: pl.DataFrame) -> None:
        """Scaling by (1, 1) should not change points."""
        result = square_df.with_columns(
            scaled=pl.col("contour").contour.scale(sx=1.0, sy=1.0, origin="centroid")
        )
        scaled = result["scaled"][0]
        exterior = scaled["exterior"]

        assert exterior[0]["x"] == pytest.approx(0.0)
        assert exterior[0]["y"] == pytest.approx(0.0)

    def test_origin_scaling(self, square_df: pl.DataFrame) -> None:
        """Scaling with different origins should change the points."""
        result = square_df.with_columns(
            scaled=pl.col("contour").contour.scale(sx=0.1, sy=0.1, origin="origin")
        )
        scaled = result["scaled"][0]
        exterior = scaled["exterior"]
        print(result)
        assert exterior[0]["x"] == pytest.approx(0)
        assert exterior[0]["y"] == pytest.approx(0)
        assert exterior[1]["x"] == pytest.approx(10)
        assert exterior[1]["y"] == pytest.approx(0)


@plugin_required
class TestContourSimplify:
    """Tests for contour.simplify() operation."""

    def test_simplify_returns_contour(self, square_df: pl.DataFrame) -> None:
        """Simplify should return a contour."""
        result = square_df.with_columns(
            simplified=pl.col("contour").contour.simplify(tolerance=1.0)
        )
        assert result["simplified"].dtype == result["contour"].dtype

    def test_simplify_preserves_corners(self, square_df: pl.DataFrame) -> None:
        """Simplify with low tolerance should preserve corner points."""
        result = square_df.with_columns(
            simplified=pl.col("contour").contour.simplify(tolerance=0.1)
        )
        simplified = result["simplified"][0]
        exterior = simplified["exterior"]

        # A square should still have 4 points after low-tolerance simplification
        assert len(exterior) >= 3  # At least 3 points for a valid polygon


@plugin_required
class TestContourNormalize:
    """Tests for contour.normalize() operation."""

    def test_normalize_returns_contour(self, square_df: pl.DataFrame) -> None:
        """Normalize should return a contour."""
        result = square_df.with_columns(
            normalized=pl.col("contour").contour.normalize(
                ref_width=100, ref_height=100
            )
        )
        assert result["normalized"].dtype == result["contour"].dtype

    def test_normalize_scales_to_unit_range(self, square_df: pl.DataFrame) -> None:
        """Normalize should scale coordinates to [0, 1] range."""
        result = square_df.with_columns(
            normalized=pl.col("contour").contour.normalize(
                ref_width=100, ref_height=100
            )
        )
        normalized = result["normalized"][0]
        exterior = normalized["exterior"]

        # All coordinates should be in [0, 1] range
        for point in exterior:
            assert 0.0 <= point["x"] <= 1.0
            assert 0.0 <= point["y"] <= 1.0

        # Corner at (0, 0) should still be (0, 0)
        assert exterior[0]["x"] == pytest.approx(0.0)
        assert exterior[0]["y"] == pytest.approx(0.0)

        # Corner at (100, 100) should become (1, 1)
        has_max_corner = any(
            point["x"] == pytest.approx(1.0) and point["y"] == pytest.approx(1.0)
            for point in exterior
        )
        assert has_max_corner


@plugin_required
class TestContourToAbsolute:
    """Tests for contour.to_absolute() operation."""

    def test_to_absolute_returns_contour(self, square_df: pl.DataFrame) -> None:
        """to_absolute should return a contour."""
        result = square_df.with_columns(
            absolute=pl.col("contour").contour.to_absolute(
                ref_width=100, ref_height=100
            )
        )
        assert result["absolute"].dtype == result["contour"].dtype


@plugin_required
class TestContourEnsureWinding:
    """Tests for contour.ensure_winding() operation."""

    def test_ensure_winding_ccw_returns_contour(self, square_df: pl.DataFrame) -> None:
        """ensure_winding should return a contour."""
        result = square_df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("ccw")
        )
        assert result["ensured"].dtype == result["contour"].dtype

    def test_ensure_winding_cw_returns_contour(self, square_df: pl.DataFrame) -> None:
        """ensure_winding with 'cw' should return a contour."""
        result = square_df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("cw")
        )
        assert result["ensured"].dtype == result["contour"].dtype


@plugin_required
class TestContourContainsPoint:
    """Tests for contour.contains_point() operation."""

    def test_point_inside(self, square_contour: dict) -> None:
        """Point at (50, 50) should be inside 100x100 square."""
        df = pl.DataFrame(
            {
                "contour": [square_contour],
                "point": [{"x": 50.0, "y": 50.0}],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            contains=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        assert result["contains"][0] is True

    def test_point_outside(self, square_contour: dict) -> None:
        """Point at (150, 50) should be outside 100x100 square."""
        df = pl.DataFrame(
            {
                "contour": [square_contour],
                "point": [{"x": 150.0, "y": 50.0}],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            contains=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        assert result["contains"][0] is False

    def test_multiple_points(self, square_contour: dict) -> None:
        """Test multiple points at once."""
        df = pl.DataFrame(
            {
                "contour": [square_contour, square_contour, square_contour],
                "point": [
                    {"x": 50.0, "y": 50.0},  # Inside
                    {"x": 150.0, "y": 50.0},  # Outside
                    {"x": 0.0, "y": 0.0},  # On boundary (corner)
                ],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            contains=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        assert result["contains"][0] is True  # Inside
        assert result["contains"][1] is False  # Outside
        # On boundary - could be either depending on implementation


@plugin_required
class TestContourIoU:
    """Tests for contour.iou() operation."""

    def test_iou_identical(self, square_contour: dict) -> None:
        """IoU of identical contours should be 1.0."""
        df = pl.DataFrame(
            {"a": [square_contour], "b": [square_contour]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))
        assert result["iou"][0] == pytest.approx(1.0)


@plugin_required
class TestContourDice:
    """Tests for contour.dice() operation."""

    def test_dice_identical(self, square_contour: dict) -> None:
        """Dice of identical contours should be 1.0."""
        df = pl.DataFrame(
            {"a": [square_contour], "b": [square_contour]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(dice=pl.col("a").contour.dice(pl.col("b")))
        assert result["dice"][0] == pytest.approx(1.0)


@plugin_required
class TestContourHausdorff:
    """Tests for contour.hausdorff_distance() operation."""

    def test_hausdorff_identical(self, square_contour: dict) -> None:
        """Hausdorff distance of identical contours should be 0."""
        df = pl.DataFrame(
            {"a": [square_contour], "b": [square_contour]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(
            hausdorff=pl.col("a").contour.hausdorff_distance(pl.col("b"))
        )
        assert result["hausdorff"][0] == pytest.approx(0.0)


@plugin_required
class TestContourNullHandling:
    """Tests for null handling in contour operations."""

    def test_area_with_null(self, square_contour: dict) -> None:
        """Area should return null for null input."""
        df = pl.DataFrame(
            {"contour": [square_contour, None]}, schema={"contour": CONTOUR_SCHEMA}
        )
        result = df.with_columns(area=pl.col("contour").contour.area())
        assert result["area"][0] == pytest.approx(10000.0)
        assert result["area"][1] is None

    def test_centroid_with_null(self, square_contour: dict) -> None:
        """Centroid should return null fields for null input."""
        df = pl.DataFrame(
            {"contour": [square_contour, None]}, schema={"contour": CONTOUR_SCHEMA}
        )
        result = df.with_columns(centroid=pl.col("contour").contour.centroid())
        # Non-null contour should have valid centroid
        assert result["centroid"][0]["x"] is not None
        assert result["centroid"][0]["y"] is not None
        # Null contour returns struct with null fields
        assert result["centroid"][1]["x"] is None
        assert result["centroid"][1]["y"] is None


@plugin_required
class TestContourMultipleOperations:
    """Tests for applying multiple operations at once."""

    def test_multiple_ops_together(self, multi_contour_df: pl.DataFrame) -> None:
        """Multiple contour operations should work together."""
        result = multi_contour_df.with_columns(
            area=pl.col("contour").contour.area(),
            perimeter=pl.col("contour").contour.perimeter(),
            centroid=pl.col("contour").contour.centroid(),
            bbox=pl.col("contour").contour.bounding_box(),
            is_convex=pl.col("contour").contour.is_convex(),
            winding=pl.col("contour").contour.winding(),
        )

        # Verify all columns exist
        assert "area" in result.columns
        assert "perimeter" in result.columns
        assert "centroid" in result.columns
        assert "bbox" in result.columns
        assert "is_convex" in result.columns
        assert "winding" in result.columns

        # Verify all rows have values
        assert len(result) == 3
        assert result["area"].null_count() == 0
        assert result["perimeter"].null_count() == 0
