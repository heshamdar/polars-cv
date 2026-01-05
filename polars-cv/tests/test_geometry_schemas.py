"""
Tests for geometry schemas.

These tests verify the schema definitions and helper functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_cv.geometry import (
    BBOX_SCHEMA,
    CONTOUR_SCHEMA,
    POINT_SCHEMA,
    POINT_SET_SCHEMA,
    RING_SCHEMA,
)
from polars_cv.geometry.schemas import (
    bbox_from_center,
    bbox_from_corners,
    contour_from_points,
    validate_contour,
    validate_point,
)

if TYPE_CHECKING:
    pass


class TestPointSchema:
    """Tests for POINT_SCHEMA."""

    def test_point_schema_structure(self) -> None:
        """POINT_SCHEMA should have x and y Float64 fields."""
        assert POINT_SCHEMA == pl.Struct(
            [
                pl.Field("x", pl.Float64),
                pl.Field("y", pl.Float64),
            ]
        )

    def test_create_point_series(self) -> None:
        """Can create a Series of points."""
        points = [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 50.0},
            {"x": 25.5, "y": 75.5},
        ]
        series = pl.Series("points", points, dtype=POINT_SCHEMA)

        assert series.len() == 3
        assert series.dtype == POINT_SCHEMA


class TestContourSchema:
    """Tests for CONTOUR_SCHEMA."""

    def test_contour_schema_structure(self) -> None:
        """CONTOUR_SCHEMA should have exterior, holes, is_closed fields."""
        # Check it's a Struct with the right fields
        assert CONTOUR_SCHEMA.base_type() == pl.Struct

        field_names = [f.name for f in CONTOUR_SCHEMA.fields]
        assert "exterior" in field_names
        assert "holes" in field_names
        assert "is_closed" in field_names

    def test_create_simple_contour(self) -> None:
        """Can create a simple contour without holes."""
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0},
                {"x": 0.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        series = pl.Series("contour", [contour], dtype=CONTOUR_SCHEMA)

        assert series.len() == 1

    def test_create_contour_with_hole(self) -> None:
        """Can create a contour with holes."""
        contour = {
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
        series = pl.Series("contour", [contour], dtype=CONTOUR_SCHEMA)

        assert series.len() == 1


class TestBBoxSchema:
    """Tests for BBOX_SCHEMA."""

    def test_bbox_schema_structure(self) -> None:
        """BBOX_SCHEMA should have x, y, width, height fields."""
        assert BBOX_SCHEMA == pl.Struct(
            [
                pl.Field("x", pl.Float64),
                pl.Field("y", pl.Float64),
                pl.Field("width", pl.Float64),
                pl.Field("height", pl.Float64),
            ]
        )

    def test_create_bbox_series(self) -> None:
        """Can create a Series of bounding boxes."""
        bboxes = [
            {"x": 10.0, "y": 10.0, "width": 50.0, "height": 30.0},
            {"x": 0.0, "y": 0.0, "width": 100.0, "height": 100.0},
        ]
        series = pl.Series("bboxes", bboxes, dtype=BBOX_SCHEMA)

        assert series.len() == 2


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_point_valid(self) -> None:
        """Valid point should pass validation."""
        assert validate_point({"x": 0.0, "y": 0.0}) is True
        assert validate_point({"x": 100, "y": 200}) is True  # int also works

    def test_validate_point_invalid(self) -> None:
        """Invalid points should fail validation."""
        assert validate_point({}) is False
        assert validate_point({"x": 0.0}) is False
        assert validate_point({"y": 0.0}) is False
        assert validate_point({"x": "0", "y": 0.0}) is False
        assert validate_point([0, 0]) is False

    def test_validate_contour_valid(self) -> None:
        """Valid contour should pass validation."""
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
                {"x": 100.0, "y": 100.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        assert validate_contour(contour) is True

    def test_validate_contour_too_few_points(self) -> None:
        """Contour with < 3 points should fail."""
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 100.0, "y": 0.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        assert validate_contour(contour) is False

    def test_validate_contour_no_exterior(self) -> None:
        """Contour without exterior should fail."""
        contour = {"holes": [], "is_closed": True}
        assert validate_contour(contour) is False


class TestHelperFunctions:
    """Tests for schema helper functions."""

    def test_contour_from_points_simple(self) -> None:
        """contour_from_points creates valid contour dict."""
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        contour = contour_from_points(points)

        assert "exterior" in contour
        assert len(contour["exterior"]) == 4
        assert contour["holes"] == []
        assert contour["is_closed"] is True

        # Validate structure
        assert validate_contour(contour) is True

    def test_contour_from_points_with_hole(self) -> None:
        """contour_from_points can create contour with holes."""
        points = [(0, 0), (100, 0), (100, 100), (0, 100)]
        holes = [[(25, 25), (75, 25), (75, 75), (25, 75)]]
        contour = contour_from_points(points, holes=holes)

        assert len(contour["holes"]) == 1
        assert len(contour["holes"][0]) == 4

    def test_bbox_from_corners(self) -> None:
        """bbox_from_corners creates valid bbox dict."""
        bbox = bbox_from_corners(10, 20, 60, 80)

        assert bbox["x"] == 10
        assert bbox["y"] == 20
        assert bbox["width"] == 50
        assert bbox["height"] == 60

    def test_bbox_from_corners_reversed(self) -> None:
        """bbox_from_corners handles reversed corners."""
        bbox = bbox_from_corners(60, 80, 10, 20)

        assert bbox["x"] == 10
        assert bbox["y"] == 20
        assert bbox["width"] == 50
        assert bbox["height"] == 60

    def test_bbox_from_center(self) -> None:
        """bbox_from_center creates valid bbox dict."""
        bbox = bbox_from_center(50, 50, 40, 20)

        assert bbox["x"] == 30
        assert bbox["y"] == 40
        assert bbox["width"] == 40
        assert bbox["height"] == 20


class TestRingSchema:
    """Tests for RING_SCHEMA."""

    def test_ring_is_list_of_points(self) -> None:
        """RING_SCHEMA should be a list of POINT_SCHEMA."""
        assert RING_SCHEMA == pl.List(POINT_SCHEMA)


class TestPointSetSchema:
    """Tests for POINT_SET_SCHEMA."""

    def test_point_set_is_list_of_points(self) -> None:
        """POINT_SET_SCHEMA should be a list of POINT_SCHEMA."""
        assert POINT_SET_SCHEMA == pl.List(POINT_SCHEMA)

    def test_create_point_set(self) -> None:
        """Can create a Series of point sets."""
        point_sets = [
            [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 10.0}],
            [{"x": 50.0, "y": 50.0}],
        ]
        series = pl.Series("keypoints", point_sets, dtype=POINT_SET_SCHEMA)

        assert series.len() == 2
