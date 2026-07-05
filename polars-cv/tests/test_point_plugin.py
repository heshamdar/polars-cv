"""
Tests for point plugin operations.

These tests verify that point operations work correctly through the Polars plugin.
"""

import math

import numpy as np
import polars as pl

# Import the point namespace to register it
import polars_cv.geometry.points  # noqa: F401
from tests.conftest import plugin_required


# Mark all tests in this module as requiring the plugin
class TestPointTransforms:
    """Tests for point coordinate transformation operations."""

    @plugin_required
    def test_normalize_basic(self):
        """Test basic coordinate normalization."""
        df = pl.DataFrame({"pt": [{"x": 50.0, "y": 100.0}]})
        result = df.with_columns(normalized=pl.col("pt").point.normalize(100, 200))

        normalized = result["normalized"][0]
        assert abs(normalized["x"] - 0.5) < 1e-10
        assert abs(normalized["y"] - 0.5) < 1e-10

    @plugin_required
    def test_normalize_multiple_points(self):
        """Test normalization with multiple points."""
        df = pl.DataFrame(
            {
                "pt": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 100.0, "y": 100.0},
                    {"x": 50.0, "y": 25.0},
                ]
            }
        )
        result = df.with_columns(normalized=pl.col("pt").point.normalize(100, 100))

        assert abs(result["normalized"][0]["x"] - 0.0) < 1e-10
        assert abs(result["normalized"][1]["x"] - 1.0) < 1e-10
        assert abs(result["normalized"][2]["x"] - 0.5) < 1e-10
        assert abs(result["normalized"][2]["y"] - 0.25) < 1e-10

    @plugin_required
    def test_to_absolute_basic(self):
        """Test converting normalized coordinates to absolute."""
        df = pl.DataFrame({"pt": [{"x": 0.5, "y": 0.5}]})
        result = df.with_columns(absolute=pl.col("pt").point.to_absolute(100, 200))

        absolute = result["absolute"][0]
        assert abs(absolute["x"] - 50.0) < 1e-10
        assert abs(absolute["y"] - 100.0) < 1e-10

    @plugin_required
    def test_normalize_to_absolute_roundtrip(self):
        """Test that normalize and to_absolute are inverses."""
        df = pl.DataFrame({"pt": [{"x": 75.0, "y": 120.0}]})
        result = df.with_columns(
            roundtrip=pl.col("pt").point.normalize(200, 300).point.to_absolute(200, 300)
        )

        roundtrip = result["roundtrip"][0]
        assert abs(roundtrip["x"] - 75.0) < 1e-10
        assert abs(roundtrip["y"] - 120.0) < 1e-10

    @plugin_required
    def test_translate_basic(self):
        """Test basic point translation."""
        df = pl.DataFrame({"pt": [{"x": 10.0, "y": 20.0}]})
        result = df.with_columns(moved=pl.col("pt").point.translate(5.0, -10.0))

        moved = result["moved"][0]
        assert abs(moved["x"] - 15.0) < 1e-10
        assert abs(moved["y"] - 10.0) < 1e-10

    @plugin_required
    def test_translate_negative(self):
        """Test translation with negative offsets."""
        df = pl.DataFrame({"pt": [{"x": 100.0, "y": 100.0}]})
        result = df.with_columns(moved=pl.col("pt").point.translate(-50.0, -50.0))

        moved = result["moved"][0]
        assert abs(moved["x"] - 50.0) < 1e-10
        assert abs(moved["y"] - 50.0) < 1e-10

    @plugin_required
    def test_scale_basic(self):
        """Test basic point scaling."""
        df = pl.DataFrame({"pt": [{"x": 10.0, "y": 20.0}]})
        result = df.with_columns(scaled=pl.col("pt").point.scale(2.0, 0.5))

        scaled = result["scaled"][0]
        assert abs(scaled["x"] - 20.0) < 1e-10
        assert abs(scaled["y"] - 10.0) < 1e-10

    @plugin_required
    def test_scale_uniform(self):
        """Test uniform scaling."""
        df = pl.DataFrame({"pt": [{"x": 5.0, "y": 10.0}]})
        result = df.with_columns(scaled=pl.col("pt").point.scale(3.0, 3.0))

        scaled = result["scaled"][0]
        assert abs(scaled["x"] - 15.0) < 1e-10
        assert abs(scaled["y"] - 30.0) < 1e-10

    @plugin_required
    def test_chained_transforms(self):
        """Test chaining multiple transforms."""
        df = pl.DataFrame({"pt": [{"x": 10.0, "y": 10.0}]})
        result = df.with_columns(
            result=pl.col("pt")
            .point.translate(10.0, 10.0)
            .point.scale(2.0, 2.0)
            .point.translate(-20.0, -20.0)
        )

        result_pt = result["result"][0]
        # (10+10)*2 - 20 = 20, same for y
        assert abs(result_pt["x"] - 20.0) < 1e-10
        assert abs(result_pt["y"] - 20.0) < 1e-10


class TestPointDistances:
    """Tests for point distance operations."""

    @plugin_required
    def test_euclidean_distance_basic(self):
        """Test basic Euclidean distance (3-4-5 triangle)."""
        df = pl.DataFrame({"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 3.0, "y": 4.0}]})
        result = df.with_columns(dist=pl.col("p1").point.distance(pl.col("p2")))

        assert abs(result["dist"][0] - 5.0) < 1e-10

    @plugin_required
    def test_euclidean_distance_same_point(self):
        """Test distance to same point is zero."""
        df = pl.DataFrame({"p1": [{"x": 5.0, "y": 5.0}], "p2": [{"x": 5.0, "y": 5.0}]})
        result = df.with_columns(dist=pl.col("p1").point.distance(pl.col("p2")))

        assert abs(result["dist"][0]) < 1e-10

    @plugin_required
    def test_euclidean_distance_multiple(self):
        """Test distance with multiple point pairs."""
        df = pl.DataFrame(
            {
                "p1": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.0, "y": 1.0},
                    {"x": 0.0, "y": 0.0},
                ],
                "p2": [
                    {"x": 1.0, "y": 0.0},
                    {"x": 1.0, "y": 1.0},
                    {"x": 10.0, "y": 0.0},
                ],
            }
        )
        result = df.with_columns(dist=pl.col("p1").point.distance(pl.col("p2")))

        assert abs(result["dist"][0] - 1.0) < 1e-10
        assert abs(result["dist"][1] - 0.0) < 1e-10
        assert abs(result["dist"][2] - 10.0) < 1e-10

    @plugin_required
    def test_manhattan_distance_basic(self):
        """Test basic Manhattan distance."""
        df = pl.DataFrame({"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 3.0, "y": 4.0}]})
        result = df.with_columns(
            dist=pl.col("p1").point.manhattan_distance(pl.col("p2"))
        )

        assert abs(result["dist"][0] - 7.0) < 1e-10

    @plugin_required
    def test_manhattan_distance_same_point(self):
        """Test Manhattan distance to same point is zero."""
        df = pl.DataFrame({"p1": [{"x": 5.0, "y": 5.0}], "p2": [{"x": 5.0, "y": 5.0}]})
        result = df.with_columns(
            dist=pl.col("p1").point.manhattan_distance(pl.col("p2"))
        )

        assert abs(result["dist"][0]) < 1e-10


class TestPointToContour:
    """Tests for point-to-contour operations."""

    @plugin_required
    def test_distance_to_contour_outside(self):
        """Test distance from point outside contour."""
        # Square from (0,0) to (10,10)
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        df = pl.DataFrame({"pt": [{"x": 15.0, "y": 5.0}], "contour": [contour]})
        result = df.with_columns(
            dist=pl.col("pt").point.distance_to_contour(pl.col("contour"))
        )

        # Point is 5 units from right edge
        assert abs(result["dist"][0] - 5.0) < 1e-10

    @plugin_required
    def test_distance_to_contour_inside(self):
        """Test distance from point inside contour."""
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        df = pl.DataFrame({"pt": [{"x": 5.0, "y": 5.0}], "contour": [contour]})
        result = df.with_columns(
            dist=pl.col("pt").point.distance_to_contour(pl.col("contour"))
        )

        # Point is 5 units from all edges (center of square)
        assert abs(result["dist"][0] - 5.0) < 1e-10

    @plugin_required
    def test_distance_to_contour_on_boundary(self):
        """Test distance from point on contour boundary."""
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        df = pl.DataFrame({"pt": [{"x": 10.0, "y": 5.0}], "contour": [contour]})
        result = df.with_columns(
            dist=pl.col("pt").point.distance_to_contour(pl.col("contour"))
        )

        # Point is on the boundary
        assert abs(result["dist"][0]) < 1e-10

    @plugin_required
    def test_signed_distance_outside(self):
        """Test signed distance from point outside contour (positive)."""
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        df = pl.DataFrame({"pt": [{"x": 15.0, "y": 5.0}], "contour": [contour]})
        result = df.with_columns(
            sdf=pl.col("pt").point.signed_distance_to_contour(pl.col("contour"))
        )

        # Point is outside, so positive
        assert result["sdf"][0] > 0
        assert abs(result["sdf"][0] - 5.0) < 1e-10

    @plugin_required
    def test_signed_distance_inside(self):
        """Test signed distance from point inside contour (negative)."""
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        df = pl.DataFrame({"pt": [{"x": 5.0, "y": 5.0}], "contour": [contour]})
        result = df.with_columns(
            sdf=pl.col("pt").point.signed_distance_to_contour(pl.col("contour"))
        )

        # Point is inside, so negative
        assert result["sdf"][0] < 0
        assert abs(result["sdf"][0] + 5.0) < 1e-10  # -5.0

    @plugin_required
    def test_nearest_point_on_contour(self):
        """Test finding nearest point on contour."""
        contour = {
            "exterior": [
                {"x": 0.0, "y": 0.0},
                {"x": 10.0, "y": 0.0},
                {"x": 10.0, "y": 10.0},
                {"x": 0.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }
        df = pl.DataFrame({"pt": [{"x": 15.0, "y": 5.0}], "contour": [contour]})
        result = df.with_columns(
            nearest=pl.col("pt").point.nearest_point_on_contour(pl.col("contour"))
        )

        # Nearest point should be on right edge at (10, 5)
        nearest = result["nearest"][0]
        assert abs(nearest["x"] - 10.0) < 1e-10
        assert abs(nearest["y"] - 5.0) < 1e-10


class TestGeometricOperations:
    """Tests for geometric point operations."""

    @plugin_required
    def test_angle_to_right(self):
        """Test angle to point directly to the right."""
        df = pl.DataFrame({"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 1.0, "y": 0.0}]})
        result = df.with_columns(angle=pl.col("p1").point.angle_to(pl.col("p2")))

        assert abs(result["angle"][0] - 0.0) < 1e-10

    @plugin_required
    def test_angle_to_up(self):
        """Test angle to point directly up."""
        df = pl.DataFrame({"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 0.0, "y": 1.0}]})
        result = df.with_columns(angle=pl.col("p1").point.angle_to(pl.col("p2")))

        assert abs(result["angle"][0] - math.pi / 2) < 1e-10

    @plugin_required
    def test_angle_to_left(self):
        """Test angle to point directly to the left."""
        df = pl.DataFrame({"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": -1.0, "y": 0.0}]})
        result = df.with_columns(angle=pl.col("p1").point.angle_to(pl.col("p2")))

        assert abs(abs(result["angle"][0]) - math.pi) < 1e-10

    @plugin_required
    def test_rotate_90_degrees(self):
        """Test rotating point 90 degrees around origin."""
        df = pl.DataFrame({"pt": [{"x": 1.0, "y": 0.0}]})
        result = df.with_columns(rotated=pl.col("pt").point.rotate(math.pi / 2))

        rotated = result["rotated"][0]
        assert abs(rotated["x"] - 0.0) < 1e-10
        assert abs(rotated["y"] - 1.0) < 1e-10

    @plugin_required
    def test_rotate_180_degrees(self):
        """Test rotating point 180 degrees around origin."""
        df = pl.DataFrame({"pt": [{"x": 1.0, "y": 0.0}]})
        result = df.with_columns(rotated=pl.col("pt").point.rotate(math.pi))

        rotated = result["rotated"][0]
        assert abs(rotated["x"] + 1.0) < 1e-10  # -1.0
        assert abs(rotated["y"]) < 1e-10

    @plugin_required
    def test_rotate_around_custom_origin(self):
        """Test rotating point around a custom origin."""
        df = pl.DataFrame(
            {
                "pt": [{"x": 2.0, "y": 0.0}],
                "origin": [{"x": 1.0, "y": 0.0}],
            }
        )
        result = df.with_columns(
            rotated=pl.col("pt").point.rotate(math.pi / 2, origin=pl.col("origin"))
        )

        rotated = result["rotated"][0]
        # Rotating (2,0) around (1,0) by 90 degrees gives (1,1)
        assert abs(rotated["x"] - 1.0) < 1e-10
        assert abs(rotated["y"] - 1.0) < 1e-10

    @plugin_required
    def test_midpoint_basic(self):
        """Test computing midpoint between two points."""
        df = pl.DataFrame(
            {"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 10.0, "y": 10.0}]}
        )
        result = df.with_columns(mid=pl.col("p1").point.midpoint(pl.col("p2")))

        mid = result["mid"][0]
        assert abs(mid["x"] - 5.0) < 1e-10
        assert abs(mid["y"] - 5.0) < 1e-10

    @plugin_required
    def test_midpoint_same_point(self):
        """Test midpoint of same point is itself."""
        df = pl.DataFrame({"p1": [{"x": 5.0, "y": 5.0}], "p2": [{"x": 5.0, "y": 5.0}]})
        result = df.with_columns(mid=pl.col("p1").point.midpoint(pl.col("p2")))

        mid = result["mid"][0]
        assert abs(mid["x"] - 5.0) < 1e-10
        assert abs(mid["y"] - 5.0) < 1e-10

    @plugin_required
    def test_interpolate_at_start(self):
        """Test interpolation at t=0 returns first point."""
        df = pl.DataFrame(
            {"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 10.0, "y": 10.0}]}
        )
        result = df.with_columns(
            interp=pl.col("p1").point.interpolate(pl.col("p2"), t=0.0)
        )

        interp = result["interp"][0]
        assert abs(interp["x"] - 0.0) < 1e-10
        assert abs(interp["y"] - 0.0) < 1e-10

    @plugin_required
    def test_interpolate_at_end(self):
        """Test interpolation at t=1 returns second point."""
        df = pl.DataFrame(
            {"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 10.0, "y": 10.0}]}
        )
        result = df.with_columns(
            interp=pl.col("p1").point.interpolate(pl.col("p2"), t=1.0)
        )

        interp = result["interp"][0]
        assert abs(interp["x"] - 10.0) < 1e-10
        assert abs(interp["y"] - 10.0) < 1e-10

    @plugin_required
    def test_interpolate_at_quarter(self):
        """Test interpolation at t=0.25."""
        df = pl.DataFrame(
            {"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 100.0, "y": 100.0}]}
        )
        result = df.with_columns(
            interp=pl.col("p1").point.interpolate(pl.col("p2"), t=0.25)
        )

        interp = result["interp"][0]
        assert abs(interp["x"] - 25.0) < 1e-10
        assert abs(interp["y"] - 25.0) < 1e-10

    @plugin_required
    def test_interpolate_extrapolate(self):
        """Test interpolation with t > 1 extrapolates."""
        df = pl.DataFrame(
            {"p1": [{"x": 0.0, "y": 0.0}], "p2": [{"x": 10.0, "y": 10.0}]}
        )
        result = df.with_columns(
            interp=pl.col("p1").point.interpolate(pl.col("p2"), t=2.0)
        )

        interp = result["interp"][0]
        assert abs(interp["x"] - 20.0) < 1e-10
        assert abs(interp["y"] - 20.0) < 1e-10

    @plugin_required
    def test_within_bbox_inside(self):
        """Test point inside bounding box."""
        df = pl.DataFrame(
            {
                "pt": [{"x": 5.0, "y": 5.0}],
                "bbox": [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}],
            }
        )
        result = df.with_columns(inside=pl.col("pt").point.within_bbox(pl.col("bbox")))

        assert result["inside"][0] is True

    @plugin_required
    def test_within_bbox_outside(self):
        """Test point outside bounding box."""
        df = pl.DataFrame(
            {
                "pt": [{"x": 15.0, "y": 5.0}],
                "bbox": [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}],
            }
        )
        result = df.with_columns(inside=pl.col("pt").point.within_bbox(pl.col("bbox")))

        assert result["inside"][0] is False

    @plugin_required
    def test_within_bbox_on_boundary(self):
        """Test point on bounding box boundary (inclusive)."""
        df = pl.DataFrame(
            {
                "pt": [{"x": 10.0, "y": 5.0}],
                "bbox": [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}],
            }
        )
        result = df.with_columns(inside=pl.col("pt").point.within_bbox(pl.col("bbox")))

        assert result["inside"][0] is True


class TestExtraction:
    """Tests for point coordinate extraction."""

    @plugin_required
    def test_x_extraction(self):
        """Test extracting X coordinate."""
        df = pl.DataFrame({"pt": [{"x": 5.0, "y": 10.0}]})
        result = df.with_columns(x=pl.col("pt").point.x())

        assert abs(result["x"][0] - 5.0) < 1e-10

    @plugin_required
    def test_y_extraction(self):
        """Test extracting Y coordinate."""
        df = pl.DataFrame({"pt": [{"x": 5.0, "y": 10.0}]})
        result = df.with_columns(y=pl.col("pt").point.y())

        assert abs(result["y"][0] - 10.0) < 1e-10


class TestNullHandling:
    """Tests for null value handling."""

    @plugin_required
    def test_transform_with_null(self):
        """Test transform operations handle nulls correctly."""
        df = pl.DataFrame(
            {"pt": [{"x": 10.0, "y": 20.0}, None, {"x": 30.0, "y": 40.0}]}
        )
        result = df.with_columns(translated=pl.col("pt").point.translate(5.0, 5.0))

        assert result["translated"][0]["x"] == 15.0
        # Polars represents null structs as structs with null fields
        assert result["translated"][1]["x"] is None
        assert result["translated"][1]["y"] is None
        assert result["translated"][2]["x"] == 35.0

    @plugin_required
    def test_distance_with_null(self):
        """Test distance operations handle nulls correctly."""
        df = pl.DataFrame(
            {
                "p1": [{"x": 0.0, "y": 0.0}, None, {"x": 0.0, "y": 0.0}],
                "p2": [{"x": 3.0, "y": 4.0}, {"x": 1.0, "y": 1.0}, None],
            }
        )
        result = df.with_columns(dist=pl.col("p1").point.distance(pl.col("p2")))

        assert abs(result["dist"][0] - 5.0) < 1e-10
        assert result["dist"][1] is None
        assert result["dist"][2] is None


class TestReferenceImplementation:
    """Reference tests comparing against numpy/scipy implementations."""

    @plugin_required
    def test_distance_reference(self):
        """Verify point distance matches numpy."""
        np.random.seed(42)
        n = 100
        points_a = np.random.rand(n, 2) * 100
        points_b = np.random.rand(n, 2) * 100

        # Reference: numpy
        expected = np.sqrt(np.sum((points_a - points_b) ** 2, axis=1))

        # Our implementation
        df = pl.DataFrame(
            {
                "p1": [{"x": float(p[0]), "y": float(p[1])} for p in points_a],
                "p2": [{"x": float(p[0]), "y": float(p[1])} for p in points_b],
            }
        )
        result = df.with_columns(dist=pl.col("p1").point.distance(pl.col("p2")))

        np.testing.assert_allclose(result["dist"].to_numpy(), expected, rtol=1e-10)

    @plugin_required
    def test_manhattan_distance_reference(self):
        """Verify Manhattan distance matches numpy."""
        np.random.seed(42)
        n = 100
        points_a = np.random.rand(n, 2) * 100
        points_b = np.random.rand(n, 2) * 100

        # Reference: numpy
        expected = np.sum(np.abs(points_a - points_b), axis=1)

        # Our implementation
        df = pl.DataFrame(
            {
                "p1": [{"x": float(p[0]), "y": float(p[1])} for p in points_a],
                "p2": [{"x": float(p[0]), "y": float(p[1])} for p in points_b],
            }
        )
        result = df.with_columns(
            dist=pl.col("p1").point.manhattan_distance(pl.col("p2"))
        )

        np.testing.assert_allclose(result["dist"].to_numpy(), expected, rtol=1e-10)

    @plugin_required
    def test_rotation_reference(self):
        """Verify rotation matches numpy rotation matrix."""
        np.random.seed(42)
        n = 50
        points = np.random.rand(n, 2) * 100
        angle = np.pi / 4  # 45 degrees

        # Reference: numpy rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        expected = points @ rotation_matrix.T

        # Our implementation
        df = pl.DataFrame(
            {"pt": [{"x": float(p[0]), "y": float(p[1])} for p in points]}
        )
        result = df.with_columns(rotated=pl.col("pt").point.rotate(angle))

        result_x = result["rotated"].struct.field("x").to_numpy()
        result_y = result["rotated"].struct.field("y").to_numpy()

        np.testing.assert_allclose(result_x, expected[:, 0], rtol=1e-10)
        np.testing.assert_allclose(result_y, expected[:, 1], rtol=1e-10)
