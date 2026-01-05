"""
Reference tests for contour operations using OpenCV.

These tests establish the expected behavior for contour geometric operations,
serving as ground truth for polars-cv implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass


class TestContourOpsReference:
    """Establish expected behavior for contour operations using OpenCV."""

    def test_area_simple_square_reference(self, simple_contour: np.ndarray) -> None:
        """
        Contour area for simple square.

        Square from (10,10) to (90,90) = 80x80 = 6400 pixels.
        """
        # OpenCV expects (N, 1, 2) shape for contours
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.int32)
        area = cv2.contourArea(contour_cv)

        # Expected: 80 * 80 = 6400
        assert abs(area - 6400) < 10  # Allow small rounding error

    def test_area_triangle_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """
        Contour area for triangle.
        """
        triangle = standard_contours["triangle"]
        contour_cv = triangle.reshape(-1, 1, 2).astype(np.int32)
        area = cv2.contourArea(contour_cv)

        # Triangle with vertices at (50,10), (10,90), (90,90)
        # Base = 80, Height = 80, Area = 0.5 * 80 * 80 = 3200
        assert abs(area - 3200) < 50

    def test_area_with_hole_reference(
        self, contour_with_hole: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Area calculation with hole subtraction.

        Net area = outer area - hole area.
        """
        outer, hole = contour_with_hole
        outer_cv = outer.reshape(-1, 1, 2).astype(np.int32)
        hole_cv = hole.reshape(-1, 1, 2).astype(np.int32)

        outer_area = cv2.contourArea(outer_cv)
        hole_area = cv2.contourArea(hole_cv)
        net_area = outer_area - hole_area

        # Outer: 100x100 = 10000
        # Hole: π * 20² ≈ 1257
        expected_hole_area = np.pi * 20**2

        assert abs(outer_area - 10000) < 10
        assert abs(hole_area - expected_hole_area) < 100
        assert net_area < outer_area

    def test_perimeter_reference(self, simple_contour: np.ndarray) -> None:
        """
        Contour perimeter (arc length).

        Square with side 80 has perimeter 320.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.int32)
        perimeter = cv2.arcLength(contour_cv, closed=True)

        # Expected: 4 * 80 = 320
        assert abs(perimeter - 320) < 5

    def test_perimeter_circle_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """
        Circle perimeter should approach 2πr.
        """
        circle = standard_contours["circle"]
        contour_cv = circle.reshape(-1, 1, 2).astype(np.int32)
        perimeter = cv2.arcLength(contour_cv, closed=True)

        # Circle with r=40: 2πr ≈ 251.3
        expected = 2 * np.pi * 40
        # Allow larger error due to polygon approximation
        assert abs(perimeter - expected) < 20

    def test_winding_direction_from_area_reference(
        self, simple_contour: np.ndarray
    ) -> None:
        """
        Winding direction determined from signed area.

        CCW = positive area, CW = negative area (Shoelace formula).
        Note: The winding convention depends on coordinate system.
        In image coordinates (y-axis down), CW visually appears CCW mathematically.
        """
        contour = simple_contour

        # Compute signed area using Shoelace formula
        n = len(contour)
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += contour[i, 0] * contour[j, 1]
            signed_area -= contour[j, 0] * contour[i, 1]
        signed_area /= 2

        winding = "ccw" if signed_area > 0 else "cw"

        # Our fixture is defined in image coordinates (top-left origin)
        # The point order [10,10], [10,90], [90,90], [90,10] is CW in image coords
        assert winding == "cw"

    def test_flip_reverses_winding_reference(self, simple_contour: np.ndarray) -> None:
        """
        Flipping (reversing) point order changes winding direction.
        """
        original = simple_contour
        flipped = original[::-1].copy()

        def signed_area(contour: np.ndarray) -> float:
            n = len(contour)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += contour[i, 0] * contour[j, 1]
                area -= contour[j, 0] * contour[i, 1]
            return area / 2

        orig_signed = signed_area(original)
        flip_signed = signed_area(flipped)

        # Signs should be opposite
        assert orig_signed < 0  # CW in image coordinates
        assert flip_signed > 0  # CCW after flip
        # Magnitudes should be equal
        assert np.isclose(abs(orig_signed), abs(flip_signed))

    def test_normalize_coordinates_reference(self, simple_contour: np.ndarray) -> None:
        """
        Normalize pixel coords to [0,1] range.
        """
        ref_width, ref_height = 100.0, 100.0
        normalized = simple_contour / np.array([ref_width, ref_height])

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert np.allclose(normalized[0], [0.1, 0.1])  # 10/100

    def test_to_absolute_coordinates_reference(self) -> None:
        """
        Convert normalized coords back to pixel coords.
        """
        normalized = np.array(
            [[0.1, 0.1], [0.1, 0.9], [0.9, 0.9], [0.9, 0.1]], dtype=np.float64
        )
        ref_width, ref_height = 100.0, 100.0

        absolute = normalized * np.array([ref_width, ref_height])

        expected = np.array([[10, 10], [10, 90], [90, 90], [90, 10]], dtype=np.float64)
        assert np.allclose(absolute, expected)

    def test_centroid_reference(self, simple_contour: np.ndarray) -> None:
        """
        Compute contour centroid using moments.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.int32)
        moments = cv2.moments(contour_cv)

        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        # Square from (10,10) to (90,90) has centroid at (50, 50)
        assert abs(cx - 50) < 1
        assert abs(cy - 50) < 1

    def test_bounding_box_reference(self, simple_contour: np.ndarray) -> None:
        """
        Compute axis-aligned bounding box.

        Note: OpenCV's boundingRect returns width/height that include the endpoint,
        so a contour from 10 to 90 has width 81 (includes both endpoints).
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.int32)
        x, y, w, h = cv2.boundingRect(contour_cv)

        assert x == 10
        assert y == 10
        # OpenCV includes endpoints: 90 - 10 + 1 = 81
        assert w == 81
        assert h == 81

    def test_convex_hull_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """
        Convex hull of a convex shape is itself.
        """
        square = standard_contours["square"]
        contour_cv = square.reshape(-1, 1, 2).astype(np.float32)
        hull = cv2.convexHull(contour_cv)

        # Square is already convex, hull should have same number of points
        assert len(hull) == len(square)

    def test_is_convex_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """
        Check if contour is convex.
        """
        square = standard_contours["square"]
        contour_cv = square.reshape(-1, 1, 2).astype(np.float32)

        is_convex = cv2.isContourConvex(contour_cv)

        assert is_convex is True

    def test_is_not_convex_reference(self) -> None:
        """
        Non-convex contour detection.
        """
        # L-shaped contour (concave)
        l_shape = np.array(
            [[0, 0], [0, 100], [50, 100], [50, 50], [100, 50], [100, 0]],
            dtype=np.float32,
        )
        contour_cv = l_shape.reshape(-1, 1, 2)

        is_convex = cv2.isContourConvex(contour_cv)

        assert is_convex is False

    def test_translate_reference(self, simple_contour: np.ndarray) -> None:
        """
        Translate contour by offset.
        """
        dx, dy = 10, 20
        translated = simple_contour + np.array([dx, dy])

        expected = np.array(
            [[20, 30], [20, 110], [100, 110], [100, 30]], dtype=np.float32
        )
        assert np.allclose(translated, expected)

    def test_scale_reference(self, simple_contour: np.ndarray) -> None:
        """
        Scale contour relative to origin.
        """
        sx, sy = 0.5, 0.5
        scaled = simple_contour * np.array([sx, sy])

        expected = np.array([[5, 5], [5, 45], [45, 45], [45, 5]], dtype=np.float32)
        assert np.allclose(scaled, expected)

    def test_scale_around_centroid_reference(self, simple_contour: np.ndarray) -> None:
        """
        Scale contour around its centroid.
        """
        # Compute centroid
        centroid = simple_contour.mean(axis=0)

        # Scale around centroid
        sx, sy = 0.5, 0.5
        scaled = (simple_contour - centroid) * np.array([sx, sy]) + centroid

        # Centroid should remain the same
        new_centroid = scaled.mean(axis=0)
        assert np.allclose(centroid, new_centroid)

    def test_simplify_reference(self) -> None:
        """
        Douglas-Peucker simplification reduces points.
        """
        # Circle has many points
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        circle = np.column_stack(
            [50 + 40 * np.cos(angles), 50 + 40 * np.sin(angles)]
        ).astype(np.float32)

        contour_cv = circle.reshape(-1, 1, 2)

        # Low epsilon = more points kept
        simplified_low = cv2.approxPolyDP(contour_cv, epsilon=1.0, closed=True)
        # High epsilon = fewer points
        simplified_high = cv2.approxPolyDP(contour_cv, epsilon=10.0, closed=True)

        assert len(simplified_high) < len(simplified_low)
        assert len(simplified_low) < len(circle)

    def test_point_in_polygon_reference(self, simple_contour: np.ndarray) -> None:
        """
        Point-in-polygon test.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.float32)

        # Point inside
        inside = cv2.pointPolygonTest(contour_cv, (50.0, 50.0), measureDist=False)
        assert inside > 0  # Positive = inside

        # Point outside
        outside = cv2.pointPolygonTest(contour_cv, (0.0, 0.0), measureDist=False)
        assert outside < 0  # Negative = outside

        # Point on edge
        on_edge = cv2.pointPolygonTest(contour_cv, (10.0, 50.0), measureDist=False)
        assert on_edge == 0  # Zero = on contour
