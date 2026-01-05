"""
Reference tests for contour rasterization using OpenCV.

These tests establish the expected behavior for converting contours
to binary masks, serving as ground truth for polars-cv.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass


class TestRasterizeReference:
    """Establish expected behavior for contour to mask conversion using OpenCV."""

    def test_rasterize_filled_square_reference(
        self, simple_contour: np.ndarray
    ) -> None:
        """
        Rasterize filled square contour to binary mask.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.int32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask, [contour_cv], 0, 255, -1)  # -1 = filled

        # Verify properties
        assert mask.dtype == np.uint8
        assert mask.shape == (100, 100)

        # Check specific regions
        assert mask[50, 50] == 255  # Center inside
        assert mask[5, 5] == 0  # Corner outside
        assert mask[15, 15] == 255  # Just inside

    def test_rasterize_filled_triangle_reference(
        self, triangle_contour: np.ndarray
    ) -> None:
        """
        Rasterize filled triangle contour to binary mask.
        """
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask, [triangle_contour], 0, 255, -1)

        assert mask.dtype == np.uint8
        assert mask[50, 50] == 255  # Center inside
        assert mask[5, 5] == 0  # Corner outside

    def test_rasterize_with_hole_reference(
        self, contour_with_hole: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Rasterize contour with hole - hole region should be empty.

        Note: The contour_with_hole fixture has outer from (0,0) to (100,100),
        so (5,5) is actually inside the outer contour.
        """
        outer, hole = contour_with_hole
        outer_cv = outer.reshape(-1, 1, 2).astype(np.int32)
        hole_cv = hole.reshape(-1, 1, 2).astype(np.int32)

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask, [outer_cv], 0, 255, -1)  # Fill outer
        cv2.drawContours(mask, [hole_cv], 0, 0, -1)  # Punch hole

        assert mask[50, 50] == 0  # Center is hole
        assert mask[5, 50] == 255  # Near edge but inside outer, outside hole

    def test_rasterize_explicit_dimensions_reference(
        self, triangle_contour: np.ndarray
    ) -> None:
        """
        Rasterize to explicit output dimensions.
        """
        for height, width in [(100, 100), (200, 200), (50, 50)]:
            # Scale contour to match dimensions
            scale = np.array([width / 100, height / 100])
            scaled_contour = (triangle_contour * scale).astype(np.int32)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [scaled_contour], 0, 255, -1)

            assert mask.shape == (height, width)

    def test_rasterize_custom_fill_value_reference(
        self, simple_contour: np.ndarray
    ) -> None:
        """
        Rasterize with custom fill value.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.int32)

        fill_value = 128
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask, [contour_cv], 0, fill_value, -1)

        assert mask[50, 50] == fill_value
        assert mask[5, 5] == 0

    def test_rasterize_outline_only_reference(self, simple_contour: np.ndarray) -> None:
        """
        Rasterize contour outline (not filled).
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.int32)

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask, [contour_cv], 0, 255, 1)  # thickness=1

        # Outline should be drawn, interior empty
        assert mask[10, 10] == 255  # On edge
        assert mask[50, 50] == 0  # Interior empty

    def test_rasterize_anti_alias_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """
        Rasterize with anti-aliasing (smooth edges).

        Note: OpenCV drawContours with lineType=cv2.LINE_AA for anti-aliasing.
        """
        circle = standard_contours["circle"]
        contour_cv = circle.reshape(-1, 1, 2).astype(np.int32)

        # Without anti-aliasing
        mask_no_aa = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask_no_aa, [contour_cv], 0, 255, -1, lineType=cv2.LINE_8)

        # With anti-aliasing (for outline)
        mask_aa = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask_aa, [contour_cv], 0, 255, 1, lineType=cv2.LINE_AA)

        # Both should have content
        assert mask_no_aa.max() == 255
        assert mask_aa.max() == 255

    def test_rasterize_multiple_contours_reference(self) -> None:
        """
        Rasterize multiple contours to single mask.
        """
        c1 = np.array([[10, 10], [10, 40], [40, 40], [40, 10]], dtype=np.int32)
        c2 = np.array([[60, 60], [60, 90], [90, 90], [90, 60]], dtype=np.int32)

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask, [c1, c2], -1, 255, -1)  # -1 = all contours

        assert mask[25, 25] == 255  # Inside first
        assert mask[75, 75] == 255  # Inside second
        assert mask[50, 50] == 0  # Between (empty)

    def test_rasterize_contour_set_reference(self) -> None:
        """
        Rasterize a set of contours with hierarchy.
        """
        # Outer contour
        outer = np.array([[5, 5], [5, 95], [95, 95], [95, 5]], dtype=np.int32)
        # Inner hole
        inner = np.array([[20, 20], [20, 80], [80, 80], [80, 20]], dtype=np.int32)

        # Draw with hierarchy (outer filled, inner as hole)
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask, [outer], 0, 255, -1)
        cv2.drawContours(mask, [inner], 0, 0, -1)

        # Check regions
        assert mask[10, 10] == 255  # Between outer edge and hole
        assert mask[50, 50] == 0  # Inside hole
        assert mask[2, 2] == 0  # Outside outer

    def test_rasterize_preserves_aspect_ratio_reference(self) -> None:
        """
        Rasterizing to different dimensions preserves relative positions.
        """
        # Normalized contour (0-1 range)
        normalized = np.array(
            [[0.1, 0.1], [0.1, 0.9], [0.9, 0.9], [0.9, 0.1]], dtype=np.float32
        )

        for height, width in [(100, 100), (200, 100), (100, 200)]:
            # Scale to target dimensions
            scaled = (normalized * np.array([width, height])).astype(np.int32)

            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [scaled], 0, 255, -1)

            assert mask.shape == (height, width)

            # Check that relative positions are preserved
            center_y, center_x = height // 2, width // 2
            assert mask[center_y, center_x] == 255  # Center inside

    def test_rasterize_float_contour_reference(self) -> None:
        """
        Float contours are converted to int for rasterization.
        """
        float_contour = np.array(
            [[10.5, 10.5], [10.5, 89.5], [89.5, 89.5], [89.5, 10.5]], dtype=np.float32
        )

        # OpenCV requires int32
        int_contour = float_contour.astype(np.int32)

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask, [int_contour], 0, 255, -1)

        assert mask[50, 50] == 255
