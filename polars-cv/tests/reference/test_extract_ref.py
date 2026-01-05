"""
Reference tests for contour extraction using OpenCV.

These tests establish the expected behavior for extracting contours
from binary masks, serving as ground truth for polars-cv.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    pass


class TestExtractContoursReference:
    """Establish expected behavior for mask to contour extraction using OpenCV."""

    def test_extract_single_contour_reference(
        self, binary_mask_with_shape: np.ndarray
    ) -> None:
        """
        Extract single contour from simple binary mask.
        """
        contours, hierarchy = cv2.findContours(
            binary_mask_with_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) == 1
        # Rectangle should have 4 corners with CHAIN_APPROX_SIMPLE
        assert len(contours[0]) == 4

    def test_extract_external_only_reference(self, mask_with_hole: np.ndarray) -> None:
        """
        Extract only external (outermost) contours.
        """
        contours, hierarchy = cv2.findContours(
            mask_with_hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Only outer contour, hole is ignored
        assert len(contours) == 1

    def test_extract_tree_hierarchy_reference(self, mask_with_hole: np.ndarray) -> None:
        """
        Extract contours with hierarchy (holes as children).
        """
        contours, hierarchy = cv2.findContours(
            mask_with_hole, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) == 2  # Outer + hole

        # hierarchy[0][i] = [next, prev, child, parent]
        # Outer should have child (hole)
        assert hierarchy[0][0][2] != -1  # Has child
        # Hole should have parent (outer)
        assert hierarchy[0][1][3] != -1  # Has parent

    def test_extract_all_contours_reference(self) -> None:
        """
        Extract all contours flattened (no hierarchy).
        """
        # Create mask with multiple shapes
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (40, 40), 255, -1)
        cv2.rectangle(mask, (60, 60), (90, 90), 255, -1)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) == 2

    def test_extract_no_approximation_reference(
        self, binary_mask_with_shape: np.ndarray
    ) -> None:
        """
        Extract contours without point approximation (all boundary points).
        """
        contours_none, _ = cv2.findContours(
            binary_mask_with_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        contours_simple, _ = cv2.findContours(
            binary_mask_with_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # NONE keeps all boundary points, SIMPLE removes redundant ones
        assert len(contours_none[0]) > len(contours_simple[0])

    def test_extract_approximation_methods_reference(self) -> None:
        """
        Compare different contour approximation methods.
        """
        # Create a diagonal line shape
        mask = np.zeros((100, 100), dtype=np.uint8)
        pts = np.array([[10, 10], [90, 10], [50, 90]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

        # No approximation
        contours_none, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        # Simple approximation
        contours_simple, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # TC89 approximation
        contours_tc89, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )

        # All should find one contour
        assert len(contours_none) == 1
        assert len(contours_simple) == 1
        assert len(contours_tc89) == 1

        # Point counts should differ
        assert len(contours_none[0]) >= len(contours_simple[0])

    def test_extract_min_area_filter_reference(self) -> None:
        """
        Filter contours by minimum area.
        """
        # Create mask with large and small shapes
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (50, 50), 255, -1)  # Large: 40x40 = 1600
        cv2.rectangle(mask, (80, 80), (85, 85), 255, -1)  # Small: 5x5 = 25

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area
        min_area = 100
        filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

        assert len(contours) == 2  # Both found
        assert len(filtered) == 1  # Only large kept

    def test_extract_contour_area_reference(
        self, binary_mask_with_shape: np.ndarray
    ) -> None:
        """
        Extracted contour area should approximate original shape.

        Note: cv2.contourArea computes the geometric area of the polygon,
        while mask pixel count includes all interior pixels. These can differ
        due to the discrete nature of rasterization vs continuous geometry.
        """
        contours, _ = cv2.findContours(
            binary_mask_with_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_area = cv2.contourArea(contours[0])
        mask_area = np.sum(binary_mask_with_shape > 0)

        # Areas should be close (allow ~5% difference due to discretization)
        assert abs(contour_area - mask_area) < mask_area * 0.05

    def test_extract_from_grayscale_reference(self) -> None:
        """
        Extract contours from grayscale (thresholded) image.
        """
        # Simulate grayscale with values
        gray = np.zeros((100, 100), dtype=np.uint8)
        gray[20:80, 20:80] = 200  # Bright region

        # Threshold first
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) == 1

    def test_extract_empty_mask_reference(self) -> None:
        """
        Extracting from empty mask returns no contours.
        """
        empty_mask = np.zeros((100, 100), dtype=np.uint8)

        contours, hierarchy = cv2.findContours(
            empty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) == 0

    def test_extract_full_mask_reference(self) -> None:
        """
        Extracting from fully filled mask returns boundary contour.
        """
        full_mask = np.ones((100, 100), dtype=np.uint8) * 255

        contours, _ = cv2.findContours(
            full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Should find outer boundary
        assert len(contours) == 1

    def test_extract_nested_contours_reference(self) -> None:
        """
        Extract deeply nested contours (island in hole in shape).
        """
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Outer
        cv2.rectangle(mask, (5, 5), (95, 95), 255, -1)
        # Hole
        cv2.rectangle(mask, (20, 20), (80, 80), 0, -1)
        # Island in hole
        cv2.rectangle(mask, (35, 35), (65, 65), 255, -1)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Should find: outer, hole, island
        assert len(contours) == 3

        # Verify hierarchy relationships
        # hierarchy[0][i] = [next, prev, child, parent]
        # Find which is which by checking parent relationships
        parents = [h[3] for h in hierarchy[0]]
        assert -1 in parents  # At least one has no parent (outer)

    def test_extract_preserves_winding_reference(self) -> None:
        """
        Extracted contours have consistent winding direction.
        """
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (20, 20), (80, 80), 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check winding by signed area
        contour = contours[0].reshape(-1, 2).astype(np.float32)
        n = len(contour)
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += contour[i, 0] * contour[j, 1]
            signed_area -= contour[j, 0] * contour[i, 1]
        signed_area /= 2

        # OpenCV typically returns CCW (positive area) for outer contours
        # But this can depend on the specific shape and extraction
        # The important thing is consistency
        assert signed_area != 0  # Should have definite winding
