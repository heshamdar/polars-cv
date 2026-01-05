"""
Reference tests for pairwise contour measures using OpenCV and NumPy.

These tests establish the expected behavior for IoU, Dice, and other
pairwise metrics between contours, serving as ground truth for polars-cv.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

if TYPE_CHECKING:
    pass


class TestPairwiseMeasuresReference:
    """Establish expected behavior for pairwise contour measures."""

    def _contour_to_mask(
        self, contour: np.ndarray, shape: tuple[int, int] = (100, 100)
    ) -> np.ndarray:
        """
        Helper to rasterize contour to binary mask.

        Args:
            contour: Contour points with shape (N, 2).
            shape: Output mask shape (height, width).

        Returns:
            Binary mask with 1 inside contour, 0 outside.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        contour_cv = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(mask, [contour_cv], 0, 1, -1)
        return mask

    def test_iou_identical_reference(self, simple_contour: np.ndarray) -> None:
        """
        IoU of identical contours = 1.0.
        """
        mask = self._contour_to_mask(simple_contour)

        intersection = np.sum(mask & mask)
        union = np.sum(mask | mask)
        iou = intersection / union

        assert iou == 1.0

    def test_iou_no_overlap_reference(
        self, non_overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        IoU of non-overlapping contours = 0.0.
        """
        c1, c2 = non_overlapping_contours

        mask1 = self._contour_to_mask(c1)
        mask2 = self._contour_to_mask(c2)

        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        iou = intersection / union if union > 0 else 0.0

        assert iou == 0.0

    def test_iou_partial_overlap_reference(
        self, overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        IoU of partially overlapping contours.

        c1: (10,10) to (60,60) = 50x50 = 2500
        c2: (40,40) to (90,90) = 50x50 = 2500
        Intersection: (40,40) to (60,60) = 20x20 = 400
        Union: 2500 + 2500 - 400 = 4600
        IoU: 400 / 4600 ≈ 0.087
        """
        c1, c2 = overlapping_contours

        mask1 = self._contour_to_mask(c1)
        mask2 = self._contour_to_mask(c2)

        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        iou = intersection / union

        expected_intersection = 20 * 20
        expected_union = 50 * 50 + 50 * 50 - expected_intersection
        expected_iou = expected_intersection / expected_union

        # Allow some tolerance for rasterization artifacts
        assert abs(iou - expected_iou) < 0.05

    def test_iou_symmetric_reference(
        self, overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        IoU is symmetric: IoU(A, B) == IoU(B, A).
        """
        c1, c2 = overlapping_contours

        mask1 = self._contour_to_mask(c1)
        mask2 = self._contour_to_mask(c2)

        iou_12 = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
        iou_21 = np.sum(mask2 & mask1) / np.sum(mask2 | mask1)

        assert iou_12 == iou_21

    def test_dice_identical_reference(self, simple_contour: np.ndarray) -> None:
        """
        Dice coefficient of identical contours = 1.0.
        """
        mask = self._contour_to_mask(simple_contour)

        intersection = np.sum(mask & mask)
        dice = 2 * intersection / (np.sum(mask) + np.sum(mask))

        assert dice == 1.0

    def test_dice_no_overlap_reference(
        self, non_overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Dice coefficient of non-overlapping contours = 0.0.
        """
        c1, c2 = non_overlapping_contours

        mask1 = self._contour_to_mask(c1)
        mask2 = self._contour_to_mask(c2)

        intersection = np.sum(mask1 & mask2)
        dice = 2 * intersection / (np.sum(mask1) + np.sum(mask2))

        assert dice == 0.0

    def test_dice_partial_overlap_reference(
        self, overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Dice coefficient of partially overlapping contours.

        Dice = 2 * intersection / (area1 + area2)
        Dice = 2 * 400 / (2500 + 2500) = 800 / 5000 = 0.16
        """
        c1, c2 = overlapping_contours

        mask1 = self._contour_to_mask(c1)
        mask2 = self._contour_to_mask(c2)

        intersection = np.sum(mask1 & mask2)
        dice = 2 * intersection / (np.sum(mask1) + np.sum(mask2))

        expected_dice = 2 * 400 / 5000

        assert abs(dice - expected_dice) < 0.05

    def test_dice_vs_iou_relationship_reference(
        self, overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Dice and IoU have a known relationship: Dice = 2*IoU / (1+IoU).
        """
        c1, c2 = overlapping_contours

        mask1 = self._contour_to_mask(c1)
        mask2 = self._contour_to_mask(c2)

        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        area_sum = np.sum(mask1) + np.sum(mask2)

        iou = intersection / union
        dice = 2 * intersection / area_sum

        # Relationship: Dice = 2*IoU / (1+IoU)
        dice_from_iou = 2 * iou / (1 + iou)

        assert abs(dice - dice_from_iou) < 0.01

    def test_hausdorff_identical_reference(self, simple_contour: np.ndarray) -> None:
        """
        Hausdorff distance of identical contours = 0.0.
        """
        contour1 = simple_contour.astype(np.float32)
        contour2 = simple_contour.astype(np.float32)

        try:
            from scipy.spatial.distance import directed_hausdorff

            h_dist = max(
                directed_hausdorff(contour1, contour2)[0],
                directed_hausdorff(contour2, contour1)[0],
            )
            assert h_dist == 0.0
        except ImportError:
            # Fallback: manual computation without scipy
            # For identical contours, distance is 0
            # Compute using numpy broadcasting
            diff = contour1[:, np.newaxis, :] - contour2[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            h_dist = max(dists.min(axis=1).max(), dists.min(axis=0).max())
            assert h_dist == 0.0

    def test_hausdorff_translated_reference(self, simple_contour: np.ndarray) -> None:
        """
        Hausdorff distance reflects translation amount.
        """
        contour1 = simple_contour.astype(np.float32)
        contour2 = (simple_contour + np.array([10, 0])).astype(np.float32)

        try:
            from scipy.spatial.distance import directed_hausdorff

            h_dist = max(
                directed_hausdorff(contour1, contour2)[0],
                directed_hausdorff(contour2, contour1)[0],
            )
        except ImportError:
            pytest.skip("scipy not available for Hausdorff distance")

        # Translation of 10 pixels should give Hausdorff ≈ 10
        assert 9 < h_dist < 11

    def test_hausdorff_scaled_reference(self, simple_contour: np.ndarray) -> None:
        """
        Hausdorff distance for scaled contours.
        """
        contour1 = simple_contour.astype(np.float32)
        # Scale by 2x around origin
        contour2 = (simple_contour * 2).astype(np.float32)

        try:
            from scipy.spatial.distance import directed_hausdorff

            h_dist = max(
                directed_hausdorff(contour1, contour2)[0],
                directed_hausdorff(contour2, contour1)[0],
            )
        except ImportError:
            pytest.skip("scipy not available for Hausdorff distance")

        # Distance should be significant
        assert h_dist > 0

    def test_contains_point_reference(self, simple_contour: np.ndarray) -> None:
        """
        Point-in-polygon test for contours.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.float32)

        # Point inside
        inside_result = cv2.pointPolygonTest(
            contour_cv, (50.0, 50.0), measureDist=False
        )
        assert inside_result > 0  # Positive = inside

        # Point outside
        outside_result = cv2.pointPolygonTest(contour_cv, (0.0, 0.0), measureDist=False)
        assert outside_result < 0  # Negative = outside

        # Point on edge
        edge_result = cv2.pointPolygonTest(contour_cv, (10.0, 50.0), measureDist=False)
        assert edge_result == 0  # Zero = on contour

    def test_contains_point_distance_reference(
        self, simple_contour: np.ndarray
    ) -> None:
        """
        Point-in-polygon with signed distance.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.float32)

        # Distance to center (should be positive, inside)
        dist_center = cv2.pointPolygonTest(contour_cv, (50.0, 50.0), measureDist=True)
        assert dist_center > 0

        # Distance to outside point (should be negative)
        dist_outside = cv2.pointPolygonTest(contour_cv, (0.0, 0.0), measureDist=True)
        assert dist_outside < 0

    def test_iou_with_holes_reference(
        self, contour_with_hole: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        IoU computation with contours that have holes.
        """
        outer, hole = contour_with_hole

        # Create mask with hole
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask1, [outer.reshape(-1, 1, 2).astype(np.int32)], 0, 1, -1)
        cv2.drawContours(mask1, [hole.reshape(-1, 1, 2).astype(np.int32)], 0, 0, -1)

        # Compare with simple square (no hole)
        simple = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=np.int32)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask2, [simple.reshape(-1, 1, 2)], 0, 1, -1)

        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        iou = intersection / union

        # mask1 has hole, mask2 is full -> IoU < 1.0
        assert 0 < iou < 1.0
