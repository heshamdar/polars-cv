"""
Reference tests for pairwise contour measures: polars-cv against OpenCV, NumPy, SciPy.

IoU / Dice are compared with pixel counts on OpenCV-rasterized masks (and the
hand-computed analytic value), Hausdorff distance with SciPy, and point
containment with ``cv2.pointPolygonTest``.
"""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl
import pytest

from polars_cv import CONTOUR_SCHEMA, POINT_SCHEMA
from polars_cv.geometry.schemas import contour_from_points
from tests.conftest import plugin_required

REL = 1e-9
"""Analytic polygon measures are computed in f64."""

MASK_TOL = 0.01
"""abs tolerance for polars-cv (exact polygon) vs OpenCV mask counts.

``cv2.drawContours(..., -1)`` fills edge-inclusive, so a 50-wide square covers
51x51 pixels. Measured on the overlap fixture: IoU 0.0870 vs 0.0926 (diff 0.0056),
Dice 0.1600 vs 0.1695 (diff 0.0095).
"""


def _contour(points, holes=None) -> dict:
    return contour_from_points(
        [(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)],
        [
            [(float(x), float(y)) for x, y in np.asarray(h, dtype=np.float64)]
            for h in holes
        ]
        if holes
        else None,
    )


def _pair(a, b, expr: pl.Expr, holes_a=None, holes_b=None):
    """Evaluate a two-operand ``.contour`` expression over columns ``a``, ``b``."""
    df = pl.DataFrame(
        {"a": [_contour(a, holes_a)], "b": [_contour(b, holes_b)]},
        schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
    )
    return df.select(expr).item()


def _iou(a, b, **kw) -> float:
    return _pair(a, b, pl.col("a").contour.iou(pl.col("b")), **kw)


def _dice(a, b, **kw) -> float:
    return _pair(a, b, pl.col("a").contour.dice(pl.col("b")), **kw)


def _hausdorff(a, b) -> float:
    return _pair(a, b, pl.col("a").contour.hausdorff_distance(pl.col("b")))


def _contains(points, point: tuple[float, float]) -> bool:
    df = pl.DataFrame(
        {"c": [_contour(points)], "p": [{"x": point[0], "y": point[1]}]},
        schema={"c": CONTOUR_SCHEMA, "p": POINT_SCHEMA},
    )
    return df.select(pl.col("c").contour.contains_point(pl.col("p"))).item()


def _scipy_hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric vertex-to-vertex Hausdorff distance (the documented semantics)."""
    directed_hausdorff = pytest.importorskip(
        "scipy.spatial.distance"
    ).directed_hausdorff
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return max(directed_hausdorff(a64, b64)[0], directed_hausdorff(b64, a64)[0])


@plugin_required
@pytest.mark.parity(
    ops=[
        "contour.iou",
        "contour.dice",
        "contour.hausdorff_distance",
        "contour.contains_point",
    ],
    against="OpenCV `drawContours` masks / `cv2.pointPolygonTest`, SciPy `directed_hausdorff`",
    tolerance=f"IoU/Dice abs {MASK_TOL} vs edge-inclusive OpenCV masks and rel {REL} vs the "
    "analytic value; Hausdorff rel 1e-9 vs SciPy; containment exact",
)
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
        """IoU of identical contours = 1.0."""
        mask = self._contour_to_mask(simple_contour)
        ref = np.sum(mask & mask) / np.sum(mask | mask)

        assert ref == 1.0
        assert _iou(simple_contour, simple_contour) == pytest.approx(ref, rel=REL)

    def test_iou_no_overlap_reference(
        self, non_overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """IoU of non-overlapping contours = 0.0."""
        c1, c2 = non_overlapping_contours
        mask1, mask2 = self._contour_to_mask(c1), self._contour_to_mask(c2)
        ref = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

        assert ref == 0.0
        assert _iou(c1, c2) == 0.0

    def test_iou_partial_overlap_reference(
        self, overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        IoU of partially overlapping contours.

        c1: (10,10) to (60,60) = 50x50 = 2500
        c2: (40,40) to (90,90) = 50x50 = 2500
        Intersection: 20x20 = 400, union 4600, IoU = 400 / 4600.
        """
        c1, c2 = overlapping_contours
        mask1, mask2 = self._contour_to_mask(c1), self._contour_to_mask(c2)
        mask_iou = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

        expected = 400 / 4600
        got = _iou(c1, c2)

        assert got == pytest.approx(expected, rel=REL)
        assert abs(got - mask_iou) < MASK_TOL

    def test_iou_symmetric_reference(
        self, overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """IoU is symmetric: IoU(A, B) == IoU(B, A)."""
        c1, c2 = overlapping_contours
        mask1, mask2 = self._contour_to_mask(c1), self._contour_to_mask(c2)

        ref_12 = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)
        ref_21 = np.sum(mask2 & mask1) / np.sum(mask2 | mask1)

        assert ref_12 == ref_21
        assert _iou(c1, c2) == pytest.approx(_iou(c2, c1), rel=REL)

    def test_dice_identical_reference(self, simple_contour: np.ndarray) -> None:
        """Dice coefficient of identical contours = 1.0."""
        mask = self._contour_to_mask(simple_contour)
        ref = 2 * np.sum(mask & mask) / (np.sum(mask) + np.sum(mask))

        assert ref == 1.0
        assert _dice(simple_contour, simple_contour) == pytest.approx(ref, rel=REL)

    def test_dice_no_overlap_reference(
        self, non_overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Dice coefficient of non-overlapping contours = 0.0."""
        c1, c2 = non_overlapping_contours
        mask1, mask2 = self._contour_to_mask(c1), self._contour_to_mask(c2)
        ref = 2 * np.sum(mask1 & mask2) / (np.sum(mask1) + np.sum(mask2))

        assert ref == 0.0
        assert _dice(c1, c2) == 0.0

    def test_dice_partial_overlap_reference(
        self, overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Dice = 2 * intersection / (area1 + area2) = 2 * 400 / 5000 = 0.16."""
        c1, c2 = overlapping_contours
        mask1, mask2 = self._contour_to_mask(c1), self._contour_to_mask(c2)
        mask_dice = 2 * np.sum(mask1 & mask2) / (np.sum(mask1) + np.sum(mask2))

        expected = 2 * 400 / 5000
        got = _dice(c1, c2)

        assert got == pytest.approx(expected, rel=REL)
        assert abs(got - mask_dice) < MASK_TOL

    def test_dice_vs_iou_relationship_reference(
        self, overlapping_contours: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Dice and IoU are related by Dice = 2*IoU / (1+IoU), in both implementations."""
        c1, c2 = overlapping_contours
        mask1, mask2 = self._contour_to_mask(c1), self._contour_to_mask(c2)
        intersection = np.sum(mask1 & mask2)
        ref_iou = intersection / np.sum(mask1 | mask2)
        ref_dice = 2 * intersection / (np.sum(mask1) + np.sum(mask2))

        assert abs(ref_dice - 2 * ref_iou / (1 + ref_iou)) < 1e-12

        iou, dice = _iou(c1, c2), _dice(c1, c2)
        assert dice == pytest.approx(2 * iou / (1 + iou), rel=REL)

    def test_hausdorff_identical_reference(self, simple_contour: np.ndarray) -> None:
        """Hausdorff distance of identical contours = 0.0."""
        assert _scipy_hausdorff(simple_contour, simple_contour) == 0.0
        assert _hausdorff(simple_contour, simple_contour) == 0.0

    def test_hausdorff_translated_reference(self, simple_contour: np.ndarray) -> None:
        """Translating by 10 px along x: vertex-to-vertex Hausdorff matches SciPy."""
        moved = simple_contour + np.array([10, 0], dtype=np.float32)
        ref = _scipy_hausdorff(simple_contour, moved)
        got = _hausdorff(simple_contour, moved)

        assert got == pytest.approx(ref, rel=REL)
        assert 9 < got < 11

    def test_hausdorff_scaled_reference(self, simple_contour: np.ndarray) -> None:
        """Hausdorff distance for a 2x scaled contour matches SciPy."""
        scaled = simple_contour * 2
        ref = _scipy_hausdorff(simple_contour, scaled)
        got = _hausdorff(simple_contour, scaled)

        assert ref > 0
        assert got == pytest.approx(ref, rel=REL)

    def test_contains_point_reference(self, simple_contour: np.ndarray) -> None:
        """
        Point-in-polygon test for contours.

        OpenCV: > 0 inside, < 0 outside, 0 on the contour. polars-cv returns
        True only strictly inside; a point on the edge is not contained.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.float32)
        points = {"inside": (50.0, 50.0), "outside": (0.0, 0.0), "edge": (10.0, 50.0)}

        for name, point in points.items():
            ref = cv2.pointPolygonTest(contour_cv, point, measureDist=False)
            assert _contains(simple_contour, point) == (ref > 0), name

        assert _contains(simple_contour, points["inside"]) is True
        assert _contains(simple_contour, points["outside"]) is False
        assert _contains(simple_contour, points["edge"]) is False

    def test_contains_point_distance_reference(
        self, simple_contour: np.ndarray
    ) -> None:
        """
        Point-in-polygon with signed distance.

        The sign of OpenCV's signed distance is the containment polars-cv
        reports, over a grid that includes inside and outside points.
        """
        contour_cv = simple_contour.reshape(-1, 1, 2).astype(np.float32)
        grid = [
            (x, y) for x in (0.0, 25.5, 50.0, 89.5, 120.0) for y in (5.0, 50.0, 95.5)
        ]

        for point in grid:
            dist = cv2.pointPolygonTest(contour_cv, point, measureDist=True)
            assert _contains(simple_contour, point) == (dist > 0), point

        assert cv2.pointPolygonTest(contour_cv, (50.0, 50.0), measureDist=True) == 40.0
        assert cv2.pointPolygonTest(contour_cv, (0.0, 0.0), measureDist=True) < 0

    def test_iou_with_holes_reference(
        self, contour_with_hole: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        IoU of a region with a hole against the full square.

        The hole is wholly inside the square, so IoU = (square - hole) / square
        with the hole's polygon area (from OpenCV). The mask count agrees to
        within MASK_TOL.
        """
        outer, hole = contour_with_hole

        mask1 = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask1, [outer.reshape(-1, 1, 2).astype(np.int32)], 0, 1, -1)
        cv2.drawContours(mask1, [hole.reshape(-1, 1, 2).astype(np.int32)], 0, 0, -1)

        simple = np.array([[0, 0], [0, 100], [100, 100], [100, 0]], dtype=np.int32)
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(mask2, [simple.reshape(-1, 1, 2)], 0, 1, -1)
        mask_iou = np.sum(mask1 & mask2) / np.sum(mask1 | mask2)

        hole_area = cv2.contourArea(hole.reshape(-1, 1, 2).astype(np.float32))
        expected = (10000 - hole_area) / 10000
        got = _iou(outer, simple, holes_a=[hole])

        assert 0 < mask_iou < 1.0
        assert got == pytest.approx(expected, rel=REL)
        # The mask hole is the int-truncated polygon; the difference is a few pixels.
        assert abs(got - mask_iou) < MASK_TOL
