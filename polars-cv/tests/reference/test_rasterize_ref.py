"""
Reference tests for contour rasterization against OpenCV.

Each test draws a contour with ``cv2.drawContours`` and with polars-cv's
``rasterize()`` and compares the masks.

The conventions differ in one documented way. ``cv2.drawContours`` paints the
pixels a vertex or edge *touches*, so a rectangle between integer vertices
``a`` and ``b`` covers pixels ``a..b`` inclusive. ``rasterize()`` paints a
pixel when its *centre* ``(x + 0.5, y + 0.5)`` lies inside the contour. The
same pixels therefore come from the polygon whose far edges are one pixel
further out (``b + 1``), and for axis-aligned shapes the two masks are
identical. Diagonal and curved edges cut through pixels: there OpenCV
also paints the boundary pixels the centre rule leaves out, so the masks agree
inside and differ along the boundary only, by well under the perimeter.
"""

from __future__ import annotations

import cv2
import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_column
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA
from tests.conftest import plugin_required


def _ring(points: np.ndarray) -> list[dict]:
    return [
        {"x": float(x), "y": float(y)} for x, y in np.asarray(points).reshape(-1, 2)
    ]


def _contour(exterior: np.ndarray, holes: tuple[np.ndarray, ...] = ()) -> dict:
    return {
        "exterior": _ring(exterior),
        "holes": [_ring(h) for h in holes],
        "is_closed": True,
    }


def _edge_rect(points: np.ndarray) -> np.ndarray:
    """An axis-aligned rectangle's vertices in pixel-edge coordinates.

    OpenCV paints ``a..b`` inclusive, so the polygon covering exactly those
    pixel centres runs from ``a`` to ``b + 1``.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts + (pts == pts.max(axis=0))


def _rasterize(
    contours: list[dict], height: int, width: int, *, stroke: dict | None = None, **kw
) -> np.ndarray:
    """polars-cv's ``(height, width)`` mask of a contour set."""
    pipe = Pipeline().source("contour")
    if stroke is not None:
        pipe = pipe.stroke(**stroke)
    pipe = pipe.rasterize(width=width, height=height, **kw)
    df = pl.DataFrame({"c": [contours]}, schema={"c": CONTOUR_SET_SCHEMA})
    out = df.select(pl.col("c").cv.pipe(pipe).sink("numpy"))
    (mask,) = numpy_from_column(out.to_series(), copy=True)
    assert mask is not None and mask.shape == (height, width, 1)
    return mask[:, :, 0]


def _assert_within_boundary_band(
    ours: np.ndarray, theirs: np.ndarray, perimeter: float
) -> None:
    """Same mask but for pixels on the boundary of a diagonal or curved edge.

    Measured: the differing pixels number 0.47-0.49 x the perimeter
    (OpenCV paints the boundary pixel that the centre rule leaves out), so the
    bound is the perimeter itself.
    """
    assert ours.shape == theirs.shape
    differing = int((ours != theirs).sum())
    assert 0 < differing <= perimeter, (differing, perimeter)


@plugin_required
@pytest.mark.parity(
    ops=["rasterize", "stroke"],
    against="OpenCV `cv2.drawContours` (filled, thickness=1 outline, holes punched with value 0) on the same contours",
    tolerance=(
        "exact for axis-aligned shapes (far edges at b + 1 for OpenCV's inclusive pixels) "
        "and for the 1px outline; diagonal/curved edges differ on boundary pixels only, "
        "<= the perimeter (measured ~0.48x)"
    ),
)
class TestRasterizeReference:
    """``rasterize()`` paints the same masks as ``cv2.drawContours``."""

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

        ours = _rasterize([_contour(_edge_rect(simple_contour))], 100, 100)
        assert ours.dtype == np.uint8
        np.testing.assert_array_equal(ours, mask)

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

        ours = _rasterize([_contour(triangle_contour)], 100, 100)
        assert ours.dtype == np.uint8
        # Centre-rule pixels are a subset of OpenCV's boundary-inclusive ones.
        assert not ((ours > 0) & (mask == 0)).any()
        _assert_within_boundary_band(
            ours, mask, cv2.arcLength(triangle_contour.reshape(-1, 1, 2), True)
        )

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

        # The hole is a ring of the contour (the float vertices; OpenCV
        # truncated them). The outer square covers the whole canvas in both.
        ours = _rasterize([_contour(outer, (hole,))], 100, 100)
        assert ours[50, 50] == 0 and ours[5, 50] == 255
        # OpenCV's hole also takes its boundary pixels: its foreground is a
        # subset of ours.
        assert not ((mask > 0) & (ours == 0)).any()
        _assert_within_boundary_band(ours, mask, cv2.arcLength(hole_cv, True))

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

            ours = _rasterize([_contour(scaled_contour)], height, width)
            assert not ((ours > 0) & (mask == 0)).any()
            _assert_within_boundary_band(
                ours, mask, cv2.arcLength(scaled_contour.reshape(-1, 1, 2), True)
            )

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

        ours = _rasterize(
            [_contour(_edge_rect(simple_contour))],
            100,
            100,
            fill_value=fill_value,
            background=0,
        )
        np.testing.assert_array_equal(ours, mask)

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

        # A 1px outline is polars-cv's `stroke(width=1)`; OpenCV draws along
        # the pixel centres, so the ring runs through (x + 0.5, y + 0.5).
        ours = _rasterize(
            [_contour(simple_contour + 0.5)],
            100,
            100,
            stroke={"width": 1.0, "join": "miter"},
        )
        np.testing.assert_array_equal(ours, mask)

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

        ours = _rasterize([_contour(_edge_rect(c)) for c in (c1, c2)], 100, 100)
        np.testing.assert_array_equal(ours, mask)

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

        # polars-cv carries the hole as a ring of the contour.
        ours = _rasterize([_contour(_edge_rect(outer), (_edge_rect(inner),))], 100, 100)
        np.testing.assert_array_equal(ours, mask)

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

            ours = _rasterize([_contour(_edge_rect(scaled))], height, width)
            np.testing.assert_array_equal(ours, mask)
            assert ours[center_y, center_x] == 255

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

        # polars-cv takes the float vertices as they are, no truncation. With
        # fractions off .5 no pixel centre sits on an edge, and the centre
        # rule equals OpenCV's truncation to the pixel: exactly equal.
        off_centre = np.array(
            [[10.3, 10.3], [10.3, 89.7], [89.7, 89.7], [89.7, 10.3]], dtype=np.float32
        )
        reference = np.zeros((100, 100), dtype=np.uint8)
        cv2.drawContours(reference, [off_centre.astype(np.int32)], 0, 255, -1)
        np.testing.assert_array_equal(
            _rasterize([_contour(off_centre)], 100, 100), reference
        )

        # With the .5 vertices every edge passes through pixel centres, where
        # the centre rule is a tie. Measured: 80 of 6400 pixels differ (one
        # side of each tie), none inside the rows/columns 11..88.
        ours = _rasterize([_contour(float_contour)], 100, 100)
        assert ours[50, 50] == 255
        np.testing.assert_array_equal(ours[11:89, 11:89], mask[11:89, 11:89])
        assert int((ours != mask).sum()) <= 4 * 80
