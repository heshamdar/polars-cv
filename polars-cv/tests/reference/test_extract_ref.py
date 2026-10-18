"""
Reference tests for contour extraction against OpenCV.

Each test builds a binary mask, runs ``cv2.findContours`` on it and
``extract_contours()`` on the same mask, and compares the two.

The conventions differ in one documented way. OpenCV returns the *centres* of
the boundary pixels; polars-cv traces the *edges* of the boundary pixels
(pixel ``(x, y)`` is the square ``[x, x+1] x [y, y+1]``). So a region whose
OpenCV bounding rect is ``(x, y, w, h)`` has polars-cv bounds
``(x, y, x+w, y+h)``, its polars-cv area is its pixel count (OpenCV's
``contourArea`` is ``(w-1)*(h-1)`` for a rectangle), and a hole border sits one
pixel inside OpenCV's. The comparisons below are exact under that mapping.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline
from tests.conftest import plugin_required


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _pipe(**kwargs: object) -> Pipeline:
    return (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .threshold(127)
        .extract_contours(**kwargs)
    )


def _extract(mask: np.ndarray, **kwargs: object) -> list[dict]:
    """polars-cv borders of ``mask`` (foreground is level > 127)."""
    df = pl.DataFrame({"img": [_png(mask)]})
    out = df.select(out=pl.col("img").cv.pipe(_pipe(**kwargs)).sink("native"))
    return list(out["out"][0])


def _bounds(border: dict) -> tuple[float, float, float, float]:
    xs = [p["x"] for p in border["exterior"]]
    ys = [p["y"] for p in border["exterior"]]
    return (min(xs), min(ys), max(xs), max(ys))


def _signed_area(border: dict) -> float:
    pts = border["exterior"]
    n = len(pts)
    return (
        sum(
            pts[i]["x"] * pts[(i + 1) % n]["y"] - pts[(i + 1) % n]["x"] * pts[i]["y"]
            for i in range(n)
        )
        / 2
    )


def _depth(hierarchy: np.ndarray, i: int) -> int:
    depth = 0
    while hierarchy[0][i][3] != -1:
        i = hierarchy[0][i][3]
        depth += 1
    return depth


def _expected_bounds(contours, hierarchy=None) -> list[tuple[float, ...]]:
    """OpenCV contours mapped to the polars-cv (pixel-edge) bounds, sorted.

    A foreground border (even depth) is the same rect with its far edges
    pushed out by one pixel; a hole border (odd depth) is OpenCV's trace of
    the foreground ring around the hole, one pixel outside the hole itself.
    """
    out = []
    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if hierarchy is not None and _depth(hierarchy, i) % 2 == 1:
            out.append((x + 1.0, y + 1.0, x + w - 1.0, y + h - 1.0))
        else:
            out.append((float(x), float(y), x + w * 1.0, y + h * 1.0))
    return sorted(out)


@plugin_required
@pytest.mark.parity(
    ops=["extract_contours"],
    against="OpenCV `cv2.findContours` (RETR_EXTERNAL/LIST/TREE, CHAIN_APPROX_NONE/SIMPLE/TC89) on the same mask",
    tolerance=(
        "exact: contour count, bounding box under the pixel-edge vs pixel-centre "
        "mapping, area == mask pixel count, vertex count where the conventions coincide"
    ),
)
class TestExtractContoursReference:
    """``extract_contours()`` agrees with ``cv2.findContours`` on the same masks."""

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

        ours = _extract(binary_mask_with_shape, mode="external", method="simple")
        assert len(ours) == 1
        assert len(ours[0]["exterior"]) == 4
        assert [_bounds(b) for b in ours] == _expected_bounds(contours)

    def test_extract_external_only_reference(self, mask_with_hole: np.ndarray) -> None:
        """
        Extract only external (outermost) contours.
        """
        contours, hierarchy = cv2.findContours(
            mask_with_hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Only outer contour, hole is ignored
        assert len(contours) == 1

        ours = _extract(mask_with_hole, mode="external")
        assert len(ours) == 1
        assert ours[0]["holes"] == []
        assert [_bounds(b) for b in ours] == _expected_bounds(contours)

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

        # polars-cv has no hierarchy: mode="all" is a flat list, the exterior
        # plus one border per enclosed background region.
        ours = _extract(mask_with_hole, mode="all")
        assert len(ours) == 2
        assert sorted(_bounds(b) for b in ours) == _expected_bounds(contours, hierarchy)
        # The hole border is wound the other way from the exterior, so the two
        # signed areas net to the mask's pixel count.
        assert sum(_signed_area(b) for b in ours) == np.sum(mask_with_hole > 0)

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

        ours = _extract(mask, mode="all")
        assert len(ours) == 2
        assert sorted(_bounds(b) for b in ours) == _expected_bounds(contours)

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

        ours_none = _extract(binary_mask_with_shape, method="none")
        ours_simple = _extract(binary_mask_with_shape, method="simple")
        assert len(ours_none[0]["exterior"]) > len(ours_simple[0]["exterior"])
        # SIMPLE leaves the four corners in both libraries.
        assert len(ours_simple[0]["exterior"]) == len(contours_simple[0]) == 4
        # NONE: one point per boundary step. The 60 x 60 block's pixel-edge
        # outline has 4 * 60 = 240 steps, OpenCV's pixel-centre trace
        # 4 * 59 = 236 (the convention difference, exactly 4 corners).
        assert len(ours_none[0]["exterior"]) == 240
        assert len(contours_none[0]) == 236
        assert [_bounds(b) for b in ours_none] == _expected_bounds(contours_none)

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

        ours = {m: _extract(mask, method=m) for m in ("none", "simple", "approx")}
        for method, borders in ours.items():
            assert len(borders) == 1, method
            # Every method outlines the same pixels as OpenCV's, whatever the
            # vertex count.
            assert _bounds(borders[0]) == _expected_bounds(contours_none)[0], method
        n = {m: len(b[0]["exterior"]) for m, b in ours.items()}
        assert n["none"] >= n["simple"] > n["approx"], n
        # The lossless methods enclose exactly the mask's pixels; "approx"
        # simplifies the outline, so its area is only close.
        pixels = int(np.sum(mask > 0))
        assert abs(_signed_area(ours["none"][0])) == pixels
        assert abs(_signed_area(ours["simple"][0])) == pixels
        # Measured 3341 vs 3321 pixels (+0.6%): a deliberately lossy polygon.
        assert abs(_signed_area(ours["approx"][0]) - pixels) <= 0.01 * pixels

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

        # polars-cv areas are pixel counts (41 x 41 and 6 x 6, since OpenCV's
        # rectangle() is inclusive), not cv2.contourArea's 1600 and 25; both
        # sit on the same side of the threshold.
        unfiltered = _extract(mask)
        assert sorted(_signed_area(b) for b in unfiltered) == [36.0, 1681.0]
        ours = _extract(mask, min_area=min_area)
        assert len(ours) == len(filtered)
        assert [_bounds(b) for b in ours] == _expected_bounds(filtered)

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

        # polars-cv traces pixel edges, so its area is the pixel count itself.
        ours = _extract(binary_mask_with_shape)
        assert abs(_signed_area(ours[0])) == mask_area

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

        # polars-cv thresholds the grey image itself (same cut, 127).
        ours = _extract(gray)
        assert len(ours) == 1
        assert [_bounds(b) for b in ours] == _expected_bounds(contours)

    def test_extract_empty_mask_reference(self) -> None:
        """
        Extracting from empty mask returns no contours.
        """
        empty_mask = np.zeros((100, 100), dtype=np.uint8)

        contours, hierarchy = cv2.findContours(
            empty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        assert len(contours) == 0
        assert _extract(empty_mask) == []

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

        ours = _extract(full_mask)
        assert len(ours) == 1
        assert [_bounds(b) for b in ours] == _expected_bounds(contours)
        assert _bounds(ours[0]) == (0.0, 0.0, 100.0, 100.0)

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

        # polars-cv returns the same three borders flat; the nesting depth
        # fixes which are holes (odd depth), mapped as in _expected_bounds.
        ours = _extract(mask, mode="all")
        assert len(ours) == 3
        assert sorted(_bounds(b) for b in ours) == _expected_bounds(contours, hierarchy)

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

        # polars-cv: a definite winding too, and `winding()` agrees with the
        # sign of the signed area (positive = "ccw" in math coordinates).
        # Measured: OpenCV's outer trace has negative signed area, polars-cv's
        # positive, i.e. opposite orientations. Direction is not a documented
        # property of extract_contours (every contour op is winding-independent),
        # so only the region is compared: the same pixels either way.
        contour_expr = pl.col("img").cv.pipe(_pipe()).sink("native").list.first()
        df = pl.DataFrame({"img": [_png(mask)]})
        out = df.select(
            area=contour_expr.contour.area(signed=True),
            winding=contour_expr.contour.winding(),
        )
        ours_area = out["area"][0]
        assert ours_area != 0
        assert out["winding"][0] == ("ccw" if ours_area > 0 else "cw")
        assert abs(ours_area) == np.sum(mask > 0)
