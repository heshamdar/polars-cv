"""
Reference tests for contour operations: polars-cv against OpenCV and NumPy.

Each test computes a quantity with OpenCV / NumPy and with polars-cv's
``.contour`` namespace on the same vertices, then compares them.
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
"""Analytic polygon measures are computed in f64 on both sides."""


def _contour(points, holes=None) -> dict:
    """A closed CONTOUR_SCHEMA value from (N, 2) vertex arrays (exact f64)."""
    return contour_from_points(
        [(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)],
        [
            [(float(x), float(y)) for x, y in np.asarray(h, dtype=np.float64)]
            for h in holes
        ]
        if holes
        else None,
    )


def _eval(points, expr: pl.Expr, holes=None):
    """Evaluate a ``.contour`` expression on column ``c`` holding one contour."""
    df = pl.DataFrame({"c": [_contour(points, holes)]}, schema={"c": CONTOUR_SCHEMA})
    return df.select(expr).item()


def _xy(contour: dict) -> np.ndarray:
    """The exterior ring of a contour value as an (N, 2) array."""
    return np.array([[p["x"], p["y"]] for p in contour["exterior"]])


def _cv(points, dtype=np.float32) -> np.ndarray:
    """OpenCV's (N, 1, 2) contour layout."""
    return np.asarray(points).reshape(-1, 1, 2).astype(dtype)


def _shoelace(contour: np.ndarray) -> float:
    x, y = contour[:, 0].astype(np.float64), contour[:, 1].astype(np.float64)
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _contains(points, point: tuple[float, float]) -> bool:
    df = pl.DataFrame(
        {"c": [_contour(points)], "p": [{"x": point[0], "y": point[1]}]},
        schema={"c": CONTOUR_SCHEMA, "p": POINT_SCHEMA},
    )
    return df.select(pl.col("c").contour.contains_point(pl.col("p"))).item()


@plugin_required
@pytest.mark.parity(
    ops=[
        "contour.area",
        "contour.perimeter",
        "contour.winding",
        "contour.flip",
        "contour.normalize",
        "contour.to_absolute",
        "contour.centroid",
        "contour.bounding_box",
        "contour.convex_hull",
        "contour.is_convex",
        "contour.translate",
        "contour.scale",
        "contour.simplify",
        "contour.contains_point",
    ],
    against="OpenCV `cv2.contourArea`/`arcLength`/`moments`/`boundingRect`/`convexHull`/`approxPolyDP`/`pointPolygonTest` and NumPy",
    tolerance=f"rel {REL} on analytic polygon measures (area, perimeter, centroid, transforms); "
    "exact for bbox, hull vertices, winding, convexity and containment",
)
class TestContourOpsReference:
    """polars-cv's ``.contour`` operations against OpenCV / NumPy on the same vertices."""

    def test_area_simple_square_reference(self, simple_contour: np.ndarray) -> None:
        """Square from (10,10) to (90,90): 80 * 80 = 6400."""
        ref = cv2.contourArea(_cv(simple_contour, np.int32))
        got = _eval(simple_contour, pl.col("c").contour.area())

        assert ref == 6400
        assert got == pytest.approx(ref, rel=REL)

    def test_area_triangle_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """Triangle (50,10), (10,90), (90,90): 0.5 * 80 * 80 = 3200."""
        triangle = standard_contours["triangle"]
        ref = cv2.contourArea(_cv(triangle, np.int32))
        got = _eval(triangle, pl.col("c").contour.area())

        assert ref == 3200
        assert got == pytest.approx(ref, rel=REL)

    def test_area_with_hole_reference(
        self, contour_with_hole: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Area of a region with a hole is the exterior minus the hole polygon."""
        outer, hole = contour_with_hole
        outer_area = cv2.contourArea(_cv(outer))
        hole_area = cv2.contourArea(_cv(hole))
        got = _eval(outer, pl.col("c").contour.area(), holes=[hole])

        assert outer_area == 10000
        assert got == pytest.approx(outer_area - hole_area, rel=REL)
        # The hole ring is wound clockwise; the area must not care.
        assert got < outer_area

    def test_perimeter_reference(self, simple_contour: np.ndarray) -> None:
        """Closed square of side 80 has perimeter 320."""
        ref = cv2.arcLength(_cv(simple_contour), closed=True)
        got = _eval(simple_contour, pl.col("c").contour.perimeter())

        assert ref == 320
        assert got == pytest.approx(ref, rel=REL)

    def test_perimeter_circle_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """A 64-gon approximating r=40: both sum the same closed edge lengths."""
        circle = standard_contours["circle"]
        # cv2 reads float32 vertices; feed polars the same rounded values.
        verts = circle.astype(np.float32).astype(np.float64)
        ref = cv2.arcLength(_cv(verts), closed=True)
        got = _eval(verts, pl.col("c").contour.perimeter())

        assert got == pytest.approx(ref, rel=1e-6)  # cv2 sums edges in f32
        assert got == pytest.approx(2 * np.pi * 40, rel=0.01)

    def test_winding_direction_from_area_reference(
        self, simple_contour: np.ndarray
    ) -> None:
        """
        Winding follows the sign of the shoelace area.

        Positive = ccw, negative = cw. The fixture is clockwise in image
        coordinates, negative under the shoelace formula.
        """
        signed = _shoelace(simple_contour)
        got_winding = _eval(simple_contour, pl.col("c").contour.winding())
        got_signed = _eval(simple_contour, pl.col("c").contour.area(signed=True))

        assert got_winding == ("ccw" if signed > 0 else "cw")
        assert got_winding == "cw"
        assert got_signed == pytest.approx(signed, rel=REL)

    def test_flip_reverses_winding_reference(self, simple_contour: np.ndarray) -> None:
        """Flipping reverses the point order, negating the signed area."""
        flipped_ref = simple_contour[::-1]
        flipped = _eval(simple_contour, pl.col("c").contour.flip())
        flipped_xy = _xy(flipped)

        np.testing.assert_array_equal(flipped_xy, flipped_ref)
        assert _shoelace(simple_contour) < 0
        assert _shoelace(flipped_xy) > 0
        assert _eval(flipped_xy, pl.col("c").contour.winding()) == "ccw"
        assert _eval(flipped_xy, pl.col("c").contour.area()) == pytest.approx(
            abs(_shoelace(simple_contour)), rel=REL
        )

    def test_normalize_coordinates_reference(self, simple_contour: np.ndarray) -> None:
        """Normalize pixel coords to [0,1] by dividing by the image size."""
        ref = simple_contour / np.array([100.0, 100.0])
        got = _xy(_eval(simple_contour, pl.col("c").contour.normalize(100, 100)))

        np.testing.assert_allclose(got, ref, rtol=REL)
        assert got.min() >= 0.0
        assert got.max() <= 1.0

    def test_to_absolute_coordinates_reference(self) -> None:
        """Convert normalized coords back to pixel coords."""
        normalized = np.array(
            [[0.1, 0.1], [0.1, 0.9], [0.9, 0.9], [0.9, 0.1]], dtype=np.float64
        )
        ref = normalized * np.array([100.0, 100.0])
        got = _xy(_eval(normalized, pl.col("c").contour.to_absolute(100, 100)))

        np.testing.assert_allclose(got, ref, rtol=REL)
        np.testing.assert_allclose(got, [[10, 10], [10, 90], [90, 90], [90, 10]])

    def test_centroid_reference(self, simple_contour: np.ndarray) -> None:
        """Centroid of the region, from polygon moments."""
        m = cv2.moments(_cv(simple_contour))
        ref = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        got = _eval(simple_contour, pl.col("c").contour.centroid())

        assert ref == (50.0, 50.0)
        assert (got["x"], got["y"]) == pytest.approx(ref, rel=REL)

    def test_centroid_triangle_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """An asymmetric shape, where the area centroid is not the plain vertex mean."""
        triangle = standard_contours["triangle"]
        m = cv2.moments(_cv(triangle))
        ref = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        got = _eval(triangle, pl.col("c").contour.centroid())

        assert (got["x"], got["y"]) == pytest.approx(ref, rel=REL)

    def test_bounding_box_reference(self, simple_contour: np.ndarray) -> None:
        """
        Axis-aligned bounding box.

        OpenCV's boundingRect is pixel-inclusive: width/height count both
        endpoints (10..90 is 81). polars-cv's box is the geometric extent
        (width 80), so the two differ by exactly one pixel.
        """
        x, y, w, h = cv2.boundingRect(_cv(simple_contour, np.int32))
        got = _eval(simple_contour, pl.col("c").contour.bounding_box())

        assert (w, h) == (81, 81)
        assert (got["x"], got["y"]) == (x, y)
        assert (got["width"], got["height"]) == (w - 1, h - 1)

    def test_convex_hull_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """The hull of a convex shape is its own vertex set."""
        square = standard_contours["square"]
        ref = cv2.convexHull(_cv(square)).reshape(-1, 2)
        got = _xy(_eval(square, pl.col("c").contour.convex_hull()))

        assert len(ref) == len(square)
        assert len(got) == len(ref)
        assert {tuple(p) for p in got} == {tuple(p) for p in ref.astype(np.float64)}

    def test_convex_hull_concave_reference(self) -> None:
        """A concave L-shape hulls to the same vertices OpenCV finds."""
        l_shape = np.array(
            [[0, 0], [0, 100], [50, 100], [50, 50], [100, 50], [100, 0]],
            dtype=np.float64,
        )
        ref = cv2.convexHull(_cv(l_shape)).reshape(-1, 2).astype(np.float64)
        got = _xy(_eval(l_shape, pl.col("c").contour.convex_hull()))

        assert len(ref) == 5  # (50, 50) is dropped
        assert {tuple(p) for p in got} == {tuple(p) for p in ref}

    def test_is_convex_reference(
        self, standard_contours: dict[str, np.ndarray]
    ) -> None:
        """Convex square."""
        square = standard_contours["square"]
        ref = cv2.isContourConvex(_cv(square))
        got = _eval(square, pl.col("c").contour.is_convex())

        assert ref is True
        assert got is ref

    def test_is_not_convex_reference(self) -> None:
        """Concave L-shape."""
        l_shape = np.array(
            [[0, 0], [0, 100], [50, 100], [50, 50], [100, 50], [100, 0]],
            dtype=np.float64,
        )
        ref = cv2.isContourConvex(_cv(l_shape))
        got = _eval(l_shape, pl.col("c").contour.is_convex())

        assert ref is False
        assert got is ref

    def test_translate_reference(self, simple_contour: np.ndarray) -> None:
        """Translate by an offset."""
        ref = simple_contour + np.array([10, 20])
        got = _xy(_eval(simple_contour, pl.col("c").contour.translate(dx=10, dy=20)))

        np.testing.assert_allclose(got, ref, rtol=REL)
        np.testing.assert_allclose(got, [[20, 30], [20, 110], [100, 110], [100, 30]])

    def test_scale_reference(self, simple_contour: np.ndarray) -> None:
        """Scale relative to the origin."""
        ref = simple_contour * np.array([0.5, 0.5])
        got = _xy(
            _eval(
                simple_contour,
                pl.col("c").contour.scale(sx=0.5, sy=0.5, origin="origin"),
            )
        )

        np.testing.assert_allclose(got, ref, rtol=REL)
        np.testing.assert_allclose(got, [[5, 5], [5, 45], [45, 45], [45, 5]])

    def test_scale_around_centroid_reference(self, simple_contour: np.ndarray) -> None:
        """Scaling about the centroid (the default origin) keeps the centroid fixed."""
        centroid = simple_contour.astype(np.float64).mean(axis=0)
        ref = (simple_contour - centroid) * np.array([0.5, 0.5]) + centroid
        scaled = _eval(simple_contour, pl.col("c").contour.scale(sx=0.5, sy=0.5))
        got = _xy(scaled)

        np.testing.assert_allclose(got, ref, rtol=REL)
        np.testing.assert_allclose(got.mean(axis=0), centroid, rtol=REL)

    def test_simplify_reference(self) -> None:
        """Douglas-Peucker simplification reduces points."""
        angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        circle = np.column_stack(
            [50 + 40 * np.cos(angles), 50 + 40 * np.sin(angles)]
        ).astype(np.float32)

        ref_low = cv2.approxPolyDP(_cv(circle), epsilon=1.0, closed=True)
        ref_high = cv2.approxPolyDP(_cv(circle), epsilon=10.0, closed=True)
        verts = circle.astype(np.float64)
        got_low = _xy(_eval(verts, pl.col("c").contour.simplify(tolerance=1.0)))
        got_high = _xy(_eval(verts, pl.col("c").contour.simplify(tolerance=10.0)))

        assert len(got_high) < len(got_low) < len(circle)
        assert len(ref_high) < len(ref_low) < len(circle)
        assert (len(got_low), len(got_high)) == (len(ref_low), len(ref_high))

    def test_point_in_polygon_reference(self, simple_contour: np.ndarray) -> None:
        """
        Point-in-polygon test.

        OpenCV reports > 0 inside, < 0 outside, 0 on the contour. polars-cv's
        boolean is True strictly inside; a point on the edge is not contained.
        """
        contour_cv = _cv(simple_contour)
        for point in [(50.0, 50.0), (0.0, 0.0), (10.0, 50.0)]:
            ref = cv2.pointPolygonTest(contour_cv, point, measureDist=False)
            assert _contains(simple_contour, point) == (ref > 0), point

        assert _contains(simple_contour, (50.0, 50.0)) is True
        assert _contains(simple_contour, (0.0, 0.0)) is False
        assert _contains(simple_contour, (10.0, 50.0)) is False  # on edge
