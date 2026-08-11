"""
Contour operations namespace for Polars expressions.

This module provides the `.contour` accessor for operations on contour columns.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from polars_cv._namespace import _ArgBinder, _GeomNullPolicy, _PluginNamespace
from polars_cv._types import (
    LabelReduction,
    LabelRegionMode,
    ScaleOrigin,
    Winding,
    _enum_or_expr,
)


@pl.api.register_expr_namespace("contour")
class ContourNamespace(_GeomNullPolicy, _PluginNamespace):
    """
    Namespace for geometric operations on contour columns.

    Numeric parameters accept either a literal or a Polars expression; an
    expression is resolved per row at execution time.

    Example:
        >>> df.with_columns(
        ...     area=pl.col("contour").contour.area(),
        ...     bbox=pl.col("contour").contour.bounding_box(),
        ...     norm=pl.col("contour").contour.normalize(
        ...         pl.col("img_w"), pl.col("img_h")
        ...     ),
        ... )
    """

    # --- Coordinate Operations ---

    def normalize(
        self,
        width: int | float | pl.Expr,
        height: int | float | pl.Expr,
    ) -> pl.Expr:
        """
        Convert pixel coordinates to normalized [0,1] range.

        Args:
            width: Reference width for normalization (literal or expression).
            height: Reference height for normalization (literal or expression).

        Returns:
            Contour with coordinates in [0,1] range.
        """
        binder = _ArgBinder()
        binder.add_param("ref_width", width)
        binder.add_param("ref_height", height)
        return binder.call(self, "contour_normalize")

    def to_absolute(
        self,
        width: int | float | pl.Expr,
        height: int | float | pl.Expr,
    ) -> pl.Expr:
        """
        Convert normalized coordinates to pixel coordinates.

        Args:
            width: Reference width for scaling (literal or expression).
            height: Reference height for scaling (literal or expression).

        Returns:
            Contour with pixel coordinates.
        """
        binder = _ArgBinder()
        binder.add_param("ref_width", width)
        binder.add_param("ref_height", height)
        return binder.call(self, "contour_to_absolute")

    # --- Geometric Measures ---

    def winding(self) -> pl.Expr:
        """
        Compute the exterior ring's winding direction from point order.

        Returns:
            String 'ccw' for counter-clockwise, 'cw' for clockwise.

        Note:
            Winding is computed using the Shoelace formula:
            - Positive signed area = CCW
            - Negative signed area = CW

            This is purely a report on point order. Winding does not mark a ring as
            a hole — the `holes` field does — and no other operation consults it.
        """
        return self._plugin("contour_winding")

    def area(self, *, signed: bool | pl.Expr = False) -> pl.Expr:
        """
        Compute contour area.

        The area of the region the contour describes: the exterior minus the union
        of its hole rings, in either winding direction. Overlapping or nested hole
        rings are not double-subtracted.

        Args:
            signed: If True, return signed area (negative for CW winding).
                   If False, return absolute area. Accepts a boolean expression
                   for per-row selection.

        Returns:
            Float64 area value.
        """
        binder = _ArgBinder()
        binder.add_param("signed", signed, cast=bool)
        return binder.call(self, "contour_area")

    def perimeter(self) -> pl.Expr:
        """
        Compute contour perimeter (sum of edge lengths).

        Returns:
            Float64 perimeter value.
        """
        return self._plugin("contour_perimeter")

    def centroid(self) -> pl.Expr:
        """
        Compute contour centroid (center of mass).

        Measured on the same region as `area()` — the exterior minus the union of
        the hole rings — so overlapping or nested holes are not subtracted twice.

        Returns:
            Point struct with x, y coordinates.
        """
        return self._plugin("contour_centroid")

    def bounding_box(self) -> pl.Expr:
        """
        Compute axis-aligned bounding box.

        Returns:
            BBox struct with x, y, width, height.
        """
        return self._plugin("contour_bbox")

    def convex_hull(self) -> pl.Expr:
        """
        Compute convex hull of the contour.

        Returns:
            New contour representing the convex hull.
        """
        return self._plugin("contour_convex_hull")

    def is_convex(self) -> pl.Expr:
        """
        Check if contour is convex.

        Returns:
            Boolean indicating convexity.
        """
        return self._plugin("contour_is_convex")

    # --- Transformations ---

    def flip(self) -> pl.Expr:
        """
        Reverse point order (flips winding direction).

        Exterior and holes are reversed together. This changes only what
        `winding()` reports — the region the contour describes is unaffected,
        because no operation reads winding.

        Returns:
            Contour with reversed point order.
        """
        return self._plugin("contour_flip")

    def ensure_winding(self, direction: Winding | str | pl.Expr) -> pl.Expr:
        """
        Ensure contour has specified winding direction.

        Flips the contour if needed to match target winding. Use this when handing
        contours to an external consumer that expects a convention; polars-cv's own
        operations never require one.

        Args:
            direction: Target winding direction. Accepts a Polars expression for
                a per-row choice — rewinding a ring reorders its vertices and
                leaves the output schema untouched.

        Returns:
            Contour with guaranteed winding direction.
        """
        binder = _ArgBinder()
        binder.add_param(
            "direction", _enum_or_expr(direction, Winding, "direction"), cast=str
        )
        return binder.call(self, "contour_ensure_winding")

    def translate(
        self,
        dx: float | pl.Expr,
        dy: float | pl.Expr,
    ) -> pl.Expr:
        """
        Translate contour by offset.

        Args:
            dx: X offset (literal or expression).
            dy: Y offset (literal or expression).

        Returns:
            Translated contour.
        """
        binder = _ArgBinder()
        binder.add_param("dx", dx)
        binder.add_param("dy", dy)
        return binder.call(self, "contour_translate")

    def scale(
        self,
        sx: float | pl.Expr,
        sy: float | pl.Expr,
        *,
        origin: ScaleOrigin | str | pl.Expr = "origin",
    ) -> pl.Expr:
        """
        Scale contour relative to specified origin.

        Args:
            sx: X scale factor (literal or expression).
            sy: Y scale factor (literal or expression).
            origin: Point to scale around, as a literal or an expression:
                - "centroid": Center of mass
                - "bbox_center": Bounding box center
                - "origin": Coordinate origin (0, 0)

        Returns:
            Scaled contour.
        """
        binder = _ArgBinder()
        binder.add_param("sx", sx)
        binder.add_param("sy", sy)
        binder.add_param(
            "origin", _enum_or_expr(origin, ScaleOrigin, "origin"), cast=str
        )
        return binder.call(self, "contour_scale")

    def simplify(self, tolerance: float | pl.Expr) -> pl.Expr:
        """
        Simplify contour using Douglas-Peucker algorithm.

        Args:
            tolerance: Simplification tolerance. Higher = fewer points.
                Accepts a Polars expression for per-row values.

        Returns:
            Simplified contour.
        """
        binder = _ArgBinder()
        binder.add_param("tolerance", tolerance)
        return binder.call(self, "contour_simplify")

    # --- Pairwise Operations ---

    def iou(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Intersection over Union with another contour.

        Overlap is exact for arbitrary simple polygons — concave shapes and holes
        included, in either winding direction. Each contour is measured as its
        exterior minus the union of its hole rings, the same region `area()`,
        `contains_point()` and rasterization use.

        Args:
            other: Another contour column to compare with.

        Returns:
            Float64 IoU value in [0, 1].
        """
        return self._plugin("contour_iou", args=[other])

    def pairwise_iou(self, other: pl.Expr) -> pl.Expr:
        """
        Compute full pairwise IoU matrix between contour sets.

        Args:
            other: Ground-truth contour-set expression (`List[Contour]`).

        Returns:
            A nested list (`List[List[Float64]]`) representing an N x M IoU matrix.
        """
        return self._plugin("contour_pairwise_iou", args=[other])

    def match_detections(
        self,
        other: pl.Expr,
        *,
        threshold: float | pl.Expr = 0.5,
        scores: pl.Expr | None = None,
        strategy: Literal["greedy"] = "greedy",
    ) -> pl.Expr:
        """
        Match prediction contour set against ground truth contour set.

        Args:
            other: Ground-truth contour-set expression (`List[Contour]`).
            threshold: IoU threshold for positive matches. Accepts a Polars
                expression for a per-row threshold.
            scores: Optional per-prediction confidence score list used for ordering.
            strategy: Matching strategy. Currently only ``"greedy"`` is supported.

        Returns:
            A struct matching ``polars_cv.geometry.MATCH_RESULT_SCHEMA`` —
            per-prediction match indices, IoUs, and TP/FP/FN counts.

        Note:
            ``n_fn`` (and the other count fields) are computed **per row**
            (typically one image). Summing ``n_fn`` over a frame that omits
            images with ground truth and no detections undercounts false
            negatives. Keep one row per image in the evaluation population
            (e.g. a full outer join against the image list) before aggregating.
        """
        binder = _ArgBinder()
        binder.add_data("other", other)
        binder.add_data("scores", scores)
        binder.add_param("threshold", threshold)
        return binder.call(self, "contour_match_detections", strategy=strategy)

    def label_reduce(
        self,
        image: pl.Expr | None = None,
        *,
        heatmap: pl.Expr | None = None,
        reduction: LabelReduction | str | pl.Expr = "max",
        region_mode: LabelRegionMode | str | pl.Expr = "interior",
    ) -> pl.Expr:
        """
        Score each contour from an image/array expression with configurable reduction.

        Runs the same engine routine as :meth:`polars_cv.Pipeline.label_reduce`,
        so the two agree on every reduction, region mode and edge case; this
        accessor differs only in taking an already-materialized contour column
        rather than extracting one inside a pipeline.

        Pixels are sampled at their centres. A contour whose region catches no
        pixel centre — a sub-pixel detection — is scored at its centroid rather
        than as 0.0.

        Args:
            image: Image/array expression aligned by row with contour sets.
            heatmap: Backward-compatible alias for ``image``.
            reduction: Aggregation method over pixels in each contour region.
                Accepts a Polars expression for a per-row choice.
            region_mode: Region selector - ``"interior"`` (pixels strictly inside),
                ``"boundary"`` (interior plus the contour boundary) or ``"bbox"``
                (everything in the bounding box). Accepts a Polars expression.

        Returns:
            A list of float scores, aligned to the input contour order.
        """
        if image is None and heatmap is None:
            msg = "Either `image` or `heatmap` must be provided."
            raise ValueError(msg)
        if image is not None and heatmap is not None:
            msg = "Provide only one of `image` or `heatmap`."
            raise ValueError(msg)
        image_expr = image if image is not None else heatmap
        assert image_expr is not None
        binder = _ArgBinder()
        binder.add_data("image", image_expr)
        binder.add_param("reduction", reduction, cast=str)
        binder.add_param("region_mode", region_mode, cast=str)
        return binder.call(self, "contour_label_reduce")

    def dice(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Dice coefficient with another contour.

        Dice = 2 * intersection / (area1 + area2)

        Args:
            other: Another contour column to compare with.

        Returns:
            Float64 Dice coefficient in [0, 1].
        """
        return self._plugin("contour_dice", args=[other])

    def hausdorff_distance(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Hausdorff distance to another contour.

        The maximum, over every *vertex* of either contour, of the distance to the
        nearest vertex of the other. This is a vertex-to-vertex measure, not
        point-to-edge: two contours tracing the same outline with different vertex
        spacing have a non-zero distance. Hole vertices are included. An empty
        contour gives `inf`.

        Args:
            other: Another contour column to compare with.

        Returns:
            Float64 Hausdorff distance.
        """
        return self._plugin("contour_hausdorff", args=[other])

    def contains_point(self, point: pl.Expr) -> pl.Expr:
        """
        Test if contour contains a point.

        Args:
            point: Point column to test.

        Returns:
            Boolean indicating if point is inside contour.
        """
        return self._plugin("contour_contains_point", args=[point])
