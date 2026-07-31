"""
Contour operations namespace for Polars expressions.

This module provides the `.contour` accessor for operations on contour columns.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from polars_cv._namespace import _ArgBinder, _GeomNullPolicy, _PluginNamespace


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
        Compute winding direction from point order.

        Returns:
            String 'ccw' for counter-clockwise, 'cw' for clockwise.

        Note:
            Winding is computed using the Shoelace formula:
            - Positive signed area = CCW
            - Negative signed area = CW
        """
        return self._plugin("contour_winding")

    def area(self, *, signed: bool | pl.Expr = False) -> pl.Expr:
        """
        Compute contour area using the Shoelace formula.

        For contours with holes, the hole areas are subtracted.

        Args:
            signed: If True, return signed area (negative for CW winding).
                   If False, return absolute area. Accepts a boolean expression
                   for per-row selection.

        Returns:
            Float64 area value.

        Raises:
            OpenContourError: If contour is not closed.
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

        Returns:
            Contour with reversed point order.
        """
        return self._plugin("contour_flip")

    def ensure_winding(self, direction: Literal["ccw", "cw"]) -> pl.Expr:
        """
        Ensure contour has specified winding direction.

        Flips the contour if needed to match target winding.

        Args:
            direction: Target winding direction.

        Returns:
            Contour with guaranteed winding direction.
        """
        return self._plugin("contour_ensure_winding", kwargs={"direction": direction})

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
        origin: Literal["centroid", "bbox_center", "origin"] = "origin",
    ) -> pl.Expr:
        """
        Scale contour relative to specified origin.

        Args:
            sx: X scale factor (literal or expression).
            sy: Y scale factor (literal or expression).
            origin: Point to scale around:
                - "centroid": Center of mass
                - "bbox_center": Bounding box center
                - "origin": Coordinate origin (0, 0)

        Returns:
            Scaled contour.
        """
        binder = _ArgBinder()
        binder.add_param("sx", sx)
        binder.add_param("sy", sy)
        return binder.call(self, "contour_scale", origin=origin)

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
            A struct containing per-prediction match indices, IoUs, and TP/FP/FN counts.
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
        reduction: Literal["max", "mean", "sum"] | pl.Expr = "max",
        region_mode: Literal["interior", "bbox"] | pl.Expr = "interior",
    ) -> pl.Expr:
        """
        Score each contour from an image/array expression with configurable reduction.

        Args:
            image: Image/array expression aligned by row with contour sets.
            heatmap: Backward-compatible alias for ``image``.
            reduction: Aggregation method over pixels in each contour region.
                Accepts a Polars expression for a per-row choice, matching
                :meth:`polars_cv.Pipeline.label_reduce`.
            region_mode: Region selector - ``"interior"`` or ``"bbox"``.
                Accepts a Polars expression.

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

        The Hausdorff distance is the maximum distance from any point
        on one contour to the nearest point on the other.

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
