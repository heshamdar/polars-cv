"""
Contour operations namespace for Polars expressions.

This module provides the `.contour` accessor for operations on contour columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    pass

# Path to the compiled Rust library
LIB_PATH = Path(__file__).parent.parent


@pl.api.register_expr_namespace("contour")
class ContourNamespace:
    """
    Namespace for geometric operations on contour columns.

    Example:
        >>> df.with_columns(
        ...     area=pl.col("contour").contour.area(),
        ...     bbox=pl.col("contour").contour.bounding_box(),
        ... )
    """

    def __init__(self, expr: pl.Expr) -> None:
        """
        Initialize the namespace.

        Args:
            expr: The Polars expression to operate on.
        """
        self._expr = expr

    # --- Coordinate Operations ---

    def normalize(
        self,
        ref_width: int | pl.Expr,
        ref_height: int | pl.Expr,
    ) -> pl.Expr:
        """
        Convert pixel coordinates to normalized [0,1] range.

        Args:
            ref_width: Reference width for normalization.
            ref_height: Reference height for normalization.

        Returns:
            Contour with coordinates in [0,1] range.

        Raises:
            CoordinateRangeError: If any coordinate exceeds ref dimensions.
        """
        if isinstance(ref_width, pl.Expr) or isinstance(ref_height, pl.Expr):
            raise TypeError(
                "contour normalize does not support expression parameters. "
                "Pass literal int values for ref_width and ref_height."
            )
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_normalize",
            args=[self._expr],
            kwargs={
                "ref_width": ref_width,
                "ref_height": ref_height,
            },
            is_elementwise=True,
        )

    def to_absolute(
        self,
        ref_width: int | pl.Expr,
        ref_height: int | pl.Expr,
    ) -> pl.Expr:
        """
        Convert normalized coordinates to pixel coordinates.

        Args:
            ref_width: Reference width for scaling.
            ref_height: Reference height for scaling.

        Returns:
            Contour with pixel coordinates.
        """
        if isinstance(ref_width, pl.Expr) or isinstance(ref_height, pl.Expr):
            raise TypeError(
                "contour to_absolute does not support expression parameters. "
                "Pass literal int values for ref_width and ref_height."
            )
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_to_absolute",
            args=[self._expr],
            kwargs={
                "ref_width": ref_width,
                "ref_height": ref_height,
            },
            is_elementwise=True,
        )

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
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_winding",
            args=[self._expr],
            is_elementwise=True,
        )

    def area(self, *, signed: bool = False) -> pl.Expr:
        """
        Compute contour area using the Shoelace formula.

        For contours with holes, the hole areas are subtracted.

        Args:
            signed: If True, return signed area (negative for CW winding).
                   If False, return absolute area.

        Returns:
            Float64 area value.

        Raises:
            OpenContourError: If contour is not closed.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_area",
            args=[self._expr],
            kwargs={"signed": signed},
            is_elementwise=True,
        )

    def perimeter(self) -> pl.Expr:
        """
        Compute contour perimeter (sum of edge lengths).

        Returns:
            Float64 perimeter value.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_perimeter",
            args=[self._expr],
            is_elementwise=True,
        )

    def centroid(self) -> pl.Expr:
        """
        Compute contour centroid (center of mass).

        Returns:
            Point struct with x, y coordinates.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_centroid",
            args=[self._expr],
            is_elementwise=True,
        )

    def bounding_box(self) -> pl.Expr:
        """
        Compute axis-aligned bounding box.

        Returns:
            BBox struct with x, y, width, height.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_bbox",
            args=[self._expr],
            is_elementwise=True,
        )

    def convex_hull(self) -> pl.Expr:
        """
        Compute convex hull of the contour.

        Returns:
            New contour representing the convex hull.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_convex_hull",
            args=[self._expr],
            is_elementwise=True,
        )

    def is_convex(self) -> pl.Expr:
        """
        Check if contour is convex.

        Returns:
            Boolean indicating convexity.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_is_convex",
            args=[self._expr],
            is_elementwise=True,
        )

    # --- Transformations ---

    def flip(self) -> pl.Expr:
        """
        Reverse point order (flips winding direction).

        Returns:
            Contour with reversed point order.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_flip",
            args=[self._expr],
            is_elementwise=True,
        )

    def ensure_winding(self, direction: Literal["ccw", "cw"]) -> pl.Expr:
        """
        Ensure contour has specified winding direction.

        Flips the contour if needed to match target winding.

        Args:
            direction: Target winding direction.

        Returns:
            Contour with guaranteed winding direction.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_ensure_winding",
            args=[self._expr],
            kwargs={"direction": direction},
            is_elementwise=True,
        )

    def translate(
        self,
        dx: float | pl.Expr,
        dy: float | pl.Expr,
    ) -> pl.Expr:
        """
        Translate contour by offset.

        Args:
            dx: X offset.
            dy: Y offset.

        Returns:
            Translated contour.
        """
        if isinstance(dx, pl.Expr) or isinstance(dy, pl.Expr):
            raise TypeError(
                "contour translate does not support expression parameters. "
                "Pass literal numeric values for dx and dy."
            )
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_translate",
            args=[self._expr],
            kwargs={
                "dx": dx,
                "dy": dy,
            },
            is_elementwise=True,
        )

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
            sx: X scale factor.
            sy: Y scale factor.
            origin: Point to scale around:
                - "centroid": Center of mass
                - "bbox_center": Bounding box center
                - "origin": Coordinate origin (0, 0)

        Returns:
            Scaled contour.
        """
        if isinstance(sx, pl.Expr) or isinstance(sy, pl.Expr):
            raise TypeError(
                "contour scale does not support expression parameters. "
                "Pass literal numeric values for sx and sy."
            )
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_scale",
            args=[self._expr],
            kwargs={
                "sx": sx,
                "sy": sy,
                "origin": origin,
            },
            is_elementwise=True,
        )

    def simplify(self, tolerance: float) -> pl.Expr:
        """
        Simplify contour using Douglas-Peucker algorithm.

        Args:
            tolerance: Simplification tolerance. Higher = fewer points.

        Returns:
            Simplified contour.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_simplify",
            args=[self._expr],
            kwargs={"tolerance": tolerance},
            is_elementwise=True,
        )

    # --- Pairwise Operations ---

    def iou(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Intersection over Union with another contour.

        Args:
            other: Another contour column to compare with.

        Returns:
            Float64 IoU value in [0, 1].
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_iou",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def pairwise_iou(self, other: pl.Expr) -> pl.Expr:
        """
        Compute full pairwise IoU matrix between contour sets.

        Args:
            other: Ground-truth contour-set expression (`List[Contour]`).

        Returns:
            A nested list (`List[List[Float64]]`) representing an N x M IoU matrix.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_pairwise_iou",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def match_detections(
        self,
        other: pl.Expr,
        *,
        threshold: float = 0.5,
        scores: pl.Expr | None = None,
        strategy: Literal["greedy"] = "greedy",
    ) -> pl.Expr:
        """
        Match prediction contour set against ground truth contour set.

        Args:
            other: Ground-truth contour-set expression (`List[Contour]`).
            threshold: IoU threshold for positive matches.
            scores: Optional per-prediction confidence score list used for ordering.
            strategy: Matching strategy. Currently only ``"greedy"`` is supported.

        Returns:
            A struct containing per-prediction match indices, IoUs, and TP/FP/FN counts.
        """
        args = [self._expr, other]
        if scores is not None:
            args.append(scores)
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_match_detections",
            args=args,
            kwargs={
                "threshold": threshold,
                "strategy": strategy,
            },
            is_elementwise=True,
        )

    def label_reduce(
        self,
        heatmap: pl.Expr,
        *,
        reduction: Literal["max", "mean", "sum"] = "max",
        region_mode: Literal["interior", "bbox"] = "interior",
    ) -> pl.Expr:
        """
        Score each contour from a heatmap with configurable reduction.

        Args:
            heatmap: Heatmap expression aligned by row with contour sets.
            reduction: Aggregation method over pixels in each contour region.
            region_mode: Region selector - ``"interior"`` or ``"bbox"``.

        Returns:
            A list of float scores, aligned to the input contour order.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_label_reduce",
            args=[self._expr, heatmap],
            kwargs={
                "reduction": reduction,
                "region_mode": region_mode,
            },
            is_elementwise=True,
        )

    def dice(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Dice coefficient with another contour.

        Dice = 2 * intersection / (area1 + area2)

        Args:
            other: Another contour column to compare with.

        Returns:
            Float64 Dice coefficient in [0, 1].
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_dice",
            args=[self._expr, other],
            is_elementwise=True,
        )

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
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_hausdorff",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def contains_point(self, point: pl.Expr) -> pl.Expr:
        """
        Test if contour contains a point.

        Args:
            point: Point column to test.

        Returns:
            Boolean indicating if point is inside contour.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="contour_contains_point",
            args=[self._expr, point],
            is_elementwise=True,
        )
