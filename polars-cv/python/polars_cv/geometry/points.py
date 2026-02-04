"""
Point operations namespace for Polars expressions.

This module provides the `.point` accessor for operations on point columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    pass

# Path to the compiled Rust library
LIB_PATH = Path(__file__).parent.parent


@pl.api.register_expr_namespace("point")
class PointNamespace:
    """
    Operations on point columns.

    This namespace provides geometric operations for point data,
    including coordinate transformations and distance calculations.

    The point column must match POINT_SCHEMA or POINT_SET_SCHEMA.
    Operations automatically handle both single points and sets of points.

    Example:
        >>> df.with_columns(
        ...     normalized=pl.col("keypoint").point.normalize(width=100, height=100),
        ...     shifted=pl.col("keypoint").point.translate(dx=10, dy=20),
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
        ref_width: int | float,
        ref_height: int | float,
    ) -> pl.Expr:
        """
        Convert pixel coordinates to normalized [0,1] range.

        Args:
            ref_width: Reference width for normalization.
            ref_height: Reference height for normalization.

        Returns:
            Point with coordinates in [0,1] range.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_normalize",
            args=[self._expr],
            kwargs={
                "ref_width": float(ref_width),
                "ref_height": float(ref_height),
            },
            is_elementwise=True,
        )

    def to_absolute(
        self,
        ref_width: int | float,
        ref_height: int | float,
    ) -> pl.Expr:
        """
        Convert normalized coordinates to pixel coordinates.

        Args:
            ref_width: Reference width for scaling.
            ref_height: Reference height for scaling.

        Returns:
            Point with pixel coordinates.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_to_absolute",
            args=[self._expr],
            kwargs={
                "ref_width": float(ref_width),
                "ref_height": float(ref_height),
            },
            is_elementwise=True,
        )

    def translate(
        self,
        dx: float,
        dy: float,
    ) -> pl.Expr:
        """
        Translate point by offset.

        Args:
            dx: X offset.
            dy: Y offset.

        Returns:
            Translated point.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_translate",
            args=[self._expr],
            kwargs={
                "dx": float(dx),
                "dy": float(dy),
            },
            is_elementwise=True,
        )

    def scale(
        self,
        sx: float,
        sy: float,
    ) -> pl.Expr:
        """
        Scale point coordinates.

        Args:
            sx: X scale factor.
            sy: Y scale factor.

        Returns:
            Scaled point.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_scale",
            args=[self._expr],
            kwargs={
                "sx": float(sx),
                "sy": float(sy),
            },
            is_elementwise=True,
        )

    # --- Distance Operations ---

    def distance(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Euclidean distance to another point.

        Args:
            other: Another point column.

        Returns:
            Float64 distance.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_distance",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def manhattan_distance(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Manhattan (L1) distance to another point.

        Args:
            other: Another point column.

        Returns:
            Float64 distance.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_manhattan_distance",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def distance_to_contour(self, contour: pl.Expr) -> pl.Expr:
        """
        Compute minimum distance from point to contour boundary.

        This computes the distance to the nearest edge of the contour,
        considering both the exterior ring and any holes.

        Args:
            contour: A contour column.

        Returns:
            Float64 distance (always non-negative).

        Example:
            >>> df.with_columns(
            ...     dist=pl.col("point").point.distance_to_contour(pl.col("polygon"))
            ... )
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_distance_to_contour",
            args=[self._expr, contour],
            is_elementwise=True,
        )

    def signed_distance_to_contour(self, contour: pl.Expr) -> pl.Expr:
        """
        Compute signed distance from point to contour boundary.

        Returns negative distance if point is inside the contour,
        positive distance if outside. Zero if on the boundary.

        Args:
            contour: A contour column.

        Returns:
            Float64 signed distance.

        Example:
            >>> df.with_columns(
            ...     sdf=pl.col("point").point.signed_distance_to_contour(pl.col("polygon"))
            ... )
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_signed_distance_to_contour",
            args=[self._expr, contour],
            is_elementwise=True,
        )

    def nearest_point_on_contour(self, contour: pl.Expr) -> pl.Expr:
        """
        Find the nearest point on a contour boundary.

        Returns the point on the contour boundary that is closest to this point.

        Args:
            contour: A contour column.

        Returns:
            Point struct with x, y coordinates of nearest boundary point.

        Example:
            >>> df.with_columns(
            ...     nearest=pl.col("point").point.nearest_point_on_contour(pl.col("polygon"))
            ... )
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_nearest_on_contour",
            args=[self._expr, contour],
            is_elementwise=True,
        )

    # --- Geometric Operations ---

    def angle_to(self, other: pl.Expr) -> pl.Expr:
        """
        Compute angle from this point to another in radians.

        Returns the angle in [-pi, pi] using atan2, where:
        - 0 radians points to the right (positive x)
        - pi/2 radians points up (positive y)
        - pi radians points left (negative x)
        - -pi/2 radians points down (negative y)

        Args:
            other: Target point column.

        Returns:
            Float64 angle in radians.

        Example:
            >>> df.with_columns(
            ...     angle=pl.col("p1").point.angle_to(pl.col("p2"))
            ... )
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_angle_to",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def rotate(self, angle: float, *, origin: pl.Expr | None = None) -> pl.Expr:
        """
        Rotate point around an origin by angle (in radians).

        Args:
            angle: Rotation angle in radians (counter-clockwise positive).
            origin: Center of rotation. If None, rotates around (0, 0).

        Returns:
            Rotated point.

        Example:
            >>> import math
            >>> df.with_columns(
            ...     rotated=pl.col("point").point.rotate(math.pi / 2)  # 90 degrees
            ... )
        """
        args = [self._expr]
        if origin is not None:
            args.append(origin)

        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_rotate",
            args=args,
            kwargs={"angle": float(angle)},
            is_elementwise=True,
        )

    def midpoint(self, other: pl.Expr) -> pl.Expr:
        """
        Compute midpoint between this point and another.

        Args:
            other: Another point column.

        Returns:
            Point at the midpoint.

        Example:
            >>> df.with_columns(
            ...     mid=pl.col("p1").point.midpoint(pl.col("p2"))
            ... )
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_midpoint",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def interpolate(self, other: pl.Expr, t: float = 0.5) -> pl.Expr:
        """
        Linear interpolation between two points.

        Args:
            other: Target point column.
            t: Interpolation parameter (0 = self, 1 = other).
               Values outside [0, 1] extrapolate beyond the endpoints.

        Returns:
            Interpolated point.

        Example:
            >>> df.with_columns(
            ...     quarter=pl.col("p1").point.interpolate(pl.col("p2"), t=0.25)
            ... )
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_interpolate",
            args=[self._expr, other],
            kwargs={"t": float(t)},
            is_elementwise=True,
        )

    def within_bbox(self, bbox: pl.Expr) -> pl.Expr:
        """
        Check if point is within a bounding box.

        Args:
            bbox: Bounding box column with x, y, width, height fields.

        Returns:
            Boolean indicating if point is within the bbox.

        Example:
            >>> df.with_columns(
            ...     inside=pl.col("point").point.within_bbox(pl.col("bbox"))
            ... )
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_within_bbox",
            args=[self._expr, bbox],
            is_elementwise=True,
        )

    # --- Extraction ---

    def x(self) -> pl.Expr:
        """
        Extract X coordinate.

        Returns:
            Float64 X coordinate.
        """
        return self._expr.struct.field("x")

    def y(self) -> pl.Expr:
        """
        Extract Y coordinate.

        Returns:
            Float64 Y coordinate.
        """
        return self._expr.struct.field("y")
