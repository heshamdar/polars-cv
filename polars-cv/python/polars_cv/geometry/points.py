"""
Point operations namespace for Polars expressions.

This module provides the `.point` accessor for operations on point columns.
"""

from __future__ import annotations

import polars as pl

from polars_cv._namespace import _ArgBinder, _GeomNullPolicy, _PluginNamespace


@pl.api.register_expr_namespace("point")
class PointNamespace(_GeomNullPolicy, _PluginNamespace):
    """
    Operations on point columns.

    This namespace provides geometric operations for point data,
    including coordinate transformations and distance calculations.

    The point column must match POINT_SCHEMA or POINT_SET_SCHEMA.
    Operations automatically handle both single points and sets of points.

    Numeric parameters accept either a literal or a Polars expression; an
    expression is resolved per row at execution time.

    Example:
        >>> df.with_columns(
        ...     normalized=pl.col("keypoint").point.normalize(width=100, height=100),
        ...     shifted=pl.col("keypoint").point.translate(dx=10, dy=20),
        ...     per_row=pl.col("keypoint").point.normalize(
        ...         width=pl.col("img_w"), height=pl.col("img_h")
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
            Point with coordinates in [0,1] range.
        """
        binder = _ArgBinder()
        binder.add_param("ref_width", width)
        binder.add_param("ref_height", height)
        return binder.call(self, "point_normalize")

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
            Point with pixel coordinates.
        """
        binder = _ArgBinder()
        binder.add_param("ref_width", width)
        binder.add_param("ref_height", height)
        return binder.call(self, "point_to_absolute")

    def translate(
        self,
        dx: float | pl.Expr,
        dy: float | pl.Expr,
    ) -> pl.Expr:
        """
        Translate point by offset.

        Args:
            dx: X offset (literal or expression).
            dy: Y offset (literal or expression).

        Returns:
            Translated point.
        """
        binder = _ArgBinder()
        binder.add_param("dx", dx)
        binder.add_param("dy", dy)
        return binder.call(self, "point_translate")

    def scale(
        self,
        sx: float | pl.Expr,
        sy: float | pl.Expr,
    ) -> pl.Expr:
        """
        Scale point coordinates.

        Args:
            sx: X scale factor (literal or expression).
            sy: Y scale factor (literal or expression).

        Returns:
            Scaled point.
        """
        binder = _ArgBinder()
        binder.add_param("sx", sx)
        binder.add_param("sy", sy)
        return binder.call(self, "point_scale")

    # --- Distance Operations ---

    def distance(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Euclidean distance to another point.

        Args:
            other: Another point column.

        Returns:
            Float64 distance.
        """
        return self._plugin("point_distance", args=[other])

    def manhattan_distance(self, other: pl.Expr) -> pl.Expr:
        """
        Compute Manhattan (L1) distance to another point.

        Args:
            other: Another point column.

        Returns:
            Float64 distance.
        """
        return self._plugin("point_manhattan_distance", args=[other])

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
        return self._plugin("point_distance_to_contour", args=[contour])

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
        return self._plugin("point_signed_distance_to_contour", args=[contour])

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
        return self._plugin("point_nearest_on_contour", args=[contour])

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
        return self._plugin("point_angle_to", args=[other])

    def rotate(
        self, angle: float | pl.Expr, *, origin: pl.Expr | None = None
    ) -> pl.Expr:
        """
        Rotate point around an origin by angle (in radians).

        Args:
            angle: Rotation angle in radians (counter-clockwise positive).
                Accepts a Polars expression for a per-row angle.
            origin: Center of rotation. If None, rotates around (0, 0).

        Returns:
            Rotated point.

        Example:
            >>> import math
            >>> df.with_columns(
            ...     rotated=pl.col("point").point.rotate(math.pi / 2)  # 90 degrees
            ... )
        """
        binder = _ArgBinder()
        binder.add_data("origin", origin)
        binder.add_param("angle", angle)
        return binder.call(self, "point_rotate")

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
        return self._plugin("point_midpoint", args=[other])

    def interpolate(self, other: pl.Expr, t: float | pl.Expr = 0.5) -> pl.Expr:
        """
        Linear interpolation between two points.

        Args:
            other: Target point column.
            t: Interpolation parameter (0 = self, 1 = other).
                Values outside [0, 1] extrapolate beyond the endpoints.
                Accepts a Polars expression for a per-row parameter.

        Returns:
            Interpolated point.

        Example:
            >>> df.with_columns(
            ...     quarter=pl.col("p1").point.interpolate(pl.col("p2"), t=0.25)
            ... )
        """
        binder = _ArgBinder()
        binder.add_data("other", other)
        binder.add_param("t", t)
        return binder.call(self, "point_interpolate")

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
        return self._plugin("point_within_bbox", args=[bbox])

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
