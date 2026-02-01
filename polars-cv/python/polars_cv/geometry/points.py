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
        ref_width: int | pl.Expr,
        ref_height: int | pl.Expr,
    ) -> pl.Expr:
        """
        Convert pixel coordinates to normalized [0,1] range.

        Args:
            ref_width: Reference width for normalization.
            ref_height: Reference height for normalization.

        Returns:
            Point with coordinates in [0,1] range.
        """
        if isinstance(ref_width, pl.Expr) or isinstance(ref_height, pl.Expr):
            raise TypeError("point.normalize() does not support pl.Expr arguments; pass literal int values")
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_normalize",
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
            Point with pixel coordinates.
        """
        if isinstance(ref_width, pl.Expr) or isinstance(ref_height, pl.Expr):
            raise TypeError("point.to_absolute() does not support pl.Expr arguments; pass literal int values")
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_to_absolute",
            args=[self._expr],
            kwargs={
                "ref_width": ref_width,
                "ref_height": ref_height,
            },
            is_elementwise=True,
        )

    def translate(
        self,
        dx: float | pl.Expr,
        dy: float | pl.Expr,
    ) -> pl.Expr:
        """
        Translate point by offset.

        Args:
            dx: X offset.
            dy: Y offset.

        Returns:
            Translated point.
        """
        if isinstance(dx, pl.Expr) or isinstance(dy, pl.Expr):
            raise TypeError("point.translate() does not support pl.Expr arguments; pass literal numeric values")
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_translate",
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
    ) -> pl.Expr:
        """
        Scale point coordinates.

        Args:
            sx: X scale factor.
            sy: Y scale factor.

        Returns:
            Scaled point.
        """
        if isinstance(sx, pl.Expr) or isinstance(sy, pl.Expr):
            raise TypeError("point.scale() does not support pl.Expr arguments; pass literal numeric values")
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="point_scale",
            args=[self._expr],
            kwargs={
                "sx": sx,
                "sy": sy,
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
