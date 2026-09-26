"""
Point operations namespace for Polars expressions.

This module provides the `.point` accessor for operations on point columns.
"""

from __future__ import annotations

import polars as pl

from polars_cv._namespace import _GeomNamespace
from polars_cv._ops_generated import _PointOpsMixin


@pl.api.register_expr_namespace("point")
class PointNamespace(_PointOpsMixin, _GeomNamespace):
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

    # --- Extraction ---

    def x(self) -> pl.Expr:
        """
        Extract X coordinate.

        Returns:
            Float64 X coordinate.
        """
        return self._expr.ext.storage().struct.field("x")

    def y(self) -> pl.Expr:
        """
        Extract Y coordinate.

        Returns:
            Float64 Y coordinate.
        """
        return self._expr.ext.storage().struct.field("y")
