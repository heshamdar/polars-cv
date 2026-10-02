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

    The point column holds one point per row (``POINT_SCHEMA``) or a point set
    (``POINT_SET_SCHEMA``, ``List(point)``) per row. Over a set every method
    gives one value per point, in input order (a null point gives a null in
    its place), as a ``List``. A contour or bbox operand is one per row and
    broadcasts against the set; two point columns broadcast either way (a set
    against a single point), and a set on both sides is refused.

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
