"""
Contour operations namespace for Polars expressions.

This module provides the `.contour` accessor for operations on contour columns.
"""

from __future__ import annotations

import polars as pl

from polars_cv._namespace import _GeomNamespace
from polars_cv._ops_generated import _ContourOpsMixin


@pl.api.register_expr_namespace("contour")
class ContourNamespace(_ContourOpsMixin, _GeomNamespace):
    """
    Namespace for geometric operations on contour columns.

    **One contour per row or a set of them — every accessor takes both.** A
    column may be a ``CONTOUR_SCHEMA`` struct per row, or the
    ``CONTOUR_SET_SCHEMA`` list of them that ``extract_contours()`` produces.
    The two are told apart by the column's dtype, and the result is wrapped to
    match: an accessor returning ``Float64`` for a single contour returns
    ``List(Float64)`` for a set, one entry per contour, in input order.

    Two-operand accessors (:meth:`iou`, :meth:`dice`,
    :meth:`hausdorff_distance`) **broadcast**: a set on one side and a single
    contour on the other gives one result per contour in the set, whichever
    side the set is on. A set on *both* sides is rejected rather than guessed —
    it could mean the N×M matrix (:meth:`pairwise_iou`) or an index-wise
    pairing (``.explode()`` one side), and the two mean different things.

    The set-level accessors (:meth:`pairwise_iou`, :meth:`correspond`,
    :meth:`label_reduce`) work the other way round for the same reason: a lone
    contour is read as a set of one.

    Numeric parameters accept either a literal or a Polars expression; an
    expression is resolved per row at execution time. A per-row parameter
    applies to every contour in that row's set — parameters vary by row, not by
    contour.

    Example:
        >>> df.with_columns(
        ...     area=pl.col("contour").contour.area(),
        ...     bbox=pl.col("contour").contour.bounding_box(),
        ...     norm=pl.col("contour").contour.normalize(
        ...         pl.col("img_w"), pl.col("img_h")
        ...     ),
        ... )
        >>> # A set column: one area per contour, per row.
        >>> df.with_columns(areas=pl.col("contours").contour.area())
    """
