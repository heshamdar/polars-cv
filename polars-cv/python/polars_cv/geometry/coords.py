"""Points and contours from plain coordinate lists.

The inverse of ``.point.to_coords()`` / ``.contour.to_coords()``. Each pair is
``[x, y]`` or ``[y, x]`` (row, column — NumPy's convention), as ``order``
says; integers and ``Array(_, 2)`` pairs are accepted alike. A pair that does
not hold exactly two non-null, finite numbers is an error naming its row.
"""

from __future__ import annotations

import polars as pl

from polars_cv import _plugin
from polars_cv._ops_generated import CoordOrder


def _order(order: str | CoordOrder) -> str:
    try:
        return CoordOrder(order).value
    except ValueError:
        names = [o.value for o in CoordOrder]
        msg = f"order must be one of {names}, got {order!r}"
        raise ValueError(msg) from None


def point_from_coords(expr: pl.Expr, *, order: str | CoordOrder = "xy") -> pl.Expr:
    """A point (``POINT_SCHEMA``) from each ``[a, b]`` pair.

    Args:
        expr: A column of pairs: ``List`` or ``Array(_, 2)`` of numbers.
        order: ``"xy"`` or ``"yx"`` (row, column).

    Returns:
        A ``POINT_SCHEMA`` expression; a null pair gives a null point.
    """
    return _plugin.call(
        "geom_point_from_coords",
        args=[expr.cast(pl.List(pl.Float64))],
        kwargs={"order": _order(order)},
    )


def contour_from_coords(
    expr: pl.Expr, *, order: str | CoordOrder = "xy", closed: bool = True
) -> pl.Expr:
    """A contour (``CONTOUR_SCHEMA``) from each list of ``[a, b]`` pairs.

    Args:
        expr: A column of rings: ``List`` of pairs.
        order: ``"xy"`` or ``"yx"`` (row, column).
        closed: ``True`` for a region (the ring closes back on its first
            point), ``False`` for an open polyline.

    Returns:
        A ``CONTOUR_SCHEMA`` expression, without holes.
    """
    return _plugin.call(
        "geom_contour_from_coords",
        args=[expr.cast(pl.List(pl.List(pl.Float64)))],
        kwargs={"order": _order(order), "closed": bool(closed)},
    )


def contour_set_from_coords(
    expr: pl.Expr, *, order: str | CoordOrder = "xy", closed: bool = True
) -> pl.Expr:
    """A contour set (``CONTOUR_SET_SCHEMA``) from each list of rings.

    Args:
        expr: A column of ring lists: ``List`` of ``List`` of pairs.
        order: ``"xy"`` or ``"yx"`` (row, column).
        closed: ``True`` for regions, ``False`` for open polylines.

    Returns:
        A ``CONTOUR_SET_SCHEMA`` expression.
    """
    return _plugin.call(
        "geom_contour_set_from_coords",
        args=[expr.cast(pl.List(pl.List(pl.List(pl.Float64))))],
        kwargs={"order": _order(order), "closed": bool(closed)},
    )
