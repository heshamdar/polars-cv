"""Points, contours and boxes from plain coordinate lists.

The inverse of ``.point.to_coords()`` / ``.contour.to_coords()``. Each pair is
``[x, y]`` or ``[y, x]`` (row, column — NumPy's convention), as ``order``
says; integers and ``Array(_, 2)`` pairs are accepted alike. A pair that does
not hold exactly two non-null, finite numbers is an error naming its row.
:func:`bbox_from_coords` reads the common box layouts into ``BBOX_SCHEMA``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, get_args

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


#: Box layouts: corners (Pascal VOC, torchvision), corner + size (COCO), and
#: centre + size (YOLO). Coordinates are in the same units as the image (or
#: all normalised); ``BBOX_SCHEMA`` is corner + size.
BoxFormat = Literal["xyxy", "xywh", "cxcywh"]


def bbox_from_coords(
    coords: str | pl.Expr | Sequence[str | pl.Expr],
    *,
    format: BoxFormat,  # noqa: A002 — the conventional name for a box layout
) -> pl.Expr:
    """A box (``BBOX_SCHEMA``: ``x, y, width, height``) from four coordinates.

    Args:
        coords: Either one column holding four numbers per row (``List`` or
            ``Array(_, 4)``), or four columns / expressions in layout order.
        format: The layout of the four numbers (required — the three are
            indistinguishable from the data):

            * ``"xyxy"``: ``x_min, y_min, x_max, y_max``;
            * ``"xywh"``: ``x_min, y_min, width, height`` (COCO);
            * ``"cxcywh"``: ``x_centre, y_centre, width, height`` (YOLO).

    Returns:
        A ``BBOX_SCHEMA`` expression named after the first input column; a
        null box gives a null struct, and a row whose box does not hold four
        numbers is an error.

    Raises:
        ValueError: An unknown ``format``, or a sequence of other than four
            columns.
    """
    formats = get_args(BoxFormat)
    if format not in formats:
        msg = f"format must be one of {list(formats)}, got {format!r}"
        raise ValueError(msg)
    if isinstance(coords, (str, pl.Expr)):
        packed = (pl.col(coords) if isinstance(coords, str) else coords).cast(
            pl.Array(pl.Float64, 4)
        )
        a, b, c, d = (packed.arr.get(i) for i in range(4))
        name, valid = packed, packed.is_not_null()
    else:
        cols = [pl.col(e) if isinstance(e, str) else e for e in coords]
        if len(cols) != 4:
            msg = f"bbox_from_coords takes one column or 4 columns, got {len(cols)}"
            raise ValueError(msg)
        a, b, c, d = (e.cast(pl.Float64) for e in cols)
        name, valid = cols[0], pl.lit(True)
    if format == "xyxy":
        fields = (a, b, c - a, d - b)
    elif format == "xywh":
        fields = (a, b, c, d)
    else:
        fields = (a - c / 2.0, b - d / 2.0, c, d)
    box = pl.struct(
        fields[0].alias("x"),
        fields[1].alias("y"),
        fields[2].alias("width"),
        fields[3].alias("height"),
    )
    return pl.when(valid).then(box).otherwise(None).alias(name.meta.output_name())
