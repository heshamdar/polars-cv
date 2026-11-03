"""The matchers' ``carry=`` and ``iou_threshold=`` arguments, read once.

Generic: nothing here knows how a matcher pairs its predictions, so the
pre-matched adapter and the polars-cv matchers share it (see
``tests/_metrics_boundary.py``).
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from .._types import (
    COL_GROUP_ID,
    COL_IOU_THRESHOLD,
    IMAGE_META_SCHEMA,
    ensure_columns_exist,
)

#: What a matcher's ``carry=`` takes: columns of the frame that defines
#: ``image_metadata``, or expressions over it, each named by its output name.
Carry = Sequence[str | pl.Expr]

#: The ``image_metadata`` columns a table defines itself.
_RESERVED = (*IMAGE_META_SCHEMA, COL_GROUP_ID, COL_IOU_THRESHOLD)


def carried(carry: Carry | None, schema_names: Sequence[str]) -> dict[str, pl.Expr]:
    """``carry`` as ``{name: expression}``, checked against ``schema_names``.

    The one reading of every matcher's ``carry=``: a string is that column, an
    expression is named by its output name (so it must have one: a bare
    literal is refused rather than called ``"literal"``).

    Raises:
        ValueError: A named column is missing, an expression has no name of
            its own or several outputs, a name repeats, or a name is one
            ``image_metadata`` defines itself.
        TypeError: An entry is neither a string nor an expression.
    """
    out: dict[str, pl.Expr] = {}
    for item in carry or ():
        if isinstance(item, str):
            name, expr = item, pl.col(item)
        elif isinstance(item, pl.Expr):
            if item.meta.has_multiple_outputs():
                raise ValueError(f"carry: {item} has several outputs; carry each")
            aliased = not item.meta.undo_aliases().meta.eq(item)
            if not aliased and not item.meta.root_names():
                raise ValueError(
                    f"carry: {item} reads no column, so its name would be "
                    "'literal'; name it with .alias()"
                )
            name, expr = item.meta.output_name(), item
        else:
            raise TypeError(f"carry takes column names or expressions, got {item!r}")
        if name in _RESERVED:
            raise ValueError(
                f"carry: {name!r} is reserved: image_metadata defines it itself"
            )
        if name in out:
            raise ValueError(f"carry: {name!r} is carried more than once")
        if isinstance(item, str):
            ensure_columns_exist(schema_names, [item])
        out[name] = expr
    return out


def iou_thresholds(value: float | Sequence[float]) -> tuple[float, ...]:
    """A matcher's ``iou_threshold`` as a tuple of thresholds.

    One number is one matching; a sequence is a sweep (one matching per
    threshold, COCO-style). Each must lie in ``(0, 1]``, and a sweep's must be
    distinct.

    Raises:
        ValueError: An empty sequence, a repeated threshold, or one outside
            ``(0, 1]``.
    """
    ts = (
        (float(value),)
        if isinstance(value, (int, float))
        else tuple(float(t) for t in value)
    )
    if not ts:
        raise ValueError("`iou_threshold` needs at least one threshold.")
    if len(set(ts)) != len(ts):
        raise ValueError(f"`iou_threshold` thresholds must be distinct, got {ts}.")
    if not all(0.0 < t <= 1.0 for t in ts):
        raise ValueError(f"`iou_threshold` must be in (0, 1], got {ts}.")
    return ts
