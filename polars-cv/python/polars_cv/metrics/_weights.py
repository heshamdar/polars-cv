"""Per-key weight resolution — the single authority for collapsing duplicate
metadata weights to one value per lookup key.

``image_metadata`` may carry more than one row for a ``(image[, class])`` key
(one rendered image owned by two cases, a bootstrap redraw, …). Weighted FROC/
LROC needs exactly one weight per key: the numerator attaches a weight to each
detection while the denominators aggregate metadata, so the two must read the
*same* per-key weight or the result is order-dependent.

Rather than guard against disagreeing duplicates with an eager collect (which
would break the lazy/streaming plan), this resolves each key deterministically
via a caller-chosen aggregate. ``"first"`` keeps the cheap ``unique`` semantics
and is **not** guaranteed stable when a key's weights actually disagree — that
is the caller's responsibility. The other aggregates are order-independent.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from ._types import COL_CLASS_ID, COL_IMAGE_ID, COL_WEIGHT

#: How to reduce a key's duplicate weights to a single value. ``"first"`` is the
#: default and matches the historical ``unique(keep="first")`` behaviour.
WeightAgg = Literal["first", "min", "max", "mean", "sum"]

_WEIGHT_AGGS: tuple[WeightAgg, ...] = ("first", "min", "max", "mean", "sum")


def resolve_key_weights(
    meta: pl.LazyFrame,
    keys: list[str],
    agg: WeightAgg = "first",
) -> pl.LazyFrame:
    """Reduce ``COL_WEIGHT`` to one value per ``keys`` group, lazily.

    Args:
        meta: Image-metadata lazy frame carrying ``keys`` and ``COL_WEIGHT``.
        keys: Columns identifying one weight-lookup unit (e.g. ``[image_id]`` or
            ``[image_id, class_id]``).
        agg: Resolution policy. ``"first"`` returns
            ``select(*keys, weight).unique(subset=keys, keep="first")`` (order of
            the kept row is not guaranteed when weights disagree); ``"min"`` /
            ``"max"`` / ``"mean"`` / ``"sum"`` return the order-independent
            aggregate per key.

    Returns:
        A lazy frame with columns ``[*keys, COL_WEIGHT]``, one row per key.

    Raises:
        ValueError: If ``agg`` is not a recognised policy.
    """
    if agg not in _WEIGHT_AGGS:
        raise ValueError(
            f"Unknown weight_agg {agg!r}. Expected one of {list(_WEIGHT_AGGS)}."
        )
    lookup = meta.select(*keys, COL_WEIGHT)
    if agg == "first":
        return lookup.unique(subset=keys, keep="first")
    weight = pl.col(COL_WEIGHT)
    reduced = {
        "min": weight.min(),
        "max": weight.max(),
        "mean": weight.mean(),
        "sum": weight.sum(),
    }[agg]
    return lookup.group_by(keys).agg(reduced.alias(COL_WEIGHT))


def image_weight_keys(meta: pl.LazyFrame) -> list[str]:
    """The lookup key identifying one weight unit on the metadata frame."""
    return (
        [COL_IMAGE_ID, COL_CLASS_ID]
        if COL_CLASS_ID in set(meta.collect_schema().names())
        else [COL_IMAGE_ID]
    )


def attach_resolved_weight(
    detections: pl.LazyFrame,
    meta: pl.LazyFrame,
    *,
    weight_agg: WeightAgg = "first",
) -> pl.LazyFrame:
    """Join the resolved per-(image[, class]) weight onto a detections frame.

    Detections carry no weight of their own; the weighted detection-level
    Mann-Whitney statistic needs each detection tagged with its image's weight.
    The weight is resolved once per key via :func:`resolve_key_weights` (so a
    repeated metadata row cannot fan detections or make the value order-dependent)
    and missing keys default to ``1.0``.

    Args:
        detections: Per-detection frame carrying ``image_id`` (and ``class_id``
            when the metadata is keyed on it).
        meta: Image-metadata frame carrying the weight.
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        ``detections`` with a ``COL_WEIGHT`` column.
    """
    keys = image_weight_keys(meta)
    resolved = resolve_key_weights(meta, keys, weight_agg)
    return detections.join(resolved, on=keys, how="left").with_columns(
        pl.col(COL_WEIGHT).fill_null(1.0)
    )
