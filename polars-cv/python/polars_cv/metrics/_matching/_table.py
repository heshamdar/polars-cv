"""The matchers' shared tail: per-image correspondences → ``DetectionTable``.

Every matcher prepares one row per (image, class) holding its predictions'
scores and its ground-truth objects, and a way to pair them at an IoU
threshold. What follows is the same for all of them and lives here once:
matching at each requested threshold, exploding the pairings into
per-detection rows that keep their row's class (and threshold), and the
per-(image, class) metadata.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import polars as pl

from ...geometry.schemas import CORRESPONDENCE_SCHEMA
from .._types import (
    COL_CLASS_ID,
    COL_DET_IDX,
    COL_GROUP_ID,
    COL_GT_IDX,
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IOU,
    COL_IOU_THRESHOLD,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DEFAULT_CLASS,
    DetectionTable,
)

_RIGHT_IDX, _OVERLAP, _DUPLICATE = (f.name for f in CORRESPONDENCE_SCHEMA.fields)


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


def matched_table(
    prepared: pl.LazyFrame,
    *,
    match: Callable[[float | pl.Expr], pl.Expr],
    thresholds: tuple[float, ...] | pl.Expr,
    scores_col: str,
    gt_col: str,
    class_col: str | None,
    group_col: str | None,
    ignore_duplicates: bool = False,
) -> DetectionTable:
    """Match ``prepared`` at every threshold and build the table.

    Args:
        prepared: One row per (image, class) with ``image_id``, ``weight``,
            the prediction scores (``scores_col``, a list aligned with the
            predictions) and the ground-truth objects (``gt_col``, a list).
        match: The correspondence expression at a threshold (a
            ``CORRESPONDENCE_SCHEMA`` struct per row).
        thresholds: The thresholds (several for a sweep), or one per-row
            expression.
        scores_col: The prediction score list column.
        gt_col: The ground-truth object list column (its length is ``n_gts``).
        class_col: The row's class column, or ``None`` for one class.
        group_col: An optional grouping column copied to the metadata.
        ignore_duplicates: Drop a repeated hit on a claimed target rather than
            count it as a false positive.
    """
    ts: Sequence[float | pl.Expr]
    if isinstance(thresholds, pl.Expr):
        # A per-row threshold has no single value to warn against.
        ts, matching, sweep_ts = (thresholds,), None, ()
    else:
        ts, matching = thresholds, min(thresholds)
        sweep_ts = thresholds if len(thresholds) > 1 else ()
    sweep = bool(sweep_ts)
    # A sweep reads the prepared rows once per threshold; caching them keeps
    # any extraction upstream to one run.
    base = prepared.cache() if sweep else prepared
    gt = pl.col(gt_col)
    levels = [
        base.with_columns(_match=match(t)).select(
            COL_IMAGE_ID,
            COL_WEIGHT,
            pl.col(scores_col).alias("_scores"),
            gt.list.len().fill_null(0).cast(pl.Int64).alias(COL_N_GTS),
            (gt.list.len().fill_null(0) > 0).alias(COL_GT_LABEL),
            pl.col("_match").struct.field(_RIGHT_IDX).alias("_gt_idx"),
            pl.col("_match").struct.field(_OVERLAP).alias("_iou"),
            pl.col("_match").struct.field(_DUPLICATE).alias("_dup"),
            (
                pl.col(class_col).cast(pl.String)
                if class_col is not None
                else pl.lit(DEFAULT_CLASS)
            ).alias(COL_CLASS_ID),
            *(
                [pl.col(group_col).cast(pl.String).alias(COL_GROUP_ID)]
                if group_col is not None
                else []
            ),
            *([pl.lit(t, dtype=pl.Float64).alias(COL_IOU_THRESHOLD)] if sweep else []),
        )
        for t in ts
    ]
    # Cached so the detections and the metadata derived below run the
    # correspond graph once under the caller's single collect.
    image_level = (levels[0] if len(levels) == 1 else pl.concat(levels)).cache()
    carry = [COL_CLASS_ID, *([COL_IOU_THRESHOLD] if sweep else [])]
    detections = explode_matches(
        image_level, carry=carry, ignore_duplicates=ignore_duplicates
    )
    meta = image_level.select(
        COL_IMAGE_ID,
        COL_CLASS_ID,
        COL_N_GTS,
        COL_WEIGHT,
        COL_GT_LABEL,
        *([COL_GROUP_ID] if group_col is not None else []),
        *([COL_IOU_THRESHOLD] if sweep else []),
    )
    table = DetectionTable.from_matched(
        detections, meta, matching_iou_threshold=matching
    )
    return replace(table, _sweep=sweep_ts) if sweep else table


def explode_matches(
    image_level: pl.LazyFrame,
    *,
    carry: list[str],
    ignore_duplicates: bool = False,
) -> pl.LazyFrame:
    """Per-image pairings → one row per detection, keeping the ``carry`` columns.

    ``image_level`` holds ``image_id``, the ``_scores`` list and the aligned
    ``_gt_idx`` / ``_iou`` / ``_dup`` lists. Each detection keeps its own
    row's ``carry`` values (its class, its threshold) — never another row's of
    the same image.
    """
    # A row whose ground-truth column was null gets a null payload, because
    # `correspond` declines to answer when an operand is null. Metrics has
    # already decided what a null GT column means -- `n_gts` reads it as zero
    # -- so the payload follows: no pairings, one per prediction. Without this
    # the predictions on a ground-truth-free image would vanish instead of
    # counting as false positives, which is exactly what they are.
    scores = pl.col("_scores")
    unpaired = pl.col("_gt_idx").is_null()
    payload = image_level.select(
        COL_IMAGE_ID,
        *carry,
        "_scores",
        _gt_idx=pl.when(unpaired)
        .then(scores.list.eval(pl.lit(None, dtype=pl.UInt32)))
        .otherwise(pl.col("_gt_idx")),
        _iou=pl.when(unpaired)
        .then(scores.list.eval(pl.lit(0.0)))
        .otherwise(pl.col("_iou")),
        _det_ord=pl.int_ranges(0, scores.list.len()),
        _dup=(
            pl.when(unpaired)
            .then(scores.list.eval(pl.lit(False)))
            .otherwise(pl.col("_dup"))
            if ignore_duplicates
            else scores.list.eval(pl.lit(False))
        ),
    )
    # One explode, no join: the payload is positionally aligned with the
    # scores, so the ordinal is the position.
    return (
        payload.explode(
            "_scores", "_gt_idx", "_iou", "_det_ord", "_dup", empty_as_null=True
        )
        # An image with no predictions explodes to one null row: not a detection.
        .filter(pl.col("_scores").is_not_null())
        # A repeated hit on a claimed target is dropped, not scored as an FP.
        .filter(~pl.col("_dup").fill_null(False))
        .select(
            pl.col(COL_IMAGE_ID),
            pl.col(COL_CLASS_ID),
            pl.col("_scores").alias(COL_SCORE),
            pl.col("_gt_idx").is_not_null().alias(COL_IS_TP),
            pl.col("_gt_idx").cast(pl.UInt32).alias(COL_GT_IDX),
            pl.col("_iou").fill_null(0.0).alias(COL_IOU),
            pl.col("_det_ord").cast(pl.UInt32).alias(COL_DET_IDX),
            *(pl.col(c) for c in carry if c != COL_CLASS_ID),
        )
    )
