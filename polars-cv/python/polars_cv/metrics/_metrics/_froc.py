"""FROC metric: sensitivity vs false positives per image (lazy, expression-valued).

The eager ``froc_curve``/``FROCResult`` API was removed in favour of the
expression-valued functions here: :func:`froc_curve_lazy` (group-aware curve),
:func:`froc_auc` (one row per group), :func:`froc_sensitivity_at_fp` and
:func:`froc_summary_table`. Confidence intervals come from
``froc_auc_ci_lazy`` in :mod:`polars_cv.metrics._bootstrap`.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .._auc import CorrectionMethod
from .._auc_expr import (
    collapse_curve,
    collapse_scores,
    interpolate_curve_lazy,
    mann_whitney_auc_expr,
    partial_auc_expr,
    trapz_auc_expr,
)
from .._types import (
    COL_CLASS_ID,
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DetectionTable,
)
from .._weights import (
    WeightAgg,
    attach_resolved_weight,
    image_weight_keys,
    resolve_key_weights,
)

# Internal dummy group key used to run the group-aware curve/AUC path with a
# single implicit group. Dropped from every public result.
_DUMMY_GROUP = "_froc_grp"


def _normalize_group_by(group_by: str | list[str] | None) -> list[str]:
    """Normalize the ``group_by`` argument to a list of column names."""
    if group_by is None:
        return []
    if isinstance(group_by, str):
        return [group_by]
    return list(group_by)


def froc_curve_lazy(
    table: DetectionTable,
    *,
    group_by: str | list[str] | None = None,
    thresholds: list[float] | None = None,
    weight_agg: WeightAgg = "first",
) -> pl.LazyFrame:
    """Build the FROC curve as a lazy, group-aware frame.

    Every cumulative sum runs ``.over`` the group and the weighted denominators
    are per-group aggregations rather than eager ``.item()`` Python floats. With
    ``group_by=None`` the whole table is one group; with a grouping column each
    group is computed as if on its filtered sub-table (so
    ``froc_curve_lazy(group_by="class_id")`` per class equals ``froc_curve_lazy``
    on ``table.filter_class(cid)``).

    Args:
        table: Canonical detection table produced by a matcher.
        group_by: Optional column(s) partitioning the curve. May name a column
            on the detections (e.g. ``class_id``) or one only on the metadata
            (e.g. ``group_id``); metadata-only keys are joined onto detections
            by ``image_id``.
        thresholds: Optional explicit score thresholds to keep.
        weight_agg: How to resolve duplicate metadata weights for one
            ``(image[, class])`` key (see :func:`resolve_key_weights`). ``"first"``
            (default) keeps whichever row survives ``unique``; ``"min"`` / ``"max"``
            / ``"mean"`` / ``"sum"`` are order-independent. Supplying consistent
            weights is the caller's responsibility — conflicting weights resolve
            per this policy rather than raising.

    Returns:
        A ``LazyFrame`` with ``[*group_by, threshold, tp, fp, fn, total_gts,
        fp_per_image, sensitivity]`` (no group columns when ``group_by`` is
        ``None``).
    """
    group_keys = _normalize_group_by(group_by)

    det = table.detections.with_columns(pl.lit(0, dtype=pl.Int32).alias(_DUMMY_GROUP))
    meta = table.image_metadata.with_columns(
        pl.lit(0, dtype=pl.Int32).alias(_DUMMY_GROUP)
    )
    keys = [_DUMMY_GROUP, *group_keys]

    curve = _froc_curve_grouped(det, meta, keys, thresholds, weight_agg)
    return curve.drop(_DUMMY_GROUP)


def _froc_curve_grouped(
    det: pl.LazyFrame,
    meta: pl.LazyFrame,
    keys: list[str],
    thresholds: list[float] | None,
    weight_agg: WeightAgg = "first",
) -> pl.LazyFrame:
    """Group-aware FROC curve over ``keys`` (always non-empty; carries dummy)."""
    det_schema = set(det.collect_schema().names())
    meta_schema = set(meta.collect_schema().names())

    # Resolve the metadata weight to one value per (image[, class]) key up front,
    # so the numerator lookup and the summed denominators below read the *same*
    # weight. This keeps the path pure-lazy (no build-time guard): disagreeing
    # duplicate weights resolve per `weight_agg` instead of making the result
    # order-dependent.
    weight_keys = (
        [COL_IMAGE_ID, COL_CLASS_ID] if COL_CLASS_ID in meta_schema else [COL_IMAGE_ID]
    )
    resolved = resolve_key_weights(meta, weight_keys, weight_agg)
    meta_orig = meta
    # Fan the resolved per-key weight back onto every metadata row: the
    # per-target denominator sums n_gts*weight over rows (a shared image owned by
    # two cases contributes each case's n_gts), and every row of a key now carries
    # the same resolved weight.
    meta = meta.drop(COL_WEIGHT).join(resolved, on=weight_keys, how="left")

    # Attach any group key that lives only on the metadata (e.g. group_id) to
    # each detection, keyed by image_id.
    meta_only = [k for k in keys if k not in det_schema and k in meta_schema]
    if meta_only:
        det = det.join(
            meta.select(COL_IMAGE_ID, *meta_only).unique(),
            on=COL_IMAGE_ID,
            how="left",
        )

    # Numerator: the resolved per-key weight attached to each detection.
    det_w = det.join(resolved, on=weight_keys, how="left").with_columns(
        pl.col(COL_WEIGHT).fill_null(1.0)
    )

    # Per-group denominators, each scoped to the group's metadata subset so a
    # grouped curve equals the curve of that group's filtered sub-table.
    gt_stats = meta.group_by(keys).agg(
        total_targets=pl.col(COL_N_GTS).sum().cast(pl.Int64),
        _tw_gts=(pl.col(COL_N_GTS).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum(),
    )
    # FP-per-image counts images: resolve one weight per (group, image_id) from
    # the *original* metadata (not the fanned `meta`, which would sum a key's
    # weight once per duplicate row), then sum over images. `resolve_key_weights`
    # keeps this order-independent when an image spans several metadata rows.
    weight_stats = (
        resolve_key_weights(meta_orig, [*keys, COL_IMAGE_ID], weight_agg)
        .group_by(keys)
        .agg(_weight_sum=pl.col(COL_WEIGHT).sum())
    )
    group_stats = gt_stats.join(weight_stats, on=keys, how="left").with_columns(
        _tw_gts_f=pl.max_horizontal(pl.col("_tw_gts"), pl.lit(1.0)),
        _weight_sum_f=pl.max_horizontal(pl.col("_weight_sum"), pl.lit(1.0)),
    )

    bucketed = (
        det_w.group_by(*keys, COL_SCORE)
        .agg(
            tp_count=pl.col(COL_IS_TP).sum().cast(pl.Int64),
            fp_count=(~pl.col(COL_IS_TP)).sum().cast(pl.Int64),
            weighted_tp=(pl.col(COL_IS_TP).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum(),
            weighted_fp=(
                (~pl.col(COL_IS_TP)).cast(pl.Float64) * pl.col(COL_WEIGHT)
            ).sum(),
        )
        .sort(*keys, COL_SCORE, descending=[False] * len(keys) + [True])
        .with_columns(
            tp=pl.col("tp_count").cum_sum().over(keys),
            fp=pl.col("fp_count").cum_sum().over(keys),
            cum_weighted_tp=pl.col("weighted_tp").cum_sum().over(keys),
            cum_weighted_fp=pl.col("weighted_fp").cum_sum().over(keys),
        )
        .rename({COL_SCORE: "threshold"})
        .join(group_stats, on=keys, how="left")
        .with_columns(
            total_gts=pl.col("total_targets"),
            fn=(pl.col("total_targets") - pl.col("tp")).clip(lower_bound=0),
            sensitivity=pl.col("cum_weighted_tp") / pl.col("_tw_gts_f"),
            fp_per_image=pl.col("cum_weighted_fp") / pl.col("_weight_sum_f"),
        )
        .select(
            *keys,
            "threshold",
            "tp",
            "fp",
            "fn",
            "total_gts",
            "fp_per_image",
            "sensitivity",
        )
    )

    # One origin point (threshold=+inf, everything zero) per group.
    origin = group_stats.select(
        *keys,
        threshold=pl.lit(float("inf")),
        tp=pl.lit(0, dtype=pl.Int64),
        fp=pl.lit(0, dtype=pl.Int64),
        fn=pl.col("total_targets"),
        total_gts=pl.col("total_targets"),
        fp_per_image=pl.lit(0.0),
        sensitivity=pl.lit(0.0),
    )

    curve = pl.concat([origin, bucketed], how="vertical").sort(
        *keys, "threshold", descending=[False] * len(keys) + [True]
    )
    if thresholds is not None:
        threshold_set = set(thresholds)
        curve = curve.filter(pl.col("threshold").is_in(threshold_set))
    return curve


def froc_auc(
    table: DetectionTable,
    *,
    method: Literal["trapezoidal", "mann_whitney"] = "trapezoidal",
    fp_range: tuple[float, float] | None = None,
    correction: CorrectionMethod = None,
    level: Literal["detection", "image"] = "detection",
    group_by: str | list[str] | None = None,
    weight_agg: WeightAgg = "first",
) -> pl.LazyFrame:
    """Compute FROC AUC as a lazy, group-aware frame — one row per group.

    The single authority for the FROC integral: the reusable expressions in
    :mod:`polars_cv.metrics._auc_expr`. A scalar is ``froc_auc(table).collect().item()``.

    Both families are weighted by ``image_metadata.weight``: the trapezoidal path
    through the weighted curve, and Mann-Whitney through a weighted rank-sum
    (``collapse_scores`` + ``mann_whitney_auc_expr``). Unit weights recover the
    unweighted statistic.

    Args:
        table: Canonical detection table.
        method: ``"trapezoidal"`` integrates the curve; ``"mann_whitney"``
            computes a rank statistic.
        fp_range: Optional ``(lo, hi)`` partial-AUC range (trapezoidal only).
        correction: Partial-AUC correction (trapezoidal only).
        level: Mann-Whitney granularity — ``"detection"`` (P(TP > FP), the
            default) or ``"image"`` (P(positive-image score > negative-image)).
        group_by: Optional grouping column(s). ``None`` yields a single row.
        weight_agg: Duplicate-weight resolution policy (see
            :func:`resolve_key_weights`).

    Returns:
        A ``LazyFrame`` with ``[*group_by, auc]``.
    """
    group_keys = _normalize_group_by(group_by)

    if method == "mann_whitney":
        if fp_range is not None or correction is not None:
            raise ValueError(
                "fp_range and correction are not supported with "
                "method='mann_whitney'. Mann-Whitney computes a global rank "
                "statistic, not a curve integral."
            )
        if level == "detection":
            det = table.detections.with_columns(
                pl.lit(0, dtype=pl.Int32).alias(_DUMMY_GROUP)
            )
            meta = table.image_metadata
            if group_keys:
                meta_only = [
                    k for k in group_keys if k not in set(det.collect_schema().names())
                ]
                if meta_only:
                    det = det.join(
                        meta.select(COL_IMAGE_ID, *meta_only).unique(),
                        on=COL_IMAGE_ID,
                        how="left",
                    )
            det = attach_resolved_weight(det, meta, weight_agg=weight_agg)
            keys = [_DUMMY_GROUP, *group_keys]
            bucketed = collapse_scores(
                det,
                score=COL_SCORE,
                label=COL_IS_TP,
                weight=COL_WEIGHT,
                group_keys=keys,
            )
            return (
                bucketed.group_by(keys)
                .agg(auc=mann_whitney_auc_expr())
                .drop(_DUMMY_GROUP)
            )
        if level == "image":
            # Per-image score: positive images commit their best TP score
            # (0 if none), negative images their max detection score (0 if none);
            # label is the image's gt_label. P(pos-image score > neg-image score).
            per_img = table.detections.group_by(COL_IMAGE_ID).agg(
                _max_tp=pl.when(pl.col(COL_IS_TP))
                .then(pl.col(COL_SCORE))
                .otherwise(None)
                .max(),
                _max_any=pl.col(COL_SCORE).max(),
            )
            meta = table.image_metadata
            resolved = resolve_key_weights(meta, image_weight_keys(meta), weight_agg)
            joined = (
                meta.drop(COL_WEIGHT)
                .join(resolved, on=image_weight_keys(meta), how="left")
                .with_columns(
                    pl.col(COL_WEIGHT).fill_null(1.0),
                    pl.lit(0, dtype=pl.Int32).alias(_DUMMY_GROUP),
                )
                .join(per_img, on=COL_IMAGE_ID, how="left")
            )
            score_expr = (
                pl.when(pl.col(COL_GT_LABEL))
                .then(pl.col("_max_tp").fill_null(0.0))
                .otherwise(pl.col("_max_any").fill_null(0.0))
                .fill_null(0.0)
            )
            keys = [_DUMMY_GROUP, *group_keys]
            bucketed = collapse_scores(
                joined,
                score=score_expr,
                label=pl.col(COL_GT_LABEL),
                weight=COL_WEIGHT,
                group_keys=keys,
            )
            return (
                bucketed.group_by(keys)
                .agg(auc=mann_whitney_auc_expr())
                .drop(_DUMMY_GROUP)
            )
        raise ValueError(
            f"Unsupported level {level!r}. Expected 'detection' or 'image'."
        )

    if method != "trapezoidal":
        raise ValueError(
            f"Unknown method {method!r}. Expected 'trapezoidal' or 'mann_whitney'."
        )

    curve = froc_curve_lazy(table, group_by=group_by, weight_agg=weight_agg)
    collapsed = collapse_curve(
        curve, x_col="fp_per_image", y_col="sensitivity", group_keys=group_keys
    )
    if fp_range is None:
        auc_expr = trapz_auc_expr(
            x="fp_per_image", y="sensitivity", correction=correction
        )
    else:
        auc_expr = partial_auc_expr(
            x="fp_per_image",
            y="sensitivity",
            lo=fp_range[0],
            hi=fp_range[1],
            correction=correction,
        )

    if group_keys:
        return collapsed.group_by(group_keys).agg(auc=auc_expr)
    return collapsed.select(auc=auc_expr)


def froc_sensitivity_at_fp(
    table: DetectionTable,
    fp_per_image: float,
    *,
    thresholds: list[float] | None = None,
    weight_agg: WeightAgg = "first",
) -> pl.LazyFrame:
    """Interpolate FROC sensitivity at a requested FP/image rate, lazily.

    Builds the curve via :func:`froc_curve_lazy` and interpolates it with the
    shared lazy geometry (upper-envelope, no extrapolation) — nothing is
    collected here; the caller owns the collect.

    Args:
        table: Canonical detection table.
        fp_per_image: Target false-positive-per-image rate.
        thresholds: Optional explicit score thresholds to keep.
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        A one-row ``LazyFrame`` ``[fp_per_image, sensitivity]``; ``sensitivity``
        is null when ``fp_per_image`` is outside the observed range of the curve.
        A scalar is ``froc_sensitivity_at_fp(t, fp).collect().item()``.
    """
    curve = froc_curve_lazy(table, thresholds=thresholds, weight_agg=weight_agg)
    return interpolate_curve_lazy(
        curve, x_col="fp_per_image", y_col="sensitivity", at=[fp_per_image]
    )


def froc_summary_table(
    table: DetectionTable,
    fp_rates: list[float] | None = None,
    *,
    weight_agg: WeightAgg = "first",
) -> pl.LazyFrame:
    """Sensitivity at standard FP/image operating points, lazily.

    Args:
        table: Canonical detection table.
        fp_rates: Operating points. Defaults to the standard radiology set.
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        A ``LazyFrame`` with ``fp_per_image`` and ``sensitivity`` columns, one row
        per operating point (``sensitivity`` null off the curve). The caller
        collects.
    """
    rates = fp_rates or [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    curve = froc_curve_lazy(table, weight_agg=weight_agg)
    return interpolate_curve_lazy(
        curve, x_col="fp_per_image", y_col="sensitivity", at=rates
    )
