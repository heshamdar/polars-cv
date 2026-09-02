"""LROC metric: localization sensitivity vs FP fraction (lazy, expression-valued).

The eager ``lroc_curve``/``LROCResult`` API was removed in favour of the
expression-valued functions here: :func:`lroc_curve_lazy`, :func:`lroc_auc`,
:func:`lroc_sensitivity_at_fpf`. Confidence intervals come from
``bootstrap_lroc_auc`` in :mod:`polars_cv.metrics._bootstrap`.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from .._auc import CorrectionMethod
from .._auc_expr import (
    collapse_curve,
    mann_whitney_auc_expr,
    partial_auc_expr,
    trapz_auc_expr,
)
from .._result import MetricResult
from .._types import (
    COL_GT_LABEL,
    COL_IS_TP,
    COL_SCORE,
    COL_WEIGHT,
    DetectionTable,
)

# Internal dummy group key used to run the group-aware curve/AUC path with a
# single implicit group. Dropped from every public result.
_DUMMY_GROUP = "_lroc_grp"


def _normalize_group_by(group_by: str | list[str] | None) -> list[str]:
    """Normalize the ``group_by`` argument to a list of column names."""
    if group_by is None:
        return []
    if isinstance(group_by, str):
        return [group_by]
    return list(group_by)


def _scored_per_image_lazy(
    table: DetectionTable,
    variant: Literal["best_tp", "top_scoring"],
) -> pl.LazyFrame:
    """Per-image table with the variant's ``max_score`` / ``top_is_tp`` columns.

    The single authority for LROC's per-image commitment logic.
    """
    per_image_lf = table.to_per_image()
    if "detections" in per_image_lf.collect_schema().names():
        per_image_lf = per_image_lf.with_columns(
            _best_tp_score=pl.col("detections")
            .list.eval(
                pl.when(pl.element().struct.field(COL_IS_TP))
                .then(pl.element().struct.field(COL_SCORE))
                .otherwise(None)
            )
            .list.max(),
            _max_det_score=pl.col("detections")
            .list.eval(pl.element().struct.field(COL_SCORE))
            .list.max(),
            _top_det_is_tp=pl.col("detections")
            .list.eval(pl.element().struct.field(COL_IS_TP))
            .list.first(),
        )

        if variant == "best_tp":
            per_image_lf = per_image_lf.with_columns(
                max_score=pl.when(pl.col(COL_GT_LABEL))
                .then(pl.col("_best_tp_score"))
                .otherwise(pl.col("_max_det_score")),
                top_is_tp=pl.when(pl.col(COL_GT_LABEL))
                .then(pl.col("_best_tp_score").is_not_null())
                .otherwise(pl.lit(False)),
            )
        else:
            # top_scoring: commit to the highest-scoring detection
            per_image_lf = per_image_lf.with_columns(
                max_score=pl.col("_max_det_score"),
                top_is_tp=pl.when(pl.col(COL_GT_LABEL))
                .then(pl.col("_top_det_is_tp").fill_null(False))
                .otherwise(pl.lit(False)),
            )

    return per_image_lf


def lroc_curve_lazy(
    table: DetectionTable,
    *,
    variant: Literal["best_tp", "top_scoring"] = "best_tp",
    group_by: str | list[str] | None = None,
) -> pl.LazyFrame:
    """Build the LROC curve as a lazy, group-aware frame.

    Cumulative sums run ``.over`` the group and the weighted denominators are
    per-group aggregations.

    Args:
        table: Canonical detection table produced by a matcher.
        variant: ``"best_tp"`` or ``"top_scoring"``.
        group_by: Optional per-image column(s) partitioning the curve (e.g.
            ``group_id``).

    Returns:
        A ``LazyFrame`` with ``[*group_by, threshold, fpf, sensitivity]``.
    """
    if variant not in ("best_tp", "top_scoring"):
        raise ValueError(
            f"Invalid variant {variant!r}. Expected 'best_tp' or 'top_scoring'."
        )
    group_keys = _normalize_group_by(group_by)
    per_image = _scored_per_image_lazy(table, variant).with_columns(
        pl.lit(0, dtype=pl.Int32).alias(_DUMMY_GROUP)
    )
    keys = [_DUMMY_GROUP, *group_keys]
    curve = _build_lroc_curve_grouped(per_image, keys)
    return curve.drop(_DUMMY_GROUP)


def _build_lroc_curve_grouped(
    per_image: pl.LazyFrame,
    keys: list[str],
) -> pl.LazyFrame:
    """Group-aware LROC curve over ``keys`` (non-empty; carries the dummy)."""
    group_stats = (
        per_image.group_by(keys)
        .agg(
            _tw_pos=(pl.col(COL_GT_LABEL).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum(),
            _tw_neg=(
                (~pl.col(COL_GT_LABEL)).cast(pl.Float64) * pl.col(COL_WEIGHT)
            ).sum(),
            _n_pos=pl.col(COL_GT_LABEL).sum().cast(pl.Float64),
            _n_neg=(~pl.col(COL_GT_LABEL)).sum().cast(pl.Float64),
        )
        .with_columns(
            _n_pos_f=pl.max_horizontal(pl.col("_n_pos"), pl.lit(1.0)),
            _n_neg_f=pl.max_horizontal(pl.col("_n_neg"), pl.lit(1.0)),
        )
    )

    scored = per_image.filter(pl.col("max_score").is_not_null())
    bucketed = (
        scored.group_by(*keys, "max_score")
        .agg(
            pos_detected=(pl.col(COL_GT_LABEL) & pl.col("top_is_tp")).sum(),
            neg_detected=(~pl.col(COL_GT_LABEL)).sum(),
            weighted_pos_detected=(
                (pl.col(COL_GT_LABEL) & pl.col("top_is_tp")).cast(pl.Float64)
                * pl.col(COL_WEIGHT)
            ).sum(),
            weighted_neg_detected=(
                (~pl.col(COL_GT_LABEL)).cast(pl.Float64) * pl.col(COL_WEIGHT)
            ).sum(),
        )
        .rename({"max_score": "threshold"})
        .sort(*keys, "threshold", descending=[False] * len(keys) + [True])
        .with_columns(
            cum_pos_detected=pl.col("pos_detected").cum_sum().over(keys),
            cum_neg_detected=pl.col("neg_detected").cum_sum().over(keys),
            cum_weighted_pos_detected=pl.col("weighted_pos_detected")
            .cum_sum()
            .over(keys),
            cum_weighted_neg_detected=pl.col("weighted_neg_detected")
            .cum_sum()
            .over(keys),
        )
        .join(group_stats, on=keys, how="left")
        .with_columns(
            sensitivity=pl.when(pl.col("_tw_pos") > 0.0)
            .then(pl.col("cum_weighted_pos_detected") / pl.col("_tw_pos"))
            .otherwise(pl.col("cum_pos_detected") / pl.col("_n_pos_f")),
            fpf=pl.when(pl.col("_tw_neg") > 0.0)
            .then(pl.col("cum_weighted_neg_detected") / pl.col("_tw_neg"))
            .otherwise(pl.col("cum_neg_detected") / pl.col("_n_neg_f")),
        )
        .select(*keys, "threshold", "fpf", "sensitivity")
    )

    # Origin (threshold=+inf, fpf=0, sensitivity=0) and lower-right endpoint
    # (threshold=-inf, fpf=1, sensitivity=max achievable), one of each per group.
    group_ids = group_stats.select(keys)
    origin = group_ids.select(
        *keys,
        threshold=pl.lit(float("inf")),
        fpf=pl.lit(0.0),
        sensitivity=pl.lit(0.0),
    )
    max_sens = bucketed.group_by(keys).agg(_max_sens=pl.col("sensitivity").max())
    lower_right = (
        group_ids.join(max_sens, on=keys, how="left")
        .with_columns(pl.col("_max_sens").fill_null(0.0))
        .select(
            *keys,
            threshold=pl.lit(float("-inf")),
            fpf=pl.lit(1.0),
            sensitivity=pl.col("_max_sens"),
        )
    )

    return pl.concat([bucketed, origin, lower_right], how="vertical").sort(
        *keys, "threshold", descending=[False] * len(keys) + [True]
    )


def lroc_auc(
    table: DetectionTable,
    *,
    variant: Literal["best_tp", "top_scoring"] = "best_tp",
    method: Literal["trapezoidal", "mann_whitney"] = "trapezoidal",
    fpf_range: tuple[float, float] | None = None,
    correction: CorrectionMethod = None,
    level: Literal["detection", "image"] = "image",
    group_by: str | list[str] | None = None,
) -> pl.LazyFrame:
    """Compute LROC AUC as a lazy, group-aware frame — one row per group.

    The single authority for the LROC integral. A scalar is
    ``lroc_auc(table).collect().item()``.

    Args:
        table: Canonical detection table.
        variant: ``"best_tp"`` or ``"top_scoring"`` (curve/image-MW only).
        method: ``"trapezoidal"`` integrates the curve; ``"mann_whitney"`` is
            the rank statistic.
        fpf_range: Optional ``(lo, hi)`` partial-AUC range (trapezoidal only).
        correction: Partial-AUC correction (trapezoidal only).
        level: Mann-Whitney granularity — ``"image"`` (default) or
            ``"detection"``.
        group_by: Optional grouping column(s). ``None`` yields a single row.

    Returns:
        A ``LazyFrame`` with ``[*group_by, auc]``.
    """
    group_keys = _normalize_group_by(group_by)

    if method == "mann_whitney":
        if fpf_range is not None or correction is not None:
            raise ValueError(
                "fpf_range and correction are not supported with "
                "method='mann_whitney'. Mann-Whitney computes a global rank "
                "statistic, not a curve integral."
            )
        if level == "detection":
            det = table.detections.with_columns(
                pl.lit(0, dtype=pl.Int32).alias(_DUMMY_GROUP)
            )
            return (
                det.group_by([_DUMMY_GROUP, *group_keys])
                .agg(auc=mann_whitney_auc_expr(score=COL_SCORE, label=COL_IS_TP))
                .drop(_DUMMY_GROUP)
            )
        if level == "image":
            per_image = _scored_per_image_lazy(table, variant).with_columns(
                pl.lit(0, dtype=pl.Int32).alias(_DUMMY_GROUP)
            )
            # Image score: positives commit their best-TP score (0 if none),
            # negatives their max detection score (0 if none).
            score_expr = (
                pl.when(pl.col(COL_GT_LABEL))
                .then(
                    pl.when(pl.col("top_is_tp"))
                    .then(pl.col("max_score"))
                    .otherwise(0.0)
                )
                .otherwise(pl.col("max_score"))
                .fill_null(0.0)
            )
            return (
                per_image.group_by([_DUMMY_GROUP, *group_keys])
                .agg(
                    auc=mann_whitney_auc_expr(
                        score=score_expr, label=pl.col(COL_GT_LABEL)
                    )
                )
                .drop(_DUMMY_GROUP)
            )
        raise ValueError(
            f"Unsupported level {level!r}. Expected 'detection' or 'image'."
        )

    if method != "trapezoidal":
        raise ValueError(
            f"Unknown method {method!r}. Expected 'trapezoidal' or 'mann_whitney'."
        )

    curve = lroc_curve_lazy(table, variant=variant, group_by=group_by)
    collapsed = collapse_curve(
        curve, x_col="fpf", y_col="sensitivity", group_keys=group_keys
    )
    if fpf_range is None:
        auc_expr = trapz_auc_expr(x="fpf", y="sensitivity", correction=correction)
    else:
        auc_expr = partial_auc_expr(
            x="fpf",
            y="sensitivity",
            lo=fpf_range[0],
            hi=fpf_range[1],
            correction=correction,
        )
    if group_keys:
        return collapsed.group_by(group_keys).agg(auc=auc_expr)
    return collapsed.select(auc=auc_expr)


def lroc_sensitivity_at_fpf(
    table: DetectionTable,
    fpf: float,
    *,
    variant: Literal["best_tp", "top_scoring"] = "best_tp",
) -> float | None:
    """Interpolate LROC sensitivity at a requested false-positive fraction.

    Args:
        table: Canonical detection table.
        fpf: Target false-positive fraction.
        variant: ``"best_tp"`` or ``"top_scoring"``.

    Returns:
        Interpolated sensitivity, or ``None`` when ``fpf`` is outside the
        observed range of the curve.
    """
    curve = lroc_curve_lazy(table, variant=variant).collect(engine="streaming")
    return MetricResult(curve=curve).interpolate(
        x_col="fpf", y_col="sensitivity", at=fpf
    )
