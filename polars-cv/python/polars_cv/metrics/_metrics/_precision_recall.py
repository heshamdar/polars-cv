"""Precision-Recall metrics: PR curve, AP, mAP, P/R/F1 at threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from .._auc import trapz_auc
from .._result import MetricResult
from .._types import (
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    DEFAULT_CLASS,
    DetectionTable,
)


@dataclass(frozen=True)
class PrecisionRecallResult(MetricResult):
    """Precision-Recall curve result.

    Attributes:
        curve: DataFrame with ``score``, ``precision``, ``recall``,
            ``cum_tp``, ``cum_fp``.
        total_gts: Total ground-truth count for this class.
        class_id: Class this curve was computed for.
        detection_table: The underlying ``DetectionTable`` for bootstrap.
    """

    total_gts: int = 0
    class_id: str = DEFAULT_CLASS
    detection_table: DetectionTable | None = None

    def auc(  # type: ignore[override]
        self,
        *,
        method: Literal["all_points", "11_point", "trapezoidal"] = "all_points",
    ) -> float:
        """Compute Average Precision (AUC of the PR curve).

        Args:
            method: Computation method.
                ``"all_points"`` (default) applies the standard monotonically
                decreasing precision envelope before trapezoidal integration
                (matches COCO / scikit-learn AP).
                ``"11_point"`` uses the Pascal VOC 11-point method.
                ``"trapezoidal"`` computes raw trapezoidal AUC without the
                monotone-envelope correction. The global envelope is not
                applied, but points sharing one recall value (a run of false
                positives leaves recall unchanged) still collapse to the
                highest precision among them — those points span zero width,
                so the only question they pose is which precision the
                trapezoid leaving them uses, and "an arbitrary one" is not an
                answer.

        Returns:
            Average Precision value.
        """
        if method == "all_points":
            return _all_points_ap(self.curve)
        if method == "11_point":
            return _eleven_point_ap(self.curve)
        if method == "trapezoidal":
            return super().auc(x_col="recall", y_col="precision")
        raise ValueError(
            f"Unknown method {method!r}. Expected 'all_points', "
            f"'11_point', or 'trapezoidal'."
        )

    def precision_at(self, threshold: float) -> float:
        """Precision at a given score threshold.

        Args:
            threshold: Score threshold.

        Returns:
            Precision value.
        """
        filtered = self.curve.filter(pl.col("score") >= threshold)
        if filtered.height == 0:
            return 1.0
        return float(filtered.select(pl.col("precision").last()).item())

    def recall_at(self, threshold: float) -> float:
        """Recall at a given score threshold.

        Args:
            threshold: Score threshold.

        Returns:
            Recall value.
        """
        filtered = self.curve.filter(pl.col("score") >= threshold)
        if filtered.height == 0:
            return 0.0
        return float(filtered.select(pl.col("recall").last()).item())

    # ------------------------------------------------------------------
    # Bootstrap hooks
    # ------------------------------------------------------------------

    def _get_detection_table(self) -> DetectionTable:
        """Return the underlying DetectionTable."""
        if self.detection_table is None:
            raise ValueError("bootstrap_ci requires detection_table to be set.")
        return self.detection_table

    def _bootstrap_grouped(
        self,
        boot_table: DetectionTable,
        method: str,
        kwargs: dict[str, object],
    ) -> tuple[pl.LazyFrame, float]:
        """Grouped PR metric over the resampled table, keyed by ``bootstrap_id``.

        Reads the shared lazy authorities (:func:`all_points_ap_by_group`,
        :func:`threshold_counts_by_group`) so a replicate uses the exact
        estimator the scalar point estimate does. Only the vectorizable PR
        metrics are supported; anything else raises rather than silently
        degrading.
        """
        det = boot_table.detections
        meta = boot_table.image_metadata

        if method == "auc":
            ap_method = kwargs.get("method", "all_points")
            if ap_method != "all_points":
                raise ValueError(
                    "bootstrap_ci vectorizes auc only for method='all_points'; "
                    f"got {ap_method!r}."
                )
            gts = meta.group_by("bootstrap_id").agg(
                total_gts=pl.col(COL_N_GTS).sum().cast(pl.Float64)
            )
            expanded = (
                det.select("bootstrap_id", COL_SCORE, COL_IS_TP)
                .drop_nulls(COL_SCORE)
                .join(gts, on="bootstrap_id", how="left")
            )
            grouped = all_points_ap_by_group(expanded, group_col="bootstrap_id").rename(
                {"ap": "auc"}
            )
            return grouped, 0.0

        if method in ("precision_at", "recall_at"):
            threshold = float(kwargs["threshold"])  # type: ignore[arg-type]
            counts = threshold_counts_by_group(
                det, meta, group_col="bootstrap_id", threshold=threshold
            )
            denom = pl.col("tp") + pl.col("fp")
            if method == "precision_at":
                value = (
                    pl.when(denom == 0)
                    .then(pl.lit(1.0))
                    .otherwise(pl.col("tp") / denom)
                )
                empty_value = 1.0
            else:
                value = (
                    pl.when(pl.col("total_gts") == 0)
                    .then(pl.lit(0.0))
                    .otherwise(pl.col("tp") / pl.col("total_gts"))
                )
                empty_value = 0.0
            grouped = counts.select("bootstrap_id", auc=value.cast(pl.Float64))
            return grouped, empty_value

        raise ValueError(
            f"metric {method!r} has no vectorized bootstrap form on "
            f"{type(self).__name__}."
        )


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def precision_recall_curve(
    table: DetectionTable,
    *,
    class_id: str | None = None,
) -> PrecisionRecallResult:
    """Compute a precision-recall curve from a DetectionTable.

    Detections are sorted by confidence score (descending). At each rank, cumulative
    TP/FP are computed and precision/recall derived. All computation uses Polars
    lazy expressions.

    Args:
        table: Canonical detection table.
        class_id: Restrict to a specific class. ``None`` uses all detections.

    Returns:
        ``PrecisionRecallResult`` with the PR curve.
    """
    if class_id is not None:
        table = table.filter_class(class_id)
    resolved_class = class_id or DEFAULT_CLASS

    det_df, meta_df = table.collect(engine="streaming")

    total_gts = int(meta_df.select(pl.col(COL_N_GTS).sum()).item())

    if det_df.height == 0 or total_gts == 0:
        empty_curve = pl.DataFrame(
            schema={
                "score": pl.Float64,
                "precision": pl.Float64,
                "recall": pl.Float64,
                "cum_tp": pl.Int64,
                "cum_fp": pl.Int64,
            }
        )
        return PrecisionRecallResult(
            curve=empty_curve,
            total_gts=total_gts,
            class_id=resolved_class,
            detection_table=table,
        )

    curve = (
        det_df.lazy()
        .sort(COL_SCORE, descending=True)
        .with_columns(
            cum_tp=pl.col(COL_IS_TP).cast(pl.Int64).cum_sum(),
            cum_fp=(~pl.col(COL_IS_TP)).cast(pl.Int64).cum_sum(),
        )
        .with_columns(
            precision=pl.col("cum_tp") / (pl.col("cum_tp") + pl.col("cum_fp")),
            recall=pl.col("cum_tp") / pl.lit(float(total_gts)),
        )
        .select(
            pl.col(COL_SCORE).alias("score"),
            "precision",
            "recall",
            "cum_tp",
            "cum_fp",
        )
        .collect(engine="streaming")
    )

    return PrecisionRecallResult(
        curve=curve,
        total_gts=total_gts,
        class_id=resolved_class,
        detection_table=table,
    )


def average_precision(
    table: DetectionTable,
    *,
    class_id: str | None = None,
    interpolation: Literal["all_points", "11_point"] = "all_points",
) -> float:
    """Compute Average Precision for a single class.

    Args:
        table: Canonical detection table.
        class_id: Restrict to a specific class.
        interpolation: ``"all_points"`` (trapezoidal) or ``"11_point"`` (VOC).

    Returns:
        AP value in [0, 1].
    """
    pr = precision_recall_curve(table, class_id=class_id)
    return pr.auc(method=interpolation)


def mean_average_precision(
    table: DetectionTable,
    *,
    iou_thresholds: list[float] | None = None,
    interpolation: Literal["all_points", "11_point"] = "all_points",
) -> float:
    """Compute Mean Average Precision across classes and IoU thresholds.

    If ``iou_thresholds`` is provided, the stored ``iou`` column is re-thresholded
    at each level to recompute ``is_tp`` -- **no re-matching is needed**.

    Args:
        table: Canonical detection table.
        iou_thresholds: IoU thresholds to average over. Defaults to
            ``[0.5]`` (Pascal VOC). Use ``[0.5, 0.55, ..., 0.95]`` for COCO.
        interpolation: AP interpolation method.

    Returns:
        mAP value in [0, 1].
    """
    thresholds = iou_thresholds or [0.5]
    class_ids = table.class_ids()

    ap_values: list[float] = []
    for iou_thresh in thresholds:
        rethresholded = table.at_iou_threshold(iou_thresh)
        for cid in class_ids:
            ap_values.append(
                average_precision(
                    rethresholded, class_id=cid, interpolation=interpolation
                )
            )

    if not ap_values:
        return 0.0
    return float(pl.Series("ap", ap_values).mean())  # type: ignore[arg-type]


def precision_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
) -> float:
    """Compute precision at a given score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold.
        class_id: Optional class filter.

    Returns:
        Precision value.
    """
    if class_id is not None:
        table = table.filter_class(class_id)

    counts = (
        table.detections.filter(pl.col(COL_SCORE) >= threshold)
        .select(
            tp=pl.col(COL_IS_TP).sum().cast(pl.Int64),
            fp=(~pl.col(COL_IS_TP)).sum().cast(pl.Int64),
        )
        .collect(engine="streaming")
    )
    tp = int(counts["tp"].item())
    fp = int(counts["fp"].item())
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)


def recall_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
) -> float:
    """Compute recall at a given score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold.
        class_id: Optional class filter.

    Returns:
        Recall value.
    """
    if class_id is not None:
        table = table.filter_class(class_id)

    tp_count = (
        table.detections.filter((pl.col(COL_SCORE) >= threshold) & pl.col(COL_IS_TP))
        .select(pl.len().alias("count"))
        .collect(engine="streaming")
        .item()
    )
    total_gts = int(
        table.image_metadata.select(pl.col(COL_N_GTS).sum())
        .collect(engine="streaming")
        .item()
    )
    if total_gts == 0:
        return 0.0
    return int(tp_count) / total_gts


def f1_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
) -> float:
    """Compute F1 score at a given score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold.
        class_id: Optional class filter.

    Returns:
        F1 value in [0, 1].
    """
    p = precision_at_threshold(table, threshold, class_id=class_id)
    r = recall_at_threshold(table, threshold, class_id=class_id)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _all_points_ap(curve: pl.DataFrame) -> float:
    """Compute AP using monotone-envelope interpolation.

    Applies the standard monotonically decreasing precision envelope
    (right-to-left cumulative maximum) before trapezoidal integration.
    This matches the COCO and scikit-learn AP definitions.

    Args:
        curve: PR curve DataFrame with ``recall`` and ``precision``.

    Returns:
        All-points interpolated AP with monotone envelope.
    """
    if curve.height == 0:
        return 0.0

    recall = curve["recall"].cast(pl.Float64)
    precision = curve["precision"].cast(pl.Float64)

    # Monotone decreasing envelope: reverse, cum_max, reverse back
    envelope = precision.reverse().cum_max().reverse()
    # Anchor at recall = 0 so the leftmost block (recall[0] × envelope[0])
    # is included — matches COCO / scikit-learn Σ (Rₙ − Rₙ₋₁) · Pₙ with R₀ = 0.
    recall = pl.concat([pl.Series([0.0]), recall])
    envelope = pl.concat([pl.Series([envelope[0]]), envelope])
    return float(trapz_auc(recall, envelope))


def _eleven_point_ap(curve: pl.DataFrame) -> float:
    """Compute AP using Pascal VOC 11-point interpolation.

    Cross-joins the 11 recall thresholds with the PR curve, filters to
    recall >= threshold, and takes max precision per threshold -- all as
    a single Polars lazy plan. Thresholds beyond the curve's maximum
    recall have no qualifying point and contribute a precision of 0; the
    average is always taken over all 11 thresholds.

    Args:
        curve: PR curve DataFrame with ``recall`` and ``precision``.

    Returns:
        11-point interpolated AP.
    """
    if curve.height == 0:
        return 0.0

    thresholds = pl.DataFrame({"t": [i / 10.0 for i in range(11)]})
    per_threshold = (
        thresholds.lazy()
        .join(
            curve.lazy().select(
                pl.col("recall").cast(pl.Float64),
                pl.col("precision").cast(pl.Float64),
            ),
            how="cross",
        )
        .filter(pl.col("recall") >= pl.col("t"))
        .group_by("t")
        .agg(max_p=pl.col("precision").max())
    )
    result = (
        thresholds.lazy()
        .join(per_threshold, on="t", how="left")
        .select(pl.col("max_p").fill_null(0.0).mean())
        .collect(engine="streaming")
    )
    return float(result.item())


# ---------------------------------------------------------------------------
# Grouped (lazy) PR estimators — the single authority for vectorized bootstrap
# ---------------------------------------------------------------------------


def all_points_ap_by_group(
    expanded: pl.LazyFrame,
    *,
    group_col: str,
) -> pl.LazyFrame:
    """All-points AP per group — the lazy authority shared by every bootstrap.

    ``expanded`` carries ``[group_col, score, is_tp, total_gts]`` (one row per
    detection, ``total_gts`` broadcast per group). The estimator is identical to
    the scalar :func:`_all_points_ap`: sort by score within group, cumulative
    TP/FP, the monotone decreasing precision envelope, then trapezoidal
    integration anchored at recall = 0. Keeping the sort as the last
    row-reordering step (nothing joins between it and the windowed ops) makes the
    curve stable across thread counts.

    Args:
        expanded: Per-detection frame with ``group_col``, ``score``, ``is_tp``
            and per-group ``total_gts``.
        group_col: The grouping column (e.g. ``bootstrap_id``).

    Returns:
        ``LazyFrame`` with ``[group_col, ap]``.
    """
    pr = (
        expanded.sort(group_col, COL_SCORE, descending=[False, True])
        .with_columns(
            cum_tp=pl.col(COL_IS_TP).cast(pl.Int64).cum_sum().over(group_col),
            cum_fp=(~pl.col(COL_IS_TP)).cast(pl.Int64).cum_sum().over(group_col),
        )
        .with_columns(
            precision=pl.col("cum_tp")
            / (pl.col("cum_tp") + pl.col("cum_fp")).cast(pl.Float64),
            recall=pl.col("cum_tp").cast(pl.Float64) / pl.col("total_gts"),
        )
        .with_columns(
            precision=pl.col("precision").reverse().cum_max().reverse().over(group_col),
        )
    )
    return (
        pr.with_columns(
            d_recall=(
                pl.col("recall") - pl.col("recall").shift(1).over(group_col)
            ).fill_null(pl.col("recall")),
            avg_precision=(
                (pl.col("precision") + pl.col("precision").shift(1).over(group_col))
                / 2.0
            ).fill_null(pl.col("precision")),
        )
        .with_columns(slice_area=pl.col("d_recall") * pl.col("avg_precision"))
        .group_by(group_col)
        .agg(ap=pl.col("slice_area").sum())
    )


def threshold_counts_by_group(
    det: pl.LazyFrame,
    meta: pl.LazyFrame,
    *,
    group_col: str,
    threshold: float,
) -> pl.LazyFrame:
    """Per-group ``tp``/``fp`` at a score threshold plus ``total_gts``.

    The lazy, grouped counterpart of :func:`precision_at_threshold` /
    :func:`recall_at_threshold`: one row per group (every group in ``meta``,
    even those with no detection at or above ``threshold`` — those get
    ``tp = fp = 0``). Callers derive precision (``1.0`` when ``tp + fp == 0``)
    and recall (``0.0`` when ``total_gts == 0``) from the counts, matching the
    scalar conventions.

    Args:
        det: Per-detection frame carrying ``group_col``, ``score``, ``is_tp``.
        meta: Per-(image) metadata carrying ``group_col`` and ``n_gts``.
        group_col: The grouping column (e.g. ``bootstrap_id``).
        threshold: Score threshold — detections with ``score >= threshold`` count.

    Returns:
        ``LazyFrame`` with ``[group_col, tp, fp, total_gts]``.
    """
    counts = (
        det.filter(pl.col(COL_SCORE) >= threshold)
        .group_by(group_col)
        .agg(
            tp=pl.col(COL_IS_TP).sum().cast(pl.Int64),
            fp=(~pl.col(COL_IS_TP)).sum().cast(pl.Int64),
        )
    )
    gts = meta.group_by(group_col).agg(total_gts=pl.col(COL_N_GTS).sum().cast(pl.Int64))
    return gts.join(counts, on=group_col, how="left").with_columns(
        pl.col("tp").fill_null(0),
        pl.col("fp").fill_null(0),
    )
