"""Precision-Recall metrics: PR curve, AP, mAP, P/R/F1 at threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from .._result import MetricResult
from .._types import (
    COL_IMAGE_ID,
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
        interpolation: Literal["all_points", "11_point"] = "all_points",
    ) -> float:
        """Compute Average Precision (AUC with monotone-envelope interpolation).

        Applies the standard **monotonically decreasing precision envelope**
        (right-to-left cumulative maximum) before trapezoidal integration,
        matching COCO / scikit-learn AP definitions.

        For the raw trapezoidal area without the envelope, use
        :meth:`raw_auc`.

        Args:
            interpolation: Interpolation method.
                ``"all_points"`` (default) uses the monotone-envelope
                precision before trapezoidal integration.
                ``"11_point"`` uses the Pascal VOC 11-point method.

        Returns:
            Average Precision value.
        """
        if interpolation == "11_point":
            return _eleven_point_ap(self.curve)
        return _all_points_ap(self.curve)

    def raw_auc(self) -> float:
        """Compute raw trapezoidal AUC under the precision-recall curve.

        This does **not** apply the monotonically decreasing precision
        envelope.  For the standard AP that matches COCO / scikit-learn
        definitions, use :meth:`auc` instead.

        Returns:
            Raw area under the PR curve (without envelope interpolation).
        """
        return super().auc(x_col="recall", y_col="precision")

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

    def bootstrap_ci(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int | None = None,
        *,
        metric: str = "auc",
        metric_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Estimate CI via image-level bootstrap.

        Args:
            n_bootstrap: Number of bootstrap iterations.
            confidence: Confidence level in (0, 1).
            seed: Optional RNG seed.
            metric: ``"auc"`` (standard AP with envelope).
            metric_kwargs: Extra kwargs for the metric method.

        Returns:
            ``BootstrapResult`` with percentile confidence interval.
        """
        from .._bootstrap import bootstrap_metric_sequential

        metric_kwargs = metric_kwargs or {}
        if self.detection_table is None:
            raise ValueError("bootstrap_ci requires detection_table to be set.")

        image_ids, strata = self.detection_table.image_ids_and_strata()

        def _metric(sampled_ids: list[str]) -> float:
            sampled_det = (
                pl.DataFrame({COL_IMAGE_ID: sampled_ids})
                .lazy()
                .join(
                    self.detection_table.detections,  # type: ignore[union-attr]
                    on=COL_IMAGE_ID,
                    how="left",
                )
            )
            sampled_meta = (
                pl.DataFrame({COL_IMAGE_ID: sampled_ids})
                .lazy()
                .join(
                    self.detection_table.image_metadata,  # type: ignore[union-attr]
                    on=COL_IMAGE_ID,
                    how="left",
                )
            )
            sampled_table = DetectionTable.from_matched(sampled_det, sampled_meta)
            result = precision_recall_curve(sampled_table, class_id=self.class_id)
            return result.auc(**metric_kwargs)

        point = self.auc(**metric_kwargs)
        return bootstrap_metric_sequential(
            image_ids=image_ids,
            metric_fn=_metric,
            point_estimate=point,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
            strata=strata,
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

    det_df, meta_df = table.collect()

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
    return pr.auc(interpolation=interpolation)


def mean_average_precision(
    table: DetectionTable,
    *,
    iou_thresholds: list[float] | None = None,
    interpolation: Literal["all_points", "11_point"] = "all_points",
) -> float:
    """Compute Mean Average Precision across classes and IoU thresholds.

    If ``iou_thresholds`` is provided, the stored ``iou`` column is re-thresholded
    at each level to recompute ``is_tp`` — **no re-matching is needed**.

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
    return float(np.mean(ap_values))


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

    recall = curve["recall"].cast(pl.Float64).to_numpy()
    precision = curve["precision"].cast(pl.Float64).to_numpy()

    # Monotone decreasing envelope: at each recall level, precision is the
    # maximum precision at any recall >= current recall.
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.trapezoid(precision_envelope, recall))


def _eleven_point_ap(curve: pl.DataFrame) -> float:
    """Compute AP using Pascal VOC 11-point interpolation.

    Args:
        curve: PR curve DataFrame with ``recall`` and ``precision``.

    Returns:
        11-point interpolated AP.
    """
    if curve.height == 0:
        return 0.0

    recall = curve["recall"].cast(pl.Float64).to_numpy()
    precision = curve["precision"].cast(pl.Float64).to_numpy()

    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        mask = recall >= t
        if mask.any():
            ap += float(precision[mask].max())
    return ap / 11.0
