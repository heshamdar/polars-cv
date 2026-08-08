"""Precision-Recall metrics: PR curve, AP, mAP, P/R/F1 at threshold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
        method: Literal["all_points", "11_point", "trapezoidal"] = "all_points",
    ) -> float:
        """Compute Average Precision (AUC of the PR curve).

        Args:
            method: Computation method.
                ``"all_points"`` (default) applies the standard monotonically
                decreasing precision envelope, then sums
                ``Σ (Rₙ − Rₙ₋₁) · Pₙ`` (matches COCO / scikit-learn AP).
                ``"11_point"`` uses the Pascal VOC 11-point method.
                ``"trapezoidal"`` computes raw trapezoidal AUC without the
                monotone-envelope correction. The global envelope is not
                applied, but points sharing one recall value (a bucket of pure
                false positives leaves recall unchanged) still collapse to the
                highest precision among them — those points span zero width,
                so the only question they pose is which precision the
                trapezoid leaving them uses, and "an arbitrary one" is not an
                answer.

        All three read a curve carrying one point per distinct score, so none
        of them depends on the order of the frame the detections came in.

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

    def _reconstruct(self, sampled_ids: list[str]) -> PrecisionRecallResult:
        """Rebuild a PrecisionRecallResult from bootstrap-sampled image IDs."""
        table = self._get_detection_table()

        sampled_ids_lf = pl.DataFrame({COL_IMAGE_ID: sampled_ids}).lazy()
        sampled_det = sampled_ids_lf.join(table.detections, on=COL_IMAGE_ID, how="left")
        sampled_meta = sampled_ids_lf.join(
            table.image_metadata, on=COL_IMAGE_ID, how="left"
        )
        sampled_table = DetectionTable.from_matched(sampled_det, sampled_meta)
        return precision_recall_curve(sampled_table, class_id=self.class_id)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def precision_recall_curve(
    table: DetectionTable,
    *,
    class_id: str | None = None,
) -> PrecisionRecallResult:
    """Compute a precision-recall curve from a DetectionTable.

    Detections are bucketed by confidence score and the buckets accumulated in
    descending score order, giving **one curve point per distinct score**.
    Precision and recall are derived from the running totals. All computation
    uses Polars lazy expressions.

    One point per *score*, not per detection, is what makes the curve — and
    therefore every AP computed from it — a function of the data alone. A row
    per detection puts tied detections in some order, and `sort` defaults to
    ``maintain_order=False``, so the interleaving of the TPs and FPs inside a
    tie run came from how the caller's frame happened to be laid out: the same
    detections in a different row order produced a different AP, by as much as
    0.53 vs 0.65 on a frame with heavy ties. A threshold cannot admit one
    detection of a tied group and reject another, so the intermediate points
    inside a run were not operating points the detector can be set to either.
    ``froc_curve`` has always bucketed this way; this is the PR curve catching
    up, and it matches scikit-learn, which likewise reports one point per
    distinct threshold.

    Args:
        table: Canonical detection table.
        class_id: Restrict to a specific class. ``None`` uses all detections.

    Returns:
        ``PrecisionRecallResult`` with the PR curve, one row per distinct
        score, ordered by descending score (ascending recall).
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
        # Bucket first, accumulate second. Grouping on the score is what makes
        # the result independent of the input frame's row order: every
        # detection sharing a score lands in one bucket, so there is no
        # intra-tie ordering left for the cumulative sums to depend on. The
        # sort that follows is then a total order (scores are unique after the
        # group_by), which `maintain_order=False` cannot disturb.
        .group_by(COL_SCORE)
        .agg(
            tp_count=pl.col(COL_IS_TP).sum().cast(pl.Int64),
            fp_count=(~pl.col(COL_IS_TP)).sum().cast(pl.Int64),
        )
        .sort(COL_SCORE, descending=True)
        .with_columns(
            cum_tp=pl.col("tp_count").cum_sum(),
            cum_fp=pl.col("fp_count").cum_sum(),
        )
        .with_columns(
            precision=pl.col("cum_tp")
            / (pl.col("cum_tp") + pl.col("cum_fp")).cast(pl.Float64),
            recall=pl.col("cum_tp").cast(pl.Float64) / pl.lit(float(total_gts)),
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
    (right-to-left cumulative maximum), then sums the rectangles
    ``Σ (Rₙ − Rₙ₋₁) · Pₙ`` with ``R₀ = 0`` — the COCO / scikit-learn
    definition, spelled directly.

    The rectangle sum is written out rather than delegated to
    :func:`trapz_auc`, which is what this used to call. The two agree only
    while the envelope is flat across every recall step, and that held only
    because the curve carried one row per detection: a step then came from a
    single TP, which raises raw precision, which forces the right-to-left
    cumulative maximum to be equal on both sides of it. Now that the curve
    carries one row per *score*, a bucket can add true and false positives
    together and lower the envelope across a step of non-zero width — where
    the trapezoid would credit the average of the two precisions and COCO
    credits the right-hand one. The definition the docstring claims is the one
    that runs.

    Args:
        curve: PR curve DataFrame with ``recall`` and ``precision``, ordered
            by ascending recall.

    Returns:
        All-points interpolated AP with monotone envelope.
    """
    if curve.height == 0:
        return 0.0

    recall = curve["recall"].cast(pl.Float64)
    precision = curve["precision"].cast(pl.Float64)

    # Monotone decreasing envelope: reverse, cum_max, reverse back
    envelope = precision.reverse().cum_max().reverse()
    # Widths against the previous recall, the first measured from R₀ = 0 so
    # the leftmost block is included.
    widths = recall.diff()
    widths[0] = recall[0]
    return float((widths * envelope).sum())


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
