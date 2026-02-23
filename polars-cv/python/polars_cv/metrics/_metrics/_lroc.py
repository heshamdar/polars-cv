"""LROC metric: localization sensitivity vs false-positive fraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from .._auc import mann_whitney_u_auc
from .._result import MetricResult
from .._types import (
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IS_TP,
    COL_SCORE,
    COL_WEIGHT,
    DetectionTable,
)


@dataclass(frozen=True)
class LROCResult(MetricResult):
    """LROC-specific result with sensitivity-at-FPF helpers.

    Two scoring variants are supported (controlled by ``variant``):

    - ``"best_tp"`` (default): For each positive image, the effective
      score is the highest-scoring true-positive (TP) detection.  A
      positive image counts as "correctly localized" if it has at least
      one TP above the current threshold.
    - ``"top_scoring"``: For each positive image, the single
      highest-scoring detection (regardless of TP/FP status) is the
      commitment.  It counts as "correctly localized" only if *that*
      top detection is a TP.  This is closer to the classical
      single-location-commitment LROC formulation (Swensson 1996).

    Attributes:
        curve: DataFrame with ``threshold``, ``fpf``, ``sensitivity``.
        per_image: Per-image top-detection table.
        n_positive: Number of positive images.
        n_negative: Number of negative images.
        iou_threshold: IoU threshold used for matching.
        variant: ``"best_tp"`` or ``"top_scoring"``.
        detection_table: The underlying ``DetectionTable`` for bootstrap.
    """

    per_image: pl.DataFrame = None  # type: ignore[assignment]
    n_positive: int = 0
    n_negative: int = 0
    iou_threshold: float = 0.5
    variant: Literal["best_tp", "top_scoring"] = "best_tp"
    detection_table: DetectionTable | None = None

    def auc(  # type: ignore[override]
        self,
        *,
        fpf_range: tuple[float, float] | None = None,
        normalize: bool = False,
    ) -> float:
        """Compute (partial) AUC under the LROC curve.

        Args:
            fpf_range: Optional ``(lo, hi)`` false-positive fraction range.
            normalize: Whether to normalize the AUC by the range of the x-values.
        Returns:
            Area under the LROC curve.
        """
        return super().auc(
            x_col="fpf", y_col="sensitivity", x_range=fpf_range, normalize=normalize
        )

    def sensitivity_at_fpf(self, fpf: float) -> float:
        """Interpolate sensitivity at a requested false-positive fraction.

        Args:
            fpf: Target false-positive fraction.

        Returns:
            Interpolated sensitivity value.
        """
        return self.interpolate(x_col="fpf", y_col="sensitivity", at=fpf)

    def mann_whitney_auc(
        self,
        *,
        level: str = "image",
    ) -> float:
        """Compute Mann-Whitney U statistic as a non-parametric AUC estimate.

        Args:
            level: Granularity of the comparison.
                ``"detection"`` — P(random TP score > random FP score).
                ``"image"`` — P(positive-image effective score >
                negative-image effective score).  For LROC, the positive-image
                score is ``max_score`` where ``top_is_tp`` is true (else 0);
                the negative-image score is ``max_score`` (else 0).

        Returns:
            Mann-Whitney U AUC in [0, 1].  Returns 0.5 when one group is
            empty.

        Raises:
            ValueError: If ``level`` is not ``"detection"`` or ``"image"``.
        """
        if level == "detection":
            return self._mann_whitney_detection()
        if level == "image":
            return self._mann_whitney_image()
        raise ValueError(
            f"Unsupported level {level!r}. Expected 'detection' or 'image'."
        )

    def _mann_whitney_detection(self) -> float:
        """Detection-level MW-U: P(TP score > FP score)."""
        if self.detection_table is None:
            raise ValueError(
                "mann_whitney_auc(level='detection') requires detection_table."
            )
        det_df = self.detection_table.detections.collect(engine="streaming")
        tp_scores = (
            det_df.filter(pl.col(COL_IS_TP)).select(COL_SCORE).to_numpy().flatten()
        )
        fp_scores = (
            det_df.filter(~pl.col(COL_IS_TP)).select(COL_SCORE).to_numpy().flatten()
        )
        return mann_whitney_u_auc(tp_scores, fp_scores)

    def _mann_whitney_image(self) -> float:
        """Image-level MW-U: P(pos-image score > neg-image score)."""
        if self.per_image is None:
            raise ValueError("mann_whitney_auc(level='image') requires per_image data.")
        pos = self.per_image.filter(pl.col(COL_GT_LABEL))
        neg = self.per_image.filter(~pl.col(COL_GT_LABEL))

        # Positive: score from best TP (0 if no TP)
        pos_scores = (
            pos.select(
                pl.when(pl.col("top_is_tp"))
                .then(pl.col("max_score"))
                .otherwise(0.0)
                .fill_null(0.0)
                .alias("s")
            )
            .to_numpy()
            .flatten()
        )
        # Negative: max detection score (0 if no detections)
        neg_scores = (
            neg.select(pl.col("max_score").fill_null(0.0).alias("s"))
            .to_numpy()
            .flatten()
        )
        return mann_whitney_u_auc(pos_scores, neg_scores)

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
            metric: ``"auc"`` or ``"sensitivity_at_fpf"``.
            metric_kwargs: Extra kwargs passed to the metric method.

        Returns:
            ``BootstrapResult`` with percentile confidence interval.
        """
        from .._bootstrap import bootstrap_metric_sequential

        metric_kwargs = metric_kwargs or {}
        image_ids = [str(v) for v in self.per_image[COL_IMAGE_ID].to_list()]
        strata: dict[str, str] | None = None
        if self.detection_table is not None:
            image_ids, strata = self.detection_table.image_ids_and_strata()

        def _metric(sampled_ids: list[str]) -> float:
            sampled = pl.DataFrame({COL_IMAGE_ID: sampled_ids}).join(
                self.per_image, on=COL_IMAGE_ID, how="left"
            )
            curve = _build_lroc_curve(sampled)
            temp = LROCResult(
                curve=curve,
                per_image=self.per_image,
                n_positive=int(sampled.filter(pl.col(COL_GT_LABEL)).height),
                n_negative=int(sampled.filter(~pl.col(COL_GT_LABEL)).height),
                iou_threshold=self.iou_threshold,
                variant=self.variant,
            )
            if metric == "auc":
                return temp.auc(**metric_kwargs)
            return temp.sensitivity_at_fpf(**metric_kwargs)

        point = (
            self.auc(**metric_kwargs)
            if metric == "auc"
            else self.sensitivity_at_fpf(**metric_kwargs)
        )
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
# Public function
# ---------------------------------------------------------------------------


def lroc_curve(
    table: DetectionTable,
    *,
    variant: Literal["best_tp", "top_scoring"] = "best_tp",
) -> LROCResult:
    """Compute LROC operating points from a DetectionTable.

    LROC requires:
    - ``gt_label`` in image metadata (positive/negative per image).
    - Per-image top-detection reduction from ``DetectionTable.to_per_image()``.

    Two scoring variants are supported:

    - ``"best_tp"`` (default): For each positive image, the effective
      score is the highest-scoring TP detection.  An image counts as
      localized if it has *any* TP above the threshold.
    - ``"top_scoring"``: For each positive image, take the single
      highest-scoring detection (regardless of TP/FP).  The image
      counts as localized only if that top detection is a TP.  This
      matches the classical single-commitment LROC (Swensson 1996).

    For negative images both variants use the maximum detection score.

    Args:
        table: Canonical detection table produced by a matcher.
        variant: ``"best_tp"`` or ``"top_scoring"``.

    Returns:
        ``LROCResult`` with curve, per-image summary, and metadata.
    """
    if variant not in ("best_tp", "top_scoring"):
        raise ValueError(
            f"Invalid variant {variant!r}. Expected 'best_tp' or 'top_scoring'."
        )

    meta_df = table.image_metadata.collect(engine="streaming")

    # Validate: LROC requires gt_label
    if COL_GT_LABEL not in meta_df.columns:
        raise ValueError("LROC requires `gt_label` in image metadata.")

    # Validate: gt_label must not contain nulls
    null_count = meta_df.select(pl.col(COL_GT_LABEL).is_null().sum()).item()
    if null_count > 0:
        raise ValueError(
            f"LROC requires non-null `gt_label` for every image, but found "
            f"{null_count} null value(s). This typically means the matcher "
            f"produced null gt_label — check that contour/bbox extraction "
            f"handles empty results correctly."
        )

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

    per_image = per_image_lf.collect(engine="streaming")

    curve = _build_lroc_curve(per_image)

    return LROCResult(
        curve=curve,
        per_image=per_image,
        n_positive=int(per_image.filter(pl.col(COL_GT_LABEL)).height),
        n_negative=int(per_image.filter(~pl.col(COL_GT_LABEL)).height),
        variant=variant,
        detection_table=table,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_lroc_curve(per_image: pl.DataFrame) -> pl.DataFrame:
    """Construct an LROC curve table from per-image top detections.

    Args:
        per_image: Per-image table with ``gt_label``, ``weight``,
            ``max_score``, ``top_is_tp``.

    Returns:
        DataFrame with ``threshold``, ``fpf``, ``sensitivity``.
    """
    total_weighted_pos = float(
        per_image.select(
            (pl.col(COL_GT_LABEL).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum()
        ).item()
    )
    total_weighted_neg = float(
        per_image.select(
            ((~pl.col(COL_GT_LABEL)).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum()
        ).item()
    )
    n_positive = max(per_image.filter(pl.col(COL_GT_LABEL)).height, 1)
    n_negative = max(per_image.filter(~pl.col(COL_GT_LABEL)).height, 1)

    scored = per_image.filter(pl.col("max_score").is_not_null())
    if scored.height == 0:
        return pl.DataFrame(
            {"threshold": [float("inf")], "fpf": [0.0], "sensitivity": [0.0]}
        )

    bucketed = (
        scored.group_by("max_score")
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
        .sort("threshold", descending=True)
        .with_columns(
            cum_pos_detected=pl.col("pos_detected").cum_sum(),
            cum_neg_detected=pl.col("neg_detected").cum_sum(),
            cum_weighted_pos_detected=pl.col("weighted_pos_detected").cum_sum(),
            cum_weighted_neg_detected=pl.col("weighted_neg_detected").cum_sum(),
        )
        .with_columns(
            sensitivity=pl.when(pl.lit(total_weighted_pos) > 0.0)
            .then(pl.col("cum_weighted_pos_detected") / pl.lit(total_weighted_pos))
            .otherwise(pl.col("cum_pos_detected") / pl.lit(float(n_positive))),
            fpf=pl.when(pl.lit(total_weighted_neg) > 0.0)
            .then(pl.col("cum_weighted_neg_detected") / pl.lit(total_weighted_neg))
            .otherwise(pl.col("cum_neg_detected") / pl.lit(float(n_negative))),
        )
        .select("threshold", "fpf", "sensitivity")
    )
    # Origin (threshold=inf, fpf=0, sensitivity=0)
    inf_row = pl.DataFrame(
        {"threshold": [float("inf")], "fpf": [0.0], "sensitivity": [0.0]}
    )
    # Lower-right endpoint (threshold=-inf, fpf=1.0, sensitivity=max achievable).
    # Max achievable sensitivity is the fraction of positive images that have
    # at least one TP detection at *any* score.
    max_sens = float(bucketed["sensitivity"].max()) if bucketed.height > 0 else 0.0
    lower_right = pl.DataFrame(
        {"threshold": [float("-inf")], "fpf": [1.0], "sensitivity": [max_sens]}
    )
    return pl.concat([bucketed, inf_row, lower_right], how="vertical").sort("threshold")
