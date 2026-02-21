"""LROC metric: localization sensitivity vs false-positive fraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

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

    Attributes:
        curve: DataFrame with ``threshold``, ``fpf``, ``sensitivity``.
        per_image: Per-image top-detection table.
        n_positive: Number of positive images.
        n_negative: Number of negative images.
        iou_threshold: IoU threshold used for matching.
        detection_table: The underlying ``DetectionTable`` for bootstrap.
    """

    per_image: pl.DataFrame = None  # type: ignore[assignment]
    n_positive: int = 0
    n_negative: int = 0
    iou_threshold: float = 0.5
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
) -> LROCResult:
    """Compute LROC operating points from a DetectionTable.

    LROC requires:
    - ``gt_label`` in image metadata (positive/negative per image).
    - Per-image top-detection reduction from ``DetectionTable.to_per_image()``.

    Positive images may contain multiple GT targets. LROC is computed at the
    image level: an image is counted as localized if its top detection is a TP,
    i.e. it matches at least one GT.

    Args:
        table: Canonical detection table produced by a matcher.

    Returns:
        ``LROCResult`` with curve, per-image summary, and metadata.
    """
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

    # Get per-image aggregation and choose the best localized detection.
    # For positive images, the effective score is the highest score among TP
    # detections (if any). For negative images, the effective score is the
    # highest score among all detections.
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
        ).with_columns(
            max_score=pl.when(pl.col(COL_GT_LABEL))
            .then(pl.col("_best_tp_score"))
            .otherwise(pl.col("_max_det_score")),
            top_is_tp=pl.when(pl.col(COL_GT_LABEL))
            .then(pl.col("_best_tp_score").is_not_null())
            .otherwise(pl.lit(False)),
        )
    per_image = per_image_lf.collect(engine="streaming")

    curve = _build_lroc_curve(per_image)

    return LROCResult(
        curve=curve,
        per_image=per_image,
        n_positive=int(per_image.filter(pl.col(COL_GT_LABEL)).height),
        n_negative=int(per_image.filter(~pl.col(COL_GT_LABEL)).height),
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
    inf_row = pl.DataFrame(
        {"threshold": [float("inf")], "fpf": [0.0], "sensitivity": [0.0]}
    )
    return pl.concat([bucketed, inf_row], how="vertical").sort("threshold")
