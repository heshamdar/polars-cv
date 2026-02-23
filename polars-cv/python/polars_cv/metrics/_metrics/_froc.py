"""FROC metric: sensitivity vs false positives per image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from .._auc import mann_whitney_u_auc, weighted_curve
from .._result import MetricResult
from .._types import (
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DetectionTable,
)


@dataclass(frozen=True)
class FROCResult(MetricResult):
    """FROC-specific result with sensitivity-at-FP helpers.

    Attributes:
        curve: DataFrame with columns ``threshold``, ``tp``, ``fp``, ``fn``,
            ``total_gts``, ``fp_per_image``, ``sensitivity``.
        per_image_threshold: Dense per-image/per-threshold TP/FP counts.
        n_images: Number of images in the dataset.
        total_targets: Total GT count across all images.
        iou_threshold: IoU threshold used for matching.
        detection_table: The underlying ``DetectionTable`` for bootstrap.
    """

    per_image_threshold: pl.DataFrame = None  # type: ignore[assignment]
    n_images: int = 0
    total_targets: int = 0
    iou_threshold: float = 0.5
    detection_table: DetectionTable | None = None

    def auc(  # type: ignore[override]
        self,
        *,
        fp_range: tuple[float, float] | None = None,
        normalize: bool = False,
    ) -> float:
        """Compute (partial) AUC under the FROC curve.

        Args:
            fp_range: Optional ``(lo, hi)`` false-positive-per-image range.
            normalize: Whether to normalize the AUC by the range of the x-values.

        Returns:
            Area under the FROC curve.
        """
        return super().auc(
            x_col="fp_per_image",
            y_col="sensitivity",
            x_range=fp_range,
            normalize=normalize,
        )

    def sensitivity_at_fp(self, fp_per_image: float) -> float:
        """Interpolate sensitivity at a requested FP/image rate.

        Args:
            fp_per_image: Target false-positive-per-image rate.

        Returns:
            Interpolated sensitivity value.
        """
        return self.interpolate(
            x_col="fp_per_image", y_col="sensitivity", at=fp_per_image
        )

    def summary_table(  # type: ignore[override]
        self,
        fp_rates: list[float] | None = None,
    ) -> pl.DataFrame:
        """Build a sensitivity summary at standard FP/image operating points.

        Args:
            fp_rates: Operating points. Defaults to standard radiology set.

        Returns:
            DataFrame with ``fp_per_image`` and ``sensitivity`` columns.
        """
        rates = fp_rates or [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        return super().summary_table(
            x_col="fp_per_image",
            y_col="sensitivity",
            operating_points=rates,
        )

    def mann_whitney_auc(
        self,
        *,
        level: str = "detection",
    ) -> float:
        """Compute Mann-Whitney U statistic as a non-parametric AUC estimate.

        Args:
            level: Granularity of the comparison.
                ``"detection"`` — P(random TP score > random FP score).
                ``"image"`` — P(positive-image effective score > negative-image
                effective score).  For positive images the effective score is
                the max TP detection score (0 if no TPs); for negative images
                it is the max detection score (0 if none).

        Returns:
            Mann-Whitney U AUC in [0, 1].  Returns 0.5 when one group is
            empty.

        Raises:
            ValueError: If ``level`` is not ``"detection"`` or ``"image"``,
                or if ``"image"`` level is requested but ``detection_table``
                is not available.
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
        """Image-level MW-U: P(positive-image score > negative-image score)."""
        if self.detection_table is None:
            raise ValueError(
                "mann_whitney_auc(level='image') requires detection_table."
            )
        det_df = self.detection_table.detections.collect(engine="streaming")
        meta_df = self.detection_table.image_metadata.collect(engine="streaming")

        if COL_GT_LABEL not in meta_df.columns:
            raise ValueError(
                "mann_whitney_auc(level='image') requires gt_label in metadata."
            )

        # Positive images: max TP score (0 if no TPs)
        pos_ids = meta_df.filter(pl.col(COL_GT_LABEL)).select(COL_IMAGE_ID)
        pos_scores = (
            pos_ids.join(
                det_df.filter(pl.col(COL_IS_TP))
                .group_by(COL_IMAGE_ID)
                .agg(pl.col(COL_SCORE).max()),
                on=COL_IMAGE_ID,
                how="left",
            )
            .with_columns(pl.col(COL_SCORE).fill_null(0.0))
            .select(COL_SCORE)
            .to_numpy()
            .flatten()
        )

        # Negative images: max detection score (0 if none)
        neg_ids = meta_df.filter(~pl.col(COL_GT_LABEL)).select(COL_IMAGE_ID)
        neg_scores = (
            neg_ids.join(
                det_df.group_by(COL_IMAGE_ID).agg(pl.col(COL_SCORE).max()),
                on=COL_IMAGE_ID,
                how="left",
            )
            .with_columns(pl.col(COL_SCORE).fill_null(0.0))
            .select(COL_SCORE)
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
            metric: ``"auc"`` or ``"sensitivity_at_fp"``.
            metric_kwargs: Extra kwargs passed to the metric method.

        Returns:
            ``BootstrapResult`` with percentile confidence interval.
        """
        from .._bootstrap import bootstrap_metric_sequential

        metric_kwargs = metric_kwargs or {}
        image_ids = [
            str(v) for v in self.per_image_threshold[COL_IMAGE_ID].unique().to_list()
        ]
        strata: dict[str, str] | None = None
        if self.detection_table is not None:
            image_ids, strata = self.detection_table.image_ids_and_strata()

        def _metric(sampled_ids: list[str]) -> float:
            sampled = pl.DataFrame({COL_IMAGE_ID: sampled_ids}).join(
                self.per_image_threshold, on=COL_IMAGE_ID, how="left"
            )
            curve = _curve_from_dense(sampled, weighted=True)
            sampled_total_targets = int(
                sampled.select(COL_IMAGE_ID, COL_N_GTS)
                .unique(subset=[COL_IMAGE_ID])
                .select(pl.col(COL_N_GTS).sum())
                .item()
            )
            temp = FROCResult(
                curve=curve,
                per_image_threshold=self.per_image_threshold,
                n_images=len(sampled_ids),
                total_targets=sampled_total_targets,
                iou_threshold=self.iou_threshold,
            )
            if metric == "auc":
                return temp.auc(**metric_kwargs)
            return temp.sensitivity_at_fp(**metric_kwargs)

        point = (
            self.auc(**metric_kwargs)
            if metric == "auc"
            else self.sensitivity_at_fp(**metric_kwargs)
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


def froc_curve(
    table: DetectionTable,
    *,
    thresholds: list[float] | None = None,
) -> FROCResult:
    """Compute FROC operating points from a DetectionTable.

    Args:
        table: Canonical detection table produced by a matcher.
        thresholds: Optional explicit score thresholds to evaluate.

    Returns:
        ``FROCResult`` with curve, per-image dense counts, and metadata.
    """
    det_df, meta_df = table.collect()

    if det_df.height == 0:
        return _empty_froc_result(table)

    # Derive thresholds from observed scores if not provided
    threshold_values = thresholds or _derive_thresholds(det_df)
    threshold_df = pl.DataFrame({"threshold": threshold_values}).sort("threshold")

    # Per-detection table already has image_id, score, is_tp
    per_det = det_df.lazy().select(COL_IMAGE_ID, COL_SCORE, COL_IS_TP)

    # Cross-join with thresholds, filter to kept detections, count TP/FP
    per_threshold_counts = (
        per_det.join(threshold_df.lazy(), how="cross")
        .filter(pl.col(COL_SCORE) >= pl.col("threshold"))
        .group_by(COL_IMAGE_ID, "threshold")
        .agg(
            tp=pl.col(COL_IS_TP).sum().cast(pl.Int64),
            fp=(~pl.col(COL_IS_TP)).sum().cast(pl.Int64),
        )
        .collect(engine="streaming")
    )

    # Dense grid: every image x every threshold
    all_image_thresholds = meta_df.select(COL_IMAGE_ID).join(threshold_df, how="cross")
    dense = (
        all_image_thresholds.join(
            per_threshold_counts, on=[COL_IMAGE_ID, "threshold"], how="left"
        )
        .with_columns(
            tp=pl.col("tp").fill_null(0).cast(pl.Int64),
            fp=pl.col("fp").fill_null(0).cast(pl.Int64),
        )
        .join(
            meta_df.select(COL_IMAGE_ID, COL_N_GTS, COL_WEIGHT),
            on=COL_IMAGE_ID,
            how="left",
        )
    )

    curve = _curve_from_dense(dense, weighted=True)
    total_targets = int(meta_df.select(pl.col(COL_N_GTS).sum()).item())

    return FROCResult(
        curve=curve,
        per_image_threshold=dense,
        n_images=meta_df.height,
        total_targets=total_targets,
        iou_threshold=table._matching_iou_threshold or 0.5,
        detection_table=table,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _derive_thresholds(det_df: pl.DataFrame) -> list[float]:
    """Derive thresholds from observed detection scores."""
    score_df = (
        det_df.select(pl.col(COL_SCORE).alias("score"))
        .drop_nulls()
        .unique()
        .sort("score", descending=True)
    )
    if score_df.height == 0:
        return [float("inf")]
    values = [float(v) for v in score_df["score"].to_list()]
    return [float("inf"), *values]


def _curve_from_dense(
    dense_counts: pl.DataFrame,
    *,
    weighted: bool,
) -> pl.DataFrame:
    """Aggregate dense per-image/per-threshold table into FROC curve points."""
    if weighted and COL_WEIGHT in dense_counts.columns:
        return weighted_curve(
            dense_counts,
            threshold_col="threshold",
            tp_col="tp",
            fp_col="fp",
            n_gts_col=COL_N_GTS,
            weight_col=COL_WEIGHT,
        ).select(
            "threshold", "tp", "fp", "fn", "total_gts", "fp_per_image", "sensitivity"
        )

    n_images = max(dense_counts.select(pl.col(COL_IMAGE_ID).n_unique()).item(), 1)
    # De-duplicate before summing: each image appears once per threshold in
    # the dense grid, so summing n_gts over the whole grid would overcount
    # by a factor of len(thresholds).
    total_gts = int(
        dense_counts.select(COL_IMAGE_ID, COL_N_GTS)
        .unique(subset=[COL_IMAGE_ID])
        .select(pl.col(COL_N_GTS).sum())
        .item()
    )
    return (
        dense_counts.group_by("threshold")
        .agg(
            tp=pl.col("tp").sum().cast(pl.Int64),
            fp=pl.col("fp").sum().cast(pl.Int64),
            total_gts=pl.col(COL_N_GTS).sum().cast(pl.Int64),
        )
        .with_columns(
            fn=(pl.col("total_gts") - pl.col("tp")).clip(lower_bound=0),
            fp_per_image=pl.col("fp") / pl.lit(float(n_images)),
            sensitivity=pl.when(pl.lit(total_gts) > 0)
            .then(pl.col("tp") / pl.lit(float(total_gts)))
            .otherwise(pl.lit(0.0)),
        )
        .sort("threshold")
    )


def _empty_froc_result(table: DetectionTable) -> FROCResult:
    """Return an empty FROC result."""
    empty_curve = pl.DataFrame(
        {
            "threshold": [float("inf")],
            "tp": [0],
            "fp": [0],
            "fn": [0],
            "total_gts": [0],
            "fp_per_image": [0.0],
            "sensitivity": [0.0],
        }
    )
    empty_dense = pl.DataFrame(
        schema={
            COL_IMAGE_ID: pl.String,
            "threshold": pl.Float64,
            "tp": pl.Int64,
            "fp": pl.Int64,
            COL_N_GTS: pl.Int64,
            COL_WEIGHT: pl.Float64,
        }
    )
    return FROCResult(
        curve=empty_curve,
        per_image_threshold=empty_dense,
        n_images=0,
        total_targets=0,
        detection_table=table,
    )
