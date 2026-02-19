"""FROC analyzer for heatmap + binary-mask detection workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from ._auc import partial_auc, trapz_auc, weighted_curve
from ._bootstrap import BootstrapResult, bootstrap_metric
from ._prepare import prepare_detection_table


@dataclass(frozen=True)
class FROCResult:
    """FROC outputs and post-hoc metric helpers."""

    curve: pl.DataFrame
    matched: pl.DataFrame
    per_image_threshold: pl.DataFrame
    n_images: int
    total_targets: int
    iou_threshold: float
    _image_id_col: str
    _weight_col: str | None
    _stratify_col: str | None

    def auc(self, *, fp_range: tuple[float, float] | None = None) -> float:
        """Compute (partial) AUC under the FROC curve."""
        curve = self.curve.sort("fp_per_image")
        x = curve["fp_per_image"].cast(pl.Float64).to_numpy()
        y = curve["sensitivity"].fill_null(0.0).cast(pl.Float64).to_numpy()
        if x.size == 0:
            return 0.0
        if fp_range is None:
            return trapz_auc(x, y)
        return partial_auc(x, y, fp_range[0], fp_range[1])

    def sensitivity_at_fp(self, fp_per_image: float) -> float:
        """Interpolate sensitivity at a requested FP/image rate."""
        curve = self.curve.sort("fp_per_image")
        x = curve["fp_per_image"].cast(pl.Float64).to_numpy()
        y = curve["sensitivity"].fill_null(0.0).cast(pl.Float64).to_numpy()
        if x.size == 0:
            return 0.0
        return float(np.interp(fp_per_image, x, y, left=y[0], right=y[-1]))

    def summary_table(self, fp_rates: list[float] | None = None) -> pl.DataFrame:
        """Build a sensitivity summary at standard FP/image operating points."""
        rates = fp_rates or [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        return pl.DataFrame(
            {
                "fp_per_image": rates,
                "sensitivity": [self.sensitivity_at_fp(rate) for rate in rates],
            }
        )

    def bootstrap_ci(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int | None = None,
        *,
        metric: Literal["auc", "sensitivity_at_fp"] = "auc",
        metric_kwargs: dict[str, Any] | None = None,
    ) -> BootstrapResult:
        """Estimate CI for an FROC metric via image-level bootstrap."""
        metric_kwargs = metric_kwargs or {}
        image_ids = [str(value) for value in self.matched[self._image_id_col].to_list()]
        strata: dict[str, str] | None = None
        if self._stratify_col is not None:
            strata_df = self.matched.select(
                self._image_id_col, self._stratify_col
            ).unique()
            strata = {
                str(image_id): str(stratum)
                for image_id, stratum in zip(
                    strata_df[self._image_id_col].to_list(),
                    strata_df[self._stratify_col].to_list(),
                    strict=True,
                )
            }

        def _metric(sampled_ids: list[str]) -> float:
            sampled = pl.DataFrame({self._image_id_col: sampled_ids}).join(
                self.per_image_threshold,
                on=self._image_id_col,
                how="left",
            )
            curve = _curve_from_dense(sampled, weight_col=self._weight_col)
            temp = FROCResult(
                curve=curve,
                matched=self.matched,
                per_image_threshold=self.per_image_threshold,
                n_images=len(sampled_ids),
                total_targets=self.total_targets,
                iou_threshold=self.iou_threshold,
                _image_id_col=self._image_id_col,
                _weight_col=self._weight_col,
                _stratify_col=self._stratify_col,
            )
            if metric == "auc":
                return temp.auc(**metric_kwargs)
            return temp.sensitivity_at_fp(**metric_kwargs)

        point = (
            self.auc(**metric_kwargs)
            if metric == "auc"
            else self.sensitivity_at_fp(**metric_kwargs)
        )
        return bootstrap_metric(
            image_ids=image_ids,
            metric_fn=_metric,
            point_estimate=point,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
            strata=strata,
        )


class FROCAnalyzer:
    """Compute FROC curves from prediction heatmaps and GT masks."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        extraction_threshold: float = 0.1,
        min_contour_area: float = 0.0,
        auto_resize: bool = True,
    ) -> None:
        """Initialize analyzer settings.

        Args:
            iou_threshold: IoU threshold for TP matching.
            extraction_threshold: Threshold for contour extraction from heatmaps.
            min_contour_area: Minimum extracted contour area.
            auto_resize: Whether to resize heatmaps to mask shapes automatically.
        """
        if not (0.0 < iou_threshold <= 1.0):
            raise ValueError("`iou_threshold` must be in (0, 1].")
        self._iou_threshold = iou_threshold
        self._extraction_threshold = extraction_threshold
        self._min_contour_area = min_contour_area
        self._auto_resize = auto_resize

    def compute(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        *,
        pred_col: str,
        gt_mask_col: str,
        gt_label_col: str | None = None,
        image_id_col: str | None = None,
        weight_col: str | None = None,
        stratify_col: str | None = None,
        thresholds: list[float] | None = None,
    ) -> FROCResult:
        """Compute FROC operating points.

        Args:
            data: Input frame with one image/sample per row.
            pred_col: Prediction heatmap column.
            gt_mask_col: Ground-truth mask column.
            gt_label_col: Optional precomputed image-level label column.
            image_id_col: Optional ID column; defaults to row index.
            weight_col: Optional sample-weight column.
            stratify_col: Optional stratification column for bootstrap.
            thresholds: Optional explicit score thresholds.

        Returns:
            FROCResult with curve points and bootstrap helpers.
        """
        prepared = prepare_detection_table(
            data,
            pred_col=pred_col,
            gt_mask_col=gt_mask_col,
            gt_label_col=gt_label_col,
            image_id_col=image_id_col,
            weight_col=weight_col,
            stratify_col=stratify_col,
            iou_threshold=self._iou_threshold,
            extraction_threshold=self._extraction_threshold,
            min_contour_area=self._min_contour_area,
            auto_resize=self._auto_resize,
        )

        matched = prepared.select(
            pl.col("_image_id").alias("image_id"),
            "_pred_scores",
            pred_idx=pl.col("_match").struct.field("pred_idx"),
            gt_idx=pl.col("_match").struct.field("gt_idx"),
            n_gts=pl.col("_n_gts"),
            gt_label=pl.col("_gt_label"),
            stratify=pl.col("_stratify"),
            weight=pl.col("_weight"),
        ).collect(engine="streaming")

        if matched.height == 0:
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
                {
                    "image_id": [],
                    "threshold": [],
                    "tp": [],
                    "fp": [],
                    "n_gts": [],
                    "weight": [],
                }
            )
            return FROCResult(
                curve=empty_curve,
                matched=matched,
                per_image_threshold=empty_dense,
                n_images=0,
                total_targets=0,
                iou_threshold=self._iou_threshold,
                _image_id_col="image_id",
                _weight_col="weight",
                _stratify_col="stratify",
            )

        threshold_values = (
            thresholds
            if thresholds is not None
            else _derive_thresholds_from_scores(matched)
        )
        threshold_df = pl.DataFrame({"threshold": threshold_values}).sort("threshold")
        mismatch_preview = (
            matched.lazy()
            .with_columns(
                pred_len=pl.col("pred_idx").list.len().fill_null(0),
                gt_len=pl.col("gt_idx").list.len().fill_null(0),
            )
            .filter(pl.col("pred_len") != pl.col("gt_len"))
            .select("image_id", "pred_len", "gt_len")
            .limit(5)
            .collect(engine="streaming")
        )
        if mismatch_preview.height > 0:
            raise ValueError(
                "Matched prediction-index and GT-index lists are misaligned. "
                f"Examples: {mismatch_preview.to_dicts()}"
            )

        per_det_base = (
            matched.lazy()
            .select(
                "image_id",
                "n_gts",
                "gt_label",
                "stratify",
                "weight",
                scores=pl.col("_pred_scores"),
                pred_ord=pl.int_ranges(0, pl.col("_pred_scores").list.len()),
            )
            .explode("scores", "pred_ord")
            .with_columns(pred_ord=pl.col("pred_ord").cast(pl.UInt32))
        )
        match_pairs = (
            matched.lazy()
            .select(
                "image_id",
                pred_ord=pl.col("pred_idx"),
                gt_idx=pl.col("gt_idx"),
            )
            .explode("pred_ord", "gt_idx")
            .with_columns(pred_ord=pl.col("pred_ord").cast(pl.UInt32))
        )
        per_det = per_det_base.join(
            match_pairs, on=["image_id", "pred_ord"], how="left"
        ).drop("pred_ord")

        per_threshold_counts = (
            per_det.join(threshold_df.lazy(), how="cross")
            .with_columns(
                keep=pl.col("scores") >= pl.col("threshold"),
                is_tp=pl.col("gt_idx").is_not_null(),
            )
            .filter(pl.col("keep"))
            .group_by("image_id", "threshold")
            .agg(
                tp=pl.col("is_tp").sum().cast(pl.Int64),
                fp=(~pl.col("is_tp")).sum().cast(pl.Int64),
            )
            .collect(engine="streaming")
        )

        all_image_thresholds = matched.select("image_id").join(
            threshold_df, how="cross"
        )
        dense = (
            all_image_thresholds.join(
                per_threshold_counts, on=["image_id", "threshold"], how="left"
            )
            .with_columns(
                tp=pl.col("tp").fill_null(0).cast(pl.Int64),
                fp=pl.col("fp").fill_null(0).cast(pl.Int64),
            )
            .join(
                matched.select("image_id", "n_gts", "weight"), on="image_id", how="left"
            )
        )

        curve = _curve_from_dense(dense, weight_col="weight")
        total_targets = int(matched.select(pl.col("n_gts").sum()).item())
        return FROCResult(
            curve=curve,
            matched=matched,
            per_image_threshold=dense,
            n_images=matched.height,
            total_targets=total_targets,
            iou_threshold=self._iou_threshold,
            _image_id_col="image_id",
            _weight_col="weight",
            _stratify_col="stratify",
        )


def _derive_thresholds_from_scores(with_scores: pl.DataFrame) -> list[float]:
    """Derive sklearn-like thresholds from observed detection scores."""
    score_df = (
        with_scores.select(pl.col("_pred_scores").explode().alias("score"))
        .drop_nulls()
        .unique()
        .sort("score", descending=True)
    )
    if score_df.height == 0:
        return [float("inf")]
    score_values = [float(value) for value in score_df["score"].to_list()]
    return [float("inf"), *score_values]


def _curve_from_dense(
    dense_counts: pl.DataFrame, *, weight_col: str | None
) -> pl.DataFrame:
    """Aggregate dense per-image/per-threshold table into curve points."""
    if weight_col is not None and weight_col in dense_counts.columns:
        return weighted_curve(
            dense_counts,
            threshold_col="threshold",
            tp_col="tp",
            fp_col="fp",
            n_gts_col="n_gts",
            weight_col=weight_col,
        ).select(
            "threshold", "tp", "fp", "fn", "total_gts", "fp_per_image", "sensitivity"
        )

    n_images = max(dense_counts.select(pl.col("image_id").n_unique()).item(), 1)
    total_gts = int(dense_counts.select(pl.col("n_gts").sum()).item())
    return (
        dense_counts.group_by("threshold")
        .agg(
            tp=pl.col("tp").sum().cast(pl.Int64),
            fp=pl.col("fp").sum().cast(pl.Int64),
            total_gts=pl.col("n_gts").sum().cast(pl.Int64),
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
