"""LROC analyzer for heatmap + binary-mask detection workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import polars as pl

from ._auc import partial_auc, trapz_auc
from ._bootstrap import BootstrapResult, bootstrap_metric
from ._prepare import prepare_detection_table

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class LROCResult:
    """LROC outputs and utility methods."""

    curve: pl.DataFrame
    per_image: pl.DataFrame
    n_positive: int
    n_negative: int
    iou_threshold: float
    _image_id_col: str
    _stratify_col: str

    def auc(self, *, fpf_range: tuple[float, float] | None = None) -> float:
        """Compute (partial) AUC under the LROC curve."""
        curve = self.curve.sort("fpf")
        x = curve["fpf"].cast(pl.Float64).to_numpy()
        y = curve["sensitivity"].fill_null(0.0).cast(pl.Float64).to_numpy()
        if x.size == 0:
            return 0.0
        if fpf_range is None:
            return trapz_auc(x, y)
        return partial_auc(x, y, fpf_range[0], fpf_range[1])

    def sensitivity_at_fpf(self, fpf: float) -> float:
        """Interpolate sensitivity at a requested false-positive fraction."""
        curve = self.curve.sort("fpf")
        x = curve["fpf"].cast(pl.Float64).to_numpy()
        y = curve["sensitivity"].fill_null(0.0).cast(pl.Float64).to_numpy()
        if x.size == 0:
            return 0.0
        return float(np.interp(fpf, x, y, left=y[0], right=y[-1]))

    def bootstrap_ci(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int | None = None,
        *,
        metric: Literal["auc", "sensitivity_at_fpf"] = "auc",
        metric_kwargs: dict[str, Any] | None = None,
    ) -> BootstrapResult:
        """Estimate confidence intervals for LROC metrics."""
        metric_kwargs = metric_kwargs or {}
        image_ids = [
            str(value) for value in self.per_image[self._image_id_col].to_list()
        ]
        strata_df = self.per_image.select(
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
                self.per_image,
                on=self._image_id_col,
                how="left",
            )
            curve = _build_lroc_curve(sampled)
            temp = LROCResult(
                curve=curve,
                per_image=self.per_image,
                n_positive=int(sampled.filter(pl.col("gt_label")).height),
                n_negative=int(sampled.filter(~pl.col("gt_label")).height),
                iou_threshold=self.iou_threshold,
                _image_id_col=self._image_id_col,
                _stratify_col=self._stratify_col,
            )
            if metric == "auc":
                return temp.auc(**metric_kwargs)
            return temp.sensitivity_at_fpf(**metric_kwargs)

        point = (
            self.auc(**metric_kwargs)
            if metric == "auc"
            else self.sensitivity_at_fpf(**metric_kwargs)
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


class LROCAnalyzer:
    """Compute LROC curves from heatmaps and binary masks."""

    def __init__(
        self,
        iou_threshold: float = 0.5,
        extraction_threshold: float = 0.1,
        min_contour_area: float = 0.0,
        auto_resize: bool = True,
    ) -> None:
        """Initialize analyzer settings."""
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
        gt_label_col: str,
        image_id_col: str | None = None,
        weight_col: str | None = None,
        stratify_col: str | None = None,
    ) -> LROCResult:
        """Compute LROC operating points."""
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
            gt_min_contour_area=max(self._min_contour_area, 1.0),
            auto_resize=self._auto_resize,
        ).with_columns(_gt_components=pl.col("_gt_contours").list.len().cast(pl.Int64))

        invalid_multi = prepared.filter(
            pl.col("_gt_label") & (pl.col("_gt_components") > 1)
        ).select("_image_id", "_gt_components")
        invalid_preview = invalid_multi.limit(5).collect(engine="streaming")
        if invalid_preview.height > 0:
            preview = invalid_preview.to_dicts()
            raise ValueError(
                "LROC expects <= 1 target contour for positive samples. "
                f"Examples with multiple targets: {preview}"
            )

        matched = prepared.select(
            pl.col("_image_id").alias("image_id"),
            gt_label=pl.col("_gt_label"),
            stratify=pl.col("_stratify"),
            weight=pl.col("_weight"),
            pred_scores=pl.col("_pred_scores"),
            pred_idx=pl.col("_match").struct.field("pred_idx"),
            gt_idx=pl.col("_match").struct.field("gt_idx"),
        )
        mismatch_preview = (
            matched.with_columns(
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

        image_base = matched.select("image_id", "gt_label", "stratify", "weight")
        det_base = (
            matched.select(
                "image_id",
                pred_scores=pl.col("pred_scores"),
                pred_ord=pl.int_ranges(0, pl.col("pred_scores").list.len()),
            )
            .explode("pred_scores", "pred_ord")
            .with_columns(pred_ord=pl.col("pred_ord").cast(pl.UInt32))
        )
        match_pairs = (
            matched.select(
                "image_id", pred_ord=pl.col("pred_idx"), gt_idx=pl.col("gt_idx")
            )
            .explode("pred_ord", "gt_idx")
            .with_columns(pred_ord=pl.col("pred_ord").cast(pl.UInt32))
        )
        top_det = (
            det_base.join(match_pairs, on=["image_id", "pred_ord"], how="left")
            .group_by("image_id")
            .agg(
                max_score=pl.col("pred_scores").max(),
                top_is_tp=pl.col("gt_idx")
                .sort_by(pl.col("pred_scores"), descending=True)
                .first()
                .is_not_null(),
            )
        )
        per_image = (
            image_base.join(top_det, on="image_id", how="left")
            .with_columns(top_is_tp=pl.col("top_is_tp").fill_null(False))
            .collect(engine="streaming")
        )

        curve = _build_lroc_curve(per_image)
        return LROCResult(
            curve=curve,
            per_image=per_image,
            n_positive=int(per_image.filter(pl.col("gt_label")).height),
            n_negative=int(per_image.filter(~pl.col("gt_label")).height),
            iou_threshold=self._iou_threshold,
            _image_id_col="image_id",
            _stratify_col="stratify",
        )


def _build_lroc_curve(per_image: pl.DataFrame) -> pl.DataFrame:
    """Construct an LROC curve table from per-image top detections."""
    thresholds = (
        per_image.select(
            pl.col("max_score").drop_nulls().unique().sort(descending=True)
        )
        .to_series()
        .to_list()
    )
    threshold_df = pl.DataFrame(
        {"threshold": [float("inf"), *[float(v) for v in thresholds]]}
    )

    n_positive = max(per_image.filter(pl.col("gt_label")).height, 1)
    n_negative = max(per_image.filter(~pl.col("gt_label")).height, 1)

    return (
        per_image.join(threshold_df, how="cross")
        .with_columns(
            keep=pl.col("max_score").is_not_null()
            & (pl.col("max_score") >= pl.col("threshold"))
        )
        .group_by("threshold")
        .agg(
            pos_detected=(
                pl.col("gt_label") & pl.col("keep") & pl.col("top_is_tp")
            ).sum(),
            neg_detected=((~pl.col("gt_label")) & pl.col("keep")).sum(),
            n_positive=(pl.col("gt_label")).sum(),
            n_negative=(~pl.col("gt_label")).sum(),
            weighted_pos_detected=(
                (pl.col("gt_label") & pl.col("keep") & pl.col("top_is_tp")).cast(
                    pl.Float64
                )
                * pl.col("weight")
            ).sum(),
            weighted_neg_detected=(
                ((~pl.col("gt_label")) & pl.col("keep")).cast(pl.Float64)
                * pl.col("weight")
            ).sum(),
            weighted_pos_total=(
                pl.col("gt_label").cast(pl.Float64) * pl.col("weight")
            ).sum(),
            weighted_neg_total=(
                (~pl.col("gt_label")).cast(pl.Float64) * pl.col("weight")
            ).sum(),
        )
        .with_columns(
            sensitivity=pl.when(pl.col("weighted_pos_total") > 0.0)
            .then(pl.col("weighted_pos_detected") / pl.col("weighted_pos_total"))
            .otherwise(pl.col("pos_detected") / pl.lit(float(n_positive))),
            fpf=pl.when(pl.col("weighted_neg_total") > 0.0)
            .then(pl.col("weighted_neg_detected") / pl.col("weighted_neg_total"))
            .otherwise(pl.col("neg_detected") / pl.lit(float(n_negative))),
        )
        .sort("threshold")
        .select("threshold", "fpf", "sensitivity")
    )
