"""LROC metric: localization sensitivity vs false-positive fraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from .._auc import CorrectionMethod, detection_level_mann_whitney, mann_whitney_u_auc
from .._auc_expr import (
    collapse_curve,
    mann_whitney_auc_expr,
    partial_auc_expr,
    trapz_auc_expr,
)
from .._result import MetricResult
from .._types import (
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IS_TP,
    COL_SCORE,
    COL_WEIGHT,
    DetectionTable,
)

# Internal dummy group key used to run the group-aware curve/AUC path with a
# single implicit group. Dropped from every public result.
_DUMMY_GROUP = "_lroc_grp"


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
        method: Literal["trapezoidal", "mann_whitney"] = "trapezoidal",
        fpf_range: tuple[float, float] | None = None,
        correction: CorrectionMethod = None,
        level: Literal["detection", "image"] = "image",
    ) -> float:
        """Compute AUC for the LROC curve.

        Args:
            method: AUC computation method.
                ``"trapezoidal"`` (default) integrates the LROC curve.
                ``"mann_whitney"`` computes a non-parametric AUC via the
                Mann-Whitney U statistic.
            fpf_range: Optional ``(lo, hi)`` false-positive fraction range
                for partial AUC (trapezoidal only).
            correction: Partial-AUC correction (trapezoidal only).
                ``None`` returns the raw area.
                ``"normalize"`` divides by the range width.
                ``"mcclish"`` applies McClish's standardized correction.
            level: Granularity for Mann-Whitney (ignored for trapezoidal).
                ``"detection"`` -- P(random TP score > random FP score).
                ``"image"`` (default) -- P(positive-image score >
                negative-image score).

        Returns:
            AUC value.

        Raises:
            ValueError: On invalid ``method``/``level`` or unsupported
                parameter combinations.
        """
        if method == "mann_whitney":
            if fpf_range is not None or correction is not None:
                raise ValueError(
                    "fpf_range and correction are not supported with "
                    "method='mann_whitney'. Mann-Whitney computes a global "
                    "rank statistic, not a curve integral."
                )
            return self._mann_whitney(level)
        if method == "trapezoidal":
            return super().auc(
                x_col="fpf",
                y_col="sensitivity",
                x_range=fpf_range,
                correction=correction,
            )
        raise ValueError(
            f"Unknown method {method!r}. Expected 'trapezoidal' or 'mann_whitney'."
        )

    def sensitivity_at_fpf(self, fpf: float) -> float | None:
        """Interpolate sensitivity at a requested false-positive fraction.

        Args:
            fpf: Target false-positive fraction.

        Returns:
            Interpolated sensitivity, or ``None`` when ``fpf`` is outside
            the observed range of the curve.
        """
        return self.interpolate(x_col="fpf", y_col="sensitivity", at=fpf)

    # ------------------------------------------------------------------
    # Bootstrap hooks
    # ------------------------------------------------------------------

    def _get_detection_table(self) -> DetectionTable:
        """Return the underlying DetectionTable."""
        if self.detection_table is None:
            raise ValueError("bootstrap_ci / Mann-Whitney requires detection_table.")
        return self.detection_table

    def _reconstruct(self, sampled_ids: list[str]) -> LROCResult:
        """Rebuild an LROCResult from bootstrap-sampled image IDs.

        Uses the pre-computed per-image table for fast curve reconstruction
        and also builds a sampled DetectionTable for MW and other
        detection-level metrics.
        """
        table = self._get_detection_table()

        # Fast curve rebuild from per-image table
        sampled = pl.DataFrame({COL_IMAGE_ID: sampled_ids}).join(
            self.per_image, on=COL_IMAGE_ID, how="left"
        )
        curve = _build_lroc_curve(sampled)

        # Build sampled DetectionTable for MW / detection-level metrics
        sampled_ids_lf = pl.DataFrame({COL_IMAGE_ID: sampled_ids}).lazy()
        sampled_det = sampled_ids_lf.join(table.detections, on=COL_IMAGE_ID, how="left")
        sampled_meta = sampled_ids_lf.join(
            table.image_metadata, on=COL_IMAGE_ID, how="left"
        )
        sampled_table = DetectionTable.from_matched(sampled_det, sampled_meta)

        return LROCResult(
            curve=curve,
            per_image=sampled,
            n_positive=int(sampled.filter(pl.col(COL_GT_LABEL)).height),
            n_negative=int(sampled.filter(~pl.col(COL_GT_LABEL)).height),
            iou_threshold=self.iou_threshold,
            variant=self.variant,
            detection_table=sampled_table,
        )

    # ------------------------------------------------------------------
    # Mann-Whitney internals
    # ------------------------------------------------------------------

    def _mann_whitney(self, level: str) -> float:
        """Dispatch Mann-Whitney computation by level."""
        if level == "detection":
            table = self._get_detection_table()
            det_df = table.detections.collect(engine="streaming")
            return detection_level_mann_whitney(det_df, COL_SCORE, COL_IS_TP)
        if level == "image":
            return self._mann_whitney_image()
        raise ValueError(
            f"Unsupported level {level!r}. Expected 'detection' or 'image'."
        )

    def _mann_whitney_image(self) -> float:
        """Image-level MW-U: P(pos-image score > neg-image score)."""
        if self.per_image is None:
            raise ValueError(
                "auc(method='mann_whitney', level='image') requires per_image data."
            )
        pos = self.per_image.filter(pl.col(COL_GT_LABEL))
        neg = self.per_image.filter(~pl.col(COL_GT_LABEL))

        # Positive: score from best TP (0 if no TP)
        pos_scores = pos.select(
            pl.when(pl.col("top_is_tp"))
            .then(pl.col("max_score"))
            .otherwise(0.0)
            .fill_null(0.0)
            .alias("s")
        )["s"]
        # Negative: max detection score (0 if no detections)
        neg_scores = neg.select(pl.col("max_score").fill_null(0.0).alias("s"))["s"]
        return mann_whitney_u_auc(pos_scores, neg_scores)


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

    per_image = _scored_per_image_lazy(table, variant).collect(engine="streaming")

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
# Expression-valued, group-aware curve + AUC (lazy)
# ---------------------------------------------------------------------------


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

    The single authority for LROC's per-image commitment logic, shared by the
    eager :func:`lroc_curve` and the lazy :func:`lroc_curve_lazy`.
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

    The expression-valued replacement for the eager :func:`lroc_curve` curve
    build: cumulative sums run ``.over`` the group and the weighted denominators
    are per-group aggregations. With ``group_by=None`` the numbers match
    :func:`lroc_curve` exactly.

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

    The expression-valued replacement for ``LROCResult.auc``.

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
    # Sort by descending threshold, which *is* ascending fpf: the negative
    # counts are cumulative over descending score. Thresholds are unique
    # (group_by(max_score) plus the +inf origin and -inf lower-right endpoint)
    # so this is a total order, while fpf ties and Polars' sort defaults to
    # maintain_order=False — sorting on fpf would leave the y at each tie
    # boundary, and therefore the AUC, unspecified. See `_froc.py`.
    return pl.concat([bucketed, inf_row, lower_right], how="vertical").sort(
        "threshold", descending=True
    )
