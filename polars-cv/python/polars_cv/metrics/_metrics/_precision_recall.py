"""Precision-Recall metrics: PR curve, AP, mAP, P/R/F1 at threshold.

Every metric here is weighted by ``image_metadata.weight``: a detection carries
its ``(image[, class])`` key's resolved weight (:mod:`.._weights`), precision is
``Σw·tp / Σw·(tp + fp)`` and recall ``Σw·tp / Σw·n_gts`` — scikit-learn's
``sample_weight`` semantics, so an integer weight ``k`` equals the image drawn
``k`` times. Unit weights reduce every formula to the plain counts exactly.
Zero-weight detections carry no mass and add no PR point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from .._result import MetricResult
from .._types import (
    COL_CLASS_ID,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DEFAULT_CLASS,
    DetectionTable,
)
from .._weights import WeightAgg, attach_resolved_weight, weighted_gt_mass
from ._confusion import confusion_at_threshold


@dataclass(frozen=True)
class PrecisionRecallResult(MetricResult):
    """Precision-Recall curve result.

    Attributes:
        curve: DataFrame with ``score``, ``precision``, ``recall``, the raw
            ``cum_tp``/``cum_fp`` counts and the ``cum_weighted_tp``/
            ``cum_weighted_fp`` masses precision and recall are computed from.
        total_gts: Total ground-truth count for this class.
        weighted_gts: Weighted ground-truth mass ``Σ n_gts · w`` — the recall
            denominator.
        class_id: Class this curve was computed for.
    """

    total_gts: int = 0
    weighted_gts: float = 0.0
    class_id: str = DEFAULT_CLASS

    def auc(  # type: ignore[override]
        self,
        *,
        method: Literal["all_points", "11_point", "trapezoidal"] = "all_points",
    ) -> float:  # ty: ignore[invalid-method-override]
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


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def precision_recall_curve(
    table: DetectionTable,
    *,
    class_id: str | None = None,
    weight_agg: WeightAgg = "first",
) -> PrecisionRecallResult:
    """Compute a weighted precision-recall curve from a DetectionTable.

    Detections are bucketed by distinct confidence score and the buckets
    accumulated in descending score order — one PR point per distinct score,
    after the whole tied block, so the curve does not depend on the input row
    order among equal scores. Precision and recall are weighted (see the module
    docstring).

    Args:
        table: Canonical detection table.
        class_id: Restrict to a specific class. ``None`` uses all detections.
        weight_agg: Duplicate-weight resolution policy (see
            :func:`~polars_cv.metrics._weights.resolve_key_weights`).

    Returns:
        ``PrecisionRecallResult`` with the PR curve.
    """
    if class_id is not None:
        table = table.filter_class(class_id)
    resolved_class = class_id or DEFAULT_CLASS

    meta = table.image_metadata
    det_lf = attach_resolved_weight(
        table.detections, meta, weight_agg=weight_agg
    ).filter(pl.col(COL_WEIGHT) != 0.0)
    totals_lf = pl.concat(
        [
            meta.select(total_gts=pl.col(COL_N_GTS).sum().cast(pl.Int64)),
            weighted_gt_mass(meta, [], weight_agg),
        ],
        how="horizontal",
    )
    det_df, totals = pl.collect_all([det_lf, totals_lf], engine="streaming")
    total_gts = int(totals["total_gts"].item() or 0)
    gt_mass = float(totals["gt_mass"].item() or 0.0)

    if det_df.height == 0 or total_gts == 0 or not gt_mass > 0.0:
        empty_curve = pl.DataFrame(
            schema={
                "score": pl.Float64,
                "precision": pl.Float64,
                "recall": pl.Float64,
                "cum_tp": pl.Int64,
                "cum_fp": pl.Int64,
                "cum_weighted_tp": pl.Float64,
                "cum_weighted_fp": pl.Float64,
            }
        )
        return PrecisionRecallResult(
            curve=empty_curve,
            total_gts=total_gts,
            weighted_gts=gt_mass,
            class_id=resolved_class,
        )

    curve = (
        _score_buckets(det_df.lazy(), [])
        .sort(COL_SCORE, descending=True)
        .with_columns(
            cum_tp=pl.col("_tp").cum_sum(),
            cum_fp=pl.col("_fp").cum_sum(),
            cum_weighted_tp=pl.col("_wtp").cum_sum(),
            cum_weighted_fp=pl.col("_wfp").cum_sum(),
        )
        .with_columns(
            precision=pl.col("cum_weighted_tp")
            / (pl.col("cum_weighted_tp") + pl.col("cum_weighted_fp")),
            recall=pl.col("cum_weighted_tp") / pl.lit(gt_mass),
        )
        .select(
            pl.col(COL_SCORE).alias("score"),
            "precision",
            "recall",
            "cum_tp",
            "cum_fp",
            "cum_weighted_tp",
            "cum_weighted_fp",
        )
        .collect(engine="streaming")
    )

    return PrecisionRecallResult(
        curve=curve,
        total_gts=total_gts,
        weighted_gts=gt_mass,
        class_id=resolved_class,
    )


def _score_buckets(det: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    """Per-``(keys, score)`` TP/FP counts and weighted masses of weighted detections.

    The canonical tie convention shared by the scalar curve and the grouped AP
    authority: every detection sharing a score is one bucket, so a PR point sits
    after the whole tied block.
    """
    tp = pl.col(COL_IS_TP)
    return det.group_by(*keys, COL_SCORE).agg(
        _tp=tp.cast(pl.Int64).sum(),
        _fp=(~tp).cast(pl.Int64).sum(),
        _wtp=(tp.cast(pl.Float64) * pl.col(COL_WEIGHT)).sum(),
        _wfp=((~tp).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum(),
    )


def average_precision(
    table: DetectionTable,
    *,
    class_id: str | None = None,
    interpolation: Literal["all_points", "11_point"] = "all_points",
    weight_agg: WeightAgg = "first",
) -> float:
    """Compute weighted Average Precision for a single class.

    Args:
        table: Canonical detection table.
        class_id: Restrict to a specific class.
        interpolation: ``"all_points"`` (trapezoidal) or ``"11_point"`` (VOC).
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        AP value in [0, 1].
    """
    pr = precision_recall_curve(table, class_id=class_id, weight_agg=weight_agg)
    return pr.auc(method=interpolation)


def mean_average_precision(
    table: DetectionTable,
    *,
    iou_thresholds: list[float] | None = None,
    interpolation: Literal["all_points", "11_point"] = "all_points",
    weight_agg: WeightAgg = "first",
) -> float:
    """Compute weighted Mean Average Precision across classes and IoU thresholds.

    If ``iou_thresholds`` is provided, the stored ``iou`` column is re-thresholded
    at each level to recompute ``is_tp`` -- **no re-matching is needed**.

    Args:
        table: Canonical detection table.
        iou_thresholds: IoU thresholds to average over. Defaults to
            ``[0.5]`` (Pascal VOC). Use ``[0.5, 0.55, ..., 0.95]`` for COCO.
        interpolation: AP interpolation method.
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        mAP value in [0, 1].
    """
    thresholds = iou_thresholds or [0.5]

    if interpolation == "all_points":
        return _mean_average_precision_all_points(table, thresholds, weight_agg)

    # The grouped authority implements only the all-points estimator; the VOC
    # 11-point method has no grouped form, so it keeps the per-(threshold, class)
    # eager path. This branch also validates ``interpolation``: an unknown value
    # reaches ``PrecisionRecallResult.auc`` and raises there, as before.
    class_ids = table.class_ids()
    ap_values: list[float] = []
    for iou_thresh in thresholds:
        rethresholded = table.at_iou_threshold(iou_thresh)
        for cid in class_ids:
            ap_values.append(
                average_precision(
                    rethresholded,
                    class_id=cid,
                    interpolation=interpolation,
                    weight_agg=weight_agg,
                )
            )

    if not ap_values:
        return 0.0
    return float(pl.Series("ap", ap_values).mean())  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


def _mean_average_precision_all_points(
    table: DetectionTable,
    thresholds: list[float],
    weight_agg: WeightAgg = "first",
) -> float:
    """Vectorized all-points mAP over every ``(threshold, class)`` cell.

    One lazy plan, one collect: the per-cell APs come from the shared grouped
    authority :func:`all_points_ap_by_group` (the same estimator the scalar
    ``average_precision`` uses), rather than a Python loop of eager
    ``average_precision`` collects. Re-thresholding goes through the canonical
    :meth:`DetectionTable.at_iou_threshold`, so the ``is_tp`` rule and its
    "lowering has no effect" warning are not re-implemented here.

    The averaging grid is every ``(threshold, class)`` pair — classes from
    :meth:`DetectionTable.class_ids` — so a class with no detections (or zero
    GTs) still averages in as ``AP = 0``, exactly as the eager loop did.
    """
    class_ids = table.class_ids()
    if not class_ids or not thresholds:
        return 0.0

    meta = table.image_metadata
    gts = weighted_gt_mass(meta, [COL_CLASS_ID], weight_agg)

    # Per-detection rows, stacked across thresholds with ``is_tp`` recomputed by
    # the canonical re-thresholder (which also emits the lowering warning).
    per_threshold = [
        attach_resolved_weight(
            table.at_iou_threshold(iou_thresh).detections, meta, weight_agg=weight_agg
        )
        .select(COL_CLASS_ID, COL_SCORE, COL_IS_TP, COL_WEIGHT)
        .with_columns(_iou_t=pl.lit(float(iou_thresh), dtype=pl.Float64))
        for iou_thresh in thresholds
    ]
    expanded = pl.concat(per_threshold, how="vertical").join(
        gts, on=COL_CLASS_ID, how="left"
    )
    ap = all_points_ap_by_group(expanded, group_col=["_iou_t", COL_CLASS_ID])

    grid = (
        pl.LazyFrame(
            {"_iou_t": pl.Series([float(t) for t in thresholds], dtype=pl.Float64)}
        )
        .join(
            pl.LazyFrame({COL_CLASS_ID: pl.Series(class_ids, dtype=pl.String)}),
            how="cross",
        )
        .join(gts, on=COL_CLASS_ID, how="left")
        .join(ap, on=["_iou_t", COL_CLASS_ID], how="left")
        .with_columns(
            # A cell with GT mass but no qualifying detections is a null AP →
            # 0.0; a cell whose class has no GT mass has undefined recall → 0.0.
            ap=pl.when(pl.col("gt_mass") > 0)
            .then(pl.col("ap").fill_null(0.0))
            .otherwise(0.0)
        )
        .select(pl.col("ap").mean())
    )
    result = grid.collect(engine="streaming").item()
    return float(result) if result is not None else 0.0


def precision_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
    weight_agg: WeightAgg = "first",
) -> float:
    """Weighted precision at a given score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold.
        class_id: Optional class filter.
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        Precision value; ``1.0`` when no weighted detection is at or above the
        threshold.
    """
    conf = confusion_at_threshold(
        table, threshold, class_id=class_id, weight_agg=weight_agg
    )
    predicted = conf.weighted_tp + conf.weighted_fp
    return 1.0 if predicted == 0 else conf.weighted_tp / predicted


def recall_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
    weight_agg: WeightAgg = "first",
) -> float:
    """Weighted recall at a given score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold.
        class_id: Optional class filter.
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        Recall value; ``0.0`` when there is no weighted ground-truth mass.
    """
    return confusion_at_threshold(
        table, threshold, class_id=class_id, weight_agg=weight_agg
    ).recall


def f1_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
    weight_agg: WeightAgg = "first",
) -> float:
    """Weighted F1 score at a given score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold.
        class_id: Optional class filter.
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        F1 value in [0, 1].
    """
    # One confusion pass gives the weighted tp/fp/fn; precision and recall fall
    # out with the standalone functions' edge cases (precision 1.0 with nothing
    # predicted, recall 0.0 with no ground-truth mass).
    conf = confusion_at_threshold(
        table, threshold, class_id=class_id, weight_agg=weight_agg
    )
    predicted = conf.weighted_tp + conf.weighted_fp
    p = 1.0 if predicted == 0 else conf.weighted_tp / predicted
    r = conf.recall
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _all_points_ap(curve: pl.DataFrame) -> float:
    """Compute AP using monotone-envelope interpolation — lazily.

    Applies the standard monotonically decreasing precision envelope
    (right-to-left cumulative maximum) before trapezoidal integration, matching
    the COCO and scikit-learn AP definitions. This is the ungrouped form of the
    lazy authority :func:`all_points_ap_by_group`: the same envelope and the same
    shift-based trapezoid anchored at recall = 0 (the first row's ``d_recall``
    falls back to its own recall, so the leftmost block ``recall₀ · P₀`` is
    counted, per COCO / scikit-learn ``Σ (Rₙ − Rₙ₋₁) · Pₙ`` with ``R₀ = 0``). No
    eager Series integral is used.

    Args:
        curve: PR curve DataFrame with ``recall`` and ``precision``.

    Returns:
        All-points interpolated AP with monotone envelope.
    """
    if curve.height == 0:
        return 0.0

    ap = (
        curve.lazy()
        .select(
            pl.col("recall").cast(pl.Float64),
            pl.col("precision").cast(pl.Float64),
        )
        # recall is non-decreasing in the score-descending curve; sort makes the
        # envelope and the shift-based trapezoid order-stable across thread counts.
        .sort("recall")
        .with_columns(precision=pl.col("precision").reverse().cum_max().reverse())
        .with_columns(
            d_recall=(pl.col("recall") - pl.col("recall").shift(1)).fill_null(
                pl.col("recall")
            ),
            avg_precision=(
                (pl.col("precision") + pl.col("precision").shift(1)) / 2.0
            ).fill_null(pl.col("precision")),
        )
        .select(ap=(pl.col("d_recall") * pl.col("avg_precision")).sum())
        .collect(engine="streaming")
        .item()
    )
    return float(ap)


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
    group_col: str | list[str],
) -> pl.LazyFrame:
    """Weighted all-points AP per group — the lazy authority every bootstrap shares.

    ``expanded`` carries ``[*group_col, score, is_tp, weight, gt_mass]`` (one row
    per detection; ``weight`` its resolved weight, ``gt_mass`` the group's
    ``Σ n_gts · w`` broadcast per row). The estimator is identical to the scalar
    :func:`precision_recall_curve` + :func:`_all_points_ap`: bucket by score
    (:func:`_score_buckets`), accumulate in descending score order within group,
    the monotone decreasing precision envelope, then trapezoidal integration
    anchored at recall = 0. Zero-weight detections are dropped, as there. Keeping
    the sort as the last row-reordering step (nothing joins between it and the
    windowed ops) makes the curve stable across thread counts.

    Args:
        expanded: Per-detection frame with the group key(s), ``score``, ``is_tp``,
            ``weight`` and per-group ``gt_mass``.
        group_col: The grouping column(s) — a single name (e.g. ``bootstrap_id``)
            or a list (e.g. ``[group_id, bootstrap_id]``).

    Returns:
        ``LazyFrame`` with ``[*group_col, ap]``.
    """
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    weighted = expanded.filter(pl.col(COL_WEIGHT) != 0.0)
    mass = weighted.group_by(keys).agg(pl.col("gt_mass").first())
    pr = (
        _score_buckets(weighted, keys)
        .join(mass, on=keys, how="left", nulls_equal=True)
        .sort(*keys, COL_SCORE, descending=[False] * len(keys) + [True])
        .with_columns(
            cum_wtp=pl.col("_wtp").cum_sum().over(keys),
            cum_wfp=pl.col("_wfp").cum_sum().over(keys),
        )
        .with_columns(
            precision=pl.col("cum_wtp") / (pl.col("cum_wtp") + pl.col("cum_wfp")),
            recall=pl.col("cum_wtp") / pl.col("gt_mass"),
        )
        .with_columns(
            precision=pl.col("precision").reverse().cum_max().reverse().over(keys),
        )
    )
    return (
        pr.with_columns(
            d_recall=(
                pl.col("recall") - pl.col("recall").shift(1).over(keys)
            ).fill_null(pl.col("recall")),
            avg_precision=(
                (pl.col("precision") + pl.col("precision").shift(1).over(keys)) / 2.0
            ).fill_null(pl.col("precision")),
        )
        .with_columns(slice_area=pl.col("d_recall") * pl.col("avg_precision"))
        .group_by(keys)
        .agg(ap=pl.col("slice_area").sum())
    )
