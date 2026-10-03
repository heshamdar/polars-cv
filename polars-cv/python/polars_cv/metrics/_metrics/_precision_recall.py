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

from .._grouped_scan import exact_mean, exact_sums
from .._result import MetricResult
from .._types import (
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DEFAULT_CLASS,
    DetectionTable,
)
from .._weights import WeightAgg, attach_resolved_weight, weighted_gt_mass
from ._confusion import confusion_at_threshold

#: The AP estimators. ``"all_points"`` integrates the precision envelope as a
#: step function over every recall the curve reaches (Pascal VOC 2010+);
#: ``"11_point"`` / ``"101_point"`` average the envelope on a fixed recall grid
#: (Pascal VOC 2007 / COCO).
APInterpolation = Literal["all_points", "11_point", "101_point"]

_N_POINT_METHODS: dict[int, APInterpolation] = {11: "11_point", 101: "101_point"}
_GRID_SIZES: dict[str, int] = {v: k for k, v in _N_POINT_METHODS.items()}


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
        method: Literal[
            "all_points", "11_point", "101_point", "trapezoidal"
        ] = "all_points",
    ) -> float:  # ty: ignore[invalid-method-override]
        """Compute Average Precision (AUC of the PR curve).

        Args:
            method: Computation method.
                ``"all_points"`` (default) integrates the monotonically
                decreasing precision envelope as a step function over every
                recall the curve reaches (Pascal VOC 2010+).
                ``"11_point"`` / ``"101_point"`` average the envelope on a
                fixed recall grid (Pascal VOC 2007 / COCO).
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
        if method in _GRID_SIZES:
            return _n_point_ap(self.curve, _GRID_SIZES[method])
        if method == "trapezoidal":
            return super().auc(x_col="recall", y_col="precision")
        raise ValueError(
            f"Unknown method {method!r}. Expected 'all_points', "
            f"'11_point', '101_point' or 'trapezoidal'."
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
    interpolation: APInterpolation = "all_points",
    weight_agg: WeightAgg = "first",
) -> float:
    """Compute weighted Average Precision for a single class.

    Args:
        table: Canonical detection table.
        class_id: Restrict to a specific class.
        interpolation: ``"all_points"`` (step-integrated envelope, VOC 2010+),
            ``"11_point"`` (VOC 2007) or ``"101_point"`` (COCO).
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        AP value in [0, 1].
    """
    validate_interpolation(interpolation)
    pr = precision_recall_curve(table, class_id=class_id, weight_agg=weight_agg)
    return pr.auc(method=interpolation)


def mean_average_precision(
    table: DetectionTable,
    *,
    iou_thresholds: list[float] | None = None,
    interpolation: APInterpolation = "all_points",
    weight_agg: WeightAgg = "first",
) -> float:
    """Compute weighted Mean Average Precision across classes and IoU thresholds.

    The mean of :class:`~polars_cv.metrics.AP` over every ``(threshold,
    class)`` cell of the metadata — ``MeanOver(AP(interpolation),
    undefined="zero")``: a class without ground truth averages in as ``0``
    (COCO leaves it out instead; see :func:`~polars_cv.metrics.mean_ap`).

    On a table matched at several thresholds (a matcher given a sequence of
    ``iou_threshold``), each matched threshold is evaluated on its own
    matching, as COCO does. Any other threshold — and every threshold of a
    single matching — is evaluated by re-thresholding the stored ``iou``
    (:meth:`DetectionTable.at_iou_threshold`), which does not re-match.

    Args:
        table: Canonical detection table.
        iou_thresholds: IoU thresholds to average over. Defaults to the
            table's matched thresholds on a sweep, else ``[0.5]`` (Pascal
            VOC).
        interpolation: AP interpolation method.
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        mAP value in [0, 1].
    """
    from .._statistics import AP, MeanOver

    validate_interpolation(interpolation)
    if iou_thresholds is None:
        if table._sweep:
            evaluated = table
        else:
            evaluated = table.at_iou_threshold(0.5)
    else:
        evaluated = DetectionTable.stack(
            {float(t): table.at_iou_threshold(t) for t in iou_thresholds}
        )
    statistic = MeanOver(AP(interpolation), undefined="zero")
    return statistic.value(evaluated, weight_agg=weight_agg)


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
    """All-points AP of a PR curve — the ungrouped view of :func:`ap_from_points`."""
    return _curve_ap(curve, "all_points")


def _n_point_ap(curve: pl.DataFrame, n_points: int) -> float:
    """N-point interpolated AP of a PR curve (11: Pascal VOC 2007, 101: COCO)."""
    return _curve_ap(curve, _N_POINT_METHODS[n_points])


def _curve_ap(curve: pl.DataFrame, interpolation: APInterpolation) -> float:
    """AP of one curve through the grouped authority, under a dropped dummy key."""
    if curve.height == 0:
        return 0.0
    points = curve.lazy().select(
        pl.lit(0, dtype=pl.Int32).alias("_g"),
        pl.col("score").alias(COL_SCORE),
        pl.col("recall").cast(pl.Float64),
        pl.col("precision").cast(pl.Float64),
    )
    out = ap_from_points(points, ["_g"], interpolation).collect(engine="streaming")
    return float(out["ap"].item()) if out.height else 0.0


# ---------------------------------------------------------------------------
# Grouped (lazy) PR estimators — the single authority for vectorized bootstrap
# ---------------------------------------------------------------------------


def validate_interpolation(interpolation: str) -> None:
    """Raise ``ValueError`` for an AP interpolation nothing implements."""
    if interpolation != "all_points" and interpolation not in _GRID_SIZES:
        raise ValueError(
            f"Unknown interpolation {interpolation!r}. Expected 'all_points', "
            f"'11_point' or '101_point'."
        )


def _recall_grid(n_points: int) -> list[float]:
    """The recall grid ``i · 1/(n−1)``, as ``numpy.linspace`` builds it.

    The reference implementations build their grids this way — COCO's
    ``np.linspace(0, 1, 101)`` and the VOC 2007 devkit's
    ``np.arange(0, 1.1, 0.1)`` — so a recall landing exactly on a grid value
    (``3/10``) compares the way it does there (``3 · 0.1 > 0.3``).
    """
    step = 1.0 / (n_points - 1)
    return [i * step for i in range(n_points - 1)] + [1.0]


def pr_points_by_group(expanded: pl.LazyFrame, keys: list[str]) -> pl.LazyFrame:
    """One weighted PR point per ``(keys, score)`` bucket, lazily.

    ``expanded`` carries ``[*keys, score, is_tp, weight, gt_mass]`` (one row per
    detection; ``gt_mass`` the group's ``Σ n_gts · w``). Zero-weight detections
    are dropped. Returns ``[*keys, score, recall, precision]`` with the
    cumulative counts after each whole tied block (:func:`_score_buckets`).
    """
    weighted = expanded.filter(pl.col(COL_WEIGHT) != 0.0)
    mass = weighted.group_by(keys).agg(pl.col("gt_mass").first())
    return (
        _score_buckets(weighted, keys)
        .join(mass, on=keys, how="left", nulls_equal=True)
        .sort(*keys, COL_SCORE, descending=[False] * len(keys) + [True])
        .with_columns(
            cum_wtp=pl.col("_wtp").cum_sum().over(keys),
            cum_wfp=pl.col("_wfp").cum_sum().over(keys),
        )
        .select(
            *keys,
            COL_SCORE,
            recall=pl.col("cum_wtp") / pl.col("gt_mass"),
            precision=pl.col("cum_wtp") / (pl.col("cum_wtp") + pl.col("cum_wfp")),
        )
    )


def ap_from_points(
    points: pl.LazyFrame,
    keys: list[str],
    interpolation: APInterpolation,
) -> pl.LazyFrame:
    """AP per group from PR points — the one integration every AP path shares.

    ``points`` carries ``[*keys, score, recall, precision]``. Precision is
    replaced by its envelope (the highest precision at any equal or higher
    recall), then:

    * ``"all_points"``: ``Σ (Rₖ − Rₖ₋₁) · P̂ₖ`` with ``R₀ = 0`` — the envelope
      as a step function. (A trapezoid ``(P̂ₖ + P̂ₖ₋₁)/2`` agrees on untied
      scores but overstates AP when a tied block mixes TPs and FPs, which
      lowers precision while raising recall.)
    * ``"11_point"`` / ``"101_point"``: the mean over the recall grid
      (:func:`_recall_grid`) of the envelope at the first point reaching each
      grid recall, ``0`` where none does.

    Rows are sorted by recall (score descending among equal recalls) before
    any windowed step, so the result is stable across thread counts.

    Returns:
        ``[*keys, ap]`` — one row per group that has at least one point.
    """
    validate_interpolation(interpolation)
    ordered = points.sort(
        *keys,
        "recall",
        COL_SCORE,
        descending=[False] * len(keys) + [False, True],
    ).with_columns(
        precision=pl.col("precision").reverse().cum_max().reverse().over(keys)
    )
    if interpolation == "all_points":
        d_recall = (pl.col("recall") - pl.col("recall").shift(1).over(keys)).fill_null(
            pl.col("recall")
        )
        return exact_sums(
            ordered.with_columns(_area=d_recall * pl.col("precision")),
            keys,
            ap=pl.col("_area"),
        )

    grid = pl.LazyFrame(
        {"_t": pl.Series(_recall_grid(_GRID_SIZES[interpolation]), dtype=pl.Float64)}
    )
    queries = ordered.select(keys).unique().join(grid, how="cross").sort("_t")
    # The envelope is non-increasing in recall, so its value at the first
    # point with recall >= t is the highest precision at any recall >= t.
    knots = ordered.group_by(*keys, "recall").agg(_p=pl.col("precision").max())
    reached = queries.join_asof(
        knots.select(*keys, pl.col("recall").alias("_t"), "_p").sort("_t"),
        on="_t",
        by=keys,
        strategy="forward",
        # Both sides are sorted on `_t` globally, hence within every group.
        check_sortedness=False,
    )
    return exact_mean(
        reached.with_columns(pl.col("_p").fill_null(0.0)), keys, "_p", "ap"
    )


def all_points_ap_by_group(
    expanded: pl.LazyFrame,
    *,
    group_col: str | list[str],
) -> pl.LazyFrame:
    """Weighted all-points AP per group — the lazy authority every bootstrap shares.

    ``expanded`` carries ``[*group_col, score, is_tp, weight, gt_mass]`` (one row
    per detection; ``weight`` its resolved weight, ``gt_mass`` the group's
    ``Σ n_gts · w`` broadcast per row). The estimator is identical to the scalar
    :func:`precision_recall_curve` + :func:`_all_points_ap`: one PR point per
    score bucket (:func:`pr_points_by_group`) integrated by
    :func:`ap_from_points`.

    Args:
        expanded: Per-detection frame with the group key(s), ``score``, ``is_tp``,
            ``weight`` and per-group ``gt_mass``.
        group_col: The grouping column(s) — a single name (e.g. ``bootstrap_id``)
            or a list (e.g. ``[group_id, bootstrap_id]``).

    Returns:
        ``LazyFrame`` with ``[*group_col, ap]``.
    """
    return ap_by_group(expanded, group_col=group_col, interpolation="all_points")


def ap_by_group(
    expanded: pl.LazyFrame,
    *,
    group_col: str | list[str],
    interpolation: APInterpolation,
) -> pl.LazyFrame:
    """Weighted AP per group under any interpolation (see :func:`ap_from_points`)."""
    keys = [group_col] if isinstance(group_col, str) else list(group_col)
    return ap_from_points(pr_points_by_group(expanded, keys), keys, interpolation)
