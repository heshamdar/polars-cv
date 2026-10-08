"""Confusion counts at a score threshold."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .._types import COL_IS_TP, COL_N_GTS, COL_SCORE, COL_WEIGHT, DetectionTable
from .._weights import WeightAgg, attach_resolved_weight, weighted_gt_mass


@dataclass(frozen=True)
class ConfusionResult:
    """True/false positive and false negative counts at a score threshold.

    Detection problems have no true negatives (the background is unbounded), so
    only ``tp``, ``fp`` and ``fn`` are reported — as raw counts of the table's
    detections and ground truths, and as **weighted** masses (each detection or
    ground truth carrying its ``(image[, class])`` weight). The derived
    :attr:`precision` / :attr:`recall` / :attr:`f1` read the weighted masses, so
    they agree with :func:`precision_at_threshold` and friends; under unit
    weights the two sets of numbers are equal. Call :meth:`to_dict` for the
    legacy count mapping.

    Attributes:
        tp: True positives (matched detections at or above the threshold).
        fp: False positives (unmatched detections at or above the threshold).
        fn: False negatives (ground truths with no matching detection).
        weighted_tp: ``Σ w`` over the true positives.
        weighted_fp: ``Σ w`` over the false positives.
        weighted_fn: ``max(Σ w · n_gts − weighted_tp, 0)``.
    """

    tp: int
    fp: int
    fn: int
    weighted_tp: float
    weighted_fp: float
    weighted_fn: float

    @property
    def precision(self) -> float:
        """Weighted ``tp / (tp + fp)``; ``0.0`` when nothing is predicted."""
        denom = self.weighted_tp + self.weighted_fp
        return self.weighted_tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Weighted ``tp / (tp + fn)``; ``0.0`` with no ground-truth mass."""
        denom = self.weighted_tp + self.weighted_fn
        return self.weighted_tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of :attr:`precision` and :attr:`recall`."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, int]:
        """Return the raw counts as a ``{"tp", "fp", "fn"}`` mapping."""
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn}


def confusion_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
    weight_agg: WeightAgg = "first",
) -> ConfusionResult:
    """Compute TP, FP, FN counts and weighted masses at a score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold — detections with ``score >= threshold`` are
            considered active.
        class_id: Optional class filter.
        weight_agg: Duplicate-weight resolution policy (see
            :func:`~polars_cv.metrics._weights.resolve_key_weights`).

    Returns:
        A :class:`ConfusionResult`.
    """
    if class_id is not None:
        table = table.filter_class(class_id)

    meta = table.image_metadata
    tp = pl.col(COL_IS_TP)
    w = pl.col(COL_WEIGHT)
    det = (
        attach_resolved_weight(table.detections, meta, weight_agg=weight_agg)
        .filter(pl.col(COL_SCORE) >= threshold)
        .select(
            tp=tp.sum().cast(pl.Int64),
            fp=(~tp).sum().cast(pl.Int64),
            weighted_tp=(tp.cast(pl.Float64) * w).sum(),
            weighted_fp=((~tp).cast(pl.Float64) * w).sum(),
        )
    )
    gts = pl.concat(
        [
            meta.select(total_gts=pl.col(COL_N_GTS).sum().cast(pl.Int64)),
            weighted_gt_mass(meta, [], weight_agg),
        ],
        how="horizontal",
    )
    row = pl.concat([det, gts], how="horizontal").collect().row(0, named=True)
    n_tp, wtp = int(row["tp"] or 0), float(row["weighted_tp"] or 0.0)
    return ConfusionResult(
        tp=n_tp,
        fp=int(row["fp"] or 0),
        fn=max(int(row["total_gts"] or 0) - n_tp, 0),
        weighted_tp=wtp,
        weighted_fp=float(row["weighted_fp"] or 0.0),
        weighted_fn=max(float(row["gt_mass"] or 0.0) - wtp, 0.0),
    )
