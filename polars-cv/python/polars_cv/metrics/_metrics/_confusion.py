"""Confusion counts at a score threshold."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .._types import COL_IS_TP, COL_N_GTS, COL_SCORE, DetectionTable


@dataclass(frozen=True)
class ConfusionResult:
    """True/false positive and false negative counts at a score threshold.

    Detection problems have no true negatives (the background is unbounded), so
    only ``tp``, ``fp`` and ``fn`` are reported. Returned as a frozen dataclass
    to match the other metric result type (:class:`PrecisionRecallResult`); call
    :meth:`to_dict` for the legacy mapping.

    Attributes:
        tp: True positives (matched detections at or above the threshold).
        fp: False positives (unmatched detections at or above the threshold).
        fn: False negatives (ground truths with no matching detection).
    """

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        """``tp / (tp + fp)``; ``0.0`` when there are no positive predictions."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """``tp / (tp + fn)``; ``0.0`` when there are no ground truths."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of :attr:`precision` and :attr:`recall`."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, int]:
        """Return the counts as a ``{"tp", "fp", "fn"}`` mapping."""
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn}


def confusion_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
) -> ConfusionResult:
    """Compute TP, FP, FN counts at a given score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold — detections with ``score >= threshold`` are
            considered active.
        class_id: Optional class filter.

    Returns:
        A :class:`ConfusionResult` with the ``tp``, ``fp`` and ``fn`` counts.
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

    total_gts = int(
        table.image_metadata.select(pl.col(COL_N_GTS).sum())
        .collect(engine="streaming")
        .item()
    )
    fn = max(total_gts - tp, 0)

    return ConfusionResult(tp=tp, fp=fp, fn=fn)
