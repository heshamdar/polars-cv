"""Confusion matrix at a score threshold."""

from __future__ import annotations

import polars as pl

from .._types import COL_IS_TP, COL_N_GTS, COL_SCORE, DetectionTable


def confusion_at_threshold(
    table: DetectionTable,
    threshold: float,
    *,
    class_id: str | None = None,
) -> dict[str, int]:
    """Compute TP, FP, FN counts at a given score threshold.

    Args:
        table: Canonical detection table.
        threshold: Score threshold — detections with ``score >= threshold`` are
            considered active.
        class_id: Optional class filter.

    Returns:
        Dictionary with keys ``tp``, ``fp``, ``fn``.
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

    return {"tp": tp, "fp": fp, "fn": fn}
