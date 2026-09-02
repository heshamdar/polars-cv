"""Individual metric implementations operating on DetectionTable."""

from __future__ import annotations

from ._confusion import ConfusionResult, confusion_at_threshold
from ._froc import FROCResult, froc_auc, froc_curve, froc_curve_lazy
from ._lroc import LROCResult, lroc_auc, lroc_curve, lroc_curve_lazy
from ._precision_recall import (
    PrecisionRecallResult,
    average_precision,
    f1_at_threshold,
    mean_average_precision,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
)

__all__ = [
    "ConfusionResult",
    "FROCResult",
    "LROCResult",
    "PrecisionRecallResult",
    "average_precision",
    "confusion_at_threshold",
    "f1_at_threshold",
    "froc_auc",
    "froc_curve",
    "froc_curve_lazy",
    "lroc_auc",
    "lroc_curve",
    "lroc_curve_lazy",
    "mean_average_precision",
    "precision_at_threshold",
    "precision_recall_curve",
    "recall_at_threshold",
]
