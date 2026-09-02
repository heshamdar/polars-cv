"""Individual metric implementations operating on DetectionTable."""

from __future__ import annotations

from ._confusion import ConfusionResult, confusion_at_threshold
from ._froc import (
    froc_auc,
    froc_curve_lazy,
    froc_sensitivity_at_fp,
    froc_summary_table,
)
from ._lroc import (
    lroc_auc,
    lroc_curve_lazy,
    lroc_sensitivity_at_fpf,
)
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
    "PrecisionRecallResult",
    "average_precision",
    "confusion_at_threshold",
    "f1_at_threshold",
    "froc_auc",
    "froc_curve_lazy",
    "froc_sensitivity_at_fp",
    "froc_summary_table",
    "lroc_auc",
    "lroc_curve_lazy",
    "lroc_sensitivity_at_fpf",
    "mean_average_precision",
    "precision_at_threshold",
    "precision_recall_curve",
    "recall_at_threshold",
]
