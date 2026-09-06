"""Detection metrics built on top of polars-cv primitives.

This module provides a layered metrics system:

1. **Matchers** — produce a canonical :class:`DetectionTable` from raw data.
2. **Metric functions** — compute curves and scalar metrics from a
   ``DetectionTable``.
3. **Result objects** — carry computed curves with convenience methods
   (AUC, interpolation, bootstrap CI).
"""

from __future__ import annotations

from ._bootstrap import (
    average_precision_ci_lazy,
    froc_auc_ci_lazy,
    lroc_auc_ci_lazy,
)
from ._matching import BBoxMatcher, ContourMatcher, Matcher, PreMatchedAdapter
from ._metrics import (
    ConfusionResult,
    PrecisionRecallResult,
    average_precision,
    confusion_at_threshold,
    f1_at_threshold,
    froc_auc,
    froc_curve_lazy,
    froc_sensitivity_at_fp,
    froc_summary_table,
    lroc_auc,
    lroc_curve_lazy,
    lroc_sensitivity_at_fpf,
    mean_average_precision,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
)
from ._result import MetricResult
from ._types import DetectionTable

__all__ = [
    # Core types
    "DetectionTable",
    "MetricResult",
    # Matchers
    "Matcher",
    "ContourMatcher",
    "BBoxMatcher",
    "PreMatchedAdapter",
    # Metric functions
    "froc_auc",
    "froc_curve_lazy",
    "froc_sensitivity_at_fp",
    "froc_summary_table",
    "lroc_auc",
    "lroc_curve_lazy",
    "lroc_sensitivity_at_fpf",
    "precision_recall_curve",
    "average_precision",
    "mean_average_precision",
    "precision_at_threshold",
    "recall_at_threshold",
    "f1_at_threshold",
    "confusion_at_threshold",
    # Result types
    "ConfusionResult",
    "PrecisionRecallResult",
    # Bootstrap confidence intervals (lazy, group-aware)
    "froc_auc_ci_lazy",
    "lroc_auc_ci_lazy",
    "average_precision_ci_lazy",
]
