"""Detection metrics built on top of polars-cv primitives.

This module provides a layered metrics system:

1. **Matchers** — produce a canonical :class:`DetectionTable` from raw data.
2. **Metric functions** — compute curves and scalar metrics from a
   ``DetectionTable``.
3. **Result objects** — carry computed curves with convenience methods
   (AUC, interpolation, bootstrap CI).
"""

from __future__ import annotations

from ._bootstrap import BootstrapResult, bootstrap_metric_sequential, bootstrap_pr_auc
from ._matching import BBoxMatcher, ContourMatcher, Matcher, PreMatchedAdapter
from ._metrics import (
    ConfusionResult,
    FROCResult,
    LROCResult,
    PrecisionRecallResult,
    average_precision,
    confusion_at_threshold,
    f1_at_threshold,
    froc_auc,
    froc_curve,
    froc_curve_lazy,
    lroc_curve,
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
    "BootstrapResult",
    # Matchers
    "Matcher",
    "ContourMatcher",
    "BBoxMatcher",
    "PreMatchedAdapter",
    # Metric functions
    "froc_auc",
    "froc_curve",
    "froc_curve_lazy",
    "lroc_curve",
    "precision_recall_curve",
    "average_precision",
    "mean_average_precision",
    "precision_at_threshold",
    "recall_at_threshold",
    "f1_at_threshold",
    "confusion_at_threshold",
    # Result types
    "ConfusionResult",
    "FROCResult",
    "LROCResult",
    "PrecisionRecallResult",
    # Bootstrap
    "bootstrap_metric_sequential",
    "bootstrap_pr_auc",
]
