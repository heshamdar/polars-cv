"""Detection metrics built on top of polars-cv primitives.

This module provides a layered metrics system:

1. **Matchers** — produce a canonical :class:`DetectionTable` from raw data.
2. **Metric functions** — compute curves and scalar metrics from a
   ``DetectionTable``.
3. **Statistics** — every metric as one grouped lazy reduction
   (:class:`Statistic`), so it can be read per class, per IoU threshold, per
   subgroup or per bootstrap replicate (:func:`bootstrap_ci`).
4. **Result objects** — carry computed curves with convenience methods
   (AUC, interpolation, bootstrap CI).
"""

from __future__ import annotations

from ._bootstrap import (
    average_precision_ci_lazy,
    bootstrap_ci,
    froc_auc_ci_lazy,
    lroc_auc_ci_lazy,
)
from ._inputs import group_objects, match_detections
from ._matching import BBoxMatcher, ContourMatcher, Matcher, PreMatchedAdapter
from ._metrics import (
    ConfusionResult,
    PrecisionRecallResult,
    average_precision,
    confusion_at_threshold,
    f1_at_threshold,
    froc_auc,
    froc_curve_lazy,
    froc_operating_range,
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
from ._statistics import (
    AP,
    CPM,
    FROC_RATES,
    FROCAUC,
    LROCAUC,
    F1At,
    FROCSensitivity,
    LROCSensitivity,
    MeanOver,
    PrecisionAt,
    Recall,
    RecallAt,
    Statistic,
    mean_ap,
)
from ._types import DetectionTable

__all__ = [
    # Core types
    "DetectionTable",
    "MetricResult",
    # Object tables -> DetectionTable
    "group_objects",
    "match_detections",
    # Matchers
    "Matcher",
    "ContourMatcher",
    "BBoxMatcher",
    "PreMatchedAdapter",
    # Metric functions
    "froc_auc",
    "froc_curve_lazy",
    "froc_sensitivity_at_fp",
    "froc_operating_range",
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
    # Statistics (grouped, lazy) and the one CI engine over them
    "Statistic",
    "AP",
    "Recall",
    "PrecisionAt",
    "RecallAt",
    "F1At",
    "FROCSensitivity",
    "CPM",
    "FROCAUC",
    "LROCAUC",
    "LROCSensitivity",
    "MeanOver",
    "mean_ap",
    "FROC_RATES",
    "bootstrap_ci",
    # Bootstrap confidence intervals (lazy, group-aware)
    "froc_auc_ci_lazy",
    "lroc_auc_ci_lazy",
    "average_precision_ci_lazy",
]
