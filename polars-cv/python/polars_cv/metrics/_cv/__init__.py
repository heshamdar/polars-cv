"""The polars-cv tier of the metrics: everything that reads images or geometry.

The matchers, the object-table and heatmap entry points and the segmentation
measures. This package is the only part of ``polars_cv.metrics`` that imports
polars-cv (public API only); everything outside it needs Polars alone
(``tests/_metrics_boundary.py``).
"""

from __future__ import annotations

from ._bbox import BBoxMatcher
from ._contour import ContourMatcher
from ._evaluate import evaluate_detections, evaluate_heatmaps
from ._match import match_detections
from ._segmentation import evaluate_segmentation, segmentation_measures

__all__ = [
    "BBoxMatcher",
    "ContourMatcher",
    "evaluate_detections",
    "evaluate_heatmaps",
    "evaluate_segmentation",
    "match_detections",
    "segmentation_measures",
]
