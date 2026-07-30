"""Bounding-box operations namespace for Polars expressions.

Provides the ``.bbox`` accessor for operations on ``List[BBOX_SCHEMA]`` columns.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from polars_cv._namespace import _ArgBinder, _PluginNamespace


@pl.api.register_expr_namespace("bbox")
class BBoxNamespace(_PluginNamespace):
    """Namespace for bounding-box operations on ``List[BBOX_SCHEMA]`` columns.

    Example::

        df.with_columns(
            iou_matrix=pl.col("pred_bboxes").bbox.pairwise_iou(pl.col("gt_bboxes")),
            match_result=pl.col("pred_bboxes").bbox.match_detections(
                pl.col("gt_bboxes"),
                threshold=0.5,
                scores=pl.col("pred_scores"),
            ),
        )
    """

    def pairwise_iou(self, other: pl.Expr) -> pl.Expr:
        """Compute pairwise IoU matrix between two sets of bounding boxes.

        Args:
            other: Ground-truth bboxes column (``List[BBOX_SCHEMA]``).

        Returns:
            ``List[List[Float64]]`` IoU matrix expression.
        """
        return self._plugin("bbox_pairwise_iou", args=[other])

    def match_detections(
        self,
        other: pl.Expr,
        *,
        threshold: float | pl.Expr = 0.5,
        scores: pl.Expr | None = None,
        strategy: Literal["greedy"] = "greedy",
    ) -> pl.Expr:
        """Greedy one-to-one detection matching via IoU on bounding boxes.

        Internally converts bboxes to rectangular contours and delegates to the
        existing contour matching infrastructure.

        Args:
            other: Ground-truth bboxes (``List[BBOX_SCHEMA]``).
            threshold: IoU threshold for a match to be considered a TP.
                Accepts a Polars expression for a per-row threshold.
            scores: Optional per-prediction confidence scores
                (``List[Float64]``). When provided, predictions are processed
                in descending score order.
            strategy: Matching strategy (only ``"greedy"`` is supported).

        Returns:
            Struct expression matching ``MATCH_RESULT_SCHEMA``.
        """
        binder = _ArgBinder()
        binder.add_data("other", other)
        binder.add_data("scores", scores)
        binder.add_param("threshold", threshold)
        return binder.call(self, "bbox_match_detections", strategy=strategy)
