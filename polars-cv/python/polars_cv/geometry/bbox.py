"""Bounding-box operations namespace for Polars expressions.

Provides the ``.bbox`` accessor for operations on ``List[BBOX_SCHEMA]`` columns.
"""

from __future__ import annotations

import polars as pl

from polars_cv._namespace import _ArgBinder, _GeomNullPolicy, _PluginNamespace


@pl.api.register_expr_namespace("bbox")
class BBoxNamespace(_GeomNullPolicy, _PluginNamespace):
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

