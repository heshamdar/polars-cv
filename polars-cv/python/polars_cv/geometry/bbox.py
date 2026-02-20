"""Bounding-box operations namespace for Polars expressions.

Provides the ``.bbox`` accessor for operations on ``List[BBOX_SCHEMA]`` columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    pass

LIB_PATH = Path(__file__).parent.parent


@pl.api.register_expr_namespace("bbox")
class BBoxNamespace:
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

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def pairwise_iou(self, other: pl.Expr) -> pl.Expr:
        """Compute pairwise IoU matrix between two sets of bounding boxes.

        Args:
            other: Ground-truth bboxes column (``List[BBOX_SCHEMA]``).

        Returns:
            ``List[List[Float64]]`` IoU matrix expression.
        """
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="bbox_pairwise_iou",
            args=[self._expr, other],
            is_elementwise=True,
        )

    def match_detections(
        self,
        other: pl.Expr,
        *,
        threshold: float = 0.5,
        scores: pl.Expr | None = None,
        strategy: Literal["greedy"] = "greedy",
    ) -> pl.Expr:
        """Greedy one-to-one detection matching via IoU on bounding boxes.

        Internally converts bboxes to rectangular contours and delegates to the
        existing contour matching infrastructure.

        Args:
            other: Ground-truth bboxes (``List[BBOX_SCHEMA]``).
            threshold: IoU threshold for a match to be considered a TP.
            scores: Optional per-prediction confidence scores
                (``List[Float64]``). When provided, predictions are processed
                in descending score order.
            strategy: Matching strategy (only ``"greedy"`` is supported).

        Returns:
            Struct expression matching ``MATCH_RESULT_SCHEMA``.
        """
        args = [self._expr, other]
        if scores is not None:
            args.append(scores)
        return register_plugin_function(
            plugin_path=LIB_PATH,
            function_name="bbox_match_detections",
            args=args,
            kwargs={"threshold": threshold, "strategy": strategy},
            is_elementwise=True,
        )
