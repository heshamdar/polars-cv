"""Bounding-box operations namespace for Polars expressions.

Provides the ``.bbox`` accessor for operations on ``List[BBOX_SCHEMA]`` columns.
"""

from __future__ import annotations

import polars as pl

from polars_cv._namespace import _GeomNamespace
from polars_cv._ops_generated import _BBoxOpsMixin


@pl.api.register_expr_namespace("bbox")
class BBoxNamespace(_BBoxOpsMixin, _GeomNamespace):
    """Namespace for bounding-box operations on ``List[BBOX_SCHEMA]`` columns.

    Example::

        df.with_columns(
            iou_matrix=pl.col("pred_bboxes").bbox.pairwise_iou(pl.col("gt_bboxes")),
            pairs=pl.col("pred_bboxes").bbox.correspond(
                pl.col("gt_bboxes"), threshold=0.5
            ),
        )
    """
