"""Bounding-box matcher: List[BBOX_SCHEMA] predictions + GTs -> DetectionTable."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from .._types import (
    COL_IMAGE_ID,
    COL_WEIGHT,
    DetectionTable,
    ensure_columns_exist,
    to_lazy,
)
from ._contour import _confidence_order
from ._table import iou_thresholds, matched_table


class BBoxMatcher:
    """Match detections from bounding-box lists via IoU matching.

    Expects prediction and ground-truth columns as ``List[Struct{x, y, width,
    height}]`` (i.e. ``List[BBOX_SCHEMA]``).  Scores should be provided as a
    separate ``List[Float64]`` column aligned with the prediction bboxes.

    Matching calls ``.bbox.correspond()``, supplying a confidence order: the
    IoU of two boxes is computed analytically and the greedy assignment is the
    one ``.contour.correspond()`` uses.

    Multi-class data is one row per (image, class); each detection keeps its
    own row's class.

    Args:
        iou_threshold: IoU threshold for TP matching, or a sequence of them to
            match once per threshold (COCO's ``[0.5, 0.55, …, 0.95]``): the
            table then carries an ``iou_threshold`` column, and a detection
            that lost its ground truth at one threshold can claim it at
            another (see :attr:`DetectionTable.iou_thresholds`).
    """

    def __init__(self, iou_threshold: float | Sequence[float] = 0.5) -> None:
        self._iou_thresholds = iou_thresholds(iou_threshold)

    def match(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        *,
        pred_col: str,
        gt_col: str,
        score_col: str | None = None,
        class_col: str | None = None,
        image_id_col: str | None = None,
        weight_col: str | None = None,
        group_col: str | None = None,
    ) -> DetectionTable:
        """Produce a ``DetectionTable`` from bbox prediction/GT lists.

        Args:
            data: Input frame with one image/sample per row.
            pred_col: Prediction bboxes column (``List[BBOX_SCHEMA]``).
            gt_col: Ground-truth bboxes column (``List[BBOX_SCHEMA]``).
            score_col: Per-prediction score column (``List[Float64]``).
                Required for bbox matching.
            class_col: Optional class label column.
            image_id_col: Optional image identifier column.
            weight_col: Optional sample weight column.
            group_col: Optional grouping column.

        Returns:
            Validated ``DetectionTable``.

        Raises:
            ValueError: If ``score_col`` is not provided.
        """
        if score_col is None:
            raise ValueError(
                "BBoxMatcher requires `score_col` — a List[Float64] column of "
                "per-prediction confidence scores."
            )

        lf = to_lazy(data)
        schema_names = list(lf.collect_schema().names())
        ensure_columns_exist(schema_names, [pred_col, gt_col, score_col])
        if class_col is not None:
            ensure_columns_exist(schema_names, [class_col])
        if image_id_col is not None:
            ensure_columns_exist(schema_names, [image_id_col])
        if weight_col is not None:
            ensure_columns_exist(schema_names, [weight_col])
        if group_col is not None:
            ensure_columns_exist(schema_names, [group_col])

        # Assign image_id
        if image_id_col is None:
            prepared = lf.with_row_index(name="_row_idx").with_columns(
                pl.col("_row_idx").cast(pl.String).alias(COL_IMAGE_ID)
            )
        else:
            prepared = lf.with_columns(
                pl.col(image_id_col).cast(pl.String).alias(COL_IMAGE_ID)
            )

        # Assign weight
        prepared = prepared.with_columns(
            (
                pl.col(weight_col).cast(pl.Float64)
                if weight_col is not None
                else pl.lit(1.0, dtype=pl.Float64)
            ).alias(COL_WEIGHT)
        )

        # Pair predictions with GT boxes. Confidence decides the visit order,
        # which is this layer's choice to make; `correspond` only sees overlap.
        def match(threshold: float | pl.Expr) -> pl.Expr:
            return pl.col(pred_col).bbox.correspond(  # ty: ignore[unresolved-attribute]
                pl.col(gt_col),
                threshold=threshold,
                order=_confidence_order(score_col),
            )

        return matched_table(
            prepared,
            match=match,
            thresholds=self._iou_thresholds,
            scores_col=score_col,
            gt_col=gt_col,
            class_col=class_col,
            group_col=group_col,
        )
