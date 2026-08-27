"""Bounding-box matcher: List[BBOX_SCHEMA] predictions + GTs -> DetectionTable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .._types import (
    COL_CLASS_ID,
    COL_DET_IDX,
    COL_GROUP_ID,
    COL_GT_IDX,
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IOU,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DEFAULT_CLASS,
    DetectionTable,
    ensure_columns_exist,
    to_lazy,
)
from ._contour import _OVERLAP, _RIGHT_IDX, _confidence_order, _validate_match_alignment

if TYPE_CHECKING:
    pass


class BBoxMatcher:
    """Match detections from bounding-box lists via IoU matching.

    Expects prediction and ground-truth columns as ``List[Struct{x, y, width,
    height}]`` (i.e. ``List[BBOX_SCHEMA]``).  Scores should be provided as a
    separate ``List[Float64]`` column aligned with the prediction bboxes.

    Matching calls ``.bbox.correspond()``, supplying a confidence order, which
    internally converts each bbox to a 4-point rectangular contour and
    delegates to the existing contour matching infrastructure.

    Args:
        iou_threshold: IoU threshold for TP matching.
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        if not (0.0 < iou_threshold <= 1.0):
            raise ValueError("`iou_threshold` must be in (0, 1].")
        self._iou_threshold = iou_threshold

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
        prepared = prepared.with_columns(
            _match=pl.col(pred_col).bbox.correspond(
                pl.col(gt_col),
                threshold=self._iou_threshold,
                order=_confidence_order(score_col),
            ),
            _n_gts=pl.col(gt_col).list.len().fill_null(0).cast(pl.Int64),
        )

        # Build image-level frame
        select_exprs: list[pl.Expr] = [
            pl.col(COL_IMAGE_ID),
            pl.col(COL_WEIGHT),
            pl.col(score_col).alias("_scores"),
            pl.col("_n_gts"),
            pl.col("_match").struct.field(_RIGHT_IDX).alias("gt_idx"),
            pl.col("_match").struct.field(_OVERLAP).alias("iou"),
            (pl.col(gt_col).list.len().fill_null(0) > 0).alias("_gt_label"),
            (
                pl.col(class_col).cast(pl.String).alias(COL_CLASS_ID)
                if class_col is not None
                else pl.lit(DEFAULT_CLASS).alias(COL_CLASS_ID)
            ),
        ]
        if group_col is not None:
            select_exprs.append(pl.col(group_col).cast(pl.String).alias(COL_GROUP_ID))
        image_level = prepared.select(select_exprs)

        image_level_df = image_level.collect(engine="streaming")

        if image_level_df.height == 0:
            return _empty_bbox_detection_table()

        _validate_match_alignment(
            image_level_df.lazy(), image_id_col=COL_IMAGE_ID, scores_col="_scores"
        )

        # Explode into per-detection rows
        from ._contour import _explode_match_to_detections

        detections_lf = _explode_match_to_detections(
            image_level_df.lazy(),
            image_id_col=COL_IMAGE_ID,
            scores_col="_scores",
            gt_idx_col="gt_idx",
            iou_col="iou",
            class_id=DEFAULT_CLASS,
        )

        if class_col is not None:
            detections_lf = detections_lf.drop(COL_CLASS_ID).join(
                image_level_df.lazy().select(COL_IMAGE_ID, COL_CLASS_ID).unique(),
                on=COL_IMAGE_ID,
                how="left",
            )

        # Build image metadata
        group_cols = [COL_GROUP_ID] if COL_GROUP_ID in image_level_df.columns else []
        meta_lf = image_level_df.lazy().select(
            COL_IMAGE_ID,
            COL_CLASS_ID,
            pl.col("_n_gts").alias(COL_N_GTS),
            COL_WEIGHT,
            pl.col("_gt_label").alias(COL_GT_LABEL),
            *group_cols,
        )

        return DetectionTable.from_matched(
            detections_lf,
            meta_lf,
            matching_iou_threshold=self._iou_threshold,
        )


def _empty_bbox_detection_table() -> DetectionTable:
    """Return an empty ``DetectionTable`` for edge cases."""
    det_df = pl.DataFrame(
        schema={
            COL_IMAGE_ID: pl.String,
            COL_CLASS_ID: pl.String,
            COL_SCORE: pl.Float64,
            COL_IS_TP: pl.Boolean,
            COL_GT_IDX: pl.UInt32,
            COL_IOU: pl.Float64,
            COL_DET_IDX: pl.UInt32,
        }
    )
    meta_df = pl.DataFrame(
        schema={
            COL_IMAGE_ID: pl.String,
            COL_CLASS_ID: pl.String,
            COL_N_GTS: pl.Int64,
            COL_WEIGHT: pl.Float64,
            COL_GT_LABEL: pl.Boolean,
        }
    )
    return DetectionTable.from_matched(det_df, meta_df)
