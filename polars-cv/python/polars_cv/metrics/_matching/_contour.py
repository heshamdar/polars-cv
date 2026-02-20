"""Contour-based matcher: heatmap + binary mask -> DetectionTable."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ...pipeline import Pipeline
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

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Contour schemas (used only for casting extracted contours)
# ---------------------------------------------------------------------------

_POINT_STRUCT = pl.Struct([pl.Field("x", pl.Float64), pl.Field("y", pl.Float64)])
_EXTRACTED_CONTOUR_SCHEMA = pl.Struct(
    [
        pl.Field("exterior", pl.List(_POINT_STRUCT)),
        pl.Field("holes", pl.List(pl.List(_POINT_STRUCT))),
        pl.Field("is_closed", pl.Boolean),
    ]
)
_EXTRACTED_CONTOUR_SET_SCHEMA = pl.List(_EXTRACTED_CONTOUR_SCHEMA)


# ---------------------------------------------------------------------------
# Shared pipeline helpers
# ---------------------------------------------------------------------------


def _validate_or_resize_shapes(
    lf: pl.LazyFrame,
    *,
    pred_col: str,
    gt_mask_col: str,
    auto_resize: bool,
) -> tuple[pl.LazyFrame, str]:
    """Validate heatmap/mask shape match or resize prediction heatmaps.

    Args:
        lf: Input lazy frame.
        pred_col: Prediction heatmap column.
        gt_mask_col: Ground-truth mask column.
        auto_resize: Whether to auto-resize prediction heatmaps to GT mask shape.

    Returns:
        Tuple of ``(updated_lf, aligned_pred_heatmap_col_name)``.
    """
    with_dims = lf.with_columns(
        _pred_h=pl.col(pred_col).list.len().cast(pl.Int64),
        _pred_w=pl.col(pred_col).list.first().list.len().cast(pl.Int64),
        _gt_h=pl.col(gt_mask_col).list.len().cast(pl.Int64),
        _gt_w=pl.col(gt_mask_col).list.first().list.len().cast(pl.Int64),
    ).with_columns(
        _shape_mismatch=(pl.col("_pred_h") != pl.col("_gt_h"))
        | (pl.col("_pred_w") != pl.col("_gt_w"))
    )

    mismatch_preview_df = (
        with_dims.filter(pl.col("_shape_mismatch"))
        .select("_pred_h", "_pred_w", "_gt_h", "_gt_w")
        .limit(5)
        .collect(engine="streaming")
    )
    has_mismatch = mismatch_preview_df.height > 0

    if not auto_resize:
        if has_mismatch:
            msg = (
                "Prediction heatmap and GT mask shapes differ. "
                "Enable `auto_resize=True` or pre-align shapes before metric computation. "
                f"Examples: {mismatch_preview_df.to_dicts()}"
            )
            raise ValueError(msg)
        return (
            with_dims.with_columns(pl.col(pred_col).alias("_pred_heatmap_aligned")),
            "_pred_heatmap_aligned",
        )

    if not has_mismatch:
        return (
            with_dims.with_columns(pl.col(pred_col).alias("_pred_heatmap_aligned")),
            "_pred_heatmap_aligned",
        )

    resize_pipe = (
        Pipeline()
        .source("list", dtype="f32")
        .resize(height=pl.col("_gt_h"), width=pl.col("_gt_w"))
    )
    return (
        with_dims.with_columns(
            _pred_heatmap_resized=pl.col(pred_col).cv.pipe(resize_pipe).sink("list")
        )
        .with_columns(
            # TODO: remove flattening once list resize preserves scalar channels for 2D input.
            _pred_heatmap_aligned=pl.col("_pred_heatmap_resized").list.eval(
                pl.element().list.eval(pl.element().list.first())
            )
        )
        .drop("_pred_heatmap_resized"),
        "_pred_heatmap_aligned",
    )


def _extract_contours_from_col(
    lf: pl.LazyFrame,
    *,
    source_col: str,
    threshold: float,
    min_area: float,
    output_col: str,
) -> pl.LazyFrame:
    """Extract contours from a buffer-like list column.

    Args:
        lf: Input lazy frame.
        source_col: Buffer column (nested list image-like).
        threshold: Binary threshold used prior to extraction.
        min_area: Minimum contour area.
        output_col: Name of output contour-set column.

    Returns:
        LazyFrame with ``output_col`` as a list of contours.
    """
    pipe_builder = Pipeline().source("list", dtype="f32").threshold(value=threshold)
    if min_area > 0.0:
        extract_pipe = pipe_builder.extract_contours(
            mode="external", method="simple", min_area=min_area
        )
    else:
        extract_pipe = pipe_builder.extract_contours(mode="external", method="simple")

    return lf.with_columns(
        pl.col(source_col)
        .cv.pipe(extract_pipe)
        .sink("native")
        .cast(_EXTRACTED_CONTOUR_SET_SCHEMA)
        .alias(output_col)
    )


def _score_contours_from_heatmap(
    lf: pl.LazyFrame,
    *,
    contour_col: str,
    heatmap_col: str,
    output_col: str = "_pred_scores",
) -> pl.LazyFrame:
    """Score contour sets from heatmaps using max interior value.

    Args:
        lf: Input lazy frame.
        contour_col: Contour-set column.
        heatmap_col: Heatmap column.
        output_col: Output score list column.

    Returns:
        LazyFrame with score column added.
    """
    score_pipe = (
        Pipeline()
        .source("list", dtype="f32")
        .label_reduce(
            contours=pl.col(contour_col), reduction="max", region_mode="interior"
        )
    )
    return lf.with_columns(
        pl.col(heatmap_col).cv.pipe(score_pipe).sink("native").alias(output_col)
    )


def _validate_match_alignment(
    matched_lf: pl.LazyFrame,
    *,
    image_id_col: str,
    pred_idx_col: str = "pred_idx",
    gt_idx_col: str = "gt_idx",
) -> None:
    """Raise if match payload lists are misaligned.

    Args:
        matched_lf: Lazy frame containing match results.
        image_id_col: Image identifier column.
        pred_idx_col: Prediction index list column.
        gt_idx_col: GT index list column.

    Raises:
        ValueError: If any row has mismatched list lengths.
    """
    mismatch = (
        matched_lf.with_columns(
            _pred_len=pl.col(pred_idx_col).list.len().fill_null(0),
            _gt_len=pl.col(gt_idx_col).list.len().fill_null(0),
        )
        .filter(pl.col("_pred_len") != pl.col("_gt_len"))
        .select(image_id_col, "_pred_len", "_gt_len")
        .limit(5)
        .collect(engine="streaming")
    )
    if mismatch.height > 0:
        raise ValueError(
            "Matched prediction-index and GT-index lists are misaligned. "
            f"Examples: {mismatch.to_dicts()}"
        )


def _explode_match_to_detections(
    image_level: pl.LazyFrame,
    *,
    image_id_col: str,
    scores_col: str,
    pred_idx_col: str,
    gt_idx_col: str,
    iou_col: str,
    class_id: str,
) -> pl.LazyFrame:
    """Explode per-image match results into per-detection rows.

    Args:
        image_level: One row per image with list columns.
        image_id_col: Image identifier column.
        scores_col: Score list column.
        pred_idx_col: Prediction index list column.
        gt_idx_col: GT index list column.
        iou_col: IoU list column.
        class_id: Class label to assign.

    Returns:
        Per-detection lazy frame with canonical column names.
    """
    det_base = (
        image_level.select(
            image_id_col,
            _scores=pl.col(scores_col),
            _det_ord=pl.int_ranges(0, pl.col(scores_col).list.len()),
        )
        .explode("_scores", "_det_ord")
        .with_columns(_det_ord=pl.col("_det_ord").cast(pl.UInt32))
    )
    match_pairs = (
        image_level.select(
            image_id_col,
            _det_ord=pl.col(pred_idx_col),
            _gt_idx=pl.col(gt_idx_col),
            _iou=pl.col(iou_col),
        )
        .explode("_det_ord", "_gt_idx", "_iou")
        .with_columns(_det_ord=pl.col("_det_ord").cast(pl.UInt32))
    )
    return det_base.join(match_pairs, on=[image_id_col, "_det_ord"], how="left").select(
        pl.col(image_id_col).alias(COL_IMAGE_ID),
        pl.lit(class_id).alias(COL_CLASS_ID),
        pl.col("_scores").alias(COL_SCORE),
        pl.col("_gt_idx").is_not_null().alias(COL_IS_TP),
        pl.col("_gt_idx").cast(pl.UInt32).alias(COL_GT_IDX),
        pl.col("_iou").fill_null(0.0).alias(COL_IOU),
        pl.col("_det_ord").alias(COL_DET_IDX),
    )


# ---------------------------------------------------------------------------
# ContourMatcher
# ---------------------------------------------------------------------------


class ContourMatcher:
    """Match detections from heatmaps + binary masks via contour extraction.

    This is the refactored version of the original ``prepare_detection_table``
    from ``_prepare.py``.  It extracts contours from both predictions and GT
    masks, scores predictions against the heatmap, and runs greedy IoU matching
    via ``.contour.match_detections()``.

    Args:
        iou_threshold: IoU threshold for TP matching.
        extraction_threshold: Threshold for contour extraction from heatmaps.
        min_contour_area: Minimum extracted contour area for predictions.
        auto_resize: Whether to resize heatmaps to mask shapes automatically.
        gt_min_contour_area: Separate min area for GT contours (defaults to
            ``min_contour_area``).
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        extraction_threshold: float = 0.1,
        min_contour_area: float = 0.0,
        auto_resize: bool = True,
        gt_min_contour_area: float | None = None,
    ) -> None:
        if not (0.0 < iou_threshold <= 1.0):
            raise ValueError("`iou_threshold` must be in (0, 1].")
        self._iou_threshold = iou_threshold
        self._extraction_threshold = extraction_threshold
        self._min_contour_area = min_contour_area
        self._auto_resize = auto_resize
        self._gt_min_contour_area = gt_min_contour_area

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
        """Produce a ``DetectionTable`` from heatmap + binary mask data.

        Args:
            data: Input frame with one image/sample per row.
            pred_col: Prediction heatmap column (nested ``List[List[Float64]]``).
            gt_col: Ground-truth binary mask column (same nesting).
            score_col: Unused for contour matching (scores are derived from
                heatmap peaks).
            class_col: Optional class label column for multi-class metrics.
            image_id_col: Optional image identifier column (defaults to row index).
            weight_col: Optional sample weight column.
            group_col: Optional grouping column.

        Returns:
            Validated ``DetectionTable``.
        """
        lf = to_lazy(data)
        schema_names = list(lf.collect_schema().names())
        ensure_columns_exist(schema_names, [pred_col, gt_col])
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
            prepared = lf.with_row_index(name="_metric_row_idx").with_columns(
                pl.col("_metric_row_idx").cast(pl.String).alias(COL_IMAGE_ID)
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

        # Shape validation / resize
        prepared, aligned_pred_col = _validate_or_resize_shapes(
            prepared,
            pred_col=pred_col,
            gt_mask_col=gt_col,
            auto_resize=self._auto_resize,
        )

        # Extract contours from predictions and GT
        prepared = _extract_contours_from_col(
            prepared,
            source_col=aligned_pred_col,
            threshold=self._extraction_threshold,
            min_area=self._min_contour_area,
            output_col="_pred_contours",
        )
        gt_area = (
            self._gt_min_contour_area
            if self._gt_min_contour_area is not None
            else self._min_contour_area
        )
        prepared = _extract_contours_from_col(
            prepared,
            source_col=gt_col,
            threshold=0.5,
            min_area=gt_area,
            output_col="_gt_contours",
        )

        # Score predictions
        prepared = _score_contours_from_heatmap(
            prepared,
            contour_col="_pred_contours",
            heatmap_col=aligned_pred_col,
            output_col="_pred_scores",
        )

        # Run matching
        prepared = prepared.with_columns(
            _match=pl.col("_pred_contours").contour.match_detections(
                pl.col("_gt_contours"),
                threshold=self._iou_threshold,
                scores=pl.col("_pred_scores"),
            ),
            _n_gts=pl.col("_gt_contours").list.len().fill_null(0).cast(pl.Int64),
        )

        # Build image-level frame with match results
        image_level = prepared.select(
            COL_IMAGE_ID,
            COL_WEIGHT,
            "_pred_scores",
            "_n_gts",
            pred_idx=pl.col("_match").struct.field("pred_idx"),
            gt_idx=pl.col("_match").struct.field("gt_idx"),
            iou=pl.col("_match").struct.field("iou"),
            _gt_label=(pl.col("_gt_contours").list.len().fill_null(0) > 0),
            **(
                {COL_CLASS_ID: pl.col(class_col).cast(pl.String)}
                if class_col is not None
                else {COL_CLASS_ID: pl.lit(DEFAULT_CLASS)}
            ),
            **(
                {COL_GROUP_ID: pl.col(group_col).cast(pl.String)}
                if group_col is not None
                else {}
            ),
        )

        # Materialize so we can validate and split
        image_level_df = image_level.collect(engine="streaming")

        if image_level_df.height == 0:
            return _empty_detection_table()

        # Validate match alignment
        _validate_match_alignment(
            image_level_df.lazy(),
            image_id_col=COL_IMAGE_ID,
        )

        # Explode into per-detection rows and drop spurious null-score
        # rows that arise when contour extraction returns null for images
        # with no predictions.
        detections_lf = _explode_match_to_detections(
            image_level_df.lazy(),
            image_id_col=COL_IMAGE_ID,
            scores_col="_pred_scores",
            pred_idx_col="pred_idx",
            gt_idx_col="gt_idx",
            iou_col="iou",
            class_id=DEFAULT_CLASS,
        ).filter(pl.col(COL_SCORE).is_not_null())

        # Handle per-class explode if class_col was provided
        if class_col is not None:
            detections_lf = (
                _explode_match_to_detections(
                    image_level_df.lazy(),
                    image_id_col=COL_IMAGE_ID,
                    scores_col="_pred_scores",
                    pred_idx_col="pred_idx",
                    gt_idx_col="gt_idx",
                    iou_col="iou",
                    class_id=DEFAULT_CLASS,
                )
                .filter(pl.col(COL_SCORE).is_not_null())
                .with_columns(pl.col(COL_IMAGE_ID))
            )
            # Re-join with the class_id from the image-level frame
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


def _empty_detection_table() -> DetectionTable:
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
