"""Input preparation helpers for heatmap/mask detection metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from ..pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Sequence


POINT_STRUCT = pl.Struct([pl.Field("x", pl.Float64), pl.Field("y", pl.Float64)])
EXTRACTED_CONTOUR_SCHEMA = pl.Struct(
    [
        pl.Field("exterior", pl.List(POINT_STRUCT)),
        pl.Field("holes", pl.List(pl.List(POINT_STRUCT))),
        pl.Field("is_closed", pl.Boolean),
    ]
)
EXTRACTED_CONTOUR_SET_SCHEMA = pl.List(EXTRACTED_CONTOUR_SCHEMA)


def to_lazy(data: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Normalize eager/lazy inputs to LazyFrame."""
    return data.lazy() if isinstance(data, pl.DataFrame) else data


def validate_or_resize_shapes(
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
            mismatch_preview = mismatch_preview_df.to_dicts()
            msg = (
                "Prediction heatmap and GT mask shapes differ. "
                "Enable `auto_resize=True` or pre-align shapes before metric computation. "
                f"Examples: {mismatch_preview}"
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


def extract_contours_from_col(
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
    if min_area > 0.0:
        extract_pipe = (
            Pipeline()
            .source("list", dtype="f32")
            .threshold(value=threshold)
            .extract_contours(mode="external", method="simple", min_area=min_area)
        )
    else:
        extract_pipe = (
            Pipeline()
            .source("list", dtype="f32")
            .threshold(value=threshold)
            .extract_contours(mode="external", method="simple")
        )

    return lf.with_columns(
        pl.col(source_col)
        .cv.pipe(extract_pipe)
        .sink("native")
        .cast(EXTRACTED_CONTOUR_SET_SCHEMA)
        .alias(output_col)
    )


def score_contours_from_heatmap(
    lf: pl.LazyFrame,
    *,
    contour_col: str,
    heatmap_col: str,
    output_col: str = "_pred_scores",
) -> pl.LazyFrame:
    """Score contour sets from heatmaps using max interior value."""
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


def ensure_column_exists(columns: Sequence[str], name: str) -> None:
    """Validate that a required column exists."""
    if name not in columns:
        raise ValueError(f"Required column `{name}` not found.")


def prepare_detection_table(
    data: pl.LazyFrame | pl.DataFrame,
    *,
    pred_col: str,
    gt_mask_col: str,
    gt_label_col: str | None,
    image_id_col: str | None,
    weight_col: str | None,
    stratify_col: str | None,
    iou_threshold: float,
    extraction_threshold: float,
    min_contour_area: float,
    gt_min_contour_area: float | None = None,
    auto_resize: bool,
) -> pl.LazyFrame:
    """Build a shared lazy detection table for FROC/LROC analyzers.

    The returned table contains one row per image with aligned prediction scores,
    match indices, and metadata needed for threshold sweeps.
    """
    lf = to_lazy(data)
    schema_names = list(lf.collect_schema().names())
    ensure_column_exists(schema_names, pred_col)
    ensure_column_exists(schema_names, gt_mask_col)
    if gt_label_col is not None:
        ensure_column_exists(schema_names, gt_label_col)
    if image_id_col is not None:
        ensure_column_exists(schema_names, image_id_col)
    if weight_col is not None:
        ensure_column_exists(schema_names, weight_col)
    if stratify_col is not None:
        ensure_column_exists(schema_names, stratify_col)

    resolved_image_id_col = image_id_col or "_metric_image_id"
    if image_id_col is None:
        prepared = lf.with_row_index(name=resolved_image_id_col)
    else:
        prepared = lf
    prepared = prepared.with_columns(
        pl.col(resolved_image_id_col).cast(pl.String).alias("_image_id"),
        (
            pl.col(weight_col).cast(pl.Float64)
            if weight_col is not None
            else pl.lit(1.0, dtype=pl.Float64)
        ).alias("_weight"),
    )

    prepared, aligned_pred_col = validate_or_resize_shapes(
        prepared,
        pred_col=pred_col,
        gt_mask_col=gt_mask_col,
        auto_resize=auto_resize,
    )
    prepared = extract_contours_from_col(
        prepared,
        source_col=aligned_pred_col,
        threshold=extraction_threshold,
        min_area=min_contour_area,
        output_col="_pred_contours",
    )
    prepared = extract_contours_from_col(
        prepared,
        source_col=gt_mask_col,
        threshold=0.5,
        min_area=(
            gt_min_contour_area if gt_min_contour_area is not None else min_contour_area
        ),
        output_col="_gt_contours",
    )
    prepared = score_contours_from_heatmap(
        prepared,
        contour_col="_pred_contours",
        heatmap_col=aligned_pred_col,
        output_col="_pred_scores",
    )
    return prepared.with_columns(
        _match=pl.col("_pred_contours").contour.match_detections(
            pl.col("_gt_contours"),
            threshold=iou_threshold,
            scores=pl.col("_pred_scores"),
        ),
        _n_gts=pl.col("_gt_contours").list.len().cast(pl.Int64),
        _gt_label=(
            pl.col(gt_label_col).cast(pl.Boolean)
            if gt_label_col is not None
            else (pl.col("_gt_contours").list.len() > 0)
        ),
        _stratify=(
            pl.col(stratify_col).cast(pl.String)
            if stratify_col is not None
            else (
                pl.col(gt_label_col).cast(pl.String)
                if gt_label_col is not None
                else (pl.col("_gt_contours").list.len() > 0).cast(pl.String)
            )
        ),
    ).select(
        "_image_id",
        "_pred_scores",
        "_match",
        "_n_gts",
        "_gt_label",
        "_stratify",
        "_weight",
        "_gt_contours",
    )
