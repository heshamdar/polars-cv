"""Pre-matched adapter: pass-through for already-matched detection data."""

from __future__ import annotations

import warnings
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

if TYPE_CHECKING:
    pass


class PreMatchedAdapter:
    """Adapter for data that already has per-detection TP/FP assignments.

    Expects a flat table where each row is one detection with at minimum:

    * ``score`` (float) — confidence score
    * ``is_tp`` (bool) — whether this detection is a true positive

    Plus per-image metadata either via ``image_meta`` or derived from the
    detection rows (``n_gts`` column / defaults).

    .. warning::

        Without ``image_meta``, the adapter derives the image population
        by grouping the detection frame. Images with no detections are
        then missing from ``image_metadata``, which silently deletes the
        negative population and inflates recall / FP-per-image. Prefer
        passing an explicit ``image_meta`` that covers the full evaluation
        population whenever any image may carry zero detections.
    """

    def match(
        self,
        data: pl.LazyFrame | pl.DataFrame,
        *,
        pred_col: str = "score",
        gt_col: str = "is_tp",
        score_col: str | None = None,
        class_col: str | None = None,
        image_id_col: str | None = None,
        weight_col: str | None = None,
        group_col: str | None = None,
        n_gts_col: str | None = None,
        gt_label_col: str | None = None,
        iou_col: str | None = None,
        det_idx_col: str | None = None,
        image_meta: pl.LazyFrame | pl.DataFrame | None = None,
    ) -> DetectionTable:
        """Wrap pre-matched data into a ``DetectionTable``.

        Args:
            data: Input frame with one row per detection.
            pred_col: Column with confidence scores (aliased to ``score``).
            gt_col: Column with TP flag (aliased to ``is_tp``).
            score_col: Alias for ``pred_col`` (takes precedence if both set).
            class_col: Optional class label column.
            image_id_col: Image identifier column (required, or row index used).
            weight_col: Optional sample weight column.
            group_col: Optional grouping column.
            n_gts_col: Column with per-image GT count.
            gt_label_col: Column with per-image positive/negative label.
            iou_col: Optional column with per-detection IoU values.
            det_idx_col: Optional column with detection index within image.
            image_meta: Optional per-image (or per-image-class) frame that
                defines the evaluation population. Must use the canonical
                column names ``image_id`` and ``n_gts``; ``class_id``,
                ``weight``, ``gt_label`` and ``group_id`` are filled with
                defaults when absent. When provided this is the *sole* source
                of ``image_metadata`` — images with zero detections are
                retained — so it may not be combined with the per-image
                column arguments below. When omitted, metadata is derived
                from the detection rows and a ``UserWarning`` is emitted.

        Returns:
            Validated ``DetectionTable``.

        Raises:
            ValueError: If ``image_meta`` is combined with ``n_gts_col``,
                ``weight_col``, ``gt_label_col`` or ``group_col``, or if a
                named column is missing from ``data``.
        """
        if image_meta is not None:
            # image_meta is the whole population, so the per-image column
            # arguments have nothing to read: they only ever described how to
            # derive metadata *from the detection frame*. Accepting and
            # ignoring them would let a caller believe a weight column was
            # honoured when it was not.
            conflicting = {
                "n_gts_col": n_gts_col,
                "weight_col": weight_col,
                "gt_label_col": gt_label_col,
                "group_col": group_col,
            }
            supplied = sorted(name for name, val in conflicting.items() if val)
            if supplied:
                raise ValueError(
                    f"image_meta cannot be combined with {', '.join(supplied)}. "
                    "image_meta is the sole source of image_metadata, so those "
                    "arguments — which describe how to derive metadata from the "
                    "detection frame — would be silently ignored. Put the "
                    "per-image values in image_meta under the canonical names "
                    "(n_gts, weight, gt_label, group_id) instead."
                )

        lf = to_lazy(data)
        schema_names = list(lf.collect_schema().names())

        resolved_score = score_col or pred_col
        resolved_tp = gt_col
        ensure_columns_exist(schema_names, [resolved_score, resolved_tp])

        # Image ID
        if image_id_col is not None:
            ensure_columns_exist(schema_names, [image_id_col])
            lf = lf.with_columns(
                pl.col(image_id_col).cast(pl.String).alias(COL_IMAGE_ID)
            )
        else:
            lf = lf.with_row_index(name="_row_idx").with_columns(
                pl.col("_row_idx").cast(pl.String).alias(COL_IMAGE_ID)
            )

        # Build detection columns
        det_exprs: list[pl.Expr] = [
            pl.col(COL_IMAGE_ID),
            (
                pl.col(class_col).cast(pl.String).alias(COL_CLASS_ID)
                if class_col is not None
                else pl.lit(DEFAULT_CLASS).alias(COL_CLASS_ID)
            ),
            pl.col(resolved_score).cast(pl.Float64).alias(COL_SCORE),
            pl.col(resolved_tp).cast(pl.Boolean).alias(COL_IS_TP),
            (
                pl.col(iou_col).cast(pl.Float64).alias(COL_IOU)
                if iou_col is not None and iou_col in schema_names
                else pl.lit(0.0, dtype=pl.Float64).alias(COL_IOU)
            ),
        ]

        if det_idx_col is not None and det_idx_col in schema_names:
            det_exprs.append(pl.col(det_idx_col).cast(pl.UInt32).alias(COL_DET_IDX))
        else:
            # Auto-assign det_idx as row ordinal within each image
            det_exprs.append(pl.lit(0, dtype=pl.UInt32).alias(COL_DET_IDX))

        # GT index: not available for pre-matched data
        det_exprs.append(
            pl.when(pl.col(resolved_tp).cast(pl.Boolean))
            .then(pl.lit(0, dtype=pl.UInt32))
            .otherwise(pl.lit(None, dtype=pl.UInt32))
            .alias(COL_GT_IDX)
        )

        detections_lf = lf.select(det_exprs)

        # If we used a placeholder det_idx, assign proper ordinals
        if det_idx_col is None or det_idx_col not in schema_names:
            detections_lf = detections_lf.with_columns(
                pl.int_range(0, pl.len(), dtype=pl.UInt32)
                .over(COL_IMAGE_ID, COL_CLASS_ID)
                .alias(COL_DET_IDX)
            )

        if image_meta is not None:
            meta_lf = _canonicalize_image_meta(image_meta)
        else:
            warnings.warn(
                "PreMatchedAdapter.match() was called without image_meta. "
                "Image metadata is derived from the detection frame, so any "
                "image with zero detections is dropped from the evaluation "
                "population (inflating recall and FP-per-image). Pass an "
                "explicit image_meta covering the full population to avoid "
                "this.",
                UserWarning,
                stacklevel=2,
            )
            # Build image metadata from the enriched input (lf), which retains
            # all original columns alongside the canonical ones we added.
            enriched_lf = lf.with_columns(
                pl.col(resolved_tp).cast(pl.Boolean).alias(COL_IS_TP),
                *(
                    [pl.col(class_col).cast(pl.String).alias(COL_CLASS_ID)]
                    if class_col is not None
                    else [pl.lit(DEFAULT_CLASS).alias(COL_CLASS_ID)]
                ),
            )

            meta_agg: list[pl.Expr] = []

            if n_gts_col is not None and n_gts_col in schema_names:
                meta_agg.append(
                    pl.col(n_gts_col).first().cast(pl.Int64).alias(COL_N_GTS)
                )
            else:
                meta_agg.append(pl.col(COL_IS_TP).sum().cast(pl.Int64).alias(COL_N_GTS))

            if weight_col is not None and weight_col in schema_names:
                meta_agg.append(
                    pl.col(weight_col).first().cast(pl.Float64).alias(COL_WEIGHT)
                )
            else:
                meta_agg.append(pl.lit(1.0, dtype=pl.Float64).alias(COL_WEIGHT))

            if gt_label_col is not None and gt_label_col in schema_names:
                meta_agg.append(
                    pl.col(gt_label_col).first().cast(pl.Boolean).alias(COL_GT_LABEL)
                )
            else:
                meta_agg.append((pl.col(COL_IS_TP).any()).alias(COL_GT_LABEL))

            meta_lf = enriched_lf.group_by(COL_IMAGE_ID, COL_CLASS_ID).agg(meta_agg)

            if group_col is not None and group_col in list(lf.collect_schema().names()):
                group_map = lf.select(
                    pl.col(image_id_col or COL_IMAGE_ID)
                    .cast(pl.String)
                    .alias(COL_IMAGE_ID),
                    pl.col(group_col).cast(pl.String).alias(COL_GROUP_ID),
                ).unique()
                meta_lf = meta_lf.join(group_map, on=COL_IMAGE_ID, how="left")

        return DetectionTable.from_matched(detections_lf, meta_lf)


def _canonicalize_image_meta(
    image_meta: pl.LazyFrame | pl.DataFrame,
) -> pl.LazyFrame:
    """Normalize a caller-supplied image population to canonical columns.

    Args:
        image_meta: Per-image (or per-image-class) frame. Must contain
            ``image_id`` and ``n_gts``; ``class_id``, ``weight``, and
            ``gt_label`` are filled with defaults when absent.

    Returns:
        LazyFrame with the columns ``IMAGE_META_SCHEMA`` declares.

    Raises:
        ValueError: If ``image_id`` or ``n_gts`` is missing.
    """
    meta_lf = to_lazy(image_meta)
    schema_names = list(meta_lf.collect_schema().names())
    ensure_columns_exist(schema_names, [COL_IMAGE_ID, COL_N_GTS])

    select_exprs: list[pl.Expr] = [
        pl.col(COL_IMAGE_ID).cast(pl.String).alias(COL_IMAGE_ID),
        (
            pl.col(COL_CLASS_ID).cast(pl.String).alias(COL_CLASS_ID)
            if COL_CLASS_ID in schema_names
            else pl.lit(DEFAULT_CLASS).alias(COL_CLASS_ID)
        ),
        pl.col(COL_N_GTS).cast(pl.Int64).alias(COL_N_GTS),
        (
            pl.col(COL_WEIGHT).cast(pl.Float64).alias(COL_WEIGHT)
            if COL_WEIGHT in schema_names
            else pl.lit(1.0, dtype=pl.Float64).alias(COL_WEIGHT)
        ),
        (
            pl.col(COL_GT_LABEL).cast(pl.Boolean).alias(COL_GT_LABEL)
            if COL_GT_LABEL in schema_names
            else (pl.col(COL_N_GTS).cast(pl.Int64) > 0).alias(COL_GT_LABEL)
        ),
    ]
    if COL_GROUP_ID in schema_names:
        select_exprs.append(pl.col(COL_GROUP_ID).cast(pl.String).alias(COL_GROUP_ID))

    return meta_lf.select(select_exprs)
