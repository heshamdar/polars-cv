"""Semantic segmentation measured with polars-cv: one graph per row.

:func:`segmentation_measures` is one expression giving, per row, the overlap
(Dice, IoU) and boundary (ASSD, Hausdorff, HD95) agreement of a predicted and
a reference mask -- the MONAI set -- computed in one graph per row: both masks
binarised once, their pixel counts and their outlines (hole borders included)
read from the same decode. :func:`evaluate_segmentation` collects it into the
generic :class:`SegmentationReport`.

Boundaries are the exact pixel-edge outlines (``extract_contours``), measured
point-to-edge (``.contour.set_boundary_distances``): every sample of one mask's
boundary to the nearest edge of the other's. With ``frame="image"`` boundary
on the image edge is not measured -- a structure cut off by the field of view
has a boundary there that is not anatomy, and that is not an error.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

from polars_cv import Pipeline

from .._segmentation import SegmentationReport


def segmentation_measures(
    pred: str | pl.Expr,
    target: str | pl.Expr,
    *,
    threshold: float = 0.5,
    frame: Literal["image"] | None = "image",
    sample_step: float | None = 1.0,
    spacing: tuple[float, float] | None = None,
) -> pl.Expr:
    """Overlap and boundary agreement of two masks, one struct per row.

    Args:
        pred: The predicted mask column (any format a polars-cv source reads:
            encoded image bytes, a nested ``List``/``Array``, a VIEW blob).
        target: The reference mask column, the same size.
        threshold: Pixels above it are foreground (0.5 suits 0/1 and 0/255
            masks and probability maps alike).
        frame: ``"image"`` leaves boundary on the image edge unmeasured;
            ``None`` measures all of it.
        sample_step: Spacing of the boundary samples along each edge, in
            output units (pixels, or ``spacing`` units); ``None`` samples the
            outline vertices only.
        spacing: ``(row, column)`` pixel spacing (e.g. mm): boundary distances
            in those units. Overlap is unitless either way.

    Returns:
        ``Struct{dice, iou, assd, hd, hd95, mean_pred_to_target,
        mean_target_to_pred}`` (Float64). Two empty masks agree perfectly
        (Dice and IoU 1); one empty mask has Dice and IoU 0. Boundary
        distances are null whenever either mask is empty (no boundary to
        measure to) — as MONAI's are NaN.
    """
    if not (sample_step is None or sample_step > 0):
        raise ValueError(f"sample_step must be > 0 or None, got {sample_step}")
    if frame not in ("image", None):
        raise ValueError(f"frame must be 'image' or None, got {frame!r}")
    if spacing is not None and not (len(spacing) == 2 and all(s > 0 for s in spacing)):
        raise ValueError(f"spacing must be two positive numbers, got {spacing}")

    def binary(col: str | pl.Expr):
        expr = pl.col(col) if isinstance(col, str) else col
        return expr.cv.pipe(Pipeline().source("auto").threshold(value=threshold))  # ty: ignore[missing-attribute]

    p, t = binary(pred), binary(target)
    count = Pipeline().reduce_sum()
    outline = Pipeline().extract_contours(mode="all")
    parts = {
        "_i": p.bitwise_and(t).pipe(count),
        "_ps": p.pipe(count),
        "_ts": t.pipe(count),
        "_pc": p.pipe(outline),
        "_tc": t.pipe(outline),
        "_shape": t.pipe(Pipeline().extract_shape()),
    }
    named = [e.alias(k) for k, e in parts.items()]
    graph = named[0].merge_pipe(*named[1:]).sink({k: "native" for k in parts})

    i, ps, ts = (graph.struct.field(k) for k in ("_i", "_ps", "_ts"))
    union = ps + ts - i
    dice = pl.when(ps + ts == 0).then(1.0).otherwise(2.0 * i / (ps + ts))
    iou = pl.when(union == 0).then(1.0).otherwise(i / union)

    sy, sx = spacing or (1.0, 1.0)
    pc, tc = graph.struct.field("_pc"), graph.struct.field("_tc")
    if spacing is not None:
        pc = pc.contour.scale(sx, sy, origin="origin")
        tc = tc.contour.scale(sx, sy, origin="origin")
    shape = graph.struct.field("_shape")
    box = (
        pl.struct(
            pl.lit(0.0).alias("x"),
            pl.lit(0.0).alias("y"),
            (shape.list.get(1).cast(pl.Float64) * sx).alias("width"),
            (shape.list.get(0).cast(pl.Float64) * sy).alias("height"),
        )
        if frame == "image"
        else None
    )
    d = pc.contour.set_boundary_distances(tc, sample_step=sample_step, frame=box)
    return pl.struct(
        dice.alias("dice"),
        iou.alias("iou"),
        d.struct.field("assd").alias("assd"),
        d.struct.field("hd").alias("hd"),
        d.struct.field("hd95").alias("hd95"),
        d.struct.field("mean_a_to_b").alias("mean_pred_to_target"),
        d.struct.field("mean_b_to_a").alias("mean_target_to_pred"),
    ).alias("segmentation")


def evaluate_segmentation(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    pred: str,
    target: str,
    image_id: str | None = None,
    threshold: float = 0.5,
    frame: Literal["image"] | None = "image",
    sample_step: float | None = 1.0,
    spacing: tuple[float, float] | None = None,
) -> SegmentationReport:
    """Evaluate semantic segmentation masks: Dice, IoU, ASSD, HD, HD95.

    ::

        report = evaluate_segmentation(df, pred="pred_mask", target="gt_mask")
        report.summary          # mean of each measure, with undefined counts
        report.per_image        # one row per image
        report.ci("dice")

    Args:
        data: One row per image.
        pred: The predicted mask column.
        target: The reference mask column.
        image_id: An image identifier column (default: the row index).
        threshold: Foreground threshold for both masks.
        frame: ``"image"`` leaves boundary on the image edge unmeasured.
        sample_step: Boundary sample spacing (see :func:`segmentation_measures`).
        spacing: ``(row, column)`` pixel spacing for physical distances.

    Returns:
        A :class:`SegmentationReport`.
    """
    lf = data.lazy()
    if image_id is None:
        # A row index streams; an unwindowed `int_range(pl.len())` does not.
        lf = lf.with_row_index("_seg_row")
        image_id = "_seg_row"
    ids = pl.col(image_id).cast(pl.String)
    per_image = (
        lf.select(
            ids.alias("image_id"),
            segmentation_measures(
                pred,
                target,
                threshold=threshold,
                frame=frame,
                sample_step=sample_step,
                spacing=spacing,
            ),
        )
        .unnest("segmentation")
        .collect()
    )
    return SegmentationReport(per_image=per_image)
