"""One call from predictions and ground truth to a report.

:func:`evaluate_detections` (object tables: boxes, polygons, instance masks)
and :func:`evaluate_heatmaps` (dense predictions against masks) match with
polars-cv and return the generic :class:`DetectionReport`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from polars_cv.geometry import BoxFormat

from .._inputs import GT_ROW, PRED_ROW, group_objects
from .._reports import COCO_IOU_THRESHOLDS, DetectionReport, _materialized
from .._statistics import FROC_RATES
from .._types import COL_CLASS_ID, COL_IMAGE_ID
from ._contour import ContourMatcher
from ._match import match_detections

if TYPE_CHECKING:
    from .._auc import Extrapolate
    from .._metrics._precision_recall import APInterpolation


def evaluate_detections(
    predictions: pl.DataFrame | pl.LazyFrame,
    ground_truth: pl.DataFrame | pl.LazyFrame,
    *,
    geometry: str | Sequence[str] = "bbox",
    box_format: str | BoxFormat | None = None,
    image_id: str = COL_IMAGE_ID,
    class_id: str | None = COL_CLASS_ID,
    score: str = "score",
    iou_thresholds: Literal["coco"] | float | Sequence[float] = "coco",
    interpolation: APInterpolation | None = None,
    max_detections: int | None | Literal["coco"] = "coco",
    images: Iterable[str] | pl.Series | pl.DataFrame | pl.LazyFrame | None = None,
    weight: str | None = None,
    group: str | None = None,
    fp_rates: Sequence[float] | None = None,
    extrapolate: Extrapolate = "none",
) -> DetectionReport:
    """Evaluate object detections (boxes, polygons or instance masks).

    The one-call form of :func:`match_detections` plus the summary metrics,
    COCO-style by default::

        report = evaluate_detections(preds, gts, box_format="xyxy")
        report.summary            # map, map_50, map_75, mar
        report.per_class          # AP per class
        report.ci("map")          # bootstrap interval
        report.matches(0.5)       # each prediction's TP/FP, for error analysis

    With the defaults (``iou_thresholds="coco"``) matching is redone at each
    IoU threshold 0.50:0.05:0.95, AP is COCO's 101-point interpolation, each
    image keeps its 100 highest-scoring predictions per class, and a class
    without ground truth is left out of the means — COCO's evaluation, except
    that it has no area ranges and no crowd/ignore regions (and ties between
    equal scores form one PR point rather than following input order).

    Args:
        predictions: One row per predicted object (``image_id``, geometry,
            ``score``, ``class_id``).
        ground_truth: One row per ground-truth object.
        geometry: The geometry column (or four coordinate columns); see
            :func:`match_detections` for the accepted forms.
        box_format: ``"xyxy"``, ``"xywh"`` or ``"cxcywh"`` for boxes given
            as four numbers.
        image_id: The image identifier column.
        class_id: The class column, or ``None`` for one class.
        score: The prediction confidence column.
        iou_thresholds: ``"coco"``, one threshold (e.g. ``0.5``, Pascal VOC)
            or several.
        interpolation: AP interpolation; ``None`` is ``"101_point"`` for
            ``"coco"`` and ``"all_points"`` otherwise.
        max_detections: Per (image, class) cap on predictions; ``"coco"`` is
            100 with ``iou_thresholds="coco"`` and no cap otherwise.
        images: The image population (see :func:`match_detections`).
        weight: A per-image weight column of the ``images`` frame.
        group: A per-image subgroup column of the ``images`` frame.
        fp_rates: FROC operating points to summarise (``FROC_RATES`` is the
            radiology set), or ``None``.
        extrapolate: How FROC curves are read past their last point.

    Returns:
        A :class:`DetectionReport`.
    """
    coco = isinstance(iou_thresholds, str)
    if coco and iou_thresholds != "coco":
        raise ValueError(
            f"iou_thresholds must be 'coco' or numbers, got {iou_thresholds!r}"
        )
    thresholds = COCO_IOU_THRESHOLDS if coco else iou_thresholds
    cap = (100 if coco else None) if max_detections == "coco" else max_detections
    interp: APInterpolation = interpolation or ("101_point" if coco else "all_points")
    table = match_detections(
        predictions,
        ground_truth,
        geometry=geometry,
        box_format=box_format,
        image_id=image_id,
        class_id=class_id,
        score=score,
        iou_thresholds=thresholds,  # type: ignore[arg-type]
        max_detections=cap,
        images=images,
        weight=weight,
        group=group,
    )
    geo = geometry if isinstance(geometry, str) else next(iter(geometry))
    objects = group_objects(
        predictions,
        ground_truth,
        geometry=geo,
        image_id=image_id,
        class_id=class_id,
        score=score,
        images=images,
        max_detections=cap,
    ).select(COL_IMAGE_ID, COL_CLASS_ID, PRED_ROW, GT_ROW)
    return DetectionReport(
        table=_materialized(table),
        interpolation=interp,
        fp_rates=tuple(fp_rates) if fp_rates else None,
        extrapolate=extrapolate,
        _objects=objects,
        _predictions=predictions.lazy(),
        _ground_truth=ground_truth.lazy(),
    )


def evaluate_heatmaps(
    data: pl.DataFrame | pl.LazyFrame,
    *,
    heatmap: str,
    gt: str,
    image_id: str | None = None,
    class_id: str | None = None,
    iou_threshold: float | Sequence[float] = 0.5,
    fp_rates: Sequence[float] = FROC_RATES,
    extrapolate: Extrapolate = "none",
    interpolation: APInterpolation = "all_points",
    weight: str | None = None,
    group: str | None = None,
    **matcher_options: Any,
) -> DetectionReport:
    """Evaluate dense predictions (heatmaps, probability maps) against masks.

    One row per image: each heatmap is thresholded into candidate regions,
    each region scored from the heatmap, and the regions matched to the
    ground-truth mask's regions by :class:`ContourMatcher` (any of its options
    can be passed through, e.g. ``match_by="coverage"``,
    ``extraction_threshold=0.3``, ``score_reduction="mean"``). The report
    summarises FROC — sensitivity at each of ``fp_rates`` and their mean, the
    CPM — beside AP::

        report = evaluate_heatmaps(df, heatmap="prob", gt="mask")
        report.summary              # map, mar, sensitivity@…, cpm
        report.ci("cpm")

    Args:
        data: One row per image.
        heatmap: The prediction column (any format a polars-cv source reads).
        gt: The ground-truth mask (or contour) column.
        image_id: An image identifier column (default: the row index).
        class_id: A class column (one row per image and class).
        iou_threshold: The matching threshold, or several.
        fp_rates: FROC operating points (false positives per image).
        extrapolate: How the FROC curve is read past its last point
            (``"flat"`` is LUNA16's convention).
        interpolation: AP interpolation.
        weight: A per-image weight column of ``data``.
        group: A per-image subgroup column of ``data``.
        **matcher_options: :class:`ContourMatcher` options.

    Returns:
        A :class:`DetectionReport`.
    """
    # Unknown options are refused by ContourMatcher's own signature.
    matcher = ContourMatcher(iou_threshold=iou_threshold, **matcher_options)
    table = matcher.match(
        data,
        pred_col=heatmap,
        gt_col=gt,
        class_col=class_id,
        image_id_col=image_id,
        weight_col=weight,
        group_col=group,
    )
    return DetectionReport(
        table=_materialized(table),
        interpolation=interpolation,
        fp_rates=tuple(fp_rates),
        extrapolate=extrapolate,
    )
