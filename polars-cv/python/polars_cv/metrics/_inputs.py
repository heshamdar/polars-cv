"""Object-level inputs: long tables of predictions and ground truth.

Detections usually arrive one row per object — a model's output, a COCO
annotation file flattened, a CSV export — while the matchers read one row per
(image, class) holding lists. :func:`group_objects` is the one way between the
two, and :func:`match_detections` is the one entry point from object tables to
a :class:`DetectionTable` for any geometry: boxes in any layout, polygons
(contours) or per-instance masks.

The population is never inferred from one side alone. An image or class that
appears only among the predictions (false positives on an empty image) or
only among the ground truth (missed objects) is a row of the evaluation, with
an empty list on the other side; images with neither come from ``images=``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import polars as pl

from ..geometry.coords import BoxFormat, bbox_from_coords
from ..geometry.schemas import BBOX_SCHEMA, CONTOUR_SCHEMA
from ._types import COL_CLASS_ID, COL_IMAGE_ID, DEFAULT_CLASS, to_lazy

if TYPE_CHECKING:
    from ._types import DetectionTable

#: The column names of the per-(image, class) frame ``group_objects`` returns.
PRED, PRED_SCORE, PRED_ROW, GT, GT_ROW = (
    "pred",
    "pred_score",
    "pred_row",
    "gt",
    "gt_row",
)
#: The source-row index ``group_objects`` gives each object of its inputs.
_ROW = "_object_row"

Geometry = str | Sequence[str]


def group_objects(
    predictions: pl.DataFrame | pl.LazyFrame,
    ground_truth: pl.DataFrame | pl.LazyFrame,
    *,
    geometry: str,
    image_id: str = COL_IMAGE_ID,
    class_id: str | None = COL_CLASS_ID,
    score: str = "score",
    images: Iterable[str] | pl.Series | pl.DataFrame | pl.LazyFrame | None = None,
    max_detections: int | None = None,
) -> pl.LazyFrame:
    """Long object tables → one row per (image, class) with object lists.

    Args:
        predictions: One row per predicted object: ``image_id``, ``geometry``,
            ``score`` and (unless ``class_id=None``) ``class_id`` columns.
        ground_truth: One row per ground-truth object: ``image_id``,
            ``geometry`` and ``class_id``.
        geometry: The geometry column, the same name in both tables.
        image_id: The image identifier column.
        class_id: The class column, or ``None`` for one class.
        score: The prediction confidence column.
        images: The image population: ids (or a frame with an ``image_id``
            column). Images absent from both tables are evaluated as empty
            (their predictions none, their truth none). ``None`` uses the
            images in either table.
        max_detections: Keep only each (image, class)'s highest-scoring
            predictions — COCO's ``maxDets`` (100).

    Returns:
        A lazy frame with ``image_id`` (String), ``class_id`` (String),
        ``pred`` / ``gt`` (lists of geometry), ``pred_score`` (List[Float64],
        aligned with ``pred``, highest first) and ``pred_row`` / ``gt_row``
        (each object's row index in its input table). Every image of the
        population is crossed with every class of either table.
    """
    if max_detections is not None and max_detections < 1:
        raise ValueError(f"`max_detections` must be >= 1, got {max_detections}.")
    preds = to_lazy(predictions)
    gts = to_lazy(ground_truth)
    _require(
        preds,
        [image_id, geometry, score, *([class_id] if class_id else [])],
        "predictions",
    )
    _require(
        gts, [image_id, geometry, *([class_id] if class_id else [])], "ground_truth"
    )

    def keyed(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.with_row_index(_ROW).with_columns(
            pl.col(image_id).cast(pl.String).alias(COL_IMAGE_ID),
            (
                pl.col(class_id).cast(pl.String) if class_id else pl.lit(DEFAULT_CLASS)
            ).alias(COL_CLASS_ID),
        )

    keys = [COL_IMAGE_ID, COL_CLASS_ID]
    p, g = keyed(preds), keyed(gts)
    ranked = p.sort(score, _ROW, descending=[True, False], nulls_last=True)

    def capped(expr: pl.Expr) -> pl.Expr:
        # Each group's rows arrive in `ranked` order, so the cap is the head of
        # the list the aggregation builds anyway (no separate window).
        return expr if max_detections is None else expr.head(max_detections)

    pred_lists = ranked.group_by(keys, maintain_order=True).agg(
        capped(pl.col(geometry)).alias(PRED),
        capped(pl.col(score).cast(pl.Float64)).alias(PRED_SCORE),
        capped(pl.col(_ROW)).alias(PRED_ROW),
    )
    gt_lists = (
        g.sort(_ROW)
        .group_by(keys, maintain_order=True)
        .agg(pl.col(geometry).alias(GT), pl.col(_ROW).alias(GT_ROW))
    )

    population = pl.concat(
        [p.select(COL_IMAGE_ID), g.select(COL_IMAGE_ID), *_image_ids(images)]
    ).unique()
    classes = pl.concat([p.select(COL_CLASS_ID), g.select(COL_CLASS_ID)]).unique()
    geo_p = preds.collect_schema()[geometry]
    geo_g = gts.collect_schema()[geometry]
    return (
        population.join(classes, how="cross")
        .join(pred_lists, on=keys, how="left")
        .join(gt_lists, on=keys, how="left")
        .with_columns(
            pl.col(PRED).fill_null(pl.lit([], dtype=pl.List(geo_p))),
            pl.col(PRED_SCORE).fill_null(pl.lit([], dtype=pl.List(pl.Float64))),
            pl.col(PRED_ROW).fill_null(pl.lit([], dtype=pl.List(pl.UInt32))),
            pl.col(GT).fill_null(pl.lit([], dtype=pl.List(geo_g))),
            pl.col(GT_ROW).fill_null(pl.lit([], dtype=pl.List(pl.UInt32))),
        )
        .sort(keys)
    )


def _require(lf: pl.LazyFrame, cols: list[str], label: str) -> None:
    names = set(lf.collect_schema().names())
    missing = [c for c in cols if c not in names]
    if missing:
        msg = f"{label} has no column(s) {missing}; it has {sorted(names)}"
        raise ValueError(msg)


def _image_ids(
    images: Iterable[str] | pl.Series | pl.DataFrame | pl.LazyFrame | None,
) -> list[pl.LazyFrame]:
    if images is None:
        return []
    if isinstance(images, (pl.DataFrame, pl.LazyFrame)):
        lf = to_lazy(images)
        _require(lf, [COL_IMAGE_ID], "images")
        return [lf.select(pl.col(COL_IMAGE_ID).cast(pl.String))]
    if isinstance(images, str):
        msg = f"images takes a list of image ids, got the string {images!r}"
        raise TypeError(msg)
    return [pl.LazyFrame({COL_IMAGE_ID: pl.Series(list(images), dtype=pl.String)})]


# ---------------------------------------------------------------------------
# match_detections
# ---------------------------------------------------------------------------

_ACCEPTED = (
    "a BBOX_SCHEMA struct column; a List/Array of 4 numbers or a tuple of 4 "
    "column names, with box_format=; a CONTOUR_SCHEMA column (polygons); or a "
    "per-instance mask column (encoded image bytes or a 2-D List/Array)"
)


def match_detections(
    predictions: pl.DataFrame | pl.LazyFrame,
    ground_truth: pl.DataFrame | pl.LazyFrame,
    *,
    geometry: Geometry = "bbox",
    box_format: BoxFormat | None = None,
    image_id: str = COL_IMAGE_ID,
    class_id: str | None = COL_CLASS_ID,
    score: str = "score",
    iou_thresholds: float | Sequence[float] = 0.5,
    max_detections: int | None = None,
    images: Iterable[str] | pl.Series | pl.DataFrame | pl.LazyFrame | None = None,
    weight: str | None = None,
    group: str | None = None,
) -> DetectionTable:
    """Match object-level predictions to ground truth: the general entry point.

    Takes the long tables detections usually come in — one row per object —
    and returns the :class:`DetectionTable` every metric, statistic and
    confidence interval reads. The geometry decides the matcher:

    * **boxes** — a ``BBOX_SCHEMA`` column, or four numbers per object (a
      ``List``/``Array`` column, or a tuple of four column names) in
      ``box_format``: matched by box IoU (:class:`BBoxMatcher`);
    * **polygons** — a ``CONTOUR_SCHEMA`` column: matched by region IoU
      (:class:`ContourMatcher`);
    * **instance masks** — one binary mask per object (encoded image bytes or
      a 2-D ``List``/``Array``): its outline is extracted and matched as a
      polygon. A mask must hold exactly one connected region; a mask with
      several (or none) fails the query, naming its image — split it, or
      pass polygons.

    Matching is greedy in descending confidence, each prediction taking the
    unmatched ground truth it overlaps most at or above the threshold — the
    COCO rule. A sequence of ``iou_thresholds`` matches once per threshold
    (``[0.5, 0.55, …, 0.95]`` for COCO), so a detection that loses its ground
    truth at one threshold can claim another at a higher one.

    Args:
        predictions: One row per predicted object (``image_id``, geometry,
            ``score``, and ``class_id`` unless ``class_id=None``).
        ground_truth: One row per ground-truth object.
        geometry: The geometry column (same name in both tables), or a tuple
            of four coordinate columns.
        box_format: ``"xyxy"``, ``"xywh"`` or ``"cxcywh"``: required for boxes
            given as four numbers, refused otherwise.
        image_id: The image identifier column.
        class_id: The class column, or ``None`` for one class.
        score: The prediction confidence column.
        iou_thresholds: One IoU threshold, or several to match at each.
        max_detections: Keep each (image, class)'s ``max_detections``
            highest-scoring predictions (COCO: 100).
        images: The image population: ids, or a frame with an ``image_id``
            column (and the ``weight`` / ``group`` columns when named).
        weight: A per-image weight column of the ``images`` frame.
        group: A per-image subgroup column of the ``images`` frame (e.g. a
            scanner or site), for grouped and stratified evaluation.

    Returns:
        The :class:`DetectionTable`. Its ``det_idx`` is the prediction's
        position in its (image, class) list; :func:`group_objects` maps it
        back to the input row.
    """
    from ._matching import BBoxMatcher, ContourMatcher

    preds, gts = to_lazy(predictions), to_lazy(ground_truth)
    if not isinstance(geometry, str):
        cols = list(geometry)
        if box_format is None:
            raise ValueError(
                f"geometry {cols} names coordinate columns: pass box_format= "
                "('xyxy', 'xywh' or 'cxcywh') to say what they hold"
            )
        preds = preds.with_columns(
            bbox_from_coords(cols, format=box_format).alias("_box")
        )
        gts = gts.with_columns(bbox_from_coords(cols, format=box_format).alias("_box"))
        geometry, box_format = "_box", None
    kind = _geometry_kind(preds, gts, geometry, box_format)
    if kind == "coords":
        assert box_format is not None
        preds = preds.with_columns(bbox_from_coords(geometry, format=box_format))
        gts = gts.with_columns(bbox_from_coords(geometry, format=box_format))
        kind = "bbox"
    elif kind == "mask":
        preds = _instance_outlines(preds, geometry, image_id)
        gts = _instance_outlines(gts, geometry, image_id)
        kind = "contour"

    grouped = group_objects(
        preds,
        gts,
        geometry=geometry,
        image_id=image_id,
        class_id=class_id,
        score=score,
        images=images,
        max_detections=max_detections,
    )
    extra = _image_columns(images, weight, group)
    if extra is not None:
        grouped = grouped.join(extra, on=COL_IMAGE_ID, how="left")
    weight_col, group_col = weight, group
    common = {
        "pred_col": PRED,
        "gt_col": GT,
        "score_col": PRED_SCORE,
        "class_col": COL_CLASS_ID,
        "image_id_col": COL_IMAGE_ID,
        "weight_col": weight_col,
        "group_col": group_col,
    }
    if kind == "bbox":
        return BBoxMatcher(iou_threshold=iou_thresholds).match(grouped, **common)
    grouped = grouped.with_columns(
        pl.col(PRED).cast(pl.List(CONTOUR_SCHEMA)),
        pl.col(GT).cast(pl.List(CONTOUR_SCHEMA)),
    )
    return ContourMatcher(iou_threshold=iou_thresholds, auto_resize=False).match(
        grouped, **common
    )


def _image_columns(
    images: object, weight: str | None, group: str | None
) -> pl.LazyFrame | None:
    """The ``images`` frame's ``weight`` / ``group`` columns, by ``image_id``."""
    if weight is None and group is None:
        return None
    if not isinstance(images, (pl.DataFrame, pl.LazyFrame)):
        raise ValueError("weight= / group= name columns of an images= frame")
    lf = to_lazy(images)
    named = [c for c in (weight, group) if c is not None]
    _require(lf, [COL_IMAGE_ID, *named], "images")
    return lf.select(pl.col(COL_IMAGE_ID).cast(pl.String), *named)


def _geometry_kind(
    preds: pl.LazyFrame, gts: pl.LazyFrame, geometry: str, box_format: str | None
) -> str:
    """``"bbox"``, ``"coords"``, ``"contour"`` or ``"mask"``, from the dtype."""
    kinds = []
    for label, lf in (("predictions", preds), ("ground_truth", gts)):
        _require(lf, [geometry], label)
        kinds.append(_kind_of(lf.collect_schema()[geometry]))
    if kinds[0] != kinds[1]:
        msg = (
            f"geometry {geometry!r} is {kinds[0]} in predictions but {kinds[1]} "
            "in ground_truth"
        )
        raise ValueError(msg)
    kind = kinds[0]
    if kind is None:
        dtype = preds.collect_schema()[geometry]
        msg = f"geometry {geometry!r} has dtype {dtype}; expected {_ACCEPTED}"
        raise ValueError(msg)
    if kind == "coords" and box_format is None:
        raise ValueError(
            f"geometry {geometry!r} holds four numbers per object: pass "
            "box_format= ('xyxy', 'xywh' or 'cxcywh') to say which"
        )
    if kind != "coords" and box_format is not None:
        raise ValueError(
            f"box_format applies to boxes given as four numbers, not {kind}"
        )
    return kind


def _kind_of(dtype: pl.DataType) -> str | None:
    if dtype == BBOX_SCHEMA:
        return "bbox"
    if dtype == CONTOUR_SCHEMA:
        return "contour"
    if isinstance(dtype, pl.Array) and dtype.size == 4 and dtype.inner.is_numeric():
        return "coords"
    if isinstance(dtype, pl.List) and dtype.inner.is_numeric():
        return "coords"
    if isinstance(dtype, pl.Binary):
        return "mask"
    if isinstance(dtype, (pl.List, pl.Array)) and isinstance(
        dtype.inner, (pl.List, pl.Array)
    ):
        return "mask"
    return None


def _instance_outlines(lf: pl.LazyFrame, geometry: str, image_id: str) -> pl.LazyFrame:
    """Each instance mask → its one outline (``CONTOUR_SCHEMA``).

    Extraction is the matcher's own (threshold at 0.5, external outlines). A
    mask that is not exactly one region fails the query: picking one region
    of several would silently score a different object than the mask's.
    """
    from ._matching._contour import (
        _detect_source_info,
        _extract_contours_via,
        _SourceHandle,
    )

    handle = _SourceHandle.from_column(
        geometry, _detect_source_info(dict(lf.collect_schema()), geometry)
    )
    extracted = _extract_contours_via(
        lf, handle, threshold=0.5, min_area=0.0, output_col="_regions"
    )

    # The plugin refuses a set of any other size, naming the image.
    regions = pl.col("_regions").contour  # ty: ignore[unresolved-attribute]
    one = regions.single(label=pl.col(image_id).cast(pl.String))
    return extracted.with_columns(one.alias(geometry)).drop("_regions")
