"""Object tables to a ``DetectionTable``, for any polars-cv geometry.

:func:`match_detections` is the one entry point from long tables of
predictions and ground truth (see ``.._inputs.group_objects``) to a
:class:`DetectionTable`: boxes in any layout, polygons (contours) or
per-instance masks. The geometry's dtype decides the matcher.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from polars_cv import BBOX_SCHEMA, CONTOUR_SCHEMA
from polars_cv.geometry import BoxFormat, bbox_from_coords

from .._inputs import GT, PRED, PRED_SCORE, Geometry, _require, group_objects
from .._matching._carry import Carry, carried
from .._types import COL_CLASS_ID, COL_IMAGE_ID, DetectionTable, to_lazy
from ._bbox import BBoxMatcher
from ._contour import (
    ContourMatcher,
    _detect_source_info,
    _extract_contours_via,
    _SourceHandle,
)

#: The box formats, as the refusals name them (read from the enum).
_FORMATS = ", ".join(repr(f.value) for f in BoxFormat)

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
    box_format: str | BoxFormat | None = None,
    image_id: str = COL_IMAGE_ID,
    class_id: str | None = COL_CLASS_ID,
    score: str = "score",
    iou_thresholds: float | Sequence[float] = 0.5,
    max_detections: int | None = None,
    images: Iterable[str] | pl.Series | pl.DataFrame | pl.LazyFrame | None = None,
    weight: str | None = None,
    group: str | None = None,
    carry: Carry | None = None,
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
            column (and the ``weight`` / ``group`` / ``carry`` columns when
            named). With any of those, the frame must hold one row per
            image; this is not checked (a repeated ``image_id`` would repeat
            that image's objects).
        weight: A per-image weight column of the ``images`` frame.
        group: A per-image subgroup column of the ``images`` frame (e.g. a
            scanner or site), for grouped and stratified evaluation.
        carry: Further per-image columns of the ``images`` frame to copy into
            ``image_metadata``: names, or expressions over it named by their
            output name (see :meth:`BBoxMatcher.match`). Each is then a
            grouping key like ``group``.

    Returns:
        The :class:`DetectionTable`. Its ``det_idx`` is the prediction's
        position in its (image, class) list; :func:`group_objects` maps it
        back to the input row.
    """
    preds, gts = to_lazy(predictions), to_lazy(ground_truth)
    if not isinstance(geometry, str):
        cols = list(geometry)
        if box_format is None:
            raise ValueError(
                f"geometry {cols} names coordinate columns: pass box_format= "
                f"({_FORMATS}) to say what they hold"
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

    carry_cols = (
        carried(carry, to_lazy(images).collect_schema().names())
        if carry and isinstance(images, (pl.DataFrame, pl.LazyFrame))
        else {}
    )
    extra = _image_columns(images, weight, group, carry, carry_cols)
    grouped = group_objects(
        preds,
        gts,
        geometry=geometry,
        image_id=image_id,
        class_id=class_id,
        score=score,
        images=images if extra is None else extra,
        max_detections=max_detections,
        image_columns=[] if extra is None else extra.collect_schema().names()[1:],
    )
    weight_col, group_col = weight, group
    common = {
        "pred_col": PRED,
        "gt_col": GT,
        "score_col": PRED_SCORE,
        "class_col": COL_CLASS_ID,
        "image_id_col": COL_IMAGE_ID,
        "weight_col": weight_col,
        "group_col": group_col,
        "carry": list(carry_cols),
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
    images: object,
    weight: str | None,
    group: str | None,
    carry: Carry | None,
    carry_cols: dict[str, pl.Expr],
) -> pl.LazyFrame | None:
    """The ``images`` frame's ``weight`` / ``group`` / carried columns, by ``image_id``."""
    if weight is None and group is None and not carry:
        return None
    if not isinstance(images, (pl.DataFrame, pl.LazyFrame)):
        raise ValueError("weight= / group= / carry= name columns of an images= frame")
    lf = to_lazy(images)
    named = [c for c in (weight, group) if c is not None]
    _require(lf, [COL_IMAGE_ID, *named], "images")
    clash = set(named) & set(carry_cols)
    if clash:
        raise ValueError(f"carry: {sorted(clash)} is already weight= or group=")
    named += list(carry_cols)
    # One row per image is the caller's contract (see `match_detections`),
    # taken as given: checking it would cost a pass over `images`.
    return lf.select(
        pl.col(COL_IMAGE_ID).cast(pl.String),
        *(pl.col(c) for c in named if c not in carry_cols),
        *(e.alias(c) for c, e in carry_cols.items()),
    )


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
            f"box_format= ({_FORMATS}) to say which"
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
    handle = _SourceHandle.from_column(
        geometry, _detect_source_info(dict(lf.collect_schema()), geometry)
    )
    extracted = _extract_contours_via(
        lf, handle, threshold=0.5, min_area=0.0, output_col="_regions"
    )

    # The plugin refuses a set of any other size, naming the image.
    regions = pl.col("_regions").contour  # ty: ignore[missing-attribute]
    one = regions.single(label=pl.col(image_id).cast(pl.String))
    return extracted.with_columns(one.alias(geometry)).drop("_regions")
