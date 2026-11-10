"""Object-level inputs: long tables of predictions and ground truth.

Detections usually arrive one row per object — a model's output, a COCO
annotation file flattened, a CSV export — while the matchers read one row per
(image, class) holding lists. :func:`group_objects` is the one way between the
two; it knows nothing of the geometry it groups (``_cv._match.match_detections``
is the entry point that does).

The population is never inferred from one side alone. An image or class that
appears only among the predictions (false positives on an empty image) or
only among the ground truth (missed objects) is a row of the evaluation, with
an empty list on the other side; images with neither come from ``images=``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from ._types import COL_CLASS_ID, COL_IMAGE_ID, DEFAULT_CLASS, to_lazy

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
    image_columns: Sequence[str] = (),
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
        image_columns: Columns of the ``images`` frame to copy onto each of
            the image's rows (null for an image it does not list). The frame
            must hold one row per image; this is not checked.

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
    if image_columns:
        if not isinstance(images, (pl.DataFrame, pl.LazyFrame)):
            raise ValueError("image_columns names columns of an images= frame")
        per_image = to_lazy(images)
        _require(per_image, [COL_IMAGE_ID, *image_columns], "images")
        # Joined here, before the final sort, rather than onto the result: the
        # sort re-chunks its output into even morsels for the matcher's
        # parallel pass, while a join after it hands the matcher one morsel.
        population = population.join(
            per_image.select(pl.col(COL_IMAGE_ID).cast(pl.String), *image_columns),
            on=COL_IMAGE_ID,
            how="left",
            nulls_equal=True,
        )
    classes = pl.concat([p.select(COL_CLASS_ID), g.select(COL_CLASS_ID)]).unique()
    geo_p = preds.collect_schema()[geometry]
    geo_g = gts.collect_schema()[geometry]
    return (
        population.join(classes, how="cross")
        .join(pred_lists, on=keys, how="left", nulls_equal=True)
        .join(gt_lists, on=keys, how="left", nulls_equal=True)
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
