"""
Mask comparison and per-region statistics over pipeline expressions.

``mask_iou`` and ``mask_dice`` compare two binary masks; ``region_stats``
reduces per-pixel maps over contour regions. Re-exported from
:mod:`polars_cv`.
"""

from __future__ import annotations

import polars as pl

from .lazy import LazyPipelineExpr
from .pipeline import Pipeline


def mask_iou(
    pred: LazyPipelineExpr,
    target: LazyPipelineExpr,
    *,
    epsilon: float = 1e-7,
) -> pl.Expr:
    """
    Compute Intersection over Union (IoU) between two binary masks.

    Args:
        pred: Mask expression (binary 0/255).
        target: Target mask expression.
    """
    # Compute intersection and union, then reduce to scalars
    intersection = (
        pred.bitwise_and(target)
        .pipe(Pipeline().reduce_sum())
        .alias("_iou_intersection")
    )
    union = pred.bitwise_or(target).pipe(Pipeline().reduce_sum()).alias("_iou_union")

    # Sink both as native scalars (Float64)
    result = intersection.merge_pipe(union).sink(
        {
            "_iou_intersection": "native",
            "_iou_union": "native",
        }
    )

    # Compute IoU using Polars scalar operations
    intersection_sum = result.struct.field("_iou_intersection")
    union_sum = result.struct.field("_iou_union")

    return intersection_sum / (union_sum + epsilon)


def mask_dice(
    pred: LazyPipelineExpr,
    target: LazyPipelineExpr,
    *,
    epsilon: float = 1e-7,
) -> pl.Expr:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        pred: Mask expression (binary 0/255).
        target: Target mask expression.
    """
    # Compute intersection, pred sum, and target sum as scalars
    intersection = (
        pred.bitwise_and(target)
        .pipe(Pipeline().reduce_sum())
        .alias("_dice_intersection")
    )
    pred_sum = pred.pipe(Pipeline().reduce_sum()).alias("_dice_pred")
    target_sum = target.pipe(Pipeline().reduce_sum()).alias("_dice_target")

    # Sink all three as native scalars (Float64)
    result = intersection.merge_pipe(pred_sum, target_sum).sink(
        {
            "_dice_intersection": "native",
            "_dice_pred": "native",
            "_dice_target": "native",
        }
    )

    # Compute Dice using Polars scalar operations
    inter = result.struct.field("_dice_intersection")
    total = result.struct.field("_dice_pred") + result.struct.field("_dice_target")

    return (2.0 * inter) / (total + epsilon)


def region_stats(
    regions: pl.Expr,
    maps: "dict[str, tuple[LazyPipelineExpr, str]]",
    *,
    region_mode: str = "interior",
) -> pl.Expr:
    """
    Reduce several per-pixel maps over each contour region: regionprops for
    any map.

    Each entry is one ``label_reduce`` of its map over ``regions``, by a
    statistic named as ``statistics()`` names them (``"max"``, ``"median"``,
    ``"p90"``, ``"count"``, ...). The result is a struct of equal-length lists,
    one value per region in ``regions``' order; ``.unnest()`` then
    ``.explode(...)`` gives one row per region.

    Args:
        regions: A contour-set expression (``List[Contour]``), one set per row.
        maps: Output name -> ``(map, statistic)``; each map a single-channel
            ``LazyPipelineExpr`` aligned with ``regions``.
        region_mode: Which pixels a region holds: "interior" (default),
            "boundary" or "bbox", as ``label_reduce``.

    Returns:
        A ``Struct`` expression with one ``List(Float64)`` field per entry.

    Example:
        ```python
        >>> stats = polars_cv.region_stats(pl.col("regions"), {
        ...     "heat_max": (heat, "max"),
        ...     "heat_p90": (heat, "p90"),
        ...     "skin_min": (skin_distance, "min"),
        ... })
        >>> df.with_columns(stats=stats).unnest("stats").explode(
        ...     ["heat_max", "heat_p90", "skin_min"])
        ```
    """
    if not maps:
        raise ValueError("region_stats needs at least one (map, statistic) entry")
    nodes = []
    for name, (image, stat) in maps.items():
        step = Pipeline().label_reduce(regions, reduction=stat, region_mode=region_mode)
        nodes.append(image.pipe(step).alias(name))
    merged = nodes[0] if len(nodes) == 1 else nodes[0].merge_pipe(*nodes[1:])
    return merged.sink({name: "native" for name in maps})
