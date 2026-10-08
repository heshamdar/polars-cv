"""Semantic segmentation: overlap and boundary measures of mask pairs.

:func:`segmentation_measures` is one expression giving, per row, the overlap
(Dice, IoU) and boundary (ASSD, Hausdorff, HD95) agreement of a predicted and
a reference mask — the MONAI set — computed in one graph per row: both masks
binarised once, their pixel counts and their outlines (hole borders included)
read from the same decode. :func:`evaluate_segmentation` adds the per-dataset
summary and bootstrap intervals over images.

Boundaries are the exact pixel-edge outlines (``extract_contours``), measured
point-to-edge (``.contour.set_boundary_distances``): every sample of one mask's
boundary to the nearest edge of the other's. With ``frame="image"`` boundary
on the image edge is not measured — a structure cut off by the field of view
has a boundary there that is not anatomy, and that is not an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import polars as pl

from ..pipeline import Pipeline

#: The per-row measures, in output order.
MEASURES = (
    "dice",
    "iou",
    "assd",
    "hd",
    "hd95",
    "mean_pred_to_target",
    "mean_target_to_pred",
)


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
        return expr.cv.pipe(Pipeline().source("auto").threshold(value=threshold))  # ty: ignore[unresolved-attribute]

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


@dataclass(frozen=True, repr=False)
class SegmentationReport:
    """Per-image measures of a segmentation evaluation and their summary.

    Attributes:
        per_image: ``[image_id, dice, iou, assd, hd, hd95, …]``, one row per
            image.
    """

    per_image: pl.DataFrame

    @cached_property
    def summary(self) -> pl.DataFrame:
        """``[metric, value, n, n_undefined]``: the mean of each measure over
        the images where it is defined (boundary distances are undefined when
        a mask is empty), and how many images that left out."""
        rows = [
            self.per_image.select(
                pl.lit(m).alias("metric"),
                pl.col(m).mean().alias("value"),
                pl.col(m).count().cast(pl.Int64).alias("n"),
                pl.col(m).null_count().cast(pl.Int64).alias("n_undefined"),
            )
            for m in MEASURES
        ]
        return pl.concat(rows)

    def value(self, metric: str) -> float | None:
        """One summary mean by name."""
        return self.summary.filter(pl.col("metric") == metric).item(0, "value")

    def ci(
        self,
        metric: str | Sequence[str] | None = None,
        *,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """Percentile bootstrap intervals of the summary means over images.

        Images are drawn by the same seeded hash resampler as the detection
        intervals (:func:`~polars_cv.metrics.bootstrap_ci`), so a seed means the
        same thing in both.

        Returns:
            ``[metric, value, ci_lower, ci_upper]``.
        """
        from ._bootstrap import _COL_BOOT, _lazy_resample, _validate_ci_params

        _validate_ci_params(n_bootstrap, confidence)
        names = (
            list(MEASURES)
            if metric is None
            else [metric]
            if isinstance(metric, str)
            else list(metric)
        )
        unknown = [n for n in names if n not in MEASURES]
        if unknown:
            raise ValueError(f"unknown metric(s) {unknown}; expected {list(MEASURES)}")
        images = self.per_image.lazy().with_row_index("_unit")
        draws = _lazy_resample(
            images.select("_unit"),
            unit_col="_unit",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        replicates = draws.join(images, on="_unit", how="left")
        alpha = (1.0 - confidence) / 2.0
        means = replicates.group_by(_COL_BOOT).agg(pl.col(n).mean() for n in names)
        q = means.select(
            *(pl.col(n).quantile(alpha, "linear").alias(f"{n}:lo") for n in names),
            *(
                pl.col(n).quantile(1.0 - alpha, "linear").alias(f"{n}:hi")
                for n in names
            ),
        ).collect()
        bounds = pl.DataFrame(
            {
                "metric": names,
                "ci_lower": [q.item(0, f"{n}:lo") for n in names],
                "ci_upper": [q.item(0, f"{n}:hi") for n in names],
            },
            schema={
                "metric": pl.String,
                "ci_lower": pl.Float64,
                "ci_upper": pl.Float64,
            },
        )
        return self.summary.select("metric", "value").join(
            bounds, on="metric", how="inner", maintain_order="right"
        )

    def __repr__(self) -> str:
        lines = [f"SegmentationReport ({self.per_image.height} images)"]
        for metric, value, _, undefined in self.summary.iter_rows():
            shown = "n/a" if value is None else f"{value:.4f}"
            note = f"  ({undefined} undefined)" if undefined else ""
            lines.append(f"  {metric:<20}  {shown}{note}")
        return "\n".join(lines)


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
    ids = (
        pl.col(image_id).cast(pl.String)
        if image_id is not None
        else pl.int_range(pl.len()).cast(pl.String)
    )
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
