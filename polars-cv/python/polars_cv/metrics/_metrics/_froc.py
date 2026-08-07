"""FROC metric: sensitivity vs false positives per image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from .._auc import CorrectionMethod, detection_level_mann_whitney, mann_whitney_u_auc
from .._result import MetricResult
from .._types import (
    COL_CLASS_ID,
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
    COL_WEIGHT,
    DetectionTable,
)

# Columns the bootstrap uses to name each draw before it becomes the image_id.
_COL_DRAW_ID = "_draw_id"
_COL_DRAW_IDX = "_draw_idx"


def _draw_frame(sampled_ids: list[str]) -> pl.LazyFrame:
    """Name each bootstrap draw as a distinct evaluation unit.

    Returns a frame mapping ``image_id`` (the drawn id, for joining) to
    ``_draw_id`` (``<image_id>#draw<n>``, which becomes the replicate's
    ``image_id``). Sampling is with replacement, so the same image can appear
    several times and each appearance is its own unit.
    """
    return (
        pl.DataFrame({COL_IMAGE_ID: sampled_ids})
        .with_row_index(_COL_DRAW_IDX)
        .with_columns(
            (
                pl.col(COL_IMAGE_ID)
                + pl.lit("#draw")
                + pl.col(_COL_DRAW_IDX).cast(pl.String)
            ).alias(_COL_DRAW_ID)
        )
        .drop(_COL_DRAW_IDX)
        .lazy()
    )


def _count_images(meta_df: pl.DataFrame) -> int:
    """Count evaluation units in an image-metadata frame.

    Metadata carries one row per (image, class), so the row count is not the
    image count — using it divides FP-per-image by the number of classes.
    This is the single definition both ``froc_curve`` and
    ``FROCResult._reconstruct`` read.
    """
    return int(meta_df.select(pl.col(COL_IMAGE_ID).n_unique()).item())


@dataclass(frozen=True)
class FROCResult(MetricResult):
    """FROC-specific result with sensitivity-at-FP helpers.

    Attributes:
        curve: DataFrame with columns ``threshold``, ``tp``, ``fp``, ``fn``,
            ``total_gts``, ``fp_per_image``, ``sensitivity``.
        n_images: Number of images in the dataset.
        total_targets: Total GT count across all images.
        iou_threshold: IoU threshold used for matching.
        detection_table: The underlying ``DetectionTable`` for bootstrap.
    """

    n_images: int = 0
    total_targets: int = 0
    iou_threshold: float = 0.5
    detection_table: DetectionTable | None = None

    def auc(  # type: ignore[override]
        self,
        *,
        method: Literal["trapezoidal", "mann_whitney"] = "trapezoidal",
        fp_range: tuple[float, float] | None = None,
        correction: CorrectionMethod = None,
        level: Literal["detection", "image"] = "detection",
    ) -> float:
        """Compute AUC for the FROC curve.

        Args:
            method: AUC computation method.
                ``"trapezoidal"`` (default) integrates the FROC curve.
                ``"mann_whitney"`` computes a non-parametric AUC via the
                Mann-Whitney U statistic.
            fp_range: Optional ``(lo, hi)`` false-positive-per-image range
                for partial AUC (trapezoidal only).
            correction: Partial-AUC correction (trapezoidal only).
                ``None`` returns the raw area.
                ``"normalize"`` divides by the range width.
                ``"mcclish"`` applies McClish's standardized correction.
            level: Granularity for Mann-Whitney (ignored for trapezoidal).
                ``"detection"`` -- P(random TP score > random FP score).
                ``"image"`` -- P(positive-image score > negative-image score).

        Returns:
            AUC value.

        Raises:
            ValueError: On invalid ``method``/``level`` or unsupported
                parameter combinations.
        """
        if method == "mann_whitney":
            if fp_range is not None or correction is not None:
                raise ValueError(
                    "fp_range and correction are not supported with "
                    "method='mann_whitney'. Mann-Whitney computes a global "
                    "rank statistic, not a curve integral."
                )
            return self._mann_whitney(level)
        if method == "trapezoidal":
            return super().auc(
                x_col="fp_per_image",
                y_col="sensitivity",
                x_range=fp_range,
                correction=correction,
            )
        raise ValueError(
            f"Unknown method {method!r}. Expected 'trapezoidal' or 'mann_whitney'."
        )

    def sensitivity_at_fp(self, fp_per_image: float) -> float | None:
        """Interpolate sensitivity at a requested FP/image rate.

        Args:
            fp_per_image: Target false-positive-per-image rate.

        Returns:
            Interpolated sensitivity, or ``None`` when ``fp_per_image`` is
            outside the observed range of the curve.
        """
        return self.interpolate(
            x_col="fp_per_image", y_col="sensitivity", at=fp_per_image
        )

    def summary_table(  # type: ignore[override]
        self,
        fp_rates: list[float] | None = None,
    ) -> pl.DataFrame:
        """Build a sensitivity summary at standard FP/image operating points.

        Args:
            fp_rates: Operating points. Defaults to standard radiology set.

        Returns:
            DataFrame with ``fp_per_image`` and ``sensitivity`` columns.
        """
        rates = fp_rates or [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        return super().summary_table(
            x_col="fp_per_image",
            y_col="sensitivity",
            operating_points=rates,
        )

    # ------------------------------------------------------------------
    # Bootstrap hooks
    # ------------------------------------------------------------------

    def _get_detection_table(self) -> DetectionTable:
        """Return the underlying DetectionTable."""
        if self.detection_table is None:
            raise ValueError("bootstrap_ci / Mann-Whitney requires detection_table.")
        return self.detection_table

    def _reconstruct(self, sampled_ids: list[str]) -> FROCResult:
        """Rebuild a FROCResult from bootstrap-sampled image IDs.

        Joins sampled IDs against the detections and metadata tables and
        re-derives the FROC curve via cumulative sums.

        Sampling is with replacement, so the same image can be drawn more than
        once and each draw is its own evaluation unit: it contributes its own
        detections, its own ``n_gts``, and its own slot in the FP-per-image
        denominator. Each draw therefore gets a distinct synthetic
        ``image_id`` before the join. Carrying the duplicates as repeated
        ``image_id``s instead would leave every downstream count guessing
        whether a repeat is a redraw or one image legitimately owned twice —
        which is exactly the ambiguity ``_curve_from_detections`` has to raise
        on.
        """
        table = self._get_detection_table()

        draws = _draw_frame(sampled_ids)

        sampled_det = (
            draws.join(table.detections, on=COL_IMAGE_ID, how="left")
            .drop(COL_IMAGE_ID)
            .rename({_COL_DRAW_ID: COL_IMAGE_ID})
        )
        sampled_meta = (
            draws.join(table.image_metadata, on=COL_IMAGE_ID, how="left")
            .drop(COL_IMAGE_ID)
            .rename({_COL_DRAW_ID: COL_IMAGE_ID})
        )

        det_df = sampled_det.collect(engine="streaming")
        meta_df = sampled_meta.collect(engine="streaming")

        n_images = _count_images(meta_df)
        # Sum over every row: one row per (draw, class), so a repeatedly-drawn
        # image contributes its n_gts once per draw.
        total_targets = int(meta_df.select(pl.col(COL_N_GTS).sum()).item())

        curve = _curve_from_detections(det_df, meta_df, total_targets)

        sampled_table = DetectionTable.from_matched(sampled_det, sampled_meta)

        return FROCResult(
            curve=curve,
            n_images=n_images,
            total_targets=total_targets,
            iou_threshold=self.iou_threshold,
            detection_table=sampled_table,
        )

    # ------------------------------------------------------------------
    # Mann-Whitney internals
    # ------------------------------------------------------------------

    def _mann_whitney(self, level: str) -> float:
        """Dispatch Mann-Whitney computation by level."""
        if level == "detection":
            table = self._get_detection_table()
            det_df = table.detections.collect(engine="streaming")
            return detection_level_mann_whitney(det_df, COL_SCORE, COL_IS_TP)
        if level == "image":
            return self._mann_whitney_image()
        raise ValueError(
            f"Unsupported level {level!r}. Expected 'detection' or 'image'."
        )

    def _mann_whitney_image(self) -> float:
        """Image-level MW-U: P(positive-image score > negative-image score)."""
        table = self._get_detection_table()
        det_df = table.detections.collect(engine="streaming")
        meta_df = table.image_metadata.collect(engine="streaming")

        if COL_GT_LABEL not in meta_df.columns:
            raise ValueError(
                "auc(method='mann_whitney', level='image') requires "
                "gt_label in metadata."
            )

        # Positive images: max TP score (0 if no TPs)
        pos_ids = meta_df.filter(pl.col(COL_GT_LABEL)).select(COL_IMAGE_ID)
        pos_scores = (
            pos_ids.join(
                det_df.filter(pl.col(COL_IS_TP))
                .group_by(COL_IMAGE_ID)
                .agg(pl.col(COL_SCORE).max()),
                on=COL_IMAGE_ID,
                how="left",
            ).with_columns(pl.col(COL_SCORE).fill_null(0.0))
        )[COL_SCORE]

        # Negative images: max detection score (0 if none)
        neg_ids = meta_df.filter(~pl.col(COL_GT_LABEL)).select(COL_IMAGE_ID)
        neg_scores = (
            neg_ids.join(
                det_df.group_by(COL_IMAGE_ID).agg(pl.col(COL_SCORE).max()),
                on=COL_IMAGE_ID,
                how="left",
            ).with_columns(pl.col(COL_SCORE).fill_null(0.0))
        )[COL_SCORE]

        return mann_whitney_u_auc(pos_scores, neg_scores)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def froc_curve(
    table: DetectionTable,
    *,
    thresholds: list[float] | None = None,
) -> FROCResult:
    """Compute FROC operating points from a DetectionTable.

    Uses a cumulative-sum approach: detections are sorted by score
    (descending) and TP/FP counts are accumulated, producing one curve
    point per unique score. This avoids the O(images x thresholds)
    dense grid that the previous implementation used.

    Args:
        table: Canonical detection table produced by a matcher.
        thresholds: Optional explicit score thresholds. When provided,
            the curve is filtered to these thresholds only.

    Returns:
        ``FROCResult`` with curve and metadata.
    """
    det_df, meta_df = table.collect(engine="streaming")

    if det_df.height == 0:
        return _empty_froc_result(table)

    n_images = _count_images(meta_df)
    total_targets = int(meta_df.select(pl.col(COL_N_GTS).sum()).item())

    curve = _curve_from_detections(det_df, meta_df, total_targets)

    if thresholds is not None:
        threshold_set = set(thresholds)
        curve = curve.filter(pl.col("threshold").is_in(threshold_set))

    return FROCResult(
        curve=curve,
        n_images=n_images,
        total_targets=total_targets,
        iou_threshold=table._matching_iou_threshold or 0.5,
        detection_table=table,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _curve_from_detections(
    det_df: pl.DataFrame,
    meta_df: pl.DataFrame,
    total_targets: int,
) -> pl.DataFrame:
    """Build a FROC curve from detections using cumulative sums.

    Sorts detections by score descending and computes running TP/FP counts,
    weighting each image by its ``weight`` in ``meta_df``.

    The image count is read from ``meta_df`` rather than passed in: the
    FP-per-image denominator and the ``n_images`` a caller reports must be the
    same number, and two ways of arriving at it is how they came to disagree
    between the point estimate and the bootstrap replicate.

    Args:
        det_df: Collected detections DataFrame.
        meta_df: Collected image metadata DataFrame, one row per
            (image, class). Bootstrap draws arrive as distinct ``image_id``s.
        total_targets: Total ground-truth target count.

    Returns:
        DataFrame with ``threshold``, ``tp``, ``fp``, ``fn``, ``total_gts``,
        ``fp_per_image``, ``sensitivity``.
    """
    if det_df.height == 0:
        return pl.DataFrame(
            {
                "threshold": [float("inf")],
                "tp": [0],
                "fp": [0],
                "fn": [total_targets],
                "total_gts": [total_targets],
                "fp_per_image": [0.0],
                "sensitivity": [0.0],
            },
            schema={
                "threshold": pl.Float64,
                "tp": pl.Int64,
                "fp": pl.Int64,
                "fn": pl.Int64,
                "total_gts": pl.Int64,
                "fp_per_image": pl.Float64,
                "sensitivity": pl.Float64,
            },
        )

    # `weight` is in IMAGE_META_REQUIRED and `from_matched` validates it, so
    # every metadata frame reaching here is weighted — an all-1.0 weight column
    # reduces the formulas below to the plain counts exactly. There is no
    # unweighted branch to fall into, and adding one back would be a second
    # implementation of the same curve.
    total_weighted_gts = float(
        meta_df.select(
            (pl.col(COL_N_GTS).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum()
        ).item()
    )
    total_weighted_gts_f = max(total_weighted_gts, 1.0)

    # Attach one weight per lookup key so a repeated metadata row (one rendered
    # image owned by two cases) does not fan every detection out before
    # aggregation. Conflicting weights for the same key are ill-defined — the
    # numerator would pick an arbitrary row while the denominator sums every
    # row — so raise instead.
    weight_keys = _weight_lookup_keys(meta_df)
    # Check on image_id alone, which subsumes the (image_id, class_id) check:
    # a weight is a property of an *image*, and the FP-per-image denominator
    # below dedupes on image_id, so two classes of one image disagreeing about
    # its weight is exactly as ill-defined as two rows of one class doing so.
    _raise_on_conflicting_weights(meta_df, [COL_IMAGE_ID])
    lookup_cols = [*weight_keys, COL_WEIGHT]
    weight_lookup = meta_df.select(lookup_cols).unique(subset=weight_keys, keep="first")

    # FP-per-image counts *images*, so its denominator sums one weight per
    # image_id — not one per metadata row. A multi-class table carries a row
    # per (image, class), and summing those would divide the false-positive
    # rate by the number of classes.
    weight_sum = float(
        meta_df.select(COL_IMAGE_ID, COL_WEIGHT)
        .unique(subset=[COL_IMAGE_ID], keep="first")
        .select(pl.col(COL_WEIGHT).sum())
        .item()
    )
    weight_sum_f = max(weight_sum, 1.0)

    det_with_weight = det_df.join(
        weight_lookup,
        on=weight_keys,
        how="left",
    ).with_columns(pl.col(COL_WEIGHT).fill_null(1.0))

    # Bucket detections by score, aggregate weighted TP/FP
    bucketed = (
        det_with_weight.group_by(COL_SCORE)
        .agg(
            tp_count=pl.col(COL_IS_TP).sum().cast(pl.Int64),
            fp_count=(~pl.col(COL_IS_TP)).sum().cast(pl.Int64),
            weighted_tp=(pl.col(COL_IS_TP).cast(pl.Float64) * pl.col(COL_WEIGHT)).sum(),
            weighted_fp=(
                (~pl.col(COL_IS_TP)).cast(pl.Float64) * pl.col(COL_WEIGHT)
            ).sum(),
        )
        .sort(COL_SCORE, descending=True)
        .with_columns(
            tp=pl.col("tp_count").cum_sum(),
            fp=pl.col("fp_count").cum_sum(),
            cum_weighted_tp=pl.col("weighted_tp").cum_sum(),
            cum_weighted_fp=pl.col("weighted_fp").cum_sum(),
        )
        .rename({COL_SCORE: "threshold"})
        .with_columns(
            total_gts=pl.lit(total_targets, dtype=pl.Int64),
            fn=(pl.lit(total_targets, dtype=pl.Int64) - pl.col("tp")).clip(
                lower_bound=0
            ),
            sensitivity=pl.col("cum_weighted_tp") / pl.lit(total_weighted_gts_f),
            fp_per_image=pl.col("cum_weighted_fp") / pl.lit(weight_sum_f),
        )
        .select(
            "threshold",
            "tp",
            "fp",
            "fn",
            "total_gts",
            "fp_per_image",
            "sensitivity",
        )
    )

    # Prepend origin point (threshold=inf, everything zero)
    inf_row = pl.DataFrame(
        {
            "threshold": [float("inf")],
            "tp": [0],
            "fp": [0],
            "fn": [total_targets],
            "total_gts": [total_targets],
            "fp_per_image": [0.0],
            "sensitivity": [0.0],
        },
        schema={
            "threshold": pl.Float64,
            "tp": pl.Int64,
            "fp": pl.Int64,
            "fn": pl.Int64,
            "total_gts": pl.Int64,
            "fp_per_image": pl.Float64,
            "sensitivity": pl.Float64,
        },
    )
    # Sort by descending threshold, which *is* ascending fp_per_image: tp/fp
    # are cumulative over descending score, so a lower threshold can only add
    # false positives. Sorting on fp_per_image directly would look equivalent
    # and is not — thresholds are unique (group_by(score) plus the +inf origin)
    # so this is a total order, while fp_per_image ties constantly (every
    # bucket that adds only TPs leaves it unchanged) and Polars' sort defaults
    # to maintain_order=False, leaving those ties in an unspecified order.
    # Descending threshold also puts each tie group's rows in ascending
    # sensitivity, so the plotting order is the upper envelope.
    return pl.concat([inf_row, bucketed], how="vertical").sort(
        "threshold", descending=True
    )


def _weight_lookup_keys(meta_df: pl.DataFrame) -> list[str]:
    """Return the columns that identify a weight-lookup unit.

    Prefer ``(image_id, class_id)`` when class is present so multi-class
    rows that share an image are not collapsed; otherwise ``image_id``.
    """
    if COL_CLASS_ID in meta_df.columns:
        return [COL_IMAGE_ID, COL_CLASS_ID]
    return [COL_IMAGE_ID]


def _raise_on_conflicting_weights(
    meta_df: pl.DataFrame,
    weight_keys: list[str],
) -> None:
    """Raise when a lookup key has more than one distinct weight.

    Equal weights on duplicate keys are fine (shared image / bootstrap
    redraws). Conflicting weights make the weighted FROC numerator
    (first-row weight) disagree with the denominators (sum of every row)
    in an order-dependent way.

    Args:
        meta_df: Image metadata frame that includes ``weight``.
        weight_keys: Columns identifying a weight-lookup unit.

    Raises:
        ValueError: If any key group has more than one distinct weight.
    """
    conflicts = (
        meta_df.group_by(weight_keys)
        .agg(
            n_weights=pl.col(COL_WEIGHT).n_unique(),
            weights=pl.col(COL_WEIGHT).unique().sort(),
        )
        .filter(pl.col("n_weights") > 1)
    )
    if conflicts.height == 0:
        return

    examples: list[str] = []
    for row in conflicts.head(3).iter_rows(named=True):
        key_parts = [f"{k}={row[k]!r}" for k in weight_keys]
        examples.append(f"({', '.join(key_parts)}): weights={row['weights']}")
    raise ValueError(
        "image_metadata has conflicting weights for the same "
        f"{'+'.join(weight_keys)} key(s). Weighted FROC attaches one weight "
        "per key to detections while denominators sum every metadata row, "
        "so disagreeing weights make sensitivity order-dependent. Use a "
        "single weight per unit, or a composite image_id when each ownership "
        f"is a distinct evaluation unit. Examples: {'; '.join(examples)}"
    )


def _empty_froc_result(table: DetectionTable) -> FROCResult:
    """Return an empty FROC result."""
    empty_curve = pl.DataFrame(
        {
            "threshold": [float("inf")],
            "tp": [0],
            "fp": [0],
            "fn": [0],
            "total_gts": [0],
            "fp_per_image": [0.0],
            "sensitivity": [0.0],
        }
    )
    return FROCResult(
        curve=empty_curve,
        n_images=0,
        total_targets=0,
        detection_table=table,
    )
