"""The segmentation report: per-image measures and their summary.

:class:`SegmentationReport` summarises a frame of per-image measures
(:data:`MEASURES`: Dice, IoU, ASSD, Hausdorff, HD95 — the MONAI set) and
bootstraps intervals over images. Computing the measures from masks is
polars-cv's (``_cv._segmentation``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import polars as pl

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
        replicates = draws.join(images, on="_unit", how="left", nulls_equal=True)
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
            bounds, on="metric", how="inner", maintain_order="right", nulls_equal=True
        )

    def __repr__(self) -> str:
        lines = [f"SegmentationReport ({self.per_image.height} images)"]
        for metric, value, _, undefined in self.summary.iter_rows():
            shown = "n/a" if value is None else f"{value:.4f}"
            note = f"  ({undefined} undefined)" if undefined else ""
            lines.append(f"  {metric:<20}  {shown}{note}")
        return "\n".join(lines)
