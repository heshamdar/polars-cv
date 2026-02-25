"""Bootstrap helpers for detection metrics — sequential and vectorized paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import polars as pl

from ._types import COL_IMAGE_ID, COL_IS_TP, COL_N_GTS, COL_SCORE

if TYPE_CHECKING:
    from ._types import DetectionTable


@dataclass(frozen=True)
class BootstrapResult:
    """Container for bootstrap confidence interval results.

    Attributes:
        point_estimate: Metric value on the original sample.
        ci_lower: Lower confidence bound.
        ci_upper: Upper confidence bound.
        confidence: Confidence level used for the interval.
        distribution: Raw bootstrap metric values.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    distribution: list[float]


# ---------------------------------------------------------------------------
# Sequential fallback (original approach, works with any metric_fn callback)
# ---------------------------------------------------------------------------


def bootstrap_metric_sequential(
    *,
    image_ids: list[str],
    metric_fn: Callable[[list[str]], float],
    point_estimate: float,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    strata: dict[str, str] | None = None,
) -> BootstrapResult:
    """Estimate confidence intervals by image-level bootstrap sampling.

    This is the sequential fallback that calls *metric_fn* once per iteration.
    Use :func:`bootstrap_pr_auc` for a fully vectorized Polars-native path when
    computing PR-based AUC.

    Args:
        image_ids: Base image IDs to sample with replacement.
        metric_fn: Callback computing a scalar metric from sampled image IDs.
        point_estimate: Metric on the original sample.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed.
        strata: Optional image->stratum mapping for stratified resampling.

    Returns:
        ``BootstrapResult`` with percentile confidence interval.
    """
    _validate_bootstrap_params(n_bootstrap, confidence, image_ids)
    distribution: list[float] = []

    if strata is None:
        ids_series = pl.Series(COL_IMAGE_ID, image_ids)
        for i in range(n_bootstrap):
            iter_seed = (seed + i) if seed is not None else None
            sampled = ids_series.sample(
                n=ids_series.len(), with_replacement=True, seed=iter_seed
            )
            distribution.append(float(metric_fn(sampled.to_list())))
    else:
        grouped: dict[str, list[str]] = {}
        for image_id in image_ids:
            key = strata.get(image_id, "__missing__")
            grouped.setdefault(key, []).append(image_id)
        grouped_series = {k: pl.Series(k, v) for k, v in grouped.items()}

        for i in range(n_bootstrap):
            iter_seed = (seed + i) if seed is not None else None
            sampled_ids: list[str] = []
            for j, (key, s) in enumerate(grouped_series.items()):
                stratum_seed = (
                    (iter_seed * len(grouped_series) + j)
                    if iter_seed is not None
                    else None
                )
                sampled = s.sample(n=s.len(), with_replacement=True, seed=stratum_seed)
                sampled_ids.extend(sampled.to_list())
            distribution.append(float(metric_fn(sampled_ids)))

    return _finalize(distribution, point_estimate, confidence)


# ---------------------------------------------------------------------------
# Vectorized bootstrap for PR AUC (single Polars lazy plan)
# ---------------------------------------------------------------------------


def bootstrap_pr_auc(
    table: DetectionTable,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    class_id: str | None = None,
) -> BootstrapResult:
    """Vectorized bootstrap for precision-recall AUC.

    Generates all bootstrap samples as a single DataFrame, joins with the
    detection table, and computes AP per bootstrap iteration using Polars
    window functions -- all in one lazy plan.

    Args:
        table: Canonical detection table.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed.
        class_id: Optional class filter.

    Returns:
        ``BootstrapResult`` with percentile confidence interval.
    """
    image_ids, strata = table.image_ids_and_strata()
    _validate_bootstrap_params(n_bootstrap, confidence, image_ids)

    if class_id is not None:
        table = table.filter_class(class_id)

    from ._metrics._precision_recall import average_precision

    point = average_precision(table, class_id=class_id)

    samples_df = _generate_bootstrap_samples(
        image_ids=image_ids,
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Get total GTs per bootstrap by joining with metadata
    meta_df = table.image_metadata.collect(engine="streaming")
    boot_gts = (
        samples_df.lazy()
        .join(
            meta_df.lazy().select(COL_IMAGE_ID, COL_N_GTS),
            on=COL_IMAGE_ID,
            how="left",
        )
        .group_by("bootstrap_id")
        .agg(total_gts=pl.col(COL_N_GTS).sum().cast(pl.Float64))
    )

    # Join samples with detections
    det_df = table.detections.collect(engine="streaming")
    expanded = (
        samples_df.lazy()
        .join(
            det_df.lazy().select(COL_IMAGE_ID, COL_SCORE, COL_IS_TP),
            on=COL_IMAGE_ID,
            how="left",
        )
        .drop_nulls(COL_SCORE)
    )

    # Compute PR curve per bootstrap: sort by score, cumulative TP/FP, then
    # trapezoidal AUC via diff + product.
    pr_per_boot = (
        expanded.sort("bootstrap_id", COL_SCORE, descending=[False, True])
        .with_columns(
            cum_tp=pl.col(COL_IS_TP).cast(pl.Int64).cum_sum().over("bootstrap_id"),
            cum_fp=(~pl.col(COL_IS_TP)).cast(pl.Int64).cum_sum().over("bootstrap_id"),
        )
        .join(boot_gts, on="bootstrap_id", how="left")
        .with_columns(
            precision=pl.col("cum_tp")
            / (pl.col("cum_tp") + pl.col("cum_fp")).cast(pl.Float64),
            recall=pl.col("cum_tp").cast(pl.Float64) / pl.col("total_gts"),
        )
    )

    # Trapezoidal AUC per bootstrap_id using shift + diff
    auc_per_boot = (
        pr_per_boot.with_columns(
            d_recall=(
                pl.col("recall") - pl.col("recall").shift(1).over("bootstrap_id")
            ).fill_null(0.0),
            avg_precision=(
                (
                    pl.col("precision")
                    + pl.col("precision").shift(1).over("bootstrap_id")
                )
                / 2.0
            ).fill_null(pl.col("precision")),
        )
        .with_columns(
            slice_area=pl.col("d_recall") * pl.col("avg_precision"),
        )
        .group_by("bootstrap_id")
        .agg(ap=pl.col("slice_area").sum())
        .sort("bootstrap_id")
        .collect(engine="streaming")
    )

    distribution = auc_per_boot["ap"].to_list()
    return _finalize(distribution, point, confidence)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _generate_bootstrap_samples(
    *,
    image_ids: list[str],
    strata: dict[str, str] | None,
    n_bootstrap: int,
    seed: int | None,
) -> pl.DataFrame:
    """Generate all bootstrap sample rows as a single DataFrame.

    Returns a DataFrame with columns ``bootstrap_id`` (Int32) and
    ``image_id`` (String), with ``n_bootstrap * len(image_ids)`` rows.

    Uses ``pl.Series.sample()`` for Polars-native resampling.

    Args:
        image_ids: Base image IDs.
        strata: Optional image->stratum mapping for stratified sampling.
        n_bootstrap: Number of bootstrap iterations.
        seed: Optional RNG seed.

    Returns:
        DataFrame with ``bootstrap_id`` and ``image_id`` columns.
    """
    boot_ids: list[int] = []
    img_ids: list[str] = []

    if strata is None:
        ids_series = pl.Series(COL_IMAGE_ID, image_ids)
        for b in range(n_bootstrap):
            iter_seed = (seed + b) if seed is not None else None
            sampled = ids_series.sample(
                n=ids_series.len(), with_replacement=True, seed=iter_seed
            )
            boot_ids.extend([b] * sampled.len())
            img_ids.extend(sampled.to_list())
    else:
        grouped: dict[str, list[str]] = {}
        for iid in image_ids:
            key = strata.get(iid, "__missing__")
            grouped.setdefault(key, []).append(iid)
        grouped_series = {k: pl.Series(k, v) for k, v in grouped.items()}

        for b in range(n_bootstrap):
            iter_seed = (seed + b) if seed is not None else None
            for j, (key, s) in enumerate(grouped_series.items()):
                stratum_seed = (
                    (iter_seed * len(grouped_series) + j)
                    if iter_seed is not None
                    else None
                )
                sampled = s.sample(n=s.len(), with_replacement=True, seed=stratum_seed)
                boot_ids.extend([b] * sampled.len())
                img_ids.extend(sampled.to_list())

    return pl.DataFrame(
        {
            "bootstrap_id": pl.Series("bootstrap_id", boot_ids, dtype=pl.Int32),
            COL_IMAGE_ID: pl.Series(COL_IMAGE_ID, img_ids, dtype=pl.String),
        }
    )


def _validate_bootstrap_params(
    n_bootstrap: int,
    confidence: float,
    image_ids: list[str],
) -> None:
    """Validate bootstrap parameters."""
    if n_bootstrap <= 0:
        raise ValueError("`n_bootstrap` must be > 0.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("`confidence` must be in (0, 1).")
    if not image_ids:
        raise ValueError("`image_ids` cannot be empty.")


def _finalize(
    distribution: list[float],
    point_estimate: float,
    confidence: float,
) -> BootstrapResult:
    """Build a ``BootstrapResult`` from a distribution."""
    alpha = (1.0 - confidence) / 2.0
    dist = pl.Series("v", distribution)
    lower = float(dist.quantile(alpha, interpolation="linear"))
    upper = float(dist.quantile(1.0 - alpha, interpolation="linear"))
    return BootstrapResult(
        point_estimate=float(point_estimate),
        ci_lower=lower,
        ci_upper=upper,
        confidence=confidence,
        distribution=[float(v) for v in distribution],
    )


# Backward-compatible alias
bootstrap_metric = bootstrap_metric_sequential
