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

    # Join samples with detections. The per-bootstrap total_gts is joined in
    # here too, *before* the sort below: every operation after the sort
    # (cum_sum, the precision envelope, the trapezoid shift) depends on rows
    # staying in sorted-by-score order within each bootstrap group, and a
    # `join` does not preserve row order — its parallel execution reorders
    # rows differently depending on the runtime thread count. A join placed
    # between the sort and those windowed ops therefore scrambles the curve
    # nondeterministically across platforms (e.g. it silently yielded
    # negative AUCs on macOS CI). Keeping the sort as the last row-reordering
    # step guarantees a stable, correct curve everywhere.
    det_df = table.detections.collect(engine="streaming")
    expanded = (
        samples_df.lazy()
        .join(
            det_df.lazy().select(COL_IMAGE_ID, COL_SCORE, COL_IS_TP),
            on=COL_IMAGE_ID,
            how="left",
        )
        .drop_nulls(COL_SCORE)
        .join(boot_gts, on="bootstrap_id", how="left")
    )

    # Compute PR curve per bootstrap: sort by score, cumulative TP/FP, then
    # the same monotone precision envelope average_precision applies (the
    # point estimate's estimator), then trapezoidal AUC via diff + product.
    pr_per_boot = (
        expanded.sort("bootstrap_id", COL_SCORE, descending=[False, True])
        .with_columns(
            cum_tp=pl.col(COL_IS_TP).cast(pl.Int64).cum_sum().over("bootstrap_id"),
            cum_fp=(~pl.col(COL_IS_TP)).cast(pl.Int64).cum_sum().over("bootstrap_id"),
        )
        .with_columns(
            precision=pl.col("cum_tp")
            / (pl.col("cum_tp") + pl.col("cum_fp")).cast(pl.Float64),
            recall=pl.col("cum_tp").cast(pl.Float64) / pl.col("total_gts"),
        )
        # Monotone decreasing envelope per replicate (reverse, cum_max,
        # reverse back) — mirrors `_all_points_ap` so every replicate uses
        # the same estimator as the point estimate.
        .with_columns(
            precision=pl.col("precision")
            .reverse()
            .cum_max()
            .reverse()
            .over("bootstrap_id"),
        )
    )

    # Trapezoidal AUC per bootstrap_id using shift + diff
    auc_per_boot = (
        pr_per_boot.with_columns(
            # Anchor the first point at recall = 0 (fill_null with the current
            # recall so the first slice width is recall − 0), mirroring
            # `_all_points_ap`.
            d_recall=(
                pl.col("recall") - pl.col("recall").shift(1).over("bootstrap_id")
            ).fill_null(pl.col("recall")),
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
# Vectorized bootstrap for FROC / LROC AUC (single Polars lazy plan)
# ---------------------------------------------------------------------------


def _bootstrap_table_with_draws(
    table: DetectionTable,
    samples_df: pl.DataFrame,
) -> DetectionTable:
    """Build a per-replicate ``DetectionTable`` keyed by ``bootstrap_id``.

    Each sampled image draw is given a distinct synthetic ``image_id`` so that an
    image drawn more than once within a replicate counts once per draw — the same
    reason the sequential ``_reconstruct`` renames draws. The ``bootstrap_id``
    rides along on both frames so ``froc_auc`` / ``lroc_auc`` can group by it.
    """
    from ._types import DetectionTable

    samples = (
        samples_df.lazy()
        .with_row_index("_ridx")
        .with_columns(
            _draw_uid=pl.col(COL_IMAGE_ID)
            + pl.lit("#d")
            + pl.col("_ridx").cast(pl.String)
        )
    )

    det_boot = (
        samples.join(table.detections, on=COL_IMAGE_ID, how="left")
        .drop_nulls(COL_SCORE)  # zero-detection draws contribute no rows
        .with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID))
        .drop("_draw_uid", "_ridx")
    )
    meta_boot = (
        samples.join(table.image_metadata, on=COL_IMAGE_ID, how="left")
        .with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID))
        .drop("_draw_uid", "_ridx")
    )
    return DetectionTable.from_matched(
        det_boot, meta_boot, matching_iou_threshold=table._matching_iou_threshold
    )


def _bootstrap_distribution(
    auc_lf: pl.LazyFrame,
    n_bootstrap: int,
    empty_value: float,
) -> list[float]:
    """Collect one AUC per ``bootstrap_id``, filling absent replicates.

    A replicate whose resample produced no detections is absent from a
    detection-grouped result (e.g. Mann-Whitney); it is filled with
    *empty_value* so the distribution always has ``n_bootstrap`` entries in
    replicate order.
    """
    all_ids = pl.LazyFrame(
        {"bootstrap_id": pl.int_range(0, n_bootstrap, dtype=pl.Int32, eager=True)}
    )
    filled = (
        all_ids.join(auc_lf, on="bootstrap_id", how="left")
        .with_columns(pl.col("auc").fill_null(empty_value))
        .sort("bootstrap_id")
        .collect(engine="streaming")
    )
    return filled["auc"].to_list()


def bootstrap_froc_auc(
    table: DetectionTable,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    method: str = "trapezoidal",
    fp_range: tuple[float, float] | None = None,
    correction: str | None = None,
) -> BootstrapResult:
    """Vectorized, seed-reproducible bootstrap for FROC AUC.

    Generates all replicates as one seeded ``(bootstrap_id, image_id)`` frame,
    joins it to the detection table, and computes ``froc_auc`` grouped by
    ``bootstrap_id`` — every replicate in one lazy plan. A given ``seed`` yields
    an identical confidence interval, unlike the sequential path.

    Args:
        table: Canonical detection table.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed (set it for reproducible bounds).
        method: ``"trapezoidal"`` or ``"mann_whitney"``.
        fp_range: Optional ``(lo, hi)`` partial-AUC range (trapezoidal only).
        correction: Partial-AUC correction (trapezoidal only).

    Returns:
        ``BootstrapResult`` with percentile confidence interval.
    """
    from ._metrics import froc_auc

    image_ids, strata = table.image_ids_and_strata()
    # image_ids_and_strata reads a `.unique()`, whose row order is not stable
    # across calls; sort so a given seed samples the same images every time.
    image_ids = sorted(image_ids)
    _validate_bootstrap_params(n_bootstrap, confidence, image_ids)

    point = (
        froc_auc(table, method=method, fp_range=fp_range, correction=correction)
        .collect()
        .item()
    )

    samples_df = _generate_bootstrap_samples(
        image_ids=image_ids, strata=strata, n_bootstrap=n_bootstrap, seed=seed
    )
    boot_table = _bootstrap_table_with_draws(table, samples_df)

    auc_lf = froc_auc(
        boot_table,
        method=method,
        fp_range=fp_range,
        correction=correction,
        group_by="bootstrap_id",
    )
    empty_value = 0.5 if method == "mann_whitney" else 0.0
    distribution = _bootstrap_distribution(auc_lf, n_bootstrap, empty_value)
    return _finalize(distribution, point, confidence)


def bootstrap_lroc_auc(
    table: DetectionTable,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    variant: str = "best_tp",
    method: str = "trapezoidal",
    fpf_range: tuple[float, float] | None = None,
    correction: str | None = None,
    level: str = "image",
) -> BootstrapResult:
    """Vectorized, seed-reproducible bootstrap for LROC AUC.

    The LROC counterpart of :func:`bootstrap_froc_auc`: ``lroc_auc`` grouped by
    ``bootstrap_id`` over one seeded resample frame.

    Args:
        table: Canonical detection table.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed.
        variant: ``"best_tp"`` or ``"top_scoring"``.
        method: ``"trapezoidal"`` or ``"mann_whitney"``.
        fpf_range: Optional ``(lo, hi)`` partial-AUC range (trapezoidal only).
        correction: Partial-AUC correction (trapezoidal only).
        level: Mann-Whitney granularity (``"image"`` or ``"detection"``).

    Returns:
        ``BootstrapResult`` with percentile confidence interval.
    """
    from ._metrics import lroc_auc

    image_ids, strata = table.image_ids_and_strata()
    # See bootstrap_froc_auc: sort for seed-stable sampling.
    image_ids = sorted(image_ids)
    _validate_bootstrap_params(n_bootstrap, confidence, image_ids)

    point = (
        lroc_auc(
            table,
            variant=variant,
            method=method,
            fpf_range=fpf_range,
            correction=correction,
            level=level,
        )
        .collect()
        .item()
    )

    samples_df = _generate_bootstrap_samples(
        image_ids=image_ids, strata=strata, n_bootstrap=n_bootstrap, seed=seed
    )
    boot_table = _bootstrap_table_with_draws(table, samples_df)

    auc_lf = lroc_auc(
        boot_table,
        variant=variant,
        method=method,
        fpf_range=fpf_range,
        correction=correction,
        level=level,
        group_by="bootstrap_id",
    )
    empty_value = 0.5 if method == "mann_whitney" else 0.0
    distribution = _bootstrap_distribution(auc_lf, n_bootstrap, empty_value)
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
