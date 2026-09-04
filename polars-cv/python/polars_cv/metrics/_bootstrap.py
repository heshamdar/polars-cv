"""Bootstrap helpers for detection metrics — sequential and vectorized paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import polars as pl

from ._types import COL_GT_LABEL, COL_IMAGE_ID, COL_IS_TP, COL_N_GTS, COL_SCORE

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

    Generates the resample as a lazy hash-expression frame (``_lazy_resample``),
    joins it with the detection table, and computes all-points AP per bootstrap
    iteration via the shared ``all_points_ap_by_group`` authority -- one
    streaming plan, no eager collect of the source frames.

    Args:
        table: Canonical detection table.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed.
        class_id: Optional class filter.

    Returns:
        ``BootstrapResult`` with percentile confidence interval.
    """
    # Sampling base is the (unfiltered) image population, stratified by gt_label;
    # detections / gts below come from the class-filtered table.
    base = table.image_metadata.select(COL_IMAGE_ID, COL_GT_LABEL).unique()

    if class_id is not None:
        table = table.filter_class(class_id)

    from ._metrics._precision_recall import average_precision

    point = average_precision(table)

    samples = _lazy_resample(
        base,
        unit_col=COL_IMAGE_ID,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        strata_col=COL_GT_LABEL,
    )

    # Get total GTs per bootstrap by joining with metadata (lazy throughout).
    boot_gts = (
        samples.join(
            table.image_metadata.select(COL_IMAGE_ID, COL_N_GTS),
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
    expanded = (
        samples.join(
            table.detections.select(COL_IMAGE_ID, COL_SCORE, COL_IS_TP),
            on=COL_IMAGE_ID,
            how="left",
        )
        .drop_nulls(COL_SCORE)
        .join(boot_gts, on="bootstrap_id", how="left")
    )

    # Per-bootstrap all-points AP via the shared lazy authority (same estimator
    # as the scalar point estimate). A replicate with no detections is absent
    # from the grouped result and filled with AP = 0.
    from ._metrics._precision_recall import all_points_ap_by_group

    auc_lf = all_points_ap_by_group(expanded, group_col="bootstrap_id").rename(
        {"ap": "auc"}
    )
    distribution = _bootstrap_distribution(auc_lf, n_bootstrap, 0.0)
    return _finalize(distribution, point, confidence)


# ---------------------------------------------------------------------------
# Vectorized bootstrap for FROC / LROC AUC (single Polars lazy plan)
# ---------------------------------------------------------------------------


def _resolve_bootstrap_samples(
    table: DetectionTable,
    *,
    sample_col: str | None,
    n_bootstrap: int,
    confidence: float,
    seed: int | None,
) -> pl.LazyFrame:
    """Seeded, lazy ``(bootstrap_id, image_id)`` resample frame.

    Image-level (``sample_col is None``) resamples images directly, stratified by
    ``gt_label``. Entity-level (``sample_col`` set, e.g. ``case_id``) resamples
    entities and expands each drawn entity to its images — the expansion map is
    built lazily (``group_by``/``explode``), never as a Python dict. The whole
    frame is a ``LazyFrame`` so the caller collects once at the streaming
    boundary.
    """
    if sample_col is None:
        base = table.image_metadata.select(COL_IMAGE_ID, COL_GT_LABEL).unique()
        return _lazy_resample(
            base,
            unit_col=COL_IMAGE_ID,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
            strata_col=COL_GT_LABEL,
        )

    # Entity-level: resample entities (unstratified), then expand to images.
    entity = pl.col(sample_col).cast(pl.String).alias("_entity")
    base = table.image_metadata.select(entity).unique()
    ent_samples = _lazy_resample(
        base,
        unit_col="_entity",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        strata_col=None,
    )
    ent_map = (
        table.image_metadata.select(entity, COL_IMAGE_ID)
        .unique()
        .group_by("_entity")
        .agg(pl.col(COL_IMAGE_ID))
    )
    return (
        ent_samples.join(ent_map, on="_entity", how="left")
        # every entity maps to >=1 image, so empty_as_null is a no-op — set it
        # explicitly to pin the Polars-2.0 behaviour and silence the warning.
        .explode(COL_IMAGE_ID, empty_as_null=True)
        .select("bootstrap_id", COL_IMAGE_ID, "_slot")
    )


def _bootstrap_table_with_draws(
    table: DetectionTable,
    samples_df: pl.LazyFrame,
) -> DetectionTable:
    """Build a per-replicate ``DetectionTable`` keyed by ``bootstrap_id``.

    Each sampled image draw is given a distinct synthetic ``image_id`` so that an
    image drawn more than once within a replicate counts once per draw — the same
    reason the sequential ``_reconstruct`` renames draws. The ``bootstrap_id``
    rides along on both frames so ``froc_auc`` / ``lroc_auc`` can group by it.

    ``samples_df`` is a ``LazyFrame``; the synthetic id uses a global row index,
    so it stays unique per draw regardless of the (possibly nondeterministic)
    row order the resample join emits — the per-replicate metric is invariant to
    that labeling.
    """
    from ._types import DetectionTable

    # ``_slot`` is unique per image draw (image-level) and unique per
    # (entity draw, image) (entity-level); pairing it with the base image_id
    # yields a synthetic id that is unique per draw *and* deterministic for a
    # given seed, so the resulting table — and every grouped metric over it — is
    # reproducible.
    samples = samples_df.with_columns(
        _draw_uid=pl.col(COL_IMAGE_ID) + pl.lit("#d") + pl.col("_slot").cast(pl.String)
    )

    det_boot = (
        samples.join(table.detections, on=COL_IMAGE_ID, how="left")
        .drop_nulls(COL_SCORE)  # zero-detection draws contribute no rows
        .with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID))
        .drop("_draw_uid", "_slot")
    )
    meta_boot = (
        samples.join(table.image_metadata, on=COL_IMAGE_ID, how="left")
        .with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID))
        .drop("_draw_uid", "_slot")
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
    level: str = "detection",
    sample_col: str | None = None,
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
        level: Mann-Whitney granularity — ``"detection"`` or ``"image"``.
        sample_col: Optional entity column (e.g. ``"case_id"``) to resample at
            the entity level, expanding to images; ``None`` resamples images.

    Returns:
        ``BootstrapResult`` with percentile confidence interval.
    """
    from ._metrics import froc_auc

    point = (
        froc_auc(
            table, method=method, fp_range=fp_range, correction=correction, level=level
        )
        .collect()
        .item()
    )

    samples_df = _resolve_bootstrap_samples(
        table,
        sample_col=sample_col,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    boot_table = _bootstrap_table_with_draws(table, samples_df)

    auc_lf = froc_auc(
        boot_table,
        method=method,
        fp_range=fp_range,
        correction=correction,
        level=level,
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
    sample_col: str | None = None,
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
        sample_col: Optional entity column to resample at the entity level.

    Returns:
        ``BootstrapResult`` with percentile confidence interval.
    """
    from ._metrics import lroc_auc

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

    samples_df = _resolve_bootstrap_samples(
        table,
        sample_col=sample_col,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
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


def _lazy_resample(
    base: pl.LazyFrame,
    *,
    unit_col: str,
    n_bootstrap: int,
    confidence: float,
    seed: int | None,
    strata_col: str | None = None,
) -> pl.LazyFrame:
    """Lazy, streaming ``(bootstrap_id, unit_col)`` resample-with-replacement frame.

    Every draw is a *position-independent* hash of its global slot index, so the
    result is bit-identical across thread counts and streaming morsels and
    reproducible for a given ``seed`` — the same property that lets the whole
    bootstrap run as one streaming plan instead of a Python loop over eager
    ``pl.Series.sample`` calls. ``seed=None`` maps to a fixed constant, so lazy
    mode is deterministic even without an explicit seed (a deliberate change from
    the old eager path).

    Sampling is stratified within ``strata_col`` — each stratum is redrawn to its
    own size, matching the eager path's per-stratum resampling. ``strata_col=None``
    treats the whole frame as one stratum.

    Args:
        base: Distinct sampling units, one row per unit (plus ``strata_col`` when
            stratifying). Must be a ``LazyFrame``.
        unit_col: Column naming the sampling unit (e.g. ``image_id``).
        n_bootstrap: Number of replicates.
        confidence: Confidence level in ``(0, 1)`` — validated here so the lazy
            paths share one validation point.
        seed: Optional RNG seed (``None`` → deterministic constant).
        strata_col: Optional stratum column on ``base``.

    Returns:
        A ``LazyFrame`` with ``bootstrap_id`` (Int32) and ``unit_col`` (String),
        ``n_bootstrap * n_units`` rows.

    Note:
        The draw is ``hash(slot) % stratum_size``; the modulo bias for
        ``u64 % n`` is negligible for realistic unit counts.
    """
    if n_bootstrap <= 0:
        raise ValueError("`n_bootstrap` must be > 0.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("`confidence` must be in (0, 1).")

    hash_seed = 0 if seed is None else int(seed)
    strata = strata_col if strata_col is not None else "_strata"

    b = base
    if strata_col is None:
        b = b.with_columns(pl.lit(0, dtype=pl.Int32).alias("_strata"))
    # Deterministic order (this replaces the eager path's Python `sorted()` that
    # closed the `.unique()` order-dependence); within-stratum index + stratum
    # size; and a global position for the slot -> stratum lookup below.
    b = (
        b.sort([strata, unit_col])
        .with_columns(
            _sidx=pl.int_range(pl.len(), dtype=pl.Int64).over(strata),
            _s=pl.len().over(strata),
        )
        .with_row_index("_pos")
    )

    n = int(b.select(pl.len()).collect(engine="streaming").item())
    if n == 0:
        raise ValueError("`image_ids` cannot be empty.")

    # n_bootstrap * n draw slots as a streamable integer source.
    slots = pl.select(
        pl.int_range(0, n_bootstrap * n, dtype=pl.Int64).alias("_slot")
    ).lazy()
    slots = slots.with_columns(
        bootstrap_id=(pl.col("_slot") // n).cast(pl.Int32),
        _slot_pos=(pl.col("_slot") % n),
    )
    # Each slot inherits its base unit's stratum + stratum size, then draws a
    # within-stratum index by hashing its own global slot id (position-free).
    slots = slots.join(
        b.select("_pos", strata, "_s"),
        left_on="_slot_pos",
        right_on="_pos",
        how="left",
    ).with_columns(
        _draw=(pl.col("_slot").hash(seed=hash_seed) % pl.col("_s")).cast(pl.Int64),
    )
    # ``_slot`` is a globally-unique, deterministic per-draw id; it rides along so
    # ``_bootstrap_table_with_draws`` can mint reproducible synthetic image ids
    # (a post-hoc ``with_row_index`` over the join output would be
    # order-nondeterministic).
    return slots.join(
        b.select(strata, "_sidx", unit_col),
        left_on=[strata, "_draw"],
        right_on=[strata, "_sidx"],
        how="left",
    ).select("bootstrap_id", unit_col, "_slot")


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
