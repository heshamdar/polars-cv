"""Lazy, group-aware bootstrap confidence intervals for detection AUC metrics.

The public seam is three free functions — :func:`froc_auc_ci_lazy`,
:func:`lroc_auc_ci_lazy`, :func:`average_precision_ci_lazy` — each returning a
``pl.LazyFrame`` and **never collecting internally**. A downstream compiler builds
its plan with no data present, so the confidence interval must stay lazy until the
caller's final ``.collect()``, and must carry one ``ci_lower``/``ci_upper`` row per
group so it can be *joined* onto the point-metric frame instead of looped over in
Python.

Everything is one Polars plan:

* the resample (:func:`_lazy_resample`) is a position-independent hash-expression
  draw over a cross-join skeleton — no materialization, group-partitioned so each
  group resamples within itself and stratified within ``gt_label``;
* the per-replicate metric reuses the existing lazy group-aware authorities
  (``froc_auc`` / ``lroc_auc`` / ``all_points_ap_by_group`` keyed by
  ``bootstrap_id``);
* the interval is a lazy per-group ``quantile`` aggregation
  (:func:`_bootstrap_ci_from_replicates`), with degenerate groups (no positive
  targets) nulling their bounds rather than raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl

from ._types import (
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IS_TP,
    COL_N_GTS,
    COL_SCORE,
)

if TYPE_CHECKING:
    from .._auc import CorrectionMethod
    from ._types import DetectionTable

# Internal slot column carrying a globally-unique, deterministic per-draw id.
_COL_BOOT = "bootstrap_id"
_COL_SLOT = "_slot"


def _normalize_group_by(group_by: str | list[str] | None) -> list[str]:
    """Normalize the ``group_by`` argument to a list of column names."""
    if group_by is None:
        return []
    if isinstance(group_by, str):
        return [group_by]
    return list(group_by)


def _validate_ci_params(n_bootstrap: int, confidence: float) -> None:
    """Validate the shared bootstrap parameters."""
    if n_bootstrap <= 0:
        raise ValueError("`n_bootstrap` must be > 0.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("`confidence` must be in (0, 1).")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def froc_auc_ci_lazy(
    table: DetectionTable,
    *,
    group_by: str | list[str] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    method: Literal["trapezoidal", "mann_whitney"] = "trapezoidal",
    fp_range: tuple[float, float] | None = None,
    correction: CorrectionMethod = None,
    level: Literal["detection", "image"] = "detection",
    sample_col: str | None = None,
) -> pl.LazyFrame:
    """Lazy, group-aware bootstrap confidence interval for FROC AUC.

    Returns a ``LazyFrame`` with columns ``[*group_by, auc, ci_lower, ci_upper]``
    — one row per group (a single row when ``group_by`` is ``None``). The ``auc``
    column is the deterministic point estimate (``froc_auc``); only the bounds are
    bootstrapped, and they are seed-reproducible. Nothing is collected here — the
    caller owns the collect.

    Each group resamples within itself (its own images, to its own size),
    stratified by ``gt_label``. A **degenerate group** (no positive targets) keeps
    its point estimate but yields null ``ci_lower``/``ci_upper`` instead of raising.

    Args:
        table: Canonical detection table.
        group_by: Optional grouping column(s). ``None`` yields one ungrouped row.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed. ``None`` maps to a fixed constant, so the bounds
            are deterministic even without an explicit seed.
        method: ``"trapezoidal"`` or ``"mann_whitney"``.
        fp_range: Optional ``(lo, hi)`` partial-AUC range (trapezoidal only).
        correction: Partial-AUC correction (trapezoidal only).
        level: Mann-Whitney granularity — ``"detection"`` or ``"image"``.
        sample_col: Optional entity column (e.g. ``"case_id"``) to resample at the
            entity level within each group, expanding to images.

    Returns:
        ``LazyFrame`` with ``[*group_by, auc, ci_lower, ci_upper]``.
    """
    from ._metrics import froc_auc

    _validate_ci_params(n_bootstrap, confidence)
    group_keys = _normalize_group_by(group_by)

    def metric(tbl: DetectionTable, keys: list[str]) -> pl.LazyFrame:
        return froc_auc(
            tbl,
            method=method,
            fp_range=fp_range,
            correction=correction,
            level=level,
            group_by=keys or None,
        )

    empty_value = 0.5 if method == "mann_whitney" else 0.0
    return _auc_ci_lazy(
        table,
        metric=metric,
        group_keys=group_keys,
        value_col="auc",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        sample_col=sample_col,
        empty_value=empty_value,
        require_both_classes=method == "mann_whitney",
    )


def lroc_auc_ci_lazy(
    table: DetectionTable,
    *,
    group_by: str | list[str] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    variant: Literal["best_tp", "top_scoring"] = "best_tp",
    method: Literal["trapezoidal", "mann_whitney"] = "trapezoidal",
    fpf_range: tuple[float, float] | None = None,
    correction: CorrectionMethod = None,
    level: Literal["detection", "image"] = "image",
    sample_col: str | None = None,
) -> pl.LazyFrame:
    """Lazy, group-aware bootstrap confidence interval for LROC AUC.

    The LROC counterpart of :func:`froc_auc_ci_lazy`; see it for the shared
    behavior. Returns ``[*group_by, auc, ci_lower, ci_upper]``.

    Args:
        table: Canonical detection table.
        group_by: Optional grouping column(s). ``None`` yields one ungrouped row.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed (``None`` → deterministic constant).
        variant: ``"best_tp"`` or ``"top_scoring"``.
        method: ``"trapezoidal"`` or ``"mann_whitney"``.
        fpf_range: Optional ``(lo, hi)`` partial-AUC range (trapezoidal only).
        correction: Partial-AUC correction (trapezoidal only).
        level: Mann-Whitney granularity — ``"image"`` or ``"detection"``.
        sample_col: Optional entity column to resample at the entity level.

    Returns:
        ``LazyFrame`` with ``[*group_by, auc, ci_lower, ci_upper]``.
    """
    from ._metrics import lroc_auc

    _validate_ci_params(n_bootstrap, confidence)
    group_keys = _normalize_group_by(group_by)

    def metric(tbl: DetectionTable, keys: list[str]) -> pl.LazyFrame:
        return lroc_auc(
            tbl,
            variant=variant,
            method=method,
            fpf_range=fpf_range,
            correction=correction,
            level=level,
            group_by=keys or None,
        )

    empty_value = 0.5 if method == "mann_whitney" else 0.0
    return _auc_ci_lazy(
        table,
        metric=metric,
        group_keys=group_keys,
        value_col="auc",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        sample_col=sample_col,
        empty_value=empty_value,
        require_both_classes=method == "mann_whitney",
    )


def average_precision_ci_lazy(
    table: DetectionTable,
    *,
    group_by: str | list[str] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    class_id: str | None = None,
    sample_col: str | None = None,
) -> pl.LazyFrame:
    """Lazy, group-aware bootstrap confidence interval for all-points AP.

    Returns ``[*group_by, ap, ci_lower, ci_upper]``. The ``ap`` column is the
    deterministic point estimate (the same all-points estimator as
    :func:`~polars_cv.metrics.average_precision`); only the bounds are
    bootstrapped. Nothing is collected here.

    Args:
        table: Canonical detection table.
        group_by: Optional grouping column(s). ``None`` yields one ungrouped row.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed (``None`` → deterministic constant).
        class_id: Optional class filter applied before sampling and scoring.
        sample_col: Optional entity column to resample at the entity level.

    Returns:
        ``LazyFrame`` with ``[*group_by, ap, ci_lower, ci_upper]``.
    """
    _validate_ci_params(n_bootstrap, confidence)
    group_keys = _normalize_group_by(group_by)

    if class_id is not None:
        table = table.filter_class(class_id)

    def metric(tbl: DetectionTable, keys: list[str]) -> pl.LazyFrame:
        return _all_points_ap_grouped(tbl, keys)

    return _auc_ci_lazy(
        table,
        metric=metric,
        group_keys=group_keys,
        value_col="ap",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        sample_col=sample_col,
        empty_value=0.0,
    )


# ---------------------------------------------------------------------------
# Shared CI composition
# ---------------------------------------------------------------------------


def _auc_ci_lazy(
    table: DetectionTable,
    *,
    metric,
    group_keys: list[str],
    value_col: str,
    n_bootstrap: int,
    confidence: float,
    seed: int | None,
    sample_col: str | None,
    empty_value: float,
    require_both_classes: bool = False,
) -> pl.LazyFrame:
    """Compose point estimate + per-group bootstrap quantiles into one plan.

    ``metric(table, keys)`` returns a lazy ``[*keys, value_col]`` frame for the
    given grouping (the shared lazy authority for this family). It is called once
    for the point estimate (``keys = group_keys``) and once per replicate
    (``keys = [*group_keys, bootstrap_id]``). ``require_both_classes`` tightens the
    degeneracy rule for the two-class rank statistics (Mann-Whitney).
    """
    point = metric(table, group_keys)

    samples = _resolve_bootstrap_samples(
        table,
        sample_col=sample_col,
        n_bootstrap=n_bootstrap,
        seed=seed,
        group_keys=group_keys,
    )
    boot = _bootstrap_table_with_draws(table, samples)
    replicates = metric(boot, [*group_keys, _COL_BOOT])

    ci = _bootstrap_ci_from_replicates(
        replicates,
        table,
        group_keys=group_keys,
        value_col=value_col,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        empty_value=empty_value,
        require_both_classes=require_both_classes,
    )
    return _join_point_and_ci(point, ci, group_keys, value_col)


def _bootstrap_ci_from_replicates(
    replicates: pl.LazyFrame,
    table: DetectionTable,
    *,
    group_keys: list[str],
    value_col: str,
    n_bootstrap: int,
    confidence: float,
    empty_value: float,
    require_both_classes: bool = False,
) -> pl.LazyFrame:
    """Per-group percentile bounds from a per-replicate grouped-metric frame.

    ``replicates`` carries ``[*group_keys, bootstrap_id, value_col]`` (one row per
    replicate that produced a value). The complete group set and a per-group
    viability flag come from ``table.image_metadata``. Absent replicates are
    filled with ``empty_value`` (a resample that drew no detections legitimately
    scores ``0.0`` / ``0.5``). A **non-viable group** nulls its bounds instead of
    reporting a spurious interval: viability needs at least one positive target,
    and — for the two-class rank statistics (``require_both_classes``, i.e.
    Mann-Whitney) — at least one negative as well, since the AUC is undefined
    without both classes.

    Returns a ``LazyFrame`` with ``[*group_keys, ci_lower, ci_upper]``.
    """
    alpha = (1.0 - confidence) / 2.0
    meta = table.image_metadata
    viable = pl.col(COL_GT_LABEL).cast(pl.Int64).sum() > 0
    if require_both_classes:
        viable = viable & ((~pl.col(COL_GT_LABEL)).cast(pl.Int64).sum() > 0)
    viable_expr = viable.alias("_viable")

    reps = pl.LazyFrame(
        {_COL_BOOT: pl.int_range(0, n_bootstrap, dtype=pl.Int32, eager=True)}
    )
    rep_marked = replicates.with_columns(pl.lit(1, dtype=pl.Int64).alias("_present"))

    if group_keys:
        groups = meta.group_by(group_keys).agg(viable_expr)
        grid = groups.join(reps, how="cross")
        joined = grid.join(
            rep_marked, on=[*group_keys, _COL_BOOT], how="left"
        ).with_columns(
            pl.col(value_col).fill_null(empty_value),
            pl.col("_present").fill_null(0),
        )
        agg = joined.group_by(group_keys).agg(
            pl.col(value_col).quantile(alpha, "linear").alias("ci_lower"),
            pl.col(value_col).quantile(1.0 - alpha, "linear").alias("ci_upper"),
            pl.col("_present").sum().alias("_n_present"),
            pl.col("_viable").first().alias("_viable"),
        )
    else:
        groups = meta.select(viable_expr)
        grid = groups.join(reps, how="cross")
        joined = grid.join(rep_marked, on=_COL_BOOT, how="left").with_columns(
            pl.col(value_col).fill_null(empty_value),
            pl.col("_present").fill_null(0),
        )
        agg = joined.select(
            pl.col(value_col).quantile(alpha, "linear").alias("ci_lower"),
            pl.col(value_col).quantile(1.0 - alpha, "linear").alias("ci_upper"),
            pl.col("_present").sum().alias("_n_present"),
            pl.col("_viable").first().alias("_viable"),
        )

    viable = pl.col("_viable") & (pl.col("_n_present") > 0)
    return agg.with_columns(
        pl.when(viable).then(pl.col("ci_lower")).otherwise(None).alias("ci_lower"),
        pl.when(viable).then(pl.col("ci_upper")).otherwise(None).alias("ci_upper"),
    ).select(*group_keys, "ci_lower", "ci_upper")


def _join_point_and_ci(
    point: pl.LazyFrame,
    ci: pl.LazyFrame,
    group_keys: list[str],
    value_col: str,
) -> pl.LazyFrame:
    """Join the point estimate and the CI bounds on the group keys."""
    if group_keys:
        return point.join(ci, on=group_keys, how="left").select(
            *group_keys, value_col, "ci_lower", "ci_upper"
        )
    # Both are single-row (or empty) frames; a cross join pairs them.
    return point.join(ci, how="cross").select(value_col, "ci_lower", "ci_upper")


# ---------------------------------------------------------------------------
# Group-aware all-points AP (point + per-replicate), reusing the PR authority
# ---------------------------------------------------------------------------


def _all_points_ap_grouped(
    table: DetectionTable,
    group_keys: list[str],
) -> pl.LazyFrame:
    """All-points AP per group as a lazy ``[*group_keys, ap]`` frame.

    Builds the per-detection ``expanded`` frame (``score``/``is_tp`` plus per-group
    ``total_gts``) and reduces it with the shared
    :func:`~polars_cv.metrics._metrics._precision_recall.all_points_ap_by_group`
    authority — the same estimator the scalar ``average_precision`` uses. An empty
    ``group_keys`` runs under a single dropped dummy group.
    """
    from ._metrics._precision_recall import all_points_ap_by_group

    det = table.detections
    meta = table.image_metadata
    det_names = set(det.collect_schema().names())
    meta_only = [k for k in group_keys if k not in det_names]
    if meta_only:
        det = det.join(
            meta.select(COL_IMAGE_ID, *meta_only).unique(), on=COL_IMAGE_ID, how="left"
        )

    if group_keys:
        gts = meta.group_by(group_keys).agg(
            total_gts=pl.col(COL_N_GTS).sum().cast(pl.Float64)
        )
        expanded = (
            det.select(*group_keys, COL_SCORE, COL_IS_TP)
            .drop_nulls(COL_SCORE)
            .join(gts, on=group_keys, how="left")
        )
        return all_points_ap_by_group(expanded, group_col=group_keys)

    _dummy = "_pr_grp"
    gts = meta.select(total_gts=pl.col(COL_N_GTS).sum().cast(pl.Float64))
    expanded = (
        det.select(COL_SCORE, COL_IS_TP)
        .drop_nulls(COL_SCORE)
        .join(gts, how="cross")
        .with_columns(pl.lit(0, dtype=pl.Int32).alias(_dummy))
    )
    return all_points_ap_by_group(expanded, group_col=_dummy).drop(_dummy)


# ---------------------------------------------------------------------------
# Resample construction (lazy, collect-free, group-partitioned)
# ---------------------------------------------------------------------------


def _resolve_bootstrap_samples(
    table: DetectionTable,
    *,
    sample_col: str | None,
    n_bootstrap: int,
    seed: int | None,
    group_keys: list[str] | None = None,
) -> pl.LazyFrame:
    """Seeded, lazy ``(bootstrap_id, image_id)`` resample frame.

    Image-level (``sample_col is None``) resamples images directly, stratified by
    ``gt_label``. Entity-level (``sample_col`` set) resamples entities and expands
    each drawn entity to its images. Both partition draws within ``group_keys`` so
    an image (or entity) is only ever redrawn to replace one in the same group. The
    whole frame stays lazy — the caller collects once at the streaming boundary.
    """
    group_keys = list(group_keys or [])
    meta = table.image_metadata

    if sample_col is None:
        base = meta.select(COL_IMAGE_ID, COL_GT_LABEL, *group_keys).unique()
        return _lazy_resample(
            base,
            unit_col=COL_IMAGE_ID,
            n_bootstrap=n_bootstrap,
            seed=seed,
            strata_col=COL_GT_LABEL,
            partition_cols=group_keys,
        )

    # Entity-level: resample entities (unstratified) within group, expand to images.
    entity = pl.col(sample_col).cast(pl.String).alias("_entity")
    base = meta.select(entity, *group_keys).unique()
    ent_samples = _lazy_resample(
        base,
        unit_col="_entity",
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata_col=None,
        partition_cols=group_keys,
    )
    ent_map = (
        meta.select(entity, COL_IMAGE_ID)
        .unique()
        .group_by("_entity")
        .agg(pl.col(COL_IMAGE_ID))
    )
    return (
        ent_samples.join(ent_map, on="_entity", how="left")
        .explode(COL_IMAGE_ID, empty_as_null=True)
        .select(_COL_BOOT, COL_IMAGE_ID, _COL_SLOT)
    )


def _bootstrap_table_with_draws(
    table: DetectionTable,
    samples_df: pl.LazyFrame,
) -> DetectionTable:
    """Build a per-replicate ``DetectionTable`` keyed by ``bootstrap_id``.

    Each sampled draw is given a distinct synthetic ``image_id`` so an image drawn
    more than once within a replicate counts once per draw. The ``bootstrap_id``
    rides along on both frames so the grouped metric can key by it. ``samples_df``
    is a ``LazyFrame``; the synthetic id uses the deterministic per-draw ``_slot``,
    so the resulting table — and every grouped metric over it — is reproducible.
    """
    from ._types import DetectionTable

    samples = samples_df.with_columns(
        _draw_uid=pl.col(COL_IMAGE_ID)
        + pl.lit("#d")
        + pl.col(_COL_SLOT).cast(pl.String)
    )

    det_boot = (
        samples.join(table.detections, on=COL_IMAGE_ID, how="left")
        .drop_nulls(COL_SCORE)  # zero-detection draws contribute no rows
        .with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID))
        .drop("_draw_uid", _COL_SLOT)
    )
    meta_boot = (
        samples.join(table.image_metadata, on=COL_IMAGE_ID, how="left")
        .with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID))
        .drop("_draw_uid", _COL_SLOT)
    )
    return DetectionTable.from_matched(
        det_boot, meta_boot, matching_iou_threshold=table._matching_iou_threshold
    )


def _lazy_resample(
    base: pl.LazyFrame,
    *,
    unit_col: str,
    n_bootstrap: int,
    seed: int | None,
    strata_col: str | None = None,
    partition_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Lazy, collect-free ``(bootstrap_id, unit_col, _slot)`` resample frame.

    Every draw is a *position-independent* hash of its global slot id, so the
    result is bit-identical across thread counts and streaming morsels and
    reproducible for a given ``seed``. ``seed=None`` maps to a fixed constant, so
    the draw is deterministic even without an explicit seed.

    The draw skeleton is a **cross-join** of a constant-length reps frame
    (``int_range(0, n_bootstrap)``) against the ``base`` units — so the total unit
    count is never materialized (the old path collected it for a modulus). Sampling
    is stratified within ``strata_col`` and partitioned within ``partition_cols``:
    each ``(partition, stratum)`` is redrawn to its own size, so a grouped resample
    never crosses a group boundary. An empty ``base`` (or empty group) simply
    yields no rows — it does not raise.

    Args:
        base: Distinct sampling units (plus ``strata_col`` / ``partition_cols`` when
            given). Must be a ``LazyFrame``.
        unit_col: Column naming the sampling unit (e.g. ``image_id``).
        n_bootstrap: Number of replicates (> 0).
        seed: Optional RNG seed (``None`` → deterministic constant).
        strata_col: Optional stratum column on ``base``.
        partition_cols: Optional partition columns on ``base`` (e.g. group keys).

    Returns:
        ``LazyFrame`` with ``bootstrap_id`` (Int32), ``unit_col`` and ``_slot``.

    Note:
        The draw is ``hash(slot) % stratum_size``; the modulo bias for ``u64 % n``
        is negligible for realistic unit counts.
    """
    if n_bootstrap <= 0:
        raise ValueError("`n_bootstrap` must be > 0.")

    part = list(partition_cols or [])
    hash_seed = 0 if seed is None else int(seed)
    strata = strata_col if strata_col is not None else "_strata"

    b = base
    if strata_col is None:
        b = b.with_columns(pl.lit(0, dtype=pl.Int32).alias("_strata"))
    keys = [*part, strata]
    # Deterministic order; within-(partition, stratum) index + size; and a global
    # position that seeds the unique per-draw slot id.
    b = (
        b.sort([*keys, unit_col])
        .with_columns(
            _sidx=pl.int_range(pl.len(), dtype=pl.Int64).over(keys),
            _s=pl.len().over(keys),
        )
        .with_row_index("_pos")
    )

    reps = pl.LazyFrame(
        {_COL_BOOT: pl.int_range(0, n_bootstrap, dtype=pl.Int32, eager=True)}
    )
    # Cross join gives, per replicate, one slot per base unit — so a replicate
    # redraws exactly the base's (partition, stratum) sizes. `_ntot` (the base
    # size) is a window count, never a materialized scalar.
    slots = (
        reps.join(b, how="cross")
        .with_columns(_ntot=pl.len().over(_COL_BOOT))
        .with_columns(
            **{
                _COL_SLOT: (
                    pl.col(_COL_BOOT).cast(pl.Int64) * pl.col("_ntot") + pl.col("_pos")
                )
            }
        )
        .with_columns(
            _draw=(pl.col(_COL_SLOT).hash(seed=hash_seed) % pl.col("_s")).cast(pl.Int64)
        )
        .select(_COL_BOOT, *keys, "_draw", _COL_SLOT)
    )
    # Map each slot's within-(partition, stratum) draw index back to a unit.
    return slots.join(
        b.select(*keys, "_sidx", unit_col),
        left_on=[*keys, "_draw"],
        right_on=[*keys, "_sidx"],
        how="left",
    ).select(_COL_BOOT, unit_col, _COL_SLOT)
