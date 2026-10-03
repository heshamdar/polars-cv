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
  group resamples within itself and stratified within ``gt_label`` and, for the
  weighted FROC/LROC families, within each weight cell (:func:`_sampling_cells`);
* the per-replicate metric reuses the existing lazy group-aware authorities
  (``froc_auc`` / ``lroc_auc`` / ``all_points_ap_by_group`` keyed by
  ``bootstrap_id``);
* the interval is a lazy per-group ``quantile`` aggregation
  (:func:`_bootstrap_ci_from_replicates`), with degenerate groups (no positive
  targets) nulling their bounds rather than raising.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import polars as pl

from ._types import (
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_IS_TP,
    COL_SCORE,
    COL_WEIGHT,
)

if TYPE_CHECKING:
    from ._auc import CorrectionMethod, Extrapolate
    from ._types import DetectionTable

# Internal slot column carrying a globally-unique, deterministic per-draw id.
_COL_BOOT = "bootstrap_id"
_COL_SLOT = "_slot"
# A sampling unit's weight cell: the sorted distinct weight clusters (crossed
# with any `strata` columns) of its metadata rows.
_COL_CELL = "_cell"
# A metadata row's weight cluster: weights within `weight_rtol` share one.
_COL_WCLUSTER = "_wc"


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
    correction: CorrectionMethod = "normalize",
    level: Literal["detection", "image"] = "detection",
    sample_col: str | None = None,
    extrapolate: Extrapolate = "none",
    strata: str | list[str] | None = None,
    weight_rtol: float = 1e-6,
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
    So does a group any of whose replicates is **undefined** — a resampled curve
    that stops short of ``fp_range`` under ``extrapolate="none"``: its AUC was
    not observed, and is never scored as an empty draw's ``0.0``.

    **Weighted tables.** The resample is stratified on *weight cells* as well as
    ``gt_label``: units (images, or ``sample_col`` entities) with the same weight
    form one cell, and each ``(group, cell)`` is redrawn to its own size. The
    weighted statistics are weight-scale-invariant ratios, so a weight that is a
    function of its cell's count — an importance weight ``p / q̂`` estimated from
    the sample, post-stratification, raking over crossed cells — is then exactly
    the weight re-estimated inside every replicate. Weights computed per group or
    globally are both exact (draws never cross a group). Unit weights form one
    cell, which leaves the resample unchanged. Weights within ``weight_rtol`` of
    each other share a cell. Under ``sample_col`` the draw redraws entities, so a
    replicate's image mix can still wander (entities differ in size): each
    image's weight is then rescaled by ``(n_c/N) / (n*_c/N*)`` — its cell's
    full-sample over replicate share of the group's images — which is the
    ``p / q̂`` weight re-estimated on the replicate (exactly ``1`` for image-level
    draws). A group with a weight cell holding a **single** unit nulls its
    bounds: that cell has no bootstrap variance, and a continuous weight (every
    unit its own cell) would otherwise report a zero-width interval.

    Args:
        table: Canonical detection table.
        group_by: Optional grouping column(s). ``None`` yields one ungrouped row.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed. ``None`` maps to a fixed constant, so the bounds
            are deterministic even without an explicit seed.
        method: ``"trapezoidal"`` or ``"mann_whitney"``.
        fp_range: ``(lo, hi)`` FP-per-image window (**required** for
            ``method="trapezoidal"`` — ``froc_auc`` raises without it).
        correction: Partial-AUC correction (trapezoidal only); ``"normalize"``
            (default) gives mean sensitivity over ``fp_range``.
        level: Mann-Whitney granularity — ``"detection"`` or ``"image"``.
        sample_col: Optional entity column (e.g. ``"case_id"``) to resample at the
            entity level within each group, expanding to images.
        extrapolate: Off-curve policy passed to :func:`froc_auc` for the point
            estimate and every replicate (``"none"`` or ``"flat"``).
        strata: Optional metadata column(s) crossed into the weight cells — the
            columns the weights were computed over (e.g. ``"vendor"``). Needed
            only when two cells share a weight (several at ``1.0``).
        weight_rtol: Relative tolerance within which two weights share a cell
            (sorted distinct weights split where consecutive values differ by
            more than ``weight_rtol`` times the larger). The default ``1e-6``
            absorbs arithmetic noise; ``0.0`` compares exactly.

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
            extrapolate=extrapolate,
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
        strata=_normalize_group_by(strata),
        weight_rtol=weight_rtol,
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
    correction: CorrectionMethod = "normalize",
    level: Literal["detection", "image"] = "image",
    sample_col: str | None = None,
    extrapolate: Extrapolate = "none",
    strata: str | list[str] | None = None,
    weight_rtol: float = 1e-6,
) -> pl.LazyFrame:
    """Lazy, group-aware bootstrap confidence interval for LROC AUC.

    The LROC counterpart of :func:`froc_auc_ci_lazy`; see it for the shared
    behavior, weight-cell stratification included. Returns
    ``[*group_by, auc, ci_lower, ci_upper]``.

    Args:
        table: Canonical detection table.
        group_by: Optional grouping column(s). ``None`` yields one ungrouped row.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed (``None`` → deterministic constant).
        variant: ``"best_tp"`` or ``"top_scoring"``.
        method: ``"trapezoidal"`` or ``"mann_whitney"``.
        fpf_range: ``(lo, hi)`` FPF window (trapezoidal only); defaults to the
            full ``(0.0, 1.0)`` LROC domain.
        correction: Partial-AUC correction (trapezoidal only); ``"normalize"``
            (default) gives mean sensitivity over the window.
        level: Mann-Whitney granularity — ``"image"`` or ``"detection"``.
        sample_col: Optional entity column to resample at the entity level.
        extrapolate: Off-curve policy passed to :func:`lroc_auc`.
        strata: Optional metadata column(s) crossed into the weight cells (see
            :func:`froc_auc_ci_lazy`).
        weight_rtol: Relative tolerance within which weights share a cell.

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
            extrapolate=extrapolate,
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
        strata=_normalize_group_by(strata),
        weight_rtol=weight_rtol,
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
    strata: str | list[str] | None = None,
    weight_rtol: float = 1e-6,
) -> pl.LazyFrame:
    """Lazy, group-aware bootstrap confidence interval for weighted all-points AP.

    Returns ``[*group_by, ap, ci_lower, ci_upper]``. The ``ap`` column is the
    deterministic point estimate (the same all-points estimator as
    :func:`~polars_cv.metrics.average_precision`); only the bounds are
    bootstrapped. Nothing is collected here. The AP is weighted by
    ``image_metadata.weight`` and its resample is stratified on weight cells, as
    for :func:`froc_auc_ci_lazy` (see it for the rules).

    Args:
        table: Canonical detection table.
        group_by: Optional grouping column(s). ``None`` yields one ungrouped row.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed (``None`` → deterministic constant).
        class_id: Optional class filter applied before sampling and scoring.
        sample_col: Optional entity column to resample at the entity level.
        strata: Optional metadata column(s) crossed into the weight cells.
        weight_rtol: Relative tolerance within which weights share a cell.

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
        strata=_normalize_group_by(strata),
        weight_rtol=weight_rtol,
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
    strata: list[str],
    weight_rtol: float,
    empty_value: float,
    require_both_classes: bool = False,
) -> pl.LazyFrame:
    """Compose point estimate + per-group bootstrap quantiles into one plan.

    ``metric(table, keys)`` returns a lazy ``[*keys, value_col]`` frame for the
    given grouping (the shared lazy authority for this family). It is called once
    for the point estimate (``keys = group_keys``) and once per replicate
    (``keys = [*group_keys, bootstrap_id]``). ``require_both_classes`` tightens the
    degeneracy rule for the two-class rank statistics (Mann-Whitney). The
    replicates come from :func:`_replicate_table`, whose sampling-unit weight
    cells also feed the interval's singleton rule.
    """
    point = metric(table, group_keys)

    boot, cells = _replicate_table(
        table,
        group_keys=group_keys,
        sample_col=sample_col,
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata=strata,
        weight_rtol=weight_rtol,
    )
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
        cells=cells,
    )
    return _join_point_and_ci(point, ci, group_keys, value_col)


def _replicate_table(
    table: DetectionTable,
    *,
    group_keys: list[str],
    sample_col: str | None,
    n_bootstrap: int,
    seed: int | None,
    strata: list[str],
    weight_rtol: float,
) -> tuple[DetectionTable, pl.LazyFrame]:
    """The weighted replicate table and its sampling units' weight cells.

    The one way every CI builds its replicates: the resample is stratified on the
    sampling units' weight cells (:func:`_sampling_cells`) and each drawn image's
    weight is rescaled to its cell's full-sample share
    (:func:`_bootstrap_table_with_draws`). Returns ``(replicates, unit_cells)``;
    the cells feed the interval's singleton rule.

    The cells and the draw are ``cache()``-d (still lazy): the replicate metric
    reads the replicate frames many times, and projection pushdown leaves each
    read a slightly different subplan the optimizer cannot merge, so without a
    cache the streaming engine recomputed the resample at every read (~15x).
    """
    image_cells = _sampling_cells(
        table,
        sample_col=None,
        group_keys=group_keys,
        strata=strata,
        weight_rtol=weight_rtol,
    ).cache()
    unit_cells = (
        image_cells
        if sample_col is None
        else _sampling_cells(
            table,
            sample_col=sample_col,
            group_keys=group_keys,
            strata=strata,
            weight_rtol=weight_rtol,
        ).cache()
    )
    samples = _resolve_bootstrap_samples(
        table,
        sample_col=sample_col,
        n_bootstrap=n_bootstrap,
        seed=seed,
        group_keys=group_keys,
        cells=unit_cells,
    ).cache()
    boot = _bootstrap_table_with_draws(
        table, samples, image_cells=image_cells, group_keys=group_keys
    )
    return boot, unit_cells


def _bootstrap_ci_from_replicates(
    replicates: pl.LazyFrame,
    table: DetectionTable,
    *,
    group_keys: list[str],
    value_col: str,
    n_bootstrap: int,
    confidence: float,
    empty_value: float,
    cells: pl.LazyFrame,
    require_both_classes: bool = False,
) -> pl.LazyFrame:
    """Per-group percentile bounds from a per-replicate grouped-metric frame.

    ``replicates`` carries ``[*group_keys, bootstrap_id, value_col]`` (one row per
    replicate that produced a row). The complete group set and a per-group
    viability flag come from ``table.image_metadata``. Absent replicates are
    filled with ``empty_value`` (a resample that drew no detections legitimately
    scores ``0.0`` / ``0.5``). A replicate that is *present with a null value*
    is undefined (its curve does not reach the window), which is different: it
    nulls its group's bounds rather than being filled. A **non-viable group** nulls its bounds instead of
    reporting a spurious interval: viability needs at least one positive target,
    and — for the two-class rank statistics (``require_both_classes``, i.e.
    Mann-Whitney) — at least one negative as well, since the AUC is undefined
    without both classes. Given the sampling units' weight ``cells``
    (:func:`_sampling_cells`), a group is also non-viable when any of its weight cells holds a single unit:
    every replicate redraws that unit, so its variance is invisible — and a
    continuous weight, all singletons, would report a zero-width interval.

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
    rep_marked = replicates.with_columns(
        pl.lit(1, dtype=pl.Int64).alias("_present"),
        pl.col(value_col).is_null().cast(pl.Int64).alias("_undefined"),
    )

    cell_sizes = cells.group_by(*group_keys, _COL_CELL).agg(_n=pl.len())
    min_cell = pl.col("_n").min().alias("_min_cell")

    if group_keys:
        groups = _require_no_singleton_cell(
            meta.group_by(group_keys)
            .agg(viable_expr)
            .join(
                cell_sizes.group_by(group_keys).agg(min_cell),
                on=group_keys,
                how="left",
                nulls_equal=True,
            )
        )
        grid = groups.join(reps, how="cross")
        joined = grid.join(
            rep_marked, on=[*group_keys, _COL_BOOT], how="left"
        ).with_columns(
            pl.when(pl.col("_present").is_null())
            .then(pl.lit(empty_value))
            .otherwise(pl.col(value_col))
            .alias(value_col),
            pl.col("_present").fill_null(0),
            pl.col("_undefined").fill_null(0),
        )
        agg = joined.group_by(group_keys).agg(
            pl.col(value_col).quantile(alpha, "linear").alias("ci_lower"),
            pl.col(value_col).quantile(1.0 - alpha, "linear").alias("ci_upper"),
            pl.col("_present").sum().alias("_n_present"),
            pl.col("_undefined").sum().alias("_n_undefined"),
            pl.col("_viable").first().alias("_viable"),
        )
    else:
        groups = _require_no_singleton_cell(
            meta.select(viable_expr).join(cell_sizes.select(min_cell), how="cross")
        )
        grid = groups.join(reps, how="cross")
        joined = grid.join(rep_marked, on=_COL_BOOT, how="left").with_columns(
            pl.when(pl.col("_present").is_null())
            .then(pl.lit(empty_value))
            .otherwise(pl.col(value_col))
            .alias(value_col),
            pl.col("_present").fill_null(0),
            pl.col("_undefined").fill_null(0),
        )
        agg = joined.select(
            pl.col(value_col).quantile(alpha, "linear").alias("ci_lower"),
            pl.col(value_col).quantile(1.0 - alpha, "linear").alias("ci_upper"),
            pl.col("_present").sum().alias("_n_present"),
            pl.col("_undefined").sum().alias("_n_undefined"),
            pl.col("_viable").first().alias("_viable"),
        )

    viable = (
        pl.col("_viable") & (pl.col("_n_present") > 0) & (pl.col("_n_undefined") == 0)
    )
    return agg.with_columns(
        pl.when(viable).then(pl.col("ci_lower")).otherwise(None).alias("ci_lower"),
        pl.when(viable).then(pl.col("ci_upper")).otherwise(None).alias("ci_upper"),
    ).select(*group_keys, "ci_lower", "ci_upper")


def _require_no_singleton_cell(groups: pl.LazyFrame) -> pl.LazyFrame:
    """Fold ``_min_cell >= 2`` into a groups frame's ``_viable`` flag."""
    return groups.with_columns(
        (pl.col("_viable") & (pl.col("_min_cell") >= 2)).alias("_viable")
    ).drop("_min_cell")


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
    """Weighted all-points AP per group as a lazy ``[*group_keys, ap]`` frame.

    Builds the per-detection ``expanded`` frame (``score``/``is_tp``, the
    detection's resolved ``weight`` and the group's ``gt_mass``) and reduces it
    with the shared
    :func:`~polars_cv.metrics._metrics._precision_recall.all_points_ap_by_group`
    authority — the same estimator the scalar ``average_precision`` uses. An empty
    ``group_keys`` runs under a single dropped dummy group.
    """
    from ._metrics._precision_recall import all_points_ap_by_group
    from ._weights import attach_resolved_weight, weighted_gt_mass

    meta = table.image_metadata
    det = attach_resolved_weight(table.detections, meta)
    det_names = set(det.collect_schema().names())
    meta_only = [k for k in group_keys if k not in det_names]
    if meta_only:
        det = det.join(
            meta.select(COL_IMAGE_ID, *meta_only).unique(), on=COL_IMAGE_ID, how="left"
        )

    columns = (COL_SCORE, COL_IS_TP, COL_WEIGHT)
    if group_keys:
        expanded = (
            det.select(*group_keys, *columns)
            .drop_nulls(COL_SCORE)
            .join(
                weighted_gt_mass(meta, group_keys),
                on=group_keys,
                how="left",
                nulls_equal=True,
            )
        )
        return all_points_ap_by_group(expanded, group_col=group_keys)

    _dummy = "_pr_grp"
    expanded = (
        det.select(*columns)
        .drop_nulls(COL_SCORE)
        .join(weighted_gt_mass(meta, []), how="cross")
        .with_columns(pl.lit(0, dtype=pl.Int32).alias(_dummy))
    )
    return all_points_ap_by_group(expanded, group_col=_dummy).drop(_dummy)


# ---------------------------------------------------------------------------
# Resample construction (lazy, collect-free, group-partitioned)
# ---------------------------------------------------------------------------


def _unit_expr(sample_col: str | None) -> pl.Expr:
    """The sampling unit: the image, or the ``sample_col`` entity (as String)."""
    if sample_col is None:
        return pl.col(COL_IMAGE_ID)
    return pl.col(sample_col).cast(pl.String).alias("_entity")


def _sampling_cells(
    table: DetectionTable,
    *,
    sample_col: str | None,
    group_keys: list[str],
    strata: list[str],
    weight_rtol: float,
) -> pl.LazyFrame:
    """Each sampling unit's weight cell, lazily: ``[unit, *group_keys, _cell]``.

    The single authority for which cell a unit belongs to — the resample
    stratifies on it, the replicate rescale and the interval's singleton rule
    count it. Within each ``(group, *strata)`` the distinct weights are clustered
    (:func:`_weight_clusters`), and a unit's cell is the sorted list of the
    distinct clusters (or ``(*strata, cluster)`` structs) across its metadata
    rows. A weight is a property of an image, so this is normally one value; when
    an image's class rows (or an entity's images) disagree, the unit still has
    one well-defined cell — its combination.

    Raises:
        ValueError: A ``strata`` column is not on ``image_metadata``, or
            ``weight_rtol`` is not a finite number ``>= 0`` (both checked at build
            time; nothing collects).
    """
    if not (math.isfinite(weight_rtol) and weight_rtol >= 0.0):
        raise ValueError(
            f"`weight_rtol` must be a finite number >= 0, got {weight_rtol!r}."
        )
    meta = table.image_metadata
    names = set(meta.collect_schema().names())
    missing = [c for c in strata if c not in names]
    if missing:
        raise ValueError(
            f"`strata` column(s) {missing!r} not found in image_metadata; "
            f"available: {sorted(names)}."
        )
    extra = [c for c in dict.fromkeys(strata) if c != COL_WEIGHT]
    part = [*group_keys, *(c for c in extra if c not in group_keys)]
    labelled = meta.join(
        _weight_clusters(meta, part, weight_rtol),
        on=[*part, COL_WEIGHT],
        how="left",
        nulls_equal=True,
    )
    value = pl.struct(*extra, _COL_WCLUSTER) if extra else pl.col(_COL_WCLUSTER)
    return labelled.group_by(_unit_expr(sample_col), *group_keys).agg(
        value.unique().sort().alias(_COL_CELL)
    )


def _weight_clusters(meta: pl.LazyFrame, part: list[str], rtol: float) -> pl.LazyFrame:
    """Cluster ids for the distinct weights within each ``part``, lazily.

    Sorted ascending, a new cluster starts wherever a weight is not within
    ``rtol`` (relative to the larger magnitude) of the one before it, so values
    that differ only by arithmetic noise share a cluster without any rounding
    boundary to straddle. A null or NaN weight is a cluster of its own. Returns
    ``[*part, weight, _wc]``.
    """
    w = pl.col(COL_WEIGHT)

    def within(expr: pl.Expr) -> pl.Expr:
        return expr.over(part) if part else expr

    prev = within(w.shift(1))
    close = (w - prev).abs() <= rtol * pl.max_horizontal(w.abs(), prev.abs())
    return (
        meta.select(*part, COL_WEIGHT)
        .unique()
        .sort(*part, COL_WEIGHT)
        .with_columns(within((~close).fill_null(True).cum_sum()).alias(_COL_WCLUSTER))
    )


def _resolve_bootstrap_samples(
    table: DetectionTable,
    *,
    sample_col: str | None,
    n_bootstrap: int,
    seed: int | None,
    group_keys: list[str] | None = None,
    cells: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Seeded, lazy ``(bootstrap_id, image_id)`` resample frame.

    Image-level (``sample_col is None``) resamples images directly, stratified by
    ``gt_label``. Entity-level (``sample_col`` set) resamples entities and expands
    each drawn entity to its images. Both partition draws within ``group_keys`` so
    an image (or entity) is only ever redrawn to replace one in the same group.
    ``cells`` (from :func:`_sampling_cells`, built with the same ``sample_col`` and
    ``group_keys``) further stratifies each partition by weight cell. The whole
    frame stays lazy — the caller collects once at the streaming boundary.
    """
    group_keys = list(group_keys or [])
    meta = table.image_metadata
    unit = _unit_expr(sample_col)
    unit_col = unit.meta.output_name()

    base = meta.select(
        unit, *([COL_GT_LABEL] if sample_col is None else []), *group_keys
    )
    base = base.unique()
    strata_cols = [COL_GT_LABEL] if sample_col is None else []
    if cells is not None:
        base = base.join(
            cells, on=[unit_col, *group_keys], how="left", nulls_equal=True
        )
        strata_cols.append(_COL_CELL)

    unit_samples = _lazy_resample(
        base,
        unit_col=unit_col,
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata_cols=strata_cols,
        partition_cols=group_keys,
    )
    if sample_col is None:
        return unit_samples

    # Entity-level: expand each drawn entity to its images.
    ent_map = (
        meta.select(unit, COL_IMAGE_ID)
        .unique()
        .group_by(unit_col)
        .agg(pl.col(COL_IMAGE_ID))
    )
    return (
        unit_samples.join(ent_map, on=unit_col, how="left")
        .explode(COL_IMAGE_ID, empty_as_null=True)
        .select(_COL_BOOT, COL_IMAGE_ID, _COL_SLOT)
    )


def _bootstrap_table_with_draws(
    table: DetectionTable,
    samples_df: pl.LazyFrame,
    *,
    image_cells: pl.LazyFrame | None = None,
    group_keys: list[str] | None = None,
) -> DetectionTable:
    """Build a per-replicate ``DetectionTable`` keyed by ``bootstrap_id``.

    Each sampled draw is given a distinct synthetic ``image_id`` so an image drawn
    more than once within a replicate counts once per draw. The ``bootstrap_id``
    rides along on both frames so the grouped metric can key by it. ``samples_df``
    is a ``LazyFrame``; the synthetic id uses the deterministic per-draw ``_slot``,
    so the resulting table — and every grouped metric over it — is reproducible.

    With ``image_cells`` (:func:`_sampling_cells` at image level), each drawn
    image's weight is rescaled to its cell's full-sample share of the group's
    images (:func:`_rescale_to_cell_shares`).
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
    meta_boot = samples.join(table.image_metadata, on=COL_IMAGE_ID, how="left")
    if image_cells is not None:
        meta_boot = _rescale_to_cell_shares(
            meta_boot, image_cells, list(group_keys or [])
        )
    meta_boot = meta_boot.with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID)).drop(
        "_draw_uid", _COL_SLOT
    )
    return DetectionTable.from_matched(
        det_boot, meta_boot, matching_iou_threshold=table._matching_iou_threshold
    )


def _rescale_to_cell_shares(
    meta_boot: pl.LazyFrame,
    image_cells: pl.LazyFrame,
    group_keys: list[str],
) -> pl.LazyFrame:
    """Rescale each drawn image's weight by ``(n_c/N) / (n*_c/N*)``, lazily.

    ``n_c/N`` is the image's weight cell's share of its group's images in the
    full sample, ``n*_c/N*`` its share of the replicate's drawn images. The
    weighted statistics are scale-invariant within a group, so this is a
    ``p / q̂`` weight re-estimated on the replicate — the recomputation a coarser
    (entity) draw needs, assigned back to its images. An image-level draw redraws
    every ``(group, cell)`` to its own size, so the factor is exactly ``1``
    there. A missing weight counts as ``1.0``, as in the metrics.
    """
    g = group_keys

    def within(expr: pl.Expr, *extra: str) -> pl.Expr:
        keys = [*extra, *g]
        return expr.over(keys) if keys else expr

    full = (
        image_cells.group_by(*g, _COL_CELL)
        .agg(_n=pl.len())
        .with_columns(_share=pl.col("_n") / within(pl.col("_n").sum()))
        .select(*g, _COL_CELL, "_share")
    )
    drawn = pl.col("_draw_uid").n_unique()
    replicate_share = within(drawn, _COL_BOOT, _COL_CELL) / within(drawn, _COL_BOOT)
    return (
        meta_boot.join(image_cells, on=[COL_IMAGE_ID, *g], how="left", nulls_equal=True)
        .join(full, on=[*g, _COL_CELL], how="left", nulls_equal=True)
        .with_columns(
            (
                pl.col(COL_WEIGHT).fill_null(1.0) * (pl.col("_share") / replicate_share)
            ).alias(COL_WEIGHT)
        )
        .drop(_COL_CELL, "_share")
    )


def _lazy_resample(
    base: pl.LazyFrame,
    *,
    unit_col: str,
    n_bootstrap: int,
    seed: int | None,
    strata_cols: list[str] | None = None,
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
    is stratified within ``strata_cols`` and partitioned within ``partition_cols``:
    each ``(partition, stratum)`` is redrawn to its own size, so a grouped resample
    never crosses a group boundary. An empty ``base`` (or empty group) simply
    yields no rows — it does not raise.

    Args:
        base: Distinct sampling units (plus ``strata_cols`` / ``partition_cols``
            when given). Must be a ``LazyFrame``.
        unit_col: Column naming the sampling unit (e.g. ``image_id``).
        n_bootstrap: Number of replicates (> 0).
        seed: Optional RNG seed (``None`` → deterministic constant).
        strata_cols: Optional stratum columns on ``base``; a stratum is one
            combination of their values.
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
    strata = list(strata_cols or [])

    b = base
    if not strata:
        b = b.with_columns(pl.lit(0, dtype=pl.Int32).alias("_strata"))
        strata = ["_strata"]
    keys = [*part, *strata]
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
