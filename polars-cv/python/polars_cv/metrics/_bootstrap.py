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
  group resamples within itself and stratified within ``gt_label`` or, for
  weighted tables, the ``weight_scheme``'s weight cells (:func:`_sampling_units`);
* the per-replicate metric reuses the existing lazy group-aware authorities
  (``froc_auc`` / ``lroc_auc`` / ``all_points_ap_by_group`` keyed by
  ``bootstrap_id``);
* the interval is a lazy per-group linear ``quantile``, read as ranks
  (:func:`_bootstrap_ci_from_replicates`), with degenerate groups (no positive
  targets) nulling their bounds rather than raising.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, get_args

import polars as pl

from ._grouped_scan import RowIndex, grouped_scan
from ._types import (
    COL_GT_LABEL,
    COL_IMAGE_ID,
    COL_SCORE,
    COL_WEIGHT,
)

if TYPE_CHECKING:
    from ._auc import CorrectionMethod, Extrapolate
    from ._statistics import Statistic
    from ._types import DetectionTable
    from ._weights import WeightAgg

# Internal slot column carrying a globally-unique, deterministic per-draw id.
_COL_BOOT = "bootstrap_id"
_COL_SLOT = "_slot"
# A sampling unit's weight cell: the sorted distinct weight clusters (crossed
# with any `strata` columns) of its metadata rows.
_COL_CELL = "_cell"
# A metadata row's weight cluster: weights within `weight_rtol` share one.
_COL_WCLUSTER = "_wc"
# An image-level unit's `gt_label` stratum: its label, or null where the draw
# does not stratify on the label (a multi-cell group under "reestimate").
_COL_LSTRATUM = "_label_stratum"
# Whether every weight of a unit is zero (its cell is exempt from the
# singleton rule: it contributes nothing to any statistic).
_COL_ZERO = "_zero_weight"

#: What a table's weights are, and so how the bootstrap draw treats them:
#: ``"reestimate"`` (sample-estimated ``p / q̂``), ``"stratified"`` (cell counts
#: fixed by design) or ``"fixed"`` (known weights). See :func:`froc_auc_ci_lazy`.
WeightScheme = Literal["reestimate", "stratified", "fixed"]

#: The ``weight_rtol`` a cell-forming scheme uses when none is given.
_DEFAULT_WEIGHT_RTOL = 1e-6


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
    weight_rtol: float | None = None,
    weight_scheme: WeightScheme = "reestimate",
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
    not observed, and is never scored as an empty draw's ``0.0`` — or
    **degenerate**, a draw with no positive image (or, for Mann-Whitney, no
    negative one), which only a draw that does not fix the ``gt_label`` counts
    can make (a cell-stratified or ``sample_col`` draw).

    **Weighted tables.** ``weight_scheme`` says what the weights are:

    * ``"reestimate"`` (default) — weights estimated from the sample: an
      importance weight ``p / q̂`` over target distributions (conditional ones
      included), post-stratification, raking over crossed cells. Units (images,
      or ``sample_col`` entities) with the same weight form a *weight cell*, and
      each ``(group, cell)`` is redrawn to its own size. The weighted statistics
      are weight-scale-invariant ratios, so a weight that is a function of its
      cell's count is then exactly the weight re-estimated inside every
      replicate — computed from the weights given, with no target distributions
      needed. ``gt_label`` is **not** crossed with the cells: each cell's
      positive count is random when the weights were estimated over all images,
      and fixing it narrows the interval (in simulation, an SE about 10% low and
      a 95% interval covering 0.90-0.93). Targets conditioned on ``gt_label``
      give each label its own weights, so their cells already carry the label. A
      group with a single cell (unit weights) keeps the ``gt_label`` stratum, so
      its draw is the unweighted one.
    * ``"stratified"`` — each cell's positive count was fixed by the study
      design: the draw keeps every ``(group, gt_label, cell)`` count.
    * ``"fixed"`` — known weights (design weights, a continuous weight): carried
      unchanged on the ``gt_label``-stratified draw, with no weight cells, so
      ``strata`` and ``weight_rtol`` raise.

    Weights computed per group or globally are both exact (draws never cross a
    group). Weights within ``weight_rtol`` of each other share a cell. Under
    ``sample_col`` the draw redraws entities, so a replicate's image mix can
    still wander (entities differ in size): each image's weight is then rescaled
    by ``(n_c/N) / (n*_c/N*)`` — its cell's full-sample over replicate share of
    the group's images — which is the ``p / q̂`` weight re-estimated on the
    replicate (exactly ``1`` for image-level draws). With weight cells, a group
    with a cell holding a **single** unit nulls its bounds: that cell has no
    bootstrap variance, and a continuous weight (every unit its own cell) would
    otherwise report a zero-width interval — use ``"fixed"`` for one. A cell of
    zero weights (images outside the target) is exempt.

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
            more than ``weight_rtol`` times the larger). ``None`` is ``1e-6``,
            which absorbs arithmetic noise; ``0.0`` compares exactly.
        weight_scheme: ``"reestimate"``, ``"stratified"`` or ``"fixed"``: what
            the weights are, and so how the draw treats them (see above).

    Returns:
        ``LazyFrame`` with ``[*group_by, auc, ci_lower, ci_upper]``.
    """
    from ._statistics import FROCAUC

    return bootstrap_ci(
        table,
        FROCAUC(
            fp_range=fp_range,
            method=method,
            correction=correction,
            level=level,
            extrapolate=extrapolate,
        ),
        group_by=group_by,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        sample_col=sample_col,
        strata=strata,
        weight_rtol=weight_rtol,
        weight_scheme=weight_scheme,
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
    weight_rtol: float | None = None,
    weight_scheme: WeightScheme = "reestimate",
) -> pl.LazyFrame:
    """Lazy, group-aware bootstrap confidence interval for LROC AUC.

    The LROC counterpart of :func:`froc_auc_ci_lazy`; see it for the shared
    behavior, ``weight_scheme`` included. Returns
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
        weight_rtol: Relative tolerance within which weights share a cell
            (``None``: ``1e-6``).
        weight_scheme: What the weights are — ``"reestimate"``,
            ``"stratified"`` or ``"fixed"`` (see :func:`froc_auc_ci_lazy`).

    Returns:
        ``LazyFrame`` with ``[*group_by, auc, ci_lower, ci_upper]``.
    """
    from ._statistics import LROCAUC

    return bootstrap_ci(
        table,
        LROCAUC(
            variant=variant,
            method=method,
            fpf_range=fpf_range,
            correction=correction,
            level=level,
            extrapolate=extrapolate,
        ),
        group_by=group_by,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        sample_col=sample_col,
        strata=strata,
        weight_rtol=weight_rtol,
        weight_scheme=weight_scheme,
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
    weight_rtol: float | None = None,
    weight_scheme: WeightScheme = "reestimate",
) -> pl.LazyFrame:
    """Lazy, group-aware bootstrap confidence interval for weighted all-points AP.

    Returns ``[*group_by, ap, ci_lower, ci_upper]``. The ``ap`` column is the
    deterministic point estimate (the same all-points estimator as
    :func:`~polars_cv.metrics.average_precision`); only the bounds are
    bootstrapped. Nothing is collected here. The AP is weighted by
    ``image_metadata.weight`` and its resample follows ``weight_scheme``, as for
    :func:`froc_auc_ci_lazy` (see it for the rules).

    Args:
        table: Canonical detection table.
        group_by: Optional grouping column(s). ``None`` yields one ungrouped row.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed (``None`` → deterministic constant).
        class_id: Optional class filter applied before sampling and scoring.
        sample_col: Optional entity column to resample at the entity level.
        strata: Optional metadata column(s) crossed into the weight cells.
        weight_rtol: Relative tolerance within which weights share a cell
            (``None``: ``1e-6``).
        weight_scheme: What the weights are — ``"reestimate"``,
            ``"stratified"`` or ``"fixed"`` (see :func:`froc_auc_ci_lazy`).

    Returns:
        ``LazyFrame`` with ``[*group_by, ap, ci_lower, ci_upper]``.
    """
    from ._statistics import AP

    if class_id is not None:
        table = table.filter_class(class_id)
    return bootstrap_ci(
        table,
        AP(),
        group_by=group_by,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        sample_col=sample_col,
        strata=strata,
        weight_rtol=weight_rtol,
        weight_scheme=weight_scheme,
    )


def bootstrap_ci(
    table: DetectionTable,
    statistic: Statistic,
    *,
    group_by: str | list[str] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    sample_col: str | None = None,
    strata: str | list[str] | None = None,
    weight_rtol: float | None = None,
    weight_scheme: WeightScheme = "reestimate",
    weight_agg: WeightAgg = "first",
) -> pl.LazyFrame:
    """Lazy, group-aware percentile bootstrap CI for any :class:`Statistic`.

    The one CI engine: ``froc_auc_ci_lazy``, ``lroc_auc_ci_lazy`` and
    ``average_precision_ci_lazy`` are this with ``FROCAUC``, ``LROCAUC`` and
    ``AP``. The point estimate is ``statistic.by_group(table, group_by)``; each
    replicate is the same statistic on a resample of the table keyed by
    ``bootstrap_id``. Resampling, stratification (``gt_label`` and the
    ``weight_scheme``'s weight cells), entity-level draws and the
    degenerate-group rules are those of :func:`froc_auc_ci_lazy` (see it for
    the details).

    Images are the resampling unit and ``group_by`` partitions the draw, so
    keys a statistic averages *over* (a :class:`MeanOver` facet such as
    ``class_id`` or ``iou_threshold``) stay paired: a drawn image brings its
    rows for every class and threshold into the same replicate. Pass them as
    ``group_by`` instead to get one independently-resampled interval per
    facet.

    Args:
        table: Canonical detection table.
        statistic: The statistic to bootstrap, e.g. ``AP("101_point")`` or
            ``mean_ap()``.
        group_by: Optional grouping column(s). ``None`` yields one row.
        n_bootstrap: Number of bootstrap replicates.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed (``None`` → deterministic constant).
        sample_col: Optional entity column to resample at the entity level.
        strata: Optional metadata column(s) crossed into the weight cells.
        weight_rtol: Relative tolerance within which weights share a cell
            (``None``: ``1e-6``).
        weight_scheme: What the weights are — ``"reestimate"``,
            ``"stratified"`` or ``"fixed"`` (see :func:`froc_auc_ci_lazy`).
        weight_agg: Duplicate-weight resolution policy.

    Returns:
        ``LazyFrame`` with ``[*group_by, <statistic.name>, ci_lower, ci_upper]``.
    """
    _validate_ci_params(n_bootstrap, confidence)

    def metric(tbl: DetectionTable, keys: list[str]) -> pl.LazyFrame:
        return statistic.by_group(tbl, keys, weight_agg=weight_agg)

    return _auc_ci_lazy(
        table,
        metric=metric,
        group_keys=_normalize_group_by(group_by),
        value_col=statistic.name,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        sample_col=sample_col,
        strata=_normalize_group_by(strata),
        weight_rtol=weight_rtol,
        weight_scheme=weight_scheme,
        empty_value=statistic.empty_value,
        require_both_classes=statistic.require_both_classes,
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
    weight_rtol: float | None,
    weight_scheme: WeightScheme,
    empty_value: float,
    require_both_classes: bool = False,
) -> pl.LazyFrame:
    """Compose point estimate + per-group bootstrap quantiles into one plan.

    ``metric(table, keys)`` returns a lazy ``[*keys, value_col]`` frame for the
    given grouping (the shared lazy authority for this family). It is called once
    for the point estimate (``keys = group_keys``) and once per replicate
    (``keys = [*group_keys, bootstrap_id]``). ``require_both_classes`` tightens the
    degeneracy rule for the two-class rank statistics (Mann-Whitney). The
    replicates come in batches from :func:`_replicate_tables`, whose
    sampling-unit weight cells also feed the interval's singleton rule.
    """
    point = metric(table, group_keys)

    boots, cells = _replicate_tables(
        table,
        group_keys=group_keys,
        sample_col=sample_col,
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata=strata,
        weight_rtol=weight_rtol,
        weight_scheme=weight_scheme,
        batch=_REPLICATES_PER_BATCH,
    )
    replicates = pl.concat([metric(boot, [*group_keys, _COL_BOOT]) for boot in boots])

    ci = _bootstrap_ci_from_replicates(
        replicates,
        table,
        boots=boots,
        group_keys=group_keys,
        value_col=value_col,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        empty_value=empty_value,
        require_both_classes=require_both_classes,
        cells=cells,
    )
    return _join_point_and_ci(point, ci, group_keys, value_col)


#: Replicates evaluated together. The replicate frame holds
#: ``replicates × detections`` rows and the metric's sorts and group-bys hold
#: all of them, so the batch bounds peak memory: the batches run one after
#: another within the one plan. See ``_replicate_tables``.
_REPLICATES_PER_BATCH = 50


def _replicate_tables(
    table: DetectionTable,
    *,
    group_keys: list[str],
    sample_col: str | None,
    n_bootstrap: int,
    seed: int | None,
    strata: list[str],
    weight_rtol: float | None,
    weight_scheme: WeightScheme,
    batch: int | None,
) -> tuple[list[DetectionTable], pl.LazyFrame | None]:
    """The weighted replicate tables, one per batch, and their sampling units.

    The one way every CI builds its replicates. ``weight_scheme``
    (:func:`_weight_cells`) decides whether the units carry weight cells. With
    cells the resample is stratified on them (:func:`_sampling_units`), and
    under ``sample_col`` each drawn image's weight is rescaled to its cell's
    full-sample share (:func:`_rescale_to_cell_shares`). An image-level draw
    redraws every ``(group, cell)`` to its own size, so its rescale factor
    would be exactly ``1`` and is not applied. ``"fixed"`` has no cells: the
    weights ride along unchanged. Returns ``(replicate tables, units)``; the
    units feed the interval's singleton rule (``None`` without weight cells).

    **Batches.** Replicates ``k·batch .. (k+1)·batch − 1`` form table ``k``
    (``batch=None``: one table). Each batch draws its own range
    (:func:`_lazy_resample`'s ``first``), which is exactly the slice one
    whole-range draw would give. The caller evaluates the metric per table and
    concatenates, and the streaming engine runs those inputs one after another,
    so peak memory follows the batch rather than ``n_bootstrap``. (Filtering
    one cached whole-range draw by ``bootstrap_id`` instead tripped a polars
    optimizer panic, "expected filter", in 1.44.2.)

    The units and each batch's draw are ``cache()``-d (still lazy): the
    replicate metric reads the replicate frames many times, and projection
    pushdown leaves each read a slightly different subplan the streaming engine
    cannot merge, so without a cache it re-runs the draw at every read (~4x
    here). This needs polars >= 1.44.2 (the declared floor): 1.43.2 returned
    wrong rows from a cached frame read under different projections.
    """
    cells, rtol = _weight_cells(weight_scheme, strata, weight_rtol)
    label_in_cells = weight_scheme == "stratified"
    units = _sampling_units(
        table,
        sample_col=sample_col,
        group_keys=group_keys,
        strata=cells,
        weight_rtol=rtol,
        label_in_cells=label_in_cells,
    ).cache()
    image_cells = (
        None
        if sample_col is None or cells is None
        else _sampling_units(
            table,
            sample_col=None,
            group_keys=group_keys,
            strata=cells,
            weight_rtol=rtol,
            label_in_cells=label_in_cells,
        ).cache()
    )
    step = n_bootstrap if batch is None else batch
    tables = []
    for first in range(0, n_bootstrap, step):
        samples = _resolve_bootstrap_samples(
            table,
            sample_col=sample_col,
            n_bootstrap=min(step, n_bootstrap - first),
            seed=seed,
            group_keys=group_keys,
            units=units,
            first=first,
        ).cache()
        tables.append(
            _bootstrap_table_with_draws(
                table, samples, image_cells=image_cells, group_keys=group_keys
            )
        )
    return tables, (None if cells is None else units)


def _weight_cells(
    weight_scheme: WeightScheme, strata: list[str], weight_rtol: float | None
) -> tuple[list[str] | None, float]:
    """The weight-cell columns and tolerance a scheme draws with.

    Returns ``(strata, weight_rtol)`` for :func:`_sampling_units`: ``strata`` is
    ``None`` under ``"fixed"`` (no weight cells), else the given columns (which
    may be ``[]``: the weight alone forms the cells).

    Raises:
        ValueError: ``weight_scheme`` is not a :data:`WeightScheme`, or
            ``"fixed"`` is given ``strata`` or ``weight_rtol`` — both describe
            weight cells, which it does not form, so neither would be read.
    """
    schemes = get_args(WeightScheme)
    if weight_scheme not in schemes:
        raise ValueError(
            f"Unknown weight_scheme {weight_scheme!r}. Expected one of {list(schemes)}."
        )
    if weight_scheme == "fixed":
        for name, given in (
            ("strata", bool(strata)),
            ("weight_rtol", weight_rtol is not None),
        ):
            if given:
                raise ValueError(
                    f"`{name}` describes weight cells, which "
                    "weight_scheme='fixed' does not form (its weights are "
                    "carried unchanged); drop it, or use 'reestimate' or "
                    "'stratified'."
                )
        return None, 0.0
    return strata, _DEFAULT_WEIGHT_RTOL if weight_rtol is None else weight_rtol


def _bootstrap_ci_from_replicates(
    replicates: pl.LazyFrame,
    table: DetectionTable,
    *,
    boots: list[DetectionTable],
    group_keys: list[str],
    value_col: str,
    n_bootstrap: int,
    confidence: float,
    empty_value: float,
    cells: pl.LazyFrame | None,
    require_both_classes: bool = False,
) -> pl.LazyFrame:
    """Per-group percentile bounds from a per-replicate grouped-metric frame.

    ``replicates`` carries ``[*group_keys, bootstrap_id, value_col]`` (one row per
    replicate that produced a row). The complete group set and a per-group
    viability flag come from ``table.image_metadata``. Absent replicates are
    filled with ``empty_value`` (a resample that drew no detections legitimately
    scores ``0.0`` / ``0.5``). A replicate that is *present with a null value*
    is undefined (its curve does not reach the window), which is different: it
    nulls its group's bounds rather than being filled. So does a **degenerate
    replicate**: one whose draw in ``boots`` (the replicate tables) fails the
    viability rule below. Its statistic still has a value (``NaN`` for AP, ``0.5``
    or ``0.0`` for the AUCs), but it describes no resample of the group, so it is
    never scored. A draw that keeps the ``gt_label`` counts cannot be degenerate;
    a cell-stratified (``"reestimate"``) or entity-level one can. A **non-viable group** nulls its bounds instead of
    reporting a spurious interval: viability needs at least one positive target,
    and — for the two-class rank statistics (``require_both_classes``, i.e.
    Mann-Whitney) — at least one negative as well, since the AUC is undefined
    without both classes. Given the sampling units' weight ``cells``
    (:func:`_sampling_units`), a group is also non-viable when any of its weight cells holds a single unit:
    every replicate redraws that unit, so its variance is invisible — and a
    continuous weight, all singletons, would report a zero-width interval. A
    cell whose weights are all zero is exempt (it adds nothing to any
    statistic), and ``cells=None`` (``weight_scheme="fixed"``) has no cells.

    Returns a ``LazyFrame`` with ``[*group_keys, ci_lower, ci_upper]``.
    """
    alpha = (1.0 - confidence) / 2.0
    meta = table._all_rows()[1]
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

    if group_keys:
        groups = meta.group_by(group_keys).agg(viable_expr)
        if cells is not None:
            groups = _require_no_singleton_cell(
                groups.join(
                    _min_cell_size(cells, group_keys),
                    on=group_keys,
                    how="left",
                    nulls_equal=True,
                )
            )
    else:
        groups = meta.select(viable_expr)
        if cells is not None:
            groups = _require_no_singleton_cell(
                groups.join(_min_cell_size(cells, []), how="cross")
            )
    on = [*group_keys, _COL_BOOT]
    degenerate = pl.concat(
        [
            boot._all_rows()[1]
            .group_by(on)
            .agg((~viable).cast(pl.Int64).alias("_degenerate"))
            for boot in boots
        ]
    )
    joined = (
        groups.join(reps, how="cross")
        .join(rep_marked, on=on, how="left", nulls_equal=True)
        .join(degenerate, on=on, how="left", nulls_equal=True)
        .with_columns(
            pl.when(pl.col("_present").is_null())
            .then(pl.lit(empty_value))
            .otherwise(pl.col(value_col))
            .alias(value_col),
            pl.col("_present").fill_null(0),
            pl.max_horizontal(
                pl.col("_undefined").fill_null(0), pl.col("_degenerate").fill_null(0)
            ).alias("_undefined"),
        )
        .drop("_degenerate")
    )
    agg = _percentile_bounds(joined, group_keys, value_col, n_bootstrap, alpha)

    viable = (
        pl.col("_viable") & (pl.col("_n_present") > 0) & (pl.col("_n_undefined") == 0)
    )
    return agg.with_columns(
        pl.when(viable).then(pl.col("ci_lower")).otherwise(None).alias("ci_lower"),
        pl.when(viable).then(pl.col("ci_upper")).otherwise(None).alias("ci_upper"),
    ).select(*group_keys, "ci_lower", "ci_upper")


def _percentile_bounds(
    joined: pl.LazyFrame,
    group_keys: list[str],
    value_col: str,
    n_bootstrap: int,
    alpha: float,
) -> pl.LazyFrame:
    """``ci_lower``/``ci_upper`` and the viability counts per group.

    ``joined`` holds exactly ``n_bootstrap`` rows per group (the replicate grid),
    so the linear-interpolation positions are known up front. The bounds are
    ``quantile(q, "linear")`` computed bit for bit as polars does,
    ``v[⌊h⌋] + (h − ⌊h⌋)·(v[⌊h⌋+1] − v[⌊h⌋])`` with ``h = (n − 1)·q``. Reading
    them as ranks from one grouped scan keeps the plan on the streaming
    engine, where ``quantile`` inside a group-by is not native. A null value
    (an undefined replicate) sorts last; such a group's bounds are nulled by
    ``_n_undefined`` anyway.
    """
    rank, v = pl.col("_rank"), pl.col(value_col)
    picks: list[pl.Expr] = []
    bounds: dict[str, pl.Expr] = {}
    for name, q in (("ci_lower", alpha), ("ci_upper", 1.0 - alpha)):
        h = float(n_bootstrap - 1) * q
        lo = math.floor(h)
        hi = min(lo + 1, n_bootstrap - 1)
        picks += [
            pl.when(rank == lo).then(v).max().alias(f"_{name}_lo"),
            pl.when(rank == hi).then(v).max().alias(f"_{name}_hi"),
        ]
        a, b = pl.col(f"_{name}_lo"), pl.col(f"_{name}_hi")
        bounds[name] = a + (h - lo) * (b - a)
    aggs = [
        *picks,
        pl.col("_present").sum().alias("_n_present"),
        pl.col("_undefined").sum().alias("_n_undefined"),
        pl.col("_viable").first().alias("_viable"),
    ]
    ranked = grouped_scan(
        joined,
        group_keys,
        by=[value_col],
        descending=[False],
        nulls_last=True,
        _rank=RowIndex(),
    )
    agged = ranked.group_by(group_keys).agg(aggs) if group_keys else ranked.select(aggs)
    return agged.with_columns(**bounds).select(
        *group_keys, *bounds, "_n_present", "_n_undefined", "_viable"
    )


def _min_cell_size(cells: pl.LazyFrame, group_keys: list[str]) -> pl.LazyFrame:
    """``[*group_keys, _min_cell]``: each group's smallest weight cell.

    Counts units per ``(group, cell)``; a cell whose weights are all zero is
    left out (an image outside the target adds nothing to any statistic, so a
    lone one hides no variance).
    """
    sizes = cells.group_by(*group_keys, _COL_CELL).agg(
        _n=pl.len(), _zero=pl.col(_COL_ZERO).all()
    )
    smallest = pl.col("_n").filter(~pl.col("_zero")).min().alias("_min_cell")
    return (
        sizes.group_by(group_keys).agg(smallest)
        if group_keys
        else sizes.select(smallest)
    )


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
# Resample construction (lazy, collect-free, group-partitioned)
# ---------------------------------------------------------------------------


def _unit_expr(sample_col: str | None) -> pl.Expr:
    """The sampling unit: the image, or the ``sample_col`` entity (as String)."""
    if sample_col is None:
        return pl.col(COL_IMAGE_ID)
    return pl.col(sample_col).cast(pl.String).alias("_entity")


def _sampling_units(
    table: DetectionTable,
    *,
    sample_col: str | None,
    group_keys: list[str],
    strata: list[str] | None = None,
    weight_rtol: float = 0.0,
    label_in_cells: bool = True,
) -> pl.LazyFrame:
    """One row per sampling unit and group: the resample's base, lazily.

    The single authority for what a unit is and which stratum it is drawn in —
    the resample draws from it, the replicate rescale and the interval's
    singleton rule count it. Columns: ``[unit, *group_keys]``, plus:

    * ``gt_label`` (image-level units): positive if **any** of the image's
      ``(image, class)`` rows is. One row per image, so an image positive for one
      class and negative for another has one draw slot, not one per label.
    * ``_label_stratum`` (image-level units): the ``gt_label`` stratum the draw
      uses — the label, or null in a group of several weight cells unless
      ``label_in_cells``. Crossing the label with the cells fixes each cell's
      positive count, which is random when the weights are estimated over all
      images, and the weighted statistics depend on it: the interval comes out
      too narrow. A group with one cell (unit weights) keeps the label stratum,
      so its draw is the unweighted one.
    * ``_cell`` and ``_zero_weight`` (when ``strata`` is given): the sorted
      distinct weight clusters (:func:`_with_weight_clusters`) — or
      ``(*strata, cluster)`` structs — across the unit's metadata rows, and
      whether all its weights are zero. A weight is a property of an image, so
      the cell is normally one value; a unit whose rows disagree still has one
      well-defined cell, its combination.

    Raises:
        ValueError: A ``strata`` column is not on ``image_metadata``, or
            ``weight_rtol`` is not a finite number ``>= 0`` (both checked at build
            time; nothing collects).
    """
    meta = table._all_rows()[1]
    aggs: list[pl.Expr] = []
    if sample_col is None:
        aggs.append(pl.col(COL_GT_LABEL).any())
    if strata is not None:
        if not (math.isfinite(weight_rtol) and weight_rtol >= 0.0):
            raise ValueError(
                f"`weight_rtol` must be a finite number >= 0, got {weight_rtol!r}."
            )
        names = set(meta.collect_schema().names())
        missing = [c for c in strata if c not in names]
        if missing:
            raise ValueError(
                f"`strata` column(s) {missing!r} not found in image_metadata; "
                f"available: {sorted(names)}."
            )
        extra = [c for c in dict.fromkeys(strata) if c != COL_WEIGHT]
        part = [*group_keys, *(c for c in extra if c not in group_keys)]
        meta = _with_weight_clusters(meta, part, weight_rtol)
        value = pl.struct(*extra, _COL_WCLUSTER) if extra else pl.col(_COL_WCLUSTER)
        # `_cell` stays the last aggregate: the list build is a known streaming
        # fallback (`tests/_streaming_guard.py`), matched by that position.
        aggs.append((pl.col(COL_WEIGHT).fill_null(1.0) == 0.0).all().alias(_COL_ZERO))
        aggs.append(value.unique().sort().alias(_COL_CELL))
    units = meta.group_by(_unit_expr(sample_col), *group_keys).agg(*aggs)
    if sample_col is not None:
        return units
    label = pl.col(COL_GT_LABEL)
    if strata is None or label_in_cells:
        return units.with_columns(label.alias(_COL_LSTRATUM))
    n_cells = units.group_by(*group_keys, _COL_CELL).agg(pl.len())
    n_cells = (
        n_cells.group_by(group_keys).agg(_n_cells=pl.len())
        if group_keys
        else n_cells.select(_n_cells=pl.len())
    )
    joined = (
        units.join(n_cells, on=group_keys, how="left", nulls_equal=True)
        if group_keys
        else units.join(n_cells, how="cross")
    )
    single = pl.col("_n_cells") == 1
    return joined.with_columns(pl.when(single).then(label).alias(_COL_LSTRATUM)).drop(
        "_n_cells"
    )


def _with_weight_clusters(
    meta: pl.LazyFrame, part: list[str], rtol: float
) -> pl.LazyFrame:
    """``meta`` with each row's weight cluster id within ``part``, in one pass.

    Sorted by weight within each ``part``, a new cluster starts wherever a
    weight is neither equal to the one before it nor within ``rtol`` (relative
    to the larger magnitude), so values that differ only by arithmetic noise
    share a cluster without any rounding boundary to straddle. Null weights
    share one cluster, as do NaN weights. A sort and frame-wide scans, every
    step native to the streaming engine and no self-join, so the subplan stays
    cheap however often the replicate frames re-read it.
    """
    w, prev = pl.col(COL_WEIGHT), pl.col(COL_WEIGHT).shift(1)
    close = w.eq_missing(prev) | (
        (w - prev).abs() <= rtol * pl.max_horizontal(w.abs(), prev.abs())
    )
    # A part's first row compares against no predecessor (`prev` null there),
    # as `shift().over(part)` did; the frame-wide shift reads the previous
    # part's last row instead, so a part start is forced to `prev` = null.
    starts = pl.lit(False)
    for k in part:
        starts = starts | pl.col(k).ne_missing(pl.col(k).shift(1))
    close = (
        pl.when(starts | (pl.int_range(pl.len()) == 0))
        .then(w.eq_missing(pl.lit(None, dtype=pl.Float64)))
        .otherwise(close)
    )
    opens = (~close).fill_null(True).cast(pl.Int64)
    # The running count of openings, frame-wide, minus its value before the
    # part began (the part's minimum of the exclusive count, which never
    # decreases): the per-part count, in integers, so exactly. Every step is a
    # native streaming node; `min().over` becomes a group-by and a join.
    # (`min().over` of a column, not of an expression: only the former is
    # rewritten natively.)
    run = pl.col("_opens").cum_sum()
    counted = (
        meta.sort(*part, COL_WEIGHT, maintain_order=True)
        .with_columns(_opens=opens)
        .with_columns(_run=run, _before=run - pl.col("_opens"))
    )
    restart = pl.col("_before").min().over(part) if part else pl.lit(0)
    return counted.with_columns((pl.col("_run") - restart).alias(_COL_WCLUSTER)).drop(
        "_opens", "_run", "_before"
    )


def _resolve_bootstrap_samples(
    table: DetectionTable,
    *,
    sample_col: str | None,
    n_bootstrap: int,
    seed: int | None,
    group_keys: list[str] | None = None,
    units: pl.LazyFrame | None = None,
    first: int = 0,
) -> pl.LazyFrame:
    """Seeded, lazy ``(bootstrap_id, *group_keys, image_id, _slot)`` resample.

    Image-level (``sample_col is None``) resamples images directly, stratified by
    their ``gt_label`` stratum (:func:`_sampling_units`). Entity-level (``sample_col`` set) resamples entities and expands
    each drawn entity to its images. Both partition draws within ``group_keys`` so
    an image (or entity) is only ever redrawn to replace one in the same group,
    and each draw carries its partition's keys so it brings only that group's
    rows. ``units`` (from :func:`_sampling_units`, built with the same
    ``sample_col`` and ``group_keys``) is the base to draw from; its
    ``_label_stratum`` (image level) and ``_cell`` columns are the strata. Without it the base is the plain
    :func:`_sampling_units`. ``first`` starts the replicate range
    (:func:`_lazy_resample`). The whole frame stays lazy — the caller collects
    once at the streaming boundary.
    """
    group_keys = list(group_keys or [])
    meta = table._all_rows()[1]
    unit = _unit_expr(sample_col)
    unit_col = unit.meta.output_name()

    base = (
        _sampling_units(table, sample_col=sample_col, group_keys=group_keys)
        if units is None
        else units
    )
    strata_cols = [_COL_LSTRATUM] if sample_col is None else []
    if _COL_CELL in base.collect_schema().names():
        strata_cols.append(_COL_CELL)

    unit_samples = _lazy_resample(
        base,
        unit_col=unit_col,
        n_bootstrap=n_bootstrap,
        seed=seed,
        strata_cols=strata_cols,
        partition_cols=group_keys,
        first=first,
    )
    if sample_col is None:
        return unit_samples

    # Entity-level: expand each drawn entity to its images within its group.
    ent_map = (
        meta.select(unit, *group_keys, COL_IMAGE_ID)
        .unique()
        .group_by(unit_col, *group_keys)
        .agg(pl.col(COL_IMAGE_ID))
    )
    return (
        unit_samples.join(
            ent_map, on=[unit_col, *group_keys], how="left", nulls_equal=True
        )
        .explode(COL_IMAGE_ID, empty_as_null=True)
        .select(_COL_BOOT, *group_keys, COL_IMAGE_ID, _COL_SLOT)
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

    A draw brings only its own group's rows: metadata joins on ``image_id`` and
    every ``group_keys`` column, detections on ``image_id`` and the group keys
    they carry (a metadata-only key such as ``group_id`` reaches the detections
    through the draw's metadata). Otherwise, under ``group_by="class_id"`` a
    draw in one class's partition would bring the image's other classes too.

    With ``image_cells`` (:func:`_sampling_units` at image level), each drawn
    image's weight is rescaled to its cell's full-sample share of the group's
    images (:func:`_rescale_to_cell_shares`).
    """
    from ._types import DetectionTable

    g = list(group_keys or [])
    samples = samples_df.with_columns(
        _draw_uid=pl.col(COL_IMAGE_ID)
        + pl.lit("#d")
        + pl.col(_COL_SLOT).cast(pl.String)
    )
    det_all, meta_all = table._all_rows()
    det_names = set(det_all.collect_schema().names())
    det_keys = [COL_IMAGE_ID, *(k for k in g if k in det_names)]
    meta_only = [k for k in g if k not in det_names]

    det_boot = (
        samples.drop(*meta_only)
        .join(det_all, on=det_keys, how="left", nulls_equal=True)
        .drop_nulls(COL_SCORE)  # zero-detection draws contribute no rows
        .with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID))
        .drop("_draw_uid", _COL_SLOT)
    )
    meta_boot = samples.join(
        meta_all, on=[COL_IMAGE_ID, *g], how="left", nulls_equal=True
    )
    if image_cells is not None:
        meta_boot = _rescale_to_cell_shares(meta_boot, image_cells, g)
    meta_boot = meta_boot.with_columns(pl.col("_draw_uid").alias(COL_IMAGE_ID)).drop(
        "_draw_uid", _COL_SLOT
    )
    # `replace` keeps every property of the table (a sweep's thresholds
    # included): a replicate is the same kind of table, resampled.
    return replace(
        DetectionTable.from_matched(det_boot, meta_boot),
        _matching_iou_threshold=table._matching_iou_threshold,
        _has_iou=table._has_iou,
        _sweep=table._sweep,
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
    (entity) draw needs, assigned back to its images. A missing weight counts as
    ``1.0``, as in the metrics.
    """
    g = group_keys

    def within(expr: pl.Expr, *extra: str) -> pl.Expr:
        keys = [*extra, *g]
        return expr.over(keys) if keys else expr

    cells = image_cells.select(COL_IMAGE_ID, *g, _COL_CELL)
    full = (
        cells.group_by(*g, _COL_CELL)
        .agg(_n=pl.len())
        .with_columns(_share=pl.col("_n") / within(pl.col("_n").sum()))
        .select(*g, _COL_CELL, "_share")
    )
    drawn = pl.col("_draw_uid").n_unique()
    replicate_share = within(drawn, _COL_BOOT, _COL_CELL) / within(drawn, _COL_BOOT)
    return (
        meta_boot.join(cells, on=[COL_IMAGE_ID, *g], how="left", nulls_equal=True)
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
    first: int = 0,
    strata_cols: list[str] | None = None,
    partition_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Lazy, collect-free ``(bootstrap_id, *partition, unit_col, _slot)`` resample.

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
        first: The first replicate's ``bootstrap_id``: replicates
            ``first .. first + n_bootstrap - 1`` are drawn. A draw hashes its
            global slot id, so a range draws exactly what it would within one
            larger run (how :func:`_replicate_tables` batches).
        strata_cols: Optional stratum columns on ``base``; a stratum is one
            combination of their values.
        partition_cols: Optional partition columns on ``base`` (e.g. group keys).

    Returns:
        ``LazyFrame`` with ``bootstrap_id`` (Int32), the ``partition_cols``,
        ``unit_col`` and ``_slot``.

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
        grouped_scan(b, keys, by=[unit_col], descending=[False], _sidx=RowIndex())
        .with_row_index("_pos")
        .with_columns(_s=pl.len().over(keys))
    )

    reps = pl.LazyFrame(
        {
            _COL_BOOT: pl.int_range(
                first, first + n_bootstrap, dtype=pl.Int32, eager=True
            )
        }
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
        nulls_equal=True,
    ).select(_COL_BOOT, *part, unit_col, _COL_SLOT)
