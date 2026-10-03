"""AUC integrals over lazy curves — the single authority for curve→AUC.

Every AUC in the package reduces through these functions. Each takes a lazy
curve (or score buckets) and returns ``[*keys, auc]``: one row per group, or
one row for the whole frame with no ``keys``. A ``float`` is then
``trapz_auc(lf).collect().item()``. There is no second, eager implementation.
The Series-based ``trapz_auc`` / ``partial_auc`` that once lived in ``_auc.py``
were removed once ``MetricResult.auc`` and the FROC/LROC/PR paths all routed
through here.

They used to be ``pl.Expr`` reductions for ``group_by().agg(...)``. A sort and
a shift inside an aggregation are not native to polars' streaming engine, which
then collected the whole input, and a bootstrap's input is
``n_bootstrap × curve points``. They are now ordered scans per group
(:func:`~._grouped_scan.grouped_scan`) followed by exact sums
(:func:`~._grouped_scan.exact_sums`), all native streaming steps.

The two stages are kept separate:

* :func:`collapse_curve` turns a raw curve into strictly-increasing ``x`` with
  the upper-envelope ``y`` **per group**. :func:`collapse_scores` is its
  Mann-Whitney counterpart: one bucket per ``(group, distinct score)`` carrying
  the positive/negative weight mass.
* :func:`trapz_auc` / :func:`partial_auc` / :func:`mann_whitney_auc` are the
  integrals. They assume each group's ``x`` (or score bucket) is already
  unique (i.e. the curve/scores were collapsed) — call :func:`collapse_curve` or
  :func:`collapse_scores` first.

:func:`interpolate_curve_lazy` is the single lazy interpolation authority: it
reads y at requested x operating points off a collapsed curve without collecting,
so the FROC/LROC curve helpers (``froc_sensitivity_at_fp`` / ``froc_summary_table``
/ ``lroc_sensitivity_at_fpf``) return a ``LazyFrame`` and the caller owns the
collect.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from ._auc import (
    CorrectionMethod,
    Extrapolate,
    validate_correction,
    validate_extrapolate,
)
from ._grouped_scan import CumSum, IsFirst, IsLast, Lag, exact_sums, grouped_scan


def _as_expr(value: str | pl.Expr) -> pl.Expr:
    """Accept a column name or an expression, return an expression."""
    return pl.col(value) if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# Stage 1 — curve geometry (strictly-increasing x, upper-envelope y)
# ---------------------------------------------------------------------------


def collapse_curve(
    lf: pl.LazyFrame,
    *,
    x_col: str,
    y_col: str,
    group_keys: list[str] | None = None,
) -> pl.LazyFrame:
    """Collapse a curve to strictly-increasing ``x`` with the upper-envelope ``y``.

    The single curve-geometry authority for every AUC (``MetricResult.auc`` and
    the FROC/LROC paths all call it). A
    curve carries many rows tied at one ``x`` (a FROC threshold bucket that adds
    only true positives leaves ``fp_per_image`` unchanged); collapsing each tie
    group to its maximum ``y`` is deterministic and is the ROC/FROC convention
    (the best operating point reachable at that ``x``). Grouping by
    ``group_keys`` keeps every group's envelope independent.

    Args:
        lf: Curve frame carrying ``x_col``, ``y_col`` and every group key.
        x_col: Column name for the x-axis.
        y_col: Column name for the y-axis.
        group_keys: Optional grouping columns. When empty/None the whole frame
            is a single group.

    Returns:
        A ``LazyFrame`` with columns ``[*group_keys, x_col, y_col]``, one row per
        ``(group, x)``, ``y`` the maximum over that tie group.
    """
    keys = list(group_keys or [])
    return (
        lf.select(
            *keys,
            pl.col(x_col).cast(pl.Float64),
            pl.col(y_col).fill_null(0.0).cast(pl.Float64),
        )
        .group_by([*keys, x_col])
        .agg(pl.col(y_col).max())
    )


# ---------------------------------------------------------------------------
# Stage 2 — the integrals, per group
# ---------------------------------------------------------------------------


def _scanned_curve(lf: pl.LazyFrame, keys: list[str], x: str, y: str) -> pl.LazyFrame:
    """Each group's curve in ascending ``x`` with the previous point alongside
    (``_xp``/``_yp``, null at the group's first point) and its end flags."""
    return grouped_scan(
        lf.select(*keys, x, y),
        keys,
        by=[x],
        descending=[False],
        _xp=Lag(x),
        _yp=Lag(y),
        _first=IsFirst(),
        _last=IsLast(),
    )


def trapz_auc(
    lf: pl.LazyFrame,
    *,
    x: str = "x",
    y: str = "y",
    keys: Sequence[str] = (),
    correction: CorrectionMethod = None,
    out: str = "auc",
) -> pl.LazyFrame:
    """Trapezoidal integral of ``y(x)``: ``[*keys, out]``, one row per group.

    With no ``keys`` the whole frame is one curve and the result is one row
    (also for an empty frame). Assumes each group's ``x`` values are unique —
    call :func:`collapse_curve` first. The segments sum exactly
    (:func:`~._grouped_scan.exact_sums`), so the area is reproducible bit for
    bit and the plan stays on the streaming engine.

    Args:
        lf: The curve(s).
        x: Monotonic x-axis column.
        y: y-axis column aligned with ``x``.
        keys: Grouping columns.
        correction: ``None`` returns the raw area; ``"normalize"`` divides by the
            observed x-span (average y-value).
        out: The result column.

    Returns:
        The (corrected) area per group; ``0.0`` for < 2 points.
    """
    validate_correction(correction)
    keys = list(keys)
    xc, yc = pl.col(x), pl.col(y)
    segments = _scanned_curve(lf, keys, x, y).with_columns(
        _area=(xc - pl.col("_xp")) * ((yc + pl.col("_yp")) / 2.0)
    )
    summed = exact_sums(
        segments,
        keys,
        extra=[xc.max().alias("_x_max"), xc.min().alias("_x_min")],
        _raw=pl.col("_area"),
    )
    raw = pl.col("_raw")
    if correction == "normalize":
        span = pl.col("_x_max") - pl.col("_x_min")
        raw = pl.when(span > 0.0).then(raw / span).otherwise(0.0)
    return summed.select(*keys, raw.alias(out))


def partial_auc(
    lf: pl.LazyFrame,
    *,
    x: str = "x",
    y: str = "y",
    lo: float,
    hi: float,
    keys: Sequence[str] = (),
    correction: CorrectionMethod = None,
    extrapolate: Extrapolate = "none",
    out: str = "auc",
) -> pl.LazyFrame:
    """Partial trapezoidal AUC over ``[lo, hi]``: ``[*keys, out]`` per group.

    Each consecutive segment contributes the trapezoid over its overlap with
    ``[lo, hi]`` (``y`` linearly interpolated at the clamped ends). Where the
    window leaves the observed x-range the area is not known:
    ``extrapolate="none"`` (default) makes the result null, ``"flat"`` fills it
    at the nearest endpoint's ``y``. :func:`interpolate_curve_lazy` reads
    off-curve points by the same policy. Assumes unique, collapsed ``x``; with
    no ``keys`` the result is one row.

    Args:
        lf: The curve(s).
        x: Monotonic x-axis column.
        y: y-axis column aligned with ``x``.
        lo: Lower x bound.
        hi: Upper x bound.
        keys: Grouping columns.
        correction: ``None`` raw; ``"normalize"`` divides by ``(hi - lo)``,
            giving the mean y-value over the window.
        extrapolate: ``"none"`` (null when ``[lo, hi]`` leaves the curve) or
            ``"flat"`` (extend the endpoints).
        out: The result column.

    Returns:
        The (corrected) partial area per group; ``0.0`` when ``hi <= lo`` or
        the curve is empty; null when the window leaves the curve and
        ``extrapolate="none"``.
    """
    validate_correction(correction)
    validate_extrapolate(extrapolate)
    keys = list(keys)
    lo_f = float(lo)
    hi_f = float(hi)
    span = hi_f - lo_f
    if span <= 0.0:
        groups = lf.select(keys).unique() if keys else pl.LazyFrame({out: [0.0]})
        return groups.with_columns(pl.lit(0.0).alias(out)) if keys else groups

    xc, yc = pl.col(x), pl.col(y)
    x_prev, y_prev = pl.col("_xp"), pl.col("_yp")

    seg_lo = pl.max_horizontal(x_prev, pl.lit(lo_f))
    seg_hi = pl.min_horizontal(xc, pl.lit(hi_f))
    seg_width = xc - x_prev
    overlaps = (seg_width > 0.0) & (seg_hi > seg_lo)

    # Linear interpolation of y at the clamped segment ends.
    t_lo = (seg_lo - x_prev) / seg_width
    t_hi = (seg_hi - x_prev) / seg_width
    y_at_lo = y_prev + t_lo * (yc - y_prev)
    y_at_hi = y_prev + t_hi * (yc - y_prev)

    seg_area = (
        pl.when(overlaps)
        .then((seg_hi - seg_lo) * (y_at_lo + y_at_hi) / 2.0)
        .otherwise(0.0)
    )
    summed = exact_sums(
        _scanned_curve(lf, keys, x, y).with_columns(_seg=seg_area),
        keys,
        extra=[
            xc.min().alias("_x_min"),
            xc.max().alias("_x_max"),
            pl.when(pl.col("_first")).then(yc).max().alias("_y_first"),
            pl.when(pl.col("_last")).then(yc).max().alias("_y_last"),
            xc.count().alias("_n"),
        ],
        _interior=pl.col("_seg"),
    )

    # Flat fill outside the observed range, clamped to [lo, hi]; with
    # extrapolate="none" a window that needs it is null instead (below).
    left_width = (
        pl.min_horizontal(pl.col("_x_min"), pl.lit(hi_f)) - pl.lit(lo_f)
    ).clip(lower_bound=0.0)
    right_width = (
        pl.lit(hi_f) - pl.max_horizontal(pl.col("_x_max"), pl.lit(lo_f))
    ).clip(lower_bound=0.0)
    raw = (
        pl.col("_interior")
        + left_width * pl.col("_y_first")
        + right_width * pl.col("_y_last")
    )

    if extrapolate == "none":
        off_curve = (left_width > 0.0) | (right_width > 0.0)
        raw = pl.when(off_curve).then(None).otherwise(raw)

    # Empty curve → 0.0 (degenerate range already returned above).
    raw = pl.when(pl.col("_n") == 0).then(0.0).otherwise(raw)

    if correction == "normalize":
        raw = raw / span
    return summed.select(*keys, raw.alias(out))


def collapse_scores(
    lf: pl.LazyFrame,
    *,
    score: str | pl.Expr,
    label: str | pl.Expr,
    weight: str | pl.Expr | None = None,
    group_keys: list[str] | None = None,
) -> pl.LazyFrame:
    """Collapse rows to one bucket per ``(group, distinct score)`` for MW-U.

    The Mann-Whitney counterpart of :func:`collapse_curve`: a weighted rank
    statistic with ties needs the per-score positive/negative *weight* mass, and
    Polars' ``rank`` produces unweighted mid-ranks. Bucketing by distinct score
    removes every tie, so the downstream reduction is a plain cumulative sum that
    is correct for weighted and unweighted (weight ``= 1``) data alike. Rows with
    a null score are dropped — an unrankable score contributes to neither class,
    matching the eager reference.

    Args:
        lf: Frame carrying ``score``, ``label``, optional ``weight`` and every
            group key.
        score: Score column name or expression.
        label: Positive/negative flag (truthy = positive).
        weight: Per-row weight column/expression; ``None`` means unit weights, so
            the collapsed masses are plain counts and the reduction is the
            standard tie-averaged Mann-Whitney AUC.
        group_keys: Optional grouping columns (empty/None ⇒ one group).

    Returns:
        A ``LazyFrame`` with ``[*group_keys, "score", "w_pos", "w_neg"]``, one row
        per ``(group, score)``: ``w_pos`` the positive weight mass at that score,
        ``w_neg`` the negative mass.
    """
    keys = list(group_keys or [])
    sc = _as_expr(score)
    lb = _as_expr(label).cast(pl.Float64)
    w = pl.lit(1.0) if weight is None else _as_expr(weight).cast(pl.Float64)
    return (
        lf.select(
            *keys,
            sc.alias("score"),
            lb.alias("_lb"),
            w.alias("_w"),
        )
        .filter(pl.col("score").is_not_null())
        .group_by([*keys, "score"])
        .agg(
            w_pos=(pl.col("_lb") * pl.col("_w")).sum(),
            w_neg=((1.0 - pl.col("_lb")) * pl.col("_w")).sum(),
        )
    )


def mann_whitney_auc(
    lf: pl.LazyFrame,
    *,
    keys: Sequence[str] = (),
    score: str = "score",
    w_pos: str = "w_pos",
    w_neg: str = "w_neg",
    out: str = "auc",
) -> pl.LazyFrame:
    """Weighted Mann-Whitney U AUC over collapsed score buckets, per group.

    ``P(positive score > negative score)`` with ties counted at ½, computed as a
    weighted rank-sum: for each score bucket (ascending), the positive mass beats
    all strictly-lower negative mass plus half its tied negative mass. Assumes the
    input is one row per distinct score — call :func:`collapse_scores` first,
    exactly as :func:`trapz_auc` assumes a collapsed curve. With no ``keys`` the
    result is one row.

    With unit weights the masses are class counts and this reduces to the standard
    tie-averaged Mann-Whitney AUC, so the unweighted case is literally the
    weighted case with every weight ``= 1`` — one implementation, not two.

    Args:
        lf: Score buckets.
        keys: Grouping columns.
        score: Bucket score column to order by.
        w_pos: Positive weight-mass column.
        w_neg: Negative weight-mass column.
        out: The result column.

    Returns:
        ``[*keys, out]``: the MW-U AUC in ``[0, 1]``; ``0.5`` when either class
        has zero mass.
    """
    keys = list(keys)
    wp, wn = pl.col("_wp"), pl.col("_wn")
    buckets = lf.select(
        *keys,
        pl.col(score),
        pl.col(w_pos).cast(pl.Float64).alias("_wp"),
        pl.col(w_neg).cast(pl.Float64).alias("_wn"),
    )
    # Strictly-lower negative mass at each bucket (exclusive prefix). Buckets are
    # unique in score, so the running sum minus the bucket's own mass is exact.
    scanned = grouped_scan(
        buckets, keys, by=[score], descending=[False], _cum_neg=CumSum("_wn")
    ).with_columns(_beats=wp * ((pl.col("_cum_neg") - wn) + 0.5 * wn))
    summed = exact_sums(scanned, keys, _num=pl.col("_beats"), _pos=wp, _neg=wn)
    auc = pl.col("_num") / (pl.col("_pos") * pl.col("_neg"))
    return summed.select(
        *keys,
        pl.when((pl.col("_pos") == 0) | (pl.col("_neg") == 0))
        .then(pl.lit(0.5))
        .otherwise(auc)
        .alias(out),
    )


# ---------------------------------------------------------------------------
# Lazy curve interpolation (operating points)
# ---------------------------------------------------------------------------


def interpolate_curve_lazy(
    curve_lf: pl.LazyFrame,
    *,
    x_col: str,
    y_col: str,
    at: list[float],
    extrapolate: Extrapolate = "none",
    group_keys: list[str] | None = None,
) -> pl.LazyFrame:
    """Interpolate ``y`` at requested ``x`` operating points, lazily.

    The single lazy interpolation authority (the FROC/LROC sensitivity/summary
    helpers build on it): it
    collapses the curve to the strictly-increasing upper envelope
    (:func:`collapse_curve`), then brackets each query point with a backward and a
    forward as-of join and linearly interpolates. A point outside the observed
    ``[min x, max x]`` yields a null ``y`` with ``extrapolate="none"`` (default)
    or the nearest endpoint's ``y`` with ``"flat"`` — the policy
    :func:`partial_auc_expr` integrates by. An exact knot (and the endpoints)
    yields that knot's collapsed ``y``. With ``group_keys`` every group's curve
    is read independently. Nothing is collected — the caller owns the collect.

    Args:
        curve_lf: Curve carrying ``x_col``, ``y_col`` and every group key.
        x_col: X-axis column name.
        y_col: Y-axis column name.
        at: X operating points to report ``y`` for.
        extrapolate: ``"none"`` (null off the curve) or ``"flat"``.
        group_keys: Optional grouping columns; ``None`` reads one curve.

    Returns:
        A ``LazyFrame`` with columns ``[*group_keys, x_col, y_col]``, one row per
        group and element of ``at`` (in the given order within a group);
        ``y_col`` is Float64 and null off the curve.
    """
    validate_extrapolate(extrapolate)
    keys = list(group_keys or [])
    collapsed = collapse_curve(curve_lf, x_col=x_col, y_col=y_col, group_keys=keys)
    # `sort` after `select` keeps the join key flagged sorted for `join_asof`.
    ref = collapsed.select(
        *keys, pl.col(x_col), _xk=pl.col(x_col), _yk=pl.col(y_col)
    ).sort(x_col)

    points = (
        pl.LazyFrame({x_col: [float(a) for a in at]})
        .with_columns(pl.col(x_col).cast(pl.Float64))
        .with_row_index("_ord")
    )
    query = (
        collapsed.select(keys).unique().join(points, how="cross") if keys else points
    ).sort(x_col)
    by = keys or None
    # Both sides are sorted on `x_col` globally, so every `by` group is too;
    # polars cannot verify that per group and would warn on every call.
    bracketed = query.join_asof(
        ref, on=x_col, by=by, strategy="backward", check_sortedness=False
    ).rename({"_xk": "_x_lo", "_yk": "_y_lo"})
    bracketed = (
        bracketed.sort(x_col)
        .join_asof(ref, on=x_col, by=by, strategy="forward", check_sortedness=False)
        .rename({"_xk": "_x_hi", "_yk": "_y_hi"})
    )

    denom = pl.col("_x_hi") - pl.col("_x_lo")
    t = (
        pl.when(denom != 0.0)
        .then((pl.col(x_col) - pl.col("_x_lo")) / denom)
        .otherwise(0.0)
    )
    interp = pl.col("_y_lo") + t * (pl.col("_y_hi") - pl.col("_y_lo"))
    # Off the observed range (no knot on one side) ⇒ null, or the knot on the
    # other side under "flat"; exact knot ⇒ its y.
    below = pl.col("_x_lo").is_null()
    above = pl.col("_x_hi").is_null()
    off = (
        pl.when(below).then(pl.col("_y_hi")).otherwise(pl.col("_y_lo"))
        if extrapolate == "flat"
        else pl.lit(None, dtype=pl.Float64)
    )
    y_out = (
        pl.when(below | above)
        .then(off)
        .when(pl.col("_x_lo") == pl.col("_x_hi"))
        .then(pl.col("_y_lo"))
        .otherwise(interp)
        .cast(pl.Float64)
    )
    return bracketed.sort(*keys, "_ord").select(
        *keys, pl.col(x_col), y_out.alias(y_col)
    )
