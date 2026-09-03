"""AUC integrals as Polars expressions — the single authority for curve→AUC.

The eager scalar integrals in ``_auc.py`` reduced a curve to a Python ``float``.
This module re-expresses the same math as ``pl.Expr`` reductions that run inside
``group_by(group).agg(...)`` (one AUC per group, fully vectorized) *and* in a
plain ``select`` over an ungrouped curve (a single scalar). A ``float`` is then
``lf.select(...).collect().item()`` — never a second implementation of the
integral.

The two stages are kept separate, mirroring the eager path
(``MetricResult._curve_xy`` then ``trapz_auc``):

* :func:`collapse_curve` turns a raw curve into strictly-increasing ``x`` with
  the upper-envelope ``y`` **per group** (the lazy replacement for
  ``_curve_xy``). It changes the row count, so it is a ``LazyFrame`` transform.
  :func:`collapse_scores` is its Mann-Whitney counterpart: one bucket per
  ``(group, distinct score)`` carrying the positive/negative weight mass.
* :func:`trapz_auc_expr` / :func:`partial_auc_expr` / :func:`mann_whitney_auc_expr`
  are the reductions. They assume each group's ``x`` (or score bucket) is already
  unique (i.e. the curve/scores were collapsed), matching the eager helpers'
  precondition.

:func:`interpolate_curve_lazy` is the lazy replacement for
``MetricResult.interpolate`` / ``summary_table``: it reads y at requested x
operating points off a collapsed curve without collecting, so the curve helpers
return a ``LazyFrame`` and the caller owns the collect.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

CorrectionMethod = Literal["normalize", "mcclish"] | None


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

    This is the lazy, group-aware replacement for ``MetricResult._curve_xy``. A
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
# Stage 2 — the integrals, as aggregation/reduction expressions
# ---------------------------------------------------------------------------


def trapz_auc_expr(
    *,
    x: str | pl.Expr = "x",
    y: str | pl.Expr = "y",
    correction: CorrectionMethod = None,
) -> pl.Expr:
    """Trapezoidal integral of ``y(x)`` as a reduction expression.

    Usable inside ``group_by(group).agg(trapz_auc_expr(...))`` (one value per
    group) and in ``select`` over an ungrouped, collapsed curve. Assumes each
    group's ``x`` values are unique — call :func:`collapse_curve` first.

    Args:
        x: Monotonic x-axis column name or expression.
        y: y-axis column name or expression aligned with ``x``.
        correction: ``None`` returns the raw area; ``"normalize"`` divides by the
            observed x-span (average y-value).

    Returns:
        A ``pl.Expr`` reducing to the (corrected) area; ``0.0`` for < 2 points.
    """
    xc = _as_expr(x)
    yc = _as_expr(y)

    x_sorted = xc.sort()
    y_sorted = yc.sort_by(xc)

    dx = x_sorted.diff()
    avg_y = (y_sorted + y_sorted.shift(1)) / 2.0
    # diff()/shift() leave the first element null; the null slice drops out of
    # sum(), reproducing trapz_auc's `.slice(1)`. sum() of an all-null column
    # (< 2 points) is 0.0, matching trapz_auc's early return.
    raw = (dx * avg_y).sum()

    if correction == "normalize":
        span = x_sorted.max() - x_sorted.min()
        return pl.when(span > 0.0).then(raw / span).otherwise(0.0)
    return raw


def partial_auc_expr(
    *,
    x: str | pl.Expr = "x",
    y: str | pl.Expr = "y",
    lo: float,
    hi: float,
    correction: CorrectionMethod = None,
) -> pl.Expr:
    """Partial trapezoidal AUC over ``[lo, hi]`` as a reduction expression.

    Reproduces ``_auc.partial_auc`` without ``search_sorted`` or Python control
    flow: each consecutive segment contributes the trapezoid over its overlap
    with ``[lo, hi]`` (``y`` linearly interpolated at the clamped ends), and the
    region outside the observed x-range is filled flat at the nearest endpoint's
    ``y`` (``partial_auc``'s clamp behaviour). Assumes unique, collapsed ``x``.

    Args:
        x: Monotonic x-axis column name or expression.
        y: y-axis column name or expression aligned with ``x``.
        lo: Lower x bound.
        hi: Upper x bound.
        correction: ``None`` raw; ``"normalize"`` divides by ``(hi - lo)``;
            ``"mcclish"`` applies McClish's standardized correction.

    Returns:
        A ``pl.Expr`` reducing to the (corrected) partial area; ``0.0`` when
        ``hi <= lo`` or the curve is empty.
    """
    lo_f = float(lo)
    hi_f = float(hi)
    span = hi_f - lo_f
    if span <= 0.0:
        return pl.lit(0.0)

    xc = _as_expr(x)
    yc = _as_expr(y)

    x_sorted = xc.sort()
    y_sorted = yc.sort_by(xc)

    x_prev = x_sorted.shift(1)
    y_prev = y_sorted.shift(1)

    seg_lo = pl.max_horizontal(x_prev, pl.lit(lo_f))
    seg_hi = pl.min_horizontal(x_sorted, pl.lit(hi_f))
    seg_width = x_sorted - x_prev
    overlaps = (seg_width > 0.0) & (seg_hi > seg_lo)

    # Linear interpolation of y at the clamped segment ends.
    t_lo = (seg_lo - x_prev) / seg_width
    t_hi = (seg_hi - x_prev) / seg_width
    y_at_lo = y_prev + t_lo * (y_sorted - y_prev)
    y_at_hi = y_prev + t_hi * (y_sorted - y_prev)

    seg_area = (
        pl.when(overlaps)
        .then((seg_hi - seg_lo) * (y_at_lo + y_at_hi) / 2.0)
        .otherwise(0.0)
    )
    interior = seg_area.sum()

    # Flat fill outside the observed range, clamped to [lo, hi] (partial_auc
    # extends the curve at its endpoint y rather than extrapolating).
    x_min = x_sorted.min()
    x_max = x_sorted.max()
    y_first = y_sorted.first()
    y_last = y_sorted.last()

    left_width = (pl.min_horizontal(x_min, pl.lit(hi_f)) - pl.lit(lo_f)).clip(
        lower_bound=0.0
    )
    right_width = (pl.lit(hi_f) - pl.max_horizontal(x_max, pl.lit(lo_f))).clip(
        lower_bound=0.0
    )
    raw = interior + left_width * y_first + right_width * y_last

    # Empty curve → 0.0 (degenerate range already returned above).
    raw = pl.when(xc.count() == 0).then(0.0).otherwise(raw)

    if correction == "normalize":
        return raw / span
    if correction == "mcclish":
        return _mcclish_correction_expr(raw, lo_f, hi_f)
    return raw


def _mcclish_correction_expr(raw: pl.Expr, lo: float, hi: float) -> pl.Expr:
    """McClish standardized partial-AUC correction as an expression.

    Maps a raw partial AUC to ``[0.5, 1.0]`` where ``0.5`` is chance level over
    ``[lo, hi]``. Ports ``_auc.mcclish_correction``; ``lo``/``hi`` are Python
    floats so every bound below is a constant.

    Reference:
        McClish DC. Analyzing a portion of the ROC curve.
        Medical Decision Making. 1989;9(3):190-195.
    """
    span = hi - lo
    if span <= 0:
        return pl.lit(0.5)
    min_pauc = (lo + hi) * span / 2.0  # diagonal (chance-level)
    max_pauc = span  # perfect classifier
    denom = max_pauc - min_pauc
    if denom <= 0:
        return pl.lit(0.5)
    return (1.0 + (raw - min_pauc) / denom) / 2.0


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


def mann_whitney_auc_expr(
    *,
    score: str | pl.Expr = "score",
    w_pos: str | pl.Expr = "w_pos",
    w_neg: str | pl.Expr = "w_neg",
) -> pl.Expr:
    """Weighted Mann-Whitney U AUC as a reduction over collapsed score buckets.

    ``P(positive score > negative score)`` with ties counted at ½, computed as a
    weighted rank-sum: for each score bucket (ascending), the positive mass beats
    all strictly-lower negative mass plus half its tied negative mass. Assumes the
    input is one row per distinct score — call :func:`collapse_scores` first,
    exactly as :func:`trapz_auc_expr` assumes a collapsed curve. Usable inside
    ``group_by(group).agg(...)`` and in ``select``.

    With unit weights the masses are class counts and this reduces to the standard
    tie-averaged Mann-Whitney AUC, so the unweighted case is literally the
    weighted case with every weight ``= 1`` — one implementation, not two.

    Args:
        score: Bucket score column/expression to order by.
        w_pos: Positive weight-mass column/expression.
        w_neg: Negative weight-mass column/expression.

    Returns:
        A ``pl.Expr`` reducing to the MW-U AUC in ``[0, 1]``; ``0.5`` when either
        class has zero mass.
    """
    sc = _as_expr(score)
    wp = _as_expr(w_pos).cast(pl.Float64)
    wn = _as_expr(w_neg).cast(pl.Float64)

    wp_sorted = wp.sort_by(sc)
    wn_sorted = wn.sort_by(sc)
    # Strictly-lower negative mass at each bucket (exclusive prefix). Buckets are
    # unique in score, so cum_sum minus the bucket's own mass is exact.
    cum_neg_below = wn_sorted.cum_sum() - wn_sorted
    numerator = (wp_sorted * (cum_neg_below + 0.5 * wn_sorted)).sum()

    total_pos = wp.sum()
    total_neg = wn.sum()
    auc = numerator / (total_pos * total_neg)
    return pl.when((total_pos == 0) | (total_neg == 0)).then(pl.lit(0.5)).otherwise(auc)


# ---------------------------------------------------------------------------
# Lazy curve interpolation (operating points)
# ---------------------------------------------------------------------------


def interpolate_curve_lazy(
    curve_lf: pl.LazyFrame,
    *,
    x_col: str,
    y_col: str,
    at: list[float],
) -> pl.LazyFrame:
    """Interpolate ``y`` at requested ``x`` operating points, lazily.

    The lazy replacement for ``MetricResult.interpolate`` / ``summary_table``: it
    collapses the curve to the strictly-increasing upper envelope
    (:func:`collapse_curve`), then brackets each query point with a backward and a
    forward as-of join and linearly interpolates. A point outside the observed
    ``[min x, max x]`` yields a null ``y`` (no extrapolation); an exact knot (and
    the endpoints) yields that knot's collapsed ``y``. Nothing is collected — the
    caller owns the collect.

    Args:
        curve_lf: Curve carrying ``x_col`` and ``y_col``.
        x_col: X-axis column name.
        y_col: Y-axis column name.
        at: X operating points to report ``y`` for.

    Returns:
        A ``LazyFrame`` with columns ``[x_col, y_col]``, one row per element of
        ``at`` in the given order; ``y_col`` is Float64 and null off the curve.
    """
    collapsed = collapse_curve(curve_lf, x_col=x_col, y_col=y_col)
    # `sort` after `select` keeps the join key flagged sorted for `join_asof`.
    ref = collapsed.select(pl.col(x_col), _xk=pl.col(x_col), _yk=pl.col(y_col)).sort(
        x_col
    )

    query = (
        pl.LazyFrame({x_col: [float(a) for a in at]})
        .with_columns(pl.col(x_col).cast(pl.Float64))
        .with_row_index("_ord")
        .sort(x_col)
    )
    bracketed = query.join_asof(ref, on=x_col, strategy="backward").rename(
        {"_xk": "_x_lo", "_yk": "_y_lo"}
    )
    bracketed = (
        bracketed.sort(x_col)
        .join_asof(ref, on=x_col, strategy="forward")
        .rename({"_xk": "_x_hi", "_yk": "_y_hi"})
    )

    denom = pl.col("_x_hi") - pl.col("_x_lo")
    t = (
        pl.when(denom != 0.0)
        .then((pl.col(x_col) - pl.col("_x_lo")) / denom)
        .otherwise(0.0)
    )
    interp = pl.col("_y_lo") + t * (pl.col("_y_hi") - pl.col("_y_lo"))
    # Off the observed range (no knot on one side) ⇒ null; exact knot ⇒ its y.
    y_out = (
        pl.when(pl.col("_x_lo").is_null() | pl.col("_x_hi").is_null())
        .then(None)
        .when(pl.col("_x_lo") == pl.col("_x_hi"))
        .then(pl.col("_y_lo"))
        .otherwise(interp)
        .cast(pl.Float64)
    )
    return bracketed.sort("_ord").select(pl.col(x_col), y_out.alias(y_col))
