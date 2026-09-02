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
* :func:`trapz_auc_expr` / :func:`partial_auc_expr` / :func:`mann_whitney_auc_expr`
  are the reductions. They assume each group's ``x`` is already unique (i.e.
  the curve was collapsed), matching the eager helpers' precondition.
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


def mann_whitney_auc_expr(
    *,
    score: str | pl.Expr = "score",
    label: str | pl.Expr = "label",
) -> pl.Expr:
    """Mann-Whitney U AUC — P(positive score > negative score) — as an expression.

    Non-parametric AUC via the O(n log n) rank-sum. ``label`` is ``1.0`` for
    positives and ``0.0`` for negatives; ties in ``score`` receive their average
    rank. Usable inside ``group_by(group).agg(...)`` and in ``select``.

    Args:
        score: Score column name or expression.
        label: Positive/negative flag (truthy = positive).

    Returns:
        A ``pl.Expr`` reducing to the MW-U AUC in ``[0, 1]``; ``0.5`` when either
        class is empty.
    """
    sc = _as_expr(score).cast(pl.Float64)
    lb = _as_expr(label).cast(pl.Float64)

    # 1-based ranks, averaged within tied-score groups. `rank("average")` keeps
    # the result aligned with row order, so it multiplies against `lb`
    # element-wise with no sort — and works inside `.agg()` (no window needed).
    avg_rank = sc.rank(method="average").cast(pl.Float64)

    n_pos = lb.sum()
    n_neg = lb.len() - n_pos
    rank_sum_pos = (avg_rank * lb).sum()
    u = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0
    auc = u / (n_pos * n_neg)
    return pl.when((n_pos == 0) | (n_neg == 0)).then(pl.lit(0.5)).otherwise(auc)
