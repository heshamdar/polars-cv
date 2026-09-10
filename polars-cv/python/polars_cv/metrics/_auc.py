"""AUC helpers for FROC/LROC curves."""

from __future__ import annotations

import warnings
from typing import Literal

import polars as pl

CorrectionMethod = Literal["normalize"] | None


def validate_correction(correction: CorrectionMethod) -> None:
    """Reject any correction outside the supported vocabulary.

    The single authority for the ``correction`` vocabulary, so an unrecognised
    value fails loudly at the entry point rather than being silently treated as
    the raw area (the ``if correction == "normalize": ...; return raw`` shape
    would otherwise swallow it). ``"mcclish"`` — McClish's standardized ROC
    partial-AUC correction — was removed here: it standardizes against the ROC
    chance diagonal ``y = x`` on the unit square, which is only meaningful when
    the x-axis is a probability bounded to ``[0, 1]``. FROC's x-axis (false
    positives per image) is unbounded, LROC's chance line is not the diagonal,
    and PR's chance line is horizontal at prevalence — so none of the curves in
    this package is the ROC curve McClish assumes.

    Args:
        correction: The correction requested by the caller.

    Raises:
        ValueError: If ``correction`` is neither ``None`` nor ``"normalize"``.
    """
    if correction is not None and correction != "normalize":
        raise ValueError(
            f"Unknown correction {correction!r}. Expected 'normalize' or None. "
            "(The McClish standardized correction was removed: it assumes a "
            "bounded [0,1] ROC x-axis with a y=x chance diagonal, which the "
            "FROC/LROC/PR curves are not.)"
        )


def trapz_auc(
    x: pl.Series,
    y: pl.Series,
    correction: CorrectionMethod = None,
) -> float:
    """Integrate y(x) with the trapezoidal rule.

    Args:
        x: Monotonic x-values (Polars Series).
        y: y-values aligned with ``x`` (Polars Series).
        correction: Optional correction for the result.
            ``None`` returns the raw area.
            ``"normalize"`` divides by the x-range (gives average y-value).

    Returns:
        Area under the curve.
    """
    validate_correction(correction)
    if x.len() < 2 or y.len() < 2:
        return 0.0

    dx = x.diff().slice(1)
    avg_y = (y + y.shift(1)).slice(1) / 2.0
    raw = float((dx * avg_y).sum())

    if correction == "normalize":
        span = float(x[-1]) - float(x[0])
        return raw / span if span > 0 else 0.0
    return raw


def partial_auc(
    x: pl.Series,
    y: pl.Series,
    lo: float,
    hi: float,
    correction: CorrectionMethod = None,
) -> float:
    """Compute partial AUC over ``[lo, hi]`` with linear interpolation.

    Args:
        x: Monotonic x-values (Polars Series).
        y: y-values aligned with ``x`` (Polars Series).
        lo: Lower x bound.
        hi: Upper x bound.
        correction: Optional correction for the partial area.
            ``None`` returns the raw partial area.
            ``"normalize"`` divides by the range width ``(hi - lo)``, giving the
            mean y-value over the window (bounded to ``[0, 1]`` when ``y`` is).

    Returns:
        Area under the clipped curve (optionally corrected).
    """
    validate_correction(correction)
    if hi <= lo:
        return 0.0
    if x.len() == 0 or y.len() == 0:
        return 0.0

    x0_val = float(x[0])
    y0_val = float(y[0])

    if lo < x0_val:
        gap = x0_val - lo
        span = hi - lo
        if span > 0 and gap / span > 0.1:
            warnings.warn(
                f"partial_auc: requested lower bound {lo} is below the "
                f"curve's minimum x ({x0_val:.4g}). The gap covers "
                f"{gap / span:.0%} of the integration range and will be "
                f"filled by clamping y to {y0_val:.4g}. This may "
                f"overstate the partial AUC.",
                UserWarning,
                stacklevel=2,
            )

    # Boundary points are built as Float64 explicitly. `lo`/`hi` come straight
    # from the caller's `fp_range`, and the natural spelling of a bound is an
    # int — `fp_range=(0, 8)` — which `pl.Series` would infer as Int64 and
    # then refuse to concat onto a Float64 curve.
    lo = float(lo)
    hi = float(hi)

    def _bound(name: str, *values: float) -> pl.Series:
        return pl.Series(name, list(values), dtype=pl.Float64)

    # Vectorized clip: keep points within [lo, hi]
    mask = (x >= lo) & (x <= hi)
    clipped_x = x.filter(mask)
    clipped_y = y.filter(mask)

    # Prepend lo boundary (clamped or interpolated). partial_auc fills the
    # integration window to [lo, hi] even when the curve does not span it,
    # so out-of-range bounds fall back to endpoint y (unlike MetricResult
    # interpolate, which returns None).
    if lo < x0_val:
        clipped_x = pl.concat([_bound("x", lo), clipped_x])
        clipped_y = pl.concat([_bound("y", y0_val), clipped_y])
    elif clipped_x.len() == 0 or float(clipped_x[0]) > lo:
        y_lo = _interp(x, y, lo)
        if y_lo is None:
            y_lo = y0_val
        clipped_x = pl.concat([_bound("x", lo), clipped_x])
        clipped_y = pl.concat([_bound("y", y_lo), clipped_y])

    # Append hi boundary (interpolated) if curve doesn't reach hi
    if clipped_x.len() == 0:
        y_lo = _interp(x, y, lo)
        y_hi = _interp(x, y, hi)
        if y_lo is None:
            y_lo = y0_val
        if y_hi is None:
            y_hi = float(y[-1])
        clipped_x = _bound("x", lo, hi)
        clipped_y = _bound("y", y_lo, y_hi)
    elif float(clipped_x[-1]) < hi:
        y_hi = _interp(x, y, hi)
        if y_hi is None:
            y_hi = float(y[-1])
        clipped_x = pl.concat([clipped_x, _bound("x", hi)])
        clipped_y = pl.concat([clipped_y, _bound("y", y_hi)])

    raw = trapz_auc(clipped_x, clipped_y)

    if correction == "normalize":
        span = hi - lo
        return raw / span if span > 0 else 0.0
    return raw


def _interp(x: pl.Series, y: pl.Series, xq: float) -> float | None:
    """Interpolate y(xq) linearly; return ``None`` outside the observed range.

    Args:
        x: Sorted x-values (Polars Series).
        y: y-values aligned with ``x`` (Polars Series).
        xq: Query x-value.

    Returns:
        Interpolated y-value, or ``None`` when ``xq`` falls outside
        ``[x[0], x[-1]]`` (no extrapolation).
    """
    if xq < float(x[0]) or xq > float(x[-1]):
        return None
    if xq == float(x[0]):
        return float(y[0])
    if xq == float(x[-1]):
        return float(y[-1])
    idx = x.search_sorted(xq, side="right") - 1
    x0_val = float(x[idx])
    x1_val = float(x[idx + 1])
    y0_val = float(y[idx])
    y1_val = float(y[idx + 1])
    denom = x1_val - x0_val
    t = (xq - x0_val) / denom if denom != 0.0 else 0.0
    return y0_val + t * (y1_val - y0_val)
