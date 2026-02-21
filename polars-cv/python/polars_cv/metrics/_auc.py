"""AUC helpers for FROC/LROC curves."""

from __future__ import annotations

import numpy as np
import polars as pl


def trapz_auc(x: np.ndarray, y: np.ndarray, normalize: bool = False) -> float:
    """Integrate y(x) with the trapezoidal rule.

    Args:
        x: Monotonic x-values.
        y: y-values aligned with ``x``.

    Returns:
        Area under the curve.
    """
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size == 1:
        return 0.0

    if normalize:
        return float(np.trapezoid(y, x)) / (x[-1] - x[0])
    return float(np.trapezoid(y, x))


def partial_auc(
    x: np.ndarray, y: np.ndarray, lo: float, hi: float, normalize: bool = False
) -> float:
    """Compute partial AUC over ``[lo, hi]`` with linear interpolation.

    Args:
        x: Monotonic x-values.
        y: y-values aligned with ``x``.
        lo: Lower x bound.
        hi: Upper x bound.

    Returns:
        Area under the clipped curve.
    """
    if hi <= lo:
        return 0.0
    if x.size == 0 or y.size == 0:
        return 0.0

    clipped_x: list[float] = []
    clipped_y: list[float] = []

    if lo < x[0]:
        clipped_x.append(lo)
        clipped_y.append(float(y[0]))

    for idx in range(x.size):
        xi = float(x[idx])
        yi = float(y[idx])
        if lo <= xi <= hi:
            clipped_x.append(xi)
            clipped_y.append(yi)

    if not clipped_x:
        y_lo = _interp(x, y, lo)
        y_hi = _interp(x, y, hi)
        return trapz_auc(np.array([lo, hi]), np.array([y_lo, y_hi]), normalize)

    if clipped_x[0] > lo:
        clipped_x.insert(0, lo)
        clipped_y.insert(0, _interp(x, y, lo))
    if clipped_x[-1] < hi:
        clipped_x.append(hi)
        clipped_y.append(_interp(x, y, hi))

    return trapz_auc(np.array(clipped_x), np.array(clipped_y), normalize)


def weighted_curve(
    dense_counts: pl.DataFrame,
    *,
    threshold_col: str = "threshold",
    tp_col: str = "tp",
    fp_col: str = "fp",
    n_gts_col: str = "n_gts",
    weight_col: str = "weight",
) -> pl.DataFrame:
    """Aggregate weighted threshold operating points from dense counts.

    Args:
        dense_counts: Dense per-image/per-threshold table.
        threshold_col: Threshold column name.
        tp_col: True-positive count column name.
        fp_col: False-positive count column name.
        n_gts_col: Ground-truth target count column name.
        weight_col: Sample-weight column name.

    Returns:
        DataFrame with weighted sensitivity and weighted FP/image.
    """
    weighted = dense_counts.group_by(threshold_col).agg(
        weighted_tp=(pl.col(tp_col) * pl.col(weight_col)).sum(),
        weighted_fp=(pl.col(fp_col) * pl.col(weight_col)).sum(),
        weighted_total_gts=(pl.col(n_gts_col) * pl.col(weight_col)).sum(),
        weight_sum=pl.col(weight_col).sum(),
        tp=pl.col(tp_col).sum().cast(pl.Int64),
        fp=pl.col(fp_col).sum().cast(pl.Int64),
        total_gts=pl.col(n_gts_col).sum().cast(pl.Int64),
    )
    return weighted.with_columns(
        fn=(pl.col("total_gts") - pl.col("tp")).clip(lower_bound=0),
        fp_per_image=pl.when(pl.col("weight_sum") > 0.0)
        .then(pl.col("weighted_fp") / pl.col("weight_sum"))
        .otherwise(pl.lit(None, dtype=pl.Float64)),
        sensitivity=pl.when(pl.col("weighted_total_gts") > 0.0)
        .then(pl.col("weighted_tp") / pl.col("weighted_total_gts"))
        .otherwise(pl.lit(None, dtype=pl.Float64)),
    ).sort(threshold_col)


def _interp(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Interpolate y(xq) linearly with endpoint clamping."""
    if xq <= float(x[0]):
        return float(y[0])
    if xq >= float(x[-1]):
        return float(y[-1])
    return float(np.interp(xq, x, y))
