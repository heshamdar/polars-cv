"""AUC helpers for FROC/LROC curves."""

from __future__ import annotations

import warnings

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
        gap = float(x[0]) - lo
        span = hi - lo
        if span > 0 and gap / span > 0.1:
            warnings.warn(
                f"partial_auc: requested lower bound {lo} is below the "
                f"curve's minimum x ({float(x[0]):.4g}). The gap covers "
                f"{gap / span:.0%} of the integration range and will be "
                f"filled by clamping y to {float(y[0]):.4g}. This may "
                f"overstate the partial AUC.",
                UserWarning,
                stacklevel=2,
            )
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


def mann_whitney_u_auc(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
) -> float:
    """Compute Mann-Whitney U statistic as a non-parametric AUC estimate.

    Returns P(random positive score > random negative score), which is
    equivalent to the AUC of the ROC curve.  Uses the O(n log n) rank-sum
    algorithm to avoid O(n*m) cross-join.

    Args:
        positive_scores: Scores for positive instances.
        negative_scores: Scores for negative instances.

    Returns:
        Mann-Whitney U AUC in [0, 1].  Returns 0.5 if either group is empty.
    """
    n_pos = len(positive_scores)
    n_neg = len(negative_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Combine and sort; use label to distinguish groups.
    combined = np.concatenate([positive_scores, negative_scores])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # Sort by score; use label as tiebreaker (positive first for
    # mid-rank calculation to be symmetric).
    order = np.lexsort((labels, combined))
    labels_sorted = labels[order]

    # Compute average ranks (handle ties via cumulative count).
    n = len(combined)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and combined[order[j]] == combined[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank for tie group
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # U = rank_sum_positive - n_pos*(n_pos+1)/2
    rank_sum_pos = ranks[labels_sorted == 1.0].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def _interp(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Interpolate y(xq) linearly with endpoint clamping."""
    if xq <= float(x[0]):
        return float(y[0])
    if xq >= float(x[-1]):
        return float(y[-1])
    return float(np.interp(xq, x, y))
