"""AUC helpers for FROC/LROC curves."""

from __future__ import annotations

import warnings
from typing import Literal

import polars as pl

CorrectionMethod = Literal["normalize", "mcclish"] | None


def mcclish_correction(raw_pauc: float, lo: float, hi: float) -> float:
    """Apply McClish's standardized partial AUC correction.

    Maps a raw partial AUC to a [0.5, 1.0] scale where 0.5 corresponds to
    chance-level performance within the restricted interval.

    Reference:
        McClish DC. Analyzing a portion of the ROC curve.
        Medical Decision Making. 1989;9(3):190-195.

    Args:
        raw_pauc: Raw partial AUC value (trapezoidal integration over [lo, hi]).
        lo: Lower x-axis bound of the partial region.
        hi: Upper x-axis bound of the partial region.

    Returns:
        Corrected partial AUC in [0.5, 1.0].
    """
    span = hi - lo
    if span <= 0:
        return 0.5
    min_pauc = (lo + hi) * span / 2  # diagonal (chance-level)
    max_pauc = span  # perfect classifier
    denom = max_pauc - min_pauc
    if denom <= 0:
        return 0.5
    return (1 + (raw_pauc - min_pauc) / denom) / 2


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
            ``"normalize"`` divides by the range width ``(hi - lo)``.
            ``"mcclish"`` applies McClish's standardized correction,
            mapping the result to [0.5, 1.0] where 0.5 = chance level.

    Returns:
        Area under the clipped curve (optionally corrected).
    """
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

    # Vectorized clip: keep points within [lo, hi]
    mask = (x >= lo) & (x <= hi)
    clipped_x = x.filter(mask)
    clipped_y = y.filter(mask)

    # Prepend lo boundary (clamped or interpolated). partial_auc fills the
    # integration window to [lo, hi] even when the curve does not span it,
    # so out-of-range bounds fall back to endpoint y (unlike MetricResult
    # interpolate, which returns None).
    if lo < x0_val:
        clipped_x = pl.concat([pl.Series("x", [lo]), clipped_x])
        clipped_y = pl.concat([pl.Series("y", [y0_val]), clipped_y])
    elif clipped_x.len() == 0 or float(clipped_x[0]) > lo:
        y_lo = _interp(x, y, lo)
        if y_lo is None:
            y_lo = y0_val
        clipped_x = pl.concat([pl.Series("x", [lo]), clipped_x])
        clipped_y = pl.concat([pl.Series("y", [y_lo]), clipped_y])

    # Append hi boundary (interpolated) if curve doesn't reach hi
    if clipped_x.len() == 0:
        y_lo = _interp(x, y, lo)
        y_hi = _interp(x, y, hi)
        if y_lo is None:
            y_lo = y0_val
        if y_hi is None:
            y_hi = float(y[-1])
        clipped_x = pl.Series("x", [lo, hi])
        clipped_y = pl.Series("y", [y_lo, y_hi])
    elif float(clipped_x[-1]) < hi:
        y_hi = _interp(x, y, hi)
        if y_hi is None:
            y_hi = float(y[-1])
        clipped_x = pl.concat([clipped_x, pl.Series("x", [hi])])
        clipped_y = pl.concat([clipped_y, pl.Series("y", [y_hi])])

    raw = trapz_auc(clipped_x, clipped_y)

    if correction == "normalize":
        span = hi - lo
        return raw / span if span > 0 else 0.0
    if correction == "mcclish":
        return mcclish_correction(raw, lo, hi)
    return raw


def mann_whitney_u_auc(
    positive_scores: pl.Series,
    negative_scores: pl.Series,
) -> float:
    """Compute Mann-Whitney U statistic as a non-parametric AUC estimate.

    Returns P(random positive score > random negative score), which is
    equivalent to the AUC of the ROC curve.  Uses the O(n log n) rank-sum
    algorithm via Polars sort + ``over()`` window to avoid O(n*m) cross-join.

    Args:
        positive_scores: Scores for positive instances (Polars Series).
        negative_scores: Scores for negative instances (Polars Series).

    Returns:
        Mann-Whitney U AUC in [0, 1].  Returns 0.5 if either group is empty.
    """
    n_pos = positive_scores.len()
    n_neg = negative_scores.len()
    if n_pos == 0 or n_neg == 0:
        return 0.5

    combined = pl.DataFrame(
        {
            "score": pl.concat(
                [
                    positive_scores.cast(pl.Float64),
                    negative_scores.cast(pl.Float64),
                ]
            ),
            "label": pl.concat(
                [
                    pl.Series("label", [1.0] * n_pos),
                    pl.Series("label", [0.0] * n_neg),
                ]
            ),
        }
    ).sort("score", "label")

    # Average ranks: row index (1-based), then mean over tied score groups
    ranked = combined.with_row_index("_rank_idx").with_columns(
        avg_rank=(pl.col("_rank_idx") + 1).cast(pl.Float64).mean().over("score")
    )

    rank_sum_pos = float(ranked.filter(pl.col("label") == 1.0)["avg_rank"].sum())
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def detection_level_mann_whitney(
    det_df: pl.DataFrame,
    score_col: str = "score",
    is_tp_col: str = "is_tp",
) -> float:
    """Compute detection-level MW-U: P(TP score > FP score).

    Args:
        det_df: Collected detections DataFrame with score and is_tp columns.
        score_col: Column name for detection scores.
        is_tp_col: Column name for true-positive flag.

    Returns:
        Mann-Whitney U AUC in [0, 1].
    """
    tp_scores = det_df.filter(pl.col(is_tp_col))[score_col]
    fp_scores = det_df.filter(~pl.col(is_tp_col))[score_col]
    return mann_whitney_u_auc(tp_scores, fp_scores)


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
