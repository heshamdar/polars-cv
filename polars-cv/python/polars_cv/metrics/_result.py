"""Base metric result type with shared AUC and interpolation logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ._auc import CorrectionMethod, partial_auc, trapz_auc
from ._auc_expr import interpolate_curve_lazy


@dataclass(frozen=True)
class MetricResult:
    """Base class for all detection metric results.

    Subclasses (e.g. ``PrecisionRecallResult``) add metric-specific convenience
    methods with pre-bound column names. The FROC/LROC metrics interpolate their
    curves through this base directly (see ``froc_sensitivity_at_fp``).

    Attributes:
        curve: DataFrame containing the computed metric curve.
        metadata: Arbitrary metadata about the computation.
    """

    curve: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Curve access
    # ------------------------------------------------------------------

    def _curve_xy(self, x_col: str, y_col: str) -> tuple[pl.Series, pl.Series]:
        """Return the curve as strictly increasing x with the upper envelope y.

        Every consumer of a curve's geometry goes through here — ``auc`` and
        ``interpolate`` must not sort for themselves. A curve carries many rows
        tied at one x (a FROC threshold bucket that adds only true positives
        leaves ``fp_per_image`` unchanged), and Polars' ``sort`` defaults to
        ``maintain_order=False``, so a sort on x alone leaves the y at each tie
        boundary unspecified — the trapezoid there, and therefore the AUC,
        would vary run to run. Collapsing each tie group to its maximum y is
        both deterministic and the standard ROC/FROC convention: the operating
        point reachable at that x is the best one, not an arbitrary one.

        Args:
            x_col: Column name for the x-axis.
            y_col: Column name for the y-axis.

        Returns:
            ``(x, y)`` as Float64 Series, x strictly increasing.
        """
        collapsed = (
            self.curve.select(
                pl.col(x_col).cast(pl.Float64),
                pl.col(y_col).fill_null(0.0).cast(pl.Float64),
            )
            .group_by(x_col)
            .agg(pl.col(y_col).max())
            .sort(x_col)
        )
        return collapsed[x_col], collapsed[y_col]

    # ------------------------------------------------------------------
    # AUC
    # ------------------------------------------------------------------

    def auc(
        self,
        *,
        x_col: str,
        y_col: str,
        x_range: tuple[float, float] | None = None,
        correction: CorrectionMethod = None,
    ) -> float:
        """Compute (partial) AUC under the curve.

        Args:
            x_col: Column name for the x-axis values.
            y_col: Column name for the y-axis values.
            x_range: Optional ``(lo, hi)`` bounds for partial AUC.
            correction: Optional correction for partial AUC.
                ``None`` returns the raw area.
                ``"normalize"`` divides by the x-range width.
                ``"mcclish"`` applies McClish's standardized correction
                (only valid with ``x_range``).

        Returns:
            Area under the curve (or partial area).
        """
        x, y = self._curve_xy(x_col, y_col)
        if x.len() == 0:
            return 0.0
        if x_range is None:
            return trapz_auc(x, y, correction)
        return partial_auc(x, y, x_range[0], x_range[1], correction)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(self, *, x_col: str, y_col: str, at: float) -> float | None:
        """Linearly interpolate a y-value at a given x-value.

        Delegates to :func:`~polars_cv.metrics._auc_expr.interpolate_curve_lazy`
        — the single interpolation authority the lazy FROC/LROC helpers also use
        — and collects at this eager boundary.

        Args:
            x_col: Column name for the x-axis.
            y_col: Column name for the y-axis.
            at: The x-value at which to interpolate.

        Returns:
            Interpolated y-value, or ``None`` when ``at`` falls outside the
            observed x-range of the curve (no extrapolation). At an x the
            curve visits more than once, the highest y there is returned.
        """
        result = interpolate_curve_lazy(
            self.curve.lazy(), x_col=x_col, y_col=y_col, at=[float(at)]
        ).collect()
        value = result[y_col][0]
        return None if value is None else float(value)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary_table(
        self,
        *,
        x_col: str,
        y_col: str,
        operating_points: list[float],
    ) -> pl.DataFrame:
        """Build a summary at specific operating points.

        Args:
            x_col: Column for x-axis values.
            y_col: Column for y-axis values.
            operating_points: x-values at which to report interpolated y.

        Returns:
            DataFrame with ``x_col`` and ``y_col`` columns. ``y_col`` is
            null for operating points outside the observed x-range, and is
            always Float64 — an all-null column must still be a sensitivity
            column, not a ``Null``-dtype one that breaks arithmetic
            downstream.
        """
        return interpolate_curve_lazy(
            self.curve.lazy(),
            x_col=x_col,
            y_col=y_col,
            at=operating_points,
        ).collect()
