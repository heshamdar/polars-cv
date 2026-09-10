"""Base metric result type with shared AUC and interpolation logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ._auc import CorrectionMethod, partial_auc, trapz_auc


@dataclass(frozen=True)
class MetricResult:
    """Base class for all detection metric results.

    Subclasses (e.g. ``PrecisionRecallResult``) add metric-specific convenience
    methods with pre-bound column names. This base carries the eager PR-curve
    ``auc`` (with optional partial-AUC range and correction). FROC/LROC curve
    interpolation does **not** go through here — those helpers build on the lazy
    ``_auc_expr.interpolate_curve_lazy`` authority directly (see
    ``froc_sensitivity_at_fp`` / ``froc_summary_table``).

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

        Every consumer of a curve's geometry goes through here — ``auc`` must
        not sort for itself. A curve carries many rows
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
                ``"normalize"`` divides by the x-range width (mean y-value).

        Returns:
            Area under the curve (or partial area).
        """
        x, y = self._curve_xy(x_col, y_col)
        if x.len() == 0:
            return 0.0
        if x_range is None:
            return trapz_auc(x, y, correction)
        return partial_auc(x, y, x_range[0], x_range[1], correction)
