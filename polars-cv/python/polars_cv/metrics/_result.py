"""Base metric result type with shared AUC, interpolation, and bootstrap logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ._auc import partial_auc, trapz_auc


@dataclass(frozen=True)
class MetricResult:
    """Base class for all detection metric results.

    Subclasses add metric-specific convenience methods with pre-bound column
    names (e.g., ``FROCResult.sensitivity_at_fp``).

    Attributes:
        curve: DataFrame containing the computed metric curve.
        metadata: Arbitrary metadata about the computation.
    """

    curve: pl.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # AUC
    # ------------------------------------------------------------------

    def auc(
        self,
        *,
        x_col: str,
        y_col: str,
        x_range: tuple[float, float] | None = None,
        normalize: bool = False,
    ) -> float:
        """Compute (partial) AUC under the curve.

        Args:
            x_col: Column name for the x-axis values.
            y_col: Column name for the y-axis values.
            x_range: Optional ``(lo, hi)`` bounds for partial AUC.
            normalize: Whether to normalize the AUC by the range of the x-values.

        Returns:
            Area under the curve (or partial area).
        """
        sorted_curve = self.curve.sort(x_col)
        x = sorted_curve[x_col].cast(pl.Float64).to_numpy()
        y = sorted_curve[y_col].fill_null(0.0).cast(pl.Float64).to_numpy()
        if x.size == 0:
            return 0.0
        if x_range is None:
            return trapz_auc(x, y, normalize)
        return partial_auc(x, y, x_range[0], x_range[1], normalize)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(self, *, x_col: str, y_col: str, at: float) -> float:
        """Linearly interpolate a y-value at a given x-value.

        Args:
            x_col: Column name for the x-axis.
            y_col: Column name for the y-axis.
            at: The x-value at which to interpolate.

        Returns:
            Interpolated y-value.
        """
        sorted_curve = self.curve.sort(x_col)
        x = sorted_curve[x_col].cast(pl.Float64).to_numpy()
        y = sorted_curve[y_col].fill_null(0.0).cast(pl.Float64).to_numpy()
        if x.size == 0:
            return 0.0
        return float(np.interp(at, x, y, left=y[0], right=y[-1]))

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
            DataFrame with ``x_col`` and ``y_col`` columns.
        """
        return pl.DataFrame(
            {
                x_col: operating_points,
                y_col: [
                    self.interpolate(x_col=x_col, y_col=y_col, at=pt)
                    for pt in operating_points
                ],
            }
        )
