"""Base metric result type with shared AUC and interpolation logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from ._auc import CorrectionMethod
from ._auc_expr import collapse_curve, partial_auc, trapz_auc


@dataclass(frozen=True)
class MetricResult:
    """Base class for all detection metric results.

    Subclasses (e.g. ``PrecisionRecallResult``) add metric-specific convenience
    methods with pre-bound column names. This base carries the PR-curve ``auc``
    (with optional partial-AUC range and correction), computed through the lazy
    ``_auc_expr`` integral authority — never a second, eager implementation.
    FROC/LROC curve interpolation does **not** go through here — those helpers
    build on the lazy ``_auc_expr.interpolate_curve_lazy`` authority directly
    (see ``froc_sensitivity_at_fp`` / ``froc_summary_table``).

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
        correction: CorrectionMethod = None,
    ) -> float:
        """Compute (partial) AUC under the curve.

        Routes through the lazy ``_auc_expr`` authority: :func:`collapse_curve`
        reduces the curve to strictly-increasing x with the upper-envelope y (a
        curve carries many rows tied at one x — a run that changes only y leaves
        x fixed — and collapsing each tie to its max y is deterministic and the
        ROC/FROC convention), then :func:`trapz_auc` / :func:`partial_auc`
        integrates it in one streaming collect.

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
        if self.curve.height == 0:
            return 0.0
        collapsed = collapse_curve(self.curve.lazy(), x_col=x_col, y_col=y_col)
        if x_range is None:
            auc = trapz_auc(collapsed, x=x_col, y=y_col, correction=correction)
        else:
            auc = partial_auc(
                collapsed,
                x=x_col,
                y=y_col,
                lo=x_range[0],
                hi=x_range[1],
                correction=correction,
            )
        return auc.collect(engine="streaming").item()
