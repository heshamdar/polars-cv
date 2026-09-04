"""Base metric result type with shared AUC, interpolation, and bootstrap logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from ._auc import CorrectionMethod, partial_auc, trapz_auc
from ._auc_expr import interpolate_curve_lazy
from ._types import COL_IMAGE_ID

if TYPE_CHECKING:
    from ._bootstrap import BootstrapResult
    from ._types import DetectionTable


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

    # ------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------

    def bootstrap_ci(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int | None = None,
        *,
        metric: str | dict[str, dict[str, Any]] = "auc",
        metric_kwargs: dict[str, Any] | None = None,
        sample_col: str | None = None,
    ) -> BootstrapResult | dict[str, BootstrapResult]:
        """Estimate confidence intervals via bootstrap sampling.

        Supports single-metric (backward compatible) and multi-metric modes.

        **Single-metric** (``metric`` is a ``str``)::

            ci = result.bootstrap_ci(metric="auc", metric_kwargs={"method": "mann_whitney"})

        **Multi-metric** (``metric`` is a ``dict``)::

            cis = result.bootstrap_ci(
                metric={
                    "mw_auc": {"metric": "auc", "method": "mann_whitney"},
                    "partial": {"metric": "auc", "fp_range": (0, 2)},
                },
            )
            # cis["mw_auc"].ci_lower, cis["partial"].point_estimate, ...

        Each dict value must contain a ``"metric"`` key naming the method
        to call; remaining keys are forwarded as keyword arguments.

        Args:
            n_bootstrap: Number of bootstrap iterations.
            confidence: Confidence level in ``(0, 1)``.
            seed: Optional RNG seed.
            metric: Either a single method name (str) or a dict mapping
                labels to ``{"metric": ..., **kwargs}`` specs.
            metric_kwargs: Keyword arguments for single-metric mode.
                Ignored when ``metric`` is a dict.
            sample_col: Column name for entity-level sampling. When
                ``None`` (default), sampling happens at the ``image_id``
                level. When set (e.g. ``"case_id"``), sampling happens
                at the entity level and expands back to image IDs.

        Returns:
            ``BootstrapResult`` for single-metric mode, or
            ``dict[str, BootstrapResult]`` for multi-metric mode.

        Raises:
            ValueError: If ``detection_table`` is not available or the
                metric name is invalid.
        """
        from ._bootstrap import (
            _bootstrap_distribution,
            _bootstrap_table_with_draws,
            _finalize,
            _resolve_bootstrap_samples,
        )

        table = self._get_detection_table()

        # Normalize to multi-metric internally
        if isinstance(metric, str):
            multi_mode = False
            metric_specs: dict[str, dict[str, Any]] = {
                metric: {"metric": metric, **(metric_kwargs or {})}
            }
        else:
            multi_mode = True
            metric_specs = metric

        # One lazy, streaming resample shared across every requested metric, and
        # one per-replicate DetectionTable keyed by ``bootstrap_id``. Replaces the
        # old sequential Python loop that reconstructed a whole result — and
        # collected — once per replicate per metric.
        samples = _resolve_bootstrap_samples(
            table,
            sample_col=sample_col,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
        boot_table = _bootstrap_table_with_draws(table, samples)

        results: dict[str, BootstrapResult] = {}
        for name, spec in metric_specs.items():
            kw = {k: v for k, v in spec.items() if k != "metric"}
            grouped, empty_value = self._bootstrap_grouped(
                boot_table, spec["metric"], kw
            )
            distribution = _bootstrap_distribution(grouped, n_bootstrap, empty_value)
            point = self._resolve_metric(spec["metric"], kw)
            results[name] = _finalize(distribution, point, confidence)

        if multi_mode:
            return results
        return next(iter(results.values()))

    # ------------------------------------------------------------------
    # Subclass hooks for bootstrap
    # ------------------------------------------------------------------

    def _resolve_metric(self, metric: str, kwargs: dict[str, Any]) -> float:
        """Resolve a metric name to a callable and invoke it.

        Args:
            metric: Public method name on this result (e.g. ``"auc"``).
            kwargs: Keyword arguments to pass to the method.

        Returns:
            Scalar metric value.

        Raises:
            ValueError: If the method does not exist or is not callable.
        """
        fn = getattr(self, metric, None)
        if fn is None or not callable(fn):
            raise ValueError(
                f"Unknown metric {metric!r} on {type(self).__name__}. "
                f"Expected a public method name."
            )
        return float(fn(**kwargs))

    def _get_detection_table(self) -> DetectionTable:
        """Return the underlying ``DetectionTable``.

        Subclasses must override this.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_detection_table() "
            f"to support bootstrap_ci."
        )

    def _bootstrap_grouped(
        self,
        boot_table: DetectionTable,
        method: str,
        kwargs: dict[str, Any],
    ) -> tuple[pl.LazyFrame, float]:
        """Vectorized metric over a per-replicate table, keyed by ``bootstrap_id``.

        Subclasses override this to map each bootstrap-able metric to a single
        grouped lazy plan — replacing the old per-replicate reconstruct loop.

        Args:
            boot_table: The resampled ``DetectionTable`` carrying ``bootstrap_id``
                on both frames (from ``_bootstrap_table_with_draws``).
            method: The metric method name (as in ``_resolve_metric``).
            kwargs: Keyword arguments for that metric.

        Returns:
            ``(grouped, empty_value)`` where ``grouped`` is a ``LazyFrame`` with
            ``[bootstrap_id, auc]`` (one row per replicate that produced a value)
            and ``empty_value`` fills replicates absent from ``grouped``.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _bootstrap_grouped() "
            f"to support bootstrap_ci."
        )


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _resolve_sampling_entities(
    table: DetectionTable,
    sample_col: str | None,
) -> tuple[list[str], dict[str, list[str]] | None]:
    """Determine the sampling units and their image-ID expansion.

    Args:
        table: Detection table containing image metadata.
        sample_col: Column name for entity-level sampling, or ``None``
            for image-level sampling.

    Returns:
        Tuple of ``(entity_ids, entity_to_images_map | None)``.
        When ``sample_col`` is ``None``, returns ``(image_ids, None)``
        — each entity IS an image.
        When ``sample_col`` is set, returns ``(unique_entity_values,
        {entity -> [image_ids]})``.
    """
    if sample_col is None:
        ids, _ = table.image_ids_and_strata()
        return ids, None

    meta_df = (
        table.image_metadata.select(COL_IMAGE_ID, sample_col)
        .unique()
        .collect(engine="streaming")
    )

    entity_to_images: dict[str, list[str]] = {}
    for row in meta_df.iter_rows(named=True):
        entity = str(row[sample_col])
        entity_to_images.setdefault(entity, []).append(str(row[COL_IMAGE_ID]))

    return list(entity_to_images.keys()), entity_to_images
