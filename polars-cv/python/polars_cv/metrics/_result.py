"""Base metric result type with shared AUC, interpolation, and bootstrap logic."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import polars as pl

from ._auc import CorrectionMethod, _interp, partial_auc, trapz_auc
from ._types import COL_IMAGE_ID

if TYPE_CHECKING:
    from ._bootstrap import BootstrapResult
    from ._types import DetectionTable


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

        Args:
            x_col: Column name for the x-axis.
            y_col: Column name for the y-axis.
            at: The x-value at which to interpolate.

        Returns:
            Interpolated y-value, or ``None`` when ``at`` falls outside the
            observed x-range of the curve (no extrapolation). At an x the
            curve visits more than once, the highest y there is returned.
        """
        x, y = self._curve_xy(x_col, y_col)
        if x.len() == 0:
            return None
        return _interp(x, y, at)

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
        return pl.DataFrame(
            {
                x_col: pl.Series(x_col, operating_points, dtype=pl.Float64),
                y_col: pl.Series(
                    y_col,
                    [
                        self.interpolate(x_col=x_col, y_col=y_col, at=pt)
                        for pt in operating_points
                    ],
                    dtype=pl.Float64,
                ),
            }
        )

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
        from ._bootstrap import _finalize, _validate_bootstrap_params

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

        # Resolve sampling entities and get image ID expansion
        sample_ids, entity_to_images = _resolve_sampling_entities(table, sample_col)
        _, strata = table.image_ids_and_strata()

        _validate_bootstrap_params(n_bootstrap, confidence, sample_ids)

        # Compute point estimates
        point_estimates: dict[str, float] = {}
        for name, spec in metric_specs.items():
            kw = {k: v for k, v in spec.items() if k != "metric"}
            point_estimates[name] = self._resolve_metric(spec["metric"], kw)

        # Bootstrap loop with shared reconstruction
        distributions: dict[str, list[float]] = defaultdict(list)
        ids_series = pl.Series("id", sample_ids)

        for i in range(n_bootstrap):
            iter_seed = (seed + i) if seed is not None else None
            sampled_entities = ids_series.sample(
                n=ids_series.len(), with_replacement=True, seed=iter_seed
            ).to_list()

            if entity_to_images is not None:
                sampled_image_ids: list[str] = []
                for eid in sampled_entities:
                    sampled_image_ids.extend(entity_to_images[eid])
            else:
                sampled_image_ids = sampled_entities

            reconstructed = self._reconstruct(sampled_image_ids)

            for name, spec in metric_specs.items():
                kw = {k: v for k, v in spec.items() if k != "metric"}
                distributions[name].append(
                    reconstructed._resolve_metric(spec["metric"], kw)
                )

        # Build results
        results: dict[str, BootstrapResult] = {}
        for name in metric_specs:
            results[name] = _finalize(
                distributions[name], point_estimates[name], confidence
            )

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

    def _reconstruct(self, sampled_ids: list[str]) -> MetricResult:
        """Reconstruct a result from bootstrap-sampled image IDs.

        Subclasses must override this to rebuild their specific result
        type from resampled data.

        Args:
            sampled_ids: Image IDs sampled with replacement.

        Raises:
            NotImplementedError: Always, unless overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _reconstruct() "
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
