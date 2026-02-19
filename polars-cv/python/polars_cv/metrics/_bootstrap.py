"""Bootstrap helpers shared by detection metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    """Container for bootstrap confidence interval results.

    Attributes:
        point_estimate: Metric value on the original sample.
        ci_lower: Lower confidence bound.
        ci_upper: Upper confidence bound.
        confidence: Confidence level used for the interval.
        distribution: Raw bootstrap metric values.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    distribution: list[float]


def bootstrap_metric(
    *,
    image_ids: list[str],
    metric_fn: Callable[[list[str]], float],
    point_estimate: float,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
    strata: dict[str, str] | None = None,
) -> BootstrapResult:
    """Estimate confidence intervals by image-level bootstrap sampling.

    Args:
        image_ids: Base image IDs to sample with replacement.
        metric_fn: Callback computing a metric from sampled image IDs.
        point_estimate: Metric on the original sample.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level in ``(0, 1)``.
        seed: Optional RNG seed.
        strata: Optional image->stratum mapping for stratified resampling.

    Returns:
        BootstrapResult with percentile confidence interval.
    """
    if n_bootstrap <= 0:
        raise ValueError("`n_bootstrap` must be > 0.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("`confidence` must be in (0, 1).")
    if not image_ids:
        raise ValueError("`image_ids` cannot be empty.")

    rng = np.random.default_rng(seed)
    distribution: list[float] = []

    if strata is None:
        ids_np = np.asarray(image_ids, dtype=object)
        for _ in range(n_bootstrap):
            sampled = rng.choice(ids_np, size=ids_np.size, replace=True)
            distribution.append(float(metric_fn([str(value) for value in sampled])))
    else:
        grouped: dict[str, list[str]] = {}
        for image_id in image_ids:
            stratum = strata.get(image_id)
            key = "__missing__" if stratum is None else str(stratum)
            grouped.setdefault(key, []).append(image_id)

        grouped_arrays = {
            key: np.asarray(values, dtype=object) for key, values in grouped.items()
        }
        grouped_sizes = {key: len(values) for key, values in grouped.items()}

        for _ in range(n_bootstrap):
            sampled_ids: list[str] = []
            for key, values in grouped_arrays.items():
                sampled = rng.choice(values, size=grouped_sizes[key], replace=True)
                sampled_ids.extend(str(value) for value in sampled)
            distribution.append(float(metric_fn(sampled_ids)))

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(distribution, alpha))
    upper = float(np.quantile(distribution, 1.0 - alpha))
    return BootstrapResult(
        point_estimate=float(point_estimate),
        ci_lower=lower,
        ci_upper=upper,
        confidence=confidence,
        distribution=distribution,
    )
