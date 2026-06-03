"""Benchmark regression harness for polars-cv.

A thin layer over the existing benchmark suite (``benchmarks.scenarios`` +
``benchmarks.frameworks``) that adds the one missing capability: running a
fixed, reproducible matrix and comparing two runs to detect performance
regressions.

Workflow::

    # baseline (before the change), release build
    python -m benchmarks.regression.run_suite --out baseline.json

    # candidate (after the change), release build
    python -m benchmarks.regression.run_suite --out candidate.json

    # gate: exits non-zero if any result regressed
    python -m benchmarks.regression.compare baseline.json candidate.json

See ``README.md`` in this package for the full procedure.
"""

from __future__ import annotations

from .config import DEFAULT, POLARS_CV_ADAPTERS, SuiteConfig, Thresholds

__all__ = ["DEFAULT", "POLARS_CV_ADAPTERS", "SuiteConfig", "Thresholds"]
