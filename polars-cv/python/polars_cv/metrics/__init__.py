"""Detection metrics built on top of polars-cv primitives."""

from __future__ import annotations

from ._bootstrap import BootstrapResult
from ._froc import FROCAnalyzer, FROCResult
from ._lroc import LROCAnalyzer, LROCResult

__all__ = [
    "BootstrapResult",
    "FROCAnalyzer",
    "FROCResult",
    "LROCAnalyzer",
    "LROCResult",
]
