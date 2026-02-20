"""Detection matching adapters that produce a canonical DetectionTable."""

from __future__ import annotations

from ._bbox import BBoxMatcher
from ._contour import ContourMatcher
from ._prematched import PreMatchedAdapter
from ._protocol import Matcher

__all__ = [
    "BBoxMatcher",
    "ContourMatcher",
    "Matcher",
    "PreMatchedAdapter",
]
