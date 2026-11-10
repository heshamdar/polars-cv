"""The generic matching layer: what any matcher shares, and the pre-matched one.

``Matcher`` is the protocol; ``PreMatchedAdapter`` builds a table from rows
matched elsewhere; ``_carry`` and ``_table`` are the argument readers and the
shared tail a matcher ends in. The polars-cv matchers live in ``.._cv``.
"""

from __future__ import annotations

from ._prematched import PreMatchedAdapter
from ._protocol import Matcher

__all__ = [
    "Matcher",
    "PreMatchedAdapter",
]
