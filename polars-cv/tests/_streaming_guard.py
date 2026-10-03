"""Which steps of a lazy plan the streaming engine hands to the in-memory engine.

Polars' streaming engine runs what it can natively and wraps everything else in
an ``in-memory-map`` node, which collects that node's whole input before it
runs. That is the memory blow-up the metrics must avoid. A native node that
happens to buffer (``sort``, ``group-by``, a join build side) is *not* a
fallback; it is the streaming engine doing its job.

:func:`in_memory_nodes` reads the physical streaming graph
(``show_graph(engine="streaming", plan_stage="physical", raw_output=True)``)
and returns the label of every ``in-memory-map`` node.

Limits: this scans a debug rendering. It relies on two things:

* polars naming the fallback node ``in-memory-map``;
* the ``plan_stage`` argument of ``show_graph``.

If either changes, the scan silently matches nothing. The two
``test_guard_*`` fixtures in ``test_streaming_plans.py`` are what catch that:
a known fallback that stops being reported fails them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

_LABEL = re.compile(r'label="((?:[^"\\]|\\.)*)"')
_FALLBACK = "in-memory-map"


def in_memory_nodes(lf: pl.LazyFrame) -> list[str]:
    """The labels of ``lf``'s in-memory fallback nodes, one line each."""
    dot = lf.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True, show=False
    )
    assert isinstance(dot, str)
    labels = (
        m.group(1).replace("\\n", " ").replace('\\"', '"') for m in _LABEL.finditer(dot)
    )
    return [" ".join(label.split()) for label in labels if label.startswith(_FALLBACK)]


@dataclass(frozen=True)
class KnownFallback:
    """A fallback the streaming engine cannot avoid, and why."""

    pattern: str
    reason: str

    def matches(self, label: str) -> bool:
        return re.search(self.pattern, label) is not None


#: The closed list of accepted fallbacks. Anything else is a regression.
KNOWN_FALLBACKS: tuple[KnownFallback, ...] = (
    KnownFallback(
        pattern=r"^in-memory-map AGGREGATE.*\.implode\(\)",
        reason=(
            "Building a list per group (`group_by().agg(pl.col(x))`): polars "
            "cannot implode inside a streaming group-by. The matchers are "
            "elementwise over one row per (image, class), so the per-image "
            "object lists have to be built. That costs O(objects) memory, "
            "not O(pixels)."
        ),
    ),
)


def unexplained(labels: list[str]) -> list[str]:
    """The fallback labels that no :data:`KNOWN_FALLBACKS` entry accounts for."""
    return [
        label for label in labels if not any(k.matches(label) for k in KNOWN_FALLBACKS)
    ]
