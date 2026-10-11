"""Which steps of a lazy plan the streaming engine hands to the in-memory engine.

Polars' streaming engine runs what it can natively and hands everything else
to the in-memory engine, which collects that node's whole input before it runs:
an ``in-memory-map``, an ``in-memory-join`` (e.g. ``join_asof(by=...)``), or a
``columnar-function`` (an unwindowed ``int_range(pl.len())``, a Python UDF, a
plugin call not declared elementwise). A native node that happens to buffer
(``sort``, ``group-by``, a join's build side) is *not* a fallback; it is the
streaming engine doing its job.

:func:`in_memory_nodes` reads the physical streaming graph
(``show_graph(engine="streaming", plan_stage="physical", raw_output=True)``)
and returns every node polars itself marks as an in-memory engine fallback:
the graph's legend names the fill colour of that class, and every node filled
with it is one. The classification is polars', not a list of node names kept
here (that list once missed ``in-memory-join`` and polars' own columnar
functions, so both passed for streaming).

Since polars 2.0 a window ``.over()`` is a native streaming node (``window``),
not a fallback; what remains are the list-building aggregations
:data:`KNOWN_FALLBACKS` names, each with its reason.

Limits: this scans a debug rendering, relying on its legend wording ("in-memory
engine fallback"), its ``fillcolor`` attribute and ``show_graph(plan_stage=)``.
A legend that no longer names the class raises rather than matching nothing;
the ``test_guard_*`` fixtures in ``test_streaming_plans.py`` catch the rest: a
known fallback that stops being reported fails them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

_NODE = re.compile(r'^(\d+) \[label="((?:[^"\\]|\\.)*)"(.*)\];$', re.M)
_FILL = re.compile(r'fillcolor="([^"]*)"')
#: The legend entry polars renders for the fallback class: its colour, then
#: the class name.
_LEGEND = re.compile(r'<FONT COLOR="([^"]*)">[^<]*</FONT> in-memory engine fallback')


@dataclass(frozen=True)
class Node:
    """A fallback node: its label, on one line."""

    label: str


def _flat(label: str) -> str:
    return " ".join(label.replace("\\n", " ").replace('\\"', '"').split())


def in_memory_nodes(lf: pl.LazyFrame) -> list[Node]:
    """``lf``'s nodes that polars runs on the in-memory engine."""
    dot = lf.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True, show=False
    )
    assert isinstance(dot, str)
    legend = _LEGEND.search(dot)
    if legend is None:
        raise AssertionError(
            "the streaming graph's legend no longer names the in-memory engine "
            "fallback colour; the scanner would match nothing"
        )
    fallback = legend.group(1)
    nodes = []
    for _, raw, attrs in _NODE.findall(dot):
        fill = _FILL.search(attrs)
        if fill is not None and fill.group(1) == fallback:
            nodes.append(Node(_flat(raw)))
    return nodes


@dataclass(frozen=True)
class KnownFallback:
    """A fallback the streaming engine cannot avoid or need not, and why."""

    pattern: str
    reason: str

    def matches(self, node: Node) -> bool:
        return re.search(self.pattern, node.label) is not None


#: The closed list of accepted fallbacks. Anything else is a regression.
KNOWN_FALLBACKS: tuple[KnownFallback, ...] = (
    KnownFallback(
        pattern=(
            r'^in-memory-map AGGREGATE\[[^]]*\] \[(col\("[^"]+"\)(\.slice\([^)]*\))?'
            r'\.implode\(\)\.alias\("(pred|pred_score|pred_row|gt|gt_row)"\)'
            r"(, )?)+\] BY"
        ),
        reason=(
            "group_objects: each (image, class)'s object lists. The matchers "
            "are elementwise over one row per (image, class) (greedy matching "
            "is sequential within an image), so the lists have to exist, and "
            "polars 2.0 builds a list only in memory: aggregations, "
            "`implode().over()` and `over(mapping_strategy='join')` all fall "
            "back. What it holds is object geometry, never an image: an "
            "instance mask is reduced to its outline elementwise before it. "
            "Callers who hold one row per image (lists, heatmaps) call a "
            "matcher directly and skip this step."
        ),
    ),
)


def unexplained(nodes: list[Node]) -> list[str]:
    """The labels of the fallbacks no :data:`KNOWN_FALLBACKS` entry accounts for."""
    return [
        node.label
        for node in nodes
        if not any(k.matches(node) for k in KNOWN_FALLBACKS)
    ]
