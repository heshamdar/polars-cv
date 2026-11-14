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
not a fallback.

Limits: this scans a debug rendering, relying on its legend wording ("in-memory
engine fallback"), its ``fillcolor`` attribute and ``show_graph(plan_stage=)``.
A legend that no longer names the class raises rather than matching nothing;
the ``test_guard_*`` fixtures in ``test_streaming_plans.py`` catch the rest: a
plan that falls back and is no longer reported fails them.
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
