"""Which steps of a lazy plan the streaming engine hands to the in-memory engine.

Polars' streaming engine runs what it can natively and wraps everything else in
an ``in-memory-map`` node, which collects that node's whole input before it
runs. A native node that happens to buffer (``sort``, ``group-by``, a join's
build side) is *not* a fallback; it is the streaming engine doing its job.

:func:`in_memory_nodes` reads the physical streaming graph
(``show_graph(engine="streaming", plan_stage="physical", raw_output=True)``)
and returns every ``in-memory-map`` node, with the node that feeds it, plus
any Python UDF that is handed its whole input column.

Not every fallback costs memory. A ``.over()`` scan right after a ``sort`` on
its keys reads the frame that sort has already collected. The streaming
alternative (``grouped_scan`` doing all of it natively) was measured to hold
just as much memory and to run 2.4x slower. Such a node is accepted only when
the graph shows exactly that: its input is a ``sort`` whose leading keys are
the window's keys (:data:`KNOWN_FALLBACKS`).

Limits: this scans a debug rendering, relying on polars' node names
(``in-memory-map``, ``sort``, ``multiplexer``, a projection's ``select``), its
label layout and
``show_graph(plan_stage=)``. If any of these changes, the scan could silently
match nothing. The ``test_guard_*`` fixtures in ``test_streaming_plans.py``
catch that: a known fallback that stops being reported fails them.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

_NODE = re.compile(r'^(\d+) \[label="((?:[^"\\]|\\.)*)"', re.M)
_EDGE = re.compile(r"^(\d+) -> (\d+);", re.M)
_FALLBACK = "in-memory-map"
#: A Python UDF that is not marked elementwise is handed its whole input
#: column at once: as costly as a fallback, under another node name.
_WHOLE_COLUMN_UDF = re.compile(r"^columnar-function .*python_udf")


def _passes_through(label: str) -> bool:
    """A node that only fans a stream out or drops columns: looked through to
    find what feeds a fallback. A ``select`` that computes anything is not."""
    return label == "multiplexer" or (label.startswith("select ") and "=" not in label)


@dataclass(frozen=True)
class Node:
    """A fallback node: its label (one line) and its input's label."""

    label: str
    input: str | None


def _flat(label: str) -> str:
    return " ".join(label.replace("\\n", " ").replace('\\"', '"').split())


def in_memory_nodes(lf: pl.LazyFrame) -> list[Node]:
    """``lf``'s in-memory fallback nodes and whole-column Python UDFs."""
    dot = lf.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True, show=False
    )
    assert isinstance(dot, str)
    labels = {node: _flat(label) for node, label in _NODE.findall(dot)}
    feeds = {dst: src for src, dst in _EDGE.findall(dot)}

    def input_of(node: str) -> str | None:
        src = feeds.get(node)
        while src is not None and _passes_through(labels[src]):
            src = feeds.get(src)
        return None if src is None else labels[src]

    return [
        Node(label, input_of(node))
        for node, label in labels.items()
        if label.startswith(_FALLBACK) or _WHOLE_COLUMN_UDF.match(label)
    ]


@dataclass(frozen=True)
class KnownFallback:
    """A fallback the streaming engine cannot avoid or need not, and why."""

    pattern: str
    reason: str
    #: A further condition on the node, beyond its label.
    holds: Callable[[Node], bool] = lambda node: True

    def matches(self, node: Node) -> bool:
        return re.search(self.pattern, node.label) is not None and self.holds(node)


#: One window expression of a sorted scan: ``_POLARS_TMP_n = <scan>.over([keys])``,
#: ``<scan>`` one of grouped_scan's: a running sum or max (polars prints
#: ``cum_max(reverse=True)`` as ``cum_max()``), a shift, or the row position
#: alone or compared with the first or last position.
_POSITION = r"0\.int_range\(\[len\(\)\.cast\(Int64\)\]\)"
_WINDOW = re.compile(
    r"_POLARS_TMP_\d+ = (?:"
    r'col\("[^"]+"\)\.(?:cum_sum\(\)|cum_max\(\)|shift\(\[dyn int: -?\d+\]\))'
    rf"|{_POSITION}"
    rf"|\({_POSITION} == (?:0|\(len\(\) - 1\)\.cast\(Int64\))\)"
    r')\.over\(\[((?:col\("[^"]+"\)(?:, )?)+)\]\)'
)
_SORTED_SCAN = re.compile(r"^in-memory-map SELECT \[ (?:" + _WINDOW.pattern + r" )+\]$")
_COL = re.compile(r'col\("([^"]+)"\)')


def _scan_on_its_sort(node: Node) -> bool:
    """The node's input is a ``sort`` whose leading keys are every window's keys."""
    if node.input is None or not node.input.startswith("sort "):
        return False
    sort_keys = re.findall(r'(\S+) = col\("\1"\)', node.input)
    for window in _WINDOW.finditer(node.label):
        keys = _COL.findall(window.group(1))
        if sort_keys[: len(keys)] != keys:
            return False
    return True


#: The closed list of accepted fallbacks. Anything else is a regression.
KNOWN_FALLBACKS: tuple[KnownFallback, ...] = (
    KnownFallback(
        pattern=_SORTED_SCAN.pattern,
        holds=_scan_on_its_sort,
        reason=(
            "grouped_scan: running sums/maxima, lags and row indices .over() "
            "the keys, straight after the sort on those keys. The sort has "
            "already collected the frame, so the window adds no memory; the "
            "all-native alternative held as much and ran 2.4x slower."
        ),
    ),
    KnownFallback(
        pattern=(
            r'^in-memory-map AGGREGATE\[[^]]*\] \[(col\("[^"]+"\)(\.slice\([^)]*\))?'
            r'\.implode\(\)\.alias\("(pred|pred_score|pred_row|gt|gt_row)"\)'
            r"(, )?)+\] BY"
        ),
        reason=(
            "group_objects: each (image, class)'s object lists. The matchers "
            "are elementwise over one row per (image, class), so the lists "
            "have to exist; polars cannot build a list inside a streaming "
            "group-by. Users who already hold per-image lists skip this."
        ),
    ),
    KnownFallback(
        pattern=r'\.unique\(\)\.sort\(asc\)\.implode\(\)\.alias\("_cell"\)\] BY',
        reason=(
            "_sampling_units: the set of weight cells a sampling unit's "
            "metadata rows fall in (normally one). One row per unit."
        ),
    ),
    KnownFallback(
        pattern=(
            r'^in-memory-map AGGREGATE\[[^]]*\] \[col\("image_id"\)\.implode\(\)\] '
            r'BY \[col\("_entity"\)\]$'
        ),
        reason=(
            "_resolve_bootstrap_samples: the images of each sampling entity "
            "(sample_col=), exploded per draw. One row per entity."
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
