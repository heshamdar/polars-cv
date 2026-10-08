"""Which steps of a lazy plan the streaming engine hands to the in-memory engine.

Polars' streaming engine runs what it can natively and wraps everything else in
an ``in-memory-map`` node, which collects that node's whole input before it
runs. A native node that happens to buffer (``sort``, ``group-by``, a join's
build side) is *not* a fallback; it is the streaming engine doing its job.

:func:`in_memory_nodes` reads the physical streaming graph
(``show_graph(engine="streaming", plan_stage="physical", raw_output=True)``)
and returns every ``in-memory-map`` node, plus any Python UDF or compiled
plugin call that is handed its whole input column (a ``columnar-function``).
A plugin call is one only when it is not declared elementwise: every polars-cv
function is, in ``polars_cv._plugin.call``, so that each streaming morsel is
its own call.

Since polars 2.0 a window ``.over()`` is a native streaming node (``window``),
not a fallback; what remains are the list-building aggregations
:data:`KNOWN_FALLBACKS` names, each with its reason.

Limits: this scans a debug rendering, relying on polars' node names
(``in-memory-map``, ``columnar-function``), its label layout and
``show_graph(plan_stage=)``. If any of these changes, the scan could silently
match nothing. The ``test_guard_*`` fixtures in ``test_streaming_plans.py``
catch that: a known fallback that stops being reported fails them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

_NODE = re.compile(r'^(\d+) \[label="((?:[^"\\]|\\.)*)"', re.M)
_FALLBACK = "in-memory-map"
#: A Python UDF or a compiled plugin call that is not marked elementwise is
#: handed its whole input column at once: as costly as a fallback, under
#: another node name. A plugin call names its library (``<path>.so:<fn>``);
#: polars' own columnar functions (``int_range``) do not.
_WHOLE_COLUMN_UDF = re.compile(
    r"^columnar-function .*(python_udf|\.(so|pyd|dll|dylib):)"
)


@dataclass(frozen=True)
class Node:
    """A fallback node: its label, on one line."""

    label: str


def _flat(label: str) -> str:
    return " ".join(label.replace("\\n", " ").replace('\\"', '"').split())


def in_memory_nodes(lf: pl.LazyFrame) -> list[Node]:
    """``lf``'s in-memory fallback nodes and whole-column Python UDFs."""
    dot = lf.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True, show=False
    )
    assert isinstance(dot, str)
    return [
        Node(label)
        for label in (_flat(raw) for _, raw in _NODE.findall(dot))
        if label.startswith(_FALLBACK) or _WHOLE_COLUMN_UDF.match(label)
    ]


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
            "are elementwise over one row per (image, class), so the lists "
            "have to exist; polars cannot build a list inside a streaming "
            "group-by. Users who already hold per-image lists skip this."
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
