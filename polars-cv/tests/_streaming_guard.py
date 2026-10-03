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
#: A Python UDF that is not marked elementwise is handed its whole input
#: column at once: as costly as a fallback, under another node name.
_WHOLE_COLUMN_UDF = re.compile(r"^columnar-function .*python_udf")


def in_memory_nodes(lf: pl.LazyFrame) -> list[str]:
    """The labels of ``lf``'s in-memory fallback nodes, one line each.

    Also a Python UDF that receives its whole input column (``map_batches``
    without ``is_elementwise=True``), which collects that column just the same.
    """
    dot = lf.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True, show=False
    )
    assert isinstance(dot, str)
    labels = (
        m.group(1).replace("\\n", " ").replace('\\"', '"') for m in _LABEL.finditer(dot)
    )
    flat = (" ".join(label.split()) for label in labels)
    return [
        label
        for label in flat
        if label.startswith(_FALLBACK) or _WHOLE_COLUMN_UDF.match(label)
    ]


@dataclass(frozen=True)
class KnownFallback:
    """A fallback the streaming engine cannot avoid, and why."""

    pattern: str
    reason: str

    def matches(self, label: str) -> bool:
        return re.search(self.pattern, label) is not None


#: The closed list of accepted fallbacks. Anything else is a regression.
#:
#: Each builds a list per key with ``group_by().agg(...)``, which polars cannot
#: do inside a streaming group-by. Each runs on a per-object or per-image frame
#: (O(objects) or O(images)), never on the bootstrap's replicate frame
#: (O(n_bootstrap × detections)), and never on pixel data.
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
            "have to exist. Users who already hold per-image lists skip this."
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
        pattern=r'^in-memory-map AGGREGATE\[[^]]*\] \[col\("image_id"\)\.implode\(\)\] BY \[col\("_entity"\)\]$',
        reason=(
            "_resolve_bootstrap_samples: the images of each sampling entity "
            "(sample_col=), exploded per draw. One row per entity."
        ),
    ),
)


def unexplained(labels: list[str]) -> list[str]:
    """The fallback labels that no :data:`KNOWN_FALLBACKS` entry accounts for."""
    return [
        label for label in labels if not any(k.matches(label) for k in KNOWN_FALLBACKS)
    ]
