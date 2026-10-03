"""Every public metrics plan stays on polars' native streaming engine.

A step the streaming engine cannot run natively becomes an ``in-memory-map``
node, and that node collects its whole input first. Most of polars-cv is
elementwise and streams. The metrics are the one place where an innocent
``.over()``, a sort inside an ``agg`` or a list aggregation quietly turned
streaming into "collect everything".

``test_plan_stays_streaming`` builds each public lazy entry point and fails on
any fallback that :data:`KNOWN_FALLBACKS` does not explain. The ``test_guard_*``
fixtures make sure the scanner itself still sees fallbacks, so a polars
upgrade that renames the node cannot turn this file green by matching nothing.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

import polars_cv.metrics as M
from tests._streaming_guard import KNOWN_FALLBACKS, in_memory_nodes, unexplained

from .conftest import plugin_required

pytestmark = pytest.mark.structural

_LF = pl.LazyFrame({"g": [1, 1, 2], "x": [1.0, 2.0, 3.0], "t": [3, 2, 1]})


# -- the scanner ------------------------------------------------------------


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(
            _LF.with_columns(pl.col("x").cum_sum().over("g")), id="cum_sum-over"
        ),
        pytest.param(
            _LF.group_by("g").agg(pl.col("x").sort().cum_sum().last()),
            id="sorted-scan-in-agg",
        ),
        pytest.param(
            _LF.with_columns(pl.int_range(pl.len()).over("g")), id="int_range-over"
        ),
        pytest.param(
            _LF.with_columns(pl.col("x").map_batches(lambda s: s)), id="column-udf"
        ),
        pytest.param(
            _LF.sort("x", "g").with_columns(pl.col("x").cum_sum().over("g")),
            id="over-after-a-sort-on-other-keys",
        ),
        pytest.param(
            _LF.sort("g", "x").with_columns(pl.col("x").rank().over("g")),
            id="unlisted-window-after-its-sort",
        ),
        pytest.param(
            # A computed column between the sort and the window: the window's
            # input is that computation, not the frame the sort holds.
            _LF.sort("g", "x")
            .with_columns(y=pl.col("x").cum_sum())
            .with_columns(pl.col("y").cum_sum().over("g")),
            id="over-not-straight-after-the-sort",
        ),
    ],
)
def test_guard_rejects_known_fallbacks(plan: pl.LazyFrame) -> None:
    nodes = in_memory_nodes(plan)
    assert nodes, "the scanner no longer sees in-memory fallbacks"
    assert unexplained(nodes) == [n.label for n in nodes]


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(
            _LF.group_by("g").agg(pl.col("x").alias("pred"), pl.col("t").alias("gt")),
            id="object-lists",
        ),
        pytest.param(
            _LF.sort("g", "x").with_columns(
                pl.col("x").cum_sum().over("g"),
                pl.col("x").cum_max(reverse=True).over("g").alias("m"),
                pl.col("x").shift(1).over("g").alias("p"),
                pl.int_range(pl.len(), dtype=pl.Int64).over("g").alias("i"),
            ),
            id="scans-straight-after-their-sort",
        ),
    ],
)
def test_guard_accepts_the_known_fallbacks(plan: pl.LazyFrame) -> None:
    nodes = in_memory_nodes(plan)
    assert nodes
    assert unexplained(nodes) == []


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(_LF.group_by("g").agg(pl.col("x")), id="unnamed-list"),
        pytest.param(
            _LF.group_by("g").agg(pl.col("x").sort().alias("pred")), id="sorted-list"
        ),
    ],
)
def test_guard_rejects_other_list_building(plan: pl.LazyFrame) -> None:
    """The allow-list names its sites; any other list aggregation is new."""
    nodes = in_memory_nodes(plan)
    assert nodes
    assert unexplained(nodes) == [n.label for n in nodes]


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(_LF.sort("x").with_columns(pl.col("x").cum_sum()), id="sort-scan"),
        pytest.param(_LF.group_by("g").agg(pl.col("x").sum()), id="group-sum"),
        pytest.param(_LF.sort("x").with_columns(pl.col("x").shift()), id="shift"),
        pytest.param(
            _LF.join(pl.LazyFrame({"t": [0.5]}), how="cross"), id="cross-join"
        ),
        pytest.param(
            _LF.with_columns(pl.col("x").map_batches(lambda s: s, is_elementwise=True)),
            id="elementwise-udf",
        ),
        pytest.param(_LF.with_columns(pl.col("x").rank()), id="rank"),
    ],
)
def test_guard_accepts_native_nodes(plan: pl.LazyFrame) -> None:
    assert in_memory_nodes(plan) == []


# -- the metrics plans ------------------------------------------------------

_PREDS = pl.DataFrame(
    {
        "image_id": ["a", "a", "b", "c", "c", "d"],
        "class_id": ["cat", "cat", "dog", "dog", "cat", "dog"],
        "bbox": [
            [0, 0, 10, 6],
            [0, 0, 10, 8],
            [5, 5, 10, 10],
            [0, 0, 4, 4],
            [2, 2, 5, 5],
            [1, 1, 3, 3],
        ],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    }
)
_GTS = pl.DataFrame(
    {
        "image_id": ["a", "b", "c", "d"],
        "class_id": ["cat", "dog", "dog", "cat"],
        "bbox": [[0, 0, 10, 10], [5, 5, 10, 10], [1, 1, 4, 4], [1, 1, 3, 3]],
    }
)
_IMAGES = pl.DataFrame(
    {
        "image_id": ["a", "b", "c", "d"],
        "weight": [1.0, 2.0, 0.5, 1.5],
        "site": ["x", "x", "y", "y"],
    }
)

_SQUARE = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
_MASKS = pl.DataFrame({"image_id": ["a", "b"], "class_id": "c", "mask": [_SQUARE] * 2})


def _table() -> M.DetectionTable:
    return M.match_detections(
        _PREDS,
        _GTS,
        box_format="xywh",
        images=_IMAGES,
        weight="weight",
        group="site",
    )


_BOOT = {"n_bootstrap": 4, "seed": 0}

PLANS: dict[str, Callable[[], pl.LazyFrame]] = {
    "group_objects": lambda: M.group_objects(_PREDS, _GTS, geometry="bbox"),
    "group_objects(max_detections)": lambda: M.group_objects(
        _PREDS, _GTS, geometry="bbox", max_detections=1
    ),
    "match_detections": lambda: _table().detections,
    "match_detections(mask)": lambda: (
        M.match_detections(
            _MASKS.with_columns(score=pl.lit(0.9)), _MASKS, geometry="mask"
        ).detections
    ),
    "PreMatchedAdapter": lambda: (
        M.PreMatchedAdapter()
        .match(
            pl.DataFrame(
                {
                    "image_id": ["a", "a", "b"],
                    "score": [0.9, 0.4, 0.7],
                    "is_tp": [True, False, True],
                }
            ),
            image_id_col="image_id",
            image_meta=pl.DataFrame({"image_id": ["a", "b", "c"], "n_gts": [1, 1, 0]}),
        )
        .detections
    ),
    "AP(all_points)": lambda: M.AP().by_group(_table(), "class_id"),
    "AP(11_point)": lambda: M.AP("11_point").by_group(_table(), "class_id"),
    "AP(101_point)": lambda: M.AP("101_point").by_group(_table()),
    "mean_ap": lambda: M.mean_ap().by_group(_table()),
    "Recall": lambda: M.Recall().by_group(_table(), "class_id"),
    "PrecisionAt": lambda: M.PrecisionAt(0.5).by_group(_table()),
    "FROCSensitivity": lambda: M.FROCSensitivity(1.0).by_group(_table()),
    "CPM": lambda: M.CPM().by_group(_table(), "group_id"),
    "FROCAUC": lambda: M.FROCAUC(fp_range=(0.0, 2.0)).by_group(_table()),
    "LROCAUC": lambda: M.LROCAUC().by_group(_table()),
    "LROCSensitivity": lambda: M.LROCSensitivity(0.5).by_group(_table()),
    "froc_curve_lazy": lambda: M.froc_curve_lazy(_table(), group_by="group_id"),
    "froc_sensitivity_at_fp": lambda: M.froc_sensitivity_at_fp(_table(), 1.0),
    "froc_summary_table": lambda: M.froc_summary_table(_table()),
    "froc_operating_range": lambda: M.froc_operating_range(_table()),
    "lroc_curve_lazy": lambda: M.lroc_curve_lazy(_table(), group_by="group_id"),
    "lroc_sensitivity_at_fpf": lambda: M.lroc_sensitivity_at_fpf(_table(), 0.5),
    "bootstrap_ci(AP)": lambda: M.bootstrap_ci(_table(), M.AP(), **_BOOT),
    "bootstrap_ci(mean_ap, strata)": lambda: M.bootstrap_ci(
        _table(), M.mean_ap(), strata="group_id", **_BOOT
    ),
    "bootstrap_ci(Recall, sample_col)": lambda: M.bootstrap_ci(
        _table(), M.Recall(), sample_col="group_id", **_BOOT
    ),
    "average_precision_ci_lazy": lambda: M.average_precision_ci_lazy(
        _table(), group_by="class_id", **_BOOT
    ),
    "froc_auc_ci_lazy": lambda: M.froc_auc_ci_lazy(
        _table(), fp_range=(0.0, 2.0), **_BOOT
    ),
    "lroc_auc_ci_lazy": lambda: M.lroc_auc_ci_lazy(_table(), **_BOOT),
}


@plugin_required
@pytest.mark.parametrize("name", list(PLANS))
def test_plan_stays_streaming(name: str) -> None:
    offenders = unexplained(in_memory_nodes(PLANS[name]()))
    assert offenders == [], (
        f"{name} falls back to the in-memory engine at:\n  "
        + "\n  ".join(offenders)
        + "\nWrite the scan through metrics/_grouped_scan.py, or rewrite the "
        "step natively, or add it to KNOWN_FALLBACKS with the reason it "
        "cannot be."
    )


@plugin_required
def test_every_known_fallback_is_still_hit() -> None:
    """A fallback that was fixed must leave the allow-list too."""
    nodes = [node for build in PLANS.values() for node in in_memory_nodes(build())]
    stale = [k.pattern for k in KNOWN_FALLBACKS if not any(map(k.matches, nodes))]
    assert stale == []
