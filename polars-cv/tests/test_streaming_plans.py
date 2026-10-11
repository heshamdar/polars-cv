"""Every public metrics plan stays on polars' native streaming engine.

A step the streaming engine cannot run natively is handed to the in-memory
engine (an ``in-memory-map``, ``in-memory-join`` or ``columnar-function``
node), and that node collects its whole input first. Most of polars-cv is
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
from polars_cv import Pipeline
from tests._streaming_guard import KNOWN_FALLBACKS, in_memory_nodes, unexplained

from .conftest import plugin_required

pytestmark = pytest.mark.structural

_LF = pl.LazyFrame({"g": [1, 1, 2], "x": [1.0, 2.0, 3.0], "t": [3, 2, 1]})


# -- the scanner ------------------------------------------------------------


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(
            _LF.group_by("g").agg(pl.col("x").sort().cum_sum().last()),
            id="sorted-scan-in-agg",
        ),
        pytest.param(
            _LF.with_columns(pl.col("x").map_batches(lambda s: s)), id="column-udf"
        ),
        # Polars' own columnar function: the whole frame, then a range over it.
        pytest.param(
            _LF.sort("x").with_columns(pl.int_range(pl.len()) == 0),
            id="unwindowed-int_range",
        ),
        pytest.param(
            _LF.sort("x").join_asof(_LF.sort("x"), on="x", by="g"),
            id="join_asof-by",
        ),
        # A frame-wide rank (polars-cv ranks only inside `list.eval`).
        pytest.param(_LF.with_columns(pl.col("x").rank()), id="rank"),
    ],
)
def test_guard_rejects_known_fallbacks(plan: pl.LazyFrame) -> None:
    nodes = in_memory_nodes(plan)
    assert nodes, "the scanner no longer sees in-memory fallbacks"
    assert unexplained(nodes) == [n.label for n in nodes]


def test_guard_refuses_a_graph_without_the_fallback_legend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the legend naming the class, nothing could match: raise."""
    plan = _LF.group_by("g").agg(pl.col("x"))
    dot = plan.show_graph(
        engine="streaming", plan_stage="physical", raw_output=True, show=False
    )
    assert isinstance(dot, str)
    renamed = dot.replace("in-memory engine fallback", "fallback")
    monkeypatch.setattr(pl.LazyFrame, "show_graph", lambda *a, **k: renamed)
    with pytest.raises(AssertionError, match="no longer names"):
        in_memory_nodes(plan)


@plugin_required
def test_guard_rejects_a_whole_column_plugin_call() -> None:
    """A plugin call not declared elementwise gets the whole column in one
    call; the scanner must report it (polars-cv declares every call
    elementwise in ``_plugin.call``, the one way in)."""
    from polars.plugins import register_plugin_function

    from polars_cv._plugin import plugin_path

    def call(elementwise: bool) -> pl.LazyFrame:
        expr = register_plugin_function(
            plugin_path=plugin_path(),
            function_name="read_file_bytes",
            args=[pl.col("p")],
            is_elementwise=elementwise,
        )
        return pl.LazyFrame({"p": ["a.png"]}).select(expr)

    nodes = in_memory_nodes(call(elementwise=False))
    assert nodes, "the scanner no longer sees a whole-column plugin call"
    assert unexplained(nodes) == [n.label for n in nodes]
    assert in_memory_nodes(call(elementwise=True)) == []


@pytest.mark.parametrize(
    "plan",
    [
        pytest.param(
            _LF.group_by("g").agg(pl.col("x").alias("pred"), pl.col("t").alias("gt")),
            id="object-lists",
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
        pytest.param(_LF.sort("x").with_row_index("i"), id="row-index"),
        pytest.param(
            _LF.sort("x").join_asof(_LF.sort("x"), on="x"), id="join_asof-no-by"
        ),
        pytest.param(
            _LF.sort("g", "x").with_columns(
                pl.col("x").forward_fill().over("g"),
                pl.col("t").backward_fill().over("g"),
            ),
            id="fill-over",
        ),
        # Windows are native streaming nodes since polars 2.0.
        pytest.param(_LF.with_columns(pl.col("x").cum_sum().over("g")), id="over"),
        pytest.param(
            _LF.with_columns(pl.int_range(pl.len()).over("g")), id="int_range-over"
        ),
        pytest.param(
            _LF.sort("x", "g").with_columns(pl.col("x").cum_sum().over("g")),
            id="over-after-a-sort-on-other-keys",
        ),
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

_PATHS = pl.LazyFrame({"p": ["a.png", "b.png"], "b": [b"x", b"y"]})
_CONTOURS = pl.LazyFrame(
    {
        "c": [
            {
                "exterior": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 4.0, "y": 0.0},
                    {"x": 0.0, "y": 4.0},
                ],
                "holes": [],
                "is_closed": True,
            }
        ],
        "q": [{"x": 1.0, "y": 1.0}],
    }
)

PLANS: dict[str, Callable[[], pl.LazyFrame]] = {
    # The plugin's own entry points: each streaming morsel is one call.
    "cv.pipe": lambda: _PATHS.select(
        pl.col("b").cv.pipe(Pipeline().source("raw", dtype="u8")).sink("blob")
    ),
    "cv.read_bytes": lambda: _PATHS.select(pl.col("p").cv.read_bytes()),
    "cv.width": lambda: _PATHS.select(pl.col("p").cv.width()),
    "contour.area": lambda: _CONTOURS.select(pl.col("c").contour.area()),
    "point.translate": lambda: _CONTOURS.select(pl.col("q").point.translate(1.0, 2.0)),
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
    # Every public lazy entry point and the paths its options take: grouped
    # (the bracketing joins), partial and extrapolated (interpolation), the
    # weighted bootstrap's weight clusters, and an unweighted table.
    "match_detections(unweighted)": lambda: (
        M.match_detections(_PREDS, _GTS, box_format="xywh").detections
    ),
    "F1At": lambda: M.F1At(0.5).by_group(_table(), "class_id"),
    "RecallAt": lambda: M.RecallAt(0.5).by_group(_table()),
    "froc_auc": lambda: M.froc_auc(_table(), fp_range=(0.0, 2.0)),
    "froc_auc(mann_whitney, grouped)": lambda: M.froc_auc(
        _table(), method="mann_whitney", group_by="group_id"
    ),
    "froc_auc(partial, flat, image)": lambda: M.froc_auc(
        _table(), fp_range=(0.0, 2.0), extrapolate="flat", level="image"
    ),
    "froc_auc(partial, grouped)": lambda: M.froc_auc(
        _table(), fp_range=(0.0, 2.0), group_by="group_id"
    ),
    "lroc_auc": lambda: M.lroc_auc(_table()),
    "lroc_auc(top_scoring, mann_whitney)": lambda: M.lroc_auc(
        _table(), variant="top_scoring", method="mann_whitney"
    ),
    "lroc_auc(partial, grouped)": lambda: M.lroc_auc(
        _table(), fpf_range=(0.0, 0.5), group_by="group_id", extrapolate="flat"
    ),
    "froc_summary_table(fp_rates, grouped)": lambda: M.froc_summary_table(
        _table(), [0.5, 1.0, 2.0], group_by="group_id"
    ),
    "froc_sensitivity_at_fp(grouped)": lambda: M.froc_sensitivity_at_fp(
        _table(), 1.0, group_by="group_id"
    ),
    "lroc_sensitivity_at_fpf(grouped)": lambda: M.lroc_sensitivity_at_fpf(
        _table(), 0.5, group_by="group_id"
    ),
    "froc_auc_ci_lazy(fixed weights, grouped)": lambda: M.froc_auc_ci_lazy(
        _table(),
        fp_range=(0.0, 2.0),
        group_by="group_id",
        weight_scheme="fixed",
        **_BOOT,
    ),
    "froc_auc_ci_lazy(strata, sample_col)": lambda: M.froc_auc_ci_lazy(
        _table(),
        method="mann_whitney",
        strata="group_id",
        sample_col="group_id",
        **_BOOT,
    ),
    "lroc_auc_ci_lazy(partial, grouped)": lambda: M.lroc_auc_ci_lazy(
        _table(), fpf_range=(0.0, 0.5), group_by="group_id", **_BOOT
    ),
    "average_precision_ci_lazy(fixed weights)": lambda: M.average_precision_ci_lazy(
        _table(), weight_scheme="fixed", **_BOOT
    ),
}


@plugin_required
@pytest.mark.parametrize("name", list(PLANS))
def test_plan_stays_streaming(name: str) -> None:
    offenders = unexplained(in_memory_nodes(PLANS[name]()))
    assert offenders == [], (
        f"{name} falls back to the in-memory engine at:\n  "
        + "\n  ".join(offenders)
        + "\nRewrite the step natively, or add it to KNOWN_FALLBACKS with the "
        "reason it cannot be."
    )


@plugin_required
def test_every_known_fallback_is_still_hit() -> None:
    """A fallback that was fixed must leave the allow-list too."""
    nodes = [node for build in PLANS.values() for node in in_memory_nodes(build())]
    stale = [k.pattern for k in KNOWN_FALLBACKS if not any(map(k.matches, nodes))]
    assert stale == []


# -- the eager entry points -------------------------------------------------
#
# These build a plan and collect it themselves, so no lazy frame reaches the
# caller to scan. Every `collect` / `collect_all` they make is scanned instead.
# (Polars runs eager DataFrame methods through a separate path, so only real
# lazy collects are seen.)

_HEAT = [[0.0] * 8 for _ in range(8)]
_HEAT[2][2] = _HEAT[2][3] = _HEAT[3][2] = _HEAT[3][3] = 0.9
_GT = [[1.0 if 2 <= y < 4 and 2 <= x < 4 else 0.0 for x in range(8)] for y in range(8)]
_HEATMAPS = pl.DataFrame({"heat": [_HEAT, _GT], "gt": [_GT, _GT]})

EAGER: dict[str, Callable[[], object]] = {
    "average_precision": lambda: M.average_precision(_table()),
    "average_precision(101_point)": lambda: M.average_precision(
        _table(), interpolation="101_point"
    ),
    "mean_average_precision": lambda: M.mean_average_precision(_table()),
    "precision_recall_curve": lambda: M.precision_recall_curve(_table()),
    "confusion_at_threshold": lambda: M.confusion_at_threshold(_table(), 0.5),
    "precision_at_threshold": lambda: M.precision_at_threshold(_table(), 0.5),
    "recall_at_threshold": lambda: M.recall_at_threshold(_table(), 0.5),
    "f1_at_threshold": lambda: M.f1_at_threshold(_table(), 0.5),
    "evaluate_detections": lambda: M.evaluate_detections(
        _PREDS,
        _GTS,
        box_format="xywh",
        images=_IMAGES,
        weight="weight",
        group="site",
        fp_rates=[0.5, 1.0],
    ),
    "evaluate_heatmaps": lambda: M.evaluate_heatmaps(
        _HEATMAPS, heatmap="heat", gt="gt", fp_rates=[0.5, 1.0]
    ),
    "evaluate_segmentation": lambda: M.evaluate_segmentation(
        _HEATMAPS, pred="heat", target="gt"
    ),
}


@plugin_required
@pytest.mark.parametrize("name", list(EAGER))
def test_eager_collects_stay_streaming(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    plans: list[pl.LazyFrame] = []
    collect, collect_all = pl.LazyFrame.collect, pl.collect_all

    def scanned_collect(self: pl.LazyFrame, *args: object, **kwargs: object):  # noqa: ANN202
        plans.append(self)
        return collect(self, *args, **kwargs)

    def scanned_collect_all(frames, *args: object, **kwargs: object):  # noqa: ANN001, ANN202
        frames = list(frames)
        plans.extend(frames)
        return collect_all(frames, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", scanned_collect)
    monkeypatch.setattr(pl, "collect_all", scanned_collect_all)
    EAGER[name]()
    monkeypatch.undo()
    assert plans, f"{name} collected nothing the scan could see"
    offenders = [label for lf in plans for label in unexplained(in_memory_nodes(lf))]
    assert offenders == [], (
        f"{name} collects a plan that falls back to the in-memory engine at:\n  "
        + "\n  ".join(dict.fromkeys(offenders))
    )
