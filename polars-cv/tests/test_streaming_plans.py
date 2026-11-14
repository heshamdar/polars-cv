"""The plugin's entry points stay on polars' native streaming engine.

A step the streaming engine cannot run natively is handed to the in-memory
engine (an ``in-memory-map``, ``in-memory-join`` or ``columnar-function``
node), and that node collects its whole input first. Every plugin call is
elementwise, so each streaming morsel is one call; a call that lost that
declaration would be handed the whole column at once.

``test_plan_stays_streaming`` builds each entry point and fails on any
fallback. The ``test_guard_*`` fixtures make sure the scanner itself still sees
fallbacks, so a polars upgrade that renames the node cannot turn this file
green by matching nothing.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pytest

from polars_cv import Pipeline
from tests._streaming_guard import in_memory_nodes

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
    assert in_memory_nodes(call(elementwise=True)) == []


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


# -- the plans ---------------------------------------------------------------

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
}


@plugin_required
@pytest.mark.parametrize("name", list(PLANS))
def test_plan_stays_streaming(name: str) -> None:
    offenders = [node.label for node in in_memory_nodes(PLANS[name]())]
    assert offenders == [], (
        f"{name} falls back to the in-memory engine at:\n  " + "\n  ".join(offenders)
    )
