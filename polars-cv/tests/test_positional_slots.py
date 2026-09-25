"""Expression parameters cross the plugin boundary as positional slots.

Typed-op plan P1. An expression parameter used to be identified by its display
text — with a ``#n`` suffix, from a process-wide registry, when
two live expressions printed alike — and Rust bound that name to an input
column through a parallel list of those names. The key depended on
which other expressions happened to be alive, so the graph JSON (the
compiled-graph cache key) was not a function of the pipeline alone.

Now a graph assigns each distinct expression (``Expr.meta.eq``) an input
position and parameters carry ``{"$slot": n}``: no text, no names, no registry.
"""

from __future__ import annotations

import json
import re

import numpy as np
import polars as pl

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import make_image_png, plugin_required


def _graph(expr_pipe: Pipeline):
    return pl.col("img").cv.pipe(expr_pipe).sink("numpy", return_expr=False)


def _params(spec: dict) -> list:
    return [
        v
        for node in spec["nodes"].values()
        for op in node["ops"]
        for k, v in op.items()
        if k != "op"
    ]


@plugin_required
class TestWireForm:
    def test_an_expression_parameter_is_a_slot(self) -> None:
        pipe = Pipeline().source("image_bytes").resize(height=pl.col("h"), width=8)
        spec = json.loads(_graph(pipe)._to_json())
        assert {"$slot": 1} in _params(spec)  # input 0 is the image column

    def test_the_graph_carries_no_expression_text(self) -> None:
        a = pl.lit(pl.Series("f", [1]))
        b = pl.lit(pl.Series("f", [2]))  # prints like `a`
        pipe = Pipeline().source("image_bytes").resize(height=a, width=b)
        text = _graph(pipe)._to_json()
        assert "Series" not in text
        assert '"expr"' not in text

    def test_json_does_not_depend_on_unrelated_live_expressions(self) -> None:
        def build() -> str:
            x = pl.lit(pl.Series("f", [3]))
            pipe = Pipeline().source("image_bytes").resize(height=x, width=8)
            # Node ids are random per build (CR-50); only the expression
            # encoding is under test here.
            return re.sub(r"node_[0-9a-f]+", "node", _graph(pipe)._to_json())

        alone = build()
        _other = pl.lit(pl.Series("f", [4]))  # a live look-alike
        assert build() == alone


@plugin_required
class TestSlotsAcrossPipelines:
    """Two pipelines' slot 0 are different inputs unless the expressions are."""

    def _frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"img": [make_image_png(16, 16, 3, seed=3)] * 2, "a": [4, 6], "b": [8, 10]}
        )

    def test_distinct_expressions_at_the_same_position_do_not_merge(self) -> None:
        pa = Pipeline().source("image_bytes").resize(height=pl.col("a"), width=4)
        pb = Pipeline().source("image_bytes").resize(height=pl.col("b"), width=4)
        out = self._frame().select(
            pl.col("img")
            .cv.pipe(pa)
            .alias("x")
            .merge_pipe(pl.col("img").cv.pipe(pb).alias("y"))
            .sink({"x": "numpy", "y": "numpy"})
            .alias("o")
        )
        heights = [
            (
                numpy_from_struct(row["x"]).shape[0],
                numpy_from_struct(row["y"]).shape[0],
            )
            for row in out["o"].to_list()
        ]
        assert heights == [(4, 8), (6, 10)]

    def test_equal_expressions_share_one_input(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("a"), width=pl.col("a"))
        )
        spec = json.loads(_graph(pipe)._to_json())
        slots = [p for p in _params(spec) if isinstance(p, dict) and "$slot" in p]
        assert slots == [{"$slot": 1}, {"$slot": 1}]
        out = self._frame().select(pl.col("img").cv.pipe(pipe).sink("numpy"))
        shapes = [numpy_from_struct(r).shape[:2] for r in out["img"].to_list()]
        assert shapes == [(4, 4), (6, 6)]
        assert np.all(np.array(shapes) > 0)
