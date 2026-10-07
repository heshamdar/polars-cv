"""Polars' common-subexpression elimination reaches polars-cv's plugin calls.

Since polars 2.0 a plugin function registered ``is_deterministic=True`` is
eligible for CSE: two equal ``vb_graph`` calls in one query run once and are
read twice. Polars compares a plugin call by its serialized kwargs, so a graph
must serialize identically every time the same pipeline is built — nothing
build-specific (such as a generated node id) may reach the wire.

Verified at the user-facing entry point: the optimized plan (``explain()``),
which is what polars executes.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

pytestmark = plugin_required


def _png(value: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.full((6, 5, 3), value, dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


_DF = pl.DataFrame({"img": [_png(v) for v in (0, 90, 200)]})
_PIPE = Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()


def _plugin_calls(lf: pl.LazyFrame) -> int:
    # A call with expression inputs prints as `vb_graph([...])`, one without
    # as `vb_graph()`: match the name and the opening parenthesis only.
    return lf.explain().count(":vb_graph(")


def _sink() -> pl.Expr:
    return pl.col("img").cv.pipe(_PIPE).sink("blob")


def test_separately_built_equal_pipelines_are_equal_expressions() -> None:
    assert _sink().meta.eq(_sink())


def test_separately_built_equal_pipelines_run_once() -> None:
    lf = _DF.lazy().with_columns(a=_sink().bin.size(), b=_sink().bin.size() * 2)
    assert _plugin_calls(lf) == 1
    out = lf.collect()
    assert (out["b"] == out["a"] * 2).all()


def test_composed_graphs_built_twice_are_equal() -> None:
    def build() -> pl.Expr:
        img = pl.col("img").cv.pipe(_PIPE).alias("base")
        gray = img.pipe(Pipeline().threshold(100)).alias("mask")
        return gray.sink({"base": "numpy", "mask": "png"})

    assert build().meta.eq(build())


def test_ops_reading_other_nodes_built_twice_are_equal() -> None:
    # A binary op and a node-sized rasterize name another node inside the op.
    def binary() -> pl.Expr:
        a = pl.col("img").cv.pipe(_PIPE)
        b = pl.col("img").cv.pipe(_PIPE.invert())
        return a.add(b).sink("numpy")

    def raster() -> pl.Expr:
        img = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
        return (
            pl.col("c")
            .cv.pipe(Pipeline().source("contour").rasterize(shape=img))
            .sink("numpy")
        )

    assert binary().meta.eq(binary())
    assert raster().meta.eq(raster())
    lf = _DF.lazy().with_columns(a=binary(), b=binary())
    assert _plugin_calls(lf) == 1
    out = lf.collect()
    assert out["a"].to_list() == out["b"].to_list()


def test_per_row_parameters_merge_only_on_the_same_column() -> None:
    # A per-row parameter is an extra plugin input, so polars compares the
    # column it reads, not just the `{"$slot": n}` in the kwargs.
    df = _DF.with_columns(h=pl.Series([2, 3, 4]), h2=pl.Series([5, 6, 7]))

    def sized(column: str) -> pl.Expr:
        pipe = Pipeline().source("image_bytes").resize(height=pl.col(column), width=4)
        return pl.col("img").cv.pipe(pipe).sink("numpy")

    same = df.lazy().with_columns(a=sized("h"), b=sized("h"))
    different = df.lazy().with_columns(a=sized("h"), b=sized("h2"))
    assert _plugin_calls(same) == 1
    assert _plugin_calls(different) == 2
    out = different.collect()
    assert out["a"].to_list() != out["b"].to_list()


def test_different_pipelines_are_not_merged() -> None:
    other = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale())
    lf = _DF.lazy().with_columns(a=_sink(), b=other.sink("blob"))
    assert _plugin_calls(lf) == 2
    out = lf.collect()
    assert out["a"].to_list() != out["b"].to_list()


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_merged_calls_give_the_unmerged_results(engine: str) -> None:
    lf = _DF.lazy().with_columns(a=_sink(), b=_sink())
    merged = lf.collect(engine=engine)
    unmerged = lf.collect(
        engine=engine, optimizations=pl.QueryOptFlags(comm_subexpr_elim=False)
    )
    assert merged.equals(unmerged)
    assert merged["a"].equals(merged["b"].alias("a"))
