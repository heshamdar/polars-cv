"""``PipelineGraph.show_graph()`` renders each node's alias, domain and dtype.

The visualizer used to read those from the plugin wire format, which carried
them only for its sake; it now reads them from the Python graph. This pins the
rendered labels so that move cannot drop them.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline

pytest.importorskip("networkx")
pytest.importorskip("graphviz")
pytest.importorskip("pydot")


def test_the_rendered_graph_names_alias_and_dtype() -> None:
    img = pl.col("img").cv.pipe(Pipeline().source("image_bytes", dtype="u8"))
    small = img.pipe(Pipeline().resize(height=8, width=8).cast("f32")).alias("small")
    graph = small.sink("numpy", return_expr=False)
    dot = graph.show_graph().source
    assert "small" in dot
    assert "f32" in dot
    assert "buffer" in dot or "resize" in dot
