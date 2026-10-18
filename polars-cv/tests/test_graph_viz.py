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


def _labels(graph) -> dict[str, str]:
    """Each rendered node's label, by node name: what a reader sees, not the
    rest of the DOT text."""
    import pydot

    (dot,) = pydot.graph_from_dot_data(graph.show_graph().source)
    return {
        n.get_name(): n.get_label().strip('"').replace("\\n", "\n")
        for n in dot.get_nodes()
        if n.get_label()
    }


def test_a_source_node_renders_its_ops() -> None:
    """A root node holds the decode *and* the ops after it (CSE moves a shared
    prefix there); the label shows both."""
    gray = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale())
    (label, _sink) = _labels(gray.sink("numpy", return_expr=False)).values()
    assert "image_bytes" in label
    assert "grayscale" in label


def test_nodes_are_labelled_by_the_id_operands_name() -> None:
    """An operand is named by its node's wire id, so each node's label leads
    with that id, and no ``None`` stands in for a missing alias."""
    img = pl.col("img").cv.pipe(Pipeline().source("image_bytes", dtype="u8"))
    gray = img.pipe(Pipeline().grayscale())
    graph = gray.maximum(img.pipe(Pipeline().grayscale().blur(1.0))).sink(
        "numpy", return_expr=False
    )
    labels = _labels(graph)
    assert not any("None" in label for label in labels.values())
    (operand,) = (
        n["ops"][0]["other"]
        for n in graph._to_dict()["nodes"].values()
        if n.get("ops") and n["ops"][0]["op"] == "maximum"
    )
    assert labels[operand].splitlines()[0] == operand
    assert any(f"maximum(other={operand})" in label for label in labels.values())


def test_the_dot_source_carries_no_python_reprs() -> None:
    """Only display attributes reach the DOT text, not the node payloads."""
    small = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale())
    dot = small.sink("numpy", return_expr=False).show_graph().source
    assert "payload" not in dot
    assert "ComputeNode" not in dot
