"""``show_graph()`` lays out the plugin's description of the graph.

What each box says is held to ``explain()`` by
``test_explain.py::TestShowGraph`` (one description, two renderings); this
pins the layout: a box per node led by the id an operand names it by, an edge
per read (dashed for an op's operand), every output an edge into one ``SINK``
box, and nothing in the DOT text but display attributes.
"""

from __future__ import annotations

import re

import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

pytestmark = plugin_required

pytest.importorskip("graphviz")


def _boxes(source: str) -> dict[str, list[str]]:
    """Each box's lines, by node name."""
    return {
        name: label.replace('\\"', '"').split(r"\l")[:-1]
        for name, label in re.findall(
            r'^\t(\w+) \[label="((?:[^"\\]|\\.)*)"', source, re.M
        )
    }


def _edges(source: str) -> list[tuple[str, str, str]]:
    """``(from, to, attributes)`` for each edge."""
    return re.findall(r"^\t(\w+) -> (\w+)(?: \[(.*)\])?$", source, re.M)


def test_each_output_is_an_edge_into_the_sink_box() -> None:
    img = pl.col("img").cv.pipe(Pipeline().source("image_bytes", dtype="u8"))
    small = img.pipe(Pipeline().resize(height=8, width=8).cast("f32")).alias("small")
    source = (
        small.merge_pipe(img.alias("full"))
        .show_graph({"small": "tensor", "full": "png"})
        .source
    )
    boxes = _boxes(source)
    sink = boxes["SINK"]
    assert sink[0] == "SINK"
    assert any(line.startswith("small: ") for line in sink)
    assert any(
        line.startswith("full: ") and line.endswith(":: binary") for line in sink
    )
    into_sink = {(a, attrs) for a, b, attrs in _edges(source) if b == "SINK"}
    assert {attrs for _, attrs in into_sink} == {"label=small", "label=full"}
    assert any("f32" in line for lines in boxes.values() for line in lines)


def test_a_root_box_holds_its_decode_and_its_ops() -> None:
    gray = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale())
    boxes = _boxes(gray.show_graph("tensor").source)
    (root,) = (lines for name, lines in boxes.items() if name != "SINK")
    assert root[0].startswith('n0: decode col("img") as image_bytes(')
    assert any("grayscale(" in line for line in root[1:])


def test_an_operand_is_a_dashed_edge_from_the_box_it_names() -> None:
    img = pl.col("img").cv.pipe(Pipeline().source("image_bytes", dtype="u8"))
    gray = img.pipe(Pipeline().grayscale())
    both = gray.maximum(img.pipe(Pipeline().grayscale().blur(1.0)))
    source = both.show_graph("tensor").source
    boxes = _boxes(source)
    (reader,) = (
        n for n, lines in boxes.items() if any("maximum(" in ln for ln in lines)
    )
    (operand,) = re.findall(r"maximum\(other='(n\d+)'\)", "\n".join(boxes[reader]))
    assert boxes[operand][0].startswith(f"{operand}: ")
    assert (operand, reader, "label=maximum style=dashed") in _edges(source)


def test_the_dot_source_carries_no_python_reprs() -> None:
    small = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale())
    dot = small.show_graph("tensor").source
    assert "payload" not in dot
    assert "None" not in dot
