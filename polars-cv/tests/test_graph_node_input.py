"""A graph node's input is declared once (CR-117).

A node either decodes a column through a source or reads another node; the
nodes it depends on are that input plus the nodes its ops read, plus, for a
``merge_pipe`` node, the branches it keeps reachable. The wire (version 2)
says this as ``{"input": {"decode": {"column", "source"}} | {"node": id},
"ops": [...]}``: no ``column_bindings`` map, no positional ``upstream`` list,
no source on a node that reads another.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import make_test_png, plugin_required

pytestmark = plugin_required


def _graph(expr):  # noqa: ANN001, ANN202
    return expr.sink("numpy", return_expr=False)


def test_a_wire_node_is_its_input_and_its_ops() -> None:
    img = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale())
    blurred = img.pipe(Pipeline().blur(3))
    both = img.add(blurred)
    spec = _graph(both)._to_dict()
    assert spec["version"] == 2
    assert "column_bindings" not in spec
    for node_id, node in spec["nodes"].items():
        assert set(node) == {"input", "ops"}, (node_id, node)
    inputs = [node["input"] for node in spec["nodes"].values()]
    roots = [i["decode"] for i in inputs if "decode" in i]
    assert [(r["column"], r["source"]["format"]) for r in roots] == [(0, "image_bytes")]
    reads = [i["node"] for i in inputs if "node" in i]
    assert set(reads) <= set(spec["nodes"]), reads


def test_an_operand_is_a_dependency_once() -> None:
    src = Pipeline().source("image_bytes")
    a = pl.col("a").cv.pipe(src)
    b = pl.col("b").cv.pipe(src)
    total = a.add(b)
    deps = total._dependencies()
    assert [d is a for d in deps].count(True) == 1
    assert [d is b for d in deps].count(True) == 1
    assert len(deps) == 2
    # The root reads its column and nothing else.
    assert a._dependencies() == []


def test_merge_pipe_keeps_unread_branches_reachable() -> None:
    gray = pl.col("img").cv.pipe(Pipeline().source("image_bytes").grayscale())
    small = gray.pipe(Pipeline().resize(height=4, width=4)).alias("small")
    flat = gray.pipe(Pipeline().blur(3)).alias("flat")
    merged = small.merge_pipe(flat)
    assert any(d is flat for d in merged._dependencies())
    df = pl.DataFrame({"img": [make_test_png(8, 8)]})
    out = df.select(merged.sink({"small": "numpy", "flat": "numpy"}))
    assert out.columns == ["img"]
    assert set(out["img"].struct.fields) == {"small", "flat"}


def test_the_plugin_refuses_a_version_1_graph() -> None:
    from polars_cv._lib import check_graph

    v1 = {
        "version": 1,
        "nodes": {"n0": {"source": {"format": "image_bytes"}, "ops": []}},
        "outputs": {"_output": {"node": "n0", "sink": {"format": "numpy"}}},
        "column_bindings": {"n0": 0},
    }
    with pytest.raises(ValueError, match="version 1.*rebuild"):
        check_graph(json.dumps(v1))


def test_a_node_written_from_a_pipeline_keeps_source_and_input_consistent() -> None:
    """A node's wire form is written by Rust (``Plan.to_node``, through
    ``Pipeline._to_node_dict``): a pipeline without a source cannot be written
    as a root, nor one with a source as a node reading another."""
    root = Pipeline().source("image_bytes").grayscale()
    follow = Pipeline().blur(1.0)

    def no_slots(expr: pl.Expr) -> int:
        raise AssertionError(expr)

    assert root._to_node_dict(no_slots, {}, 0)["input"]["decode"]["column"] == 0
    assert follow._to_node_dict(no_slots, {}, "n0")["input"] == {"node": "n0"}
    with pytest.raises(ValueError, match="has no source"):
        follow._to_node_dict(no_slots, {}, 0)
    with pytest.raises(ValueError, match="has a source"):
        root._to_node_dict(no_slots, {}, "n0")


def test_a_node_depends_on_what_its_ops_read_not_on_stale_references() -> None:
    """Dependencies come from the ops (Rust's ``operands()``, through
    ``Plan.operands``), as Rust derives them: a reference the pipeline still
    holds after the op that read it is gone is not a dependency."""
    from polars_cv._graph import PipelineGraph

    img = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
    other = pl.col("other").cv.pipe(Pipeline().source("image_bytes"))
    pipe = Pipeline().blur(1.0)
    pipe._read_node(other)  # held, as CSE's rebuilt nodes hold them, but unread
    graph = PipelineGraph()
    graph.add_node(img.node_id, img.pipeline, img.column)
    graph.add_node(other.node_id, other.pipeline, other.column)
    graph.add_node("n", pipe, input=img.node_id)
    assert graph._dependencies("n") == [img.node_id]
