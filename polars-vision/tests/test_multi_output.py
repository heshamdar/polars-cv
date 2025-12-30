"""
Tests for multi-output pipeline support.

This module tests the alias functionality and multi-output sink mode
for both Pipeline (eager) and LazyPipelineExpr (lazy) modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_vision import Pipeline
from polars_vision.lazy import LazyPipelineExpr

if TYPE_CHECKING:
    pass


class TestPipelineAlias:
    """Tests for Pipeline.alias() method."""

    def test_alias_basic(self) -> None:
        """Test basic alias creation."""
        pipe = Pipeline().source("image_bytes").alias("original")

        assert "original" in pipe.get_aliases()
        assert pipe.get_aliases()["original"] == -1  # After source, before ops

    def test_alias_after_operation(self) -> None:
        """Test alias after an operation."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=100, width=200)
            .alias("resized")
        )

        aliases = pipe.get_aliases()
        assert "resized" in aliases
        assert aliases["resized"] == 0  # After first operation (index 0)

    def test_multiple_aliases(self) -> None:
        """Test multiple aliases in one pipeline."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .alias("original")
            .resize(height=100, width=200)
            .alias("resized")
            .grayscale()
            .alias("gray")
        )

        aliases = pipe.get_aliases()
        assert len(aliases) == 3
        assert "original" in aliases
        assert "resized" in aliases
        assert "gray" in aliases
        # Check ordering
        assert aliases["original"] < aliases["resized"] < aliases["gray"]

    def test_alias_duplicate_raises(self) -> None:
        """Test that duplicate alias names raise an error."""
        pipe = Pipeline().source("image_bytes").alias("test")

        with pytest.raises(ValueError, match="already defined"):
            pipe.resize(height=100, width=200).alias("test")

    def test_alias_preserved_through_clone(self) -> None:
        """Test that aliases are preserved when cloning pipeline."""
        pipe1 = Pipeline().source("image_bytes").alias("original")
        pipe2 = pipe1.resize(height=100, width=200)

        assert "original" in pipe1.get_aliases()
        assert "original" in pipe2.get_aliases()


class TestPipelineMultiSink:
    """Tests for Pipeline.sink() with multi-output dict."""

    def test_single_sink_backward_compatible(self) -> None:
        """Test that single format string still works."""
        pipe = Pipeline().source("image_bytes").sink("numpy")

        assert not pipe.is_multi_output()
        assert pipe._sink is not None

    def test_multi_sink_with_aliases(self) -> None:
        """Test multi-output sink with aliased pipeline."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .alias("original")
            .resize(height=100, width=200)
            .alias("resized")
        )

        pipe_with_sink = pipe.sink({"original": "png", "resized": "numpy"})

        assert pipe_with_sink.is_multi_output()
        multi_sink = pipe_with_sink.get_multi_sink()
        assert multi_sink is not None
        assert "original" in multi_sink.outputs
        assert "resized" in multi_sink.outputs

    def test_multi_sink_undefined_alias_raises(self) -> None:
        """Test that undefined alias in sink raises error."""
        pipe = Pipeline().source("image_bytes").alias("defined")

        with pytest.raises(ValueError, match="not found"):
            pipe.sink({"undefined": "numpy"})

    def test_multi_sink_validates_formats(self) -> None:
        """Test that invalid formats are rejected."""
        pipe = Pipeline().source("image_bytes").alias("test")

        with pytest.raises(ValueError, match="Invalid format"):
            pipe.sink({"test": "invalid_format"})


class TestLazyPipelineExprAlias:
    """Tests for LazyPipelineExpr.alias() method."""

    def test_alias_creates_named_node(self) -> None:
        """Test that alias creates a named node."""
        pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
        expr = LazyPipelineExpr(
            column=pl.col("image"),
            pipeline=pipe,
        ).alias("processed")

        assert expr.alias_name == "processed"

    def test_alias_preserves_node_id(self) -> None:
        """Test that alias preserves the node ID."""
        pipe = Pipeline().source("image_bytes")
        expr1 = LazyPipelineExpr(column=pl.col("image"), pipeline=pipe)
        expr2 = expr1.alias("named")

        assert expr1.node_id == expr2.node_id
        assert expr2.alias_name == "named"

    def test_repr_includes_alias(self) -> None:
        """Test that repr includes alias information."""
        pipe = Pipeline().source("image_bytes")
        expr = LazyPipelineExpr(column=pl.col("image"), pipeline=pipe).alias("test")

        repr_str = repr(expr)
        assert "alias='test'" in repr_str


class TestPipelineGraphMultiOutput:
    """Tests for PipelineGraph with multi-output mode."""

    def test_graph_tracks_aliases(self) -> None:
        """Test that graph correctly tracks node aliases."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")
        graph = PipelineGraph()

        graph.add_node(
            node_id="node1",
            pipeline=pipe,
            column=pl.col("image"),
            alias="original",
        )

        assert "original" in graph._alias_to_node
        assert graph._alias_to_node["original"] == "node1"

    def test_graph_set_multi_output(self) -> None:
        """Test setting multiple outputs on graph."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")
        graph = PipelineGraph()

        graph.add_node("node1", pipe, pl.col("image"), alias="img1")
        graph.add_node("node2", pipe, pl.col("image"), alias="img2")

        graph.set_multi_output({"img1": "numpy", "img2": "png"})

        assert graph.is_multi_output()
        assert "img1" in graph._multi_output.outputs
        assert "img2" in graph._multi_output.outputs

    def test_graph_multi_output_undefined_alias_raises(self) -> None:
        """Test that undefined alias raises error."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")
        graph = PipelineGraph()
        graph.add_node("node1", pipe, pl.col("image"), alias="defined")

        with pytest.raises(ValueError, match="not found"):
            graph.set_multi_output({"undefined": "numpy"})

    def test_graph_topological_order_multi_output(self) -> None:
        """Test topological order includes all output nodes."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")
        graph = PipelineGraph()

        graph.add_node("a", pipe, pl.col("image"), alias="out_a")
        graph.add_node("b", pipe, pl.col("image"), upstream=["a"], alias="out_b")
        graph.add_node("c", pipe, pl.col("image"), upstream=["b"])

        graph.set_multi_output({"out_a": "numpy", "out_b": "png"})

        order = graph.topological_order()
        assert "a" in order
        assert "b" in order
        # c is not reachable from outputs so may not be included

    def test_graph_get_output_nodes(self) -> None:
        """Test getting output node IDs."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")
        graph = PipelineGraph()

        graph.add_node("node1", pipe, pl.col("image"), alias="out1")
        graph.add_node("node2", pipe, pl.col("image"), alias="out2")

        graph.set_multi_output({"out1": "numpy", "out2": "png"})

        output_nodes = graph.get_output_nodes()
        assert "node1" in output_nodes
        assert "node2" in output_nodes


class TestMultiSinkSerialization:
    """Tests for multi-output serialization."""

    def test_pipeline_to_json_single_sink(self) -> None:
        """Test JSON serialization with single sink."""
        import json

        pipe = Pipeline().source("image_bytes").sink("numpy")
        json_str = pipe._to_json()
        data = json.loads(json_str)

        assert "sink" in data
        assert "multi_sink" not in data

    def test_pipeline_to_json_multi_sink(self) -> None:
        """Test JSON serialization with multi-sink."""
        import json

        pipe = (
            Pipeline()
            .source("image_bytes")
            .alias("original")
            .resize(height=100, width=200)
            .alias("resized")
        )
        pipe = pipe.sink({"original": "png", "resized": "numpy"})

        json_str = pipe._to_json()
        data = json.loads(json_str)

        assert "multi_sink" in data
        assert "sink" not in data
        assert "aliases" in data

    def test_graph_to_json_single_output(self) -> None:
        """Test graph JSON serialization with single output.

        The unified format always uses "outputs" dict, with "_output"
        as the key for single-output graphs.
        """
        import json

        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")
        graph = PipelineGraph()
        graph.add_node("node1", pipe, pl.col("image"))
        graph.set_output("node1", "numpy")

        json_str = graph._to_json()
        data = json.loads(json_str)

        # Unified format uses "outputs" for both single and multi-output
        assert "outputs" in data
        assert "_output" in data["outputs"]
        assert data["outputs"]["_output"]["node"] == "node1"

    def test_graph_to_json_multi_output(self) -> None:
        """Test graph JSON serialization with multi-output.

        Multi-output graphs have named outputs in the "outputs" dict.
        """
        import json

        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")
        graph = PipelineGraph()
        graph.add_node("node1", pipe, pl.col("image"), alias="out1")
        graph.add_node("node2", pipe, pl.col("image"), alias="out2")
        graph.set_multi_output({"out1": "numpy", "out2": "png"})

        json_str = graph._to_json()
        data = json.loads(json_str)

        assert "outputs" in data
        assert "out1" in data["outputs"]
        assert "out2" in data["outputs"]
        assert data["outputs"]["out1"]["node"] == "node1"
        assert data["outputs"]["out2"]["node"] == "node2"
