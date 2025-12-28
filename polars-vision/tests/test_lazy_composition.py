"""
Tests for lazy pipeline composition.

These tests verify the LazyPipelineExpr class and graph-based pipeline fusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_vision import LazyPipelineExpr, Pipeline

if TYPE_CHECKING:
    pass


class TestLazyPipelineExpr:
    """Tests for the LazyPipelineExpr class."""

    def test_pipe_returns_lazy_expr(self) -> None:
        """Verify .cv.pipe() returns LazyPipelineExpr, not pl.Expr."""
        pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
        result = pl.col("image").cv.pipe(pipe)

        assert isinstance(result, LazyPipelineExpr)
        assert not isinstance(result, pl.Expr)

    def test_lazy_expr_has_node_id(self) -> None:
        """LazyPipelineExpr should have a unique node ID."""
        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("image").cv.pipe(pipe)
        expr2 = pl.col("image").cv.pipe(pipe)

        assert expr1.node_id != expr2.node_id

    def test_lazy_expr_repr(self) -> None:
        """LazyPipelineExpr repr should include guidance."""
        pipe = Pipeline().source("image_bytes")
        expr = pl.col("image").cv.pipe(pipe)

        repr_str = repr(expr)
        assert "LazyPipelineExpr" in repr_str
        assert "sink" in repr_str.lower()

    def test_lazy_expr_preserves_column(self) -> None:
        """LazyPipelineExpr should preserve the column reference."""
        pipe = Pipeline().source("image_bytes")
        col = pl.col("my_column")
        expr = col.cv.pipe(pipe)

        assert expr.column is col

    def test_lazy_expr_preserves_pipeline(self) -> None:
        """LazyPipelineExpr should preserve the pipeline."""
        pipe = Pipeline().source("image_bytes").resize(height=100, width=200)
        expr = pl.col("image").cv.pipe(pipe)

        assert expr.pipeline is pipe


class TestLazyComposition:
    """Tests for composing LazyPipelineExpr instances."""

    def test_apply_mask_creates_new_lazy_expr(self) -> None:
        """apply_mask should return a new LazyPipelineExpr."""
        img_pipe = Pipeline().source("image_bytes")
        mask_pipe = Pipeline().source("image_bytes")

        img = pl.col("image").cv.pipe(img_pipe)
        mask = pl.col("mask").cv.pipe(mask_pipe)

        result = img.apply_mask(mask)

        assert isinstance(result, LazyPipelineExpr)
        assert result.node_id != img.node_id
        assert result.node_id != mask.node_id

    def test_apply_mask_tracks_upstream(self) -> None:
        """apply_mask should track both inputs as upstream."""
        img_pipe = Pipeline().source("image_bytes")
        mask_pipe = Pipeline().source("image_bytes")

        img = pl.col("image").cv.pipe(img_pipe)
        mask = pl.col("mask").cv.pipe(mask_pipe)

        result = img.apply_mask(mask)

        assert len(result._upstream) == 2
        assert img in result._upstream
        assert mask in result._upstream

    def test_add_composition(self) -> None:
        """add should compose two LazyPipelineExpr instances."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = expr1.add(expr2)

        assert isinstance(result, LazyPipelineExpr)
        assert len(result._upstream) == 2

    def test_subtract_composition(self) -> None:
        """subtract should compose two LazyPipelineExpr instances."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = expr1.subtract(expr2)

        assert isinstance(result, LazyPipelineExpr)
        assert len(result._upstream) == 2

    def test_multiply_composition(self) -> None:
        """multiply should compose two LazyPipelineExpr instances."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = expr1.multiply(expr2)

        assert isinstance(result, LazyPipelineExpr)

    def test_divide_composition(self) -> None:
        """divide should compose two LazyPipelineExpr instances."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = expr1.divide(expr2)

        assert isinstance(result, LazyPipelineExpr)

    def test_chained_composition(self) -> None:
        """Multiple operations can be chained."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")
        pipe3 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)
        expr3 = pl.col("img3").cv.pipe(pipe3)

        # Chain: (expr1 + expr2) * expr3
        result = expr1.add(expr2).multiply(expr3)

        assert isinstance(result, LazyPipelineExpr)
        # Should have expr1+expr2 intermediate and expr3 as upstream
        assert len(result._upstream) == 2

    def test_apply_contour_mask(self) -> None:
        """apply_contour_mask creates rasterize node automatically."""
        img_pipe = Pipeline().source("image_bytes")
        contour_pipe = Pipeline().source("contour")

        img = pl.col("image").cv.pipe(img_pipe)
        contour = pl.col("contour").cv.pipe(contour_pipe)

        result = img.apply_contour_mask(contour)

        assert isinstance(result, LazyPipelineExpr)


class TestCycleDetection:
    """Tests for circular dependency detection."""

    def test_no_cycle_simple(self) -> None:
        """Simple composition should not raise cycle error."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result = expr1.add(expr2)

        # Should not raise
        result._validate_no_cycles()

    def test_no_cycle_complex(self) -> None:
        """Complex but acyclic composition should not raise."""
        pipe = Pipeline().source("image_bytes")

        a = pl.col("a").cv.pipe(pipe)
        b = pl.col("b").cv.pipe(pipe)
        c = a.add(b)
        d = c.multiply(a)  # Reuses 'a', but not a cycle

        # Should not raise
        d._validate_no_cycles()


class TestDependencyGraph:
    """Tests for dependency graph collection."""

    def test_collect_single_node(self) -> None:
        """Single node should return just itself."""
        pipe = Pipeline().source("image_bytes")
        expr = pl.col("image").cv.pipe(pipe)

        graph = expr._collect_dependency_graph()

        assert len(graph) == 1
        assert graph[0] is expr

    def test_collect_two_nodes(self) -> None:
        """Two composed nodes should return both in order."""
        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)
        result = expr1.add(expr2)

        graph = result._collect_dependency_graph()

        # Should have expr1, expr2, and result (in some valid topological order)
        assert len(graph) == 3
        # Result should be last (it depends on the others)
        assert graph[-1] is result

    def test_collect_deduplicates(self) -> None:
        """Shared dependencies should only appear once."""
        pipe = Pipeline().source("image_bytes")

        a = pl.col("a").cv.pipe(pipe)
        b = a.add(a)  # 'a' used twice, but should appear once

        graph = b._collect_dependency_graph()

        # Should have: a, b
        assert len(graph) == 2


class TestPipelineGraphSerialization:
    """Tests for pipeline graph serialization."""

    def test_graph_to_json_single_node(self) -> None:
        """Single node graph can be serialized."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")

        graph = PipelineGraph()
        graph.add_node("node1", pipe, pl.col("image"))
        graph.set_output("node1", "numpy")

        json_str = graph._to_json()

        import json

        parsed = json.loads(json_str)

        assert "nodes" in parsed
        assert "node1" in parsed["nodes"]
        assert parsed["output"]["node"] == "node1"
        assert parsed["output"]["sink"]["format"] == "numpy"

    def test_graph_topological_order(self) -> None:
        """Graph should compute correct topological order."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")

        graph = PipelineGraph()
        graph.add_node("a", pipe, pl.col("a"))
        graph.add_node("b", pipe, pl.col("b"))
        graph.add_node("c", pipe, pl.col("c"), upstream=["a", "b"])
        graph.set_output("c", "numpy")

        order = graph.topological_order()

        # 'c' must come after 'a' and 'b'
        assert order.index("c") > order.index("a")
        assert order.index("c") > order.index("b")

    def test_graph_column_bindings(self) -> None:
        """Graph should correctly bind columns."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")

        graph = PipelineGraph()
        graph.add_node("node1", pipe, pl.col("col_a"))
        graph.add_node("node2", pipe, pl.col("col_b"))
        graph.set_output("node2", "numpy")

        # Build bindings
        graph._build_column_bindings()

        assert graph._column_bindings["node1"] == 0
        assert graph._column_bindings["node2"] == 1

    def test_graph_deduplicates_same_column(self) -> None:
        """Same column used by multiple nodes should be deduplicated."""
        from polars_vision._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")

        graph = PipelineGraph()
        graph.add_node("node1", pipe, pl.col("same_col"))
        graph.add_node("node2", pipe, pl.col("same_col"))
        graph.set_output("node2", "numpy")

        graph._build_column_bindings()
        columns = graph._get_ordered_columns()

        # Only one unique column
        assert len(columns) == 1
        # Both nodes should point to same index
        assert graph._column_bindings["node1"] == graph._column_bindings["node2"]
