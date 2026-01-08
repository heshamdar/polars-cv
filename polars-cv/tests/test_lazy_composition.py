"""
Tests for lazy pipeline composition.

These tests verify the LazyPipelineExpr class and graph-based pipeline fusion.
Tests in TestLazyCompositionExecution MUST actually execute the plugin to verify
that binary operations are correctly implemented in the Rust backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl

from polars_cv import LazyPipelineExpr, Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


# Import plugin_required marker from conftest
from tests.conftest import plugin_required


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
        # Contour source now requires dimensions for rasterization
        contour_pipe = Pipeline().source("contour", width=100, height=100)

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
        from polars_cv._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes")

        graph = PipelineGraph()
        graph.add_node("node1", pipe, pl.col("image"))
        graph.set_output("node1", "numpy")

        json_str = graph._to_json()

        import json

        parsed = json.loads(json_str)

        assert "nodes" in parsed
        assert "node1" in parsed["nodes"]
        # Unified format uses "outputs" with "_output" key for single output
        assert "outputs" in parsed
        assert "_output" in parsed["outputs"]
        assert parsed["outputs"]["_output"]["node"] == "node1"
        assert parsed["outputs"]["_output"]["sink"]["format"] == "numpy"

    def test_graph_topological_order(self) -> None:
        """Graph should compute correct topological order."""
        from polars_cv._graph import PipelineGraph

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
        from polars_cv._graph import PipelineGraph

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
        from polars_cv._graph import PipelineGraph

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


@plugin_required
class TestLazyCompositionExecution:
    """
    Tests that verify binary operations EXECUTE correctly through the plugin.

    These tests will FAIL until binary operations are implemented in the Rust backend.
    This is the expected and correct behavior - tests should fail for unimplemented features.
    """

    def test_apply_mask_execution(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """apply_mask should execute and produce correct output."""
        # Create a color image and a grayscale mask (as 3-channel for simplicity)
        img_bytes = create_test_png(100, 100, (200, 100, 50))
        # Mask: white (255) means keep, black (0) means zero out
        mask_bytes = create_test_png(100, 100, (255, 255, 255))

        df = pl.DataFrame(
            {
                "image": [img_bytes],
                "mask": [mask_bytes],
            }
        )

        img_pipe = Pipeline().source("image_bytes")
        mask_pipe = Pipeline().source("image_bytes").grayscale()

        img = pl.col("image").cv.pipe(img_pipe)
        mask = pl.col("mask").cv.pipe(mask_pipe)

        result_expr = img.apply_mask(mask).sink("numpy")

        # This will FAIL until apply_mask is implemented in execute.rs
        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        assert output.shape == (100, 100, 3)
        assert output.dtype == np.uint8

    def test_add_execution(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """add should execute and produce correct output."""
        img1_bytes = create_test_png(100, 100, (100, 100, 100))
        img2_bytes = create_test_png(100, 100, (50, 50, 50))

        df = pl.DataFrame(
            {
                "img1": [img1_bytes],
                "img2": [img2_bytes],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result_expr = expr1.add(expr2).sink("numpy")

        # This will FAIL until add is implemented in execute.rs
        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        assert output.shape == (100, 100, 3)
        assert output.dtype == np.uint8
        # Verify addition: 100 + 50 = 150
        np.testing.assert_array_equal(output[0, 0], [150, 150, 150])

    def test_subtract_execution(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """subtract should execute and produce correct output."""
        img1_bytes = create_test_png(100, 100, (150, 150, 150))
        img2_bytes = create_test_png(100, 100, (50, 50, 50))

        df = pl.DataFrame(
            {
                "img1": [img1_bytes],
                "img2": [img2_bytes],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result_expr = expr1.subtract(expr2).sink("numpy")

        # This will FAIL until subtract is implemented in execute.rs
        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        assert output.shape == (100, 100, 3)
        assert output.dtype == np.uint8
        # Verify subtraction: 150 - 50 = 100
        np.testing.assert_array_equal(output[0, 0], [100, 100, 100])

    def test_multiply_execution(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """multiply should execute and produce correct output."""
        img1_bytes = create_test_png(100, 100, (100, 100, 100))
        img2_bytes = create_test_png(100, 100, (128, 128, 128))

        df = pl.DataFrame(
            {
                "img1": [img1_bytes],
                "img2": [img2_bytes],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result_expr = expr1.multiply(expr2).sink("numpy")

        # This will FAIL until multiply is implemented in execute.rs
        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        assert output.shape == (100, 100, 3)
        assert output.dtype == np.uint8

    def test_divide_execution(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """divide should execute and produce correct output."""
        img1_bytes = create_test_png(100, 100, (200, 200, 200))
        img2_bytes = create_test_png(100, 100, (100, 100, 100))

        df = pl.DataFrame(
            {
                "img1": [img1_bytes],
                "img2": [img2_bytes],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)

        result_expr = expr1.divide(expr2).sink("numpy")

        # This will FAIL until divide is implemented in execute.rs
        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        assert output.shape == (100, 100, 3)
        assert output.dtype == np.uint8

    def test_chained_composition_execution(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """Chained operations (add then multiply) should execute correctly."""
        img1_bytes = create_test_png(100, 100, (50, 50, 50))
        img2_bytes = create_test_png(100, 100, (50, 50, 50))
        img3_bytes = create_test_png(100, 100, (128, 128, 128))

        df = pl.DataFrame(
            {
                "img1": [img1_bytes],
                "img2": [img2_bytes],
                "img3": [img3_bytes],
            }
        )

        pipe1 = Pipeline().source("image_bytes")
        pipe2 = Pipeline().source("image_bytes")
        pipe3 = Pipeline().source("image_bytes")

        expr1 = pl.col("img1").cv.pipe(pipe1)
        expr2 = pl.col("img2").cv.pipe(pipe2)
        expr3 = pl.col("img3").cv.pipe(pipe3)

        # Chain: (expr1 + expr2) * expr3
        result_expr = expr1.add(expr2).multiply(expr3).sink("numpy")

        # This will FAIL until both add and multiply are implemented
        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        assert output.shape == (100, 100, 3)
        assert output.dtype == np.uint8

    def test_apply_contour_mask_execution(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """Apply a rasterized contour as mask to an image."""
        img_bytes = create_test_png(100, 100, (200, 100, 50))

        # Create a simple square contour using CONTOUR_SCHEMA format
        contour_data = {
            "exterior": [
                {"x": 25.0, "y": 25.0},
                {"x": 25.0, "y": 75.0},
                {"x": 75.0, "y": 75.0},
                {"x": 75.0, "y": 25.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame(
            {
                "image": [img_bytes],
                "contour": [contour_data],
            }
        )

        img_pipe = Pipeline().source("image_bytes")
        # Contour source with explicit dimensions rasterizes the contour to a mask
        contour_pipe = Pipeline().source("contour", width=100, height=100)

        img = pl.col("image").cv.pipe(img_pipe)
        # The contour source already produces a rasterized mask
        mask = pl.col("contour").cv.pipe(contour_pipe)

        # Use apply_mask directly (contour source already rasterizes)
        result_expr = img.apply_mask(mask).sink("numpy")

        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        assert output.shape == (100, 100, 3)
        # Pixels outside contour should be zeroed
        assert np.all(output[0, 0] == 0)
        # Pixels inside contour should have original values
        assert np.any(output[50, 50] > 0)

    def test_contour_source_with_shape_inference(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """Contour source with shape= infers dimensions from another LazyPipelineExpr."""
        img_bytes = create_test_png(120, 80, (200, 100, 50))

        # Create a contour that spans most of the image
        contour_data = {
            "exterior": [
                {"x": 10.0, "y": 10.0},
                {"x": 10.0, "y": 70.0},
                {"x": 110.0, "y": 70.0},
                {"x": 110.0, "y": 10.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame(
            {
                "image": [img_bytes],
                "contour": [contour_data],
            }
        )

        # Define the image pipeline first
        img_pipe = Pipeline().source("image_bytes")
        img = pl.col("image").cv.pipe(img_pipe)

        # Contour source with shape= infers dimensions from the image
        contour_pipe = Pipeline().source("contour", shape=img)
        mask = pl.col("contour").cv.pipe(contour_pipe)

        # Apply the mask
        result_expr = img.apply_mask(mask).sink("numpy")

        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        # Image is 120x80 (WxH), so output should be (80, 120, 3) in HWC
        assert output.shape == (80, 120, 3)
        # Pixels outside contour should be zeroed
        assert np.all(output[0, 0] == 0)
        # Pixels inside contour (center) should have original values
        assert np.any(output[40, 60] > 0)

    def test_apply_contour_mask_convenience(
        self,
        create_test_png: Callable[[int, int, tuple[int, int, int]], bytes],
    ) -> None:
        """apply_contour_mask convenience method auto-infers dimensions."""
        img_bytes = create_test_png(100, 100, (200, 100, 50))

        # Create a simple square contour
        contour_data = {
            "exterior": [
                {"x": 25.0, "y": 25.0},
                {"x": 25.0, "y": 75.0},
                {"x": 75.0, "y": 75.0},
                {"x": 75.0, "y": 25.0},
            ],
            "holes": [],
            "is_closed": True,
        }

        df = pl.DataFrame(
            {
                "image": [img_bytes],
                "contour": [contour_data],
            }
        )

        img_pipe = Pipeline().source("image_bytes")
        # For apply_contour_mask, we don't need dimensions - they're inferred
        contour_pipe = Pipeline().source("contour", width=1, height=1)  # Dummy dims

        img = pl.col("image").cv.pipe(img_pipe)
        contour = pl.col("contour").cv.pipe(contour_pipe)

        # apply_contour_mask should auto-infer dimensions from the image
        result_expr = img.apply_contour_mask(contour).sink("numpy")

        result = df.select(output=result_expr)
        output = numpy_from_struct(result.row(0)[0])

        assert output.shape == (100, 100, 3)
        # Pixels outside contour should be zeroed
        assert np.all(output[0, 0] == 0)
        # Pixels inside contour should have original values
        assert np.any(output[50, 50] > 0)
