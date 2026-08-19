"""
Tests filling gaps in lazy composition coverage.

Covers: blend/ratio composition structure, merge_pipe with 3+ branches,
statistics with custom include lists, sink(return_expr=False), and
bitwise operation composition.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl

from polars_cv import Pipeline, numpy_from_struct
from polars_cv.lazy import LazyPipelineExpr
from tests.conftest import plugin_required

# ---------------------------------------------------------------------------
# Composition structure tests (no plugin needed)
# ---------------------------------------------------------------------------


class TestBlendRatioComposition:
    """Verify blend() and ratio() create proper LazyPipelineExpr."""

    def test_blend_returns_lazy_expr(self) -> None:
        """blend() should return a LazyPipelineExpr."""
        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)
        result = expr1.blend(expr2)
        assert isinstance(result, LazyPipelineExpr)

    def test_ratio_returns_lazy_expr(self) -> None:
        """ratio() should return a LazyPipelineExpr."""
        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)
        result = expr1.ratio(expr2)
        assert isinstance(result, LazyPipelineExpr)


class TestBitwiseComposition:
    """Verify bitwise operations create proper composition structure."""

    def test_bitwise_and_composition(self) -> None:
        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)
        result = expr1.bitwise_and(expr2)
        assert isinstance(result, LazyPipelineExpr)

    def test_bitwise_or_composition(self) -> None:
        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)
        result = expr1.bitwise_or(expr2)
        assert isinstance(result, LazyPipelineExpr)

    def test_bitwise_xor_composition(self) -> None:
        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)
        result = expr1.bitwise_xor(expr2)
        assert isinstance(result, LazyPipelineExpr)


class TestMaximumMinimumComposition:
    """Verify maximum/minimum create proper composition structure."""

    def test_maximum_composition(self) -> None:
        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)
        result = expr1.maximum(expr2)
        assert isinstance(result, LazyPipelineExpr)

    def test_minimum_composition(self) -> None:
        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)
        result = expr1.minimum(expr2)
        assert isinstance(result, LazyPipelineExpr)


# ---------------------------------------------------------------------------
# merge_pipe with 3+ branches
# ---------------------------------------------------------------------------


class TestMergePipeMultipleBranches:
    """Test merge_pipe with more than 2 branches."""

    def test_merge_three_branches(self) -> None:
        """merge_pipe should accept 3 upstream branches."""
        pipe = Pipeline().source("image_bytes")
        expr_a = pl.col("a").cv.pipe(pipe)
        expr_b = pl.col("b").cv.pipe(pipe)
        expr_c = pl.col("c").cv.pipe(pipe)

        merged = expr_a.merge_pipe(expr_b, expr_c)
        assert isinstance(merged, LazyPipelineExpr)

    def test_merge_preserves_all_dependencies(self) -> None:
        """All merged branch dependencies should appear in dependency graph."""
        pipe = Pipeline().source("image_bytes")
        expr_a = pl.col("a").cv.pipe(pipe)
        expr_b = pl.col("b").cv.pipe(pipe)
        expr_c = pl.col("c").cv.pipe(pipe)

        merged = expr_a.merge_pipe(expr_b, expr_c)
        graph = merged._collect_dependency_graph()
        # Should have at least 4 nodes (3 sources + 1 merge)
        assert len(graph) >= 4


# ---------------------------------------------------------------------------
# statistics with custom include lists
# ---------------------------------------------------------------------------


class TestStatisticsVariants:
    """Test statistics() and statistics_lazy() with various include combinations."""

    def test_statistics_returns_pl_expr(self) -> None:
        """statistics() should return a pl.Expr (finalized)."""
        pipe = Pipeline().source("image_bytes")
        expr = pl.col("img").cv.pipe(pipe)
        result = expr.statistics()
        assert isinstance(result, pl.Expr)

    def test_statistics_single_stat(self) -> None:
        """statistics(include=['mean']) should return pl.Expr."""
        pipe = Pipeline().source("image_bytes")
        expr = pl.col("img").cv.pipe(pipe)
        result = expr.statistics(include=["mean"])
        assert isinstance(result, pl.Expr)

    def test_statistics_max_min(self) -> None:
        """statistics(include=['max', 'min']) should return pl.Expr."""
        pipe = Pipeline().source("image_bytes")
        expr = pl.col("img").cv.pipe(pipe)
        result = expr.statistics(include=["max", "min"])
        assert isinstance(result, pl.Expr)

    def test_statistics_lazy_returns_lazy_expr(self) -> None:
        """statistics_lazy() should return a LazyPipelineExpr."""
        pipe = Pipeline().source("image_bytes")
        expr = pl.col("img").cv.pipe(pipe)
        result = expr.statistics_lazy()
        assert isinstance(result, LazyPipelineExpr)

    def test_statistics_lazy_with_include(self) -> None:
        """statistics_lazy(include=[...]) should return LazyPipelineExpr."""
        pipe = Pipeline().source("image_bytes")
        expr = pl.col("img").cv.pipe(pipe)
        for stat in ["mean", "std", "min", "max", "sum"]:
            result = expr.statistics_lazy(include=[stat])
            assert isinstance(result, LazyPipelineExpr)


# ---------------------------------------------------------------------------
# sink(return_expr=False)
# ---------------------------------------------------------------------------


class TestSinkReturnExpr:
    """Test sink() with return_expr=False."""

    def test_sink_return_expr_false_returns_graph(self) -> None:
        """sink(return_expr=False) should return a PipelineGraph, not pl.Expr."""
        from polars_cv._graph import PipelineGraph

        pipe = Pipeline().source("image_bytes").grayscale()
        expr = pl.col("img").cv.pipe(pipe)
        graph = expr.sink("numpy", return_expr=False)
        assert isinstance(graph, PipelineGraph)

    def test_sink_return_expr_true_returns_pl_expr(self) -> None:
        """sink(return_expr=True) should return a pl.Expr (default)."""
        pipe = Pipeline().source("image_bytes").grayscale()
        expr = pl.col("img").cv.pipe(pipe)
        result = expr.sink("numpy", return_expr=True)
        assert isinstance(result, pl.Expr)


# ---------------------------------------------------------------------------
# Execution tests for blend, ratio, bitwise
# ---------------------------------------------------------------------------


@plugin_required
class TestBlendRatioExecution:
    """Execute blend/ratio and verify output is valid."""

    def test_blend_execution_produces_valid_output(self, encode_png: Callable) -> None:
        """blend should execute without error and produce correct shape."""
        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        df = pl.DataFrame(
            {
                "a": [encode_png(img1)],
                "b": [encode_png(img2)],
            }
        )

        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)

        result = df.select(out=expr1.blend(expr2).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        assert arr.shape == (30, 30, 3)
        assert arr.dtype == np.uint8

    def test_ratio_execution_produces_valid_output(self, encode_png: Callable) -> None:
        """ratio should execute without error and produce correct shape."""
        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (30, 30, 3), dtype=np.uint8)
        df = pl.DataFrame(
            {
                "a": [encode_png(img1)],
                "b": [encode_png(img2)],
            }
        )

        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)

        result = df.select(out=expr1.ratio(expr2).sink("numpy"))
        arr = numpy_from_struct(result.row(0)[0])
        # ratio is true division: u8 operands promote to f32.
        assert arr.shape == (30, 30, 3)
        assert arr.dtype == np.float32


@plugin_required
class TestBitwiseExecution:
    """Execute bitwise operations end-to-end."""

    def test_bitwise_and_execution(self, encode_png: Callable) -> None:
        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        df = pl.DataFrame(
            {
                "a": [encode_png(img1)],
                "b": [encode_png(img2)],
            }
        )

        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)

        result = df.select(out=expr1.bitwise_and(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        expected = np.bitwise_and(img1, img2)
        np.testing.assert_array_equal(actual, expected)

    def test_bitwise_or_execution(self, encode_png: Callable) -> None:
        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        img2 = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        df = pl.DataFrame(
            {
                "a": [encode_png(img1)],
                "b": [encode_png(img2)],
            }
        )

        pipe = Pipeline().source("image_bytes")
        expr1 = pl.col("a").cv.pipe(pipe)
        expr2 = pl.col("b").cv.pipe(pipe)

        result = df.select(out=expr1.bitwise_or(expr2).sink("numpy"))
        actual = numpy_from_struct(result.row(0)[0])
        expected = np.bitwise_or(img1, img2)
        np.testing.assert_array_equal(actual, expected)
