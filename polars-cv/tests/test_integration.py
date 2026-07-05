"""
End-to-end integration tests for polars-cv.

These tests verify the full pipeline from Python to Rust and back,
using synthetic image data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import make_test_png as create_test_png
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


class TestPipelineBuilderIntegration:
    """Test Pipeline builder creates valid specifications."""

    def test_simple_pipeline_to_json(self) -> None:
        """Test simple pipeline serializes to valid JSON."""
        pipe = Pipeline().source("image_bytes").resize(height=224, width=224)

        json_str = pipe._to_json()
        assert '"source"' in json_str
        assert '"ops"' in json_str
        assert '"sink"' not in json_str

    def test_complex_pipeline_to_json(self) -> None:
        """Test complex pipeline serializes correctly."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(channels=3)
            .resize(height=256, width=256)
            .crop(top=16, left=16, height=224, width=224)
            .flip_h()
            .grayscale()
            .cast("f32")
            .scale(1.0 / 255.0)
            .normalize(method="minmax")
        )

        json_str = pipe._to_json()
        # Verify it's valid JSON by loading it
        import json

        data = json.loads(json_str)
        assert len(data["ops"]) == 7

    def test_dynamic_pipeline_to_json(self) -> None:
        """Test pipeline with expressions serializes correctly."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("w"))
        )

        json_str = pipe._to_json()
        import json

        data = json.loads(json_str)
        assert data["ops"][0]["height"]["type"] == "expr"
        assert data["ops"][0]["width"]["type"] == "expr"


class TestPolarsNamespace:
    """Test the cv namespace on Polars expressions."""

    def test_cv_namespace_exists(self) -> None:
        """Test that cv namespace is registered."""
        # Import should register the namespace
        import polars_cv.expressions  # noqa: F401

        expr = pl.col("images")
        assert hasattr(expr, "cv")

    def test_cv_pipeline_method_removed(self) -> None:
        """Test that legacy pipeline method is removed from namespace."""
        import polars_cv.expressions  # noqa: F401

        expr = pl.col("images")
        assert not hasattr(expr.cv, "pipeline")
        assert hasattr(expr.cv, "pipe")


# Check if plugin is available by checking if the .so file exists
# Mark tests with plugin_required marker for easy filtering
@plugin_required
class TestPluginExecution:
    """Tests that require the compiled Rust plugin."""

    def test_simple_pipeline_execution(self) -> None:
        """Test basic pipeline execution."""
        pipe = Pipeline().source("image_bytes").resize(height=10, width=10)

        png_bytes = create_test_png(10, 10)
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipe(pipe).sink("blob"))
        assert "processed" in result.columns
        assert result["processed"].dtype == pl.Binary

    def test_pipeline_with_expression_args(self) -> None:
        """Test pipeline with expression arguments."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("w"))
        )

        png_bytes = create_test_png(10, 10)
        df = pl.DataFrame({"images": [png_bytes, png_bytes], "h": [5, 8], "w": [5, 8]})

        result = df.with_columns(processed=pl.col("images").cv.pipe(pipe).sink("blob"))
        assert "processed" in result.columns

    def test_grayscale_pipeline(self) -> None:
        """Test grayscale conversion."""
        pipe = Pipeline().source("image_bytes").grayscale()

        png_bytes = create_test_png(5, 5, (100, 150, 200))
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipe(pipe).sink("blob"))
        assert result["processed"].dtype == pl.Binary

    def test_multiple_operations(self) -> None:
        """Test pipeline with multiple operations."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=20, width=20)
            .flip_v()
            .grayscale()
            .threshold(128)
        )

        png_bytes = create_test_png(10, 10)
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipe(pipe).sink("blob"))
        assert result["processed"].dtype == pl.Binary


class TestPipelineValidation:
    """Test pipeline validation."""

    def test_incomplete_pipeline_raises(self) -> None:
        """Test source-only pipeline is considered valid."""
        pipe = Pipeline().source()
        pipe.validate()  # Should not raise

    def test_pipeline_without_source_raises(self) -> None:
        """Test that pipeline without source raises on to_json."""
        pipe = Pipeline().resize(height=100, width=100)

        with pytest.raises(ValueError, match="must have a source"):
            pipe._to_json()

    def test_valid_pipeline_passes_validation(self) -> None:
        """Test that valid pipeline passes validation."""
        pipe = Pipeline().source().resize(height=100, width=100)

        pipe.validate()  # Should not raise


class TestExpressionTracking:
    """Test expression column tracking."""

    def test_no_expressions(self) -> None:
        """Test pipeline with no expressions has empty expr list."""
        pipe = Pipeline().source().resize(height=100, width=100)

        assert len(pipe._get_expr_columns()) == 0

    def test_single_expression(self) -> None:
        """Test single expression is tracked."""
        pipe = Pipeline().source().resize(height=pl.col("h"), width=100)

        exprs = pipe._get_expr_columns()
        assert len(exprs) == 1

    def test_multiple_expressions(self) -> None:
        """Test multiple expressions are tracked."""
        pipe = (
            Pipeline()
            .source()
            .resize(height=pl.col("h"), width=pl.col("w"))
            .crop(top=pl.col("t"), left=pl.col("l"))
        )

        exprs = pipe._get_expr_columns()
        assert len(exprs) == 4

    def test_duplicate_expressions_not_tracked_twice(self) -> None:
        """Test same expression object is not duplicated."""
        h_expr = pl.col("h")
        pipe = Pipeline().source().resize(height=h_expr, width=h_expr)

        exprs = pipe._get_expr_columns()
        # Same expression object used twice should only be tracked once
        assert len(exprs) == 1
