"""
End-to-end integration tests for polars-cv.

These tests verify the full pipeline from Python to Rust and back,
using synthetic image data.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import polars as pl
import pytest

from polars_cv import Pipeline

if TYPE_CHECKING:
    pass


def create_test_png(
    width: int = 10, height: int = 10, color: tuple = (255, 0, 0)
) -> bytes:
    """
    Create a minimal test PNG image.

    Args:
        width: Image width.
        height: Image height.
        color: RGB color tuple.

    Returns:
        PNG bytes.
    """
    # We'll use a simple approach - create raw RGB data and use PNG encoding
    # Since we don't have PIL in tests, we'll use a pre-generated minimal PNG
    # or create one programmatically

    # Minimal 1x1 red PNG for basic testing
    if width == 1 and height == 1:
        return bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,  # 1x1
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,  # 8-bit RGB
                0x90,
                0x77,
                0x53,
                0xDE,  # CRC
                0x00,
                0x00,
                0x00,
                0x0C,
                0x49,
                0x44,
                0x41,
                0x54,  # IDAT chunk
                0x08,
                0xD7,
                0x63,
                0xF8,
                0xCF,
                0xC0,
                0x00,
                0x00,  # Compressed data
                0x00,
                0x03,
                0x00,
                0x01,  # Compressed data cont.
                0x00,
                0x18,
                0xDD,
                0x8D,
                0xB4,  # CRC
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,  # IEND chunk
                0xAE,
                0x42,
                0x60,
                0x82,  # CRC
            ]
        )

    # For larger images, we need PIL/Pillow
    try:
        from PIL import Image

        img = Image.new("RGB", (width, height), color)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""


class TestPipelineBuilderIntegration:
    """Test Pipeline builder creates valid specifications."""

    def test_simple_pipeline_to_json(self) -> None:
        """Test simple pipeline serializes to valid JSON."""
        pipe = (
            Pipeline().source("image_bytes").resize(height=224, width=224).sink("numpy")
        )

        json_str = pipe._to_json()
        assert '"source"' in json_str
        assert '"ops"' in json_str
        assert '"sink"' in json_str

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
            .sink("numpy")
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
            .sink("numpy")
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

    def test_cv_pipeline_method_exists(self) -> None:
        """Test that pipeline method exists on namespace."""
        import polars_cv.expressions  # noqa: F401

        expr = pl.col("images")
        assert hasattr(expr.cv, "pipeline")


# Check if plugin is available by checking if the .so file exists
def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    from pathlib import Path

    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


# Mark tests with plugin_required marker for easy filtering
plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@plugin_required
class TestPluginExecution:
    """Tests that require the compiled Rust plugin."""

    def test_simple_pipeline_execution(self) -> None:
        """Test basic pipeline execution."""
        pipe = Pipeline().source("image_bytes").resize(height=10, width=10).sink("blob")

        png_bytes = create_test_png(10, 10)
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
        assert "processed" in result.columns
        assert result["processed"].dtype == pl.Binary

    def test_pipeline_with_expression_args(self) -> None:
        """Test pipeline with expression arguments."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("w"))
            .sink("blob")
        )

        png_bytes = create_test_png(10, 10)
        df = pl.DataFrame({"images": [png_bytes, png_bytes], "h": [5, 8], "w": [5, 8]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
        assert "processed" in result.columns

    def test_grayscale_pipeline(self) -> None:
        """Test grayscale conversion."""
        pipe = Pipeline().source("image_bytes").grayscale().sink("blob")

        png_bytes = create_test_png(5, 5, (100, 150, 200))
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
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
            .sink("blob")
        )

        png_bytes = create_test_png(10, 10)
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
        assert result["processed"].dtype == pl.Binary


class TestPipelineValidation:
    """Test pipeline validation."""

    def test_incomplete_pipeline_raises(self) -> None:
        """Test that incomplete pipeline raises on validation."""
        pipe = Pipeline().source()

        with pytest.raises(ValueError, match="must have a sink"):
            pipe.validate()

    def test_pipeline_without_source_raises(self) -> None:
        """Test that pipeline without source raises on to_json."""
        pipe = Pipeline().resize(height=100, width=100).sink("numpy")

        with pytest.raises(ValueError, match="must have a source"):
            pipe._to_json()

    def test_valid_pipeline_passes_validation(self) -> None:
        """Test that valid pipeline passes validation."""
        pipe = Pipeline().source().resize(height=100, width=100).sink("numpy")

        pipe.validate()  # Should not raise


class TestExpressionTracking:
    """Test expression column tracking."""

    def test_no_expressions(self) -> None:
        """Test pipeline with no expressions has empty expr list."""
        pipe = Pipeline().source().resize(height=100, width=100).sink("numpy")

        assert len(pipe._get_expr_columns()) == 0

    def test_single_expression(self) -> None:
        """Test single expression is tracked."""
        pipe = Pipeline().source().resize(height=pl.col("h"), width=100).sink("numpy")

        exprs = pipe._get_expr_columns()
        assert len(exprs) == 1

    def test_multiple_expressions(self) -> None:
        """Test multiple expressions are tracked."""
        pipe = (
            Pipeline()
            .source()
            .resize(height=pl.col("h"), width=pl.col("w"))
            .crop(top=pl.col("t"), left=pl.col("l"))
            .sink("numpy")
        )

        exprs = pipe._get_expr_columns()
        assert len(exprs) == 4

    def test_duplicate_expressions_not_tracked_twice(self) -> None:
        """Test same expression object is not duplicated."""
        h_expr = pl.col("h")
        pipe = Pipeline().source().resize(height=h_expr, width=h_expr).sink("numpy")

        exprs = pipe._get_expr_columns()
        # Same expression object used twice should only be tracked once
        assert len(exprs) == 1
