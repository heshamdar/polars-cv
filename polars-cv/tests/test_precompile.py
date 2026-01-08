"""
Tests for pre-compile optimization in pipeline execution.

These tests verify that:
1. Literal pipelines produce correct results (before/after optimization)
2. Expression pipelines still resolve per-row correctly
3. Mixed literal/expression pipelines work correctly
4. Expression fallback doesn't accidentally cache first row's values

These tests should pass BEFORE implementing the optimization (establishing baseline),
and must continue to pass AFTER the optimization is implemented.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


def create_test_png(
    width: int = 10, height: int = 10, color: tuple[int, int, int] = (255, 0, 0)
) -> bytes:
    """
    Create a test PNG image.

    Args:
        width: Image width.
        height: Image height.
        color: RGB color tuple.

    Returns:
        PNG bytes.
    """
    try:
        from PIL import Image

        img = Image.new("RGB", (width, height), color)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")
        return b""


def get_shape(struct_val: dict) -> tuple[int, ...]:
    """
    Extract shape from numpy output struct.

    Args:
        struct_val: The struct value from a numpy sink output.

    Returns:
        Shape as a tuple of integers.
    """
    if struct_val is None:
        return ()
    shape_list = struct_val.get("shape")
    if shape_list is None:
        return ()
    if isinstance(shape_list, pl.Series):
        return tuple(int(x) for x in shape_list.to_list())
    return tuple(int(x) for x in shape_list)


def _plugin_available() -> bool:
    """Check if the compiled plugin is available."""
    from pathlib import Path

    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


@plugin_required
class TestLiteralPipelineCorrectness:
    """Test that pipelines with all-literal parameters produce correct results."""

    def test_literal_resize_produces_correct_dimensions(self) -> None:
        """Fixed resize dimensions should produce correct output shape."""
        pipe = (
            Pipeline().source("image_bytes").resize(height=50, width=50).sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
        output_struct = result["processed"][0]

        # Verify dimensions using struct shape field
        shape = get_shape(output_struct)
        assert shape[0] == 50, f"Expected height 50, got {shape[0]}"
        assert shape[1] == 50, f"Expected width 50, got {shape[1]}"

    def test_literal_crop_produces_correct_dimensions(self) -> None:
        """Fixed crop dimensions should produce correct output shape."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .crop(top=10, left=10, height=30, width=40)
            .sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))
        output_struct = result["processed"][0]

        shape = get_shape(output_struct)
        assert shape[0] == 30, f"Expected height 30, got {shape[0]}"
        assert shape[1] == 40, f"Expected width 40, got {shape[1]}"

    def test_literal_multi_op_pipeline(self) -> None:
        """Pipeline with multiple literal ops should produce consistent results."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .grayscale()
            .threshold(128)
            .sink("numpy")
        )

        png_bytes = create_test_png(100, 100, (200, 100, 50))
        df = pl.DataFrame({"images": [png_bytes, png_bytes, png_bytes]})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # All rows should produce identical output (same input, same literal pipeline)
        arr0 = numpy_from_struct(result["processed"][0])
        arr1 = numpy_from_struct(result["processed"][1])
        arr2 = numpy_from_struct(result["processed"][2])

        np.testing.assert_array_equal(arr0, arr1)
        np.testing.assert_array_equal(arr1, arr2)

    def test_literal_batch_consistency(self) -> None:
        """Large batch with literal pipeline should produce consistent results."""
        pipe = (
            Pipeline().source("image_bytes").resize(height=32, width=32).sink("numpy")
        )

        png_bytes = create_test_png(50, 50)
        # Create a batch of 100 identical images
        df = pl.DataFrame({"images": [png_bytes] * 100})

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # All outputs should have same shape
        first_shape = get_shape(result["processed"][0])
        for i in range(100):
            shape = get_shape(result["processed"][i])
            assert shape == first_shape, f"Row {i} has different shape: {shape}"


@plugin_required
class TestExpressionPipelineCorrectness:
    """Test that pipelines with expression parameters resolve per-row correctly."""

    def test_expression_resize_per_row(self) -> None:
        """Expression-based resize should produce different dimensions per row."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("w"))
            .sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame(
            {
                "images": [png_bytes, png_bytes, png_bytes],
                "h": [10, 20, 30],
                "w": [15, 25, 35],
            }
        )

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Each row should have different dimensions based on h/w columns
        shape0 = get_shape(result["processed"][0])
        shape1 = get_shape(result["processed"][1])
        shape2 = get_shape(result["processed"][2])

        assert shape0[0] == 10 and shape0[1] == 15, (
            f"Row 0: expected (10,15), got {shape0[:2]}"
        )
        assert shape1[0] == 20 and shape1[1] == 25, (
            f"Row 1: expected (20,25), got {shape1[:2]}"
        )
        assert shape2[0] == 30 and shape2[1] == 35, (
            f"Row 2: expected (30,35), got {shape2[:2]}"
        )

    def test_expression_crop_per_row(self) -> None:
        """Expression-based crop should use different params per row."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .crop(top=pl.col("t"), left=pl.col("l"), height=20, width=20)
            .sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame(
            {
                "images": [png_bytes, png_bytes],
                "t": [0, 50],
                "l": [0, 50],
            }
        )

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Both should produce 20x20 crops, but from different positions
        shape0 = get_shape(result["processed"][0])
        shape1 = get_shape(result["processed"][1])

        assert shape0[0] == 20 and shape0[1] == 20
        assert shape1[0] == 20 and shape1[1] == 20

    def test_expression_fallback_not_cached(self) -> None:
        """
        Ensure expression path doesn't accidentally cache first row's values.

        This is the critical test for the pre-compile optimization:
        if we accidentally use the first row's values for all rows, this test will fail.
        """
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("target_size"), width=pl.col("target_size"))
            .sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame(
            {
                "images": [png_bytes, png_bytes, png_bytes, png_bytes],
                "target_size": [10, 20, 30, 40],  # Each row should have different size
            }
        )

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Verify each row has the correct size - NOT all using first row's value
        for i, expected_size in enumerate([10, 20, 30, 40]):
            shape = get_shape(result["processed"][i])
            assert shape[0] == expected_size, (
                f"Row {i}: expected size {expected_size}, got {shape[0]}. "
                "Expression values may be cached incorrectly!"
            )
            assert shape[1] == expected_size, (
                f"Row {i}: expected size {expected_size}, got {shape[1]}. "
                "Expression values may be cached incorrectly!"
            )


@plugin_required
class TestMixedLiteralExpressionPipeline:
    """Test pipelines with a mix of literal and expression parameters."""

    def test_mixed_resize_literal_height_expr_width(self) -> None:
        """Literal height with expression width should work correctly."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=50, width=pl.col("w"))  # height literal, width expression
            .sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame(
            {
                "images": [png_bytes, png_bytes, png_bytes],
                "w": [20, 40, 60],
            }
        )

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Height should be 50 for all, width should vary
        shape0 = get_shape(result["processed"][0])
        shape1 = get_shape(result["processed"][1])
        shape2 = get_shape(result["processed"][2])

        # All heights should be 50 (literal)
        assert shape0[0] == 50
        assert shape1[0] == 50
        assert shape2[0] == 50

        # Widths should match the expression values
        assert shape0[1] == 20
        assert shape1[1] == 40
        assert shape2[1] == 60

    def test_mixed_multi_op_some_literal_some_expr(self) -> None:
        """Pipeline with some literal ops and some expression ops."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)  # All literal
            .crop(
                top=pl.col("crop_t"),
                left=pl.col("crop_l"),
                height=32,  # literal
                width=32,  # literal
            )
            .sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame(
            {
                "images": [png_bytes, png_bytes],
                "crop_t": [0, 16],
                "crop_l": [0, 16],
            }
        )

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Both should produce 32x32 output (literal crop dimensions)
        shape0 = get_shape(result["processed"][0])
        shape1 = get_shape(result["processed"][1])

        assert shape0[0] == 32 and shape0[1] == 32
        assert shape1[0] == 32 and shape1[1] == 32


@plugin_required
class TestStreamingVsEagerPrecompile:
    """Test pre-compile behavior works with both streaming and eager execution."""

    def test_literal_pipeline_streaming(self) -> None:
        """Literal pipeline with streaming execution."""
        pipe = (
            Pipeline().source("image_bytes").resize(height=32, width=32).sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame({"images": [png_bytes] * 10})

        # Use streaming execution
        result = (
            df.lazy()
            .with_columns(processed=pl.col("images").cv.pipeline(pipe))
            .collect(engine="streaming")
        )

        # All should have same shape
        for i in range(10):
            shape = get_shape(result["processed"][i])
            assert shape[0] == 32 and shape[1] == 32

    def test_expression_pipeline_streaming(self) -> None:
        """Expression pipeline with streaming execution."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("w"))
            .sink("numpy")
        )

        png_bytes = create_test_png(100, 100)
        df = pl.DataFrame(
            {
                "images": [png_bytes, png_bytes],
                "h": [25, 50],
                "w": [25, 50],
            }
        )

        # Use streaming execution
        result = (
            df.lazy()
            .with_columns(processed=pl.col("images").cv.pipeline(pipe))
            .collect(engine="streaming")
        )

        shape0 = get_shape(result["processed"][0])
        shape1 = get_shape(result["processed"][1])

        assert shape0[0] == 25 and shape0[1] == 25
        assert shape1[0] == 50 and shape1[1] == 50
