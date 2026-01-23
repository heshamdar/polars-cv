"""
Tests for sink typing - ensuring output types are correctly preserved.

This module tests that:
1. List/Array sinks preserve the buffer's actual dtype (not force Float64)
2. Null values don't break type inference
3. Native sink types are correctly inferred for different domains
4. Unified graph entry works for both single and multi-output

These tests define the EXPECTED behavior. Many will fail until implementation
is complete.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from PIL import Image
from polars_cv import Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_image_bytes() -> bytes:
    """Create a simple 32x32 grayscale test image."""
    img = np.full((32, 32), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def rgb_image_bytes() -> bytes:
    """Create a simple 32x32 RGB test image."""
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def binary_mask_image_bytes() -> bytes:
    """Create a binary mask image (white square on black background)."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 255  # White square in center
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Test Class: Dtype Preservation for List/Array Sinks
# ============================================================


def get_innermost_dtype(dtype: pl.DataType) -> pl.DataType:
    """Extract the innermost dtype from a nested List/Array type."""
    current = dtype
    while hasattr(current, "inner") and current.inner is not None:
        current = current.inner
    return current


class TestDtypePreservationListSink:
    """
    Tests verifying that list sink preserves the buffer's actual dtype.

    The buffer dtype is determined by the FINAL operation in the pipeline,
    not the original source.

    Note: List sink now preserves shape as nested lists. For a 3D buffer
    [H, W, C], the result is List[List[List[inner_dtype]]]. These tests
    verify that the innermost dtype is correctly preserved.
    """

    def test_perceptual_hash_list_returns_uint8(self, rgb_image_bytes: bytes) -> None:
        """
        Perceptual hash outputs U8 bytes, list sink should return List[UInt8].

        The perceptual hash operation produces a 1D U8 buffer (shape [8]
        for 64-bit hash). When sunk as list, this should be a flat List[UInt8].
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        # Check the inner dtype of the list column
        hash_col = result["hash"]
        # Perceptual hash is 1D, so it should be a flat List[UInt8]
        assert hash_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8] for perceptual hash, got {hash_col.dtype}"
        )

        # Verify the values are in valid U8 range (0-255)
        hash_values = hash_col[0].to_list()
        assert len(hash_values) == 8, "64-bit hash should have 8 bytes"
        assert all(0 <= v <= 255 for v in hash_values), "Values should be in U8 range"

    def test_grayscale_list_returns_uint8(self, rgb_image_bytes: bytes) -> None:
        """
        Grayscale outputs U8, list sink should preserve UInt8 as innermost type.

        The grayscale operation has Fixed(U8) output dtype rule.
        Shape is [H, W, 1], so result is List[List[List[UInt8]]].
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("list")
        result = df.with_columns(gray=pl.col("image").cv.pipeline(pipe))

        gray_col = result["gray"]
        # Grayscale produces 3D shape [H, W, 1] -> nested lists
        expected_dtype = pl.List(pl.List(pl.List(pl.UInt8)))
        assert gray_col.dtype == expected_dtype, (
            f"Expected List[List[List[UInt8]]] for grayscale, got {gray_col.dtype}"
        )
        # Also verify innermost dtype is UInt8
        assert get_innermost_dtype(gray_col.dtype) == pl.UInt8, (
            "Innermost dtype should be UInt8"
        )

    def test_normalize_list_returns_float32(self, simple_image_bytes: bytes) -> None:
        """
        Normalize outputs F32 by default, list sink should preserve Float32 as innermost type.

        The normalize operation has Configurable(F32) output dtype rule.
        Shape is [H, W, C], so result is nested lists with Float32 innermost.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").normalize(method="minmax").sink("list")
        result = df.with_columns(normalized=pl.col("image").cv.pipeline(pipe))

        norm_col = result["normalized"]
        # Normalize produces 3D shape -> nested lists with Float32
        assert get_innermost_dtype(norm_col.dtype) == pl.Float32, (
            f"Expected innermost Float32 for normalize, got {get_innermost_dtype(norm_col.dtype)}"
        )

    def test_scale_list_returns_float32(self, simple_image_bytes: bytes) -> None:
        """
        Scale operation promotes to float, list sink should preserve Float32 as innermost type.

        The scale operation has PromoteToFloat output dtype rule.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").scale(factor=2.0).sink("list")
        result = df.with_columns(scaled=pl.col("image").cv.pipeline(pipe))

        scaled_col = result["scaled"]
        assert get_innermost_dtype(scaled_col.dtype) == pl.Float32, (
            f"Expected innermost Float32 for scale, got {get_innermost_dtype(scaled_col.dtype)}"
        )

    def test_threshold_list_returns_uint8(self, simple_image_bytes: bytes) -> None:
        """
        Threshold outputs U8, list sink should preserve UInt8 as innermost type.

        The threshold operation has Fixed(U8) output dtype rule.
        Note: Threshold requires single-channel input, so we convert to grayscale first.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().threshold(128).sink("list")
        result = df.with_columns(thresh=pl.col("image").cv.pipeline(pipe))

        thresh_col = result["thresh"]
        assert get_innermost_dtype(thresh_col.dtype) == pl.UInt8, (
            f"Expected innermost UInt8 for threshold, got {get_innermost_dtype(thresh_col.dtype)}"
        )

    def test_resize_list_returns_uint8(self, rgb_image_bytes: bytes) -> None:
        """
        Resize outputs U8, list sink should preserve UInt8 as innermost type.

        The resize operation has Fixed(U8) output dtype rule.
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").resize(height=16, width=16).sink("list")
        result = df.with_columns(resized=pl.col("image").cv.pipeline(pipe))

        resized_col = result["resized"]
        assert get_innermost_dtype(resized_col.dtype) == pl.UInt8, (
            f"Expected innermost UInt8 for resize, got {get_innermost_dtype(resized_col.dtype)}"
        )


class TestDtypePreservationArraySink:
    """
    Tests verifying that array sink preserves the buffer's actual dtype.
    """

    def test_perceptual_hash_array_returns_uint8(self, rgb_image_bytes: bytes) -> None:
        """
        Perceptual hash with array sink should return Array[UInt8, 8].
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = (
            Pipeline().source("image_bytes").perceptual_hash().sink("array", shape=[8])
        )
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        hash_col = result["hash"]
        # Array type should be Array[UInt8, 8]
        assert hash_col.dtype == pl.Array(pl.UInt8, 8), (
            f"Expected Array[UInt8, 8] for perceptual hash, got {hash_col.dtype}"
        )


# ============================================================
# Test Class: Null Value Handling
# ============================================================


class TestNullValueHandling:
    """
    Tests verifying that null values don't break type inference.

    Types are determined at planning time from the OutputSpec, not by
    inspecting runtime data. Even with all-null inputs, the output
    column has the correct type with proper nesting.
    """

    def test_null_values_preserve_type_list(self, simple_image_bytes: bytes) -> None:
        """
        Null values in input should still result in correctly typed list column.

        With shape preservation, grayscale produces [H, W, 1] -> nested lists.
        """
        df = pl.DataFrame({"image": [None, simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("list")
        result = df.with_columns(gray=pl.col("image").cv.pipeline(pipe))

        gray_col = result["gray"]
        # Type should have UInt8 as innermost type even with null first row
        assert get_innermost_dtype(gray_col.dtype) == pl.UInt8, (
            f"Expected innermost UInt8 with null first row, got {get_innermost_dtype(gray_col.dtype)}"
        )

        # First row should be null, second should have values
        assert gray_col[0] is None
        assert gray_col[1] is not None

    def test_all_null_values_preserve_type(self) -> None:
        """
        All null values should still result in correctly typed column.

        Even when input is entirely null (Polars Null dtype), the output
        type is determined from the pipeline's OutputSpec at planning time.
        """
        df = pl.DataFrame({"image": [None, None]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("list")
        result = df.with_columns(gray=pl.col("image").cv.pipeline(pipe))

        gray_col = result["gray"]
        # Type should have UInt8 as innermost type even with all nulls
        assert get_innermost_dtype(gray_col.dtype) == pl.UInt8, (
            f"Expected innermost UInt8 with all nulls, got {get_innermost_dtype(gray_col.dtype)}"
        )

    def test_mixed_null_and_values_perceptual_hash(
        self, rgb_image_bytes: bytes
    ) -> None:
        """
        Mixed null and valid values should preserve UInt8 type for hash.

        Perceptual hash is 1D, so it remains a flat List[UInt8].
        """
        df = pl.DataFrame({"image": [None, rgb_image_bytes, None, rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        hash_col = result["hash"]
        # Perceptual hash is 1D -> flat list
        assert hash_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8] with mixed nulls, got {hash_col.dtype}"
        )


# ============================================================
# Test Class: Native Sink Types
# ============================================================


class TestNativeSinkTypes:
    """
    Tests verifying native sink returns correct Polars types based on domain.
    """

    def test_reduce_sum_native_returns_float64(self, simple_image_bytes: bytes) -> None:
        """
        Scalar reduction with native sink should return Float64.

        reduce_sum() transitions to scalar domain, native sink should
        return Float64.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_sum().sink("native")
        result = df.with_columns(pixel_sum=pl.col("image").cv.pipeline(pipe))

        sum_col = result["pixel_sum"]
        assert sum_col.dtype == pl.Float64, (
            f"Expected Float64 for reduce_sum native, got {sum_col.dtype}"
        )

        # Value should be positive (sum of grayscale pixels)
        assert sum_col[0] > 0

    def test_extract_contours_native_returns_struct(
        self, binary_mask_image_bytes: bytes
    ) -> None:
        """
        Contour extraction with native sink should return Struct.

        extract_contours() transitions to contour domain, native sink should
        return a struct matching CONTOUR_SCHEMA.
        """
        df = pl.DataFrame({"image": [binary_mask_image_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()
            .sink("native")
        )
        result = df.with_columns(contour=pl.col("image").cv.pipeline(pipe))

        contour_col = result["contour"]
        # Should be a Struct type
        assert contour_col.dtype.base_type() == pl.Struct, (
            f"Expected Struct for extract_contours native, got {contour_col.dtype}"
        )

    def test_buffer_domain_native_errors(self, simple_image_bytes: bytes) -> None:
        """
        Buffer domain with native sink should raise an error.

        Native sink is only for scalar/contour/vector domains, not buffer.
        Users should explicitly specify numpy/png/jpeg/etc for buffers.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("native")

        with pytest.raises(Exception) as exc_info:
            df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Should mention that buffer requires explicit format
        error_msg = str(exc_info.value).lower()
        assert "buffer" in error_msg or "native" in error_msg or "format" in error_msg


# ============================================================
# Test Class: Unified Graph Entry
# ============================================================


class TestUnifiedGraphEntry:
    """
    Tests verifying that unified graph entry works for all scenarios.
    """

    def test_single_output_binary_through_unified(
        self, simple_image_bytes: bytes
    ) -> None:
        """
        Single output with binary sink should work through unified path.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_col = result["output"]
        assert isinstance(output_col.dtype, pl.Struct), (
            f"Expected Struct for numpy sink, got {output_col.dtype}"
        )

        # Verify we can decode the output
        arr = numpy_from_struct(output_col[0])
        assert arr.shape[0] == 32  # Height matches input
        assert arr.shape[1] == 32  # Width matches input

    def test_single_output_list_through_unified(
        self, simple_image_bytes: bytes
    ) -> None:
        """
        Single output with list sink should work through unified path.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("list")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_col = result["output"]
        # Should be properly typed list
        assert output_col.dtype.base_type() == pl.List

    def test_single_output_scalar_through_unified(
        self, simple_image_bytes: bytes
    ) -> None:
        """
        Single output scalar (native) should work through unified path.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_sum().sink("native")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_col = result["output"]
        assert output_col.dtype == pl.Float64, (
            f"Expected Float64 for scalar native, got {output_col.dtype}"
        )

    def test_multi_output_returns_struct(self, simple_image_bytes: bytes) -> None:
        """
        Multi-output should return Struct with correctly typed fields.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        # Build multi-output pipeline
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").grayscale())
            .alias("gray")
        )

        thresh = base.pipe(Pipeline().threshold(128)).alias("thresh")

        result = df.with_columns(
            outputs=thresh.sink({"gray": "numpy", "thresh": "numpy"})
        )

        outputs_col = result["outputs"]
        assert outputs_col.dtype.base_type() == pl.Struct, (
            f"Expected Struct for multi-output, got {outputs_col.dtype}"
        )

        # Each field should be Binary
        gray_field = outputs_col.struct.field("gray")
        thresh_field = outputs_col.struct.field("thresh")

        assert isinstance(gray_field.dtype, pl.Struct), (
            f"Expected Struct for gray numpy sink, got {gray_field.dtype}"
        )
        assert isinstance(thresh_field.dtype, pl.Struct), (
            f"Expected Struct for thresh numpy sink, got {thresh_field.dtype}"
        )

    def test_mixed_domain_multi_output(self, binary_mask_image_bytes: bytes) -> None:
        """
        Multi-output with different domains should have correctly typed fields.

        This tests a pipeline that produces both buffer and scalar outputs.
        """
        df = pl.DataFrame({"image": [binary_mask_image_bytes]})

        # Build pipeline with buffer and scalar outputs
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").grayscale().threshold(128))
            .alias("mask")
        )

        pixel_sum = base.pipe(Pipeline().reduce_sum()).alias("sum")

        result = df.with_columns(
            outputs=pixel_sum.sink({"mask": "numpy", "sum": "native"})
        )

        outputs_col = result["outputs"]
        assert outputs_col.dtype.base_type() == pl.Struct

        # mask should be Binary, sum should be Float64
        mask_field = outputs_col.struct.field("mask")
        sum_field = outputs_col.struct.field("sum")

        assert isinstance(mask_field.dtype, pl.Struct), (
            f"Expected Struct for mask numpy sink, got {mask_field.dtype}"
        )
        assert sum_field.dtype == pl.Float64, (
            f"Expected Float64 for sum, got {sum_field.dtype}"
        )


# ============================================================
# Test Class: Operation Chain Dtype Propagation
# ============================================================


class TestOperationChainDtype:
    """
    Tests verifying that dtype flows correctly through operation chains.

    With shape-preserving nested lists, the innermost dtype should
    match the final operation's output dtype.
    """

    def test_grayscale_then_normalize_is_float32(self, rgb_image_bytes: bytes) -> None:
        """
        grayscale (U8) -> normalize (F32) -> list should have innermost Float32.

        Shape is [H, W, 1], so result is nested lists with Float32 innermost.
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .normalize(method="minmax")
            .sink("list")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_col = result["output"]
        assert get_innermost_dtype(output_col.dtype) == pl.Float32, (
            f"Expected innermost Float32, got {get_innermost_dtype(output_col.dtype)}"
        )

    def test_grayscale_then_threshold_is_uint8(self, simple_image_bytes: bytes) -> None:
        """
        grayscale (U8) -> threshold (U8) -> list should have innermost UInt8.

        Threshold with Fixed(U8) output rule should produce U8 output.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)  # Threshold on U8 range
            .sink("list")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_col = result["output"]
        assert get_innermost_dtype(output_col.dtype) == pl.UInt8, (
            f"Expected innermost UInt8, got {get_innermost_dtype(output_col.dtype)}"
        )

    def test_resize_then_perceptual_hash_is_uint8(self, rgb_image_bytes: bytes) -> None:
        """
        resize (U8) -> perceptual_hash (U8) -> list should be List[UInt8].

        Perceptual hash is 1D, so it should be a flat list.
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .perceptual_hash()
            .sink("list")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_col = result["output"]
        # Perceptual hash is 1D -> flat list
        assert output_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8], got {output_col.dtype}"
        )


# ============================================================
# Test Class: Nested List Shape Preservation
# ============================================================


class TestNestedListShape:
    """
    Tests verifying that list sink preserves nested structure based on buffer shape.

    When a buffer has shape [H, W, C], the list sink should create
    nested lists that match this structure, not flatten everything.
    Note: After source(), images are always [H, W, 3] (RGB).
    After grayscale(), images are [H, W, 1].
    """

    def test_grayscale_list_preserves_3d_shape(self, simple_image_bytes: bytes) -> None:
        """
        Grayscale image (32x32x1) should return nested List[List[List[UInt8]]].

        After grayscale(), the shape is [32, 32, 1]. The list sink should
        create a nested structure: List[List[List[UInt8]]] where:
        - Outer list has 32 elements (height)
        - Middle list has 32 elements (width)
        - Inner list has 1 element (grayscale channel)
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("list")
        result = df.with_columns(gray=pl.col("image").cv.pipeline(pipe))

        gray_col = result["gray"]

        # Should be triple-nested list: List[List[List[UInt8]]]
        # Shape [32, 32, 1] -> 3 levels of nesting
        expected_dtype = pl.List(pl.List(pl.List(pl.UInt8)))
        assert gray_col.dtype == expected_dtype, (
            f"Expected List[List[List[UInt8]]] for grayscale (shape [32,32,1]), got {gray_col.dtype}"
        )

        # Verify the nested structure by converting to Python list
        nested_list = gray_col[0].to_list()
        assert nested_list is not None, "First row should not be null"

        # Should have 32 rows (height)
        assert len(nested_list) == 32, (
            f"Expected 32 rows (height), got {len(nested_list)}"
        )

        # Each row should be a list with 32 elements (width)
        for row in nested_list:
            assert isinstance(row, list), "Each row should be a list"
            assert len(row) == 32, (
                f"Expected 32 elements per row (width), got {len(row)}"
            )
            # Each pixel should be a list with 1 element (grayscale channel)
            for pixel in row:
                assert isinstance(pixel, list), "Each pixel should be a list"
                assert len(pixel) == 1, (
                    f"Expected 1 channel for grayscale, got {len(pixel)}"
                )
                assert isinstance(pixel[0], int) and 0 <= pixel[0] <= 255, (
                    "Channel value should be UInt8"
                )

    def test_rgb_list_preserves_3d_shape(self, rgb_image_bytes: bytes) -> None:
        """
        RGB image (32x32x3) should return nested List[List[List[UInt8]]].

        A 32x32x3 RGB image has shape [32, 32, 3]. The list sink should
        create a nested structure: List[List[List[UInt8]]] where:
        - Outer list has 32 elements (height)
        - Middle list has 32 elements (width)
        - Inner list has 3 elements (channels)
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").sink("list")
        result = df.with_columns(rgb=pl.col("image").cv.pipeline(pipe))

        rgb_col = result["rgb"]

        # Should be triple-nested list: List[List[List[UInt8]]]
        expected_dtype = pl.List(pl.List(pl.List(pl.UInt8)))
        assert rgb_col.dtype == expected_dtype, (
            f"Expected List[List[List[UInt8]]] for 3D RGB, got {rgb_col.dtype}"
        )

        # Verify the nested structure by converting to Python list
        nested_list = rgb_col[0].to_list()
        assert nested_list is not None, "First row should not be null"

        # Should have 32 rows (height)
        assert len(nested_list) == 32, (
            f"Expected 32 rows (height), got {len(nested_list)}"
        )

        # Each row should have 32 columns (width)
        for row in nested_list:
            assert isinstance(row, list), "Each row should be a list"
            assert len(row) == 32, f"Expected 32 columns (width), got {len(row)}"

            # Each pixel should be a list with 3 channels
            for pixel in row:
                assert isinstance(pixel, list), "Each pixel should be a list"
                assert len(pixel) == 3, f"Expected 3 channels, got {len(pixel)}"
                # All values should be UInt8 (0-255)
                assert all(isinstance(v, int) and 0 <= v <= 255 for v in pixel), (
                    "All channel values should be UInt8"
                )

    def test_resize_list_preserves_shape(self, rgb_image_bytes: bytes) -> None:
        """
        Resized image should preserve the new shape in nested list structure.

        Resizing to 16x16 maintains RGB format, so shape is [16, 16, 3].
        This should create List[List[List[UInt8]]] with 16x16x3 structure.
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").resize(height=16, width=16).sink("list")
        result = df.with_columns(resized=pl.col("image").cv.pipeline(pipe))

        resized_col = result["resized"]

        # Resize on RGB image maintains shape [16, 16, 3]
        # So should be List[List[List[UInt8]]]
        expected_dtype = pl.List(pl.List(pl.List(pl.UInt8)))
        assert resized_col.dtype == expected_dtype, (
            f"Expected List[List[List[UInt8]]] for resized image (shape [16,16,3]), got {resized_col.dtype}"
        )

        # Verify the nested structure matches resize dimensions
        nested_list = resized_col[0].to_list()
        assert nested_list is not None, "First row should not be null"
        assert len(nested_list) == 16, (
            f"Expected 16 rows (height), got {len(nested_list)}"
        )

        for row in nested_list:
            assert isinstance(row, list), "Each row should be a list"
            assert len(row) == 16, f"Expected 16 columns (width), got {len(row)}"
            # Each pixel should have 3 channels
            for pixel in row:
                assert isinstance(pixel, list), "Each pixel should be a list"
                assert len(pixel) == 3, f"Expected 3 channels, got {len(pixel)}"
