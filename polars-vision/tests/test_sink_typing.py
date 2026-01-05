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

from polars_vision import Pipeline, numpy_from_bytes

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


class TestDtypePreservationListSink:
    """
    Tests verifying that list sink preserves the buffer's actual dtype.

    The buffer dtype is determined by the FINAL operation in the pipeline,
    not the original source.
    """

    def test_perceptual_hash_list_returns_uint8(
        self, rgb_image_bytes: bytes
    ) -> None:
        """
        Perceptual hash outputs U8 bytes, list sink should return List[UInt8].

        The perceptual hash operation produces a fixed-size U8 buffer (shape [8]
        for 64-bit hash). When sunk as list, this should remain UInt8, not
        be converted to Float64.
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        # Check the inner dtype of the list column
        hash_col = result["hash"]
        assert hash_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8] for perceptual hash, got {hash_col.dtype}"
        )

        # Verify the values are in valid U8 range (0-255)
        hash_values = hash_col[0].to_list()
        assert len(hash_values) == 8, "64-bit hash should have 8 bytes"
        assert all(0 <= v <= 255 for v in hash_values), "Values should be in U8 range"

    def test_grayscale_list_returns_uint8(self, rgb_image_bytes: bytes) -> None:
        """
        Grayscale outputs U8, list sink should return List[UInt8].

        The grayscale operation has Fixed(U8) output dtype rule.
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("list")
        result = df.with_columns(gray=pl.col("image").cv.pipeline(pipe))

        gray_col = result["gray"]
        assert gray_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8] for grayscale, got {gray_col.dtype}"
        )

    def test_normalize_list_returns_float32(self, simple_image_bytes: bytes) -> None:
        """
        Normalize outputs F32 by default, list sink should return List[Float32].

        The normalize operation has Configurable(F32) output dtype rule.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").normalize(method="minmax").sink("list")
        result = df.with_columns(normalized=pl.col("image").cv.pipeline(pipe))

        norm_col = result["normalized"]
        assert norm_col.dtype == pl.List(pl.Float32), (
            f"Expected List[Float32] for normalize, got {norm_col.dtype}"
        )

    def test_scale_list_returns_float32(self, simple_image_bytes: bytes) -> None:
        """
        Scale operation promotes to float, list sink should return List[Float32].

        The scale operation has PromoteToFloat output dtype rule.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").scale(factor=2.0).sink("list")
        result = df.with_columns(scaled=pl.col("image").cv.pipeline(pipe))

        scaled_col = result["scaled"]
        assert scaled_col.dtype == pl.List(pl.Float32), (
            f"Expected List[Float32] for scale, got {scaled_col.dtype}"
        )

    def test_threshold_list_returns_uint8(self, simple_image_bytes: bytes) -> None:
        """
        Threshold outputs U8, list sink should return List[UInt8].

        The threshold operation has Fixed(U8) output dtype rule.
        """
        df = pl.DataFrame({"image": [simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").threshold(128).sink("list")
        result = df.with_columns(thresh=pl.col("image").cv.pipeline(pipe))

        thresh_col = result["thresh"]
        assert thresh_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8] for threshold, got {thresh_col.dtype}"
        )

    def test_resize_list_returns_uint8(self, rgb_image_bytes: bytes) -> None:
        """
        Resize outputs U8, list sink should return List[UInt8].

        The resize operation has Fixed(U8) output dtype rule.
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=16, width=16)
            .sink("list")
        )
        result = df.with_columns(resized=pl.col("image").cv.pipeline(pipe))

        resized_col = result["resized"]
        assert resized_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8] for resize, got {resized_col.dtype}"
        )


class TestDtypePreservationArraySink:
    """
    Tests verifying that array sink preserves the buffer's actual dtype.
    """

    def test_perceptual_hash_array_returns_uint8(
        self, rgb_image_bytes: bytes
    ) -> None:
        """
        Perceptual hash with array sink should return Array[UInt8, 8].
        """
        df = pl.DataFrame({"image": [rgb_image_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .perceptual_hash()
            .sink("array", shape=[8])
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
    column has the correct type (e.g., List[UInt8], Binary, etc.).
    """

    def test_null_values_preserve_type_list(
        self, simple_image_bytes: bytes
    ) -> None:
        """
        Null values in input should still result in correctly typed list column.
        """
        df = pl.DataFrame({"image": [None, simple_image_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("list")
        result = df.with_columns(gray=pl.col("image").cv.pipeline(pipe))

        gray_col = result["gray"]
        # Type should be List[UInt8] even with null first row
        assert gray_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8] with null first row, got {gray_col.dtype}"
        )

        # First row should be null, second should have values
        assert gray_col[0] is None or gray_col[0].is_empty()
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
        # Type should be List[UInt8] even with all nulls
        assert gray_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8] with all nulls, got {gray_col.dtype}"
        )

    def test_mixed_null_and_values_perceptual_hash(
        self, rgb_image_bytes: bytes
    ) -> None:
        """
        Mixed null and valid values should preserve UInt8 type for hash.
        """
        df = pl.DataFrame({"image": [None, rgb_image_bytes, None, rgb_image_bytes]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        hash_col = result["hash"]
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

    def test_reduce_sum_native_returns_float64(
        self, simple_image_bytes: bytes
    ) -> None:
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
        assert output_col.dtype == pl.Binary, (
            f"Expected Binary for numpy sink, got {output_col.dtype}"
        )

        # Verify we can decode the output
        arr = numpy_from_bytes(output_col[0])
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
        base = pl.col("image").cv.pipe(
            Pipeline().source("image_bytes").grayscale()
        ).alias("gray")

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

        assert gray_field.dtype == pl.Binary
        assert thresh_field.dtype == pl.Binary

    def test_mixed_domain_multi_output(self, binary_mask_image_bytes: bytes) -> None:
        """
        Multi-output with different domains should have correctly typed fields.

        This tests a pipeline that produces both buffer and scalar outputs.
        """
        df = pl.DataFrame({"image": [binary_mask_image_bytes]})

        # Build pipeline with buffer and scalar outputs
        base = pl.col("image").cv.pipe(
            Pipeline().source("image_bytes").grayscale().threshold(128)
        ).alias("mask")

        pixel_sum = base.pipe(Pipeline().reduce_sum()).alias("sum")

        result = df.with_columns(
            outputs=pixel_sum.sink({"mask": "numpy", "sum": "native"})
        )

        outputs_col = result["outputs"]
        assert outputs_col.dtype.base_type() == pl.Struct

        # mask should be Binary, sum should be Float64
        mask_field = outputs_col.struct.field("mask")
        sum_field = outputs_col.struct.field("sum")

        assert mask_field.dtype == pl.Binary, (
            f"Expected Binary for mask, got {mask_field.dtype}"
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
    """

    def test_grayscale_then_normalize_is_float32(
        self, rgb_image_bytes: bytes
    ) -> None:
        """
        grayscale (U8) -> normalize (F32) -> list should be List[Float32].
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
        assert output_col.dtype == pl.List(pl.Float32), (
            f"Expected List[Float32], got {output_col.dtype}"
        )

    def test_grayscale_then_threshold_is_uint8(
        self, simple_image_bytes: bytes
    ) -> None:
        """
        grayscale (U8) -> threshold (U8) -> list should be List[UInt8].

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
        assert output_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8], got {output_col.dtype}"
        )

    def test_resize_then_perceptual_hash_is_uint8(
        self, rgb_image_bytes: bytes
    ) -> None:
        """
        resize (U8) -> perceptual_hash (U8) -> list should be List[UInt8].
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
        assert output_col.dtype == pl.List(pl.UInt8), (
            f"Expected List[UInt8], got {output_col.dtype}"
        )

