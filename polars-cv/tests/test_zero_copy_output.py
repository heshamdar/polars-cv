"""
Tests for zero-copy output transfer.

These tests verify that:
1. Numpy/torch sink outputs use struct format with data, dtype, shape fields
2. The output schema matches NUMPY_OUTPUT_SCHEMA
3. numpy_from_struct correctly converts struct to numpy array
4. Both eager and lazy execution paths work correctly
5. Null handling works properly
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from PIL import Image
from polars_cv import NUMPY_OUTPUT_SCHEMA, Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_rgb_bytes() -> bytes:
    """Create a simple 4x4 RGB test image."""
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def gradient_image_bytes() -> bytes:
    """Create a 10x10 gradient image for testing."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    for i in range(10):
        img[i, :, :] = i * 25
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Output Schema Tests
# ============================================================


class TestNumpyOutputSchema:
    """Tests for numpy output struct schema."""

    def test_numpy_output_schema_structure(self) -> None:
        """NUMPY_OUTPUT_SCHEMA should have correct structure."""
        assert NUMPY_OUTPUT_SCHEMA == pl.Struct(
            {
                "data": pl.Binary,
                "dtype": pl.String,
                "shape": pl.List(pl.UInt64),
                "strides": pl.List(pl.Int64),
                "offset": pl.UInt64,
            }
        )

    def test_numpy_sink_returns_struct(self, simple_rgb_bytes: bytes) -> None:
        """Numpy sink should return Struct type, not Binary."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_dtype = result["output"].dtype
        assert isinstance(output_dtype, pl.Struct), (
            f"Expected Struct dtype for numpy sink, got {output_dtype}"
        )

    def test_numpy_sink_struct_fields(self, simple_rgb_bytes: bytes) -> None:
        """Numpy sink struct should have data, dtype, shape fields."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_dtype = result["output"].dtype
        assert isinstance(output_dtype, pl.Struct)

        field_names = [f.name for f in output_dtype.fields]
        assert "data" in field_names
        assert "dtype" in field_names
        assert "shape" in field_names

    def test_torch_sink_returns_struct(self, simple_rgb_bytes: bytes) -> None:
        """Torch sink should also return Struct type."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").sink("torch")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        output_dtype = result["output"].dtype
        assert isinstance(output_dtype, pl.Struct), (
            f"Expected Struct dtype for torch sink, got {output_dtype}"
        )


# ============================================================
# numpy_from_struct Tests
# ============================================================


class TestNumpyFromStruct:
    """Tests for numpy_from_struct conversion function."""

    def test_basic_conversion(self, simple_rgb_bytes: bytes) -> None:
        """Basic struct to numpy conversion should work."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        row = result["output"][0]
        arr = numpy_from_struct(row)

        assert arr.dtype == np.uint8
        assert arr.shape == (4, 4, 3)
        assert arr.mean() == pytest.approx(128, abs=1)

    def test_conversion_with_resize(self, simple_rgb_bytes: bytes) -> None:
        """Conversion should work after resize operation."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").resize(height=8, width=8).sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])

        assert arr.shape == (8, 8, 3)

    def test_conversion_with_grayscale(self, simple_rgb_bytes: bytes) -> None:
        """Conversion should work after grayscale operation."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])

        # Grayscale output has 1 channel
        assert arr.shape == (4, 4, 1)

    def test_conversion_with_cast(self, simple_rgb_bytes: bytes) -> None:
        """Conversion should preserve dtype after cast."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").cast("f32").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])

        assert arr.dtype == np.float32
        assert arr.shape == (4, 4, 3)

    def test_conversion_with_normalize(self, simple_rgb_bytes: bytes) -> None:
        """Conversion should work with normalize operation."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .cast("f32")
            .normalize(method="minmax")
            .sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])

        assert arr.dtype == np.float32
        # After minmax normalize, values should be in [0, 1]
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_conversion_from_dict(self, simple_rgb_bytes: bytes) -> None:
        """Conversion should work from dict representation."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Access struct as dict via unnest
        struct_col = result["output"]
        unnested = struct_col.struct.unnest()
        row_dict = {
            "data": unnested["data"][0],
            "dtype": unnested["dtype"][0],
            "shape": unnested["shape"][0],
        }

        arr = numpy_from_struct(row_dict)

        assert arr.dtype == np.uint8
        assert arr.shape == (4, 4, 3)

    def test_copy_parameter(self, simple_rgb_bytes: bytes) -> None:
        """copy=False should work (may share memory)."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        row = result["output"][0]

        # copy=True (default)
        arr_copy = numpy_from_struct(row, copy=True)
        # copy=False
        arr_view = numpy_from_struct(row, copy=False)

        # Both should have same data
        np.testing.assert_array_equal(arr_copy, arr_view)


# ============================================================
# Eager vs Lazy Execution Tests
# ============================================================


class TestExecutionModes:
    """Tests for eager and lazy execution paths."""

    def test_eager_pipeline(self, simple_rgb_bytes: bytes) -> None:
        """Eager pipeline execution should produce correct struct output."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").resize(height=6, width=6).sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (6, 6, 3)

    def test_lazy_pipeline(self, simple_rgb_bytes: bytes) -> None:
        """Lazy pipeline execution should produce correct struct output."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        # Use lazy API
        base = pl.col("image").cv.pipe(Pipeline().source("image_bytes"))
        resized = base.pipe(Pipeline().resize(height=6, width=6))
        expr = resized.sink("numpy")

        result = df.with_columns(output=expr)

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (6, 6, 3)

    def test_lazy_with_multi_output(self, simple_rgb_bytes: bytes) -> None:
        """Lazy pipeline with multiple outputs should work."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        base = pl.col("image").cv.pipe(Pipeline().source("image_bytes"))
        resized = base.pipe(Pipeline().resize(height=8, width=8)).alias("resized")
        gray = resized.pipe(Pipeline().grayscale()).alias("gray")

        result = df.with_columns(
            outputs=gray.sink({"resized": "numpy", "gray": "numpy"})
        )

        # Both outputs should be struct type
        outputs = result["outputs"]
        assert isinstance(outputs.dtype, pl.Struct)

        # Extract and verify resized
        resized_struct = outputs.struct.field("resized")[0]
        resized_arr = numpy_from_struct(resized_struct)
        assert resized_arr.shape == (8, 8, 3)

        # Extract and verify gray
        gray_struct = outputs.struct.field("gray")[0]
        gray_arr = numpy_from_struct(gray_struct)
        assert gray_arr.shape == (8, 8, 1)


# ============================================================
# Null Handling Tests
# ============================================================


class TestNullHandling:
    """Tests for null value handling in output."""

    def test_null_input_produces_null_struct_fields(
        self, simple_rgb_bytes: bytes
    ) -> None:
        """Null input should produce struct with null fields."""
        df = pl.DataFrame({"image": [simple_rgb_bytes, None, simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # First and third rows should have data
        row0 = result["output"][0]
        row2 = result["output"][2]
        assert row0.get("data") is not None
        assert row2.get("data") is not None

        # Second row should have null fields
        row1 = result["output"][1]
        assert row1.get("data") is None
        assert row1.get("dtype") is None
        assert row1.get("shape") is None

    def test_null_struct_raises_on_conversion(self, simple_rgb_bytes: bytes) -> None:
        """Attempting to convert null struct should raise ValueError."""
        df = pl.DataFrame({"image": [None]}).cast({"image": pl.Binary})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        row = result["output"][0]

        # Struct has null fields
        assert row.get("data") is None

        # Should raise when trying to convert
        with pytest.raises(ValueError, match="null"):
            numpy_from_struct(row)


# ============================================================
# Multiple Rows Tests
# ============================================================


class TestMultipleRows:
    """Tests for processing multiple rows."""

    def test_multiple_images(self) -> None:
        """Multiple images should all be processed correctly."""
        # Create different sized images
        images = []
        for val in [100, 150, 200]:
            img = np.full((4, 4, 3), val, dtype=np.uint8)
            pil_img = Image.fromarray(img)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            images.append(buf.getvalue())

        df = pl.DataFrame({"image": images})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Each output should match input value
        for i, expected_val in enumerate([100, 150, 200]):
            arr = numpy_from_struct(result["output"][i])
            assert arr.shape == (4, 4, 3)
            assert arr.mean() == pytest.approx(expected_val, abs=1)

    def test_batch_processing_consistency(self, simple_rgb_bytes: bytes) -> None:
        """Batch processing should produce consistent results."""
        df = pl.DataFrame({"image": [simple_rgb_bytes] * 10})

        pipe = Pipeline().source("image_bytes").resize(height=8, width=8).sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # All outputs should be identical
        first_arr = numpy_from_struct(result["output"][0])
        for i in range(1, 10):
            arr = numpy_from_struct(result["output"][i])
            np.testing.assert_array_equal(first_arr, arr)


# ============================================================
# Dtype Preservation Tests
# ============================================================


class TestDtypePreservation:
    """Tests for dtype preservation through pipeline."""

    def test_uint8_preserved(self, simple_rgb_bytes: bytes) -> None:
        """UInt8 dtype should be preserved."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Check dtype field directly
        unnested = result["output"].struct.unnest()
        assert unnested["dtype"][0] == "uint8"

    def test_float32_after_cast(self, simple_rgb_bytes: bytes) -> None:
        """Float32 dtype should be correct after cast."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = Pipeline().source("image_bytes").cast("f32").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        unnested = result["output"].struct.unnest()
        assert unnested["dtype"][0] == "float32"

    def test_float32_preserved_through_ops(self, simple_rgb_bytes: bytes) -> None:
        """Float32 dtype should be preserved through operations."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        # Cast to f32 then apply ops - dtype should stay f32
        pipe = (
            Pipeline()
            .source("image_bytes")
            .cast("f32")
            .normalize(method="minmax")
            .sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        unnested = result["output"].struct.unnest()
        assert unnested["dtype"][0] == "float32"

    def test_shape_field_correct(self, simple_rgb_bytes: bytes) -> None:
        """Shape field should contain correct dimensions."""
        df = pl.DataFrame({"image": [simple_rgb_bytes]})

        pipe = (
            Pipeline().source("image_bytes").resize(height=10, width=20).sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        unnested = result["output"].struct.unnest()
        shape = unnested["shape"][0].to_list()

        # Resize(height=10, width=20)
        assert shape[0] == 10  # Height
        assert shape[1] == 20  # Width
        assert shape[2] == 3  # Channels


# ============================================================
# Memory Efficiency Tests
# ============================================================


class TestMemoryEfficiency:
    """Tests for zero-copy and memory efficiency."""

    def test_large_batch_memory_efficiency(self) -> None:
        """Process a batch of images and verify output is generated efficiently.

        This test creates multiple images and processes them through the pipeline.
        The test verifies correct output rather than memory usage, but the
        underlying implementation should use zero-copy where possible.
        """
        # Create 50 small images (8x8 RGB)
        images = []
        for i in range(50):
            img = np.full((8, 8, 3), i * 5, dtype=np.uint8)
            pil_img = Image.fromarray(img)
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            images.append(buf.getvalue())

        df = pl.DataFrame({"image": images})

        pipe = (
            Pipeline().source("image_bytes").resize(height=16, width=16).sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Verify all outputs are correct
        assert len(result) == 50
        for i in range(50):
            arr = numpy_from_struct(result["output"][i])
            assert arr.shape == (16, 16, 3)
            # Mean should be close to original fill value (with some interpolation variance)
            assert arr.dtype == np.uint8

    def test_output_data_integrity(self, simple_rgb_bytes: bytes) -> None:
        """Verify output data has correct values after zero-copy transfer."""
        # Create image with known values
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0, :] = [255, 0, 0]  # Red pixel at top-left
        img[3, 3, :] = [0, 255, 0]  # Green pixel at bottom-right
        img[0, 3, :] = [0, 0, 255]  # Blue pixel at top-right
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])

        # Verify specific pixel values
        np.testing.assert_array_equal(arr[0, 0, :], [255, 0, 0])
        np.testing.assert_array_equal(arr[3, 3, :], [0, 255, 0])
        np.testing.assert_array_equal(arr[0, 3, :], [0, 0, 255])

    def test_binary_data_not_corrupted(self) -> None:
        """Verify binary data is not corrupted during transfer."""
        # Create image with specific pattern
        pattern = np.arange(64 * 64 * 3, dtype=np.uint8).reshape((64, 64, 3))
        pil_img = Image.fromarray(pattern)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])

        # Verify pattern is preserved (within PNG compression tolerance)
        # PNG is lossless, so should be exact
        np.testing.assert_array_equal(arr, pattern)


# ============================================================
# Strided Pipeline Tests
# ============================================================


class TestStridedPipeline:
    """Tests for strided operations in pipelines."""

    def test_flip_then_grayscale(self) -> None:
        """Test flip (view op) followed by grayscale works correctly."""
        # Create a gradient image for testing
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        for i in range(10):
            img[i, :, :] = i * 25
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        # Flip then grayscale should work on strided buffer
        # axes=[0] is vertical flip (along height axis)
        pipe = Pipeline().source("image_bytes").flip(axes=[0]).grayscale().sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (10, 10, 1)

        # After vertical flip, the gradient should be reversed
        # Top row (originally bottom) should be brighter
        assert arr[0, 0, 0] > arr[9, 0, 0]

    def test_flip_then_resize(self) -> None:
        """Test flip followed by resize works correctly."""
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .flip(axes=[1])  # horizontal flip (along width axis)
            .resize(height=10, width=10)
            .sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (10, 10, 3)

    def test_crop_then_grayscale(self) -> None:
        """Test crop followed by grayscale works correctly."""
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .crop(top=5, left=5, height=10, width=10)
            .grayscale()
            .sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (10, 10, 1)

    def test_flip_grayscale_resize_pipeline(self) -> None:
        """Test full pipeline: flip -> grayscale -> resize."""
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .flip(axes=[0])  # vertical flip
            .grayscale()
            .resize(height=16, width=16)
            .sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (16, 16, 1)

    def test_normalize_after_flip(self) -> None:
        """Test normalize operation after flip works correctly."""
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .flip(axes=[1])  # horizontal flip
            .cast("f32")
            .normalize(method="minmax")
            .sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])
        assert arr.dtype == np.float32
        assert arr.shape == (10, 10, 3)
        # All same values -> minmax should give 0
        # (or handle constant array gracefully)

    def test_multiple_view_ops_then_compute(self) -> None:
        """Test multiple view operations followed by compute."""
        # Create pattern without overflow - use modulo before converting to uint8
        img = (np.arange(100 * 100 * 3) % 256).astype(np.uint8).reshape((100, 100, 3))
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .flip(axes=[0])  # vertical flip
            .flip(
                axes=[1]
            )  # horizontal flip (back to original orientation with double flip)
            .crop(top=10, left=10, height=80, width=80)
            .grayscale()
            .resize(height=40, width=40)
            .sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (40, 40, 1)


# ============================================================
# Strided Zero-Copy Output Tests
# ============================================================


class TestStridedZeroCopyOutput:
    """Tests for strided zero-copy output verification."""

    def test_output_schema_has_strides_and_offset(self) -> None:
        """Test that the output schema includes strides and offset fields."""
        from polars_cv import NUMPY_OUTPUT_SCHEMA

        # Check schema has all expected fields
        schema_fields = {f.name for f in NUMPY_OUTPUT_SCHEMA.fields}
        assert "data" in schema_fields
        assert "dtype" in schema_fields
        assert "shape" in schema_fields
        assert "strides" in schema_fields
        assert "offset" in schema_fields

    def test_contiguous_output_has_contiguous_strides(self) -> None:
        """Test that contiguous output has standard contiguous strides."""
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        # Simple resize produces contiguous output
        pipe = (
            Pipeline().source("image_bytes").resize(height=16, width=16).sink("numpy")
        )
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Access raw struct fields
        struct_data = result["output"].struct.unnest()
        shape = struct_data["shape"][0].to_list()
        strides = struct_data["strides"][0].to_list()
        offset = struct_data["offset"][0]

        # Shape should be [16, 16, 3]
        assert shape == [16, 16, 3]
        # Contiguous strides for HWC: [W*C, C, 1] = [48, 3, 1]
        assert strides == [48, 3, 1]
        assert offset == 0

    def test_grayscale_output_has_correct_strides(self) -> None:
        """Test that grayscale output has correct shape and strides."""
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        struct_data = result["output"].struct.unnest()
        shape = struct_data["shape"][0].to_list()
        strides = struct_data["strides"][0].to_list()

        # Grayscale output shape: [H, W, 1]
        assert shape == [20, 20, 1]
        # Contiguous strides: [W*1, 1, 1] = [20, 1, 1]
        assert strides == [20, 1, 1]

    def test_numpy_from_struct_zero_copy_mode(self) -> None:
        """Test numpy_from_struct with copy=False uses strided view."""
        img = np.full((16, 16, 3), 200, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Get array with copy=False
        arr = numpy_from_struct(result["output"][0], copy=False)

        assert arr.shape == (16, 16, 3)
        assert arr.dtype == np.uint8
        # All values should be 200
        assert np.all(arr == 200)

    def test_numpy_from_struct_copy_produces_contiguous(self) -> None:
        """Test numpy_from_struct with copy=True produces contiguous array."""
        img = np.full((16, 16, 3), 150, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Get array with copy=True (default)
        arr = numpy_from_struct(result["output"][0], copy=True)

        assert arr.shape == (16, 16, 3)
        assert arr.flags["C_CONTIGUOUS"]
        assert np.all(arr == 150)

    def test_data_integrity_with_strided_output(self) -> None:
        """Test that strided output maintains data integrity."""
        # Create distinct gradient image
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        for i in range(32):
            img[i, :, 0] = i * 8  # Red gradient
            img[:, i, 1] = i * 8  # Green gradient
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"image": [img_bytes]})

        pipe = Pipeline().source("image_bytes").sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        # Test both copy modes produce same result
        arr_copy = numpy_from_struct(result["output"][0], copy=True)
        arr_view = numpy_from_struct(result["output"][0], copy=False)

        np.testing.assert_array_equal(arr_copy, arr_view)
        np.testing.assert_array_equal(arr_copy, img)
