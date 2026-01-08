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

from polars_cv import (
    NUMPY_OUTPUT_SCHEMA,
    Pipeline,
    numpy_from_struct,
)

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
        assert NUMPY_OUTPUT_SCHEMA == pl.Struct({
            "data": pl.Binary,
            "dtype": pl.String,
            "shape": pl.List(pl.UInt64),
        })

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

    def test_null_input_produces_null_struct_fields(self, simple_rgb_bytes: bytes) -> None:
        """Null input should produce struct with null fields."""
        df = pl.DataFrame({
            "image": [simple_rgb_bytes, None, simple_rgb_bytes]
        })

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
        df = pl.DataFrame({
            "image": [None]
        }).cast({"image": pl.Binary})

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
        df = pl.DataFrame({
            "image": [simple_rgb_bytes] * 10
        })

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

        row = result["output"][0]
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

        pipe = Pipeline().source("image_bytes").resize(height=10, width=20).sink("numpy")
        result = df.with_columns(output=pl.col("image").cv.pipeline(pipe))

        unnested = result["output"].struct.unnest()
        shape = unnested["shape"][0].to_list()

        # Resize(height=10, width=20)
        assert shape[0] == 10  # Height
        assert shape[1] == 20  # Width
        assert shape[2] == 3   # Channels
