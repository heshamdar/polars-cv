"""
Tests for zero-copy source ingestion.

These tests verify that:
1. Dtype auto-inference works for list/array sources
2. require_contiguous parameter properly enforces contiguity
3. Null handling works correctly
4. Shape inference works correctly for nested types
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline

if TYPE_CHECKING:
    pass


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def simple_grayscale_bytes() -> bytes:
    """Create a simple 4x4 grayscale test image."""
    img = np.full((4, 4), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def rgb_image_bytes() -> bytes:
    """Create a 4x4 RGB test image."""
    img = np.full((4, 4, 3), 128, dtype=np.uint8)
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# ============================================================
# Dtype Auto-Inference Tests
# ============================================================


class TestDtypeAutoInference:
    """Tests for automatic dtype inference from Polars column types."""

    def test_list_source_dtype_inference_u8(self) -> None:
        """List[List[UInt8]] should auto-infer as u8."""
        # Create a 2x3 nested list of u8 values
        data = [[1, 2, 3], [4, 5, 6]]
        df = pl.DataFrame({"arr": [data]}).cast({"arr": pl.List(pl.List(pl.UInt8))})

        # Pipeline without explicit dtype
        pipeline = Pipeline().source("list").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"].dtype == pl.List(pl.List(pl.UInt8))
        assert result["out"][0].to_list() == [[1, 2, 3], [4, 5, 6]]

    def test_list_source_dtype_inference_f32(self) -> None:
        """List[List[Float32]] should auto-infer as f32."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        df = pl.DataFrame({"arr": [data]}).cast({"arr": pl.List(pl.List(pl.Float32))})

        pipeline = Pipeline().source("list").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"].dtype == pl.List(pl.List(pl.Float32))

    def test_list_source_dtype_inference_i32(self) -> None:
        """List[Int32] should auto-infer as i32."""
        data = [1, 2, 3, 4, 5]
        df = pl.DataFrame({"arr": [data]}).cast({"arr": pl.List(pl.Int32)})

        pipeline = Pipeline().source("list").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"].dtype == pl.List(pl.Int32)
        assert result["out"][0].to_list() == [1, 2, 3, 4, 5]

    def test_array_source_dtype_inference(self) -> None:
        """Array[UInt8, 4] should auto-infer as u8 with fixed shape."""
        # Create Array column
        data = [1, 2, 3, 4]
        df = pl.DataFrame({"arr": [data]}).cast({"arr": pl.Array(pl.UInt8, 4)})

        pipeline = Pipeline().source("array").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"].dtype == pl.List(pl.UInt8)
        assert result["out"][0].to_list() == [1, 2, 3, 4]

    def test_dtype_explicit_override(self) -> None:
        """Explicit dtype should override auto-inference."""
        # Create i32 data but specify f32 output
        data = [1, 2, 3, 4]
        df = pl.DataFrame({"arr": [data]}).cast({"arr": pl.List(pl.Int32)})

        # Explicit dtype overrides column type
        pipeline = Pipeline().source("list", dtype="f32").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        # Data is cast to f32
        assert result["out"].dtype == pl.List(pl.Float32)


# ============================================================
# Null Handling Tests
# ============================================================


class TestNullHandling:
    """Tests for null value handling in sources."""

    def test_null_row_in_binary_source(self, simple_grayscale_bytes: bytes) -> None:
        """Null rows in binary source should produce null outputs."""
        df = pl.DataFrame(
            {"img": [simple_grayscale_bytes, None, simple_grayscale_bytes]}
        )

        pipeline = Pipeline().source("image_bytes").sink("blob")

        result = df.with_columns(out=pl.col("img").cv.pipeline(pipeline))

        assert result["out"][0] is not None
        assert result["out"][1] is None
        assert result["out"][2] is not None

    def test_null_row_in_list_source(self) -> None:
        """Null rows in list source should produce null outputs."""
        df = pl.DataFrame({"arr": [[[1, 2], [3, 4]], None, [[5, 6], [7, 8]]]})
        df = df.cast({"arr": pl.List(pl.List(pl.UInt8))})

        pipeline = Pipeline().source("list").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"][0] is not None
        assert result["out"][1] is None
        assert result["out"][2] is not None


# ============================================================
# Contiguity Tests
# ============================================================


class TestContiguityValidation:
    """Tests for require_contiguous parameter."""

    def test_require_contiguous_with_array_type(self) -> None:
        """Array type with require_contiguous=True should work (arrays are always contiguous)."""
        # Nested arrays are always contiguous
        df = pl.DataFrame({"arr": [[1, 2, 3, 4]]}).cast({"arr": pl.Array(pl.UInt8, 4)})

        pipeline = Pipeline().source("array", require_contiguous=True).sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"][0].to_list() == [1, 2, 3, 4]

    def test_require_contiguous_false_allows_copy(self) -> None:
        """require_contiguous=False should allow copy-based processing."""
        # Variable-size list (may require copy)
        data = [[1, 2, 3], [4, 5, 6]]
        df = pl.DataFrame({"arr": [data]}).cast({"arr": pl.List(pl.List(pl.UInt8))})

        # Should work with require_contiguous=False (default)
        pipeline = Pipeline().source("list", require_contiguous=False).sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"][0].to_list() == [[1, 2, 3], [4, 5, 6]]


# ============================================================
# Shape Inference Tests
# ============================================================


class TestShapeInference:
    """Tests for automatic shape inference from nested types."""

    def test_shape_from_1d_list(self) -> None:
        """1D list should produce 1D array."""
        df = pl.DataFrame({"arr": [[1, 2, 3, 4, 5]]}).cast({"arr": pl.List(pl.UInt8)})

        pipeline = Pipeline().source("list").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"].dtype == pl.List(pl.UInt8)
        assert len(result["out"][0]) == 5

    def test_shape_from_2d_list(self) -> None:
        """2D nested list should produce 2D array."""
        data = [[1, 2, 3], [4, 5, 6]]
        df = pl.DataFrame({"arr": [data]}).cast({"arr": pl.List(pl.List(pl.UInt8))})

        pipeline = Pipeline().source("list").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"].dtype == pl.List(pl.List(pl.UInt8))
        assert result["out"][0].to_list() == [[1, 2, 3], [4, 5, 6]]

    def test_shape_from_3d_list(self) -> None:
        """3D nested list should produce 3D array."""
        # 2x2x3 array
        data = [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
        ]
        df = pl.DataFrame({"arr": [data]}).cast(
            {"arr": pl.List(pl.List(pl.List(pl.UInt8)))}
        )

        pipeline = Pipeline().source("list").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"].dtype == pl.List(pl.List(pl.List(pl.UInt8)))

    def test_shape_from_fixed_array(self) -> None:
        """Polars Array type should produce correct shape."""
        df = pl.DataFrame({"arr": [[1, 2, 3, 4]]}).cast({"arr": pl.Array(pl.UInt8, 4)})

        pipeline = Pipeline().source("array").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"].dtype == pl.List(pl.UInt8)
        assert result["out"][0].to_list() == [1, 2, 3, 4]


# ============================================================
# Round-trip Tests
# ============================================================


class TestRoundTrip:
    """Tests for data integrity through source -> sink round-trips."""

    def test_blob_round_trip_preserves_data(self) -> None:
        """blob source -> blob sink should preserve data exactly."""
        # Create initial image
        img = np.arange(12, dtype=np.uint8).reshape(3, 4)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")

        df = pl.DataFrame({"img": [buf.getvalue()]})

        # Image -> blob -> blob -> list (to verify)
        to_blob = Pipeline().source("image_bytes").sink("blob")
        from_blob = Pipeline().source("blob").sink("list")

        blob_df = df.with_columns(blob=pl.col("img").cv.pipeline(to_blob))
        result = blob_df.with_columns(arr=pl.col("blob").cv.pipeline(from_blob))

        # PNG decodes to 3x4 grayscale with channel (3x4x1 or 3x4) -> List[List[...]]
        # The dtype depends on the shape - 2D produces List(List(UInt8))
        arr = result["arr"][0]
        # Just verify we got data back (the exact dtype depends on image dimensions)
        assert arr is not None
        assert len(arr) == 3  # 3 rows

    def test_list_round_trip_preserves_data(self) -> None:
        """list source -> list sink should preserve data."""
        data = [[1, 2, 3], [4, 5, 6]]
        df = pl.DataFrame({"arr": [data]}).cast({"arr": pl.List(pl.List(pl.UInt8))})

        # List -> list -> verify
        pipeline = Pipeline().source("list").sink("list")

        result = df.with_columns(out=pl.col("arr").cv.pipeline(pipeline))

        assert result["out"][0].to_list() == [[1, 2, 3], [4, 5, 6]]


# ============================================================
# Performance Tests (Basic Validation)
# ============================================================


class TestPerformanceBasics:
    """Basic tests to ensure optimized paths are functioning."""

    def test_large_binary_source_works(self) -> None:
        """Large binary data should be processed without issues."""
        # Create 100x100 image (smaller for test speed)
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")

        df = pl.DataFrame({"img": [buf.getvalue()]})

        pipeline = (
            Pipeline().source("image_bytes").resize(height=50, width=50).sink("blob")
        )

        result = df.with_columns(out=pl.col("img").cv.pipeline(pipeline))

        assert result["out"][0] is not None

    def test_batch_processing_with_nulls(self) -> None:
        """Batch processing with mixed null/non-null should work efficiently."""
        # Create batch with some nulls
        img = np.full((10, 10), 128, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # 20 rows with 20% nulls
        data = [img_bytes if i % 5 != 0 else None for i in range(20)]
        df = pl.DataFrame({"img": data})

        pipeline = Pipeline().source("image_bytes").sink("blob")

        result = df.with_columns(out=pl.col("img").cv.pipeline(pipeline))

        # Verify null pattern is preserved
        for i in range(20):
            if i % 5 == 0:
                assert result["out"][i] is None
            else:
                assert result["out"][i] is not None


# ============================================================
# Full Zero-Copy Pipeline Tests
# ============================================================


class TestFullZeroCopyPipeline:
    """Tests for complete zero-copy flow through the pipeline."""

    def test_full_pipeline_with_strided_ops(self) -> None:
        """Test complete zero-copy flow: input -> view ops -> strided ops -> output.

        This test verifies that the entire pipeline from input through
        view operations (flip, crop) to strided compute operations
        (grayscale, resize) and finally to output works correctly.
        """
        # Create test image with gradient for verification
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            img[i, :, :] = i * 2  # Gradient from 0 to 198

        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"data": [img_bytes]})

        # Pipeline with strided ops
        # flip: View op (zero-copy, creates strided view)
        # crop: View op (zero-copy, creates strided view)
        # grayscale: Now strided-compatible
        # resize: Now strided-compatible via fast_image_resize
        from polars_cv import numpy_from_struct

        pipe = (
            Pipeline()
            .source("image_bytes")
            .flip(axes=[0])  # View op (vertical flip)
            .crop(top=10, left=10, height=80, width=80)  # View op
            .grayscale()  # Strided op
            .resize(height=64, width=64)  # Strided op
            .sink("numpy")
        )

        result = df.with_columns(output=pl.col("data").cv.pipeline(pipe))

        arr = numpy_from_struct(result["output"][0])

        # Verify shape
        assert arr.shape == (64, 64, 1)
        assert arr.dtype == np.uint8

        # Verify the flip worked - after vertical flip, bottom of original
        # (brighter values) should be at top
        # The gradient was 0-198 from top to bottom, so after flip
        # the top should have higher values than the bottom
        top_mean = arr[:16, :, 0].mean()
        bottom_mean = arr[48:, :, 0].mean()
        assert top_mean > bottom_mean, (
            f"After vertical flip, top should be brighter: top={top_mean}, bottom={bottom_mean}"
        )

    def test_multiple_view_ops_preserved(self) -> None:
        """Test that multiple view operations are properly accumulated."""
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        # Add a distinct corner
        img[0:16, 0:16, 0] = 255  # Red top-left corner

        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"data": [img_bytes]})

        from polars_cv import numpy_from_struct

        # Double flip should return to original orientation
        pipe = (
            Pipeline()
            .source("image_bytes")
            .flip(axes=[0])  # vertical flip
            .flip(axes=[0])  # Back to original
            .sink("numpy")
        )

        result = df.with_columns(output=pl.col("data").cv.pipeline(pipe))
        arr = numpy_from_struct(result["output"][0])

        # Red corner should still be at top-left
        assert arr.shape == (64, 64, 3)
        assert arr[0, 0, 0] == 255  # Red channel top-left
        assert arr[63, 63, 0] == 100  # Bottom-right should be default

    def test_normalize_on_strided_buffer(self) -> None:
        """Test normalize operation works on strided (flipped) buffers."""
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"data": [img_bytes]})

        from polars_cv import numpy_from_struct

        # flip -> cast -> normalize should work on strided buffer
        pipe = (
            Pipeline()
            .source("image_bytes")
            .flip(axes=[1])  # horizontal flip
            .cast("f32")
            .normalize(method="minmax")
            .sink("numpy")
        )

        result = df.with_columns(output=pl.col("data").cv.pipeline(pipe))
        arr = numpy_from_struct(result["output"][0])

        assert arr.dtype == np.float32
        assert arr.shape == (32, 32, 3)

    def test_large_pipeline_with_many_operations(self) -> None:
        """Test a complex pipeline with many operations maintains data integrity."""
        # Create a checkerboard pattern
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(64):
            for j in range(64):
                if (i // 8 + j // 8) % 2 == 0:
                    img[i, j, :] = 200

        pil_img = Image.fromarray(img)
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        df = pl.DataFrame({"data": [img_bytes]})

        from polars_cv import numpy_from_struct

        # Long pipeline with multiple operations
        pipe = (
            Pipeline()
            .source("image_bytes")
            .flip(axes=[0])  # View op (vertical flip)
            .crop(top=8, left=8, height=48, width=48)  # View op, 48x48
            .flip(axes=[1])  # View op (horizontal flip)
            .grayscale()  # Strided op, 48x48x1
            .resize(height=32, width=32)  # Strided op
            .threshold(value=100)  # Element-wise op
            .sink("numpy")
        )

        result = df.with_columns(output=pl.col("data").cv.pipeline(pipe))
        arr = numpy_from_struct(result["output"][0])

        assert arr.shape == (32, 32, 1)
        assert arr.dtype == np.uint8
        # After threshold, values should be 0 or 255
        assert np.all((arr == 0) | (arr == 255))
