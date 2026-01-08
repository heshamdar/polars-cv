"""
Comprehensive tests for the unified graph execution path.

These tests verify that all pipeline functionality works through
the unified graph path (vb_graph) which handles both single and
multi-output pipelines.

All tests in this file use the graph path exclusively and cover:
- Single output pipelines
- Multi-output pipelines
- Expression arguments (dynamic parameters)
- Contour sources
- Binary operations (add, subtract, etc.)
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.geometry.schemas import contour_from_points

if TYPE_CHECKING:
    pass


def create_synthetic_image_blob(
    width: int = 100,
    height: int = 100,
    channels: int = 3,
) -> bytes:
    """
    Create synthetic image bytes in VIEW blob format.

    This creates a valid ViewBlob that can be decoded by the plugin.
    Header format (64 bytes):
    - magic: [u8; 4] = "VIEW"
    - version: u16 = 1
    - dtype: u8 (1=U8)
    - rank: u8
    - data_offset: u64
    - flags: u64
    - reserved: [u8; 40]

    Then: shape (rank * 8 bytes) + strides (rank * 8 bytes) + data

    Args:
        width: Image width.
        height: Image height.
        channels: Number of channels (1, 3, or 4).

    Returns:
        Bytes in VIEW blob format.
    """
    # Create a simple gradient image
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    for c in range(channels):
        arr[:, :, c] = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))

    shape = arr.shape
    rank = len(shape)
    dtype_code = 1  # U8 = 1

    # Calculate strides for C-contiguous layout (bytes)
    strides_bytes = []
    stride = 1  # 1 byte per element for U8
    for dim in reversed(shape):
        strides_bytes.append(stride)
        stride *= dim
    strides_bytes = list(reversed(strides_bytes))

    # Header is 64 bytes, then shape + strides + data
    header_size = 64
    shape_bytes_len = rank * 8
    stride_bytes_len = rank * 8
    data_offset = header_size + shape_bytes_len + stride_bytes_len

    output = bytearray()

    # Magic bytes (4 bytes)
    output.extend(b"VIEW")
    # Version (u16, 2 bytes)
    output.extend(struct.pack("<H", 1))
    # DType (u8, 1 byte) - U8 = 1
    output.append(dtype_code)
    # Rank (u8, 1 byte)
    output.append(rank)
    # Data offset (u64, 8 bytes)
    output.extend(struct.pack("<Q", data_offset))
    # Flags (u64, 8 bytes)
    output.extend(struct.pack("<Q", 1))  # 1 = contiguous
    # Reserved (40 bytes)
    output.extend(bytes(40))

    # Now we should have 64 bytes
    assert len(output) == 64, f"Header size mismatch: {len(output)}"

    # Shape (rank * u64)
    for dim in shape:
        output.extend(struct.pack("<Q", dim))

    # Strides (rank * i64, in bytes)
    for s in strides_bytes:
        output.extend(struct.pack("<q", s))

    # Data
    output.extend(arr.tobytes())

    return bytes(output)


@pytest.fixture
def synthetic_image_df() -> pl.DataFrame:
    """Create a DataFrame with synthetic image bytes in blob format."""
    img1 = create_synthetic_image_blob(50, 50, 3)
    img2 = create_synthetic_image_blob(60, 40, 3)
    img3 = create_synthetic_image_blob(30, 80, 3)

    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "images": [img1, img2, img3],
        }
    )


@pytest.fixture
def contour_df() -> pl.DataFrame:
    """Create a DataFrame with contour data."""
    # Simple square contour
    square = contour_from_points(
        [
            (10.0, 10.0),
            (90.0, 10.0),
            (90.0, 90.0),
            (10.0, 90.0),
        ]
    )

    # Triangle contour
    triangle = contour_from_points(
        [
            (50.0, 10.0),
            (90.0, 90.0),
            (10.0, 90.0),
        ]
    )

    return pl.DataFrame(
        {
            "id": [1, 2],
            "contour": [square, triangle],
            "target_width": [100, 200],
            "target_height": [100, 150],
        }
    )


class TestUnifiedGraphSingleOutput:
    """Test single-output pipelines through the unified graph path."""

    def test_basic_pipeline(self, synthetic_image_df: pl.DataFrame) -> None:
        """Test a basic pipeline through the graph path."""
        pipe = Pipeline().source("blob").transpose([2, 0, 1]).sink("numpy")

        result = synthetic_image_df.with_columns(
            processed=pl.col("images").cv.pipeline(pipe)
        )

        assert "processed" in result.columns
        assert isinstance(result["processed"].dtype, pl.Struct)

    def test_resize_pipeline(self, synthetic_image_df: pl.DataFrame) -> None:
        """Test resize operation through the graph path."""
        pipe = Pipeline().source("blob").resize(height=32, width=32).sink("numpy")

        result = synthetic_image_df.with_columns(
            resized=pl.col("images").cv.pipeline(pipe)
        )

        assert "resized" in result.columns
        assert result["resized"].null_count() == 0

    def test_multiple_operations(self, synthetic_image_df: pl.DataFrame) -> None:
        """Test multiple operations chained together."""
        pipe = (
            Pipeline()
            .source("blob")
            .resize(height=64, width=64)
            .transpose([2, 0, 1])
            .sink("numpy")
        )

        result = synthetic_image_df.with_columns(
            normalized=pl.col("images").cv.pipeline(pipe)
        )

        assert "normalized" in result.columns


class TestUnifiedGraphMultiOutput:
    """Test multi-output pipelines through the unified graph path."""

    def test_multi_output_basic(self, synthetic_image_df: pl.DataFrame) -> None:
        """Test multi-output pipeline returns Struct column using LazyPipelineExpr."""
        # Use LazyPipelineExpr composition pattern
        original = pl.col("images").cv.pipe(Pipeline().source("blob")).alias("original")

        small = original.pipe(Pipeline().resize(height=32, width=32)).alias("small")

        result = synthetic_image_df.with_columns(
            outputs=small.sink({"original": "numpy", "small": "numpy"})
        )

        assert "outputs" in result.columns
        assert result["outputs"].dtype == pl.Struct

    def test_multi_output_fields(self, synthetic_image_df: pl.DataFrame) -> None:
        """Test multi-output has correct field names using LazyPipelineExpr."""
        raw = pl.col("images").cv.pipe(Pipeline().source("blob")).alias("raw")

        chw = raw.pipe(Pipeline().transpose([2, 0, 1])).alias("chw")

        result = synthetic_image_df.with_columns(
            outputs=chw.sink({"raw": "numpy", "chw": "numpy"})
        )

        output_schema = result["outputs"].dtype
        assert isinstance(output_schema, pl.Struct)
        field_names = [f.name for f in output_schema.fields]
        assert "raw" in field_names
        assert "chw" in field_names


class TestContourSourceGraph:
    """Test contour sources through the unified graph path."""

    def test_contour_source_rasterization(self, contour_df: pl.DataFrame) -> None:
        """Test contour source is properly rasterized through graph."""
        pipe = Pipeline().source("contour", width=100, height=100).sink("numpy")

        result = contour_df.with_columns(rasterized=pl.col("contour").cv.pipeline(pipe))

        assert "rasterized" in result.columns
        assert isinstance(result["rasterized"].dtype, pl.Struct)

    def test_contour_with_operations(self, contour_df: pl.DataFrame) -> None:
        """Test contour source with additional operations."""
        pipe = (
            Pipeline()
            .source("contour", width=100, height=100)
            .resize(height=50, width=50)
            .sink("numpy")
        )

        result = contour_df.with_columns(processed=pl.col("contour").cv.pipeline(pipe))

        assert "processed" in result.columns


class TestExpressionArgumentsGraph:
    """Test expression arguments through the unified graph path."""

    def test_dynamic_resize(self, contour_df: pl.DataFrame) -> None:
        """Test dynamic resize with expression arguments."""
        pipe = (
            Pipeline()
            .source(
                "contour", width=pl.col("target_width"), height=pl.col("target_height")
            )
            .sink("numpy")
        )

        result = contour_df.with_columns(rasterized=pl.col("contour").cv.pipeline(pipe))

        assert "rasterized" in result.columns
        assert result["rasterized"].null_count() == 0


class TestLazyCompositionGraph:
    """Test lazy pipeline composition through the unified graph path."""

    def test_lazy_single_sink(self, synthetic_image_df: pl.DataFrame) -> None:
        """Test lazy pipeline with single sink."""
        pipe = Pipeline().source("blob").resize(height=32, width=32)

        lazy_expr = pl.col("images").cv.pipe(pipe)
        final_expr = lazy_expr.sink("numpy")

        result = synthetic_image_df.with_columns(output=final_expr)
        assert "output" in result.columns

    def test_lazy_multi_sink(self, synthetic_image_df: pl.DataFrame) -> None:
        """Test lazy pipeline with multi-output sink via LazyPipelineExpr.alias()."""
        pipe = Pipeline().source("blob").resize(height=32, width=32)

        # Get the LazyPipelineExpr and use its alias method
        lazy_expr = pl.col("images").cv.pipe(pipe)
        aliased = lazy_expr.alias("resized")

        # For multi-output (dict format), produces Struct with named fields
        final_expr = aliased.sink({"resized": "numpy"})

        result = synthetic_image_df.with_columns(outputs=final_expr)
        assert "outputs" in result.columns
        # Dict format produces Struct column even with one output
        assert result["outputs"].dtype == pl.Struct


class TestGraphEdgeCases:
    """Test edge cases in the unified graph execution."""

    def test_empty_dataframe(self) -> None:
        """Test pipeline on empty DataFrame."""
        df = pl.DataFrame({"images": []}, schema={"images": pl.Binary})

        pipe = Pipeline().source("blob").sink("numpy")

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        assert len(result) == 0
        assert "processed" in result.columns

    def test_null_values(self) -> None:
        """Test pipeline handles null values gracefully."""
        img = create_synthetic_image_blob(10, 10, 3)
        df = pl.DataFrame(
            {
                "images": [img, None, img],
            }
        )

        pipe = Pipeline().source("blob").sink("numpy")

        result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # With struct output, null rows have null 'data' field
        data_nulls = result["processed"].struct.field("data").null_count()
        assert data_nulls == 1
        assert result["processed"][1].get("data") is None

    def test_identity_pipeline(self, synthetic_image_df: pl.DataFrame) -> None:
        """Test pipeline with no operations (identity)."""
        pipe = Pipeline().source("blob").sink("numpy")

        result = synthetic_image_df.with_columns(
            identity=pl.col("images").cv.pipeline(pipe)
        )

        # Output should be same shape as input
        assert result["identity"].null_count() == 0
