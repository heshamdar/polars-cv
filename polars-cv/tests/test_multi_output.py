"""
Tests for multi-output pipeline support using LazyPipelineExpr composition.

This module tests the alias functionality and multi-output sink mode
using the LazyPipelineExpr composition pattern with .pipe() and .alias().
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct

if TYPE_CHECKING:
    pass


@pytest.fixture
def test_image_bytes() -> bytes:
    """Create a test image as bytes."""
    img = Image.new("RGB", (100, 100), (128, 64, 192))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def test_df(test_image_bytes: bytes) -> pl.DataFrame:
    """Create a test DataFrame with image bytes."""
    return pl.DataFrame({"image": [test_image_bytes]})


class TestLazyPipelineExprAlias:
    """Tests for LazyPipelineExpr.alias() method."""

    def test_alias_basic(self, test_df: pl.DataFrame) -> None:
        """Test basic alias creation on LazyPipelineExpr."""
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").resize(height=50, width=50))
            .alias("resized")
        )

        assert base.alias_name == "resized"

    def test_alias_chained_with_pipe(self, test_df: pl.DataFrame) -> None:
        """Test alias with .pipe() chaining."""
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").resize(height=50, width=50))
            .alias("resized")
        )

        gray = base.pipe(Pipeline().grayscale()).alias("gray")

        assert base.alias_name == "resized"
        assert gray.alias_name == "gray"

    def test_multiple_aliases_in_chain(self, test_df: pl.DataFrame) -> None:
        """Test multiple aliases in a chain using .pipe()."""
        base = (
            pl.col("image").cv.pipe(Pipeline().source("image_bytes")).alias("original")
        )

        resized = base.pipe(Pipeline().resize(height=50, width=50)).alias("resized")
        gray = resized.pipe(Pipeline().grayscale()).alias("gray")

        assert base.alias_name == "original"
        assert resized.alias_name == "resized"
        assert gray.alias_name == "gray"


class TestMultiOutputSink:
    """Tests for multi-output sink with aliases."""

    def test_single_sink_backward_compatible(self, test_df: pl.DataFrame) -> None:
        """Test that single format string still works."""
        base = pl.col("image").cv.pipe(
            Pipeline().source("image_bytes").resize(height=50, width=50)
        )

        result = test_df.with_columns(output=base.sink("numpy"))
        # Numpy sink now returns struct with data, dtype, shape fields
        assert isinstance(result["output"].dtype, pl.Struct), (
            f"Expected Struct for numpy sink, got {result['output'].dtype}"
        )

    def test_multi_sink_with_aliases(self, test_df: pl.DataFrame) -> None:
        """Test multi-output sink with aliased expressions."""
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").resize(height=50, width=50))
            .alias("resized")
        )

        gray = base.pipe(Pipeline().grayscale()).alias("gray")

        result = gray.sink({"resized": "numpy", "gray": "numpy"})
        df_result = test_df.with_columns(output=result)

        assert df_result["output"].dtype == pl.Struct
        fields = df_result["output"].struct.fields
        assert "resized" in fields
        assert "gray" in fields

    def test_multi_sink_undefined_alias_raises(self) -> None:
        """Test that undefined alias in sink raises error."""
        base = (
            pl.col("image").cv.pipe(Pipeline().source("image_bytes")).alias("defined")
        )

        with pytest.raises(ValueError, match="not found"):
            base.sink({"undefined": "numpy"})

    def test_multi_sink_three_outputs(self, test_df: pl.DataFrame) -> None:
        """Test multi-output with three aliases."""
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").resize(height=50, width=50))
            .alias("resized")
        )

        gray = base.pipe(Pipeline().grayscale()).alias("gray")
        thresh = gray.pipe(Pipeline().threshold(128)).alias("thresh")

        result = thresh.sink(
            {
                "resized": "numpy",
                "gray": "numpy",
                "thresh": "numpy",
            }
        )
        df_result = test_df.with_columns(output=result)

        fields = df_result["output"].struct.fields
        assert len(fields) == 3
        assert "resized" in fields
        assert "gray" in fields
        assert "thresh" in fields

        # Verify shapes
        resized = numpy_from_struct(df_result["output"].struct.field("resized")[0])
        gray_arr = numpy_from_struct(df_result["output"].struct.field("gray")[0])
        thresh_arr = numpy_from_struct(df_result["output"].struct.field("thresh")[0])

        assert resized.shape == (50, 50, 3)
        assert gray_arr.shape == (50, 50, 1)
        assert thresh_arr.shape == (50, 50, 1)


class TestMultiOutputFormats:
    """Tests for different output formats in multi-output mode."""

    def test_mixed_formats(self, test_df: pl.DataFrame) -> None:
        """Test that different formats can be mixed in multi-output."""
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").resize(height=50, width=50))
            .alias("numpy_out")
        )

        gray = base.pipe(Pipeline().grayscale()).alias("png_out")

        result = gray.sink(
            {
                "numpy_out": "numpy",
                "png_out": "png",
            }
        )
        df_result = test_df.with_columns(output=result)

        fields = df_result["output"].struct.fields
        assert "numpy_out" in fields
        assert "png_out" in fields

        # PNG should be a valid PNG image
        png_bytes = df_result["output"].struct.field("png_out")[0]
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


class TestBranchingPipelines:
    """Tests for branching pipelines using .pipe() and merge_pipe()."""

    def test_branch_from_base(self, test_df: pl.DataFrame) -> None:
        """Test branching from a base expression."""
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").resize(height=50, width=50))
            .alias("base")
        )

        # Two branches from base
        gray = base.pipe(Pipeline().grayscale()).alias("gray")
        blur = base.pipe(Pipeline().blur(3)).alias("blur")

        # Merge branches
        merged = gray.merge_pipe(blur)

        result = merged.sink(
            {
                "base": "numpy",
                "gray": "numpy",
                "blur": "numpy",
            }
        )
        df_result = test_df.with_columns(output=result)

        fields = df_result["output"].struct.fields
        assert "base" in fields
        assert "gray" in fields
        assert "blur" in fields

    def test_diamond_pattern(self, test_df: pl.DataFrame) -> None:
        """Test diamond pattern: base -> (branch1, branch2) -> merged."""
        base = (
            pl.col("image")
            .cv.pipe(Pipeline().source("image_bytes").resize(height=50, width=50))
            .alias("base")
        )

        gray = base.pipe(Pipeline().grayscale()).alias("gray")
        blur = base.pipe(Pipeline().blur(3)).alias("blur")

        # Both branches merge
        merged = gray.merge_pipe(blur)

        result = merged.sink(
            {
                "base": "numpy",
                "gray": "numpy",
                "blur": "numpy",
            }
        )
        df_result = test_df.with_columns(output=result)

        base_arr = numpy_from_struct(df_result["output"].struct.field("base")[0])
        gray_arr = numpy_from_struct(df_result["output"].struct.field("gray")[0])
        blur_arr = numpy_from_struct(df_result["output"].struct.field("blur")[0])

        assert base_arr.shape == (50, 50, 3)
        assert gray_arr.shape == (50, 50, 1)
        assert blur_arr.shape == (50, 50, 3)


class TestPipeMethod:
    """Tests specifically for the .pipe() method."""

    def test_pipe_without_source_continues(self, test_df: pl.DataFrame) -> None:
        """Test that .pipe() without source continues from upstream."""
        base = pl.col("image").cv.pipe(
            Pipeline().source("image_bytes").resize(height=50, width=50)
        )

        # No source = continuation
        gray = base.pipe(Pipeline().grayscale())

        result = test_df.with_columns(output=gray.sink("numpy"))
        arr = numpy_from_struct(result["output"][0])

        # Should be grayscale (1 channel)
        assert arr.shape == (50, 50, 1)

    def test_pipe_with_source_creates_new_root(self, test_df: pl.DataFrame) -> None:
        """Test that .pipe() with source creates a new root."""
        base = pl.col("image").cv.pipe(
            Pipeline().source("image_bytes").resize(height=50, width=50)
        )

        # With source = new root (ignores base's operations)
        new_root = base.pipe(Pipeline().source("image_bytes").grayscale())

        result = test_df.with_columns(output=new_root.sink("numpy"))
        arr = numpy_from_struct(result["output"][0])

        # Should be grayscale of original (100x100), not resized
        assert arr.shape == (100, 100, 1)

    def test_pipe_chains_multiple_operations(self, test_df: pl.DataFrame) -> None:
        """Test chaining multiple .pipe() calls."""
        base = pl.col("image").cv.pipe(Pipeline().source("image_bytes"))

        # Chain multiple operations
        result_expr = (
            base.pipe(Pipeline().resize(height=50, width=50))
            .pipe(Pipeline().grayscale())
            .pipe(Pipeline().threshold(128))
        )

        result = test_df.with_columns(output=result_expr.sink("numpy"))
        arr = numpy_from_struct(result["output"][0])

        assert arr.shape == (50, 50, 1)
        # Threshold should produce binary values
        assert np.all((arr == 0) | (arr == 255))
