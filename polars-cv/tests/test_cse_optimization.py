"""
Tests for Common Subexpression Elimination (CSE) optimization.

CSE automatically detects and deduplicates common operation prefixes
across pipelines, reducing redundant computation.
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


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a DataFrame with a test image."""
    # Create a simple test image with distinct regions
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    # Add a white square in the center for contour detection
    for x in range(25, 75):
        for y in range(25, 75):
            img.putpixel((x, y), (255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    return pl.DataFrame({"image": [img_bytes]})


# ============================================================
# Test Classes
# ============================================================


class TestOpSpecEquality:
    """Tests for OpSpec equality comparison."""

    def test_identical_ops_are_equal(self) -> None:
        """Two OpSpecs with same op and params should be equal."""
        from polars_cv._types import OpSpec, ParamValue

        op1 = OpSpec(
            op="resize",
            params={
                "width": ParamValue(is_expr=False, value=100),
                "height": ParamValue(is_expr=False, value=100),
            },
        )
        op2 = OpSpec(
            op="resize",
            params={
                "width": ParamValue(is_expr=False, value=100),
                "height": ParamValue(is_expr=False, value=100),
            },
        )

        assert op1 == op2
        assert hash(op1) == hash(op2)

    def test_different_params_not_equal(self) -> None:
        """OpSpecs with different param values should not be equal."""
        from polars_cv._types import OpSpec, ParamValue

        op1 = OpSpec(
            op="resize",
            params={
                "width": ParamValue(is_expr=False, value=100),
                "height": ParamValue(is_expr=False, value=100),
            },
        )
        op2 = OpSpec(
            op="resize",
            params={
                "width": ParamValue(is_expr=False, value=200),
                "height": ParamValue(is_expr=False, value=100),
            },
        )

        assert op1 != op2

    def test_different_ops_not_equal(self) -> None:
        """OpSpecs with different operation names should not be equal."""
        from polars_cv._types import OpSpec, ParamValue

        op1 = OpSpec(op="grayscale", params={})
        op2 = OpSpec(
            op="threshold", params={"level": ParamValue(is_expr=False, value=128)}
        )

        assert op1 != op2

    def test_expression_params_equality(self) -> None:
        """OpSpecs with expression params compare by string representation."""
        from polars_cv._types import OpSpec, ParamValue

        op1 = OpSpec(
            op="resize",
            params={
                "width": ParamValue(is_expr=True, value=pl.col("w")),
                "height": ParamValue(is_expr=False, value=100),
            },
        )
        op2 = OpSpec(
            op="resize",
            params={
                "width": ParamValue(is_expr=True, value=pl.col("w")),
                "height": ParamValue(is_expr=False, value=100),
            },
        )

        assert op1 == op2

    def test_different_expression_params_not_equal(self) -> None:
        """OpSpecs with different expression column refs should not be equal."""
        from polars_cv._types import OpSpec, ParamValue

        op1 = OpSpec(
            op="resize",
            params={
                "width": ParamValue(is_expr=True, value=pl.col("w1")),
            },
        )
        op2 = OpSpec(
            op="resize",
            params={
                "width": ParamValue(is_expr=True, value=pl.col("w2")),
            },
        )

        assert op1 != op2


class TestCSEBasic:
    """Basic CSE optimization tests."""

    def test_shared_prefix_extracted(self, sample_df: pl.DataFrame) -> None:
        """Two pipelines with same prefix should share computation."""
        # Define two pipelines with common prefix: resize + grayscale
        gray_pipe = (
            Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        )

        threshold_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )

        # Create lazy expressions
        gray = pl.col("image").cv.pipe(gray_pipe).alias("gray")
        thresh = pl.col("image").cv.pipe(threshold_pipe).alias("thresh")

        # Compose and sink
        result_expr = gray.merge_pipe(thresh).sink(
            {
                "gray": "numpy",
                "thresh": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        # Both outputs should be present and valid
        gray_arr = numpy_from_struct(result["outputs"].struct.field("gray")[0])
        thresh_arr = numpy_from_struct(result["outputs"].struct.field("thresh")[0])

        assert gray_arr.shape == (50, 50, 1)
        assert thresh_arr.shape == (50, 50, 1)

    def test_no_sharing_when_prefix_differs(self, sample_df: pl.DataFrame) -> None:
        """Pipelines with different prefixes should not share."""
        # Different resize dimensions = no common prefix
        pipe1 = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        pipe2 = (
            Pipeline().source("image_bytes").resize(width=100, height=100).grayscale()
        )

        expr1 = pl.col("image").cv.pipe(pipe1).alias("small")
        expr2 = pl.col("image").cv.pipe(pipe2).alias("large")

        result_expr = expr1.merge_pipe(expr2).sink(
            {
                "small": "numpy",
                "large": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        small = numpy_from_struct(result["outputs"].struct.field("small")[0])
        large = numpy_from_struct(result["outputs"].struct.field("large")[0])

        # Both should work, just with different sizes
        assert small.shape == (50, 50, 1)
        assert large.shape == (100, 100, 1)

    def test_partial_prefix_sharing(self, sample_df: pl.DataFrame) -> None:
        """Pipelines that diverge mid-way should share the common part."""
        # Common: resize + grayscale
        # Diverge: threshold vs blur
        thresh_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )

        blur_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .blur(3)
        )

        thresh = pl.col("image").cv.pipe(thresh_pipe).alias("thresh")
        blur = pl.col("image").cv.pipe(blur_pipe).alias("blur")

        result_expr = thresh.merge_pipe(blur).sink(
            {
                "thresh": "numpy",
                "blur": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        thresh_arr = numpy_from_struct(result["outputs"].struct.field("thresh")[0])
        blur_arr = numpy_from_struct(result["outputs"].struct.field("blur")[0])

        assert thresh_arr.shape == (50, 50, 1)
        assert blur_arr.shape == (50, 50, 1)

        # Threshold and blur should produce different results
        # (threshold is binary, blur is smooth)
        assert not np.array_equal(thresh_arr, blur_arr)


class TestCSEMultiplePipelines:
    """Tests for CSE with more than two pipelines."""

    def test_three_pipelines_share_prefix(self, sample_df: pl.DataFrame) -> None:
        """Three pipelines with common prefix should all share."""
        # All three pipes share: source → resize → grayscale
        pipe_a = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(64)
        )
        pipe_b = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )
        pipe_c = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(192)
        )

        a = pl.col("image").cv.pipe(pipe_a).alias("low")
        b = pl.col("image").cv.pipe(pipe_b).alias("mid")
        c = pl.col("image").cv.pipe(pipe_c).alias("high")

        result_expr = a.merge_pipe(b, c).sink(
            {
                "low": "numpy",
                "mid": "numpy",
                "high": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        low = numpy_from_struct(result["outputs"].struct.field("low")[0])
        mid = numpy_from_struct(result["outputs"].struct.field("mid")[0])
        high = numpy_from_struct(result["outputs"].struct.field("high")[0])

        # All should be valid 50x50 grayscale images
        assert low.shape == (50, 50, 1)
        assert mid.shape == (50, 50, 1)
        assert high.shape == (50, 50, 1)

        # Lower threshold = more white pixels
        assert np.sum(low) >= np.sum(mid) >= np.sum(high)


class TestCSEEdgeCases:
    """Edge cases for CSE optimization."""

    def test_single_pipeline_no_optimization(self, sample_df: pl.DataFrame) -> None:
        """Single pipeline should work without any optimization."""
        pipe = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()

        result = sample_df.with_columns(
            output=pl.col("image").cv.pipeline(pipe.sink("numpy"))
        )

        arr = numpy_from_struct(result["output"][0])
        assert arr.shape == (50, 50, 1)

    def test_identical_pipelines_fully_shared(self, sample_df: pl.DataFrame) -> None:
        """Two identical pipelines should fully share (prefix = entire pipeline)."""
        pipe1 = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        pipe2 = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()

        expr1 = pl.col("image").cv.pipe(pipe1).alias("a")
        expr2 = pl.col("image").cv.pipe(pipe2).alias("b")

        result_expr = expr1.merge_pipe(expr2).sink(
            {
                "a": "numpy",
                "b": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        a = numpy_from_struct(result["outputs"].struct.field("a")[0])
        b = numpy_from_struct(result["outputs"].struct.field("b")[0])

        # Both should be identical
        assert np.array_equal(a, b)

    def test_empty_prefix_no_sharing(self, sample_df: pl.DataFrame) -> None:
        """Pipelines with no common ops should not crash."""
        pipe1 = Pipeline().source("image_bytes").grayscale()
        pipe2 = Pipeline().source("image_bytes").resize(width=50, height=50)

        expr1 = pl.col("image").cv.pipe(pipe1).alias("gray")
        expr2 = pl.col("image").cv.pipe(pipe2).alias("resized")

        result_expr = expr1.merge_pipe(expr2).sink(
            {
                "gray": "numpy",
                "resized": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        gray = numpy_from_struct(result["outputs"].struct.field("gray")[0])
        resized = numpy_from_struct(result["outputs"].struct.field("resized")[0])

        # Gray should preserve original dimensions (100x100)
        assert gray.shape == (100, 100, 1)
        # Resized should be 50x50 RGB
        assert resized.shape == (50, 50, 3)

    def test_aliases_preserved_after_optimization(
        self, sample_df: pl.DataFrame
    ) -> None:
        """User aliases should be preserved through CSE optimization."""
        pipe1 = Pipeline().source("image_bytes").resize(width=50, height=50).grayscale()
        pipe2 = (
            Pipeline()
            .source("image_bytes")
            .resize(width=50, height=50)
            .grayscale()
            .threshold(128)
        )

        # Both have meaningful aliases
        base = pl.col("image").cv.pipe(pipe1).alias("preprocessed")
        mask = pl.col("image").cv.pipe(pipe2).alias("binary_mask")

        result_expr = base.merge_pipe(mask).sink(
            {
                "preprocessed": "numpy",
                "binary_mask": "numpy",
            }
        )

        result = sample_df.with_columns(outputs=result_expr)

        # Aliases should be preserved in output
        assert "preprocessed" in result["outputs"].struct.fields
        assert "binary_mask" in result["outputs"].struct.fields
