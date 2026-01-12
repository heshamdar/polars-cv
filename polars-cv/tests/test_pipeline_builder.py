"""
Unit tests for the Pipeline builder class.

These tests verify that the Pipeline class correctly builds and serializes
pipeline specifications without requiring the Rust plugin.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
import pytest
from polars_cv import Pipeline
from polars_cv._types import DType, SinkFormat, SourceFormat

if TYPE_CHECKING:
    pass


class TestPipelineSource:
    """Tests for Pipeline.source() method."""

    def test_source_default_format(self) -> None:
        """Test default source format is image_bytes."""
        pipe = Pipeline().source()
        assert pipe._source is not None
        assert pipe._source.format == SourceFormat.IMAGE_BYTES

    def test_source_image_bytes(self) -> None:
        """Test explicit image_bytes source format."""
        pipe = Pipeline().source("image_bytes")
        assert pipe._source is not None
        assert pipe._source.format == SourceFormat.IMAGE_BYTES

    def test_source_blob(self) -> None:
        """Test blob source format."""
        pipe = Pipeline().source("blob")
        assert pipe._source is not None
        assert pipe._source.format == SourceFormat.BLOB

    def test_source_raw_with_dtype(self) -> None:
        """Test raw source format with dtype."""
        pipe = Pipeline().source("raw", dtype="f32")
        assert pipe._source is not None
        assert pipe._source.format == SourceFormat.RAW
        assert pipe._source.dtype == DType.F32

    def test_source_raw_without_dtype_raises(self) -> None:
        """Test raw source format without dtype raises error."""
        with pytest.raises(ValueError, match="dtype is required"):
            Pipeline().source("raw")

    def test_source_invalid_format_raises(self) -> None:
        """Test invalid source format raises error."""
        with pytest.raises(ValueError, match="Invalid source format"):
            Pipeline().source("invalid_format")

    def test_source_file_path(self) -> None:
        """Test file_path source format."""
        pipe = Pipeline().source("file_path")
        assert pipe._source is not None
        assert pipe._source.format == SourceFormat.FILE_PATH


class TestPipelineSink:
    """Tests for Pipeline.sink() method."""

    def test_sink_numpy(self) -> None:
        """Test numpy sink format."""
        pipe = Pipeline().source().sink("numpy")
        assert pipe._sink is not None
        assert pipe._sink.format == SinkFormat.NUMPY

    def test_sink_torch(self) -> None:
        """Test torch sink format."""
        pipe = Pipeline().source().sink("torch")
        assert pipe._sink is not None
        assert pipe._sink.format == SinkFormat.TORCH

    def test_sink_png(self) -> None:
        """Test png sink format."""
        pipe = Pipeline().source().sink("png")
        assert pipe._sink is not None
        assert pipe._sink.format == SinkFormat.PNG

    def test_sink_jpeg_with_quality(self) -> None:
        """Test jpeg sink format with quality."""
        pipe = Pipeline().source().sink("jpeg", quality=90)
        assert pipe._sink is not None
        assert pipe._sink.format == SinkFormat.JPEG
        assert pipe._sink.quality == 90

    def test_sink_blob(self) -> None:
        """Test blob sink format."""
        pipe = Pipeline().source().sink("blob")
        assert pipe._sink is not None
        assert pipe._sink.format == SinkFormat.BLOB

    def test_sink_array_with_shape(self) -> None:
        """Test array sink format with shape."""
        pipe = Pipeline().source().sink("array", shape=[224, 224, 3])
        assert pipe._sink is not None
        assert pipe._sink.format == SinkFormat.ARRAY
        assert pipe._sink.shape == [224, 224, 3]

    def test_sink_array_without_shape_raises(self) -> None:
        """Test array sink format without shape raises error."""
        with pytest.raises(ValueError, match="shape is required"):
            Pipeline().source().sink("array")

    def test_sink_list(self) -> None:
        """Test list sink format."""
        pipe = Pipeline().source().sink("list")
        assert pipe._sink is not None
        assert pipe._sink.format == SinkFormat.LIST

    def test_sink_invalid_format_raises(self) -> None:
        """Test invalid sink format raises error."""
        with pytest.raises(ValueError, match="Invalid sink format"):
            Pipeline().source().sink("invalid_format")


class TestPipelineViewOps:
    """Tests for Pipeline view operations."""

    def test_transpose(self) -> None:
        """Test transpose operation."""
        pipe = Pipeline().source().transpose([1, 0, 2]).sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "transpose"
        assert pipe._ops[0].params["axes"].value == [1, 0, 2]

    def test_reshape(self) -> None:
        """Test reshape operation with literals."""
        pipe = Pipeline().source().reshape([100, 100, 3]).sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "reshape"

    def test_flip(self) -> None:
        """Test flip operation."""
        pipe = Pipeline().source().flip([0]).sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "flip"
        assert pipe._ops[0].params["axes"].value == [0]

    def test_flip_h(self) -> None:
        """Test horizontal flip convenience method."""
        pipe = Pipeline().source().flip_h().sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "flip"
        assert pipe._ops[0].params["axes"].value == [1]

    def test_flip_v(self) -> None:
        """Test vertical flip convenience method."""
        pipe = Pipeline().source().flip_v().sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "flip"
        assert pipe._ops[0].params["axes"].value == [0]

    def test_crop_with_literals(self) -> None:
        """Test crop operation with literal values."""
        pipe = (
            Pipeline()
            .source()
            .crop(top=10, left=20, height=100, width=100)
            .sink("numpy")
        )
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "crop"
        assert pipe._ops[0].params["top"].value == 10
        assert pipe._ops[0].params["left"].value == 20
        assert pipe._ops[0].params["height"].value == 100
        assert pipe._ops[0].params["width"].value == 100


class TestPipelineComputeOps:
    """Tests for Pipeline compute operations."""

    def test_cast(self) -> None:
        """Test cast operation."""
        pipe = Pipeline().source().cast("f32").sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "cast"
        assert pipe._ops[0].params["dtype"].value == "f32"

    def test_cast_invalid_dtype_raises(self) -> None:
        """Test cast with invalid dtype raises error."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            Pipeline().source().cast("invalid")

    def test_scale(self) -> None:
        """Test scale operation."""
        pipe = Pipeline().source().scale(2.5).sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "scale"
        assert pipe._ops[0].params["factor"].value == 2.5

    def test_normalize_minmax(self) -> None:
        """Test normalize operation with minmax."""
        pipe = Pipeline().source().normalize(method="minmax").sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "normalize"
        assert pipe._ops[0].params["method"].value == "minmax"

    def test_normalize_zscore(self) -> None:
        """Test normalize operation with zscore."""
        pipe = Pipeline().source().normalize(method="zscore").sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "normalize"
        assert pipe._ops[0].params["method"].value == "zscore"

    def test_normalize_invalid_method_raises(self) -> None:
        """Test normalize with invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid normalize method"):
            Pipeline().source().normalize(method="invalid")

    def test_clamp(self) -> None:
        """Test clamp operation."""
        pipe = Pipeline().source().clamp(0.0, 1.0).sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "clamp"
        assert pipe._ops[0].params["min"].value == 0.0
        assert pipe._ops[0].params["max"].value == 1.0


class TestPipelineImageOps:
    """Tests for Pipeline image operations."""

    def test_resize(self) -> None:
        """Test resize operation with literals."""
        pipe = Pipeline().source().resize(height=224, width=224).sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "resize"
        assert pipe._ops[0].params["height"].value == 224
        assert pipe._ops[0].params["width"].value == 224
        assert pipe._ops[0].params["filter"].value == "lanczos3"

    def test_resize_with_filter(self) -> None:
        """Test resize operation with custom filter."""
        pipe = (
            Pipeline()
            .source()
            .resize(height=100, width=100, filter="nearest")
            .sink("numpy")
        )
        assert pipe._ops[0].params["filter"].value == "nearest"

    def test_resize_invalid_filter_raises(self) -> None:
        """Test resize with invalid filter raises error."""
        with pytest.raises(ValueError, match="Invalid filter"):
            Pipeline().source().resize(height=100, width=100, filter="invalid")

    def test_grayscale(self) -> None:
        """Test grayscale operation."""
        pipe = Pipeline().source().grayscale().sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "grayscale"

    def test_threshold(self) -> None:
        """Test threshold operation."""
        pipe = Pipeline().source().threshold(128).sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "threshold"
        assert pipe._ops[0].params["value"].value == 128

    def test_blur(self) -> None:
        """Test blur operation."""
        pipe = Pipeline().source().blur(1.5).sink("numpy")
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "blur"
        assert pipe._ops[0].params["sigma"].value == 1.5


class TestPipelineShapeHints:
    """Tests for Pipeline.assert_shape() method."""

    def test_assert_shape_literals(self) -> None:
        """Test assert_shape with literal values."""
        pipe = (
            Pipeline()
            .source()
            .assert_shape(height=100, width=100, channels=3)
            .sink("numpy")
        )
        assert pipe._shape_hints.height is not None
        assert pipe._shape_hints.height.value == 100
        assert pipe._shape_hints.width is not None
        assert pipe._shape_hints.width.value == 100
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 3

    def test_assert_shape_partial(self) -> None:
        """Test assert_shape with partial values."""
        pipe = Pipeline().source().assert_shape(channels=3).sink("numpy")
        assert pipe._shape_hints.height is None
        assert pipe._shape_hints.width is None
        assert pipe._shape_hints.channels is not None
        assert pipe._shape_hints.channels.value == 3


class TestPipelineExpressionArgs:
    """Tests for Pipeline with Polars expression arguments."""

    def test_resize_with_expr(self) -> None:
        """Test resize with expression arguments."""
        pipe = (
            Pipeline()
            .source()
            .resize(height=pl.col("h"), width=pl.col("w"))
            .sink("numpy")
        )
        assert len(pipe._ops) == 1
        assert pipe._ops[0].params["height"].is_expr
        assert pipe._ops[0].params["width"].is_expr
        assert len(pipe._expr_refs) == 2

    def test_resize_mixed_literal_expr(self) -> None:
        """Test resize with mixed literal and expression."""
        pipe = Pipeline().source().resize(height=224, width=pl.col("w")).sink("numpy")
        assert not pipe._ops[0].params["height"].is_expr
        assert pipe._ops[0].params["width"].is_expr
        assert len(pipe._expr_refs) == 1

    def test_scale_with_expr(self) -> None:
        """Test scale with expression argument."""
        pipe = Pipeline().source().scale(pl.col("factor")).sink("numpy")
        assert pipe._ops[0].params["factor"].is_expr
        assert len(pipe._expr_refs) == 1

    def test_crop_with_expr(self) -> None:
        """Test crop with expression arguments."""
        pipe = (
            Pipeline()
            .source()
            .crop(top=pl.col("y"), left=pl.col("x"), height=100, width=100)
            .sink("numpy")
        )
        assert pipe._ops[0].params["top"].is_expr
        assert pipe._ops[0].params["left"].is_expr
        assert not pipe._ops[0].params["height"].is_expr
        assert len(pipe._expr_refs) == 2

    def test_assert_shape_with_expr(self) -> None:
        """Test assert_shape with expression argument."""
        pipe = (
            Pipeline()
            .source()
            .assert_shape(height=100, width=pl.col("w"))
            .sink("numpy")
        )
        assert not pipe._shape_hints.height.is_expr
        assert pipe._shape_hints.width.is_expr
        assert len(pipe._expr_refs) == 1

    def test_no_duplicate_expr_tracking(self) -> None:
        """Test that same expression is not tracked multiple times."""
        expr = pl.col("size")
        pipe = Pipeline().source().resize(height=expr, width=expr).sink("numpy")
        # Same expression object should only be tracked once
        assert len(pipe._expr_refs) == 1


class TestPipelineValidation:
    """Tests for Pipeline validation."""

    def test_validate_no_source_raises(self) -> None:
        """Test validation fails without source."""
        pipe = Pipeline().sink("numpy")
        with pytest.raises(ValueError, match="must have a source"):
            pipe.validate()

    def test_validate_no_sink_raises(self) -> None:
        """Test validation fails without sink."""
        pipe = Pipeline().source()
        with pytest.raises(ValueError, match="must have a sink"):
            pipe.validate()

    def test_validate_complete_pipeline(self) -> None:
        """Test validation passes for complete pipeline."""
        pipe = Pipeline().source().sink("numpy")
        pipe.validate()  # Should not raise


class TestPipelineSerialization:
    """Tests for Pipeline JSON serialization."""

    def test_serialize_simple_pipeline(self) -> None:
        """Test serialization of simple pipeline."""
        pipe = Pipeline().source().resize(height=224, width=224).sink("numpy")
        json_str = pipe._to_json()
        data = json.loads(json_str)

        assert data["source"]["format"] == "image_bytes"
        assert len(data["ops"]) == 1
        assert data["ops"][0]["op"] == "resize"
        assert data["sink"]["format"] == "numpy"

    def test_serialize_pipeline_with_shape_hints(self) -> None:
        """Test serialization includes shape hints."""
        pipe = Pipeline().source().assert_shape(channels=3).sink("numpy")
        json_str = pipe._to_json()
        data = json.loads(json_str)

        assert "shape_hints" in data
        assert data["shape_hints"]["channels"]["type"] == "literal"
        assert data["shape_hints"]["channels"]["value"] == 3

    def test_serialize_pipeline_with_expressions(self) -> None:
        """Test serialization of pipeline with expressions.

        Note: Expression parameters are serialized using their string representation
        as the 'col' key. This ensures unique identifiers even when multiple expressions
        share the same root column (e.g., col("x").max() and col("x").min()).
        """
        pipe = Pipeline().source().resize(height=pl.col("h"), width=224).sink("numpy")
        json_str = pipe._to_json()
        data = json.loads(json_str)

        height_param = data["ops"][0]["height"]
        width_param = data["ops"][0]["width"]

        assert height_param["type"] == "expr"
        # Expression identifier uses string representation for uniqueness
        assert height_param["col"] == 'col("h")'
        assert width_param["type"] == "literal"
        assert width_param["value"] == 224

    def test_roundtrip_json(self) -> None:
        """Test JSON can be parsed (simulating Rust-side)."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(height=100, width=100, channels=3)
            .resize(height=224, width=224)
            .grayscale()
            .normalize(method="minmax")
            .sink("numpy")
        )
        json_str = pipe._to_json()
        data = json.loads(json_str)

        # Verify structure
        assert data["source"]["format"] == "image_bytes"
        assert len(data["ops"]) == 3
        assert data["ops"][0]["op"] == "resize"
        assert data["ops"][1]["op"] == "grayscale"
        assert data["ops"][2]["op"] == "normalize"
        assert data["sink"]["format"] == "numpy"


class TestPipelineChaining:
    """Tests for Pipeline immutable chaining."""

    def test_operations_are_immutable(self) -> None:
        """Test that operations don't modify original pipeline."""
        base = Pipeline().source()
        with_resize = base.resize(height=224, width=224)
        with_grayscale = base.grayscale()

        assert len(base._ops) == 0
        assert len(with_resize._ops) == 1
        assert with_resize._ops[0].op == "resize"
        assert len(with_grayscale._ops) == 1
        assert with_grayscale._ops[0].op == "grayscale"

    def test_complex_chaining(self) -> None:
        """Test complex operation chaining."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=256, width=256)
            .crop(top=16, left=16, height=224, width=224)
            .flip_h()
            .grayscale()
            .cast("f32")
            .normalize(method="minmax")
            .sink("numpy")
        )

        assert len(pipe._ops) == 6
        assert pipe._ops[0].op == "resize"
        assert pipe._ops[1].op == "crop"
        assert pipe._ops[2].op == "flip"
        assert pipe._ops[3].op == "grayscale"
        assert pipe._ops[4].op == "cast"
        assert pipe._ops[5].op == "normalize"


class TestPipelineRepr:
    """Tests for Pipeline string representation."""

    def test_repr_empty(self) -> None:
        """Test repr of empty pipeline."""
        pipe = Pipeline()
        assert repr(pipe) == "Pipeline()"

    def test_repr_simple(self) -> None:
        """Test repr of simple pipeline."""
        pipe = Pipeline().source().sink("numpy")
        repr_str = repr(pipe)
        assert "source" in repr_str
        assert "sink" in repr_str
