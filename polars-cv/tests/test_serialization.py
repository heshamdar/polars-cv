"""
Tests for JSON serialization and deserialization.

These tests verify that pipeline specifications can be correctly
serialized to JSON and would be deserializable by the Rust plugin.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
from polars_cv import Pipeline
from polars_cv._types import ParamValue

if TYPE_CHECKING:
    pass


class TestParamValueSerialization:
    """Tests for ParamValue serialization."""

    def test_literal_int_serialization(self) -> None:
        """Test integer literal serialization."""
        param = ParamValue.from_arg(42)
        d = param.to_dict()
        assert d["type"] == "literal"
        assert d["value"] == 42

    def test_literal_float_serialization(self) -> None:
        """Test float literal serialization."""
        param = ParamValue.from_arg(3.14)
        d = param.to_dict()
        assert d["type"] == "literal"
        assert d["value"] == 3.14

    def test_literal_string_serialization(self) -> None:
        """Test string literal serialization."""
        param = ParamValue.from_arg("hello")
        d = param.to_dict()
        assert d["type"] == "literal"
        assert d["value"] == "hello"

    def test_expr_column_serialization(self) -> None:
        """Test expression column serialization.

        Note: Expression identifier uses string representation for uniqueness,
        ensuring multiple expressions with the same root column are distinguished
        (e.g., col("x").max() vs col("x").min()).
        """
        param = ParamValue.from_arg(pl.col("my_column"))
        d = param.to_dict()
        assert d["type"] == "expr"
        # Uses string representation as identifier
        assert d["col"] == 'col("my_column")'

    def test_param_is_json_serializable(self) -> None:
        """Test that serialized params are JSON-serializable."""
        param = ParamValue.from_arg(pl.col("test"))
        d = param.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


class TestPipelineJsonFormat:
    """Tests for complete pipeline JSON format."""

    def test_json_has_required_fields(self) -> None:
        """Test JSON contains all required fields."""
        pipe = Pipeline().source().sink("numpy")
        data = json.loads(pipe._to_json())

        assert "source" in data
        assert "ops" in data
        assert "sink" in data

    def test_source_spec_format(self) -> None:
        """Test source specification format."""
        pipe = Pipeline().source("blob").sink("numpy")
        data = json.loads(pipe._to_json())

        assert data["source"]["format"] == "blob"

    def test_source_spec_with_dtype(self) -> None:
        """Test source specification with dtype."""
        pipe = Pipeline().source("raw", dtype="f32").sink("numpy")
        data = json.loads(pipe._to_json())

        assert data["source"]["format"] == "raw"
        assert data["source"]["dtype"] == "f32"

    def test_sink_spec_format(self) -> None:
        """Test sink specification format."""
        pipe = Pipeline().source().sink("torch")
        data = json.loads(pipe._to_json())

        assert data["sink"]["format"] == "torch"

    def test_sink_spec_jpeg_quality(self) -> None:
        """Test sink specification with JPEG quality."""
        pipe = Pipeline().source().sink("jpeg", quality=75)
        data = json.loads(pipe._to_json())

        assert data["sink"]["format"] == "jpeg"
        assert data["sink"]["quality"] == 75

    def test_sink_spec_array_shape(self) -> None:
        """Test sink specification with array shape."""
        pipe = Pipeline().source().sink("array", shape=[224, 224, 3])
        data = json.loads(pipe._to_json())

        assert data["sink"]["format"] == "array"
        assert data["sink"]["shape"] == [224, 224, 3]

    def test_ops_format(self) -> None:
        """Test operations format in JSON."""
        pipe = Pipeline().source().resize(height=224, width=224).sink("numpy")
        data = json.loads(pipe._to_json())

        assert len(data["ops"]) == 1
        op = data["ops"][0]
        assert op["op"] == "resize"
        assert "height" in op
        assert "width" in op
        assert "filter" in op

    def test_shape_hints_format(self) -> None:
        """Test shape hints format in JSON."""
        pipe = Pipeline().source().assert_shape(height=100, width=200).sink("numpy")
        data = json.loads(pipe._to_json())

        assert "shape_hints" in data
        hints = data["shape_hints"]
        assert hints["height"]["type"] == "literal"
        assert hints["height"]["value"] == 100
        assert hints["width"]["type"] == "literal"
        assert hints["width"]["value"] == 200

    def test_shape_hints_omitted_when_empty(self) -> None:
        """Test shape hints are omitted when not set."""
        pipe = Pipeline().source().sink("numpy")
        data = json.loads(pipe._to_json())

        # Should not have shape_hints key if none set
        assert "shape_hints" not in data


class TestExpressionReferencesJson:
    """Tests for expression reference serialization."""

    def test_simple_column_reference(self) -> None:
        """Test simple column reference serialization.

        Expression identifier uses string representation for uniqueness.
        """
        pipe = Pipeline().source().resize(height=pl.col("h"), width=100).sink("numpy")
        data = json.loads(pipe._to_json())

        height_param = data["ops"][0]["height"]
        assert height_param["type"] == "expr"
        assert height_param["col"] == 'col("h")'

    def test_multiple_column_references(self) -> None:
        """Test multiple column references.

        Expression identifier uses string representation for uniqueness.
        """
        pipe = (
            Pipeline()
            .source()
            .resize(height=pl.col("h"), width=pl.col("w"))
            .crop(top=pl.col("y"), left=pl.col("x"))
            .sink("numpy")
        )
        data = json.loads(pipe._to_json())

        # resize
        assert data["ops"][0]["height"]["col"] == 'col("h")'
        assert data["ops"][0]["width"]["col"] == 'col("w")'

        # crop
        assert data["ops"][1]["top"]["col"] == 'col("y")'
        assert data["ops"][1]["left"]["col"] == 'col("x")'

    def test_expr_columns_tracking(self) -> None:
        """Test _get_expr_columns returns correct expressions."""
        pipe = (
            Pipeline()
            .source()
            .resize(height=pl.col("h"), width=pl.col("w"))
            .sink("numpy")
        )

        expr_columns = pipe._get_expr_columns()
        assert len(expr_columns) == 2

        # Check that the column names match
        col_names = set()
        for expr in expr_columns:
            root_names = expr.meta.root_names()
            col_names.update(root_names)

        assert "h" in col_names
        assert "w" in col_names


class TestJsonRustCompatibility:
    """Tests ensuring JSON is compatible with Rust deserializer."""

    def test_all_literal_types(self) -> None:
        """Test all literal types serialize correctly."""
        pipe = (
            Pipeline()
            .source()
            .resize(height=224, width=224, filter="lanczos3")  # int, int, string
            .scale(2.5)  # float
            .clamp(0.0, 1.0)  # float, float
            .sink("numpy")
        )
        json_str = pipe._to_json()

        # Should be valid JSON
        data = json.loads(json_str)

        # Verify types are preserved
        assert isinstance(data["ops"][0]["height"]["value"], int)
        assert isinstance(data["ops"][0]["filter"]["value"], str)
        assert isinstance(data["ops"][1]["factor"]["value"], float)

    def test_nested_param_list_for_reshape(self) -> None:
        """Test nested param list for reshape operation."""
        pipe = Pipeline().source().reshape([100, 100, 3]).sink("numpy")
        data = json.loads(pipe._to_json())

        # Reshape has a shape param that's a list of ParamValues
        shape_param = data["ops"][0]["shape"]
        assert shape_param["type"] == "literal"
        # The value is a list of serialized ParamValues
        shape_list = shape_param["value"]
        assert len(shape_list) == 3

    def test_flip_axes_list(self) -> None:
        """Test flip axes is serialized as int list."""
        pipe = Pipeline().source().flip([0, 1]).sink("numpy")
        data = json.loads(pipe._to_json())

        axes = data["ops"][0]["axes"]
        assert axes["type"] == "literal"
        assert axes["value"] == [0, 1]

    def test_transpose_axes_list(self) -> None:
        """Test transpose axes is serialized as int list."""
        pipe = Pipeline().source().transpose([2, 0, 1]).sink("numpy")
        data = json.loads(pipe._to_json())

        axes = data["ops"][0]["axes"]
        assert axes["type"] == "literal"
        assert axes["value"] == [2, 0, 1]


class TestComplexPipelineJson:
    """Tests for complex pipeline serialization."""

    def test_full_pipeline(self) -> None:
        """Test serialization of full pipeline."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .assert_shape(height=256, width=256, channels=3)
            .resize(height=224, width=224, filter="bilinear")
            .flip_v()
            .grayscale()
            .cast("f32")
            .scale(1.0 / 255.0)
            .normalize(method="minmax")
            .sink("numpy")
        )

        json_str = pipe._to_json()
        data = json.loads(json_str)

        # Verify structure
        assert data["source"]["format"] == "image_bytes"
        assert "shape_hints" in data
        assert len(data["ops"]) == 6
        assert data["sink"]["format"] == "numpy"

    def test_dynamic_pipeline(self) -> None:
        """Test serialization of pipeline with dynamic params."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("target_h"), width=pl.col("target_w"))
            .crop(top=pl.col("crop_y"), left=pl.col("crop_x"), height=100, width=100)
            .threshold(pl.col("threshold_val"))
            .sink("numpy")
        )

        json_str = pipe._to_json()
        data = json.loads(json_str)

        # Check dynamic params are expr type
        assert data["ops"][0]["height"]["type"] == "expr"
        assert data["ops"][0]["width"]["type"] == "expr"
        assert data["ops"][1]["top"]["type"] == "expr"
        assert data["ops"][1]["left"]["type"] == "expr"
        assert data["ops"][1]["height"]["type"] == "literal"  # Still literal
        assert data["ops"][2]["value"]["type"] == "expr"
