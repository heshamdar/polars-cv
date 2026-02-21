"""Tests for sinkless pipeline JSON serialization."""

from __future__ import annotations

import json

import polars as pl
from polars_cv import Pipeline
from polars_cv._types import ParamValue


class TestParamValueSerialization:
    """Tests for ParamValue serialization."""

    def test_literal_int_serialization(self) -> None:
        """Integer literals serialize correctly."""
        d = ParamValue.from_arg(42).to_dict()
        assert d["type"] == "literal"
        assert d["value"] == 42

    def test_expr_column_serialization(self) -> None:
        """Expression columns serialize as expr param refs."""
        d = ParamValue.from_arg(pl.col("my_column")).to_dict()
        assert d["type"] == "expr"
        assert d["col"] == 'col("my_column")'


class TestPipelineJsonFormat:
    """Tests for linear pipeline JSON output format."""

    def test_json_has_required_fields(self) -> None:
        """Serialized pipelines contain source+ops fields."""
        data = json.loads(Pipeline().source()._to_json())
        assert "source" in data
        assert "ops" in data
        assert "sink" not in data

    def test_source_spec_with_dtype(self) -> None:
        """Source dtype is serialized when provided."""
        data = json.loads(Pipeline().source("raw", dtype="f32")._to_json())
        assert data["source"]["format"] == "raw"
        assert data["source"]["dtype"] == "f32"

    def test_shape_hints_format(self) -> None:
        """Shape hints are preserved in serialization."""
        pipe = Pipeline().source().assert_shape(height=100, width=200)
        data = json.loads(pipe._to_json())
        assert "shape_hints" in data
        assert data["shape_hints"]["height"]["value"] == 100
        assert data["shape_hints"]["width"]["value"] == 200


class TestExpressionReferencesJson:
    """Tests for expression reference serialization."""

    def test_multiple_column_references(self) -> None:
        """Multiple expression refs serialize distinctly."""
        pipe = (
            Pipeline()
            .source()
            .resize(height=pl.col("h"), width=pl.col("w"))
            .crop(top=pl.col("y"), left=pl.col("x"))
        )
        data = json.loads(pipe._to_json())
        assert data["ops"][0]["height"]["col"] == 'col("h")'
        assert data["ops"][0]["width"]["col"] == 'col("w")'
        assert data["ops"][1]["top"]["col"] == 'col("y")'
        assert data["ops"][1]["left"]["col"] == 'col("x")'


class TestJsonRustCompatibility:
    """Tests ensuring JSON stays Rust-deserializer compatible."""

    def test_flip_axes_list(self) -> None:
        """Flip axes are serialized as int lists."""
        data = json.loads(Pipeline().source().flip([0, 1])._to_json())
        axes = data["ops"][0]["axes"]
        assert axes["type"] == "literal"
        assert axes["value"] == [0, 1]

    def test_transpose_axes_list(self) -> None:
        """Transpose axes are serialized as int lists."""
        data = json.loads(Pipeline().source().transpose([2, 0, 1])._to_json())
        axes = data["ops"][0]["axes"]
        assert axes["type"] == "literal"
        assert axes["value"] == [2, 0, 1]
