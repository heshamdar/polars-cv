"""Tests for sinkless pipeline JSON serialization."""

from __future__ import annotations

import json

import polars as pl

from polars_cv import Pipeline


class TestParameterSerialization:
    """A parameter crosses as its value, or as ``{"$slot": n}``: its position
    among the plugin inputs."""

    def test_literal_int_serialization(self) -> None:
        data = json.loads(Pipeline().source().scale(2)._to_json())
        assert data["ops"][0]["factor"] == 2

    def test_expr_column_serialization(self) -> None:
        pipe = Pipeline().source().scale(pl.col("my_column"))
        data = json.loads(pipe._to_json())
        # Input 0 is the pipeline's column, so its one expression is input 1.
        assert data["ops"][0]["factor"] == {"$slot": 1}


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

    def test_a_declaration_crosses_as_an_op_and_hints_do_not(self) -> None:
        """An ``assert_shape`` is an op on the wire (Rust plans and checks it);
        the planner's inferred sizes are never serialized."""
        pipe = Pipeline().source().assert_shape(height=100, width=200)
        data = json.loads(pipe._to_json())
        assert "shape_hints" not in data
        assert data["ops"] == [{"op": "assert_shape", "dims": [100, 200, None]}]


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
        # Input 0 is the pipeline's column; each distinct expression follows.
        assert data["ops"][0]["height"] == {"$slot": 1}
        assert data["ops"][0]["width"] == {"$slot": 2}
        assert data["ops"][1]["top"] == {"$slot": 3}
        assert data["ops"][1]["left"] == {"$slot": 4}


class TestJsonRustCompatibility:
    """Tests ensuring JSON stays Rust-deserializer compatible."""

    def test_flip_axes_list(self) -> None:
        """Flip axes are serialized as int lists."""
        data = json.loads(Pipeline().source().flip([0, 1])._to_json())
        # A typed op's field is the value itself.
        assert data["ops"][0]["axes"] == [0, 1]

    def test_transpose_axes_list(self) -> None:
        """Transpose axes are serialized as int lists."""
        data = json.loads(Pipeline().source().transpose([2, 0, 1])._to_json())
        assert data["ops"][0]["axes"] == [2, 0, 1]
