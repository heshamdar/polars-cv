"""Unit tests for the sinkless Pipeline builder."""

from __future__ import annotations

import json

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import DType, SourceFormat


class TestPipelineSource:
    """Tests for Pipeline source configuration."""

    def test_source_default_format(self) -> None:
        """Default source format is auto (inferred from the column dtype)."""
        pipe = Pipeline().source()
        assert pipe._source is not None
        assert pipe._source.format == SourceFormat.AUTO

    def test_source_raw_with_dtype(self) -> None:
        """Raw source requires dtype and stores it."""
        pipe = Pipeline().source("raw", dtype="f32")
        assert pipe._source is not None
        assert pipe._source.format == SourceFormat.RAW
        assert pipe._source.dtype == DType.F32

    def test_source_raw_without_dtype_raises(self) -> None:
        """Raw source without dtype raises an error."""
        with pytest.raises(ValueError, match="dtype is required"):
            Pipeline().source("raw")


class TestPipelineOps:
    """Tests for operation composition."""

    def test_resize_tracks_op(self) -> None:
        """Resize appends one resize op."""
        pipe = Pipeline().source().resize(height=224, width=224)
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "resize"

    def test_compute_ops(self) -> None:
        """Compute operations append expected op specs."""
        pipe = Pipeline().source().cast("f32").scale(2.5).clamp(0.0, 1.0)
        assert [op.op for op in pipe._ops] == ["cast", "scale", "clamp"]

    def test_domain_conversion_ops(self) -> None:
        """Contour conversions preserve expected op ordering."""
        pipe = Pipeline().source().grayscale().threshold(128).extract_contours()
        assert [op.op for op in pipe._ops][-1] == "extract_contours"


class TestExpressionTracking:
    """Tests for expression parameter tracking."""

    def test_resize_with_expr(self) -> None:
        """Expression params are stored and tracked."""
        pipe = Pipeline().source().resize(height=pl.col("h"), width=pl.col("w"))
        assert pipe._ops[0].params["height"].is_expr
        assert pipe._ops[0].params["width"].is_expr
        assert len(pipe._expr_refs) == 2

    def test_no_duplicate_expr_tracking(self) -> None:
        """The same expression is tracked only once."""
        expr = pl.col("size")
        pipe = Pipeline().source().resize(height=expr, width=expr)
        assert len(pipe._expr_refs) == 1


class TestPipelineValidation:
    """Tests for source-only validation contract."""

    def test_validate_no_source_raises(self) -> None:
        """Validation fails when source is missing."""
        with pytest.raises(ValueError, match="must have a source"):
            Pipeline().validate()

    def test_validate_with_source_passes(self) -> None:
        """Validation passes with a configured source."""
        Pipeline().source().validate()


class TestPipelineSerialization:
    """Tests for sinkless linear pipeline JSON serialization."""

    def test_serialize_simple_pipeline(self) -> None:
        """Serialized JSON contains source and ops only."""
        pipe = Pipeline().source().resize(height=224, width=224)
        data = json.loads(pipe._to_json())
        assert data["source"]["format"] == "auto"
        assert len(data["ops"]) == 1
        assert "sink" not in data

    def test_serialize_pipeline_with_shape_hints(self) -> None:
        """Shape hints are preserved in serialized output."""
        pipe = Pipeline().source().assert_shape(channels=3)
        data = json.loads(pipe._to_json())
        assert "shape_hints" in data
        assert data["shape_hints"]["channels"]["value"] == 3


class TestPipelineRepr:
    """Tests for Pipeline string representation."""

    def test_repr_empty(self) -> None:
        """Empty pipeline repr is stable."""
        assert repr(Pipeline()) == "Pipeline()"

    def test_repr_no_sink(self) -> None:
        """Repr no longer includes sink() segment."""
        repr_str = repr(Pipeline().source().resize(height=8, width=8))
        assert "source" in repr_str
        assert "resize" in repr_str
        assert "sink" not in repr_str


class TestToGraphPreservesPlannedState:
    """to_graph() must carry the pipeline's incrementally tracked
    domain/dtype/ndim into the graph node, not re-derive them by folding
    all ops on top of the already-final state (which double-applies ops).
    """

    def test_axis_reduction_ndim_preserved(self) -> None:
        """A single axis reduction: ndim 3 -> 2 must survive to_graph."""
        pipe = Pipeline().source("image_bytes", dtype="u8").reduce_max(axis=0)
        assert pipe._expected_ndim == 2

        graph = pipe.to_graph(pl.col("img"))
        node = graph._nodes["_node_0"]
        assert node.pipeline._expected_ndim == pipe._expected_ndim
        assert node.pipeline._output_dtype == pipe._output_dtype
        assert node.pipeline._current_domain == pipe._current_domain

    def test_double_axis_reduction_ndim_preserved(self) -> None:
        """Two axis reductions: ndim 3 -> 2 -> 1; re-folding from the final
        state would try to reduce below rank 0."""
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .reduce_max(axis=0)
            .reduce_max(axis=0)
        )
        assert pipe._expected_ndim == 1

        graph = pipe.to_graph(pl.col("img"))
        node = graph._nodes["_node_0"]
        assert node.pipeline._expected_ndim == pipe._expected_ndim
        assert node.pipeline._output_dtype == pipe._output_dtype
        assert node.pipeline._current_domain == pipe._current_domain

    def test_plain_image_pipeline_state_preserved(self) -> None:
        """Non-reducing pipeline: state must match exactly too."""
        pipe = Pipeline().source("image_bytes").resize(height=32, width=32).grayscale()
        graph = pipe.to_graph(pl.col("img"))
        node = graph._nodes["_node_0"]
        assert node.pipeline._expected_ndim == pipe._expected_ndim
        assert node.pipeline._output_dtype == pipe._output_dtype
        assert node.pipeline._current_domain == pipe._current_domain
