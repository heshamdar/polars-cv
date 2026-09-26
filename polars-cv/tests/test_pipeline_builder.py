"""Unit tests for the sinkless Pipeline builder."""

from __future__ import annotations

import json

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import DType, SourceFormat
from tests._plan_view import EXPR, exprs_of, op_names, ops_of, planned, source_of


class TestPipelineSource:
    """Tests for Pipeline source configuration."""

    def test_source_default_format(self) -> None:
        """Default source format is auto (inferred from the column dtype)."""
        pipe = Pipeline().source()
        assert source_of(pipe) is not None
        assert source_of(pipe).format == SourceFormat.AUTO

    def test_source_raw_with_dtype(self) -> None:
        """Raw source requires dtype and stores it."""
        pipe = Pipeline().source("raw", dtype="f32")
        assert source_of(pipe) is not None
        assert source_of(pipe).format == SourceFormat.RAW
        assert source_of(pipe).dtype == DType.F32

    def test_source_raw_without_dtype_raises(self) -> None:
        """Raw source without dtype raises an error."""
        with pytest.raises(ValueError, match="missing field `dtype`"):
            Pipeline().source("raw")


class TestPipelineOps:
    """Tests for operation composition."""

    def test_resize_tracks_op(self) -> None:
        """Resize appends one resize op."""
        pipe = Pipeline().source().resize(height=224, width=224)
        assert len(ops_of(pipe)) == 1
        assert ops_of(pipe)[0].op == "resize"

    def test_compute_ops(self) -> None:
        """Compute operations append expected op specs."""
        pipe = Pipeline().source().cast("f32").scale(2.5).clamp(0.0, 1.0)
        assert op_names(pipe) == ["cast", "scale", "clamp"]

    def test_domain_conversion_ops(self) -> None:
        """Contour conversions preserve expected op ordering."""
        pipe = Pipeline().source().grayscale().threshold(128).extract_contours()
        assert op_names(pipe)[-1] == "extract_contours"


class TestExpressionTracking:
    """Tests for expression parameter tracking."""

    def test_resize_with_expr(self) -> None:
        """Expression params are stored and tracked."""
        pipe = Pipeline().source().resize(height=pl.col("h"), width=pl.col("w"))
        assert ops_of(pipe)[0].params["height"] == EXPR
        assert ops_of(pipe)[0].params["width"] == EXPR
        assert len(exprs_of(pipe)) == 2

    def test_no_duplicate_expr_tracking(self) -> None:
        """The same expression is tracked only once."""
        expr = pl.col("size")
        pipe = Pipeline().source().resize(height=expr, width=expr)
        assert len(exprs_of(pipe)) == 1


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

    def test_serialize_pipeline_omits_shape_hints(self) -> None:
        """Shape hints are preserved in serialized output."""
        pipe = Pipeline().source().assert_shape(channels=3)
        data = json.loads(pipe._to_json())
        # Shape hints are plan-time state, not wire format (nothing in Rust
        # reads the key); `expected_shape` on the output spec carries what
        # execution needs.
        assert "shape_hints" not in data


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
        assert planned(pipe).ndim == 2

        graph = pipe.to_graph(pl.col("img"))
        node = graph._nodes["_node_0"]
        assert planned(node.pipeline).ndim == planned(pipe).ndim
        assert planned(node.pipeline).dtype == planned(pipe).dtype
        assert planned(node.pipeline).domain == planned(pipe).domain

    def test_double_axis_reduction_ndim_preserved(self) -> None:
        """Two axis reductions: ndim 3 -> 2 -> 1; re-folding from the final
        state would try to reduce below rank 0."""
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .reduce_max(axis=0)
            .reduce_max(axis=0)
        )
        assert planned(pipe).ndim == 1

        graph = pipe.to_graph(pl.col("img"))
        node = graph._nodes["_node_0"]
        assert planned(node.pipeline).ndim == planned(pipe).ndim
        assert planned(node.pipeline).dtype == planned(pipe).dtype
        assert planned(node.pipeline).domain == planned(pipe).domain

    def test_plain_image_pipeline_state_preserved(self) -> None:
        """Non-reducing pipeline: state must match exactly too."""
        pipe = Pipeline().source("image_bytes").resize(height=32, width=32).grayscale()
        graph = pipe.to_graph(pl.col("img"))
        node = graph._nodes["_node_0"]
        assert planned(node.pipeline).ndim == planned(pipe).ndim
        assert planned(node.pipeline).dtype == planned(pipe).dtype
        assert planned(node.pipeline).domain == planned(pipe).domain


class TestSugarKeywordRefusals:
    """The hand-written sugar maps its keywords onto one op; a combination
    that names no single value is refused at the call, naming the keywords."""

    def test_resize_scale_needs_a_factor(self) -> None:
        with pytest.raises(ValueError, match="'scale' or 'scale_x'/'scale_y'"):
            Pipeline().source("image_bytes").resize_scale()

    def test_resize_scale_needs_both_axes_without_scale(self) -> None:
        with pytest.raises(ValueError, match="both scale factors"):
            Pipeline().source("image_bytes").resize_scale(scale_x=0.5)

    def test_resize_scale_per_axis_overrides_uniform(self) -> None:
        pipe = Pipeline().source("image_bytes").resize_scale(scale=0.5, scale_y=2.0)
        (op,) = ops_of(pipe)
        assert (op.params["scale_x"], op.params["scale_y"]) == (0.5, 2.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({}, "not neither"),
            ({"width": 4}, "'width' and 'height' must be specified together"),
            ({"height": 4}, "'width' and 'height' must be specified together"),
        ],
    )
    def test_rasterize_needs_one_canvas(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Pipeline().source("contour").rasterize(**kwargs)

    def test_rasterize_refuses_both_canvases(self) -> None:
        img = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
        with pytest.raises(ValueError, match="not both"):
            Pipeline().source("contour").rasterize(width=4, height=4, shape=img)

    def test_rasterize_shape_must_be_a_node(self) -> None:
        with pytest.raises(TypeError, match="must be a LazyPipelineExpr"):
            Pipeline().source("contour").rasterize(shape=(4, 4))  # type: ignore[arg-type]

    def test_a_column_field_must_be_an_expression(self) -> None:
        """``label_reduce(contours=)`` reads a column as data: a literal has
        no column to give."""
        with pytest.raises(TypeError, match="must be a Polars expression"):
            Pipeline().source("image_bytes").label_reduce([1, 2])  # type: ignore[arg-type]
