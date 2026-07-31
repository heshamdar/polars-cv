"""Null values in per-row expression parameter columns.

A parameter that takes a `pl.Expr` is read from an ordinary Polars column, so
it may contain nulls. By default that fails the whole expression; the policy
opts into null-in/null-out instead, matching what a null *input image* already
does:

- `Pipeline.on_null_param("null")` for `vb_graph` pipelines.
- `pl.col(...).contour.on_null("null")` for the geometry namespaces, which
  bypass `vb_graph` and so carry the policy on the accessor.

There is deliberately no "fallback default" mode: `pl.col("h").fill_null(224)`
already expresses that, and `test_fill_null_is_the_fallback_idiom` pins it.

The mechanism is shared — one policy on `ParamCtx`, applied at the four
`ParamCol` accessors — so these tests cover the *kinds* of parameter that route
through it (numeric, enum, flag, list element) rather than every operation.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required


def _png(width: int = 8, height: int = 8, value: int = 128) -> bytes:
    from PIL import Image

    arr = np.full((height, width, 3), value, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _corrupt_png() -> bytes:
    return b"definitely not an image"


# A CCW 100x100 square, matching tests/test_expression_params.py.
SQUARE = {
    "exterior": [
        {"x": 0.0, "y": 0.0},
        {"x": 100.0, "y": 0.0},
        {"x": 100.0, "y": 100.0},
        {"x": 0.0, "y": 100.0},
    ],
    "holes": [],
    "is_closed": True,
}


class TestOnNullParamValidation:
    def test_invalid_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_null_param must be one of"):
            Pipeline().source("image_bytes").on_null_param("fallback")

    def test_policy_survives_cloning(self) -> None:
        pipe = Pipeline().source("image_bytes").on_null_param("null").grayscale()
        assert pipe._on_null_param == "null"

    def test_absent_policy_is_not_serialized(self) -> None:
        # Graphs that do not use the policy must serialize exactly as before,
        # so the compiled-graph cache key is unchanged for existing pipelines.
        pipe = Pipeline().source("image_bytes").grayscale()
        graph = pl.col("img").cv.pipe(pipe).sink("numpy", return_expr=False)
        assert "on_null_param" not in graph._to_dict()

    def test_policy_is_serialized_when_set(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale().on_null_param("null")
        graph = pl.col("img").cv.pipe(pipe).sink("numpy", return_expr=False)
        assert graph._to_dict()["on_null_param"] == "null"


@plugin_required
class TestDefaultRaises:
    """The default is unchanged: a null parameter fails the query."""

    def test_null_numeric_param_raises(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
        )
        df = pl.DataFrame({"img": [_png(), _png()], "h": [4, None]})
        with pytest.raises(pl.exceptions.ComputeError, match="has a null value at row"):
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

    def test_error_names_the_column_and_row(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("target_h"), width=pl.col("target_h"))
        )
        df = pl.DataFrame({"img": [_png()] * 3, "target_h": [4, 4, None]})
        with pytest.raises(pl.exceptions.ComputeError) as excinfo:
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        message = str(excinfo.value)
        assert "target_h" in message
        assert "row 2" in message


@plugin_required
class TestOnNullParamNull:
    def test_only_affected_rows_are_null(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .on_null_param("null")
        )
        df = pl.DataFrame({"img": [_png()] * 3, "h": [4, None, 16]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        assert list(numpy_from_struct(out["out"][0]).shape) == [4, 4, 3]
        assert out["out"][1]["data"] is None
        assert list(numpy_from_struct(out["out"][2]).shape) == [16, 16, 3]

    def test_matches_the_equivalent_literal_pipeline(self) -> None:
        # Non-null rows must be untouched by the policy: their result has to
        # equal what a literal pipeline produces for the same value.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .on_null_param("null")
        )
        df = pl.DataFrame({"img": [_png(), _png()], "h": [4, None]})
        dynamic = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        literal = Pipeline().source("image_bytes").resize(height=4, width=4)
        expected = df.head(1).with_columns(
            out=pl.col("img").cv.pipe(literal).sink("numpy")
        )
        np.testing.assert_array_equal(
            numpy_from_struct(dynamic["out"][0]), numpy_from_struct(expected["out"][0])
        )

    def test_null_enum_param(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=4, width=4, filter=pl.col("f"))
            .on_null_param("null")
        )
        df = pl.DataFrame(
            {"img": [_png(), _png()], "f": ["nearest", None]},
            schema={"img": pl.Binary, "f": pl.String},
        )
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None

    def test_null_flag_param(self) -> None:
        kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        pipe = (
            Pipeline()
            .source("image_bytes")
            .convolve2d(kernel=kernel, ksize=3, normalize=pl.col("norm"))
            .on_null_param("null")
        )
        df = pl.DataFrame(
            {"img": [_png(), _png()], "norm": [True, None]},
            schema={"img": pl.Binary, "norm": pl.Boolean},
        )
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None

    def test_null_list_element_param(self) -> None:
        # The list *length* stays structural; a null in one element still goes
        # through the same accessor as a scalar parameter.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .convolve2d(
                kernel=[0.0, 0.0, 0.0, 0.0, pl.col("center"), 0.0, 0.0, 0.0, 0.0],
                ksize=3,
            )
            .on_null_param("null")
        )
        df = pl.DataFrame({"img": [_png(), _png()], "center": [1.0, None]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None

    def test_broadcast_null_nulls_every_row(self) -> None:
        # A length-1 parameter series (an aggregation) applies to every row, so
        # a null one nulls all of them.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h").max(), width=pl.col("h").max())
            .on_null_param("null")
        )
        df = pl.DataFrame(
            {"img": [_png()] * 3, "h": [None, None, None]},
            schema={"img": pl.Binary, "h": pl.Int64},
        )
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert out["out"].struct.field("data").null_count() == 3

    def test_streaming_engine(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .on_null_param("null")
        )
        heights = [None if i % 5 == 0 else 4 for i in range(100)]
        df = pl.DataFrame(
            {"img": [_png()] * 100, "h": heights},
            schema={"img": pl.Binary, "h": pl.Int64},
        )
        out = (
            df.lazy()
            .with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
            .collect(engine="streaming")
        )
        assert out["out"].struct.field("data").null_count() == 20

    def test_plan_schema_matches_execution(self) -> None:
        # Per-row parameters cannot affect shape, rank or dtype, so nulling a
        # row must not change the column's schema.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .on_null_param("null")
        )
        lf = pl.DataFrame({"img": [_png()], "h": [4]}).lazy()
        lf = lf.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert lf.collect_schema()["out"] == lf.collect()["out"].dtype


@plugin_required
class TestScopedToTheAffectedOutput:
    """Node-scoped, unlike `on_error("null")` which nulls the whole row."""

    def test_unrelated_output_is_not_nulled(self) -> None:
        base = (
            pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes").on_null_param("null"))
            .alias("base")
        )
        resized = base.pipe(
            Pipeline().resize(height=pl.col("h"), width=pl.col("h"))
        ).alias("resized")
        expr = resized.sink({"base": "numpy", "resized": "numpy"})

        df = pl.DataFrame({"img": [_png(), _png()], "h": [4, None]})
        out = df.with_columns(outs=expr)

        base_field = out["outs"].struct.field("base")
        resized_field = out["outs"].struct.field("resized")
        # The null parameter belongs to the `resized` node only.
        assert base_field[0]["data"] is not None
        assert base_field[1]["data"] is not None
        assert resized_field[0]["data"] is not None
        assert resized_field[1]["data"] is None


@plugin_required
class TestIndependentOfOnError:
    def test_does_not_mask_a_decode_error(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale().on_null_param("null")
        df = pl.DataFrame({"img": [_png(), _corrupt_png()]})
        with pytest.raises(pl.exceptions.ComputeError, match="Decode error"):
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

    def test_null_param_records_no_error_message(self) -> None:
        # A null parameter is not an error, so `null_with_message` leaves
        # `_error` empty for those rows even though the output is null.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .on_null_param("null")
            .on_error("null_with_message")
        )
        df = pl.DataFrame({"img": [_png(), _png()], "h": [4, None]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        assert out["out"].struct.field("_error")[1] is None
        assert out["out"].struct.field("_output")[1]["data"] is None

    def test_on_error_null_still_covers_null_params(self) -> None:
        # Pre-existing behaviour: `on_error("null")` catches the null-parameter
        # error too, just less precisely (it nulls the whole row).
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .on_error("null")
        )
        df = pl.DataFrame({"img": [_png(), _png()], "h": [4, None]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None


@plugin_required
class TestComposition:
    def test_conflicting_policies_rejected(self) -> None:
        a = pl.col("img").cv.pipe(
            Pipeline().source("image_bytes").on_null_param("null")
        )
        b = pl.col("img").cv.pipe(
            Pipeline().source("image_bytes").grayscale().on_null_param("raise")
        )
        # Only distinct *non-default* policies conflict; "raise" is the default
        # and so never participates.
        merged = a.add(b)
        merged.sink("numpy")

    def test_policy_set_on_lazy_expr(self) -> None:
        expr = (
            pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes"))
            .pipe(Pipeline().resize(height=pl.col("h"), width=pl.col("h")))
            .on_null_param("null")
            .sink("numpy")
        )
        df = pl.DataFrame({"img": [_png(), _png()], "h": [4, None]})
        out = df.with_columns(out=expr)
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None


@plugin_required
class TestFallbackIdiom:
    def test_fill_null_is_the_fallback_idiom(self) -> None:
        # Why there is no "default" policy: Polars already expresses it.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h").fill_null(4), width=pl.col("h").fill_null(4))
        )
        df = pl.DataFrame({"img": [_png(), _png()], "h": [16, None]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert list(numpy_from_struct(out["out"][0]).shape) == [16, 16, 3]
        assert list(numpy_from_struct(out["out"][1]).shape) == [4, 4, 3]


@plugin_required
class TestGeometryNamespaces:
    """The `.contour` / `.point` / `.bbox` accessors carry the same policy."""

    def test_contour_default_raises(self) -> None:
        df = pl.DataFrame({"c": [SQUARE, SQUARE], "w": [100.0, None]})
        with pytest.raises(Exception, match="null"):
            df.with_columns(n=pl.col("c").contour.normalize(pl.col("w"), 100))

    def test_contour_on_null_nulls_the_row(self) -> None:
        df = pl.DataFrame({"c": [SQUARE, SQUARE], "w": [100.0, None]})
        out = df.with_columns(
            n=pl.col("c").contour.on_null("null").normalize(pl.col("w"), 100)
        )["n"].to_list()
        assert out[0] is not None
        assert out[1] is None

    def test_point_on_null_matches_a_null_input_point(self) -> None:
        # Point transforms build their struct from two Float64 columns, so a
        # null row is a struct of null fields. The policy must produce exactly
        # what a null *input* point already produces — not some third shape.
        point = {"x": 10.0, "y": 10.0}
        df = pl.DataFrame(
            {"p": [point, point, None], "dx": [5.0, None, 5.0]},
            schema={
                "p": pl.Struct({"x": pl.Float64, "y": pl.Float64}),
                "dx": pl.Float64,
            },
        )
        out = df.with_columns(
            t=pl.col("p").point.on_null("null").translate(pl.col("dx"), 0.0)
        )["t"].to_list()
        assert out[0]["x"] == 15.0
        # [1] null parameter, [2] null input — identical results.
        assert out[1] == out[2] == {"x": None, "y": None}

    def test_bbox_on_null_nulls_the_row(self) -> None:
        box = [{"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}]
        df = pl.DataFrame({"p": [box, box], "g": [box, box], "thr": [0.5, None]})
        out = df.with_columns(
            m=pl.col("p")
            .bbox.on_null("null")
            .match_detections(pl.col("g"), threshold=pl.col("thr"))
        )["m"].to_list()
        assert out[0] is not None
        assert out[1] is None

    def test_invalid_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_null must be one of"):
            pl.col("c").contour.on_null("fallback")

    def test_accessor_is_not_mutated(self) -> None:
        # `on_null` returns a copy, matching Pipeline's immutable-builder rule.
        ns = pl.col("c").contour
        configured = ns.on_null("null")
        assert ns._on_null == "raise"
        assert configured._on_null == "null"


@plugin_required
class TestNullOperandPropagation:
    """A node that went null must null its consumers, not raise.

    Regression for the conflation of "node produced nothing for this row" with
    "node is not in the graph", which made a null operand a hard error.
    """

    def test_null_bytes_in_a_merge_operand(self) -> None:
        left = pl.col("a").cv.pipe(Pipeline().source("image_bytes"))
        right = pl.col("b").cv.pipe(Pipeline().source("image_bytes"))
        expr = left.add(right).sink("numpy")

        df = pl.DataFrame(
            {"a": [_png(), _png()], "b": [_png(), None]},
            schema={"a": pl.Binary, "b": pl.Binary},
        )
        out = df.with_columns(out=expr)
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None

    def test_null_param_in_a_merge_operand(self) -> None:
        left = pl.col("a").cv.pipe(
            Pipeline().source("image_bytes").on_null_param("null")
        )
        right = pl.col("b").cv.pipe(
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .on_null_param("null")
        )
        expr = left.pipe(Pipeline().resize(height=8, width=8)).add(right).sink("numpy")

        df = pl.DataFrame(
            {"a": [_png(), _png()], "b": [_png(), _png()], "h": [8, None]}
        )
        out = df.with_columns(out=expr)
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None
