"""
Tests for the graph-level per-row error policy (`Pipeline.on_error`).

Three modes:
- "raise" (default): first failing row fails the whole expression.
- "null": failing rows yield null for all outputs; other rows proceed.
- "null_with_message": as "null", plus a reserved `_error` string field in
  the output struct carrying the failure message.

The policy covers any error while producing a row (source decode, op
resolution/execution, encode). The per-source `source(..., on_error="null")`
setting remains an independent, finer-grained control.
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
    # Bytes that fail image-format detection with a clean decode error.
    # (A *truncated* PNG is deliberately not used: its checksum path runs
    # SIMD code that, under this repo's `target-cpu=native` build flag, can
    # trap on hosts whose runtime CPU lacks the build CPU's features.)
    return b"definitely not an image"


class TestOnErrorValidation:
    def test_invalid_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="on_error must be one of"):
            Pipeline().source("image_bytes").on_error("explode")

    def test_policy_survives_cloning(self) -> None:
        pipe = Pipeline().source("image_bytes").on_error("null").grayscale()
        assert pipe._on_error == "null"


@plugin_required
class TestOnErrorRaise:
    def test_default_raises_on_corrupt_row(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale()
        df = pl.DataFrame({"img": [_png(), _corrupt_png()]})
        with pytest.raises(pl.exceptions.ComputeError, match="Decode error"):
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))


@plugin_required
class TestOnErrorNull:
    def test_corrupt_rows_become_null(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale().on_error("null")
        df = pl.DataFrame({"img": [_png(), _corrupt_png(), _png(4, 4)]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None
        assert list(numpy_from_struct(out["out"][2]).shape) == [4, 4, 1]

    def test_op_stage_error_becomes_null(self) -> None:
        # A per-row dynamic param that fails resolution (negative size) is an
        # op-stage error, not a decode error — it must also null the row.
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=pl.col("h"), width=pl.col("h"))
            .on_error("null")
        )
        df = pl.DataFrame({"img": [_png(), _png(), _png()], "h": [8, -1, 4]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        assert list(numpy_from_struct(out["out"][0]).shape) == [8, 8, 3]
        assert out["out"][1]["data"] is None
        assert list(numpy_from_struct(out["out"][2]).shape) == [4, 4, 3]

    def test_null_input_rows_still_null(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale().on_error("null")
        df = pl.DataFrame(
            {"img": [_png(), None, _corrupt_png()]}, schema={"img": pl.Binary}
        )
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None
        assert out["out"][2]["data"] is None

    def test_streaming_engine(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale().on_error("null")
        imgs = [_corrupt_png() if i % 5 == 0 else _png() for i in range(100)]
        df = pl.DataFrame({"img": imgs})
        out = (
            df.lazy()
            .with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
            .collect(engine="streaming")
        )
        data = out["out"].struct.field("data")
        assert data.null_count() == 20

    def test_multi_output_failing_row_nulls_all_aliases(self) -> None:
        base = (
            pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes").on_error("null"))
            .alias("base")
        )
        gray = base.pipe(Pipeline().grayscale()).alias("gray")
        expr = gray.sink({"base": "numpy", "gray": "numpy"})

        df = pl.DataFrame({"img": [_png(), _corrupt_png()]})
        out = df.with_columns(outs=expr)

        for alias in ("base", "gray"):
            field = out["outs"].struct.field(alias)
            assert field[0]["data"] is not None
            assert field[1]["data"] is None


@plugin_required
class TestOnErrorNullWithMessage:
    def test_error_field_populated_only_on_bad_rows(self) -> None:
        pipe = (
            Pipeline().source("image_bytes").grayscale().on_error("null_with_message")
        )
        df = pl.DataFrame({"img": [_png(), _corrupt_png()]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        errors = out["out"].struct.field("_error")
        assert errors[0] is None
        assert errors[1] is not None
        assert "Decode error" in errors[1]

        value = out["out"].struct.field("_output")
        assert value[0]["data"] is not None
        assert value[1]["data"] is None

    def test_plan_schema_matches_execution(self) -> None:
        # The `_error` field must appear identically in the planned schema
        # (collect_schema) and the executed one.
        pipe = (
            Pipeline().source("image_bytes").grayscale().on_error("null_with_message")
        )
        lf = pl.DataFrame({"img": [_png()]}).lazy()
        lf = lf.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        planned = lf.collect_schema()["out"]
        executed = lf.collect()["out"].dtype
        assert planned == executed
        assert isinstance(planned, pl.Struct)
        assert {f.name for f in planned.fields} == {"_output", "_error"}


@plugin_required
class TestOnErrorComposition:
    def test_conflicting_policies_rejected(self) -> None:
        a = pl.col("img").cv.pipe(Pipeline().source("image_bytes").on_error("null"))
        b = pl.col("img").cv.pipe(
            Pipeline().source("image_bytes").grayscale().on_error("null_with_message")
        )
        merged = a.add(b)
        with pytest.raises(ValueError, match="Conflicting on_error"):
            merged.sink("numpy")

    def test_policy_set_on_lazy_expr(self) -> None:
        # The policy can be attached anywhere in a lazy chain (it is carried
        # through continuation nodes and collected at the graph level).
        expr = (
            pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes"))
            .pipe(Pipeline().grayscale())
            .on_error("null")
            .sink("numpy")
        )
        df = pl.DataFrame({"img": [_png(), _corrupt_png()]})
        out = df.with_columns(out=expr)
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None
