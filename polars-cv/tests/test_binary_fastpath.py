"""
Row-alignment and error-policy tests for the single-output Binary fast path.

CompiledGraph::execute routes single-output blob/png/jpeg/webp/tiff sinks
(without `_error` messages) through a dedicated loop that appends each row
straight into the Arrow builder via a reusable scratch buffer. These tests
pin that the fast path keeps exactly the same observable behavior as the
generic path: per-row null semantics, raise-on-first-error, null input
passthrough, and row alignment over large batches with interleaved failures.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
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


@plugin_required
class TestBinaryFastPath:
    def test_blob_sink_raise_policy(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale()
        df = pl.DataFrame({"img": [_png(), _corrupt_png()]})
        with pytest.raises(pl.exceptions.ComputeError, match="Decode error"):
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("blob"))

    def test_blob_sink_null_policy_row_alignment_1000(self) -> None:
        # Every 7th row fails; the surviving rows must stay aligned with
        # their inputs (checked via a per-row size marker).
        pipe = Pipeline().source("image_bytes").on_error("null")
        sizes = [4 + (i % 3) for i in range(1000)]  # 4/5/6 px wide
        imgs = [
            _corrupt_png() if i % 7 == 0 else _png(width=sizes[i], height=4)
            for i in range(1000)
        ]
        df = pl.DataFrame({"img": imgs, "w": sizes})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("blob"))

        blobs = out["out"]
        assert blobs.null_count() == len([i for i in range(1000) if i % 7 == 0])
        for i in (1, 2, 6, 8, 699, 998, 999):
            if i % 7 == 0:
                continue
            blob = blobs[i]
            assert blob is not None
            # VIEW blob shape is stored as 3 little-endian u64s right after
            # the 64-byte header: the width dim must match this row's own
            # marker, proving no row shift after nulled failures.
            shape = np.frombuffer(blob[64 : 64 + 24], dtype="<u8")
            assert int(shape[1]) == sizes[i], f"row {i} misaligned"

    def test_blob_sink_null_inputs_pass_through(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale().on_error("null")
        df = pl.DataFrame(
            {"img": [_png(), None, _png(4, 4)]}, schema={"img": pl.Binary}
        )
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("blob"))
        assert out["out"][0] is not None
        assert out["out"][1] is None
        assert out["out"][2] is not None

    def test_blob_sink_streaming_alignment(self) -> None:
        pipe = Pipeline().source("image_bytes").grayscale().on_error("null")
        imgs = [_corrupt_png() if i % 5 == 0 else _png() for i in range(100)]
        out = (
            pl.LazyFrame({"img": imgs})
            .with_columns(out=pl.col("img").cv.pipe(pipe).sink("blob"))
            .collect(engine="streaming")
        )
        assert out["out"].null_count() == 20
        # Null rows must be exactly the corrupt ones.
        nulls = out["out"].is_null().to_list()
        assert all(nulls[i] == (i % 5 == 0) for i in range(100))

    def test_blob_matches_generic_path_output(self) -> None:
        # `null_with_message` routes through the generic Vec<RowResult> +
        # struct path; its blob field must be byte-identical to the fast
        # path's output for the same inputs.
        img = _png(6, 5)
        df = pl.DataFrame({"img": [img] * 3})

        fast_pipe = Pipeline().source("image_bytes").grayscale()
        fast = df.with_columns(out=pl.col("img").cv.pipe(fast_pipe).sink("blob"))[
            "out"
        ].to_list()

        generic_pipe = (
            Pipeline().source("image_bytes").grayscale().on_error("null_with_message")
        )
        generic_struct = df.with_columns(
            out=pl.col("img").cv.pipe(generic_pipe).sink("blob")
        )["out"]
        generic = generic_struct.struct.field("_output").to_list()

        assert fast == generic

    def test_png_sink_goes_through_fast_path(self) -> None:
        # png sink is also Binary-family; round-trip through PIL to verify.
        from PIL import Image

        pipe = Pipeline().source("image_bytes").grayscale()
        df = pl.DataFrame({"img": [_png(8, 8)]})
        out = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("png"))
        png_bytes = out["out"][0]
        decoded = Image.open(io.BytesIO(png_bytes))
        assert decoded.size == (8, 8)
