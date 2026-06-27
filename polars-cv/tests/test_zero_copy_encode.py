"""
Correctness guards for the zero-extra-copy output encoding.

These pin the behaviour of two encode-path changes:

  - Binary/blob/encoded sinks register each row's already-materialised bytes
    as a `BinaryViewArray` backing buffer instead of copying them into a
    `BinaryChunkedBuilder` (`src/output.rs::binary_view_series_from_rows`).
  - Typed list/array sinks move (rather than clone) their per-row buffers
    while assembling the output (`src/graph/decode.rs::build_series_from_spec`,
    which now takes its rows by value).

Both are pure optimisations: outputs must stay byte-identical to a direct
eager evaluation, survive nulls / all-null morsels, and remain stable across
the streaming engine's per-morsel invocation (the encode path runs once over
the whole column in-memory, but once per morsel under streaming).
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

pytestmark = plugin_required


def _png(seed: int, h: int = 8, w: int = 8) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Blob / binary buffer-registration path
# ---------------------------------------------------------------------------


class TestBlobBufferRegistration:
    def test_blob_roundtrip_is_byte_exact(self) -> None:
        """image -> blob (registered buffers) -> decode must reproduce pixels."""
        n = 40
        imgs = [_png(i, h=6, w=10) for i in range(n)]
        df = pl.DataFrame({"img": imgs})

        to_blob = Pipeline().source("image_bytes")
        from_blob = Pipeline().source("blob", dtype="u8")

        out = df.with_columns(
            blob=pl.col("img").cv.pipe(to_blob).sink("blob")
        ).with_columns(arr=pl.col("blob").cv.pipe(from_blob).sink("numpy"))

        for i in range(n):
            decoded = numpy_from_struct(out["arr"][i])
            expected = np.array(Image.open(io.BytesIO(imgs[i])))
            assert decoded is not None
            np.testing.assert_array_equal(decoded, expected)

    def test_blob_streaming_matches_in_memory(self) -> None:
        """Per-morsel streaming output equals the whole-column eager output."""
        n = 500  # spans many streaming morsels
        df = pl.DataFrame({"img": [_png(i, h=5, w=7) for i in range(n)]})
        pipe = Pipeline().source("image_bytes").resize(height=4, width=4)
        lf = df.lazy().with_columns(b=pl.col("img").cv.pipe(pipe).sink("blob"))

        streamed = lf.collect(engine="streaming")
        eager = lf.collect(engine="in-memory")
        assert streamed.equals(eager)
        # And byte-identical to evaluating on the eager DataFrame directly.
        direct = df.with_columns(b=pl.col("img").cv.pipe(pipe).sink("blob"))
        assert eager.equals(direct)

    def test_blob_preserves_nulls(self) -> None:
        """Null inputs yield null blobs; non-null rows stay correct."""
        imgs = [_png(i) if i % 4 else None for i in range(60)]
        df = pl.DataFrame({"img": imgs}, schema={"img": pl.Binary})
        pipe = Pipeline().source("image_bytes").grayscale()

        out = df.with_columns(b=pl.col("img").cv.pipe(pipe).sink("blob"))
        nulls = out["b"].is_null().to_list()
        assert nulls == [i % 4 == 0 for i in range(60)]
        # Streaming agrees on the null pattern and bytes.
        streamed = (
            df.lazy()
            .with_columns(b=pl.col("img").cv.pipe(pipe).sink("blob"))
            .collect(engine="streaming")
        )
        assert streamed.equals(out)

    def test_blob_all_null_morsel(self) -> None:
        """A column of all-null inputs encodes to an all-null Binary column."""
        df = pl.DataFrame({"img": [None] * 32}, schema={"img": pl.Binary})
        pipe = Pipeline().source("image_bytes").grayscale()
        out = df.with_columns(b=pl.col("img").cv.pipe(pipe).sink("blob"))
        assert out["b"].null_count() == 32
        assert out["b"].dtype == pl.Binary


# ---------------------------------------------------------------------------
# Typed list / array assembly (move-not-clone)
# ---------------------------------------------------------------------------


class TestTypedListArrayEncoding:
    @pytest.mark.parametrize(
        ("pl_dtype", "vb_dtype"),
        [
            (pl.UInt8, "u8"),
            (pl.Int16, "i16"),
            (pl.Float32, "f32"),
            (pl.Float64, "f64"),
        ],
    )
    def test_list_source_to_list_sink_roundtrip(self, pl_dtype, vb_dtype) -> None:
        """list source -> list sink preserves values exactly for each dtype."""
        rows = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
        df = pl.DataFrame({"arr": rows}).cast({"arr": pl.List(pl.List(pl_dtype))})
        pipe = Pipeline().source("list", dtype=vb_dtype)
        out = df.with_columns(out=pl.col("arr").cv.pipe(pipe).sink("list"))
        assert out["out"].to_list() == rows

    def test_array_sink_streaming_matches_in_memory(self) -> None:
        n = 300
        df = pl.DataFrame({"img": [_png(i) for i in range(n)]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=4, width=4)
            .grayscale()
            .cast("u8")
        )
        lf = df.lazy().with_columns(
            a=pl.col("img").cv.pipe(pipe).sink("array", shape=[4, 4, 1])
        )
        streamed = lf.collect(engine="streaming")
        eager = lf.collect(engine="in-memory")
        assert streamed.equals(eager)
        assert eager["a"].dtype == pl.Array(pl.UInt8, (4, 4, 1))

    def test_list_sink_preserves_nulls(self) -> None:
        imgs = [_png(i) if i % 3 else None for i in range(45)]
        df = pl.DataFrame({"img": imgs}, schema={"img": pl.Binary})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=3, width=3)
            .grayscale()
            .cast("u8")
        )
        out = df.with_columns(v=pl.col("img").cv.pipe(pipe).sink("list"))
        assert out["v"].is_null().to_list() == [i % 3 == 0 for i in range(45)]
