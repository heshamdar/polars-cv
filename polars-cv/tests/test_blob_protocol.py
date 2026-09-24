"""A hostile VIEW blob is a row error, never undefined behaviour (CR-41).

The blob header is column data, so every field in it is untrusted. Bounds were
already checked; alignment was not. A ``data_offset`` or stride that is not a
multiple of the element size used to build a misaligned typed slice. Debug
builds caught it with a ``debug_assert!`` (reported as "the engine panicked"),
but the release wheels compile that out, so the same row was undefined
behaviour. It must now be rejected as an ordinary error that names the
problem.
"""

from __future__ import annotations

import struct

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline

from .conftest import plugin_required

PANIC = "the engine panicked"
HEADER_SIZE = 64


def _f32_blob() -> bytes:
    """A valid 4x3 f32 blob, written by the plugin's own blob sink."""
    arr = np.arange(12, dtype=np.float32).reshape(4, 3)
    df = pl.DataFrame({"a": pl.Series("a", arr[np.newaxis])})
    return df.select(b=pl.col("a").cv.pipe(Pipeline().source("array")).sink("blob"))[
        "b"
    ][0]


def _shift_payload(blob: bytes, by: int) -> bytes:
    """Move the payload `by` bytes later and point `data_offset` at it."""
    (offset,) = struct.unpack_from("<Q", blob, 8)
    out = bytearray(blob[:offset]) + b"\x00" * by + blob[offset:]
    struct.pack_into("<Q", out, 8, offset + by)
    return bytes(out)


def _with_strides(blob: bytes, strides: list[int]) -> bytes:
    """Rewrite the stored strides and mark the layout non-contiguous."""
    rank = blob[7]
    assert len(strides) == rank
    out = bytearray(blob)
    struct.pack_into("<Q", out, 16, 0)  # flags: not contiguous
    for i, s in enumerate(strides):
        struct.pack_into("<q", out, HEADER_SIZE + rank * 8 + i * 8, s)
    return bytes(out)


def _run(blob: bytes) -> pl.DataFrame:
    df = pl.DataFrame({"b": [blob]}, schema={"b": pl.Binary})
    pipe = Pipeline().source("blob").scale(2.0)
    return df.select(o=pl.col("b").cv.pipe(pipe).sink("numpy"))


@plugin_required
class TestHostileBlobAlignment:
    def test_the_unmodified_blob_decodes(self) -> None:
        """Baseline: the fixtures below differ from a working blob only in
        the field under test."""
        assert _run(_f32_blob())["o"][0] is not None

    @pytest.mark.parametrize("shift", [1, 2, 3])
    def test_a_misaligned_data_offset_is_a_row_error(self, shift: int) -> None:
        with pytest.raises(pl.exceptions.ComputeError) as excinfo:
            _run(_shift_payload(_f32_blob(), shift))
        message = str(excinfo.value)
        assert PANIC not in message
        assert "align" in message

    def test_a_stride_that_is_not_a_whole_element_is_a_row_error(self) -> None:
        # Row stride 10 bytes: the furthest element ends at 3*10 + 2*4 + 4 =
        # 42 <= 48 payload bytes, so only alignment is wrong.
        with pytest.raises(pl.exceptions.ComputeError) as excinfo:
            _run(_with_strides(_f32_blob(), [10, 4]))
        message = str(excinfo.value)
        assert PANIC not in message
        assert "align" in message

    def test_a_misaligned_blob_nulls_its_row_under_on_error_null(self) -> None:
        df = pl.DataFrame(
            {"b": [_f32_blob(), _shift_payload(_f32_blob(), 1)]},
            schema={"b": pl.Binary},
        )
        pipe = Pipeline().source("blob").scale(2.0).on_error("null")
        out = df.select(o=pl.col("b").cv.pipe(pipe).sink("numpy"))["o"]
        assert out[0] is not None
        assert out[1] is None
