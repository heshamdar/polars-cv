"""The streaming guide's claims about morsels, checked at the plugin.

``docs/user-guide/concepts/streaming.md`` tells users how big a polars-cv call
is under the streaming engine, because that is what bounds its memory: a call
is one morsel, and a whole morsel's outputs are resident while it runs.

- A Parquet scan hands the plugin at most one row group per call, so the
  row-group size written is the knob a user controls.
- ``pl.Config.set_streaming_chunk_size`` bounds the morsels an in-memory
  frame is split into, and splits a larger Parquet row group.
- ``LazyFrame.collect_batches`` streams a fixed-shape ``array`` sink into one
  ``(rows, *shape)`` NumPy batch at a time, without copying.

The morsel sizes are read from the plugin itself
(``_lib._take_max_split_rows``: the most rows one call ran).
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline

from .conftest import plugin_required

_PIPE = Pipeline().source("raw", dtype="u8")


def _frame(rows: int) -> pl.DataFrame:
    return pl.DataFrame({"b": [bytes([i % 256]) * 4 for i in range(rows)]})


def _largest_call(lf: pl.LazyFrame) -> int:
    import polars_cv._lib as lib

    lib._take_max_split_rows()
    out = lf.select(pl.col("b").cv.pipe(_PIPE).sink("blob")).collect(engine="streaming")
    assert out.height > 0
    return lib._take_max_split_rows()


@plugin_required
def test_a_call_sees_at_most_one_parquet_row_group(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    _frame(600).write_parquet(path, row_group_size=100)
    largest = _largest_call(pl.scan_parquet(path))
    assert 0 < largest <= 100, largest


@plugin_required
def test_the_streaming_chunk_size_splits_a_large_row_group(tmp_path) -> None:
    path = tmp_path / "one_group.parquet"
    _frame(600).write_parquet(path, row_group_size=600)
    with pl.Config() as cfg:
        cfg.set_streaming_chunk_size(50)
        largest = _largest_call(pl.scan_parquet(path))
    assert 0 < largest <= 50, largest


@plugin_required
def test_the_streaming_chunk_size_bounds_an_in_memory_frames_calls() -> None:
    lf = _frame(1000).lazy()
    with pl.Config() as cfg:
        cfg.set_streaming_chunk_size(50)
        bounded = _largest_call(lf)
    assert 0 < bounded <= 50, bounded
    # Without it, the frame is split by the pipeline count alone.
    if pl.thread_pool_size() < 20:
        assert _largest_call(lf) > 50


@plugin_required
def test_collect_batches_streams_fixed_shape_tensors() -> None:
    from PIL import Image

    def png(i: int) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (40, 30), (i, 2 * i % 256, 3 * i % 256)).save(buf, "PNG")
        return buf.getvalue()

    lf = pl.LazyFrame({"img": [png(i) for i in range(10)]})
    pipe = Pipeline().source("image_bytes", dtype="u8").resize(height=16, width=16)
    tensors = lf.select(x=pl.col("img").cv.pipe(pipe).sink("array", shape=[16, 16, 3]))
    shapes = []
    for batch in tensors.collect_batches(chunk_size=4):
        x = batch["x"].to_numpy()
        assert x.dtype == np.uint8 and x.flags["C_CONTIGUOUS"]
        assert not x.flags["OWNDATA"], "the batch is a view of the column"
        shapes.append(x.shape)
    assert shapes == [(4, 16, 16, 3), (4, 16, 16, 3), (2, 16, 16, 3)]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
