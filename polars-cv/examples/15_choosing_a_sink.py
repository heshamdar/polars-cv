"""Choosing a sink: what each one gives, and which to reach for.

Demonstrates:
- `array`: a fixed-shape batch, one NumPy array for the whole column
- `list`: ragged images as nested lists
- `tensor`: one zero-copy tensor per row, for a loader
- `tensor` with `compact=True`: tensors that persist (Parquet) and read back
- `png` / `jpeg`: encoded bytes
- `blob`: bytes another pipeline reads back, dtype and shape included
- Several outputs from one decode, each with its own sink

The docs guide (docs/user-guide/concepts/pipelines.md, "Choosing a sink")
quotes the marked sections of this file, and tests/test_examples.py runs it.

Run:
    uv run python polars-cv/examples/15_choosing_a_sink.py
"""

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from polars_cv import Pipeline, Sink, numpy_from_struct


def make_images() -> pl.DataFrame:
    """Three PNGs of different sizes."""
    rows = []
    for i, (h, w) in enumerate([(40, 60), (50, 50), (64, 48)]):
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.stack([xx * 4, yy * 4, np.full_like(xx, 40 * i)], axis=-1)
        buf = BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
        rows.append({"image": buf.getvalue()})
    return pl.DataFrame(rows)


def main() -> None:
    df = make_images()
    decoded = Pipeline().source("image_bytes", dtype="u8")
    resized = decoded.resize(height=32, width=32)

    # --8<-- [start:array]
    # Every row the same shape, and the whole column wanted as one array. A
    # decoded image's channel count is only known per file, so the sink is
    # told the shape (a row that does not have it is an error).
    batch = df.select(
        x=pl.col("image").cv.pipe(resized).sink("array", shape=[32, 32, 3])
    )
    assert batch.schema["x"] == pl.Array(pl.UInt8, (32, 32, 3))
    assert batch["x"].to_numpy().shape == (3, 32, 32, 3)
    # --8<-- [end:array]

    # --8<-- [start:list]
    # Rows of different shapes: nested lists, which Polars can work on.
    ragged = df.select(x=pl.col("image").cv.pipe(decoded).sink("list"))
    assert ragged.schema["x"] == pl.List(pl.List(pl.List(pl.UInt8)))
    assert ragged["x"].list.len().to_list() == [40, 50, 64]  # each row's height
    # --8<-- [end:list]

    # --8<-- [start:tensor]
    # One tensor per row for a data loader: no copy on the way out.
    tensors = df.select(x=pl.col("image").cv.pipe(decoded).sink("tensor"))
    shapes = [numpy_from_struct(row).shape for row in tensors["x"]]
    assert shapes == [(40, 60, 3), (50, 50, 3), (64, 48, 3)]
    # --8<-- [end:tensor]

    # --8<-- [start:persist]
    # Persisting tensors: `compact=True` makes each row's data exactly its
    # elements, so Parquet does not write the storage a view was cut from.
    stored = df.select(x=pl.col("image").cv.pipe(decoded).sink("tensor", compact=True))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tensors.parquet"
        stored.write_parquet(path)
        back = pl.read_parquet(path)
    # The tensor sink reads back: the default source routes the column.
    again = back.select(
        x=pl.col("x").cv.pipe(Pipeline().source().flip_h()).sink("tensor")
    )
    assert numpy_from_struct(again["x"][0]).shape == (40, 60, 3)
    # --8<-- [end:persist]

    # --8<-- [start:encoded]
    # Persisting images: an encoder. PNG is lossless, JPEG takes a quality.
    files = df.select(
        png=pl.col("image").cv.pipe(resized).sink("png"),
        jpeg=pl.col("image").cv.pipe(resized).sink("jpeg", quality=80),
    )
    assert files["png"][0][:4] == b"\x89PNG"
    assert files["jpeg"][0][:2] == b"\xff\xd8"
    # --8<-- [end:encoded]

    # --8<-- [start:blob]
    # Feeding another pipeline: a blob carries its dtype and shape, so the
    # second pipeline needs nothing declared. Floats survive, unlike a PNG.
    stage_one = df.select(
        b=pl.col("image").cv.pipe(resized.scale(1 / 255)).sink("blob")
    )
    stage_two = stage_one.select(
        mean=pl.col("b").cv.pipe(Pipeline().source("blob").reduce_mean()).sink()
    )
    assert stage_two.schema["mean"] == pl.Float64
    # --8<-- [end:blob]

    # --8<-- [start:multi]
    # Several outputs from one decode, each with its own sink.
    base = pl.col("image").cv.pipe(decoded)
    x = base.pipe(Pipeline().resize(height=32, width=32)).alias("x")
    thumb = base.pipe(Pipeline().resize(height=16, width=16)).alias("thumb")
    both = df.select(
        o=x.merge_pipe(thumb).sink(
            {"x": Sink.array(shape=[32, 32, 3]), "thumb": Sink.jpeg(quality=70)}
        )
    ).unnest("o")
    assert both.schema["x"] == pl.Array(pl.UInt8, (32, 32, 3))
    assert both.schema["thumb"] == pl.Binary
    # --8<-- [end:multi]

    for name, frame in [
        ("array", batch),
        ("list", ragged),
        ("tensor", tensors),
        ("tensor (compact)", stored),
        ("png / jpeg", files),
        ("blob", stage_one),
        ("multi-output", both),
    ]:
        print(f"{name:>18}: {dict(frame.schema)}")


if __name__ == "__main__":
    main()
