"""``grayscale(keepdims=False)``: luma without the channel axis.

A decoded grayscale image is ``[H, W, 1]``, so an axis reduction over it
returned one-element lists per column. ``grayscale(keepdims=False)`` gives
``[H, W]`` from any input (gray, RGB, RGBA), known to the lazy schema.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct

from .conftest import plugin_required

pytestmark = plugin_required

H, W = 5, 7
RNG = np.random.default_rng(3)
RGB = RNG.integers(0, 256, (H, W, 3), dtype=np.uint8)
RGBA = RNG.integers(0, 256, (H, W, 4), dtype=np.uint8)
GRAY = RNG.integers(0, 256, (H, W), dtype=np.uint8)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _run(pipe: Pipeline, *images: np.ndarray) -> list[np.ndarray]:
    frame = pl.DataFrame({"img": [_png(a) for a in images]})
    out = frame.select(pl.col("img").cv.pipe(pipe).sink("numpy")).to_series()
    return [numpy_from_struct(row) for row in out]


def _gray_of(arr: np.ndarray) -> np.ndarray:
    """``grayscale()``'s output for one decoded image: the one luma authority."""
    (out,) = _run(Pipeline().source("image_bytes").grayscale(), arr)
    return out


class TestGrayscaleKeepdims:
    def test_drops_the_channel_axis(self) -> None:
        (got,) = _run(Pipeline().source("image_bytes").grayscale(keepdims=False), RGB)
        assert got.shape == (H, W)
        np.testing.assert_array_equal(got, _gray_of(RGB)[..., 0])

    def test_keeps_it_by_default(self) -> None:
        (got,) = _run(Pipeline().source("image_bytes").grayscale(), RGB)
        assert got.shape == (H, W, 1)

    def test_an_axis_reduction_gives_one_value_per_column(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale(keepdims=False)
            .cast("f64")
            .reduce_mean(axis=0)
        )
        (got,) = _run(pipe, RGB)
        np.testing.assert_allclose(got, _gray_of(RGB)[..., 0].mean(axis=0))

    def test_the_lazy_schema_has_rank_two(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .grayscale(keepdims=False)
            .resize(height=4, width=6)
        )
        query = (
            pl.DataFrame({"img": [_png(RGB)]})
            .lazy()
            .select(pl.col("img").cv.pipe(pipe).sink("array"))
        )
        assert query.collect_schema()["img"] == pl.Array(pl.UInt8, (4, 6))
        assert query.collect().schema == query.collect_schema()
