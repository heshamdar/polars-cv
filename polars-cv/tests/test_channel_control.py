"""Choosing the channel axis: ``grayscale(keepdims=)`` and ``source(channels=)``.

A decoded grayscale image is ``[H, W, 1]``, and a decoded image's channel
count is unknown until it is decoded, so an axis reduction over it returned
one-element lists per column and an ``array`` sink could not be planned.
``source(channels=1 | 3)`` makes every row decode to that many colour
channels, known to the lazy schema; ``grayscale(keepdims=False)`` drops the
channel axis.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
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


class TestSourceChannels:
    def test_one_channel_is_the_grayscale_luma(self) -> None:
        got = _run(Pipeline().source("image_bytes", channels=1), RGB, RGBA, GRAY)
        np.testing.assert_array_equal(got[0], _gray_of(RGB))
        np.testing.assert_array_equal(got[1], _gray_of(RGBA))
        np.testing.assert_array_equal(got[2], GRAY[..., None])

    def test_three_channels_replicate_gray_and_drop_alpha(self) -> None:
        got = _run(Pipeline().source("image_bytes", channels=3), RGB, RGBA, GRAY)
        np.testing.assert_array_equal(got[0], RGB)
        np.testing.assert_array_equal(got[1], RGBA[..., :3])
        np.testing.assert_array_equal(got[2], np.repeat(GRAY[..., None], 3, axis=2))

    def test_the_channel_count_is_known_to_the_lazy_schema(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8", channels=3)
            .resize(height=4, width=6)
        )
        query = (
            pl.DataFrame({"img": [_png(RGBA)]})
            .lazy()
            .select(pl.col("img").cv.pipe(pipe).sink("array"))
        )
        assert query.collect_schema()["img"] == pl.Array(pl.UInt8, (4, 6, 3))
        assert query.collect().schema == query.collect_schema()

    def test_a_file_path_source_takes_it_too(self, tmp_path) -> None:
        path = tmp_path / "a.png"
        path.write_bytes(_png(RGBA))
        frame = pl.DataFrame({"p": [str(path)]})
        pipe = Pipeline().source("file_path", channels=1)
        (got,) = [
            numpy_from_struct(r)
            for r in frame.select(pl.col("p").cv.pipe(pipe).sink("numpy")).to_series()
        ]
        np.testing.assert_array_equal(got, _gray_of(RGBA))

    @pytest.mark.parametrize("channels", [0, 2, 4])
    def test_another_count_is_refused(self, channels: int) -> None:
        with pytest.raises(ValueError, match="1 or 3"):
            Pipeline().source("image_bytes", channels=channels)
