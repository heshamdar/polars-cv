"""Area resampling and block reduction, against OpenCV and NumPy.

``resize(filter="area")`` is each output pixel's footprint mean when
shrinking, every input pixel weighted by its overlap (OpenCV's
``INTER_AREA``), which avoids the aliasing of a narrow kernel at large
factors; ``block_reduce`` pools fixed blocks with any reduction.
"""

from __future__ import annotations

import io
import math

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct

from .conftest import plugin_required

pytestmark = plugin_required

RNG = np.random.default_rng(5)
RGB = RNG.integers(0, 256, (37, 53, 3), dtype=np.uint8)
MAP = RNG.random((23, 30))


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _run_bytes(pipe: Pipeline, arr: np.ndarray) -> np.ndarray:
    frame = pl.DataFrame({"img": [_png(arr)]})
    return numpy_from_struct(
        frame.select(pl.col("img").cv.pipe(pipe).sink("numpy")).item()
    )


def _run_array(pipe: Pipeline, arr: np.ndarray) -> np.ndarray:
    frame = pl.DataFrame({"a": pl.Series("a", arr[np.newaxis])})
    return numpy_from_struct(
        frame.select(pl.col("a").cv.pipe(pipe).sink("numpy")).item()
    )


def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """``[n_out, n_in]``: each output pixel's overlap with each input pixel,
    over its footprint's length — the definition, computed directly."""
    s = n_in / n_out
    lo = np.arange(n_out)[:, None] * s
    k = np.arange(n_in)[None, :]
    return np.clip(np.minimum(lo + s, k + 1) - np.maximum(lo, k), 0, None) / s


class TestAreaResize:
    @pytest.mark.parametrize(("h", "w"), [(10, 13), (4, 4), (37, 20), (12, 53)])
    def test_u8_matches_opencv_inter_area(self, h: int, w: int) -> None:
        cv2 = pytest.importorskip("cv2")
        pipe = Pipeline().source("image_bytes").resize(height=h, width=w, filter="area")
        got = _run_bytes(pipe, RGB)
        want = cv2.resize(RGB, (w, h), interpolation=cv2.INTER_AREA)
        assert np.abs(got.astype(int) - want.astype(int)).max() <= 1

    @pytest.mark.parametrize(("h", "w"), [(5, 7), (7, 11), (23, 9)])
    def test_float_is_the_exact_overlap_mean(self, h: int, w: int) -> None:
        pipe = Pipeline().source("array").resize(height=h, width=w, filter="area")
        got = _run_array(pipe, MAP)
        want = _area_weights(MAP.shape[0], h) @ MAP @ _area_weights(MAP.shape[1], w).T
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_an_integer_factor_is_the_block_mean(self) -> None:
        img = MAP[:20, :28]
        got = _run_array(
            Pipeline().source("array").resize(height=5, width=7, filter="area"), img
        )
        want = img.reshape(5, 4, 7, 4).mean(axis=(1, 3))
        np.testing.assert_allclose(got, want, rtol=1e-9)


def _blocks(arr: np.ndarray, bh: int, bw: int, fn) -> np.ndarray:
    h, w = arr.shape[:2]
    out = np.empty((math.ceil(h / bh), math.ceil(w / bw), *arr.shape[2:]))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            block = arr[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            out[i, j] = fn(block.reshape(-1, *arr.shape[2:]), axis=0)
    return out


class TestBlockReduce:
    @pytest.mark.parametrize(
        ("reduction", "fn"),
        [("mean", np.mean), ("max", np.max), ("median", np.median), ("std", np.std)],
    )
    @pytest.mark.parametrize(("bh", "bw"), [(4, 5), (5, 7), (1, 30)])
    def test_matches_numpy_with_partial_edge_blocks(
        self, reduction, fn, bh, bw
    ) -> None:
        pipe = (
            Pipeline()
            .source("array")
            .block_reduce(height=bh, width=bw, reduction=reduction)
        )
        np.testing.assert_allclose(_run_array(pipe, MAP), _blocks(MAP, bh, bw, fn))

    def test_channels_are_pooled_separately(self) -> None:
        pipe = (
            Pipeline()
            .source("image_bytes")
            .block_reduce(height=8, width=8, reduction="max")
        )
        got = _run_bytes(pipe, RGB)
        assert got.shape == (5, 7, 3)
        np.testing.assert_allclose(got, _blocks(RGB.astype(float), 8, 8, np.max))

    def test_percentile_with_q(self) -> None:
        pipe = (
            Pipeline()
            .source("array")
            .block_reduce(height=4, width=4, reduction="percentile", q=90.0)
        )
        np.testing.assert_allclose(
            _run_array(pipe, MAP),
            _blocks(MAP, 4, 4, lambda b, axis: np.percentile(b, 90.0, axis=axis)),
        )

    def test_the_lazy_schema_knows_the_pooled_size(self) -> None:
        pipe = (
            Pipeline().source("array").block_reduce(height=4, width=7, reduction="mean")
        )
        query = (
            pl.DataFrame({"a": pl.Series("a", MAP[np.newaxis])})
            .lazy()
            .select(pl.col("a").cv.pipe(pipe).sink("array"))
        )
        assert query.collect_schema()["a"] == pl.Array(pl.Float64, (6, 5))
        assert query.collect().schema == query.collect_schema()

    def test_a_zero_block_is_refused(self) -> None:
        with pytest.raises(ValueError, match="block"):
            Pipeline().source("array").block_reduce(height=0, width=2)
