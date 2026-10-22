"""Rank filters and flat-fielding, against SciPy and NumPy.

``rank_filter(height, width, q)`` is each pixel's window ``q``-th percentile
(linear interpolation, as every reduction here), per channel;
``median_filter(size)`` is its ``q=50``. ``background_divide(sigma)`` divides
an image by its own blur.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import ndimage

from polars_cv import Pipeline, numpy_from_struct

from .conftest import plugin_required

pytestmark = plugin_required

RNG = np.random.default_rng(9)
IMG = RNG.integers(0, 256, (17, 23), dtype=np.uint8)
RGB = RNG.integers(0, 256, (11, 13, 3), dtype=np.uint8)
FMAP = RNG.random((12, 15))

#: Our border names -> scipy's.
MODES = {"replicate": "nearest", "reflect": "reflect", "zero": "constant"}


def _run(pipe: Pipeline, arr: np.ndarray) -> np.ndarray:
    frame = pl.DataFrame({"a": pl.Series("a", arr[np.newaxis])})
    return numpy_from_struct(
        frame.select(pl.col("a").cv.pipe(pipe).sink("numpy")).item()
    )


class TestMedianFilter:
    @pytest.mark.parametrize("size", [1, 3, 5])
    @pytest.mark.parametrize("border", sorted(MODES))
    def test_matches_scipy(self, size: int, border: str) -> None:
        got = _run(Pipeline().source("array").median_filter(size, border=border), IMG)
        want = ndimage.median_filter(IMG, size=size, mode=MODES[border], cval=0)
        np.testing.assert_array_equal(got, want)
        assert got.dtype == np.uint8

    def test_channels_are_filtered_separately(self) -> None:
        got = _run(Pipeline().source("array").median_filter(3), RGB)
        want = ndimage.median_filter(RGB, size=(3, 3, 1), mode="nearest")
        np.testing.assert_array_equal(got, want)


class TestRankFilter:
    def test_a_one_by_25_running_median_along_rows(self) -> None:
        got = _run(Pipeline().source("array").rank_filter(height=1, width=25), FMAP)
        want = ndimage.median_filter(FMAP, size=(1, 25), mode="nearest")
        np.testing.assert_allclose(got, want)

    @pytest.mark.parametrize("q", [0.0, 100.0])
    def test_the_extremes_are_min_and_max_filters(self, q: float) -> None:
        got = _run(Pipeline().source("array").rank_filter(height=3, width=5, q=q), FMAP)
        fn = ndimage.minimum_filter if q == 0 else ndimage.maximum_filter
        np.testing.assert_allclose(got, fn(FMAP, size=(3, 5), mode="nearest"))

    def test_an_interpolated_percentile(self) -> None:
        got = _run(
            Pipeline().source("array").rank_filter(height=3, width=3, q=30.0), FMAP
        )
        padded = np.pad(FMAP, 1, mode="edge")
        want = np.array(
            [
                [np.percentile(padded[y : y + 3, x : x + 3], 30.0) for x in range(15)]
                for y in range(12)
            ]
        )
        np.testing.assert_allclose(got, want)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"height": 2, "width": 3}, "odd"),
            ({"height": 3, "width": 0}, "odd"),
            ({"height": 3, "width": 3, "q": 101.0}, r"\[0, 100\]"),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Pipeline().source("array").rank_filter(**kwargs)


def test_background_divide_is_the_image_over_its_blur() -> None:
    img = pl.col("a").cv.pipe(Pipeline().source("array"))
    frame = pl.DataFrame({"a": pl.Series("a", (FMAP + 1.0)[np.newaxis])})
    got = numpy_from_struct(
        frame.select(img.background_divide(sigma=2.0).sink("numpy")).item()
    )
    blur = numpy_from_struct(
        frame.select(img.pipe(Pipeline().blur(2.0)).sink("numpy")).item()
    )
    np.testing.assert_allclose(got, (FMAP + 1.0) / (blur + 1e-6), rtol=1e-5)
