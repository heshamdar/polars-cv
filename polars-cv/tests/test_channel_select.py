"""``channel_select`` keeps the channel axis unless asked not to.

``channel_select(i)`` gives ``[H, W, 1]``, the shape ``grayscale`` and
``rasterize`` give, so the three combine without reshaping;
``keepdims=False`` gives ``[H, W]``. A ``[H, W]`` input is returned as it is.
``keepdims`` changes the rank, so it is a literal, never a per-row value.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests._plan_view import planned
from tests.conftest import plugin_required


def _sized() -> Pipeline:
    return (
        Pipeline()
        .source("image_bytes", dtype="u8")
        .assert_shape(height=4, width=5, channels=3)
    )


class TestPlanned:
    def test_the_channel_axis_is_kept_by_default(self) -> None:
        assert planned(_sized().channel_select(1)).dims == (4, 5, 1)

    def test_keepdims_false_drops_it(self) -> None:
        assert planned(_sized().channel_select(1, keepdims=False)).dims == (4, 5)

    def test_a_2d_input_is_returned_as_it_is(self) -> None:
        flat = _sized().channel_select(0, keepdims=False)
        assert planned(flat.channel_select(0)).dims == (4, 5)
        assert planned(flat.channel_select(0, keepdims=False)).dims == (4, 5)

    def test_keepdims_is_a_literal(self) -> None:
        with pytest.raises((TypeError, ValueError)):
            _sized().channel_select(0, keepdims=pl.col("k"))  # ty: ignore[invalid-argument-type]

    def test_a_kept_channel_combines_with_grayscale(self) -> None:
        img = pl.col("image").cv.pipe(_sized())
        both = img.pipe(Pipeline().grayscale()).maximum(img.channel_select(2))
        assert planned(both).dims == (4, 5, 1)


@plugin_required
class TestExecuted:
    @pytest.fixture
    def image(self) -> np.ndarray:
        return np.random.default_rng(5).integers(0, 256, (4, 5, 3), dtype=np.uint8)

    @pytest.fixture
    def df(self, image: np.ndarray, encode_png: Callable) -> pl.DataFrame:
        return pl.DataFrame({"image": [encode_png(image)]})

    @staticmethod
    def _run(df: pl.DataFrame, pipe: Pipeline) -> np.ndarray:
        out = df.select(o=pl.col("image").cv.pipe(pipe).sink("numpy")).item()
        return numpy_from_struct(out)

    def test_kept(self, df: pl.DataFrame, image: np.ndarray) -> None:
        out = self._run(df, _sized().channel_select(1))
        np.testing.assert_array_equal(out, image[:, :, 1:2])

    def test_dropped(self, df: pl.DataFrame, image: np.ndarray) -> None:
        out = self._run(df, _sized().channel_select(1, keepdims=False))
        np.testing.assert_array_equal(out, image[:, :, 1])

    def test_a_per_row_index_keeps_the_axis(
        self, df: pl.DataFrame, image: np.ndarray
    ) -> None:
        pipe = _sized().channel_select(pl.lit(2, dtype=pl.UInt32))
        np.testing.assert_array_equal(self._run(df, pipe), image[:, :, 2:3])
