"""Out-of-range crop windows and ragged raw buffers are errors (CR-42).

``crop`` used to clamp: a negative ``top`` became 0 while the height was kept
(a *shifted* window), a window past the edge came back smaller, and a
``width`` given without a ``height`` was dropped altogether. ``raw`` divided
the byte length by the element size and discarded the remainder. Each returned
data the caller did not ask for, without a word. Now each one is rejected: a
literal at build time, anything data-dependent as a row error that
``on_error`` handles.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline

from .conftest import plugin_required

H, W = 20, 16


def _image_frame() -> pl.DataFrame:
    arr = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)
    return pl.DataFrame({"a": pl.Series("a", arr[np.newaxis])})


def _crop(**kwargs: object) -> np.ndarray:
    pipe = Pipeline().source("array").crop(**kwargs)  # ty: ignore[invalid-argument-type]
    out = _image_frame().select(o=pl.col("a").cv.pipe(pipe).sink("numpy"))
    from polars_cv import numpy_from_struct

    return numpy_from_struct(out["o"][0])


@plugin_required
class TestCropBounds:
    def test_an_in_bounds_window_is_exact(self) -> None:
        got = _crop(top=2, left=3, height=5, width=4)
        expected = np.arange(H * W * 3, dtype=np.uint8).reshape(H, W, 3)[2:7, 3:7]
        np.testing.assert_array_equal(got, expected)

    def test_width_alone_is_honoured(self) -> None:
        """``height=None`` means "to the end"; it must not discard ``width``."""
        assert _crop(top=0, left=0, width=4).shape == (H, 4, 3)

    def test_height_alone_is_honoured(self) -> None:
        assert _crop(top=0, left=0, height=5).shape == (5, W, 3)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"top": -5, "left": 0, "height": 10, "width": 10},
            {"top": 0, "left": -1},
            {"top": 0, "left": 0, "height": -3, "width": 4},
        ],
    )
    def test_a_negative_literal_is_rejected_at_build_time(
        self, kwargs: dict[str, int]
    ) -> None:
        with pytest.raises(ValueError, match="negative"):
            Pipeline().source("array").crop(**kwargs)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"top": 15, "left": 0, "height": 10, "width": 4},  # rows 15..25 of 20
            {"top": 0, "left": 10, "height": 4, "width": 10},  # cols 10..20 of 16
            {"top": 21, "left": 0},  # origin below the image
        ],
    )
    def test_a_window_outside_the_image_is_a_row_error(
        self, kwargs: dict[str, int]
    ) -> None:
        with pytest.raises(pl.exceptions.ComputeError, match="outside"):
            _crop(**kwargs)

    def test_a_negative_per_row_offset_is_a_row_error(self) -> None:
        df = _image_frame().with_columns(t=pl.lit(-5))
        pipe = Pipeline().source("array").crop(top=pl.col("t"), left=0, height=4)
        with pytest.raises(pl.exceptions.ComputeError, match="negative"):
            df.select(o=pl.col("a").cv.pipe(pipe).sink("numpy"))

    def test_an_overrunning_row_is_nulled_under_on_error_null(self) -> None:
        df = pl.concat([_image_frame()] * 2).with_columns(t=pl.Series([0, 15]))
        pipe = (
            Pipeline()
            .source("array")
            .crop(top=pl.col("t"), left=0, height=10, width=4)
            .on_error("null")
        )
        out = df.select(o=pl.col("a").cv.pipe(pipe).sink("numpy"))["o"]
        assert out[0] is not None
        assert out[1] is None


@plugin_required
class TestRawByteLength:
    def test_a_whole_number_of_elements_decodes(self) -> None:
        df = pl.DataFrame({"b": [np.arange(3, dtype=np.float32).tobytes()]})
        out = df.select(
            o=pl.col("b").cv.pipe(Pipeline().source("raw", dtype="f32")).sink("list")
        )
        assert out["o"][0].to_list() == [0.0, 1.0, 2.0]

    def test_a_ragged_byte_length_is_a_row_error(self) -> None:
        df = pl.DataFrame({"b": [bytes(10)]})
        with pytest.raises(pl.exceptions.ComputeError, match="multiple"):
            df.select(
                o=pl.col("b")
                .cv.pipe(Pipeline().source("raw", dtype="f32"))
                .sink("list")
            )
