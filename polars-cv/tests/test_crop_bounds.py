"""A window reaching past the image's edge.

``crop(bounds=...)``: ``"error"`` (the default) refuses it and ``"clip"``
keeps the part inside the image. ``crop_pad`` keeps the requested size,
filling outside with ``value``. Checked against NumPy slicing and ``np.pad``.
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

H, W = 12, 31
IMAGE = (np.arange(H * W * 3, dtype=np.uint32) % 251).astype(np.uint8).reshape(H, W, 3)


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _run(pipe: Pipeline, **columns: list) -> list[np.ndarray]:
    rows = len(next(iter(columns.values()))) if columns else 1
    frame = pl.DataFrame({"img": [_png(IMAGE)] * rows, **columns})
    out = frame.select(pl.col("img").cv.pipe(pipe).sink("numpy")).to_series()
    return [numpy_from_struct(row) for row in out]


def _crop(**kwargs) -> Pipeline:
    return Pipeline().source("image_bytes").crop(**kwargs)


def _pad_reference(
    top: int, left: int, height: int, width: int, value: int
) -> np.ndarray:
    """The window ``[top, top + height) x [left, left + width)`` of the image
    on an infinite canvas of ``value``."""
    big = max(H, W) + height + width
    canvas = np.pad(IMAGE, ((big, big), (big, big), (0, 0)), constant_values=value)
    return canvas[big + top : big + top + height, big + left : big + left + width]


class TestClip:
    def test_a_strip_past_the_right_edge_keeps_the_part_inside(self) -> None:
        (got,) = _run(_crop(left=8, width=24, bounds="clip"))
        np.testing.assert_array_equal(got, IMAGE[:, 8:32])
        assert got.shape == (H, 23, 3)

    def test_a_negative_offset_clips_at_zero(self) -> None:
        (got,) = _run(_crop(top=-3, left=-4, height=8, width=10, bounds="clip"))
        np.testing.assert_array_equal(got, IMAGE[0:5, 0:6])

    def test_a_window_inside_is_unchanged(self) -> None:
        (got,) = _run(_crop(top=1, left=2, height=4, width=5, bounds="clip"))
        np.testing.assert_array_equal(got, IMAGE[1:5, 2:7])

    def test_per_row_offsets(self) -> None:
        got = _run(
            _crop(left=pl.col("l"), width=24, bounds="clip"),
            l=[0, 8, 20],
        )
        for row, left in zip(got, [0, 8, 20]):
            np.testing.assert_array_equal(row, IMAGE[:, left : left + 24])

    def test_a_window_entirely_outside_is_refused(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError, match="entirely outside"):
            _run(_crop(left=40, width=4, bounds="clip"))


def _crop_pad(**kwargs) -> Pipeline:
    return Pipeline().source("image_bytes").crop_pad(**kwargs)


class TestPad:
    """``crop_pad``: the window at its requested size, outside the image
    filled with ``value``. Its own op, so the size it pads to is required
    and the fill exists only where it is read."""

    @pytest.mark.parametrize(
        ("top", "left", "height", "width"),
        [
            (0, 8, 12, 24),
            (-2, -3, 5, 6),
            (10, 29, 4, 4),
            (-1, -1, H + 2, W + 2),
            (2, 3, 4, 5),
        ],
    )
    def test_matches_np_pad(self, top, left, height, width) -> None:
        (got,) = _run(
            _crop_pad(top=top, left=left, height=height, width=width, value=7)
        )
        assert got.shape == (height, width, 3)
        np.testing.assert_array_equal(got, _pad_reference(top, left, height, width, 7))

    def test_value_defaults_to_zero(self) -> None:
        (got,) = _run(_crop_pad(left=28, height=H, width=6))
        np.testing.assert_array_equal(got, _pad_reference(0, 28, H, 6, 0))

    def test_a_window_entirely_outside_is_all_padding(self) -> None:
        (got,) = _run(_crop_pad(left=40, height=3, width=4, value=9))
        np.testing.assert_array_equal(got, np.full((3, 4, 3), 9, np.uint8))

    def test_a_per_row_fill(self) -> None:
        got = _run(_crop_pad(left=29, height=2, width=4, value=pl.col("v")), v=[3, 200])
        for row, value in zip(got, [3, 200]):
            np.testing.assert_array_equal(row, _pad_reference(0, 29, 2, 4, value))

    def test_the_lazy_schema_knows_the_padded_size(self) -> None:
        frame = pl.DataFrame({"img": [_png(IMAGE)]}).lazy()
        pipe = (
            Pipeline()
            .source("image_bytes", dtype="u8")
            .crop_pad(left=8, height=12, width=24)
            # A decoded image's channel count is unknown at plan time.
            .channel_select(0, keepdims=False)
        )
        query = frame.select(pl.col("img").cv.pipe(pipe).sink("array"))
        schema = query.collect_schema()
        assert schema["img"] == pl.Array(pl.UInt8, (12, 24))
        assert query.collect().schema == schema


class TestRefused:
    def test_a_negative_offset_without_bounds_names_it(self) -> None:
        with pytest.raises(ValueError, match=r"crop: top = -1 .*bounds"):
            _crop(top=-1, height=2)

    def test_a_negative_per_row_offset_without_bounds_names_it(self) -> None:
        with pytest.raises(
            pl.exceptions.ComputeError, match=r"crop: left = -2 .*bounds"
        ):
            _run(_crop(left=pl.col("l"), width=4), l=[-2])

    def test_a_padded_window_is_crop_pad_not_a_bound(self) -> None:
        """``crop``'s bounds are what a window past the edge keeps: an error
        or the part inside. Padding is ``crop_pad``, whose fill and size are
        its own required fields rather than ``crop`` fields read under one
        bound only."""
        with pytest.raises(
            ValueError,
            match=r'unknown CropBounds "pad", expected one of \["error", "clip"\]',
        ):
            _crop(left=2, height=2, width=4, bounds="pad")

    @pytest.mark.parametrize(
        "build",
        [
            lambda p: p.crop(top=-1, height=2, width=2, bounds="clip"),
            lambda p: p.crop_pad(top=-1, height=2, width=2),
        ],
    )
    def test_a_window_on_a_1d_buffer_is_refused_not_panicked(self, build) -> None:
        """A window is over the image plane. A 1-D buffer has none: refused
        as the pipeline is built (the rank is known), where it used to reach
        the engine and panic placing the window."""
        frame = pl.DataFrame(
            {"a": pl.Series([[1, 2, 3, 4]], dtype=pl.Array(pl.UInt8, 4))}
        )
        with pytest.raises((ValueError, pl.exceptions.ComputeError)) as err:
            frame.select(
                pl.col("a").cv.pipe(build(Pipeline().source("array"))).sink("list")
            )
        assert "panicked" not in str(err.value), str(err.value)
        assert "[H, W]" in str(err.value), str(err.value)
