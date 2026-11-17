"""A pipeline that starts with ``crop`` decodes a PNG only down to the crop's
last row (the compiled graph's ``rows_needed``). The crop still runs, so the
result must equal slicing the full decode, for every PNG mode: those the
partial decoder takes (8/16-bit gray, gray+alpha, RGB, RGBA) and those it
declines (palette, transparency key), which decode in full.
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

RNG = np.random.default_rng(20261117)


def _png(mode: str, h: int, w: int) -> tuple[bytes, np.ndarray]:
    """A noisy PNG in Pillow ``mode`` and the array the full decode gives."""
    if mode == "I;16":
        arr = RNG.integers(0, 65535, (h, w), dtype=np.uint16)
        img = Image.fromarray(arr)  # uint16, 2-D: mode I;16
    else:
        channels = {"L": 1, "LA": 2, "RGB": 3, "RGBA": 4, "P": 1}[mode]
        arr = RNG.integers(0, 256, (h, w, channels), dtype=np.uint8)
        img = Image.fromarray(arr.squeeze(-1) if channels == 1 else arr)
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    data = buf.getvalue()
    full = pl.DataFrame({"img": [data]}).select(
        pl.col("img").cv.pipe(Pipeline().source("image_bytes")).sink("tensor")
    )
    return data, numpy_from_struct(full.to_series()[0])


def _cropped(data: bytes, **crop) -> np.ndarray:
    pipe = Pipeline().source("image_bytes").crop(**crop)
    out = pl.DataFrame({"img": [data]}).select(
        pl.col("img").cv.pipe(pipe).sink("tensor")
    )
    return numpy_from_struct(out.to_series()[0])


@pytest.mark.parametrize("mode", ["L", "LA", "RGB", "RGBA", "I;16", "P"])
@pytest.mark.parametrize("seed", range(6))
def test_a_leading_crop_equals_slicing_the_full_decode(mode: str, seed: int) -> None:
    h, w = int(RNG.integers(2, 90)), int(RNG.integers(1, 90))
    data, full = _png(mode, h, w)
    top = int(RNG.integers(0, h - 1))
    height = int(RNG.integers(1, h - top))
    left = int(RNG.integers(0, w))
    width = int(RNG.integers(1, w - left + 1))
    got = _cropped(data, top=top, left=left, height=height, width=width)
    want = full[top : top + height, left : left + width]
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_a_crop_to_the_last_row_and_without_a_height() -> None:
    data, full = _png("RGB", 40, 30)
    np.testing.assert_array_equal(_cropped(data, top=5, height=35), full[5:40])
    np.testing.assert_array_equal(_cropped(data, top=5), full[5:])


def test_a_per_row_crop_decodes_each_row_in_full() -> None:
    data, full = _png("RGB", 40, 30)
    pipe = Pipeline().source("image_bytes").crop(top=pl.col("t"), height=6)
    out = pl.DataFrame({"img": [data] * 3, "t": [0, 10, 34]}).select(
        pl.col("img").cv.pipe(pipe).sink("tensor")
    )
    for row, t in zip(out.to_series(), [0, 10, 34]):
        np.testing.assert_array_equal(numpy_from_struct(row), full[t : t + 6])


def test_a_crop_after_another_op_is_not_pushed_down() -> None:
    data, full = _png("RGB", 40, 30)
    pipe = Pipeline().source("image_bytes").flip_v().crop(top=2, height=5)
    out = pl.DataFrame({"img": [data]}).select(
        pl.col("img").cv.pipe(pipe).sink("tensor")
    )
    np.testing.assert_array_equal(
        numpy_from_struct(out.to_series()[0]), full[::-1][2:7]
    )
