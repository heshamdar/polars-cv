"""Canny against OpenCV: the edge maps are identical, pixel for pixel.

``canny`` computes what ``cv2.Canny(image, low, high)`` computes (aperture 3,
L1 gradient): 3x3 Sobel with a replicated border, no pre-blur, non-maximum
suppression over four directions, and 8-connected hysteresis. Blur first with
``.blur(...)`` when a smoothed edge map is wanted, as with OpenCV.

A colour input takes, per pixel, the channel with the largest gradient, as
OpenCV does; an alpha channel is not an input to the edges.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

cv2 = pytest.importorskip("cv2")

pytestmark = plugin_required


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _canny(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    pipe = (
        Pipeline().source("image_bytes").canny(low_threshold=low, high_threshold=high)
    )
    df = pl.DataFrame({"img": [_png(arr)]})
    out = numpy_from_struct(
        df.select(pl.col("img").cv.pipe(pipe).sink("numpy"))["img"][0]
    )
    assert out.shape == (*arr.shape[:2], 1)
    return out[..., 0]


def _smooth(
    rng: np.random.Generator, h: int, w: int, c: int | None = None
) -> np.ndarray:
    shape = (max(h // 6, 2), max(w // 6, 2)) + ((c,) if c else ())
    small = rng.integers(0, 256, shape).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _gray_images() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    rect = rng.integers(90, 110, (64, 80)).astype(np.uint8)
    rect[16:48, 20:60] = 200
    yy, xx = np.mgrid[0:60, 0:60]
    return {
        "smooth": _smooth(rng, 120, 150),
        "noise": rng.integers(0, 256, (64, 64)).astype(np.uint8),
        "rectangle_on_noise": rect,
        "ramp": np.tile(np.arange(0, 250, 5, dtype=np.uint8), (40, 1)),
        "diagonal_stripes": (((xx + yy) // 4 % 2) * 255).astype(np.uint8),
        "disc": (((xx - 30) ** 2 + (yy - 28) ** 2 < 300) * 220).astype(np.uint8),
    }


GRAY = _gray_images()
THRESHOLDS = [(50.0, 150.0), (10.0, 30.0), (0.0, 0.0), (50.5, 150.7), (200.0, 200.0)]


@pytest.mark.parametrize("name", sorted(GRAY))
@pytest.mark.parametrize(("low", "high"), THRESHOLDS)
def test_gray_matches_opencv(name: str, low: float, high: float) -> None:
    img = GRAY[name]
    np.testing.assert_array_equal(_canny(img, low, high), cv2.Canny(img, low, high))


def test_thresholds_in_either_order() -> None:
    # OpenCV swaps a low threshold above the high one; so does this.
    img = GRAY["smooth"]
    np.testing.assert_array_equal(_canny(img, 150.0, 50.0), cv2.Canny(img, 50.0, 150.0))


@pytest.mark.parametrize(
    "shape", [(1, 1), (1, 9), (9, 1), (2, 2), (3, 3), (3, 40), (40, 3)]
)
def test_tiny_images_match_opencv(shape: tuple[int, int]) -> None:
    img = np.random.default_rng(3).integers(0, 256, shape).astype(np.uint8)
    np.testing.assert_array_equal(_canny(img, 20.0, 60.0), cv2.Canny(img, 20.0, 60.0))


@pytest.mark.parametrize("kind", ["smooth", "noise"])
def test_colour_takes_the_strongest_channel_like_opencv(kind: str) -> None:
    rng = np.random.default_rng(11)
    img = (
        _smooth(rng, 90, 110, 3)
        if kind == "smooth"
        else rng.integers(0, 256, (48, 56, 3)).astype(np.uint8)
    )
    np.testing.assert_array_equal(_canny(img, 50.0, 150.0), cv2.Canny(img, 50.0, 150.0))


def test_alpha_is_not_an_edge_input() -> None:
    rng = np.random.default_rng(13)
    rgb = _smooth(rng, 70, 90, 3)
    alpha = rng.integers(0, 256, (70, 90, 1)).astype(np.uint8)  # edges everywhere
    rgba = np.concatenate([rgb, alpha], axis=2)
    np.testing.assert_array_equal(
        _canny(rgba, 50.0, 150.0), cv2.Canny(rgb, 50.0, 150.0)
    )


def test_gray_alpha_uses_the_gray_channel() -> None:
    rng = np.random.default_rng(17)
    gray = _smooth(rng, 60, 70)
    alpha = rng.integers(0, 256, (60, 70)).astype(np.uint8)
    la = np.stack([gray, alpha], axis=2)
    np.testing.assert_array_equal(_canny(la, 30.0, 90.0), cv2.Canny(gray, 30.0, 90.0))


@pytest.mark.parametrize("dtype", ["f32", "u16"])
def test_a_non_u8_image_of_the_same_values_matches_opencv(dtype: str) -> None:
    # u8 runs OpenCV's integer arithmetic; every other dtype the same
    # definition in f64, which is exact on these values.
    img = GRAY["rectangle_on_noise"]
    pipe = (
        Pipeline()
        .source("image_bytes")
        .cast(dtype)
        .canny(low_threshold=50.0, high_threshold=150.0)
    )
    df = pl.DataFrame({"img": [_png(img)]})
    got = numpy_from_struct(
        df.select(pl.col("img").cv.pipe(pipe).sink("numpy"))["img"][0]
    )
    np.testing.assert_array_equal(got[..., 0], cv2.Canny(img, 50.0, 150.0))
