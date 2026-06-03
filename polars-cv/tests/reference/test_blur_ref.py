"""
Reference tests for Gaussian blur against OpenCV.

polars-cv blurs via the `image` crate, whose Gaussian kernel derivation and
border handling differ from `cv2.GaussianBlur`, so the two do NOT match
byte-for-byte on high-frequency content (random noise can differ by tens of
levels at edges). On smooth / natural images, however, they agree to within a
pixel level or two, which is what these tests assert via a documented tolerance.

Native-dtype behaviour (blur preserving u16/f32) is covered in the Rust suite
(`view-buffer/tests/blur_dtype.rs`); here we pin the u8 path to OpenCV.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

# On smooth images the two implementations are essentially identical; these
# bounds (measured mean ~0.03-0.06, max ~1-2) leave margin without being loose.
MEAN_TOL = 0.5
MAX_TOL = 3


def _encode_png(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _run_pipe(pipe: Pipeline, arr: np.ndarray) -> np.ndarray:
    df = pl.DataFrame({"img": [_encode_png(arr)]})
    result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(result.row(0)[0]).squeeze()


def _assert_close_to_opencv(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    diff = np.abs(actual.astype(np.int32) - expected.astype(np.int32))
    assert diff.mean() < MEAN_TOL, f"mean abs diff {diff.mean():.3f} >= {MEAN_TOL}"
    assert diff.max() <= MAX_TOL, f"max abs diff {diff.max()} > {MAX_TOL}"


@pytest.fixture(scope="module")
def smooth_gray() -> np.ndarray:
    """128x128 smooth diagonal gradient (grayscale)."""
    yy, xx = np.mgrid[0:128, 0:128]
    return ((xx + yy) * 255 // 254).astype(np.uint8)


@pytest.fixture(scope="module")
def smooth_rgb() -> np.ndarray:
    """128x128 smooth gradient with a distinct ramp per channel."""
    yy, xx = np.mgrid[0:128, 0:128]
    r = (xx * 255 // 127).astype(np.uint8)
    g = (yy * 255 // 127).astype(np.uint8)
    b = ((xx + yy) * 255 // 254).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


@plugin_required
class TestBlurOpenCVParity:
    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    def test_blur_gray_matches_opencv(
        self, smooth_gray: np.ndarray, sigma: float
    ) -> None:
        cv2 = pytest.importorskip("cv2")
        expected = cv2.GaussianBlur(smooth_gray, (0, 0), sigmaX=sigma)
        pipe = Pipeline().source("image_bytes").blur(sigma)
        _assert_close_to_opencv(_run_pipe(pipe, smooth_gray), expected)

    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    def test_blur_rgb_matches_opencv(
        self, smooth_rgb: np.ndarray, sigma: float
    ) -> None:
        cv2 = pytest.importorskip("cv2")
        expected = cv2.GaussianBlur(smooth_rgb, (0, 0), sigmaX=sigma)
        pipe = Pipeline().source("image_bytes").blur(sigma)
        _assert_close_to_opencv(_run_pipe(pipe, smooth_rgb), expected)
