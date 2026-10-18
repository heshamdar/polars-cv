"""
Reference tests for Gaussian blur against OpenCV.

polars-cv blurs via the `image` crate, whose Gaussian kernel derivation and
border handling differ from `cv2.GaussianBlur`, so the two do NOT match
byte-for-byte on high-frequency content (random noise can differ by tens of
levels at edges). On smooth / natural images, however, they agree to within a
pixel level or two, which is what these tests assert via a documented tolerance.

Native-dtype behaviour (blur preserving u16/f32) is covered in the Rust suite
(`view-buffer/tests/it/blur_dtype.rs`); here we pin the u8 path to OpenCV.
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
@pytest.mark.parity(
    ops=["blur"],
    against="OpenCV `cv2.GaussianBlur`, u8 gray and RGB, sigma 1 and 2",
    tolerance=f"mean diff < {MEAN_TOL}, max diff <= {MAX_TOL} levels (smooth images)",
)
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


@plugin_required
@pytest.mark.parity(
    ops=["blur"],
    against="Definition: sigma 0 is the identity (as in PIL, SciPy); sigma < 0, NaN, inf refused",
    tolerance="exact",
)
class TestBlurSigmaDomain:
    """sigma = 0 is no blur, as in PIL and SciPy; below 0 is refused.

    sigma = 0 used to build the kernel exp(-0/0) = NaN, so every pixel came
    back 0, and a negative sigma was silently treated as no blur.
    """

    def test_zero_sigma_is_the_identity(self, smooth_rgb: np.ndarray) -> None:
        out = _run_pipe(Pipeline().source("image_bytes").blur(0.0), smooth_rgb)
        np.testing.assert_array_equal(out, smooth_rgb)

    def test_zero_sigma_is_the_identity_on_u16(self, smooth_gray: np.ndarray) -> None:
        arr = smooth_gray.astype(np.uint16) * 257  # a 16-bit PNG decodes to u16
        out = _run_pipe(Pipeline().source("image_bytes").blur(0.0), arr)
        np.testing.assert_array_equal(out, arr)

    def test_zero_sigma_per_row_is_the_identity(self, smooth_gray: np.ndarray) -> None:
        df = pl.DataFrame({"img": [_encode_png(smooth_gray)] * 2, "s": [0.0, 1.0]})
        pipe = Pipeline().source("image_bytes").blur(pl.col("s"))
        rows = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))["out"]
        np.testing.assert_array_equal(numpy_from_struct(rows[0]).squeeze(), smooth_gray)
        assert not np.array_equal(numpy_from_struct(rows[1]).squeeze(), smooth_gray)

    def test_literal_negative_sigma_is_refused(self) -> None:
        with pytest.raises(Exception, match="blur: sigma -1"):
            Pipeline().source("image_bytes").blur(-1.0)

    @pytest.mark.parametrize("sigma", [float("nan"), float("inf")])
    def test_literal_non_finite_sigma_is_refused(self, sigma: float) -> None:
        """Refused before the op's check: the JSON wire has no NaN or inf."""
        with pytest.raises(Exception):  # noqa: B017
            Pipeline().source("image_bytes").blur(sigma)

    @pytest.mark.parametrize("sigma", [-1.0, float("nan"), float("inf")])
    def test_per_row_sigma_outside_the_domain_is_refused(
        self, smooth_gray: np.ndarray, sigma: float
    ) -> None:
        df = pl.DataFrame({"img": [_encode_png(smooth_gray)], "s": [sigma]})
        pipe = Pipeline().source("image_bytes").blur(pl.col("s"))
        with pytest.raises(Exception, match="blur: sigma"):
            df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
