"""Reference tests for Phase 3: Convolution, Edge Detection, Histogram Equalization.

Compares polars-cv plugin output against OpenCV / NumPy / scipy ground truth
to verify correctness of convolve2d, sobel, laplacian, sharpen, canny, and
equalize_histogram operations.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


def _encode_png(arr: np.ndarray) -> bytes:
    """Encode a numpy array as PNG bytes."""
    from PIL import Image

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _run_pipe(pipe: Pipeline, png_bytes: bytes) -> np.ndarray:
    """Execute a pipeline on a single PNG image and return numpy result."""
    df = pl.DataFrame({"img": [png_bytes]})
    result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(result.row(0)[0])


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def gray_image() -> np.ndarray:
    """64x64 grayscale test image with structured content."""
    rng = np.random.default_rng(42)
    img = rng.integers(20, 200, (64, 64), dtype=np.uint8)
    # Add a bright rectangle for edge tests
    img[20:40, 20:40] = 200
    return img


@pytest.fixture(scope="module")
def gray_png(gray_image: np.ndarray) -> bytes:
    """PNG-encoded grayscale test image."""
    return _encode_png(gray_image)


@pytest.fixture(scope="module")
def rgb_image() -> np.ndarray:
    """64x64x3 RGB test image with structured content."""
    rng = np.random.default_rng(42)
    img = rng.integers(20, 200, (64, 64, 3), dtype=np.uint8)
    img[20:40, 20:40, :] = 200
    return img


@pytest.fixture(scope="module")
def rgb_png(rgb_image: np.ndarray) -> bytes:
    """PNG-encoded RGB test image."""
    return _encode_png(rgb_image)


@pytest.fixture(scope="module")
def low_contrast_image() -> np.ndarray:
    """64x64 grayscale image with narrow intensity range (for equalization tests)."""
    rng = np.random.default_rng(42)
    return rng.integers(80, 120, (64, 64), dtype=np.uint8)


@pytest.fixture(scope="module")
def low_contrast_png(low_contrast_image: np.ndarray) -> bytes:
    """PNG-encoded low-contrast test image."""
    return _encode_png(low_contrast_image)


# ===========================================================================
# Convolution: Identity kernel
# ===========================================================================


@plugin_required
class TestConvolveIdentity:
    """Identity kernel should return input unchanged."""

    def test_identity_kernel_preserves_input(
        self, gray_png: bytes, gray_image: np.ndarray
    ) -> None:
        """An identity kernel (center=1, rest=0) should preserve the image."""
        identity = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        pipe = Pipeline().source("image_bytes").grayscale().convolve2d(identity, 3)
        result = _run_pipe(pipe, gray_png)

        # Result is f32 (promote_to_float), input was u8
        # Convert gray_image to f32 for comparison
        expected = gray_image.astype(np.float32)
        # Grayscale may produce [H, W, 1] so squeeze
        if result.ndim == 3 and result.shape[2] == 1:
            result = result.squeeze(-1)
        np.testing.assert_allclose(result, expected, atol=0.5)


# ===========================================================================
# Convolution: Box blur kernel
# ===========================================================================


@plugin_required
class TestConvolveBoxBlur:
    """Box blur kernel should produce uniform averaging."""

    def test_box_blur_3x3(self, gray_png: bytes, gray_image: np.ndarray) -> None:
        """3x3 box blur with normalize=True should average neighborhood."""
        box_kernel = [1.0] * 9
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .convolve2d(box_kernel, 3, normalize=True, border="replicate")
        )
        result = _run_pipe(pipe, gray_png)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result.squeeze(-1)

        assert result.shape == gray_image.shape
        assert result.dtype == np.float32

        # Interior pixels: manual computation of 3x3 mean
        y, x = 32, 32
        expected_val = gray_image[31:34, 31:34].astype(np.float32).mean()
        np.testing.assert_allclose(result[y, x], expected_val, atol=0.5)


# ===========================================================================
# Convolution: Sobel against OpenCV
# ===========================================================================


@plugin_required
class TestSobelRef:
    """Compare Sobel operator against OpenCV reference."""

    def test_sobel_x_vs_opencv(self, gray_png: bytes, gray_image: np.ndarray) -> None:
        """Sobel X gradient should match OpenCV cv2.Sobel."""
        cv2 = pytest.importorskip("cv2")

        pipe = Pipeline().source("image_bytes").grayscale().sobel(axis="x")
        result = _run_pipe(pipe, gray_png)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result.squeeze(-1)

        expected = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        # Border handling differs, so compare interior only
        h, w = gray_image.shape
        margin = 2
        np.testing.assert_allclose(
            result[margin : h - margin, margin : w - margin],
            expected[margin : h - margin, margin : w - margin],
            atol=1.0,
        )

    def test_sobel_y_vs_opencv(self, gray_png: bytes, gray_image: np.ndarray) -> None:
        """Sobel Y gradient should match OpenCV cv2.Sobel."""
        cv2 = pytest.importorskip("cv2")

        pipe = Pipeline().source("image_bytes").grayscale().sobel(axis="y")
        result = _run_pipe(pipe, gray_png)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result.squeeze(-1)

        expected = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
        h, w = gray_image.shape
        margin = 2
        np.testing.assert_allclose(
            result[margin : h - margin, margin : w - margin],
            expected[margin : h - margin, margin : w - margin],
            atol=1.0,
        )


# ===========================================================================
# Convolution: Laplacian against OpenCV
# ===========================================================================


@plugin_required
class TestLaplacianRef:
    """Compare Laplacian operator against OpenCV reference."""

    def test_laplacian_vs_opencv(self, gray_png: bytes, gray_image: np.ndarray) -> None:
        """Laplacian should match OpenCV cv2.Laplacian with ksize=1 (4-neighbor)."""
        cv2 = pytest.importorskip("cv2")

        pipe = Pipeline().source("image_bytes").grayscale().laplacian()
        result = _run_pipe(pipe, gray_png)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result.squeeze(-1)

        # Our kernel [0,1,0,1,-4,1,0,1,0] is the 4-neighbor Laplacian,
        # which corresponds to OpenCV's ksize=1.
        expected = cv2.Laplacian(gray_image, cv2.CV_32F, ksize=1)
        h, w = gray_image.shape
        margin = 1
        np.testing.assert_allclose(
            result[margin : h - margin, margin : w - margin],
            expected[margin : h - margin, margin : w - margin],
            atol=1.0,
        )


# ===========================================================================
# Canny Edge Detection
# ===========================================================================


@plugin_required
class TestCannyRef:
    """Canny edge detection reference tests."""

    def test_canny_produces_binary_output(self, gray_png: bytes) -> None:
        """Canny should output a U8 image with only 0 and 255 values."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .canny(low_threshold=50.0, high_threshold=150.0)
        )
        result = _run_pipe(pipe, gray_png)

        assert result.dtype == np.uint8
        unique_vals = np.unique(result)
        for v in unique_vals:
            assert v in (0, 255), f"Unexpected value {v} in Canny output"

    def test_canny_detects_rectangle_edges(self, gray_png: bytes) -> None:
        """Canny should detect edges around the bright rectangle in the test image."""
        pipe = (
            Pipeline()
            .source("image_bytes")
            .canny(low_threshold=30.0, high_threshold=100.0)
        )
        result = _run_pipe(pipe, gray_png)
        if result.ndim == 3:
            result = result.squeeze(-1)

        # The rectangle is at [20:40, 20:40], so edges should appear near boundaries
        edge_region = result[18:42, 18:42]
        assert edge_region.max() == 255, "Canny should detect edges around rectangle"

        # Interior of rectangle should have no edges
        interior = result[23:37, 23:37]
        assert interior.sum() == 0 or interior.mean() < 10, (
            "Interior of uniform rectangle should be mostly edge-free"
        )

    def test_canny_vs_opencv_structure(
        self, gray_png: bytes, gray_image: np.ndarray
    ) -> None:
        """Canny edges should overlap significantly with OpenCV Canny edges."""
        cv2 = pytest.importorskip("cv2")

        pipe = (
            Pipeline()
            .source("image_bytes")
            .canny(low_threshold=50.0, high_threshold=150.0)
        )
        result = _run_pipe(pipe, gray_png)
        if result.ndim == 3:
            result = result.squeeze(-1)

        expected = cv2.Canny(gray_image, 50, 150)

        # Both should detect edges in similar areas; due to implementation
        # differences (Gaussian sigma, exact NMS), we check structural overlap.
        result_edges = result > 0
        expected_edges = expected > 0

        if expected_edges.sum() > 0:
            # Dilate expected edges to allow 1-pixel tolerance
            kernel = np.ones((3, 3), np.uint8)
            expected_dilated = (
                cv2.dilate(expected.astype(np.uint8), kernel, iterations=1) > 0
            )

            # At least 30% of our detected edges should be near OpenCV edges
            overlap = (result_edges & expected_dilated).sum()
            coverage = overlap / max(result_edges.sum(), 1)
            assert coverage > 0.3, (
                f"Only {coverage:.1%} of polars-cv Canny edges are near OpenCV edges"
            )


# ===========================================================================
# Sharpen
# ===========================================================================


@plugin_required
class TestSharpenRef:
    """Sharpen operation reference tests."""

    def test_sharpen_increases_edge_contrast(
        self, gray_png: bytes, gray_image: np.ndarray
    ) -> None:
        """Sharpening should increase the magnitude of gradients at edges."""
        pipe_original = Pipeline().source("image_bytes").grayscale()
        pipe_sharpened = (
            Pipeline().source("image_bytes").grayscale().sharpen(strength=1.0)
        )

        original = _run_pipe(pipe_original, gray_png)
        sharpened = _run_pipe(pipe_sharpened, gray_png)

        if original.ndim == 3:
            original = original.squeeze(-1)
        if sharpened.ndim == 3:
            sharpened = sharpened.squeeze(-1)

        # Compute gradients
        orig_grad = np.abs(np.diff(original.astype(np.float32), axis=1)).mean()
        sharp_grad = np.abs(np.diff(sharpened.astype(np.float32), axis=1)).mean()

        assert sharp_grad > orig_grad, (
            f"Sharpened gradient ({sharp_grad:.2f}) should exceed "
            f"original ({orig_grad:.2f})"
        )

    @pytest.mark.parametrize("strength", [0.5, 1.0, 2.0])
    def test_sharpen_vs_opencv_filter2d(
        self, gray_png: bytes, gray_image: np.ndarray, strength: float
    ) -> None:
        """Sharpen should match cv2.filter2D with the same kernel, full image.

        Our convolve2d defaults to replicate borders, so the cv2 expectation
        must use BORDER_REPLICATE (cv2's own default is BORDER_REFLECT_101).
        The sharpen kernel is symmetric, so cv2's correlation equals
        convolution and no kernel flip is needed.
        """
        cv2 = pytest.importorskip("cv2")

        pipe = Pipeline().source("image_bytes").grayscale().sharpen(strength=strength)
        result = _run_pipe(pipe, gray_png)
        if result.ndim == 3 and result.shape[2] == 1:
            result = result.squeeze(-1)

        s = strength
        kernel = np.array(
            [[-s, -s, -s], [-s, 1.0 + 8.0 * s, -s], [-s, -s, -s]],
            dtype=np.float32,
        )
        expected = cv2.filter2D(
            gray_image.astype(np.float32),
            -1,
            kernel,
            borderType=cv2.BORDER_REPLICATE,
        )

        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_sharpen_rgb_vs_opencv_filter2d(
        self, rgb_png: bytes, rgb_image: np.ndarray
    ) -> None:
        """RGB sharpen should match per-channel cv2.filter2D, full image."""
        cv2 = pytest.importorskip("cv2")

        pipe = Pipeline().source("image_bytes").sharpen(strength=1.0)
        result = _run_pipe(pipe, rgb_png)

        kernel = np.array(
            [[-1, -1, -1], [-1, 9.0, -1], [-1, -1, -1]],
            dtype=np.float32,
        )
        expected = cv2.filter2D(
            rgb_image.astype(np.float32),
            -1,
            kernel,
            borderType=cv2.BORDER_REPLICATE,
        )

        np.testing.assert_allclose(result, expected, atol=1e-3)


# ===========================================================================
# Histogram Equalization
# ===========================================================================


@plugin_required
class TestHistogramEqualizeRef:
    """Histogram equalization reference tests."""

    def test_equalize_expands_dynamic_range(
        self, low_contrast_png: bytes, low_contrast_image: np.ndarray
    ) -> None:
        """Histogram equalization should expand the dynamic range."""
        pipe = Pipeline().source("image_bytes").grayscale().equalize_histogram()
        result = _run_pipe(pipe, low_contrast_png)

        assert result.dtype == np.uint8
        if result.ndim == 3:
            result = result.squeeze(-1)

        input_range = int(low_contrast_image.max()) - int(low_contrast_image.min())
        output_range = int(result.max()) - int(result.min())

        assert output_range > input_range, (
            f"Output range ({output_range}) should exceed input range ({input_range})"
        )

    def test_equalize_produces_near_uniform_histogram(
        self, low_contrast_png: bytes
    ) -> None:
        """Equalized histogram should be approximately uniform."""
        pipe = Pipeline().source("image_bytes").grayscale().equalize_histogram()
        result = _run_pipe(pipe, low_contrast_png)
        if result.ndim == 3:
            result = result.squeeze(-1)

        hist, _ = np.histogram(result.flatten(), bins=16, range=(0, 256))
        hist_norm = hist / hist.sum()
        ideal = 1.0 / 16.0

        # Chi-squared-like measure: sum of squared deviations from uniform
        chi_sq = np.sum((hist_norm - ideal) ** 2) / ideal
        assert chi_sq < 2.0, (
            f"Equalized histogram is not approximately uniform (χ²={chi_sq:.2f})"
        )

    def test_equalize_vs_opencv(self, gray_png: bytes, gray_image: np.ndarray) -> None:
        """Equalization should match OpenCV cv2.equalizeHist."""
        cv2 = pytest.importorskip("cv2")

        pipe = Pipeline().source("image_bytes").grayscale().equalize_histogram()
        result = _run_pipe(pipe, gray_png)

        assert result.dtype == np.uint8
        if result.ndim == 3:
            result = result.squeeze(-1)

        expected = cv2.equalizeHist(gray_image)
        np.testing.assert_array_equal(
            result, expected, err_msg="Equalization doesn't match OpenCV"
        )


# ===========================================================================
# Border Modes
# ===========================================================================


@plugin_required
class TestConvolveBorderModes:
    """Test different border handling modes for convolution."""

    def test_zero_border_mode(self, gray_png: bytes) -> None:
        """Zero border mode should treat out-of-bounds as 0."""
        kernel = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .convolve2d(kernel, 3, border="zero")
        )
        result = _run_pipe(pipe, gray_png)
        assert result is not None

    def test_reflect_border_mode(self, gray_png: bytes) -> None:
        """Reflect border mode should mirror values at edges."""
        kernel = [1.0] * 9
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .convolve2d(kernel, 3, normalize=True, border="reflect")
        )
        result = _run_pipe(pipe, gray_png)
        assert result is not None


# ===========================================================================
# Validation
# ===========================================================================


class TestConvolveValidation:
    """Test Python-side validation for convolve2d parameters."""

    def test_even_ksize_rejected(self) -> None:
        """Even ksize should raise ValueError."""
        with pytest.raises(ValueError, match="odd"):
            Pipeline().source("image_bytes").convolve2d([1.0] * 4, 2)

    def test_kernel_size_mismatch(self) -> None:
        """Kernel length not matching ksize² should raise ValueError."""
        with pytest.raises(ValueError, match="doesn't match"):
            Pipeline().source("image_bytes").convolve2d([1.0, 2.0], 3)

    def test_invalid_border_mode(self) -> None:
        """Invalid border mode should raise ValueError.

        Uses the uniform ``_validate_enum`` message now that ``convolve2d``
        validates ``border`` against the view-buffer ``BorderMode`` authority
        (matching every other enum-valued parameter)."""
        with pytest.raises(ValueError, match="Invalid border mode"):
            Pipeline().source("image_bytes").convolve2d([1.0] * 9, 3, border="invalid")

    def test_sobel_invalid_ksize(self) -> None:
        """Sobel with ksize != 3 should raise ValueError."""
        with pytest.raises(ValueError, match="ksize=3"):
            Pipeline().source("image_bytes").sobel(ksize=5)

    def test_laplacian_invalid_ksize(self) -> None:
        """Laplacian with ksize != 3 should raise ValueError."""
        with pytest.raises(ValueError, match="ksize=3"):
            Pipeline().source("image_bytes").laplacian(ksize=5)


# ===========================================================================
# Multi-channel convolution
# ===========================================================================


@plugin_required
class TestConvolveMultiChannel:
    """Test convolution on multi-channel images."""

    def test_rgb_convolve(self, rgb_png: bytes, rgb_image: np.ndarray) -> None:
        """Convolution should work on RGB images (per-channel)."""
        identity = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        pipe = Pipeline().source("image_bytes").convolve2d(identity, 3)
        result = _run_pipe(pipe, rgb_png)

        assert result.ndim == 3
        assert result.shape[2] == 3
        expected = rgb_image.astype(np.float32)
        np.testing.assert_allclose(result, expected, atol=0.5)
