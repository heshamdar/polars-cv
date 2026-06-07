"""
Reference tests for morphological operations (erode, dilate, open, close, gradient).

Compares polars-cv output against OpenCV ground truth to ensure correctness.
All operations require single-channel input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.pipeline import Pipeline as PipelineClass
from tests.conftest import plugin_required

if TYPE_CHECKING:
    pass


@pytest.fixture(scope="session")
def binary_mask_image() -> np.ndarray:
    """
    Binary mask with a center square and some noise spots.

    Returns:
        np.ndarray: Binary mask with shape (100, 100) and dtype uint8.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Large center square
    mask[30:70, 30:70] = 255
    # Small noise spots (should be removed by opening)
    mask[10, 10] = 255
    mask[15, 85] = 255
    mask[90, 50] = 255
    # Small holes in the center square (should be filled by closing)
    mask[45, 45] = 0
    mask[50, 50] = 0
    mask[55, 55] = 0
    return mask


@pytest.fixture
def encode_gray_png() -> Callable[[np.ndarray], bytes]:
    """
    Encode a grayscale numpy array as PNG bytes.

    Returns:
        A callable that encodes a grayscale numpy array as PNG bytes.
    """

    def _encode(arr: np.ndarray) -> bytes:
        """
        Encode grayscale numpy array as PNG bytes.

        Args:
            arr: NumPy array with shape (H, W) and dtype uint8.

        Returns:
            PNG bytes.
        """
        try:
            import io

            from PIL import Image

            img = Image.fromarray(arr, mode="L")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except ImportError:
            pytest.skip("PIL/Pillow required for this test")
            return b""

    return _encode


def _run_pipeline(
    image_arr: np.ndarray,
    pipe: PipelineClass,
    encode_fn: Callable[[np.ndarray], bytes],
) -> np.ndarray:
    """
    Run a polars-cv pipeline on a grayscale image and return the result as ndarray.

    Args:
        image_arr: Input grayscale image (H, W) uint8.
        pipe: Pipeline to execute.
        encode_fn: Function to encode ndarray to PNG bytes.

    Returns:
        Output image as ndarray.
    """
    from polars_cv import numpy_from_struct

    png_bytes = encode_fn(image_arr)
    df = pl.DataFrame({"image": [png_bytes]})
    result = df.with_columns(output=pl.col("image").cv.pipe(pipe).sink("numpy"))
    out = numpy_from_struct(result["output"][0])
    # Squeeze trailing channel dim if present
    return out.squeeze()


@plugin_required
class TestErode:
    """Reference tests for morphological erosion."""

    def test_erode_vs_opencv(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Erode output matches OpenCV cv2.erode with 3x3 rect kernel."""
        cv2 = pytest.importorskip("cv2")

        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.erode(binary_mask_image, kernel, iterations=1)

        pipe = Pipeline().source("image_bytes").grayscale().erode(ksize=3)
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)

    def test_erode_multiple_iterations(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Erode with iterations=2 matches two sequential erodes."""
        cv2 = pytest.importorskip("cv2")

        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.erode(binary_mask_image, kernel, iterations=2)

        pipe = Pipeline().source("image_bytes").grayscale().erode(ksize=3, iterations=2)
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)

    def test_erode_ksize_1_identity(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Erode with ksize=1 should return input unchanged."""
        pipe = Pipeline().source("image_bytes").grayscale().erode(ksize=1)
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, binary_mask_image)

    def test_erode_all_zero(
        self,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Erode on all-zero mask returns all zeros."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        pipe = Pipeline().source("image_bytes").grayscale().erode(ksize=3)
        actual = _run_pipeline(mask, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, mask)

    def test_erode_all_ones(
        self,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Erode on all-255 mask: only border pixels eroded."""
        cv2 = pytest.importorskip("cv2")

        mask = np.full((50, 50), 255, dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.erode(mask, kernel, iterations=1)

        pipe = Pipeline().source("image_bytes").grayscale().erode(ksize=3)
        actual = _run_pipeline(mask, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)


@plugin_required
class TestDilate:
    """Reference tests for morphological dilation."""

    def test_dilate_vs_opencv(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Dilate output matches OpenCV cv2.dilate with 3x3 rect kernel."""
        cv2 = pytest.importorskip("cv2")

        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.dilate(binary_mask_image, kernel, iterations=1)

        pipe = Pipeline().source("image_bytes").grayscale().dilate(ksize=3)
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)

    def test_dilate_multiple_iterations(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Dilate with iterations=2 matches two sequential dilates."""
        cv2 = pytest.importorskip("cv2")

        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.dilate(binary_mask_image, kernel, iterations=2)

        pipe = (
            Pipeline().source("image_bytes").grayscale().dilate(ksize=3, iterations=2)
        )
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)

    def test_dilate_ksize_1_identity(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Dilate with ksize=1 should return input unchanged."""
        pipe = Pipeline().source("image_bytes").grayscale().dilate(ksize=1)
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, binary_mask_image)

    def test_dilate_all_zero(
        self,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Dilate on all-zero mask returns all zeros."""
        cv2 = pytest.importorskip("cv2")

        mask = np.zeros((50, 50), dtype=np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.dilate(mask, kernel, iterations=1)

        pipe = Pipeline().source("image_bytes").grayscale().dilate(ksize=3)
        actual = _run_pipeline(mask, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)


@plugin_required
class TestMorphologyOpen:
    """Reference tests for morphological opening (erode then dilate)."""

    def test_open_removes_small_noise(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Opening removes small bright noise spots."""
        cv2 = pytest.importorskip("cv2")

        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.morphologyEx(binary_mask_image, cv2.MORPH_OPEN, kernel)

        pipe = Pipeline().source("image_bytes").grayscale().morphology_open(ksize=3)
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)


@plugin_required
class TestMorphologyClose:
    """Reference tests for morphological closing (dilate then erode)."""

    def test_close_fills_small_holes(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Closing fills small dark holes."""
        cv2 = pytest.importorskip("cv2")

        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.morphologyEx(binary_mask_image, cv2.MORPH_CLOSE, kernel)

        pipe = Pipeline().source("image_bytes").grayscale().morphology_close(ksize=3)
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)


@plugin_required
class TestMorphologyGradient:
    """Reference tests for morphological gradient (dilate - erode)."""

    def test_gradient_vs_opencv(
        self,
        binary_mask_image: np.ndarray,
        encode_gray_png: Callable[[np.ndarray], bytes],
    ) -> None:
        """Gradient output matches OpenCV MORPH_GRADIENT."""
        cv2 = pytest.importorskip("cv2")

        kernel = np.ones((3, 3), np.uint8)
        expected = cv2.morphologyEx(binary_mask_image, cv2.MORPH_GRADIENT, kernel)

        pipe = Pipeline().source("image_bytes").grayscale().morphology_gradient(ksize=3)
        actual = _run_pipeline(binary_mask_image, pipe, encode_gray_png)

        np.testing.assert_array_equal(actual, expected)


class TestPipelineBuilder:
    """Tests for morphological pipeline construction (no plugin required)."""

    def test_erode_creates_op(self) -> None:
        """Erode method adds an OpSpec with correct params."""
        pipe = Pipeline().erode(ksize=5, iterations=2)
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "erode"
        assert pipe._ops[0].params["ksize"].value == 5
        assert pipe._ops[0].params["iterations"].value == 2

    def test_dilate_creates_op(self) -> None:
        """Dilate method adds an OpSpec with correct params."""
        pipe = Pipeline().dilate(ksize=7)
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "dilate"
        assert pipe._ops[0].params["ksize"].value == 7
        assert pipe._ops[0].params["iterations"].value == 1

    def test_open_is_erode_then_dilate(self) -> None:
        """Morphology open is composed of erode then dilate."""
        pipe = Pipeline().morphology_open(ksize=5)
        assert len(pipe._ops) == 2
        assert pipe._ops[0].op == "erode"
        assert pipe._ops[1].op == "dilate"
        assert pipe._ops[0].params["ksize"].value == 5
        assert pipe._ops[1].params["ksize"].value == 5

    def test_close_is_dilate_then_erode(self) -> None:
        """Morphology close is composed of dilate then erode."""
        pipe = Pipeline().morphology_close(ksize=5)
        assert len(pipe._ops) == 2
        assert pipe._ops[0].op == "dilate"
        assert pipe._ops[1].op == "erode"
        assert pipe._ops[0].params["ksize"].value == 5
        assert pipe._ops[1].params["ksize"].value == 5

    def test_gradient_creates_op(self) -> None:
        """Morphology gradient adds a single OpSpec."""
        pipe = Pipeline().morphology_gradient(ksize=3)
        assert len(pipe._ops) == 1
        assert pipe._ops[0].op == "morphology_gradient"
        assert pipe._ops[0].params["ksize"].value == 3

    def test_domain_preserved(self) -> None:
        """Morphological ops preserve the buffer domain."""
        pipe = Pipeline().source("image_bytes").grayscale().erode(ksize=3)
        assert pipe._current_domain == "buffer"

    def test_dtype_preserved(self) -> None:
        """Morphological ops preserve the output dtype."""
        pipe = Pipeline().source("image_bytes").grayscale()
        dtype_before = pipe._output_dtype
        pipe_after = pipe.erode(ksize=3)
        assert pipe_after._output_dtype == dtype_before
