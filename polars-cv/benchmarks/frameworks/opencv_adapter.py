"""
OpenCV framework adapter for benchmarking.

This module provides an adapter for OpenCV (cv2) + NumPy image processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import (
    BaseFrameworkAdapter,
    OperationParams,
    brightness_f32,
    contrast_f32,
    rotation_matrix,
)

if TYPE_CHECKING:
    import numpy.typing as npt


class OpenCVAdapter(BaseFrameworkAdapter):
    """
    Adapter for OpenCV image processing.

    Uses cv2 for image operations with NumPy array representation.

    Attributes:
        name: Human-readable name of the adapter.
    """

    name: str = "opencv"
    supports_gpu: bool = False

    def __init__(self) -> None:
        """Initialize the OpenCV adapter."""
        self._cv2: Any = None

    def is_available(self) -> bool:
        """
        Check if OpenCV is available.

        Returns:
            True if cv2 can be imported, False otherwise.
        """
        try:
            import cv2  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_cv2(self) -> Any:
        """Get the cv2 module."""
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        return self._cv2

    def load_from_file(self, path: Path) -> "npt.NDArray[np.uint8]":
        """
        Load an image from a file path.

        Args:
            path: Path to the image file.

        Returns:
            Image as NumPy array (BGR format).
        """
        cv2 = self._get_cv2()
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            msg = f"Failed to load image: {path}"
            raise ValueError(msg)
        # Convert BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_from_bytes(self, data: bytes) -> "npt.NDArray[np.uint8]":
        """
        Load an image from bytes.

        Args:
            data: Image bytes (PNG, JPEG, etc.).

        Returns:
            Image as NumPy array (RGB format).
        """
        cv2 = self._get_cv2()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            msg = "Failed to decode image from bytes"
            raise ValueError(msg)
        # Convert BGR to RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def resize(
        self, img: "npt.NDArray[np.uint8]", height: int, width: int
    ) -> "npt.NDArray[np.uint8]":
        """
        Resize an image.

        Args:
            img: Image as NumPy array.
            height: Target height.
            width: Target width.

        Returns:
            Resized image.
        """
        cv2 = self._get_cv2()
        # Bilinear, as every adapter uses. OpenCV has no antialiased bilinear:
        # on a downscale polars-cv and Pillow antialias and this does not
        # (tests/test_benchmark_adapters.py bounds the difference).
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

    def grayscale(self, img: "npt.NDArray[np.uint8]") -> "npt.NDArray[np.uint8]":
        """
        Convert image to grayscale.

        Args:
            img: Image as NumPy array (RGB).

        Returns:
            Grayscale image.
        """
        cv2 = self._get_cv2()
        if img.ndim == 2:
            return img
        # Convert RGB to grayscale
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def normalize(self, img: "npt.NDArray[np.uint8]") -> "npt.NDArray[np.float32]":
        """
        Apply min-max normalization.

        Args:
            img: Image as NumPy array.

        Returns:
            Normalized image with values in [0, 1].
        """
        img_float = img.astype(np.float32)
        min_val = img_float.min()
        max_val = img_float.max()
        if max_val - min_val > 0:
            return (img_float - min_val) / (max_val - min_val)
        return img_float

    def flip_horizontal(self, img: "npt.NDArray[np.uint8]") -> "npt.NDArray[np.uint8]":
        """
        Flip image horizontally.

        Args:
            img: Image as NumPy array.

        Returns:
            Horizontally flipped image.
        """
        cv2 = self._get_cv2()
        return cv2.flip(img, 1)  # 1 = horizontal flip

    def flip_vertical(self, img: "npt.NDArray[np.uint8]") -> "npt.NDArray[np.uint8]":
        """
        Flip image vertically.

        Args:
            img: Image as NumPy array.

        Returns:
            Vertically flipped image.
        """
        cv2 = self._get_cv2()
        return cv2.flip(img, 0)  # 0 = vertical flip

    def crop(
        self,
        img: "npt.NDArray[np.uint8]",
        top: int,
        left: int,
        height: int,
        width: int,
    ) -> "npt.NDArray[np.uint8]":
        """
        Crop image.

        Args:
            img: Image as NumPy array.
            top: Top offset.
            left: Left offset.
            height: Crop height.
            width: Crop width.

        Returns:
            Cropped image.
        """
        return img[top : top + height, left : left + width].copy()

    def blur(
        self, img: "npt.NDArray[np.uint8]", sigma: float
    ) -> "npt.NDArray[np.uint8]":
        """
        Apply Gaussian blur.

        Args:
            img: Image as NumPy array.
            sigma: Blur sigma.

        Returns:
            Blurred image.
        """
        cv2 = self._get_cv2()
        # polars-cv's kernel: radius ceil(3 sigma), replicated border.
        ksize = 2 * int(np.ceil(3 * sigma)) + 1
        return cv2.GaussianBlur(
            img, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE
        )

    def threshold(
        self, img: "npt.NDArray[np.uint8]", value: int
    ) -> "npt.NDArray[np.uint8]":
        """
        Apply binary threshold.

        Args:
            img: Image as NumPy array.
            value: Threshold value.

        Returns:
            Thresholded image.
        """
        cv2 = self._get_cv2()
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, result = cv2.threshold(img, value, 255, cv2.THRESH_BINARY)
        return result

    def rotate(
        self, img: "npt.NDArray[np.uint8]", angle: float, *, expand: bool = False
    ) -> "npt.NDArray[np.uint8]":
        """Rotate clockwise by ``angle`` degrees, as polars-cv's ``rotate``.

        A multiple of 90° is a pixel permutation (``cv2.rotate``, which swaps
        the sides of a non-square image whatever ``expand`` says); any other
        angle resamples about the image centre with ``warpAffine``, the
        expanded canvas rounded to the nearest pixel.
        """
        cv2 = self._get_cv2()
        lattice = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        quarter = angle % 360
        if quarter == 0:
            return img.copy()
        if quarter in lattice:
            return cv2.rotate(img, lattice[quarter])
        mat, (out_h, out_w) = rotation_matrix(*img.shape[:2], angle, expand=expand)
        return cv2.warpAffine(img, mat, (out_w, out_h))

    def erode(self, img: Any, ksize: int, iterations: int = 1) -> Any:
        """Apply morphological erosion."""
        cv2 = self._get_cv2()
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((ksize, ksize), dtype=np.uint8)
        return cv2.erode(img, kernel, iterations=iterations)

    def dilate(self, img: Any, ksize: int, iterations: int = 1) -> Any:
        """Apply morphological dilation."""
        cv2 = self._get_cv2()
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((ksize, ksize), dtype=np.uint8)
        return cv2.dilate(img, kernel, iterations=iterations)

    def invert(self, img: Any) -> Any:
        """Invert pixel values."""
        cv2 = self._get_cv2()
        return cv2.bitwise_not(img)

    def adjust_contrast(self, img: Any, factor: float) -> Any:
        """``(pixel - mean) * factor + mean`` over all channels, as f32, unclipped."""
        return contrast_f32(img, factor)

    def adjust_brightness(self, img: Any, factor: float) -> Any:
        """``pixel * factor`` clamped to [0, 255], as f32."""
        return brightness_f32(img, factor)

    def sharpen(self, img: Any, strength: float = 1.0) -> Any:
        """3x3 sharpening kernel (sum 1), replicated border, as f32.

        polars-cv's ``sharpen``: ``-strength`` around ``1 + 8·strength``.
        """
        cv2 = self._get_cv2()
        kernel = np.full((3, 3), -strength, dtype=np.float32)
        kernel[1, 1] = 1 + 8 * strength
        return cv2.filter2D(
            img.astype(np.float32),
            cv2.CV_32F,
            kernel,
            borderType=cv2.BORDER_REPLICATE,
        )

    def pad(
        self, img: Any, top: int, bottom: int, left: int, right: int, value: int = 0
    ) -> Any:
        """Add constant padding to image edges."""
        cv2 = self._get_cv2()
        return cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value
        )

    def histogram_equalize(self, img: Any) -> Any:
        """Apply histogram equalization."""
        cv2 = self._get_cv2()
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.equalizeHist(img)

    def canny(self, img: Any, low_threshold: float, high_threshold: float) -> Any:
        """Apply Canny edge detection."""
        cv2 = self._get_cv2()
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Canny(img, low_threshold, high_threshold)

    def sobel(self, img: Any, axis: str = "x") -> Any:
        """Apply Sobel gradient operator."""
        cv2 = self._get_cv2()
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dx, dy = (1, 0) if axis == "x" else (0, 1)
        return cv2.Sobel(
            img, cv2.CV_32F, dx, dy, ksize=3, borderType=cv2.BORDER_REPLICATE
        )

    def to_numpy(
        self, img: "npt.NDArray[np.uint8] | npt.NDArray[np.float32]"
    ) -> "npt.NDArray[np.uint8] | npt.NDArray[np.float32]":
        """
        Convert image to NumPy array (already is one).

        Args:
            img: Image as NumPy array.

        Returns:
            Same NumPy array.
        """
        return img

    def run_pipeline_batch(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list["npt.NDArray[np.uint8] | npt.NDArray[np.float32]"]:
        """
        Run a pipeline on a batch of images.

        Args:
            image_bytes_list: List of image bytes.
            operations: Operations to apply.

        Returns:
            List of processed images as NumPy arrays.
        """
        results = []
        for data in image_bytes_list:
            img = self.load_from_bytes(data)
            for op in operations:
                img = self.apply_operation(img, op)
            results.append(img)
        return results
