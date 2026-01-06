"""
OpenCV framework adapter for benchmarking.

This module provides an adapter for OpenCV (cv2) + NumPy image processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseFrameworkAdapter, OperationParams

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
        # Use bilinear interpolation for consistency across frameworks
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

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
        # Kernel size should be odd and related to sigma
        ksize = int(sigma * 6) | 1  # Ensure odd
        if ksize < 3:
            ksize = 3
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

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
