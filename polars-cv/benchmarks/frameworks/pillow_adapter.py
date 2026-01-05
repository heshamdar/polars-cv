"""
PIL/Pillow framework adapter for benchmarking.

This module provides an adapter for PIL/Pillow image processing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseFrameworkAdapter, OperationParams, OperationType

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL import Image as PILImageModule


class PillowAdapter(BaseFrameworkAdapter):
    """
    Adapter for PIL/Pillow image processing.

    Uses PIL.Image for image operations.

    Attributes:
        name: Human-readable name of the adapter.
    """

    name: str = "pillow"
    supports_gpu: bool = False

    def __init__(self) -> None:
        """Initialize the Pillow adapter."""
        self._Image: Any = None
        self._ImageFilter: Any = None

    def is_available(self) -> bool:
        """
        Check if Pillow is available.

        Returns:
            True if PIL can be imported, False otherwise.
        """
        try:
            from PIL import Image, ImageFilter  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_modules(self) -> tuple[Any, Any]:
        """Get PIL modules."""
        if self._Image is None:
            from PIL import Image, ImageFilter

            self._Image = Image
            self._ImageFilter = ImageFilter
        return self._Image, self._ImageFilter

    def load_from_file(self, path: Path) -> "PILImageModule.Image":
        """
        Load an image from a file path.

        Args:
            path: Path to the image file.

        Returns:
            PIL Image object.
        """
        Image, _ = self._get_modules()
        img = Image.open(path)
        # Convert to RGB to ensure consistent format
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def load_from_bytes(self, data: bytes) -> "PILImageModule.Image":
        """
        Load an image from bytes.

        Args:
            data: Image bytes (PNG, JPEG, etc.).

        Returns:
            PIL Image object.
        """
        Image, _ = self._get_modules()
        img = Image.open(io.BytesIO(data))
        # Convert to RGB to ensure consistent format
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def resize(
        self, img: "PILImageModule.Image", height: int, width: int
    ) -> "PILImageModule.Image":
        """
        Resize an image.

        Args:
            img: PIL Image object.
            height: Target height.
            width: Target width.

        Returns:
            Resized image.
        """
        Image, _ = self._get_modules()
        # Use bilinear interpolation for consistency across frameworks
        return img.resize((width, height), Image.Resampling.BILINEAR)

    def grayscale(self, img: "PILImageModule.Image") -> "PILImageModule.Image":
        """
        Convert image to grayscale.

        Args:
            img: PIL Image object.

        Returns:
            Grayscale image.
        """
        return img.convert("L")

    def normalize(self, img: "PILImageModule.Image") -> "npt.NDArray[np.float32]":
        """
        Apply min-max normalization.

        Note: Returns NumPy array since PIL doesn't support float images.

        Args:
            img: PIL Image object.

        Returns:
            Normalized image as NumPy array with values in [0, 1].
        """
        arr = np.array(img, dtype=np.float32)
        min_val = arr.min()
        max_val = arr.max()
        if max_val - min_val > 0:
            return (arr - min_val) / (max_val - min_val)
        return arr

    def flip_horizontal(self, img: "PILImageModule.Image") -> "PILImageModule.Image":
        """
        Flip image horizontally.

        Args:
            img: PIL Image object.

        Returns:
            Horizontally flipped image.
        """
        Image, _ = self._get_modules()
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    def flip_vertical(self, img: "PILImageModule.Image") -> "PILImageModule.Image":
        """
        Flip image vertically.

        Args:
            img: PIL Image object.

        Returns:
            Vertically flipped image.
        """
        Image, _ = self._get_modules()
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    def crop(
        self,
        img: "PILImageModule.Image",
        top: int,
        left: int,
        height: int,
        width: int,
    ) -> "PILImageModule.Image":
        """
        Crop image.

        Args:
            img: PIL Image object.
            top: Top offset.
            left: Left offset.
            height: Crop height.
            width: Crop width.

        Returns:
            Cropped image.
        """
        # PIL crop uses (left, upper, right, lower)
        return img.crop((left, top, left + width, top + height))

    def blur(self, img: "PILImageModule.Image", sigma: float) -> "PILImageModule.Image":
        """
        Apply Gaussian blur.

        Args:
            img: PIL Image object.
            sigma: Blur sigma (radius).

        Returns:
            Blurred image.
        """
        _, ImageFilter = self._get_modules()
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def threshold(
        self, img: "PILImageModule.Image", value: int
    ) -> "PILImageModule.Image":
        """
        Apply binary threshold.

        Args:
            img: PIL Image object.
            value: Threshold value.

        Returns:
            Thresholded image.
        """
        # Convert to grayscale if needed
        if img.mode != "L":
            img = img.convert("L")

        # Apply threshold using point function
        return img.point(lambda p: 255 if p > value else 0)

    def to_numpy(
        self, img: "PILImageModule.Image | npt.NDArray[np.float32]"
    ) -> "npt.NDArray[np.uint8] | npt.NDArray[np.float32]":
        """
        Convert image to NumPy array.

        Args:
            img: PIL Image object or NumPy array.

        Returns:
            NumPy array.
        """
        if isinstance(img, np.ndarray):
            return img
        return np.array(img)

    def apply_operation(
        self,
        img: "PILImageModule.Image | npt.NDArray[np.float32]",
        params: OperationParams,
    ) -> "PILImageModule.Image | npt.NDArray[np.float32]":
        """
        Apply a single operation.

        Overridden to handle the PIL/NumPy type conversion for normalize.

        Args:
            img: PIL Image or NumPy array.
            params: Operation parameters.

        Returns:
            Processed image.
        """
        Image, _ = self._get_modules()

        # Convert NumPy back to PIL if needed (except for normalize output)
        if isinstance(img, np.ndarray) and params.operation != OperationType.NORMALIZE:
            if img.dtype == np.float32:
                # Scale back to uint8
                img = (img * 255).clip(0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = Image.fromarray(img, mode="L")
            else:
                img = Image.fromarray(img, mode="RGB")

        return super().apply_operation(img, params)

    def run_pipeline_batch(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list["PILImageModule.Image | npt.NDArray[np.float32]"]:
        """
        Run a pipeline on a batch of images.

        Args:
            image_bytes_list: List of image bytes.
            operations: Operations to apply.

        Returns:
            List of processed images.
        """
        results = []
        for data in image_bytes_list:
            img: Any = self.load_from_bytes(data)
            for op in operations:
                img = self.apply_operation(img, op)
            results.append(img)
        return results
