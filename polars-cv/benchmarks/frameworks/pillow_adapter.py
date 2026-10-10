"""
PIL/Pillow framework adapter for benchmarking.

This module provides an adapter for PIL/Pillow image processing.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import (
    BaseFrameworkAdapter,
    OperationParams,
    OperationType,
    brightness_f32,
    contrast_f32,
    letterbox_geometry,
    rotation_matrix,
)

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

    def rotate(self, img: Any, angle: float, *, expand: bool = False) -> Any:
        """Rotate clockwise by ``angle`` degrees, as polars-cv's ``rotate``.

        A multiple of 90° is a pixel permutation (``transpose``, which swaps
        the sides of a non-square image whatever ``expand`` says); any other
        angle resamples bilinearly with polars-cv's centre and canvas —
        ``Image.rotate`` sizes its expanded canvas differently.
        """
        from PIL import Image

        lattice = {
            90: Image.Transpose.ROTATE_270,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_90,
        }
        quarter = angle % 360
        if quarter == 0:
            return img.copy()
        if quarter in lattice:
            return img.transpose(lattice[quarter])
        mat, (out_h, out_w) = rotation_matrix(
            img.height, img.width, angle, expand=expand
        )
        return self._affine(img, mat, out_h, out_w)

    def _affine(self, img: Any, mat: Any, out_h: int, out_w: int) -> Any:
        """Bilinear warp by the forward 2x3 ``mat`` (pixel-centre coordinates,
        OpenCV's convention) onto an ``out_h`` x ``out_w`` canvas."""
        Image, _ = self._get_modules()
        # Pillow wants the output-to-input map in pixel-corner coordinates;
        # the matrix is in pixel-centre ones (a centre is corner + 0.5).
        inv = np.linalg.inv(np.vstack([mat, [0.0, 0.0, 1.0]]))
        to_corner = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
        data = (to_corner @ inv @ np.linalg.inv(to_corner))[:2].ravel()
        return img.transform(
            (out_w, out_h),
            Image.Transform.AFFINE,
            tuple(data),
            Image.Resampling.BILINEAR,
        )

    def warp_affine(
        self, img: Any, matrix: tuple[float, ...], height: int, width: int
    ) -> Any:
        """``Image.transform`` with the inverted matrix (:meth:`_affine`)."""
        return self._affine(img, np.asarray(matrix).reshape(2, 3), height, width)

    def letterbox(self, img: Any, height: int, width: int) -> Any:
        """Bilinear fit pasted centred on a black canvas."""
        Image, _ = self._get_modules()
        new_h, new_w, top, left = letterbox_geometry(
            img.height, img.width, height, width
        )
        canvas = Image.new(img.mode, (width, height), 0)
        canvas.paste(img.resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
        return canvas

    def morphology(self, img: Any, op: str, ksize: int) -> Any:
        """Min/Max filters (Pillow's erosion and dilation) on the "L" image."""
        from PIL import ImageChops

        _, ImageFilter = self._get_modules()
        if img.mode != "L":
            img = img.convert("L")
        erode, dilate = ImageFilter.MinFilter(ksize), ImageFilter.MaxFilter(ksize)
        if op == "open":
            return img.filter(erode).filter(dilate)
        if op == "close":
            return img.filter(dilate).filter(erode)
        return ImageChops.subtract(img.filter(dilate), img.filter(erode))

    def invert(self, img: Any) -> Any:
        """Invert pixel values."""
        from PIL import ImageOps

        return ImageOps.invert(img)

    def adjust_contrast(self, img: Any, factor: float) -> Any:
        """polars-cv's contrast (f32, all-channel mean); ``ImageEnhance`` uses
        the luminance mean and truncates to u8."""
        import numpy as np

        return contrast_f32(np.asarray(img), factor)

    def adjust_brightness(self, img: Any, factor: float) -> Any:
        """polars-cv's brightness (f32, clamped); ``ImageEnhance`` truncates."""
        import numpy as np

        return brightness_f32(np.asarray(img), factor)

    def sharpen(self, img: Any, strength: float = 1.0) -> Any:
        """Not computable here: polars-cv's sharpen is an unclipped f32 3x3
        kernel, and Pillow filters only 8-bit images (``ImageFilter.SHARPEN``
        is a different, clipped kernel)."""
        raise NotImplementedError

    def pad(
        self, img: Any, top: int, bottom: int, left: int, right: int, value: int = 0
    ) -> Any:
        """Add constant padding to image edges."""
        from PIL import ImageOps

        return ImageOps.expand(img, border=(left, top, right, bottom), fill=value)

    def histogram_equalize(self, img: Any) -> Any:
        """Apply histogram equalization."""
        from PIL import ImageOps

        if img.mode != "L":
            img = img.convert("L")
        return ImageOps.equalize(img)

    def erode(self, img: Any, ksize: int, iterations: int = 1) -> Any:
        """Apply morphological erosion via MinFilter."""
        _, ImageFilter = self._get_modules()
        if img.mode != "L":
            img = img.convert("L")
        for _ in range(iterations):
            img = img.filter(ImageFilter.MinFilter(size=ksize))
        return img

    def dilate(self, img: Any, ksize: int, iterations: int = 1) -> Any:
        """Apply morphological dilation via MaxFilter."""
        _, ImageFilter = self._get_modules()
        if img.mode != "L":
            img = img.convert("L")
        for _ in range(iterations):
            img = img.filter(ImageFilter.MaxFilter(size=ksize))
        return img

    def prepare_decoded_images(self, png_bytes_list: list[bytes]) -> list[Any]:
        """Decode now: ``Image.open`` is lazy, so the first timed op on each
        image would otherwise pay its PNG decode."""
        images = super().prepare_decoded_images(png_bytes_list)
        for img in images:
            img.load()
        return images

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
