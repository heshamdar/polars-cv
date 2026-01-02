"""
Base framework adapter for benchmarking.

This module provides the abstract base class that all framework adapters
must implement to ensure consistent benchmarking across different
vision processing libraries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class OperationType(Enum):
    """Types of operations supported by benchmarks."""

    RESIZE = auto()
    GRAYSCALE = auto()
    NORMALIZE = auto()
    FLIP_H = auto()
    FLIP_V = auto()
    CROP = auto()
    BLUR = auto()
    THRESHOLD = auto()
    CAST = auto()
    SCALE = auto()


@dataclass
class OperationParams:
    """Parameters for an operation."""

    operation: OperationType
    height: int | None = None
    width: int | None = None
    sigma: float | None = None  # For blur
    threshold_value: int | None = None  # For threshold
    crop_top: int | None = None
    crop_left: int | None = None
    crop_height: int | None = None
    crop_width: int | None = None
    scale_factor: float | None = None
    dtype: str | None = None  # For cast


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    framework: str
    operation: str
    image_count: int
    image_size: tuple[int, int]
    total_time_seconds: float
    throughput_images_per_second: float
    latency_ms_per_image: float
    peak_memory_mb: float
    gpu_mode: str | None = None  # "cold", "warm", or None for CPU

    def __repr__(self) -> str:
        """Return string representation."""
        gpu_suffix = f" [{self.gpu_mode}]" if self.gpu_mode else ""
        return (
            f"BenchmarkResult({self.framework}{gpu_suffix}: "
            f"{self.throughput_images_per_second:.1f} img/s, "
            f"{self.latency_ms_per_image:.2f} ms/img, "
            f"{self.peak_memory_mb:.1f} MB)"
        )


class BaseFrameworkAdapter(ABC):
    """
    Abstract base class for framework adapters.

    All framework adapters must implement these methods to provide
    a consistent interface for benchmarking image processing operations.

    Attributes:
        name: Human-readable name of the framework.
        supports_gpu: Whether this framework supports GPU acceleration.
        gpu_device: The GPU device identifier (e.g., "mps", "cuda:0").
    """

    name: str
    supports_gpu: bool = False
    gpu_device: str | None = None

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this framework is available (dependencies installed).

        Returns:
            True if the framework can be used, False otherwise.
        """
        ...

    @abstractmethod
    def load_from_file(self, path: Path) -> Any:
        """
        Load an image from a file path.

        Args:
            path: Path to the image file.

        Returns:
            Image in the framework's native format.
        """
        ...

    @abstractmethod
    def load_from_bytes(self, data: bytes) -> Any:
        """
        Load an image from bytes.

        Args:
            data: Image bytes (PNG, JPEG, etc.).

        Returns:
            Image in the framework's native format.
        """
        ...

    @abstractmethod
    def resize(self, img: Any, height: int, width: int) -> Any:
        """
        Resize an image to the specified dimensions.

        Args:
            img: Image in the framework's native format.
            height: Target height.
            width: Target width.

        Returns:
            Resized image.
        """
        ...

    @abstractmethod
    def grayscale(self, img: Any) -> Any:
        """
        Convert an image to grayscale.

        Args:
            img: Image in the framework's native format.

        Returns:
            Grayscale image.
        """
        ...

    @abstractmethod
    def normalize(self, img: Any) -> Any:
        """
        Apply min-max normalization to an image.

        Normalizes values to [0, 1] range.

        Args:
            img: Image in the framework's native format.

        Returns:
            Normalized image.
        """
        ...

    @abstractmethod
    def flip_horizontal(self, img: Any) -> Any:
        """
        Flip an image horizontally.

        Args:
            img: Image in the framework's native format.

        Returns:
            Horizontally flipped image.
        """
        ...

    @abstractmethod
    def flip_vertical(self, img: Any) -> Any:
        """
        Flip an image vertically.

        Args:
            img: Image in the framework's native format.

        Returns:
            Vertically flipped image.
        """
        ...

    @abstractmethod
    def crop(self, img: Any, top: int, left: int, height: int, width: int) -> Any:
        """
        Crop a region from an image.

        Args:
            img: Image in the framework's native format.
            top: Top offset (y-start).
            left: Left offset (x-start).
            height: Crop height.
            width: Crop width.

        Returns:
            Cropped image.
        """
        ...

    @abstractmethod
    def blur(self, img: Any, sigma: float) -> Any:
        """
        Apply Gaussian blur to an image.

        Args:
            img: Image in the framework's native format.
            sigma: Blur sigma (standard deviation).

        Returns:
            Blurred image.
        """
        ...

    @abstractmethod
    def threshold(self, img: Any, value: int) -> Any:
        """
        Apply binary threshold to an image.

        Args:
            img: Image in the framework's native format.
            value: Threshold value.

        Returns:
            Thresholded image.
        """
        ...

    @abstractmethod
    def to_numpy(self, img: Any) -> "npt.NDArray[np.uint8] | npt.NDArray[np.float32]":
        """
        Convert image to NumPy array for validation.

        Args:
            img: Image in the framework's native format.

        Returns:
            NumPy array representation of the image.
        """
        ...

    def apply_operation(self, img: Any, params: OperationParams) -> Any:
        """
        Apply a single operation based on parameters.

        Args:
            img: Image in the framework's native format.
            params: Operation parameters.

        Returns:
            Processed image.

        Raises:
            ValueError: If operation type is not supported.
        """
        op = params.operation

        if op == OperationType.RESIZE:
            if params.height is None or params.width is None:
                msg = "Resize requires height and width"
                raise ValueError(msg)
            return self.resize(img, params.height, params.width)

        elif op == OperationType.GRAYSCALE:
            return self.grayscale(img)

        elif op == OperationType.NORMALIZE:
            return self.normalize(img)

        elif op == OperationType.FLIP_H:
            return self.flip_horizontal(img)

        elif op == OperationType.FLIP_V:
            return self.flip_vertical(img)

        elif op == OperationType.CROP:
            if (
                params.crop_top is None
                or params.crop_left is None
                or params.crop_height is None
                or params.crop_width is None
            ):
                msg = "Crop requires top, left, height, and width"
                raise ValueError(msg)
            return self.crop(
                img,
                params.crop_top,
                params.crop_left,
                params.crop_height,
                params.crop_width,
            )

        elif op == OperationType.BLUR:
            if params.sigma is None:
                msg = "Blur requires sigma"
                raise ValueError(msg)
            return self.blur(img, params.sigma)

        elif op == OperationType.THRESHOLD:
            if params.threshold_value is None:
                msg = "Threshold requires value"
                raise ValueError(msg)
            return self.threshold(img, params.threshold_value)

        else:
            msg = f"Unsupported operation: {op}"
            raise ValueError(msg)

    def run_pipeline(
        self, images: list[Any], operations: list[OperationParams]
    ) -> list[Any]:
        """
        Run a sequence of operations on a list of images.

        Args:
            images: List of images in the framework's native format.
            operations: Sequence of operations to apply.

        Returns:
            List of processed images.
        """
        results = []
        for img in images:
            result = img
            for op in operations:
                result = self.apply_operation(result, op)
            results.append(result)
        return results

    def run_pipeline_batch(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list[Any]:
        """
        Run a complete pipeline from bytes to processed images.

        This is the main method used for benchmarking end-to-end workflows.

        Args:
            image_bytes_list: List of image bytes.
            operations: Sequence of operations to apply.

        Returns:
            List of processed images.
        """
        # Load images
        images = [self.load_from_bytes(data) for data in image_bytes_list]
        # Process
        return self.run_pipeline(images, operations)

    def warmup(self, sample_image: bytes, operations: list[OperationParams]) -> None:
        """
        Perform warmup iterations to stabilize performance.

        Args:
            sample_image: Sample image bytes for warmup.
            operations: Operations to warm up.
        """
        for _ in range(3):
            self.run_pipeline_batch([sample_image], operations)

    def prepare_decoded_images(self, png_bytes_list: list[bytes]) -> list[Any]:
        """
        Pre-decode PNG bytes to the framework's native format.

        This removes image decoding overhead from subsequent benchmark runs,
        allowing for fairer comparison of pure operation performance.

        Args:
            png_bytes_list: List of PNG image bytes.

        Returns:
            List of images in the framework's native format (decoded).
        """
        return [self.load_from_bytes(data) for data in png_bytes_list]

    def run_pipeline_on_decoded(
        self,
        decoded_images: list[Any],
        operations: list[OperationParams],
    ) -> list[Any]:
        """
        Run operations on pre-decoded images (skips decode overhead).

        This is the counterpart to run_pipeline_batch for benchmarking
        pure operation performance without image decoding.

        Args:
            decoded_images: List of pre-decoded images in native format.
            operations: Operations to apply.

        Returns:
            List of processed images.
        """
        return self.run_pipeline(decoded_images, operations)
