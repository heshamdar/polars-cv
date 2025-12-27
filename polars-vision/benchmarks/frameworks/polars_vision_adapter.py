"""
Polars-vision framework adapter for benchmarking.

This module provides adapters for polars-vision in both eager and streaming modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .base import BaseFrameworkAdapter, OperationParams, OperationType

# import os

# os.environ["POLARS_VERBOSE"] = "1"


if TYPE_CHECKING:
    import numpy.typing as npt


class PolarsVisionAdapter(BaseFrameworkAdapter):
    """
    Adapter for polars-vision image processing.

    This adapter uses the polars-vision plugin to process images via Polars
    DataFrames, supporting both eager and streaming execution modes.

    Attributes:
        name: Human-readable name of the adapter.
        streaming: Whether to use streaming execution mode.
    """

    supports_gpu: bool = False

    def __init__(self, streaming: bool = False) -> None:
        """
        Initialize the polars-vision adapter.

        Args:
            streaming: If True, use streaming execution (engine='streaming').
        """
        self.streaming = streaming
        self.name = f"polars-vision-{'streaming' if streaming else 'eager'}"
        self._pipeline_module: Any = None
        self._expressions_module: Any = None

    def is_available(self) -> bool:
        """
        Check if polars-vision is available.

        Returns:
            True if polars-vision can be imported, False otherwise.
        """
        try:
            import polars_vision.expressions  # noqa: F401
            from polars_vision import Pipeline  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_pipeline_class(self) -> type:
        """Get the Pipeline class from polars_vision."""
        if self._pipeline_module is None:
            from polars_vision import Pipeline

            self._pipeline_module = Pipeline
        return self._pipeline_module

    def _ensure_expressions_registered(self) -> None:
        """Ensure the cv namespace is registered."""
        if self._expressions_module is None:
            import polars_vision.expressions

            self._expressions_module = polars_vision.expressions

    def load_from_file(self, path: Path) -> bytes:
        """
        Load image bytes from a file.

        Args:
            path: Path to the image file.

        Returns:
            Image bytes.
        """
        return path.read_bytes()

    def load_from_bytes(self, data: bytes) -> bytes:
        """
        Pass through image bytes.

        For polars-vision, images are processed as bytes in a DataFrame.

        Args:
            data: Image bytes.

        Returns:
            Same image bytes.
        """
        return data

    def _build_pipeline(
        self, operations: list[OperationParams], sink_format: str = "numpy"
    ) -> Any:
        """
        Build a polars-vision pipeline from operations.

        Args:
            operations: List of operations to apply.
            sink_format: Output format for the sink.

        Returns:
            Pipeline instance.
        """
        Pipeline = self._get_pipeline_class()
        pipe = Pipeline().source("image_bytes")

        for op in operations:
            if op.operation == OperationType.RESIZE:
                pipe = pipe.resize(height=op.height, width=op.width)
            elif op.operation == OperationType.GRAYSCALE:
                pipe = pipe.grayscale()
            elif op.operation == OperationType.NORMALIZE:
                pipe = pipe.normalize(method="minmax")
            elif op.operation == OperationType.FLIP_H:
                pipe = pipe.flip_h()
            elif op.operation == OperationType.FLIP_V:
                pipe = pipe.flip_v()
            elif op.operation == OperationType.CROP:
                pipe = pipe.crop(
                    top=op.crop_top,
                    left=op.crop_left,
                    height=op.crop_height,
                    width=op.crop_width,
                )
            elif op.operation == OperationType.BLUR:
                pipe = pipe.blur(sigma=op.sigma)
            elif op.operation == OperationType.THRESHOLD:
                pipe = pipe.threshold(value=op.threshold_value)
            elif op.operation == OperationType.CAST:
                pipe = pipe.cast(dtype=op.dtype)
            elif op.operation == OperationType.SCALE:
                pipe = pipe.scale(factor=op.scale_factor)

        return pipe.sink(sink_format)

    def resize(self, img: bytes, height: int, width: int) -> bytes:
        """
        Resize an image.

        Args:
            img: Image bytes.
            height: Target height.
            width: Target width.

        Returns:
            Resized image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=height, width=width)
            .sink("blob")
        )

        df = pl.DataFrame({"images": [img]})
        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"][0]

    def grayscale(self, img: bytes) -> bytes:
        """
        Convert image to grayscale.

        Args:
            img: Image bytes.

        Returns:
            Grayscale image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        pipe = Pipeline().source("image_bytes").grayscale().sink("blob")

        df = pl.DataFrame({"images": [img]})
        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"][0]

    def normalize(self, img: bytes) -> bytes:
        """
        Normalize image values.

        Args:
            img: Image bytes.

        Returns:
            Normalized image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        pipe = Pipeline().source("image_bytes").normalize(method="minmax").sink("blob")

        df = pl.DataFrame({"images": [img]})
        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"][0]

    def flip_horizontal(self, img: bytes) -> bytes:
        """
        Flip image horizontally.

        Args:
            img: Image bytes.

        Returns:
            Flipped image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        pipe = Pipeline().source("image_bytes").flip_h().sink("blob")

        df = pl.DataFrame({"images": [img]})
        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"][0]

    def flip_vertical(self, img: bytes) -> bytes:
        """
        Flip image vertically.

        Args:
            img: Image bytes.

        Returns:
            Flipped image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        pipe = Pipeline().source("image_bytes").flip_v().sink("blob")

        df = pl.DataFrame({"images": [img]})
        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"][0]

    def crop(self, img: bytes, top: int, left: int, height: int, width: int) -> bytes:
        """
        Crop image.

        Args:
            img: Image bytes.
            top: Top offset.
            left: Left offset.
            height: Crop height.
            width: Crop width.

        Returns:
            Cropped image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        pipe = (
            Pipeline()
            .source("image_bytes")
            .crop(top=top, left=left, height=height, width=width)
            .sink("blob")
        )

        df = pl.DataFrame({"images": [img]})
        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"][0]

    def blur(self, img: bytes, sigma: float) -> bytes:
        """
        Apply Gaussian blur.

        Args:
            img: Image bytes.
            sigma: Blur sigma.

        Returns:
            Blurred image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        pipe = Pipeline().source("image_bytes").blur(sigma=sigma).sink("blob")

        df = pl.DataFrame({"images": [img]})
        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"][0]

    def threshold(self, img: bytes, value: int) -> bytes:
        """
        Apply binary threshold.

        Args:
            img: Image bytes.
            value: Threshold value.

        Returns:
            Thresholded image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        pipe = Pipeline().source("image_bytes").threshold(value=value).sink("blob")

        df = pl.DataFrame({"images": [img]})
        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"][0]

    def to_numpy(self, img: bytes) -> "npt.NDArray[np.uint8]":
        """
        Convert image bytes to NumPy array.

        For polars-vision, we output to numpy format and decode.

        Args:
            img: Image bytes (blob format).

        Returns:
            NumPy array.
        """
        # For blob format, we need to decode
        # This is a simplified version - full implementation would parse the blob
        import io

        from PIL import Image

        # Try to load as standard image format first
        try:
            pil_img = Image.open(io.BytesIO(img))
            return np.array(pil_img)
        except Exception:
            # Assume it's raw numpy bytes
            return np.frombuffer(img, dtype=np.uint8)

    def run_pipeline_batch(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list[bytes]:
        """
        Run a pipeline on a batch of images.

        This is the main benchmarking method that processes all images at once
        using Polars' parallel execution.

        Args:
            image_bytes_list: List of image bytes.
            operations: Operations to apply.

        Returns:
            List of processed image bytes.
        """
        self._ensure_expressions_registered()
        pipe = self._build_pipeline(operations, sink_format="blob")

        df = pl.DataFrame({"images": image_bytes_list})

        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"].to_list()

    def run_pipeline_batch_to_numpy(
        self,
        image_bytes_list: list[bytes],
        operations: list[OperationParams],
    ) -> list["npt.NDArray[np.float32]"]:
        """
        Run a pipeline and return NumPy arrays.

        Args:
            image_bytes_list: List of image bytes.
            operations: Operations to apply.

        Returns:
            List of NumPy arrays.
        """
        self._ensure_expressions_registered()
        pipe = self._build_pipeline(operations, sink_format="numpy")

        df = pl.DataFrame({"images": image_bytes_list})

        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        # Convert binary output to numpy arrays
        outputs = []
        for blob in result["processed"]:
            arr = np.frombuffer(blob, dtype=np.float32)
            outputs.append(arr)

        return outputs


class PolarsVisionEagerAdapter(PolarsVisionAdapter):
    """Polars-vision adapter with eager execution."""

    def __init__(self) -> None:
        """Initialize eager adapter."""
        super().__init__(streaming=False)


class PolarsVisionStreamingAdapter(PolarsVisionAdapter):
    """Polars-vision adapter with streaming execution."""

    def __init__(self) -> None:
        """Initialize streaming adapter."""
        super().__init__(streaming=True)
