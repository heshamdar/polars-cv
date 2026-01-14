"""
polars-cv framework adapter for benchmarking.

This module provides adapters for polars-cv in both eager and streaming modes.
"""

from __future__ import annotations

import os

os.environ["POLARS_IDEAL_MORSEL_SIZE"] = "10"


from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from .base import BaseFrameworkAdapter, OperationParams, OperationType

# import os

# os.environ["POLARS_VERBOSE"] = "1"


if TYPE_CHECKING:
    import numpy.typing as npt


class PolarsCVAdapter(BaseFrameworkAdapter):
    """
    Adapter for polars-cv image processing.

    This adapter uses the polars-cv plugin to process images via Polars
    DataFrames, supporting both eager and streaming execution modes.

    Attributes:
        name: Human-readable name of the adapter.
        streaming: Whether to use streaming execution mode.
    """

    supports_gpu: bool = False

    def __init__(self, streaming: bool = False) -> None:
        """
        Initialize the polars-cv adapter.

        Args:
            streaming: If True, use streaming execution (engine='streaming').
        """
        self.streaming = streaming
        self.name = f"polars-cv-{'streaming' if streaming else 'eager'}"
        self._pipeline_module: Any = None
        self._expressions_module: Any = None

    def is_available(self) -> bool:
        """
        Check if polars-cv is available.

        Returns:
            True if polars-cv can be imported, False otherwise.
        """
        try:
            import polars_cv.expressions  # noqa: F401
            from polars_cv import Pipeline  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_pipeline_class(self) -> type:
        """Get the Pipeline class from polars_cv."""
        if self._pipeline_module is None:
            from polars_cv import Pipeline

            self._pipeline_module = Pipeline
        return self._pipeline_module

    def _ensure_expressions_registered(self) -> None:
        """Ensure the cv namespace is registered."""
        if self._expressions_module is None:
            import polars_cv.expressions

            self._expressions_module = polars_cv.expressions

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

        For polars-cv, images are processed as bytes in a DataFrame.

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
        Build a polars-cv pipeline from operations.

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
                # Use bilinear interpolation for consistency across frameworks
                pipe = pipe.resize(height=op.height, width=op.width, filter="bilinear")
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
                # adding specific handling for comparison since other frameworks implicitly convert to grayscale
                pipe = pipe.grayscale().threshold(value=op.threshold_value)
            elif op.operation == OperationType.CAST:
                pipe = pipe.cast(dtype=op.dtype)
            elif op.operation == OperationType.SCALE:
                pipe = pipe.scale(factor=op.scale_factor)

        return pipe.sink(sink_format)

    def _build_pipeline_blob_source(
        self, operations: list[OperationParams], sink_format: str = "numpy"
    ) -> Any:
        """
        Build a polars-cv pipeline with blob source (for in-memory benchmarks).

        This creates a pipeline that reads from already-decoded blob format,
        avoiding the PNG decode overhead for fair comparison with OpenCV.

        Args:
            operations: List of operations to apply.
            sink_format: Output format for the sink.

        Returns:
            Pipeline instance.
        """
        Pipeline = self._get_pipeline_class()
        pipe = Pipeline().source("blob")

        for op in operations:
            if op.operation == OperationType.RESIZE:
                pipe = pipe.resize(height=op.height, width=op.width, filter="bilinear")
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
                pipe = pipe.grayscale().threshold(value=op.threshold_value)
            elif op.operation == OperationType.CAST:
                pipe = pipe.cast(dtype=op.dtype)
            elif op.operation == OperationType.SCALE:
                pipe = pipe.scale(factor=op.scale_factor)

        return pipe.sink(sink_format)

    def prepare_blob_images(self, png_bytes_list: list[bytes]) -> list[bytes]:
        """
        Convert PNG bytes to blob format for in-memory benchmarking.

        This method decodes PNG images and re-encodes them as VIEW protocol blobs,
        removing the image decode overhead from subsequent benchmarks.

        Args:
            png_bytes_list: List of PNG image bytes.

        Returns:
            List of blob-encoded image bytes.
        """
        self._ensure_expressions_registered()
        Pipeline = self._get_pipeline_class()

        # Pipeline: decode PNG, encode to blob
        pipe = Pipeline().source("image_bytes").sink("blob")

        df = pl.DataFrame({"images": png_bytes_list})
        result = df.with_columns(blob=pl.col("images").cv.pipeline(pipe))
        return result["blob"].to_list()

    def apply_operations_blob(
        self,
        blob_images: list[bytes],
        operations: list[OperationParams],
    ) -> list[bytes]:
        """
        Apply operations to blob-encoded images (for in-memory benchmarks).

        This is the counterpart to OpenCV's direct array processing - it starts
        from already-decoded images rather than PNG bytes.

        Args:
            blob_images: List of blob-encoded image bytes (from prepare_blob_images).
            operations: List of operations to apply.

        Returns:
            List of processed image bytes.
        """
        self._ensure_expressions_registered()
        pipe = self._build_pipeline_blob_source(operations, sink_format="numpy")

        df = pl.DataFrame({"images": blob_images})

        if self.streaming:
            result = (
                df.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = df.with_columns(processed=pl.col("images").cv.pipeline(pipe))

        return result["processed"].to_list()

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
            .resize(height=height, width=width, filter="bilinear")
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

    def to_numpy(self, img: dict[str, Any] | bytes) -> "npt.NDArray[np.uint8]":
        """
        Convert image struct or bytes to NumPy array.

        For polars-cv, we output to numpy format which returns a struct
        with 'data', 'dtype', and 'shape' fields. Use numpy_from_struct
        to properly parse it.

        Args:
            img: Image struct (numpy/torch sink format) or bytes (blob format).

        Returns:
            NumPy array.
        """
        # Use numpy_from_struct to parse the numpy/torch sink struct format
        if isinstance(img, dict):
            try:
                from polars_cv import numpy_from_struct

                return numpy_from_struct(img)
            except Exception:
                pass

        # Fallback for bytes: try to load as standard image format (PNG/JPEG)
        if isinstance(img, bytes):
            try:
                import io

                from PIL import Image

                pil_img = Image.open(io.BytesIO(img))
                return np.array(pil_img)
            except Exception:
                pass

            # Last resort: raw bytes (likely incorrect shape)
            return np.frombuffer(img, dtype=np.uint8)

        raise TypeError(f"Expected dict or bytes, got {type(img).__name__}")

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
        from polars_cv import numpy_from_struct

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

        # Convert struct output to numpy arrays
        outputs = []
        for struct_row in result["processed"]:
            arr = numpy_from_struct(struct_row)
            outputs.append(arr)

        return outputs

    def prepare_decoded_images(self, png_bytes_list: list[bytes]) -> pl.DataFrame:
        """
        Pre-decode PNG bytes to a DataFrame with blob column for fair benchmarking.

        This converts PNG bytes to VIEW protocol blob format and stores them
        in a pre-constructed DataFrame, removing both image decoding AND
        DataFrame construction overhead from subsequent benchmark runs.

        This provides parity with OpenCV benchmarks where images are pre-decoded
        to numpy arrays before timing begins.

        Args:
            png_bytes_list: List of PNG image bytes.

        Returns:
            DataFrame with 'images' column containing blob-encoded image bytes.
        """
        blob_images = self.prepare_blob_images(png_bytes_list)
        return pl.DataFrame({"images": blob_images})

    def run_pipeline_on_decoded(
        self,
        decoded_images: pl.DataFrame,
        operations: list[OperationParams],
    ) -> pl.DataFrame:
        """
        Run operations on pre-decoded DataFrame (skips decode + DataFrame creation).

        This provides fair comparison with other frameworks by:
        - Starting from an already-constructed DataFrame (like OpenCV starts from arrays)
        - Returning a DataFrame (not extracting to Python list)

        Args:
            decoded_images: DataFrame with 'images' column containing blob bytes.
            operations: Operations to apply.

        Returns:
            DataFrame with 'processed' column containing results.
        """
        self._ensure_expressions_registered()
        pipe = self._build_pipeline_blob_source(operations, sink_format="numpy")

        if self.streaming:
            result = (
                decoded_images.lazy()
                .with_columns(processed=pl.col("images").cv.pipeline(pipe))
                .collect(engine="streaming")
            )
        else:
            result = decoded_images.with_columns(
                processed=pl.col("images").cv.pipeline(pipe)
            )

        return result


class PolarsCVEagerAdapter(PolarsCVAdapter):
    """polars-cv adapter with eager execution."""

    def __init__(self) -> None:
        """Initialize eager adapter."""
        super().__init__(streaming=False)


class PolarsCVStreamingAdapter(PolarsCVAdapter):
    """polars-cv adapter with streaming execution."""

    def __init__(self) -> None:
        """Initialize streaming adapter."""
        super().__init__(streaming=True)
