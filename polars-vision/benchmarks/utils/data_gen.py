"""
Test image generation utilities.

This module provides functions for generating synthetic test images
for benchmarking, including both in-memory bytes and temporary files.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


@dataclass
class GeneratedImageSet:
    """A set of generated test images."""

    image_bytes: list[bytes]
    file_paths: list[Path] | None
    size: tuple[int, int]
    channels: int
    count: int
    temp_dir: Path | None = None

    def cleanup(self) -> None:
        """Clean up temporary files if they exist."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.file_paths = None


def generate_gradient_image(
    height: int,
    width: int,
    channels: int = 3,
) -> "npt.NDArray[np.uint8]":
    """
    Generate a gradient test image.

    Creates a diagonal gradient from top-left to bottom-right.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels (1 for grayscale, 3 for RGB).

    Returns:
        NumPy array of shape (height, width, channels) with uint8 values.
    """
    # Create gradient values
    y = np.linspace(0, 255, height, dtype=np.float32)
    x = np.linspace(0, 255, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Combine into diagonal gradient
    gradient = ((xx + yy) / 2).astype(np.uint8)

    if channels == 1:
        return gradient[:, :, np.newaxis]

    # Create RGB with slight variations per channel
    result = np.zeros((height, width, channels), dtype=np.uint8)
    for c in range(channels):
        offset = c * 30  # Slight offset per channel
        result[:, :, c] = ((gradient.astype(np.int32) + offset) % 256).astype(np.uint8)

    return result


def generate_noise_image(
    height: int,
    width: int,
    channels: int = 3,
    seed: int | None = None,
) -> "npt.NDArray[np.uint8]":
    """
    Generate a random noise test image.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        seed: Random seed for reproducibility.

    Returns:
        NumPy array of shape (height, width, channels) with uint8 values.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)


def generate_pattern_image(
    height: int,
    width: int,
    channels: int = 3,
    pattern: str = "checkerboard",
    block_size: int = 32,
) -> "npt.NDArray[np.uint8]":
    """
    Generate a patterned test image.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        pattern: Pattern type ("checkerboard", "stripes_h", "stripes_v").
        block_size: Size of pattern blocks in pixels.

    Returns:
        NumPy array of shape (height, width, channels) with uint8 values.
    """
    result = np.zeros((height, width), dtype=np.uint8)

    if pattern == "checkerboard":
        for y in range(height):
            for x in range(width):
                if ((x // block_size) + (y // block_size)) % 2 == 0:
                    result[y, x] = 255
    elif pattern == "stripes_h":
        for y in range(height):
            if (y // block_size) % 2 == 0:
                result[y, :] = 255
    elif pattern == "stripes_v":
        for x in range(width):
            if (x // block_size) % 2 == 0:
                result[:, x] = 255

    if channels == 1:
        return result[:, :, np.newaxis]

    # Expand to RGB
    return np.stack([result] * channels, axis=-1)


def array_to_png_bytes(arr: "npt.NDArray[np.uint8]") -> bytes:
    """
    Convert a NumPy array to PNG bytes.

    Args:
        arr: NumPy array of shape (H, W) or (H, W, C).

    Returns:
        PNG-encoded bytes.
    """
    # Use PIL for PNG encoding
    from PIL import Image

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]

    if arr.ndim == 2:
        mode = "L"
    elif arr.shape[2] == 3:
        mode = "RGB"
    elif arr.shape[2] == 4:
        mode = "RGBA"
    else:
        msg = f"Unsupported channel count: {arr.shape[2]}"
        raise ValueError(msg)

    img = Image.fromarray(arr, mode=mode)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_image_bytes(
    height: int,
    width: int,
    channels: int = 3,
    pattern: str = "gradient",
    seed: int | None = None,
) -> bytes:
    """
    Generate a single test image as PNG bytes.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        pattern: Image pattern ("gradient", "noise", "checkerboard").
        seed: Random seed for noise pattern.

    Returns:
        PNG-encoded bytes.
    """
    if pattern == "gradient":
        arr = generate_gradient_image(height, width, channels)
    elif pattern == "noise":
        arr = generate_noise_image(height, width, channels, seed)
    elif pattern == "checkerboard":
        arr = generate_pattern_image(height, width, channels, "checkerboard")
    else:
        msg = f"Unknown pattern: {pattern}"
        raise ValueError(msg)

    return array_to_png_bytes(arr)


def generate_image_set(
    count: int,
    height: int,
    width: int,
    channels: int = 3,
    pattern: str = "gradient",
    create_files: bool = False,
    base_seed: int = 42,
) -> GeneratedImageSet:
    """
    Generate a set of test images.

    Args:
        count: Number of images to generate.
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        pattern: Image pattern ("gradient", "noise", "checkerboard", "mixed").
        create_files: Whether to create temporary files in addition to bytes.
        base_seed: Base random seed for reproducibility.

    Returns:
        GeneratedImageSet containing the generated images.
    """
    image_bytes: list[bytes] = []
    file_paths: list[Path] | None = None
    temp_dir: Path | None = None

    if create_files:
        temp_dir = Path(tempfile.mkdtemp(prefix="polars_vision_bench_"))
        file_paths = []

    patterns = (
        ["gradient", "noise", "checkerboard"] if pattern == "mixed" else [pattern]
    )

    for i in range(count):
        current_pattern = patterns[i % len(patterns)]
        seed = base_seed + i if current_pattern == "noise" else None
        img_bytes = generate_image_bytes(height, width, channels, current_pattern, seed)
        image_bytes.append(img_bytes)

        if create_files and temp_dir is not None and file_paths is not None:
            file_path = temp_dir / f"image_{i:06d}.png"
            file_path.write_bytes(img_bytes)
            file_paths.append(file_path)

    return GeneratedImageSet(
        image_bytes=image_bytes,
        file_paths=file_paths,
        size=(width, height),
        channels=channels,
        count=count,
        temp_dir=temp_dir,
    )


@contextmanager
def temporary_image_set(
    count: int,
    height: int,
    width: int,
    channels: int = 3,
    pattern: str = "gradient",
    base_seed: int = 42,
) -> Iterator[GeneratedImageSet]:
    """
    Context manager for generating temporary test images with cleanup.

    Args:
        count: Number of images to generate.
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of color channels.
        pattern: Image pattern.
        base_seed: Base random seed.

    Yields:
        GeneratedImageSet with temporary files that will be cleaned up on exit.
    """
    image_set = generate_image_set(
        count=count,
        height=height,
        width=width,
        channels=channels,
        pattern=pattern,
        create_files=True,
        base_seed=base_seed,
    )
    try:
        yield image_set
    finally:
        image_set.cleanup()
