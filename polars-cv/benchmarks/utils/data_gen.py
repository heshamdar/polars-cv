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
        temp_dir = Path(tempfile.mkdtemp(prefix="polars_cv_bench_"))
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


@dataclass
class ImageFolderDataset:
    """
    An ImageFolder-style dataset with Parquet metadata.

    This structure is compatible with both HuggingFace datasets (ImageFolder)
    and polars-cv (via the Parquet metadata file).

    The metadata file is stored OUTSIDE the images directory to avoid
    conflicts with HuggingFace's imagefolder loader.
    """

    root_dir: Path
    images_dir: Path  # Subdirectory containing class folders (for HuggingFace)
    metadata_path: Path  # Outside images_dir (for polars-cv)
    class_names: list[str]
    image_count: int
    image_size: tuple[int, int]

    def cleanup(self) -> None:
        """Clean up the dataset directory."""
        if self.root_dir.exists():
            shutil.rmtree(self.root_dir)


def generate_imagefolder_dataset(
    output_dir: str | Path,
    num_images: int = 1000,
    num_classes: int = 10,
    height: int = 224,
    width: int = 224,
    pattern: str = "mixed",
    base_seed: int = 42,
) -> ImageFolderDataset:
    """
    Generate an ImageFolder-style dataset with Parquet metadata.

    Creates a directory structure compatible with both HuggingFace datasets
    (imagefolder format) and polars-cv (via Parquet metadata).

    Directory structure:
        output_dir/
        ├── images/           <- HuggingFace imagefolder reads from here
        │   ├── class_0/
        │   │   ├── image_000000.png
        │   │   └── ...
        │   └── class_1/
        │       └── ...
        └── metadata.parquet  <- polars-cv reads from here (outside images/)

    The metadata.parquet is placed OUTSIDE the images/ directory to avoid
    conflicts with HuggingFace's imagefolder loader which looks for metadata files.

    Args:
        output_dir: Root directory for the dataset.
        num_images: Total number of images to generate.
        num_classes: Number of classification categories.
        height: Image height in pixels.
        width: Image width in pixels.
        pattern: Image pattern ("gradient", "noise", "checkerboard", "mixed").
        base_seed: Random seed for reproducibility.

    Returns:
        ImageFolderDataset with paths to the generated data.

    Example:
        ```python
        >>> dataset = generate_imagefolder_dataset("./synthetic_data", num_images=1000)
        >>> print(dataset.images_dir)     # For HuggingFace imagefolder
        >>> print(dataset.metadata_path)  # For polars-cv
        >>> print(dataset.class_names)    # ['class_0', 'class_1', ...]
        ```
    """
    import polars as pl

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create images subdirectory (HuggingFace reads from here)
    images_path = output_path / "images"
    images_path.mkdir(exist_ok=True)

    # Generate class names
    class_names = [f"class_{i}" for i in range(num_classes)]

    # Create class directories inside images/
    for class_name in class_names:
        (images_path / class_name).mkdir(exist_ok=True)

    # Generate images and track metadata
    paths: list[str] = []
    labels: list[int] = []
    class_names_col: list[str] = []

    patterns = (
        ["gradient", "noise", "checkerboard"] if pattern == "mixed" else [pattern]
    )
    rng = np.random.default_rng(base_seed)

    for i in range(num_images):
        # Assign to a class (balanced distribution)
        label = i % num_classes
        class_name = class_names[label]

        # Generate image with varied patterns
        current_pattern = patterns[i % len(patterns)]
        seed = base_seed + i if current_pattern == "noise" else None

        # Add some per-image variation to make images distinguishable
        if current_pattern == "gradient":
            # Add rotation variation via different gradient directions
            arr = generate_gradient_image(height, width, 3)
            # Add small random noise to make each image unique
            noise = rng.integers(-10, 10, size=arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif current_pattern == "noise":
            arr = generate_noise_image(height, width, 3, seed)
        else:
            arr = generate_pattern_image(height, width, 3, current_pattern)
            # Add variation
            arr = np.roll(arr, shift=i % 32, axis=0)

        # Save image inside images/ subdirectory
        img_bytes = array_to_png_bytes(arr)
        img_path = images_path / class_name / f"image_{i:06d}.png"
        img_path.write_bytes(img_bytes)

        # Track metadata (use absolute path for polars-cv compatibility)
        paths.append(str(img_path.absolute()))
        labels.append(label)
        class_names_col.append(class_name)

    # Create Parquet metadata file OUTSIDE images/ directory
    metadata_df = pl.DataFrame(
        {
            "path": paths,
            "label": labels,
            "class_name": class_names_col,
        }
    )
    metadata_path = output_path / "metadata.parquet"
    metadata_df.write_parquet(metadata_path)

    return ImageFolderDataset(
        root_dir=output_path,
        images_dir=images_path,
        metadata_path=metadata_path,
        class_names=class_names,
        image_count=num_images,
        image_size=(width, height),
    )


@dataclass
class LanceDataset:
    """
    A Lance dataset containing image bytes inline with metadata.

    All data (image bytes + label + class_name) is stored in a single
    Lance file, unlike ImageFolderDataset which uses separate image files
    alongside a metadata.parquet.

    This enables O(1-2 IO ops) random access via Lance's repetition index,
    versus Parquet's pattern of: read metadata row → seek to image file.
    """

    dataset_path: Path
    class_names: list[str]
    image_count: int
    image_size: tuple[int, int]

    def cleanup(self) -> None:
        """Clean up the Lance dataset directory."""
        if self.dataset_path.exists():
            shutil.rmtree(self.dataset_path)


def generate_lance_dataset(
    output_dir: str | Path,
    num_images: int = 1000,
    num_classes: int = 10,
    height: int = 224,
    width: int = 224,
    pattern: str = "mixed",
    base_seed: int = 42,
) -> "LanceDataset":
    """
    Generate a Lance dataset with image bytes stored inline alongside metadata.

    Unlike :func:`generate_imagefolder_dataset`, all data lives in a single
    ``.lance`` directory — no separate image files.  This is the canonical
    Lance layout for CV/ML datasets and enables Lance's fast random access.

    The images generated are identical to those produced by
    :func:`generate_imagefolder_dataset` when called with the same
    ``base_seed``, so the two functions produce directly comparable datasets.

    Schema stored in Lance::

        image_bytes : large_binary   (PNG-encoded pixel data)
        label       : int32
        class_name  : string

    Args:
        output_dir: Parent directory; a ``dataset.lance`` sub-directory will
            be created inside it.
        num_images: Total number of images to generate.
        num_classes: Number of classification categories.
        height: Image height in pixels.
        width: Image width in pixels.
        pattern: Image pattern (``"gradient"``, ``"noise"``,
            ``"checkerboard"``, ``"mixed"``).
        base_seed: Random seed — must match the seed used for the
            corresponding :func:`generate_imagefolder_dataset` call to keep
            comparisons fair.

    Returns:
        :class:`LanceDataset` pointing at the written ``.lance`` directory.

    Raises:
        ImportError: If ``lance`` is not installed.

    Example::

        >>> ds = generate_lance_dataset("./data", num_images=500)
        >>> print(ds.dataset_path)   # ./data/dataset.lance
        >>> ds.cleanup()
    """
    try:
        import lance
        import pyarrow as pa
    except ImportError as exc:
        msg = (
            "lance and pyarrow are required for Lance dataset generation. "
            "Install them with: pip install lance pyarrow"
        )
        raise ImportError(msg) from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_path = output_path / "dataset.lance"

    class_names = [f"class_{i}" for i in range(num_classes)]

    patterns = (
        ["gradient", "noise", "checkerboard"] if pattern == "mixed" else [pattern]
    )
    rng = np.random.default_rng(base_seed)

    image_bytes_list: list[bytes] = []
    labels: list[int] = []
    class_names_col: list[str] = []

    for i in range(num_images):
        label = i % num_classes
        class_name = class_names[label]
        current_pattern = patterns[i % len(patterns)]
        seed = base_seed + i if current_pattern == "noise" else None

        if current_pattern == "gradient":
            arr = generate_gradient_image(height, width, 3)
            noise = rng.integers(-10, 10, size=arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif current_pattern == "noise":
            arr = generate_noise_image(height, width, 3, seed)
        else:
            arr = generate_pattern_image(height, width, 3, current_pattern)
            arr = np.roll(arr, shift=i % 32, axis=0)

        image_bytes_list.append(array_to_png_bytes(arr))
        labels.append(label)
        class_names_col.append(class_name)

    table = pa.table(
        {
            "image_bytes": pa.array(image_bytes_list, type=pa.large_binary()),
            "label": pa.array(labels, type=pa.int32()),
            "class_name": pa.array(class_names_col, type=pa.string()),
        }
    )

    lance.write_dataset(table, str(dataset_path))

    return LanceDataset(
        dataset_path=dataset_path,
        class_names=class_names,
        image_count=num_images,
        image_size=(width, height),
    )


@contextmanager
def temporary_lance_dataset(
    num_images: int = 1000,
    num_classes: int = 10,
    height: int = 224,
    width: int = 224,
    pattern: str = "mixed",
    base_seed: int = 42,
) -> "Iterator[LanceDataset]":
    """
    Context manager for creating a temporary Lance dataset with automatic cleanup.

    Args:
        num_images: Total number of images.
        num_classes: Number of categories.
        height: Image height.
        width: Image width.
        pattern: Image pattern.
        base_seed: Random seed.

    Yields:
        :class:`LanceDataset` that will be cleaned up on exit.

    Example::

        >>> with temporary_lance_dataset(num_images=100) as ds:
        ...     # use the dataset ...
        ...     pass
        >>> # .lance directory is automatically removed
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="polars_cv_lance_"))
    dataset = generate_lance_dataset(
        output_dir=temp_dir,
        num_images=num_images,
        num_classes=num_classes,
        height=height,
        width=width,
        pattern=pattern,
        base_seed=base_seed,
    )
    try:
        yield dataset
    finally:
        dataset.cleanup()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


@contextmanager
def temporary_imagefolder_dataset(
    num_images: int = 1000,
    num_classes: int = 10,
    height: int = 224,
    width: int = 224,
    pattern: str = "mixed",
    base_seed: int = 42,
) -> Iterator[ImageFolderDataset]:
    """
    Context manager for creating a temporary ImageFolder dataset with cleanup.

    Args:
        num_images: Total number of images.
        num_classes: Number of categories.
        height: Image height.
        width: Image width.
        pattern: Image pattern.
        base_seed: Random seed.

    Yields:
        ImageFolderDataset that will be cleaned up on exit.

    Example:
        ```python
        >>> with temporary_imagefolder_dataset(num_images=100) as dataset:
        ...     df = pl.read_parquet(dataset.metadata_path)
        ...     # Use the dataset...
        >>> # Directory is automatically cleaned up
        ```
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="polars_cv_imagefolder_"))
    dataset = generate_imagefolder_dataset(
        output_dir=temp_dir,
        num_images=num_images,
        num_classes=num_classes,
        height=height,
        width=width,
        pattern=pattern,
        base_seed=base_seed,
    )
    try:
        yield dataset
    finally:
        dataset.cleanup()
