"""Image metadata extraction, display utility, and error handling.

Demonstrates:
- .cv.width(), .cv.height(), .cv.channels(), .cv.image_dtype()
- show_images() display utility
- source(..., on_error="null") for graceful error handling

Run:
    uv run python polars-cv/examples/11_image_metadata.py
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import polars as pl
from PIL import Image
from polars_cv import Pipeline, show_images


def to_png_bytes(image: np.ndarray, mode: str = "RGB") -> bytes:
    """Encode an image to PNG bytes."""
    buf = BytesIO()
    Image.fromarray(image, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def demo_metadata() -> None:
    """Extract image metadata without decoding."""
    print("\n--- Image Metadata ---")

    rng = np.random.default_rng(42)

    small_rgb = rng.integers(0, 255, (64, 48, 3), dtype=np.uint8)
    large_rgb = rng.integers(0, 255, (256, 320, 3), dtype=np.uint8)
    gray = rng.integers(0, 255, (100, 100), dtype=np.uint8)

    df = pl.DataFrame(
        {
            "name": ["small_rgb", "large_rgb", "grayscale"],
            "image": [
                to_png_bytes(small_rgb),
                to_png_bytes(large_rgb),
                to_png_bytes(gray, mode="L"),
            ],
        }
    )

    result = df.with_columns(
        width=pl.col("image").cv.width(),
        height=pl.col("image").cv.height(),
        channels=pl.col("image").cv.channels(),
        dtype=pl.col("image").cv.image_dtype(),
    )

    print(result.select("name", "width", "height", "channels", "dtype"))


def demo_filter_by_size() -> None:
    """Filter images by their dimensions using metadata."""
    print("\n--- Filter by Size ---")

    rng = np.random.default_rng(7)
    images = {
        "tiny": rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
        "small": rng.integers(0, 255, (100, 80, 3), dtype=np.uint8),
        "medium": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "large": rng.integers(0, 255, (512, 640, 3), dtype=np.uint8),
    }

    df = pl.DataFrame(
        {
            "name": list(images.keys()),
            "image": [to_png_bytes(v) for v in images.values()],
        }
    )

    large_enough = df.filter(
        (pl.col("image").cv.width() >= 200) & (pl.col("image").cv.height() >= 200)
    )

    print(f"Images >= 200x200: {large_enough['name'].to_list()}")


def demo_show_images() -> None:
    """Demonstrate the show_images display utility."""
    print("\n--- show_images() ---")

    rng = np.random.default_rng(99)
    df = pl.DataFrame(
        {
            "image": [
                to_png_bytes(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
                for _ in range(3)
            ],
        }
    )

    show_images(df, "image", max_rows=3, max_width=100)
    print("(show_images renders HTML in Jupyter; prints summary in terminal)")


def demo_on_error_null() -> None:
    """Demonstrate graceful error handling with on_error='null'."""
    print("\n--- on_error='null' ---")

    rng = np.random.default_rng(42)
    valid_png = to_png_bytes(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    corrupt_data = b"not-a-real-image"

    df = pl.DataFrame(
        {
            "name": ["valid", "corrupt", "also_valid"],
            "image": [valid_png, corrupt_data, valid_png],
        }
    )

    pipe = Pipeline().source("image_bytes", on_error="null").resize(height=32, width=32)
    result = df.with_columns(
        processed=pl.col("image").cv.pipe(pipe).sink("numpy"),
    )

    print("Results with on_error='null':")
    for row in result.iter_rows(named=True):
        status = "OK" if row["processed"] is not None else "null (decode failed)"
        print(f"  {row['name']}: {status}")


def main() -> None:
    """Run all metadata and display demos."""
    demo_metadata()
    demo_filter_by_size()
    demo_show_images()
    demo_on_error_null()
    print("\nAll metadata & display demos complete.")


if __name__ == "__main__":
    main()
