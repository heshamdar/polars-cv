"""Convolution, edge detection, and histogram equalization.

Demonstrates:
- convolve2d with custom kernels
- sobel / laplacian / sharpen
- canny edge detection
- equalize_histogram

Run:
    uv run python polars-cv/examples/10_edge_detection_and_filters.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
from polars_cv import Pipeline, numpy_from_struct

OUTPUT_DIR = Path(__file__).parent / "outputs"


def make_image(size: int = 160) -> np.ndarray:
    """Create a synthetic image with geometric shapes for edge detection."""
    img = np.full((size, size), 40, dtype=np.uint8)
    center = size // 2
    y, x = np.ogrid[:size, :size]
    circle = (x - center) ** 2 + (y - center) ** 2 < (size // 4) ** 2
    img[circle] = 200
    quarter = size // 4
    img[quarter : quarter + size // 3, quarter : quarter + size // 3] = 160
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 10, size=(size, size), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a grayscale image to PNG bytes."""
    buf = BytesIO()
    Image.fromarray(image, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def render_grid(
    images: list[np.ndarray],
    titles: list[str],
    output_name: str,
    cols: int = 3,
) -> None:
    """Render a gallery grid and save to disk."""
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for idx, (img, title) in enumerate(zip(images, titles, strict=True)):
        ax = axes[idx // cols][idx % cols]
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for idx in range(len(images), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("Saved:", path)


def demo_convolution() -> None:
    """Demonstrate custom 2D convolution kernels."""
    print("\n--- Custom Convolution ---")

    img = make_image()
    png = to_png_bytes(img)
    df = pl.DataFrame({"image": [png]})

    emboss_kernel = [-2, -1, 0, -1, 1, 1, 0, 1, 2]
    edge_kernel = [-1, -1, -1, -1, 8, -1, -1, -1, -1]
    blur_kernel = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    emboss = (
        Pipeline().source("image_bytes").grayscale().convolve2d(emboss_kernel, ksize=3)
    )
    edge = Pipeline().source("image_bytes").grayscale().convolve2d(edge_kernel, ksize=3)
    blur = (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .convolve2d(blur_kernel, ksize=3, normalize=True)
    )

    result = df.with_columns(
        emboss=pl.col("image").cv.pipe(emboss).sink("numpy"),
        edge=pl.col("image").cv.pipe(edge).sink("numpy"),
        blur=pl.col("image").cv.pipe(blur).sink("numpy"),
    )

    images = [img]
    titles = ["Original"]
    for name, label in [
        ("emboss", "Emboss Kernel"),
        ("edge", "Edge Kernel"),
        ("blur", "Box Blur (normalized)"),
    ]:
        arr = numpy_from_struct(result[name][0])
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        images.append(arr)
        titles.append(label)

    render_grid(images, titles, "10_convolution.png", cols=4)
    print("Applied custom convolution kernels: emboss, edge, box blur")


def demo_gradient_operators() -> None:
    """Demonstrate Sobel, Laplacian, and Sharpen operators."""
    print("\n--- Gradient Operators ---")

    img = make_image()
    png = to_png_bytes(img)
    df = pl.DataFrame({"image": [png]})

    sobel_x = Pipeline().source("image_bytes").grayscale().sobel(axis="x")
    sobel_y = Pipeline().source("image_bytes").grayscale().sobel(axis="y")
    lap = Pipeline().source("image_bytes").grayscale().laplacian()
    sharp = Pipeline().source("image_bytes").grayscale().sharpen(strength=2.0)

    result = df.with_columns(
        sobel_x=pl.col("image").cv.pipe(sobel_x).sink("numpy"),
        sobel_y=pl.col("image").cv.pipe(sobel_y).sink("numpy"),
        laplacian=pl.col("image").cv.pipe(lap).sink("numpy"),
        sharpened=pl.col("image").cv.pipe(sharp).sink("numpy"),
    )

    images = [img]
    titles = ["Original"]
    for name, label in [
        ("sobel_x", "Sobel X"),
        ("sobel_y", "Sobel Y"),
        ("laplacian", "Laplacian"),
        ("sharpened", "Sharpen (2x)"),
    ]:
        arr = numpy_from_struct(result[name][0])
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        images.append(arr)
        titles.append(label)

    render_grid(images, titles, "10_gradient_operators.png", cols=3)
    print("Applied Sobel X/Y, Laplacian, and Sharpen operators")


def demo_canny() -> None:
    """Demonstrate Canny edge detection."""
    print("\n--- Canny Edge Detection ---")

    img = make_image()
    png = to_png_bytes(img)
    df = pl.DataFrame({"image": [png]})

    canny_default = Pipeline().source("image_bytes").grayscale().canny()
    canny_tight = (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .canny(
            low_threshold=30.0,
            high_threshold=80.0,
        )
    )
    canny_loose = (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .canny(
            low_threshold=80.0,
            high_threshold=200.0,
        )
    )

    result = df.with_columns(
        default=pl.col("image").cv.pipe(canny_default).sink("numpy"),
        tight=pl.col("image").cv.pipe(canny_tight).sink("numpy"),
        loose=pl.col("image").cv.pipe(canny_loose).sink("numpy"),
    )

    images = [img]
    titles = ["Original"]
    for name, label in [
        ("default", "Canny (50/150)"),
        ("tight", "Canny (30/80)"),
        ("loose", "Canny (80/200)"),
    ]:
        arr = numpy_from_struct(result[name][0])
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        images.append(arr)
        titles.append(label)

    render_grid(images, titles, "10_canny.png", cols=4)
    print("Applied Canny edge detection with different thresholds")


def demo_histogram_equalization() -> None:
    """Demonstrate histogram equalization for contrast enhancement."""
    print("\n--- Histogram Equalization ---")

    rng = np.random.default_rng(42)
    low_contrast = np.clip(
        rng.normal(loc=128, scale=20, size=(160, 160)),
        0,
        255,
    ).astype(np.uint8)
    png = to_png_bytes(low_contrast)
    df = pl.DataFrame({"image": [png]})

    equalized = Pipeline().source("image_bytes").grayscale().equalize_histogram()
    result = df.with_columns(
        eq=pl.col("image").cv.pipe(equalized).sink("numpy"),
    )

    eq_arr = numpy_from_struct(result["eq"][0])
    if eq_arr.ndim == 3 and eq_arr.shape[2] == 1:
        eq_arr = eq_arr[:, :, 0]

    render_grid(
        [low_contrast, eq_arr],
        ["Low Contrast", "Equalized"],
        "10_histogram_equalization.png",
        cols=2,
    )
    print("Applied histogram equalization to low-contrast image")


def main() -> None:
    """Run all edge detection and filter demos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_convolution()
    demo_gradient_operators()
    demo_canny()
    demo_histogram_equalization()
    print("\nAll edge detection & filter demos complete.")


if __name__ == "__main__":
    main()
