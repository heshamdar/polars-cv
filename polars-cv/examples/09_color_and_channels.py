"""Color conversion, channel operations, and intensity adjustments.

Demonstrates:
- convert_color / to_hsv / to_lab / to_bgr / to_ycbcr
- channel_select / channel_swap
- adjust_contrast / adjust_gamma / adjust_brightness / invert

Run:
    uv run python polars-cv/examples/09_color_and_channels.py
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


def make_image(size: int = 128) -> np.ndarray:
    """Create a synthetic RGB image with distinct color regions."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    third = size // 3
    img[:third, :, 0] = 220
    img[:third, :, 1] = 60
    img[:third, :, 2] = 60
    img[third : 2 * third, :, 0] = 60
    img[third : 2 * third, :, 1] = 200
    img[third : 2 * third, :, 2] = 60
    img[2 * third :, :, 0] = 60
    img[2 * third :, :, 1] = 60
    img[2 * third :, :, 2] = 220
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 25, size=(size, size, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB image to PNG bytes."""
    buf = BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
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
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for idx in range(len(images), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("Saved:", path)


def demo_color_conversion() -> None:
    """Demonstrate color space conversion operations."""
    print("\n--- Color Conversion ---")

    img = make_image()
    png = to_png_bytes(img)
    df = pl.DataFrame({"image": [png]})

    to_hsv = Pipeline().source("image_bytes").to_hsv()
    to_lab = Pipeline().source("image_bytes").to_lab()
    to_ycbcr = Pipeline().source("image_bytes").to_ycbcr()
    to_gray = Pipeline().source("image_bytes").grayscale()

    result = df.with_columns(
        hsv=pl.col("image").cv.pipe(to_hsv).sink("numpy"),
        lab=pl.col("image").cv.pipe(to_lab).sink("numpy"),
        ycbcr=pl.col("image").cv.pipe(to_ycbcr).sink("numpy"),
        gray=pl.col("image").cv.pipe(to_gray).sink("numpy"),
    )

    images = [img]
    titles = ["Original RGB"]
    for name in ["hsv", "lab", "ycbcr", "gray"]:
        arr = numpy_from_struct(result[name][0])
        images.append(arr)
        titles.append(name.upper())

    render_grid(images, titles, "09_color_conversion.png", cols=3)
    print("Converted to HSV, LAB, YCbCr, and Gray color spaces")


def demo_channel_operations() -> None:
    """Demonstrate channel_select and channel_swap."""
    print("\n--- Channel Operations ---")

    img = make_image()
    png = to_png_bytes(img)
    df = pl.DataFrame({"image": [png]})

    red_ch = Pipeline().source("image_bytes").channel_select(index=0)
    green_ch = Pipeline().source("image_bytes").channel_select(index=1)
    blue_ch = Pipeline().source("image_bytes").channel_select(index=2)
    bgr_swap = Pipeline().source("image_bytes").channel_swap(order=[2, 1, 0])

    result = df.with_columns(
        red=pl.col("image").cv.pipe(red_ch).sink("numpy"),
        green=pl.col("image").cv.pipe(green_ch).sink("numpy"),
        blue=pl.col("image").cv.pipe(blue_ch).sink("numpy"),
        bgr=pl.col("image").cv.pipe(bgr_swap).sink("numpy"),
    )

    images = [img]
    titles = ["Original"]
    for name in ["red", "green", "blue", "bgr"]:
        arr = numpy_from_struct(result[name][0])
        images.append(arr)
        label = f"Channel {name.title()}" if name != "bgr" else "Channel Swap (BGR)"
        titles.append(label)

    render_grid(images, titles, "09_channel_ops.png", cols=3)
    print("Extracted R/G/B channels and swapped to BGR")


def demo_intensity_adjustments() -> None:
    """Demonstrate contrast, gamma, brightness, and invert."""
    print("\n--- Intensity Adjustments ---")

    img = make_image()
    png = to_png_bytes(img)
    df = pl.DataFrame({"image": [png]})

    contrast_hi = Pipeline().source("image_bytes").adjust_contrast(factor=2.0)
    contrast_lo = Pipeline().source("image_bytes").adjust_contrast(factor=0.5)
    gamma_bright = Pipeline().source("image_bytes").adjust_gamma(gamma=0.4)
    gamma_dark = Pipeline().source("image_bytes").adjust_gamma(gamma=2.5)
    bright = Pipeline().source("image_bytes").adjust_brightness(factor=1.5)
    inverted = Pipeline().source("image_bytes").invert()

    result = df.with_columns(
        contrast_hi=pl.col("image").cv.pipe(contrast_hi).sink("numpy"),
        contrast_lo=pl.col("image").cv.pipe(contrast_lo).sink("numpy"),
        gamma_bright=pl.col("image").cv.pipe(gamma_bright).sink("numpy"),
        gamma_dark=pl.col("image").cv.pipe(gamma_dark).sink("numpy"),
        bright=pl.col("image").cv.pipe(bright).sink("numpy"),
        inverted=pl.col("image").cv.pipe(inverted).sink("numpy"),
    )

    images = [img]
    titles = ["Original"]
    for name, label in [
        ("contrast_hi", "Contrast x2.0"),
        ("contrast_lo", "Contrast x0.5"),
        ("gamma_bright", "Gamma 0.4 (bright)"),
        ("gamma_dark", "Gamma 2.5 (dark)"),
        ("bright", "Brightness x1.5"),
        ("inverted", "Inverted"),
    ]:
        arr = numpy_from_struct(result[name][0])
        if arr.dtype != np.uint8:
            arr = (
                np.clip(arr * 255, 0, 255).astype(np.uint8)
                if arr.max() <= 1.0
                else arr.astype(np.uint8)
            )
        images.append(arr)
        titles.append(label)

    render_grid(images, titles, "09_intensity_adjustments.png", cols=4)
    print("Applied contrast, gamma, brightness, and invert adjustments")


def demo_dynamic_intensity() -> None:
    """Demonstrate per-row dynamic intensity parameters."""
    print("\n--- Dynamic Intensity (per-row) ---")

    img = make_image()
    png = to_png_bytes(img)
    df = pl.DataFrame(
        {
            "image": [png, png, png],
            "contrast_val": [0.5, 1.0, 2.0],
        }
    )

    pipe = (
        Pipeline().source("image_bytes").adjust_contrast(factor=pl.col("contrast_val"))
    )
    result = df.with_columns(
        adjusted=pl.col("image").cv.pipe(pipe).sink("numpy"),
    )

    images = []
    titles = []
    for i, contrast in enumerate(df["contrast_val"].to_list()):
        arr = numpy_from_struct(result["adjusted"][i])
        if arr.dtype != np.uint8:
            arr = (
                np.clip(arr * 255, 0, 255).astype(np.uint8)
                if arr.max() <= 1.0
                else arr.astype(np.uint8)
            )
        images.append(arr)
        titles.append(f"Contrast={contrast}")

    render_grid(images, titles, "09_dynamic_intensity.png", cols=3)
    print("Applied per-row dynamic contrast values")


def main() -> None:
    """Run all color and channel demos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_color_conversion()
    demo_channel_operations()
    demo_intensity_adjustments()
    demo_dynamic_intensity()
    print("\nAll color & channel demos complete.")


if __name__ == "__main__":
    main()
