"""Morphological operations for binary mask processing.

Demonstrates:
- erode / dilate (single and multi-iteration)
- morphology_open / morphology_close
- morphology_gradient (edge outline)
- Typical segmentation post-processing workflow

Run:
    uv run python polars-cv/examples/12_morphological_operations.py
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


def make_binary_mask(size: int = 160) -> np.ndarray:
    """Create a noisy binary mask with blobs and small-scale artifacts."""
    mask = np.zeros((size, size), dtype=np.uint8)

    center = size // 2
    y, x = np.ogrid[:size, :size]
    circle = (x - center) ** 2 + (y - center) ** 2 < (size // 4) ** 2
    mask[circle] = 255

    quarter = size // 4
    mask[
        quarter : quarter + size // 5,
        quarter + size // 2 : quarter + size // 2 + size // 5,
    ] = 255

    rng = np.random.default_rng(42)
    salt = rng.random((size, size)) < 0.02
    pepper = rng.random((size, size)) < 0.02
    mask[salt] = 255
    mask[pepper] = 0

    return mask


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a grayscale/binary image to PNG bytes."""
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
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for idx in range(len(images), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("Saved:", path)


def extract_result(col: pl.Series) -> np.ndarray:
    """Extract a 2D numpy array from a polars-cv numpy-sink result."""
    arr = numpy_from_struct(col[0])
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def demo_erode_dilate() -> None:
    """Demonstrate basic erosion and dilation."""
    print("\n--- Erode & Dilate ---")

    mask = make_binary_mask()
    png = to_png_bytes(mask)
    df = pl.DataFrame({"image": [png]})

    erode_3 = Pipeline().source("image_bytes").grayscale().erode(ksize=3)
    erode_5 = Pipeline().source("image_bytes").grayscale().erode(ksize=5)
    dilate_3 = Pipeline().source("image_bytes").grayscale().dilate(ksize=3)
    dilate_5 = Pipeline().source("image_bytes").grayscale().dilate(ksize=5)

    result = df.with_columns(
        erode_3=pl.col("image").cv.pipe(erode_3).sink("numpy"),
        erode_5=pl.col("image").cv.pipe(erode_5).sink("numpy"),
        dilate_3=pl.col("image").cv.pipe(dilate_3).sink("numpy"),
        dilate_5=pl.col("image").cv.pipe(dilate_5).sink("numpy"),
    )

    images = [
        mask,
        extract_result(result["erode_3"]),
        extract_result(result["erode_5"]),
        extract_result(result["dilate_3"]),
        extract_result(result["dilate_5"]),
    ]
    titles = ["Original", "Erode 3×3", "Erode 5×5", "Dilate 3×3", "Dilate 5×5"]

    render_grid(images, titles, "12_erode_dilate.png", cols=3)
    print("Applied erosion and dilation with different kernel sizes")


def demo_open_close() -> None:
    """Demonstrate morphological opening and closing."""
    print("\n--- Opening & Closing ---")

    mask = make_binary_mask()
    png = to_png_bytes(mask)
    df = pl.DataFrame({"image": [png]})

    opened = Pipeline().source("image_bytes").grayscale().morphology_open(ksize=3)
    closed = Pipeline().source("image_bytes").grayscale().morphology_close(ksize=3)
    clean = (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .morphology_open(ksize=3)
        .morphology_close(ksize=3)
    )

    result = df.with_columns(
        opened=pl.col("image").cv.pipe(opened).sink("numpy"),
        closed=pl.col("image").cv.pipe(closed).sink("numpy"),
        clean=pl.col("image").cv.pipe(clean).sink("numpy"),
    )

    images = [
        mask,
        extract_result(result["opened"]),
        extract_result(result["closed"]),
        extract_result(result["clean"]),
    ]
    titles = [
        "Original (noisy)",
        "Open (remove salt)",
        "Close (fill pepper)",
        "Open → Close (clean)",
    ]

    render_grid(images, titles, "12_open_close.png", cols=4)
    print("Applied opening, closing, and combined cleaning")


def demo_gradient() -> None:
    """Demonstrate morphological gradient (edge outline)."""
    print("\n--- Morphological Gradient ---")

    mask = make_binary_mask()
    clean_mask = np.zeros_like(mask)
    center = mask.shape[0] // 2
    y, x = np.ogrid[: mask.shape[0], : mask.shape[1]]
    circle = (x - center) ** 2 + (y - center) ** 2 < (mask.shape[0] // 4) ** 2
    clean_mask[circle] = 255

    png = to_png_bytes(clean_mask)
    df = pl.DataFrame({"image": [png]})

    gradient_3 = (
        Pipeline().source("image_bytes").grayscale().morphology_gradient(ksize=3)
    )
    gradient_5 = (
        Pipeline().source("image_bytes").grayscale().morphology_gradient(ksize=5)
    )

    result = df.with_columns(
        grad_3=pl.col("image").cv.pipe(gradient_3).sink("numpy"),
        grad_5=pl.col("image").cv.pipe(gradient_5).sink("numpy"),
    )

    images = [
        clean_mask,
        extract_result(result["grad_3"]),
        extract_result(result["grad_5"]),
    ]
    titles = ["Binary Mask", "Gradient 3×3", "Gradient 5×5"]

    render_grid(images, titles, "12_gradient.png", cols=3)
    print("Applied morphological gradient for edge outlines")


def demo_segmentation_workflow() -> None:
    """Demonstrate a typical segmentation post-processing pipeline."""
    print("\n--- Segmentation Post-Processing ---")

    size = 160
    rng = np.random.default_rng(99)
    raw = np.clip(rng.normal(loc=100, scale=60, size=(size, size)), 0, 255).astype(
        np.uint8
    )
    y, x = np.ogrid[:size, :size]
    center = size // 2
    signal = (x - center) ** 2 + (y - center) ** 2 < (size // 3) ** 2
    raw[signal] = np.clip(
        raw[signal].astype(np.int16) + 100,
        0,
        255,
    ).astype(np.uint8)

    png = to_png_bytes(raw)
    df = pl.DataFrame({"image": [png]})

    step_threshold = (
        Pipeline().source("image_bytes").grayscale().blur(sigma=1.0).threshold(150)
    )
    step_clean = (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .blur(sigma=1.0)
        .threshold(150)
        .morphology_open(ksize=3)
        .morphology_close(ksize=3)
    )
    step_outline = (
        Pipeline()
        .source("image_bytes")
        .grayscale()
        .blur(sigma=1.0)
        .threshold(150)
        .morphology_open(ksize=3)
        .morphology_close(ksize=3)
        .morphology_gradient(ksize=3)
    )

    result = df.with_columns(
        thresholded=pl.col("image").cv.pipe(step_threshold).sink("numpy"),
        cleaned=pl.col("image").cv.pipe(step_clean).sink("numpy"),
        outlined=pl.col("image").cv.pipe(step_outline).sink("numpy"),
    )

    images = [
        raw,
        extract_result(result["thresholded"]),
        extract_result(result["cleaned"]),
        extract_result(result["outlined"]),
    ]
    titles = [
        "Raw Image",
        "Threshold",
        "Open + Close",
        "Gradient (outline)",
    ]

    render_grid(images, titles, "12_segmentation_workflow.png", cols=4)
    print(
        "Demonstrated segmentation post-processing: threshold → open → close → gradient"
    )


def main() -> None:
    """Run all morphological operation demos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_erode_dilate()
    demo_open_close()
    demo_gradient()
    demo_segmentation_workflow()
    print("\nAll morphological operation demos complete.")


if __name__ == "__main__":
    main()
