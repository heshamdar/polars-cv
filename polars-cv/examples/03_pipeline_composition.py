"""Pipeline composition patterns: branching, merging, and binary ops.

Run:
    uv run python polars-cv/examples/03_pipeline_composition.py
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


def make_rgb(seed: int, size: int = 96) -> np.ndarray:
    """Create a deterministic synthetic RGB image."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size))
    img = np.stack(
        [
            ((x * 3) % 255).astype(np.uint8),
            ((y * 4) % 255).astype(np.uint8),
            (((x + y) * 2) % 255).astype(np.uint8),
        ],
        axis=2,
    )
    img ^= rng.integers(0, 16, size=img.shape, dtype=np.uint8)
    return img


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode a uint8 RGB image as PNG bytes."""
    buf = BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_contour(width: int, height: int) -> dict[str, object]:
    """Create a rectangular contour payload compatible with `source('contour')`."""
    return {
        "exterior": [
            {"x": 8.0, "y": 8.0},
            {"x": float(width - 8), "y": 8.0},
            {"x": float(width - 8), "y": float(height - 8)},
            {"x": 8.0, "y": float(height - 8)},
        ],
        "holes": [],
        "is_closed": True,
    }


def decode_struct_col(df: pl.DataFrame, col: str) -> np.ndarray:
    """Decode the first row of a numpy sink struct column."""
    return numpy_from_struct(df[col][0], copy=False)


def save_gallery(images: list[np.ndarray], titles: list[str], name: str) -> None:
    """Save a 2-row image gallery."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), squeeze=False)
    for idx, (img, title) in enumerate(zip(images, titles, strict=True)):
        ax = axes[idx // 4][idx % 4]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    out = OUTPUT_DIR / name
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Saved:", out)


def main() -> None:
    """Run pipeline composition demos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_a = make_rgb(seed=3)
    base_b = np.rot90(make_rgb(seed=7), k=1).copy()
    contour = make_contour(width=96, height=96)

    df = pl.DataFrame(
        {
            "image_a": [to_png_bytes(base_a)],
            "image_b": [to_png_bytes(base_b)],
            "roi_contour": [contour],
        }
    )

    left = pl.col("image_a").cv.pipe(Pipeline().source("image_bytes")).alias("left")
    right = pl.col("image_b").cv.pipe(Pipeline().source("image_bytes")).alias("right")

    left_gray = left.pipe(Pipeline().grayscale()).alias("left_gray")
    left_blur = left_gray.pipe(Pipeline().blur(sigma=1.2)).alias("left_blur")
    right_gray = right.pipe(Pipeline().grayscale()).alias("right_gray")

    # Binary operations between branches.
    add_img = left_gray.add(right_gray).alias("add")
    sub_img = left_gray.subtract(right_gray).alias("subtract")
    blend_img = left_gray.blend(right_gray).alias("blend")
    max_img = left_gray.maximum(right_gray).alias("maximum")
    xor_img = left_gray.bitwise_xor(right_gray).alias("xor")

    # Contour masking against the left image branch.
    contour_expr = (
        pl.col("roi_contour")
        .cv.pipe(
            Pipeline().source("contour", shape=left, fill_value=255, background=0),
        )
        .alias("roi_mask")
    )
    masked = left.apply_contour_mask(contour_expr, invert=False).alias("masked")

    merged = add_img.merge_pipe(sub_img, blend_img, max_img, xor_img, left_blur, masked)
    out = df.with_columns(
        outputs=merged.sink(
            {
                "add": "numpy",
                "subtract": "numpy",
                "blend": "numpy",
                "maximum": "numpy",
                "xor": "numpy",
                "left_blur": "numpy",
                "masked": "numpy",
            }
        )
    )

    # Unpack the multi-output struct for convenience.
    unpacked = out.with_columns(
        add=pl.col("outputs").struct.field("add"),
        subtract=pl.col("outputs").struct.field("subtract"),
        blend=pl.col("outputs").struct.field("blend"),
        maximum=pl.col("outputs").struct.field("maximum"),
        xor=pl.col("outputs").struct.field("xor"),
        left_blur=pl.col("outputs").struct.field("left_blur"),
        masked=pl.col("outputs").struct.field("masked"),
    )
    print(unpacked.select("add", "subtract", "blend").schema)

    images = [
        decode_struct_col(unpacked, "add"),
        decode_struct_col(unpacked, "subtract"),
        decode_struct_col(unpacked, "blend"),
        decode_struct_col(unpacked, "maximum"),
        decode_struct_col(unpacked, "xor"),
        decode_struct_col(unpacked, "left_blur"),
        decode_struct_col(unpacked, "masked"),
        np.asarray(Image.open(BytesIO(df["image_a"][0]))),
    ]
    titles = [
        "add",
        "subtract",
        "blend",
        "maximum",
        "xor",
        "left_blur",
        "masked",
        "original_a",
    ]
    save_gallery(images, titles, "03_pipeline_composition_gallery.png")


if __name__ == "__main__":
    main()
