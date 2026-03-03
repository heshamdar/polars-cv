"""Affine transforms for geometric image manipulation.

Demonstrates:
- warp_affine with a raw 2x3 matrix (translation, rotation)
- shear convenience method
- rotate_and_scale convenience method
- Pipeline fusion: consecutive affine ops are combined into one

Run:
    uv run python polars-cv/examples/13_affine_transforms.py
"""

from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
from polars_cv import Pipeline, numpy_from_struct

OUTPUT_DIR = Path(__file__).parent / "outputs"


def make_checkerboard(size: int = 200, block: int = 25) -> np.ndarray:
    """Create a checkerboard pattern useful for visualizing geometric warps."""
    board = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(0, size, block):
        for c in range(0, size, block):
            if ((r // block) + (c // block)) % 2 == 0:
                board[r : r + block, c : c + block] = [200, 200, 200]
            else:
                board[r : r + block, c : c + block] = [80, 80, 80]
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    mask = (x - cx) ** 2 + (y - cy) ** 2 < (size // 5) ** 2
    board[mask] = [220, 60, 60]
    return board


def encode_png(arr: np.ndarray) -> bytes:
    """Encode a numpy array to PNG bytes."""
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    """Run all affine transform demonstrations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img = make_checkerboard()
    h, w = img.shape[:2]

    df = pl.DataFrame({"image": [encode_png(img)]})

    # --- 1. Translation ---
    tx, ty = 30.0, 20.0
    pipe_translate = (
        Pipeline()
        .source("image_bytes")
        .warp_affine(
            matrix=[1, 0, tx, 0, 1, ty],
            output_size=(h, w),
        )
    )
    res = df.with_columns(out=pl.col("image").cv.pipe(pipe_translate).sink("numpy"))
    translated = numpy_from_struct(res["out"][0])

    # --- 2. 45-degree rotation around center ---
    angle_rad = math.radians(45)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    cx, cy = w / 2.0, h / 2.0
    rot_matrix = [
        cos_a,
        -sin_a,
        (1 - cos_a) * cx + sin_a * cy,
        sin_a,
        cos_a,
        -sin_a * cx + (1 - cos_a) * cy,
    ]
    pipe_rotate = (
        Pipeline()
        .source("image_bytes")
        .warp_affine(
            matrix=rot_matrix,
            output_size=(h, w),
            interpolation="bilinear",
        )
    )
    res = df.with_columns(out=pl.col("image").cv.pipe(pipe_rotate).sink("numpy"))
    rotated = numpy_from_struct(res["out"][0])

    # --- 3. Shear ---
    pipe_shear = Pipeline().source("image_bytes").shear(sx=0.3, output_size=(h, w))
    res = df.with_columns(out=pl.col("image").cv.pipe(pipe_shear).sink("numpy"))
    sheared = numpy_from_struct(res["out"][0])

    # --- 4. Rotate-and-scale convenience ---
    pipe_rs = (
        Pipeline()
        .source("image_bytes")
        .rotate_and_scale(
            angle=30,
            scale=0.8,
            center=(cx, cy),
            output_size=(h, w),
        )
    )
    res = df.with_columns(out=pl.col("image").cv.pipe(pipe_rs).sink("numpy"))
    rot_scaled = numpy_from_struct(res["out"][0])

    # --- 5. Pipeline fusion (two affines → one kernel call) ---
    pipe_fused = (
        Pipeline()
        .source("image_bytes")
        .warp_affine(matrix=[1, 0, 50, 0, 1, 0], output_size=(h, w))
        .warp_affine(matrix=[1, 0, 0, 0, 1, 30], output_size=(h, w))
    )
    spec = pipe_fused._to_spec_dict()
    affine_count = sum(1 for op in spec["ops"] if op["op"] == "warp_affine")
    print(f"Two warp_affine calls fused into {affine_count} op(s)")

    res = df.with_columns(out=pl.col("image").cv.pipe(pipe_fused).sink("numpy"))
    fused = numpy_from_struct(res["out"][0])

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    titles = [
        "Original",
        f"Translate ({tx}, {ty})",
        "Rotate 45°",
        "Shear (sx=0.3)",
        "Rotate 30° + Scale 0.8",
        "Fused translate (50,30)",
    ]
    images = [img, translated, rotated, sheared, rot_scaled, fused]
    for ax, title, im in zip(axes.flat, titles, images, strict=False):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("polars-cv: Affine Transforms", fontsize=16, y=0.98)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "affine_transforms.png", dpi=120)
    plt.close(fig)
    print(f"Saved → {OUTPUT_DIR / 'affine_transforms.png'}")


if __name__ == "__main__":
    main()
