"""Image transform operations in polars-cv.

Run:
    uv run python polars-cv/examples/02_image_transforms.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
from polars_cv import IMAGENET_MEAN, IMAGENET_STD, Pipeline, numpy_from_struct

OUTPUT_DIR = Path(__file__).parent / "outputs"


def make_image(seed: int, size: int = 128) -> np.ndarray:
    """Create a synthetic RGB image with gradients and noise."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size))
    r = (x * 2) % 255
    g = (y * 2) % 255
    b = ((x + y) * 3) % 255
    noise = rng.integers(0, 20, size=(size, size, 3), dtype=np.uint8)
    return np.clip(np.stack([r, g, b], axis=2) + noise, 0, 255).astype(np.uint8)


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB image to PNG bytes."""
    buf = BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def render_grid(images: list[np.ndarray], titles: list[str], output_name: str) -> None:
    """Render a gallery for transformed images."""
    cols = 3
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for idx, (img, title) in enumerate(zip(images, titles, strict=True)):
        ax = axes[idx // cols][idx % cols]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for idx in range(len(images), rows * cols):
        axes[idx // cols][idx % cols].axis("off")
    fig.tight_layout()
    path = OUTPUT_DIR / output_name
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("Saved:", path)


def decode_column_numpy(df: pl.DataFrame, col: str) -> list[np.ndarray]:
    """Decode a `sink('numpy')` output column to arrays."""
    return [numpy_from_struct(row, copy=False) for row in df[col].to_list()]


def run_static_transform_showcase(df: pl.DataFrame) -> None:
    """Demonstrate common static transform operations."""
    expr = pl.col("image").cv.pipe(Pipeline().source("image_bytes"))

    transformed = df.with_columns(
        original=expr.sink("numpy"),
        resize=expr.pipe(
            Pipeline().resize(height=96, width=96, filter="bilinear")
        ).sink("numpy"),
        crop=expr.pipe(Pipeline().crop(top=16, left=16, width=80, height=80)).sink(
            "numpy"
        ),
        rotate=expr.pipe(Pipeline().rotate(angle=20.0)).sink("numpy"),
        flip_h=expr.pipe(Pipeline().flip_h()).sink("numpy"),
        blur=expr.pipe(Pipeline().blur(sigma=1.4)).sink("numpy"),
        threshold=expr.pipe(Pipeline().grayscale().threshold(value=120)).sink("numpy"),
        pad_reflect=expr.pipe(
            Pipeline().pad(left=12, right=12, top=8, bottom=8, mode="reflect")
        ).sink("numpy"),
        letterbox=expr.pipe(Pipeline().letterbox(height=160, width=200)).sink("numpy"),
    )

    cols = [
        "original",
        "resize",
        "crop",
        "rotate",
        "flip_h",
        "blur",
        "threshold",
        "pad_reflect",
        "letterbox",
    ]
    arrays = [decode_column_numpy(transformed, c)[0] for c in cols]
    render_grid(arrays, cols, "02_transforms_static.png")


def run_value_transform_showcase(df: pl.DataFrame) -> None:
    """Demonstrate value-domain transforms and normalization methods."""
    expr = pl.col("image").cv.pipe(Pipeline().source("image_bytes"))

    transformed = df.with_columns(
        minmax=expr.pipe(Pipeline().normalize(method="minmax")).sink("numpy"),
        zscore=expr.pipe(Pipeline().normalize(method="zscore")).sink("numpy"),
        imagenet=expr.pipe(
            Pipeline().normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ).sink("numpy"),
        scaled=expr.pipe(Pipeline().cast("f32").scale(factor=0.2)).sink("numpy"),
        clamped=expr.pipe(
            Pipeline().cast("f32").scale(factor=0.2).clamp(min_val=0.0, max_val=30.0)
        ).sink("numpy"),
        relu=expr.pipe(Pipeline().cast("f32").scale(factor=-1.0).relu()).sink("numpy"),
    )

    for col in transformed.columns:
        if col == "image":
            continue
        arr = decode_column_numpy(transformed, col)[0]
        print(
            f"{col:>10} -> shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.3f}, max={arr.max():.3f}"
        )


def run_dynamic_parameter_demo(df: pl.DataFrame) -> None:
    """Demonstrate expression-driven resize and crop parameters."""
    dynamic_df = df.with_columns(
        target_h=pl.lit(72),
        target_w=pl.lit(104),
        crop_x=pl.lit(10),
        crop_y=pl.lit(14),
        crop_w=pl.lit(90),
        crop_h=pl.lit(76),
    )

    dynamic_pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=pl.col("target_h"), width=pl.col("target_w"))
        .crop(
            top=pl.col("crop_y"),
            left=pl.col("crop_x"),
            width=pl.col("crop_w"),
            height=pl.col("crop_h"),
        )
    )
    out = dynamic_df.with_columns(
        dynamic=pl.col("image").cv.pipe(dynamic_pipe).sink("numpy")
    )
    arr = decode_column_numpy(out, "dynamic")[0]
    print("\nDynamic parameter output shape:", arr.shape)


def main() -> None:
    """Run the image transform demos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = make_image(seed=42)
    df = pl.DataFrame({"image": [to_png_bytes(base)]})

    run_static_transform_showcase(df)
    run_value_transform_showcase(df)
    run_dynamic_parameter_demo(df)


if __name__ == "__main__":
    main()
