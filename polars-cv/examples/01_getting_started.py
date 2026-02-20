"""Getting started with polars-cv pipelines.

Run:
    uv run python polars-cv/examples/01_getting_started.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
from polars_cv import Pipeline, numpy_from_struct

OUTPUT_DIR = Path(__file__).parent / "outputs"


def make_test_image(size: int, seed: int) -> np.ndarray:
    """Create a synthetic RGB test image."""
    rng = np.random.default_rng(seed)
    y, x = np.indices((size, size))
    base = ((x + y) % 255).astype(np.uint8)
    noise = rng.integers(0, 40, size=(size, size), dtype=np.uint8)
    img = np.stack([base, np.roll(base, 8, axis=1), np.roll(base, 16, axis=0)], axis=2)
    return np.clip(img + noise[..., None], 0, 255).astype(np.uint8)


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an RGB uint8 image as PNG bytes."""
    buf = BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def save_comparison_grid(
    originals: list[np.ndarray],
    processed: list[np.ndarray],
    output_path: Path,
) -> None:
    """Save a side-by-side visualization of original and processed images."""
    n = len(originals)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(8, 3 * n), squeeze=False)
    for idx, (orig, proc) in enumerate(zip(originals, processed, strict=True)):
        axes[idx, 0].imshow(orig)
        axes[idx, 0].set_title(f"Original #{idx}")
        axes[idx, 0].axis("off")

        if proc.ndim == 2:
            axes[idx, 1].imshow(proc, cmap="gray")
        else:
            axes[idx, 1].imshow(proc)
        axes[idx, 1].set_title(f"Processed #{idx}")
        axes[idx, 1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run_getting_started_demo() -> pl.DataFrame:
    """Build a simple preprocessing pipeline and execute it on a batch."""
    images = [make_test_image(size=96, seed=seed) for seed in (0, 1, 2)]
    df = pl.DataFrame(
        {
            "image_id": ["img-0", "img-1", "img-2"],
            "image": [image_to_png_bytes(img) for img in images],
        }
    )

    preprocess = (
        Pipeline()
        .source("image_bytes")
        .resize(height=128, width=128, filter="bilinear")
        .grayscale()
    )

    result = df.with_columns(
        processed=pl.col("image").cv.pipe(preprocess).sink("numpy"),
    )
    return result


def decode_numpy_structs(frame: pl.DataFrame) -> list[np.ndarray]:
    """Convert `sink('numpy')` struct rows to NumPy arrays."""
    rows: list[dict[str, Any]] = frame["processed"].to_list()
    return [numpy_from_struct(row, copy=False) for row in rows]


def main() -> None:
    """Run the getting-started example end to end."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result = run_getting_started_demo()
    processed_arrays = decode_numpy_structs(result)

    print("Schema:")
    print(result.schema)
    print("\nSample row:")
    print(result.select("image_id", "processed").head(1))
    print("\nProcessed array shapes:", [arr.shape for arr in processed_arrays])

    # Recover original images for display.
    originals = [
        np.asarray(Image.open(BytesIO(raw)).convert("RGB")) for raw in result["image"]
    ]
    save_comparison_grid(
        originals=originals,
        processed=processed_arrays,
        output_path=OUTPUT_DIR / "01_getting_started_comparison.png",
    )
    print("\nSaved output image:", OUTPUT_DIR / "01_getting_started_comparison.png")


if __name__ == "__main__":
    main()
