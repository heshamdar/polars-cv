"""Perceptual hashing and similarity workflows.

Run:
    uv run python polars-cv/examples/07_perceptual_hashing.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image
from polars_cv import HashAlgorithm, Pipeline, hamming_distance, hash_similarity

OUTPUT_DIR = Path(__file__).parent / "outputs"


def make_base_image(size: int = 96) -> np.ndarray:
    """Create a base image with geometric structure."""
    y, x = np.indices((size, size))
    img = np.stack([x % 255, y % 255, ((x + y) // 2) % 255], axis=2).astype(np.uint8)
    img[18:64, 20:68, :] = np.array([230, 40, 60], dtype=np.uint8)
    return img


def to_png_bytes(image: np.ndarray) -> bytes:
    """Encode RGB image as PNG bytes."""
    buf = BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def make_variants(base: np.ndarray) -> dict[str, np.ndarray]:
    """Create near-duplicate and dissimilar variants."""
    brighter = np.clip(base.astype(np.int16) + 18, 0, 255).astype(np.uint8)
    shifted = np.roll(base, shift=4, axis=1)
    noisy = np.clip(
        base.astype(np.int16) + np.random.default_rng(10).integers(-10, 10, base.shape),
        0,
        255,
    ).astype(np.uint8)
    random = np.random.default_rng(2).integers(0, 255, size=base.shape, dtype=np.uint8)
    return {
        "base": base,
        "brighter": brighter,
        "shifted": shifted,
        "noisy": noisy,
        "random": random,
    }


def algorithm_demo(df: pl.DataFrame) -> None:
    """Compare all hash algorithms on the same image pair."""
    algorithms = [
        HashAlgorithm.AVERAGE,
        HashAlgorithm.DIFFERENCE,
        HashAlgorithm.PERCEPTUAL,
        HashAlgorithm.BLOCKHASH,
    ]

    print("Algorithm comparison (base vs brighter):")
    for algo in algorithms:
        left = pl.col("base").cv.pipe(
            Pipeline().source("image_bytes").perceptual_hash(algorithm=algo)
        )
        right = pl.col("brighter").cv.pipe(
            Pipeline().source("image_bytes").perceptual_hash(algorithm=algo)
        )
        out = df.select(
            distance=hamming_distance(left, right),
            similarity=hash_similarity(left, right, hash_bits=64),
            popcount=left.pipe(Pipeline().reduce_popcount()).sink("native"),
        )
        print(
            f"{algo.value:>10} -> distance={out['distance'][0]:.1f}, similarity={out['similarity'][0]:.2f}, hash_popcount={out['popcount'][0]:.1f}"
        )


def near_duplicate_table(df: pl.DataFrame) -> None:
    """Build a pairwise near-duplicate table with one selected algorithm."""
    pairs = [
        ("base", "brighter"),
        ("base", "shifted"),
        ("base", "noisy"),
        ("base", "random"),
    ]
    rows: list[dict[str, object]] = []
    for left_col, right_col in pairs:
        left = pl.col(left_col).cv.pipe(
            Pipeline()
            .source("image_bytes")
            .perceptual_hash(algorithm=HashAlgorithm.PERCEPTUAL),
        )
        right = pl.col(right_col).cv.pipe(
            Pipeline()
            .source("image_bytes")
            .perceptual_hash(algorithm=HashAlgorithm.PERCEPTUAL),
        )
        score_df = df.select(
            pair=pl.lit(f"{left_col} vs {right_col}"),
            hamming=hamming_distance(left, right),
            similarity=hash_similarity(left, right, hash_bits=64),
        )
        rows.append(score_df.to_dicts()[0])

    table = pl.DataFrame(rows).sort("similarity", descending=True)
    print("\nNear-duplicate ranking:")
    print(table)


def main() -> None:
    """Run perceptual hash demonstrations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = make_variants(make_base_image())
    df = pl.DataFrame({name: [to_png_bytes(img)] for name, img in variants.items()})
    algorithm_demo(df)
    near_duplicate_table(df)


if __name__ == "__main__":
    main()
