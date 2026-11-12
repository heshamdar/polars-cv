"""Training-time augmentation: random parameters are ordinary columns.

Demonstrates:
- Seeded per-row parameters as Polars columns, a function of (row id, epoch)
- A random-resized-crop, flip, rotation and contrast jitter in one pipeline
- The same seed giving the same batch, and another epoch another one
- A fixed-shape batch out of the `array` sink

The docs guide (docs/user-guide/augmentation.md) quotes the marked sections
of this file, and tests/test_examples.py runs it.

Run:
    uv run python polars-cv/examples/14_augmentation.py
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import polars as pl
from PIL import Image
from polars_cv import Pipeline

OUT = 64  # the side of every augmented image


def make_images(n: int = 8) -> pl.DataFrame:
    """`n` PNGs of different sizes, as a dataset has them."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        h, w = int(rng.integers(96, 160)), int(rng.integers(96, 160))
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) % 256], axis=-1)
        buf = BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
        rows.append({"id": i, "image": buf.getvalue(), "height": h, "width": w})
    return pl.DataFrame(rows)


# --8<-- [start:params]
def uniform(name: str, seed: int, epoch: int) -> pl.Expr:
    """A uniform [0, 1) value per row: a hash of (row id, epoch, name).

    The value depends on the row's id and not on its position, so it is the
    same whichever rows are in the batch and whatever order they arrive in.
    """
    key = pl.struct(id=pl.col("id"), epoch=pl.lit(epoch), name=pl.lit(name))
    return (key.hash(seed) % 1_000_000) / 1_000_000


def with_params(df: pl.DataFrame, *, seed: int, epoch: int) -> pl.DataFrame:
    """Add this epoch's augmentation parameters, one column each."""

    def u(name: str) -> pl.Expr:
        return uniform(name, seed, epoch)

    # Random-resized-crop: a window of 35-100% of the area, aspect 3/4 to 4/3,
    # clamped to the image, placed uniformly inside it.
    area = (0.35 + 0.65 * u("area")) * pl.col("height") * pl.col("width")
    aspect = (3 / 4) * (16 / 9) ** u("aspect")
    crop_w = pl.min_horizontal((area * aspect).sqrt(), pl.col("width"))
    crop_h = pl.min_horizontal((area / aspect).sqrt(), pl.col("height"))
    return df.with_columns(
        crop_h=crop_h.cast(pl.Int64), crop_w=crop_w.cast(pl.Int64)
    ).with_columns(
        top=(u("top") * (pl.col("height") - pl.col("crop_h") + 1)).cast(pl.Int64),
        left=(u("left") * (pl.col("width") - pl.col("crop_w") + 1)).cast(pl.Int64),
        flip=u("flip") < 0.5,
        angle=(u("angle") - 0.5) * 30.0,  # degrees, in [-15, 15)
        contrast=0.8 + 0.4 * u("contrast"),  # in [0.8, 1.2)
    )


# --8<-- [end:params]


# --8<-- [start:pipeline]
def augment() -> Pipeline:
    """The augmentation: every random value is read from a column."""
    return (
        Pipeline()
        .source("image_bytes", dtype="u8")
        .crop(
            top=pl.col("top"),
            left=pl.col("left"),
            height=pl.col("crop_h"),
            width=pl.col("crop_w"),
        )
        .resize(height=OUT, width=OUT, filter="bilinear")
        .flip_h(when=pl.col("flip"))
        .rotate(pl.col("angle"))
        .adjust_contrast(pl.col("contrast"))
    )


def epoch_batch(df: pl.DataFrame, *, seed: int, epoch: int) -> np.ndarray:
    """One epoch's images as a single `(n, OUT, OUT, 3)` float32 array
    (`adjust_contrast` computes in float; add `.cast("u8")` for bytes)."""
    out = with_params(df, seed=seed, epoch=epoch).select(
        x=pl.col("image").cv.pipe(augment()).sink("array", shape=[OUT, OUT, 3])
    )
    return out["x"].to_numpy()


# --8<-- [end:pipeline]


def main() -> None:
    df = make_images()

    # --8<-- [start:epochs]
    first = epoch_batch(df, seed=7, epoch=0)
    again = epoch_batch(df, seed=7, epoch=0)
    second = epoch_batch(df, seed=7, epoch=1)

    assert first.shape == (df.height, OUT, OUT, 3)
    assert np.array_equal(first, again)  # same seed and epoch: same batch
    assert not np.array_equal(first, second)  # another epoch: another batch

    # A row's augmentation does not depend on the rows around it.
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=1)
    order = shuffled["id"].to_numpy()
    assert np.array_equal(epoch_batch(shuffled, seed=7, epoch=0), first[order])
    # --8<-- [end:epochs]

    print(f"epoch 0: {first.shape} {first.dtype}, mean {first.mean():.1f}")
    print(f"epoch 1: {second.shape} {second.dtype}, mean {second.mean():.1f}")
    print("same seed and epoch reproduce; the batch order does not matter")


if __name__ == "__main__":
    main()
