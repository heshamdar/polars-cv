"""Patches of a whole-slide image: find the tissue, read only its patches.

Demonstrates:
- .cv.slide_info() to read a pyramidal TIFF's levels from its header
- source(level=) to decode a small level for a tissue mask
- polars_cv.patch_grid() + explode: one row per patch
- crop right after a file_path source, which reads only the patch's tiles
- a per-patch heatmap with a plain Polars pivot
- Pipeline.tile() to cut an in-memory image into patches at once

Run:
    uv run python polars-cv/examples/14_whole_slide_patches.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import tifffile

import polars_cv as cv
from polars_cv import Pipeline, numpy_from_struct

PATCH = 64


def make_slide(path: Path) -> None:
    """A 1024x1536 RGB "slide": a pink tissue blob on white, saved as a
    JPEG-tiled 3-level pyramid with an Aperio-style description."""
    h, w = 1024, 1536
    yy, xx = np.mgrid[0:h, 0:w]
    blob = ((yy - 500) / 330) ** 2 + ((xx - 700) / 520) ** 2 < 1.0
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    tissue = np.array([200, 120, 170]) + rng.integers(-30, 30, (h, w, 3))
    img[blob] = np.clip(tissue[blob], 0, 255).astype(np.uint8)
    description = "Aperio Image Library v12.0.0\r\nsynthetic|AppMag = 20|MPP = 0.5"
    with tifffile.TiffWriter(path) as tw:
        for k in range(3):
            tw.write(
                img[:: 2**k, :: 2**k],
                tile=(256, 256),
                compression="jpeg",
                photometric="rgb",
                subfiletype=1 if k else 0,
                description=description if k == 0 else None,
                metadata=None,
            )


def demo_slide_info(slides: pl.DataFrame) -> pl.DataFrame:
    """The pyramid, from the header alone."""
    print("\n--- slide_info ---")
    info = slides.with_columns(info=pl.col("path").cv.slide_info())
    levels = info.select(pl.col("info").struct.field("levels")).explode("levels").unnest("levels")
    print(levels)
    print("microns per pixel:", info["info"].struct.field("mpp_x")[0])
    return levels


def demo_tissue_patches(slides: pl.DataFrame, levels: pl.DataFrame) -> pl.DataFrame:
    """Find tissue on the smallest level, then read its level-0 patches."""
    print("\n--- tissue patches ---")
    low = levels.row(-1, named=True)
    scale = int(low["downsample"])
    # The grid at level 0, and each cell's window on the small level.
    cells = (
        slides.with_columns(
            cell=cv.patch_grid(levels["height"][0], levels["width"][0], size=PATCH)
        )
        .explode("cell")
        .unnest("cell")
        .with_columns(
            low_top=pl.col("top") // scale,
            low_left=pl.col("left") // scale,
        )
    )
    # Tissue fraction of each cell: dark-ish pixels of the small level.
    side = PATCH // scale
    tissue = (
        Pipeline()
        .source("file_path", level=low["level"])
        .crop(top=pl.col("low_top"), left=pl.col("low_left"), height=side, width=side)
        .grayscale()
        .threshold(220)
        .invert()
        .reduce_mean()
    )
    cells = cells.with_columns(tissue=pl.col("path").cv.pipe(tissue).sink("native") / 255)
    keep = cells.filter(pl.col("tissue") > 0.5)
    print(f"{keep.height} of {cells.height} patches hold tissue")

    # Only those patches are read at full resolution: each crop decodes the
    # tiles under it, not the slide.
    patch = (
        Pipeline()
        .source("file_path")
        .crop(top=pl.col("top"), left=pl.col("left"), height=PATCH, width=PATCH)
    )
    keep = keep.with_columns(patch=pl.col("path").cv.pipe(patch).sink("numpy"))
    batch = np.stack([numpy_from_struct(p) for p in keep["patch"]])
    print("batch for a model:", batch.shape, batch.dtype)
    return cells


def demo_heatmap(cells: pl.DataFrame) -> None:
    """Per-patch scores back on the slide's grid: a plain pivot."""
    print("\n--- heatmap ---")
    heat = cells.pivot(on="col", index="row", values="tissue", sort_columns=True).sort("row")
    grid = heat.drop("row").to_numpy()
    for line in grid[::2]:
        print("".join(" .:-=+*#%@"[min(int(v * 9.99), 9)] for v in line[::2]))


def demo_tile() -> None:
    """An image already in memory: cut it into patches in one step."""
    print("\n--- tile ---")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    df = pl.DataFrame({"img": [img.tobytes()]})
    pipe = (
        Pipeline()
        .source("raw", dtype="u8")
        .reshape(shape=[200, 300, 3])
        .tile(height=PATCH, width=PATCH, edge="shift")
    )
    tiles = numpy_from_struct(df.select(pl.col("img").cv.pipe(pipe).sink("numpy"))["img"][0])
    print("patches:", tiles.shape)
    assert (tiles[0] == img[:PATCH, :PATCH]).all()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "slide.svs"
        make_slide(path)
        slides = pl.DataFrame({"path": [str(path)]})
        levels = demo_slide_info(slides)
        cells = demo_tissue_patches(slides, levels)
        demo_heatmap(cells)
    demo_tile()


if __name__ == "__main__":
    main()
