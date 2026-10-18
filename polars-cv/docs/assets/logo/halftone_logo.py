"""A polar bear in polar coordinates, printed as a halftone.

An alternative polars-cv logo, made from polars and polars-cv alone:

1. **The geometry is a polars query.** Every part of the bear's face is one
   row of ``PARTS``, a polar curve ``r(theta)``: the head is a circle widened
   by a ``cos 2θ`` term, with an ear grown out of it on each side of the crown
   (a semicircular bump in ``r``); the nose is a ``cos 3θ`` rounded triangle.
   ``outlines`` samples each curve with ``int_ranges`` + ``explode``,
   evaluates it with expressions, and ``group_by`` + ``implode`` gather the
   points into contour sets: no shape is computed point by point in Python.
2. **The shading is a field over pixels.** The head's signed distance (in
   halftone cells) sets a dot radius that grows inward and fades to nothing
   outside. ``resize`` down to the cell grid and back up with ``nearest``
   samples it once per cell; ``coordinate_grid`` minus its own ``floor``
   gives each pixel's offset in its cell, and a dot is wherever that
   offset is shorter than the cell's radius. The silhouette is never
   drawn: it emerges from dot size alone.
3. **Colour is an index map.** The muzzle's dots take a second tint, the
   eyes and nose are inked over the dots, and ``apply_lut`` colours the
   indices with an RGBA palette whose background is transparent; a lanczos
   ``resize`` from the working resolution anti-aliases. The face is tinted,
   not white, and every letter sits on an ink outline, so the logo reads on
   a light page as well as a dark one.

    cd polars-cv
    uv run --no-sync python docs/assets/logo/halftone_logo.py

Writes ``polars-cv-halftone.png`` (the mark beside the wordmark of
``wordmark.py``) and ``polars-cv-halftone-mark.png`` (the square mark).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import polars as pl

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SCHEMA, CONTOUR_SET_SCHEMA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wordmark import SPACING, STROKE, wordmark  # noqa: E402

N = 360  # vertices per curve
EAR_WIDTH = 0.33  # half-width of an ear's bump, in radians
WORK = 4  # the graph draws at this multiple, then downsamples (anti-aliasing)
CELLS = 40  # halftone cells across the mark
MARGIN = 0.04  # around the face, as a fraction of the mark's box

# r(θ) = r · (1 + k2·cos 2θ + k3·cos 3(θ + π/2) + ear·(bump(π/2 − φ) + bump(π/2 + φ)))
# drawn at (cx + sx·r·cos θ, cy − sy·r·sin θ). Units: fractions of the mark's box.
PARTS = pl.DataFrame(
    [
        # layer,     cx,    cy,     r,     sx,   sy,   k2,   k3,   ear,  φ (deg)
        ("head", 0.50, 0.57, 0.30, 1.00, 0.88, 0.06, 0.00, 0.40, 42.0),
        ("muzzle", 0.50, 0.68, 0.13, 1.30, 0.85, 0.00, 0.00, 0.00, 0.0),
        ("feature", 0.50, 0.645, 0.062, 1.30, 1.00, 0.00, 0.16, 0.00, 0.0),  # nose
        ("feature", 0.40, 0.525, 0.032, 1.00, 1.00, 0.00, 0.00, 0.00, 0.0),  # eye
        ("feature", 0.60, 0.525, 0.032, 1.00, 1.00, 0.00, 0.00, 0.00, 0.0),  # eye
    ],
    schema=["layer", "cx", "cy", "r", "sx", "sy", "k2", "k3", "ear", "phi"],
    orient="row",
)

# Palette indices: 0 background, then the dots by region, then the ink (eyes,
# nose and the letters' outline), then the wordmark over it.
HALO, FACE, MUZZLE, INK, TEXT, ACCENT = 1, 2, 3, 4, 5, 6
INK_RGB = [10, 22, 40]
PALETTE = [  # RGBA
    [*INK_RGB, 0],  # background: transparent (ink-coloured, so edges fringe dark)
    [61, 115, 184, 255],  # dots outside the head
    [205, 226, 248, 255],  # dots on the face: tinted, to show on white
    [140, 186, 235, 255],  # dots on the muzzle
    [*INK_RGB, 255],  # eyes, nose and the letters' outline
    [241, 248, 255, 255],  # "polars"
    [127, 182, 236, 255],  # "-cv"
]
OUTLINE = 3.0  # the ink around each letter, in pixels at 1x

LOCKUP_WIDTH, LOCKUP_HEIGHT = 990, 220  # the width a whole number of cells
TEXT_X, BASELINE = 245.0, 160.0


def outlines(origin: tuple[float, float], size: float) -> pl.DataFrame:
    """One row: a contour-set column per layer of ``PARTS``, the mark's box
    ``size`` wide with its top-left corner at ``origin``."""
    t = pl.col("i") * (2 * math.pi / N)
    phi = pl.col("phi").radians()

    def bump(centre: pl.Expr) -> pl.Expr:
        return (1 - ((t - centre) / EAR_WIDTH).clip(-1, 1) ** 2).sqrt()

    r = pl.col("r") * (
        1
        + pl.col("k2") * (2 * t).cos()
        + pl.col("k3") * (3 * (t + math.pi / 2)).cos()
        + pl.col("ear") * (bump(math.pi / 2 - phi) + bump(math.pi / 2 + phi))
    )
    x0, y0 = origin
    contours = (
        PARTS.with_row_index("part")
        .with_columns(i=pl.int_ranges(0, N))
        .explode("i")
        .group_by("part", "layer", maintain_order=True)
        .agg(
            exterior=pl.struct(
                x=x0 + size * (pl.col("cx") + pl.col("sx") * r * t.cos()),
                y=y0 + size * (pl.col("cy") - pl.col("sy") * r * t.sin()),
            )
        )
        .select(
            "layer",
            contour=pl.struct(
                "exterior", holes=pl.lit([]), is_closed=pl.lit(True)
            ).cast(CONTOUR_SCHEMA),
        )
    )
    return contours.select(
        pl.col("contour").filter(pl.col("layer") == name).implode().alias(name)
        for name in PARTS["layer"].unique(maintain_order=True)
    )


def halftone_index(width: int, height: int, cell: float):
    """The bear as a halftone index map on a ``width`` x ``height`` working
    canvas of square cells ``cell`` pixels wide, over the columns of
    ``outlines``."""
    canvas = {"width": width, "height": height}
    contours = Pipeline().source("contour")
    head = pl.col("head").cv.pipe(contours)
    features = pl.col("feature").cv.pipe(contours)

    # The field: the head's signed distance, in cells (negative inside).
    inside = head.rasterize(**canvas, fill_value=1)
    sdf = inside.distance_transform(signed=True).scale(1 / cell)
    # Dot radius in cells: a step at the silhouette, so it reads without being
    # drawn; growing inward (up to just past touching), fading outward.
    radius = (
        sdf.neg()
        .relu()
        .sqrt()
        .scale(0.13)
        .add(sdf.relu().scale(-0.035))
        .add(inside.scale(0.16))
        .add_constant(0.13)
        .clamp(0.0, 0.56)
    )
    # One radius per cell: down to the cell grid, back up by nearest.
    radius = radius.resize(
        height=round(height / cell), width=round(width / cell), filter="bilinear"
    ).resize(**canvas, filter="nearest")
    # Each pixel's distance from its cell's centre, in cells.
    grid = sdf.coordinate_grid().scale(1 / cell)
    offset = grid.subtract(grid.floor()).add_constant(-0.5).square()
    distance = offset.channel_select(0).add(offset.channel_select(1)).sqrt()
    dot = radius.subtract(distance).threshold(0.0).scale(1 / 255)  # 0 or 1

    # Which tint each dot takes; the eyes and nose are inked over the dots.
    region = (
        pl.col("muzzle")
        .cv.pipe(contours)
        .difference(features)
        .rasterize(**canvas, fill_value=MUZZLE)
        .maximum(head.difference(features).rasterize(**canvas, fill_value=FACE))
        .clamp(HALO, MUZZLE)  # outside the head: the halo
    )
    ink = features.rasterize(**canvas, fill_value=INK)
    return dot.multiply(region).maximum(ink)


def _render(df: pl.DataFrame, index, width: int, height: int) -> bytes:
    rgb = index.clamp(0, len(PALETTE) - 1, out_dtype="u8").apply_lut(PALETTE)
    return df.select(rgb.resize(height=height, width=width).sink("png")).item()


def render_mark(size: int = 512) -> bytes:
    """The square mark, ``size`` pixels a side."""
    w = size * WORK
    df = outlines(origin=(MARGIN * w, MARGIN * w), size=(1 - 2 * MARGIN) * w)
    return _render(df, halftone_index(w, w, w / CELLS), size, size)


def render_lockup() -> bytes:
    """The mark beside the wordmark, sharing its baseline."""
    w, h = LOCKUP_WIDTH * WORK, LOCKUP_HEIGHT * WORK
    df = outlines(origin=(0.0, 0.0), size=float(h))
    *main, pen = wordmark("polars", x=TEXT_X, baseline=BASELINE, scale=WORK)
    *accent, _ = wordmark("-cv", x=pen + SPACING, baseline=BASELINE, scale=WORK)
    canvas = {"width": w, "height": h}
    contours = Pipeline().source("contour")
    letter = {
        "l": contours.stroke(width=STROKE * WORK, cap="butt"),  # lines: stroked
        "r": contours,  # regions: filled
    }
    # The outline is the letter grown by OUTLINE, in ink; `maximum` keeps the
    # letter (a higher index) over it.
    paints = [
        (pipe.offset(OUTLINE * WORK, join="miter"), kind, INK)
        for kind, pipe in letter.items()
    ] + [(pipe, kind, None) for kind, pipe in letter.items()]
    index = halftone_index(w, h, h / CELLS)
    for (lines, regions), value in ((main, TEXT), (accent, ACCENT)):
        df = df.with_columns(
            pl.Series(f"l{value}", [lines], dtype=CONTOUR_SET_SCHEMA),
            pl.Series(f"r{value}", [regions], dtype=CONTOUR_SET_SCHEMA),
        )
        for pipe, kind, ink in paints:
            index = index.maximum(
                pl.col(f"{kind}{value}").cv.pipe(
                    pipe.rasterize(**canvas, fill_value=ink or value)
                )
            )
    return _render(df, index, LOCKUP_WIDTH, LOCKUP_HEIGHT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=HERE)
    args = parser.parse_args()
    for name, png in (
        ("polars-cv-halftone.png", render_lockup()),
        ("polars-cv-halftone-mark.png", render_mark()),
    ):
        path = args.out_dir / name
        path.write_bytes(png)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
