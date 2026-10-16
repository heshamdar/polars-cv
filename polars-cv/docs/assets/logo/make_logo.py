"""Generate the polars-cv logo with polars-cv itself.

A low-poly polar bear drawn as a topographic map of itself, beside a wordmark
built from contours. "Contour" in both senses: the computer-vision kind traces
the bear, and the cartographic kind shades it.

1. ``bear.py`` sculpts the bear: primitives -> rasterize -> blur -> threshold
   (metaball fusion) -> extract_contours -> simplify (the facets).
2. The DataFrame is the canvas: one row per pixel, and
   ``.point.signed_distance_to_contour`` gives each its distance to the bear.
3. A polars-cv graph turns that field into the image. It upsamples the field
   (a smooth field interpolates to smooth iso-lines), thresholds it at nested
   depths for the terraces and at outward offsets for the contour lines, and
   ``morphology_gradient`` outlines the latter. It weights every layer by a
   per-channel colour step and sums them (the layers nest, so each pixel sums
   to its own terrace's colour), then ``channel_merge`` -> lanczos downsample
   -> PNG.

    cd polars-cv
    uv run --no-sync python docs/assets/logo/make_logo.py [--stages]

Writes ``polars-cv-logo.png`` next to this script; ``--stages`` also writes
the bear's three construction stages as ``bear-*.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SCHEMA, CONTOUR_SET_SCHEMA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bear  # noqa: E402
from wordmark import SPACING, wordmark  # noqa: E402

# --- layout (1x pixels) -------------------------------------------------------
WIDTH, HEIGHT = 980, 220
SUPERSAMPLE = 4  # every rasterized/thresholded layer runs at this multiple
FIELD_SCALE = 2  # distance-field grid, upsampled to SUPERSAMPLE before thresholding
BASELINE = 160.0  # the bear's feet and the text share it
BEAR_SCALE = 1.15
BEAR_X = 25.0
TEXT_X = 350.0

TERRACE_STEP = 5.0  # depth between terraces inside the bear
RING_STEP = 8.0  # spacing of the contour lines outside it
RING_ALPHA = (0.75, 0.45, 0.2)  # one per line, fading outward


def _hex(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


BACKGROUND = _hex("#0A1628")
TERRACES = [_hex(h) for h in ("#4F8FD6", "#7DB0E6", "#A9CDF2", "#D3E7FA", "#F7FBFF")]
RING = _hex("#3D73B8")
TEXT_MAIN = _hex("#F1F8FF")
TEXT_ACCENT = _hex("#7FB6EC")


def bear_contour() -> dict:
    """The bear outline placed on the logo canvas, feet on the baseline."""
    dy = BASELINE - 141.0 * BEAR_SCALE
    return {
        "exterior": [
            {"x": x * BEAR_SCALE + BEAR_X, "y": y * BEAR_SCALE + dy}
            for x, y in bear.outline()
        ],
        "holes": [],
        "is_closed": True,
    }


def distance_field(contour: dict) -> np.ndarray:
    """Signed distance (1x pixels, negative inside) at each field-grid pixel centre."""
    fw, fh = WIDTH * FIELD_SCALE, HEIGHT * FIELD_SCALE
    xs = (np.arange(fw) + 0.5) / FIELD_SCALE
    ys = (np.arange(fh) + 0.5) / FIELD_SCALE
    grid = pl.DataFrame({"x": np.tile(xs, fh), "y": np.repeat(ys, fw)})
    dist = grid.select(
        pl.struct("x", "y").point.signed_distance_to_contour(
            pl.lit(contour, dtype=CONTOUR_SCHEMA)
        )
    ).to_series()
    return dist.cast(pl.Float32).to_numpy().reshape(fh, fw)


def render(field: np.ndarray, main: list[dict], accent: list[dict]) -> bytes:
    w, h = WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE
    df = pl.DataFrame(
        {"field": [field], "main": [main], "accent": [accent]},
        schema={
            "field": pl.Array(pl.Float32, field.shape),
            "main": CONTOUR_SET_SCHEMA,
            "accent": CONTOUR_SET_SCHEMA,
        },
    )

    def unit(pipe: Pipeline) -> Pipeline:  # u8 mask -> f32 in [0, 1]
        return pipe.cast("f32").scale(1 / 255)

    # Depth below the bear's outline: positive inside, negative outside.
    depth = (
        Pipeline().source("array").resize(height=h, width=w, filter="bilinear").neg()
    )
    terraces = [
        pl.col("field").cv.pipe(unit(depth.threshold(k * TERRACE_STEP)))
        for k in range(len(TERRACES))
    ]
    rings = [
        pl.col("field").cv.pipe(
            unit(
                depth.threshold(-j * RING_STEP).morphology_gradient(
                    ksize=2 * SUPERSAMPLE - 1
                )
            )
        )
        for j in range(1, len(RING_ALPHA) + 1)
    ]
    glyphs = Pipeline().source("contour").rasterize(width=w, height=h).channel_select(0)
    main_mask = pl.col("main").cv.pipe(unit(glyphs))
    accent_mask = pl.col("accent").cv.pipe(unit(glyphs))

    def channel(c: int):
        # Terraces nest, so adding each one's step over the colour below it
        # sums to exactly the colour of the innermost terrace at each pixel.
        acc = terraces[0].scale(TERRACES[0][c] - BACKGROUND[c])
        for k in range(1, len(TERRACES)):
            acc = acc.add(terraces[k].scale(TERRACES[k][c] - TERRACES[k - 1][c]))
        for alpha, ring in zip(RING_ALPHA, rings, strict=True):
            acc = acc.add(ring.scale(alpha * (RING[c] - BACKGROUND[c])))
        acc = acc.add(main_mask.scale(TEXT_MAIN[c] - BACKGROUND[c]))
        acc = acc.add(accent_mask.scale(TEXT_ACCENT[c] - BACKGROUND[c]))
        return acc.add_constant(BACKGROUND[c]).clamp(0, 255).cast("u8")

    rgb = channel(0).channel_merge(channel(1), channel(2))
    return df.select(rgb.resize(height=HEIGHT, width=WIDTH).sink("png")).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "polars-cv-logo.png")
    parser.add_argument(
        "--stages", action="store_true", help="also write the bear's stages"
    )
    args = parser.parse_args()

    main_text, pen = wordmark("polars", x=TEXT_X, baseline=BASELINE, scale=SUPERSAMPLE)
    accent_text, _ = wordmark(
        "-cv", x=pen + SPACING, baseline=BASELINE, scale=SUPERSAMPLE
    )
    args.out.write_bytes(render(distance_field(bear_contour()), main_text, accent_text))
    print(f"wrote {args.out}")
    if args.stages:
        for name, png in bear.stages().items():
            path = args.out.parent / f"bear-{name}.png"
            path.write_bytes(png)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
