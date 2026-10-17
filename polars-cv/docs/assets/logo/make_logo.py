"""Generate the polars-cv logo with polars-cv itself.

A low-poly polar bear drawn as a topographic map of itself, beside a wordmark
built from contours. "Contour" in both senses: the computer-vision kind traces
the bear, and the cartographic kind shades it.

The image is an index map coloured by one palette: each layer paints its
palette index, the layers combine by ``maximum`` (a higher index is drawn
over a lower one), and ``apply_lut`` turns indices into colours. All of it is
one polars-cv graph over three contour columns:

1. ``bear.py`` sculpts the bear: primitives -> rasterize -> blur -> threshold
   (metaball fusion) -> extract_contours -> simplify (the facets).
2. Terraces: rasterize the bear, ``distance_transform(signed=True)``, and
   ``floor`` its depth over the terrace step into indices 1 to 5.
3. Contour lines: ``offset`` the bear outward, take each offset's
   ``boundary()`` and stroke it with ``rasterize(line_width=)``.
4. The wordmark: glyph centrelines stroked at the font's weight
   (``wordmark.py``).
5. ``apply_lut`` the palette, then a lanczos downsample from the working
   resolution anti-aliases every edge -> PNG.

    cd polars-cv
    uv run --no-sync python docs/assets/logo/make_logo.py [--stages]

Writes ``polars-cv-logo.png`` next to this script; ``--stages`` also writes
the bear's three construction stages as ``bear-*.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bear  # noqa: E402
from wordmark import SPACING, STROKE, wordmark  # noqa: E402

# --- layout (1x pixels) -------------------------------------------------------
WIDTH, HEIGHT = 980, 220
WORK = 4  # the graph draws at this multiple, then downsamples (anti-aliasing)
BASELINE = 160.0  # the bear's feet and the text share it
BEAR_SCALE = 1.15
BEAR_X = 25.0
TEXT_X = 350.0

TERRACE_STEP = 5.0  # depth between terraces inside the bear
RING_STEP = 8.0  # spacing of the contour lines outside it
RING_WIDTH = 1.75  # their stroke
RING_ALPHA = (0.75, 0.45, 0.2)  # one per line, fading outward


def _hex(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _mix(a: tuple[int, ...], b: tuple[int, ...], t: float) -> list[int]:
    return [round(x + (y - x) * t) for x, y in zip(a, b, strict=True)]


BACKGROUND = _hex("#0A1628")
TERRACES = [_hex(h) for h in ("#4F8FD6", "#7DB0E6", "#A9CDF2", "#D3E7FA", "#F7FBFF")]
RING = _hex("#3D73B8")
TEXT_MAIN = _hex("#F1F8FF")
TEXT_ACCENT = _hex("#7FB6EC")

# The palette, by index: background, the terraces from the edge inward, the
# contour lines (their fade baked against the background), then the text.
PALETTE = [
    list(BACKGROUND),
    *(list(c) for c in TERRACES),
    *(_mix(BACKGROUND, RING, a) for a in RING_ALPHA),
    list(TEXT_MAIN),
    list(TEXT_ACCENT),
]
FIRST_RING = 1 + len(TERRACES)
MAIN, ACCENT = FIRST_RING + len(RING_ALPHA), FIRST_RING + len(RING_ALPHA) + 1


def bear_contour() -> dict:
    """The bear outline placed on the working canvas, feet on the baseline."""
    dy = BASELINE - 141.0 * BEAR_SCALE
    return {
        "exterior": [
            {"x": (x * BEAR_SCALE + BEAR_X) * WORK, "y": (y * BEAR_SCALE + dy) * WORK}
            for x, y in bear.outline()
        ],
        "holes": [],
        "is_closed": True,
    }


def render(outline: dict, main: list[dict], accent: list[dict]) -> bytes:
    w, h = WIDTH * WORK, HEIGHT * WORK
    df = pl.DataFrame(
        {"bear": [[outline]], "main": [main], "accent": [accent]},
        schema={n: CONTOUR_SET_SCHEMA for n in ("bear", "main", "accent")},
    )
    contours = Pipeline().source("contour")

    # Terrace index: 1 at the edge, one more every TERRACE_STEP inward, 0
    # outside (the signed distance is negative inside, in working pixels).
    terraces = pl.col("bear").cv.pipe(
        contours.rasterize(width=w, height=h)
        .distance_transform(signed=True)
        .scale(-1 / (WORK * TERRACE_STEP))
        .add_constant(1.0)
        .floor()
        .clamp(0, len(TERRACES), out_dtype="u8")
    )
    image = terraces
    for j in range(len(RING_ALPHA)):
        ring = pl.col("bear").cv.pipe(
            contours.offset((j + 1) * RING_STEP * WORK)
            .boundary()
            .rasterize(
                width=w,
                height=h,
                line_width=RING_WIDTH * WORK,
                fill_value=FIRST_RING + j,
            )
        )
        image = image.maximum(ring)
    for column, index in (("main", MAIN), ("accent", ACCENT)):
        text = contours.rasterize(
            width=w, height=h, line_width=STROKE * WORK, cap="butt", fill_value=index
        )
        image = image.maximum(pl.col(column).cv.pipe(text))

    rgb = image.apply_lut(PALETTE).resize(height=HEIGHT, width=WIDTH)
    return df.select(rgb.sink("png")).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "polars-cv-logo.png")
    parser.add_argument(
        "--stages", action="store_true", help="also write the bear's stages"
    )
    args = parser.parse_args()

    main_text, pen = wordmark("polars", x=TEXT_X, baseline=BASELINE, scale=WORK)
    accent_text, _ = wordmark("-cv", x=pen + SPACING, baseline=BASELINE, scale=WORK)
    args.out.write_bytes(render(bear_contour(), main_text, accent_text))
    print(f"wrote {args.out}")
    if args.stages:
        for name, png in bear.stages().items():
            path = args.out.parent / f"bear-{name}.png"
            path.write_bytes(png)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
