"""Generate the README's cutout example: its figure and its graph.

The input is scikit-image's ``coffee.png`` (CC0, photographer Rachel Michetti),
fetched from a pinned scikit-image tag and checked against its SHA-256.
``cutout()`` is the README's example as it runs here: two cues computed from
one decode, joined into a mask, traced to an outline, rasterized back into an
alpha channel and stacked under the photo's colour channels. The figure shows
the input beside the cutout (over a checkerboard, so the transparency shows);
the graph is the example's ``show_graph()``.

    cd polars-cv
    uv run --no-sync python scripts/make_readme_cutout.py

Writes ``docs/assets/readme-cutout.png`` and ``docs/assets/readme-cutout-graph.svg``
(``show_graph()`` needs the ``viz`` extra and Graphviz's ``dot``). The README
embeds both through absolute ``raw.githubusercontent.com`` URLs so that PyPI
renders them too.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from PIL import Image  # noqa: E402

from polars_cv import LazyPipelineExpr, Pipeline  # noqa: E402

SOURCE_URL = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/"
    "v0.25.2/skimage/data/coffee.png"
)
SOURCE_SHA256 = "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7"
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
SINKS = {"cutout": "png", "area": "native"}


def fetch_source(cache: Path) -> bytes:
    """The sample image's bytes, downloaded once and verified every time."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as r:
            cache.write_bytes(r.read())
    data = cache.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(
            f"{cache} has SHA-256 {digest}, expected {SOURCE_SHA256}; "
            "delete it to re-download"
        )
    return data


def cutout() -> LazyPipelineExpr:
    """The README's example, up to the sink."""
    base = Pipeline().source("file_path").resize_max(512)
    img = pl.col("path").cv

    bright = img.pipe(base.grayscale().threshold(190))
    saturated = img.pipe(
        base.to_hsv().channel_select(1).threshold(pl.col("min_saturation"))
    )

    outline = bright.maximum(saturated).pipe(
        Pipeline().morphology_open(ksize=5).extract_contours().largest().convex_hull()
    )

    photo = img.pipe(base)
    alpha = outline.rasterize(shape=photo, supersample=4)
    cutout = photo.channel_merge(alpha).alias("cutout")

    return cutout.merge_pipe(outline.area().alias("area"))


def checkerboard(rgba: np.ndarray, cell: int = 16) -> np.ndarray:
    """``rgba`` composited over a grey checkerboard, as an image editor shows
    transparency."""
    h, w = rgba.shape[:2]
    y, x = np.mgrid[:h, :w]
    board = np.where((x // cell + y // cell) % 2, 240.0, 200.0)[..., None]
    alpha = rgba[..., 3:] / 255.0
    return (rgba[..., :3] * alpha + board * (1 - alpha)).astype(np.uint8)


def render(image: bytes, cutout_png: bytes, out: Path) -> None:
    panels = [
        ("image", np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))),
        ("cutout", checkerboard(np.asarray(Image.open(io.BytesIO(cutout_png))))),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, (title, arr) in zip(axes, panels, strict=True):
        ax.imshow(arr)
        ax.set_title(title, fontsize=10, family="monospace")
        ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ASSETS)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "polars-cv" / "coffee.png",
    )
    args = parser.parse_args()
    image = fetch_source(args.cache)
    expr = cutout()

    out = (
        pl.LazyFrame({"path": [str(args.cache)], "min_saturation": 210})
        .with_columns(o=expr.sink(SINKS))
        .unnest("o")
        .collect()
    )
    figure = args.out_dir / "readme-cutout.png"
    render(image, out["cutout"].item(), figure)
    print(f"wrote {figure} (outline area {out['area'].item()[0]:g} px)")

    graph = args.out_dir / "readme-cutout-graph.svg"
    graph.write_bytes(
        expr.sink(SINKS, return_expr=False).show_graph().pipe(format="svg")
    )
    print(f"wrote {graph}")


if __name__ == "__main__":
    main()
