"""Generate the README's before/after figure with polars-cv itself.

The input is scikit-image's ``chelsea.png`` (CC0, photographer Stefan van der
Walt), fetched from a pinned scikit-image tag and checked against its SHA-256.
Every panel is computed by polars-cv: the top row is one ``select`` whose
pipelines share a single decode, the bottom row is one pipeline whose blur
sigma and rotation angle come from per-row columns.

    cd polars-cv
    uv run --no-sync python scripts/make_readme_figure.py

Writes ``docs/assets/readme-figure.png``. The README embeds it through an
absolute ``raw.githubusercontent.com`` URL so that PyPI renders it too.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from polars_cv import Pipeline, numpy_from_column  # noqa: E402

SOURCE_URL = (
    "https://raw.githubusercontent.com/scikit-image/scikit-image/"
    "v0.25.2/skimage/data/chelsea.png"
)
SOURCE_SHA256 = "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "assets" / "readme-figure.png"


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


def gallery(image: bytes) -> list[tuple[str, np.ndarray]]:
    """Top row: several pipelines over one image, in one ``select``."""
    src = Pipeline().source("image_bytes")
    pipes = {
        "source": src,
        "grayscale()": src.grayscale(),
        "blur(sigma=3)": src.blur(3.0),
        "blur(1).canny()": src.grayscale().blur(1.0).canny(),
        "equalize_histogram()": src.grayscale().equalize_histogram(),
    }
    df = pl.DataFrame({"image": [image]})
    out = df.select(
        **{k: pl.col("image").cv.pipe(p).sink("numpy") for k, p in pipes.items()}
    )
    return [(k, numpy_from_column(out[k])[0]) for k in pipes]


def per_row(image: bytes) -> list[tuple[str, np.ndarray]]:
    """Bottom row: one pipeline, its parameters read from columns."""
    df = pl.DataFrame(
        {
            "image": [image] * 5,
            "sigma": [0.5, 1.0, 2.0, 4.0, 8.0],
            "angle": [0.0, 10.0, 20.0, 30.0, 45.0],
        }
    )
    pipe = (
        Pipeline().source("image_bytes").rotate(pl.col("angle")).blur(pl.col("sigma"))
    )
    out = df.with_columns(result=pl.col("image").cv.pipe(pipe).sink("numpy"))
    arrays = numpy_from_column(out["result"])
    return [
        (f"angle={a:g}, sigma={s:g}", arr)
        for a, s, arr in zip(out["angle"], out["sigma"], arrays, strict=True)
    ]


def render(top: list, bottom: list, out: Path) -> None:
    cols = max(len(top), len(bottom))
    fig, axes = plt.subplots(2, cols, figsize=(3.2 * cols, 5.4))
    for row, panels in zip(axes, (top, bottom), strict=True):
        for ax in row:
            ax.axis("off")
        for ax, (title, arr) in zip(row, panels, strict=False):
            gray = arr.ndim == 2 or arr.shape[-1] == 1
            ax.imshow(arr.squeeze(), cmap="gray" if gray else None, vmin=0, vmax=255)
            ax.set_title(title, fontsize=10, family="monospace")
    axes[0][0].text(
        -0.04,
        0.5,
        "one decode,\nfive outputs",
        transform=axes[0][0].transAxes,
        ha="right",
        va="center",
        fontsize=11,
    )
    axes[1][0].text(
        -0.04,
        0.5,
        "one pipeline,\nper-row params",
        transform=axes[1][0].transAxes,
        ha="right",
        va="center",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / ".cache" / "polars-cv" / "chelsea.png",
    )
    args = parser.parse_args()
    image = fetch_source(args.cache)
    render(gallery(image), per_row(image), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
