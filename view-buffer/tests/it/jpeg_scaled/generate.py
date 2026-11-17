"""Regenerate the scaled-JPEG-decode fixtures (`tests/it/jpeg_scaled.rs`).

Each `<name>.jpg` is a small synthetic image (sizes deliberately not
multiples of 8 or 16) encoded by libjpeg-turbo: through Pillow for the
baseline/progressive/restart/grayscale cases, through imagecodecs for the
4:1:1 and 4:4:0 subsamplings Pillow cannot write. Each `<name>.s<f>.png` is
libjpeg-turbo's own scaled decode of it at 1/f (Pillow's `draft`), the
reference the vendored zune-jpeg's scaled output is held to; `.L.s<f>.png`
is the same decode asked for grayscale (luma) output, and `.s1.png` a full
decode, kept only for a file zune decodes differently even at full size.

Run with: uv run --no-project --with pillow==12.0.0 --with imagecodecs \
    --with numpy python -I generate.py
"""

from __future__ import annotations

import io
import math
import pathlib

import imagecodecs
import numpy as np
from PIL import Image, features

HERE = pathlib.Path(__file__).resolve().parent
SCALES = (2, 4, 8)


def content(height: int, width: int, seed: int) -> np.ndarray:
    """Smooth gradients, a sharp-edged box, a ring and mild noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    r = 128 + 100 * np.sin(x / 7.0) * np.cos(y / 11.0)
    g = 255 * x / max(width - 1, 1)
    radius = np.hypot(x - width / 2, y - height / 2)
    b = np.where(np.abs(radius - min(height, width) / 3) < 3, 230.0, 40.0)
    box = (x > width // 5) & (x < width // 2) & (y > height // 4) & (y < height // 2)
    r[box], g[box], b[box] = 250, 20, 120
    img = np.stack([r, g, b], axis=-1) + rng.normal(0, 6, (height, width, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def pillow(img: np.ndarray, **save: object) -> bytes:
    out = io.BytesIO()
    mode = "L" if img.ndim == 2 else "RGB"
    Image.fromarray(img, mode).save(out, "JPEG", quality=90, **save)
    return out.getvalue()


def gray(img: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(img, "RGB").convert("L"))


CASES: dict[str, bytes] = {
    "b444_61x37": pillow(content(37, 61, 1), subsampling="4:4:4"),
    "b422_61x37": pillow(content(37, 61, 2), subsampling="4:2:2"),
    "b420_61x37": pillow(content(37, 61, 3), subsampling="4:2:0"),
    "b420_130x97": pillow(content(97, 130, 4), subsampling="4:2:0"),
    "b420_17x9": pillow(content(9, 17, 5), subsampling="4:2:0"),
    "b420_rst_100x75": pillow(
        content(75, 100, 6), subsampling="4:2:0", restart_marker_blocks=3
    ),
    "b411_70x45": imagecodecs.jpeg8_encode(
        content(45, 70, 7), level=90, subsampling="411"
    ),
    "b440_70x45": imagecodecs.jpeg8_encode(
        content(45, 70, 8), level=90, subsampling="440"
    ),
    "p444_61x37": pillow(content(37, 61, 9), subsampling="4:4:4", progressive=True),
    "p422_53x41": pillow(content(41, 53, 10), subsampling="4:2:2", progressive=True),
    "p420_100x75": pillow(
        content(75, 100, 11), subsampling="4:2:0", progressive=True
    ),
    "p420_rst_77x50": pillow(
        content(50, 77, 12),
        subsampling="4:2:0",
        progressive=True,
        restart_marker_blocks=2,
    ),
    "bgray_45x29": pillow(gray(content(29, 45, 13))),
    "pgray_45x29": pillow(gray(content(29, 45, 14)), progressive=True),
}

# Colour files whose luma-only scaled decode is also pinned.
LUMA = ("b420_61x37", "p420_100x75", "b422_61x37")

# Files zune-jpeg 0.5.8 decodes differently from libjpeg-turbo even at full
# size; their full decode (`.s1.png`) pins that the divergence is upstream's.
ZUNE_DIVERGENT = ("p420_rst_77x50",)


def reference(data: bytes, mode: str, factor: int) -> Image.Image:
    im = Image.open(io.BytesIO(data))
    width, height = im.size
    im.draft(mode, (max(1, width // factor), max(1, height // factor)))
    im = im.convert(mode)
    want = (math.ceil(width / factor), math.ceil(height / factor))
    assert im.size == want, (im.size, want, factor)
    return im


def main() -> None:
    assert features.check_feature("libjpeg_turbo")
    for old in HERE.glob("*.jpg"):
        old.unlink()
    for old in HERE.glob("*.png"):
        old.unlink()
    for name, data in CASES.items():
        (HERE / f"{name}.jpg").write_bytes(data)
        mode = "L" if name[1:5] == "gray" else "RGB"
        for factor in SCALES:
            reference(data, mode, factor).save(HERE / f"{name}.s{factor}.png")
            if name in LUMA:
                reference(data, "L", factor).save(HERE / f"{name}.L.s{factor}.png")
        if name in ZUNE_DIVERGENT:
            reference(data, mode, 1).save(HERE / f"{name}.s1.png")


if __name__ == "__main__":
    main()
