"""Regenerate the decode fixtures: one 16x16 RGB image per `image` feature.

Run from ``polars-cv/``: ``python tests/fixtures/formats/generate.py``. Each
file is named ``sample.<feature>``, the feature being the one in the
workspace's ``image`` dependency that decodes it, so
``tests/test_decode_formats.py`` can hold the manifest, these files and the
documented list to one another. Writers: Pillow where it has one, OpenCV for
the two float formats, and a few lines here for farbfeld, which neither has.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).parent
SIZE = 16


def pixels() -> np.ndarray:
    """The image every fixture holds: four flat 8x8 quadrants.

    Flat blocks survive the two lossy codecs here (JPEG, and DDS as DXT1)
    closely enough to compare with a small tolerance.
    """
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[:8, :8] = (200, 30, 30)
    img[:8, 8:] = (30, 200, 30)
    img[8:, :8] = (30, 30, 200)
    img[8:, 8:] = (220, 220, 60)
    return img


def farbfeld(img: np.ndarray) -> bytes:
    h, w, _ = img.shape
    rgba = np.full((h, w, 4), 0xFFFF, dtype=">u2")
    rgba[..., :3] = img.astype(">u2") * 257
    return b"farbfeld" + struct.pack(">II", w, h) + rgba.tobytes()


def main() -> None:
    img = pixels()
    pil = Image.fromarray(img)
    pil.save(HERE / "sample.bmp", format="BMP")
    pil.save(HERE / "sample.dds", format="DDS", pixel_format="DXT1")
    pil.save(HERE / "sample.gif", format="GIF")
    # `image` reads a PNG-encoded icon only as RGBA.
    pil.convert("RGBA").save(HERE / "sample.ico", format="ICO", sizes=[(SIZE, SIZE)])
    pil.save(HERE / "sample.jpeg", format="JPEG", quality=95, subsampling=0)
    pil.save(HERE / "sample.png", format="PNG")
    pil.save(HERE / "sample.pnm", format="PPM")
    pil.save(HERE / "sample.qoi", format="QOI")
    pil.save(HERE / "sample.tiff", format="TIFF")
    pil.save(HERE / "sample.webp", format="WEBP", lossless=True)
    (HERE / "sample.ff").write_bytes(farbfeld(img))
    # The float formats hold the same image scaled to [0, 1]. OpenCV writes
    # BGR, and names the file type by its extension.
    bgr = (img[..., ::-1] / 255.0).astype(np.float32)
    for feature in ("exr", "hdr"):
        tmp = HERE / f"tmp.{feature}"
        assert cv2.imwrite(str(tmp), bgr)
        tmp.rename(HERE / f"sample.{feature}")


if __name__ == "__main__":
    main()
