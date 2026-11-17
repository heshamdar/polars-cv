"""Regenerate the scaled-JPEG-decode fixtures (`tests/it/jpeg_scaled.rs`).

Each `<name>.jpg` is a small synthetic image (sizes deliberately not
multiples of 8 or 16) encoded by libjpeg-turbo: through Pillow for the
baseline/progressive/restart/grayscale cases, through imagecodecs for the
4:1:1 and 4:4:0 subsamplings Pillow cannot write. Each `<name>.s<f>.png` is
libjpeg-turbo's own scaled decode of it at 1/f (Pillow's `draft`), the
reference the vendored zune-jpeg's scaled output is held to; `.L.s<f>.png`
is the same decode asked for grayscale (luma) output, and `.s1.png` the full
decode, the reference for zune's full-size output.

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
    "p422_rstrow_200x150": pillow(
        content(150, 200, 17),
        subsampling="4:2:2",
        progressive=True,
        restart_marker_rows=1,
    ),
    "bgray_45x29": pillow(gray(content(29, 45, 13))),
    "pgray_45x29": pillow(gray(content(29, 45, 14)), progressive=True),
}

# Larger progressive files for the 1/8 decode's DC-scan selection (only the
# DC scans are entropy-decoded at 1/8): its output is pinned byte for byte to
# the decode that read every scan (`dc_scan_digests.txt`), so these carry no
# reference PNGs. libjpeg's progressive script has an interleaved first DC
# scan (Al=1) and an interleaved DC refinement scan; `split_dc_scans` makes
# a file whose DC scans are one per component instead.
DC_SCAN_CASES: dict[str, bytes] = {
    "p420_512x384": pillow(
        content(384, 512, 15), subsampling="4:2:0", progressive=True
    ),
    "p420_rst_512x384": pillow(
        content(384, 512, 16),
        subsampling="4:2:0",
        progressive=True,
        restart_marker_blocks=5,
    ),
    "pgray_256x192": pillow(gray(content(192, 256, 18)), progressive=True),
}


def split_dc_scans(data: bytes) -> bytes:
    """Re-pack each interleaved DC scan of a 4:4:4 progressive JPEG (no
    restart interval) as one non-interleaved scan per component.

    With every component at 1x1 sampling, a component's blocks come in the
    same raster order interleaved or not, so its DC predictions and hence
    its Huffman codes are unchanged: each block's code bits are copied
    as they are, only regrouped by component. The pixels are the input's.
    """
    out = bytearray(data[:2])
    tables: dict[int, dict[tuple[int, int], int]] = {}
    comps: dict[int, int] = {}  # component id -> DC table
    blocks = 0
    i = 2
    while True:
        marker = data[i + 1]
        if marker == 0xD9:
            out += data[i:]
            return bytes(out)
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        seg = data[i + 4 : i + 2 + length]
        end = i + 2 + length
        if marker == 0xDD:
            assert int.from_bytes(seg, "big") == 0, "restart intervals unsupported"
        if marker == 0xC2:
            height, width = (int.from_bytes(seg[k : k + 2], "big") for k in (1, 3))
            assert all(seg[7 + 3 * k] == 0x11 for k in range(seg[5])), "4:4:4 only"
            blocks = math.ceil(width / 8) * math.ceil(height / 8)
        if marker == 0xC4:
            k = 0
            while k < len(seg):
                tc_th, counts = seg[k], seg[k + 1 : k + 17]
                symbols = iter(seg[k + 17 : k + 17 + sum(counts)])
                code, table = 0, {}
                for bits, count in enumerate(counts, 1):
                    for _ in range(count):
                        table[(bits, code)] = next(symbols)
                        code += 1
                    code <<= 1
                if tc_th >> 4 == 0:
                    tables[tc_th & 15] = table
                k += 17 + sum(counts)
        if marker != 0xDA:
            out += data[i:end]
            i = end
            continue
        n = seg[0]
        ids = [seg[1 + 2 * k] for k in range(n)]
        for k in range(n):
            comps[ids[k]] = seg[2 + 2 * k] >> 4
        ss, se, ahal = seg[1 + 2 * n], seg[2 + 2 * n], seg[3 + 2 * n]
        j = end
        while data[j] != 0xFF or data[j + 1] == 0:
            j += 2 if data[j] == 0xFF else 1
        if ss != 0 or n == 1:
            out += data[i:j]
            i = j
            continue
        raw = data[end:j].replace(b"\xff\x00", b"\xff")
        bits = "".join(f"{b:08b}" for b in raw)
        pos = 0
        per: list[list[str]] = [[] for _ in ids]
        for _ in range(blocks):
            for c, cid in enumerate(ids):
                start = pos
                if ahal >> 4 == 0:
                    table, length_, code = tables[comps[cid]], 0, 0
                    while (length_, code) not in table:
                        code = code << 1 | int(bits[pos])
                        length_ += 1
                        pos += 1
                    pos += table[(length_, code)]
                else:
                    pos += 1
                per[c].append(bits[start:pos])
        for c, cid in enumerate(ids):
            body = "".join(per[c])
            body += "1" * (-len(body) % 8)
            ecs = int(body, 2).to_bytes(len(body) // 8, "big") if body else b""
            header = bytes([1, cid, comps[cid] << 4, ss, se, ahal])
            out += b"\xff\xda" + (len(header) + 2).to_bytes(2, "big") + header
            out += ecs.replace(b"\xff", b"\xff\x00")
        i = j


DC_SCAN_CASES["p444_dcsplit_100x75"] = split_dc_scans(
    pillow(content(75, 100, 19), subsampling="4:4:4", progressive=True)
)


# Colour files whose luma-only scaled decode is also pinned.
LUMA = ("b420_61x37", "p420_100x75", "b422_61x37")


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
        reference(data, mode, 1).save(HERE / f"{name}.s1.png")
        for factor in SCALES:
            reference(data, mode, factor).save(HERE / f"{name}.s{factor}.png")
            if name in LUMA:
                reference(data, "L", factor).save(HERE / f"{name}.L.s{factor}.png")
    for name, data in DC_SCAN_CASES.items():
        (HERE / f"{name}.jpg").write_bytes(data)
    # The split file is a different stream of the same coefficients.
    split = Image.open(io.BytesIO(DC_SCAN_CASES["p444_dcsplit_100x75"]))
    unsplit = Image.open(
        io.BytesIO(pillow(content(75, 100, 19), subsampling="4:4:4", progressive=True))
    )
    assert np.array_equal(np.asarray(split), np.asarray(unsplit))


if __name__ == "__main__":
    main()
