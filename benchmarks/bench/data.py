"""Fetch COCO val2017 and build the benchmark manifests.

    uv run python -m bench.data              # 5,000 images (coco-5k)
    uv run python -m bench.data --scale 50000  # + coco-50k (hard links)

The archive is checked against COCO's published MD5 and a SHA-256 pinned here.
The official host is tried first, then the fast.ai S3 mirror of the same file.

A manifest is a Parquet file with one row per image: ``path``, the header
dimensions, and the per-row parameters workload 4 reads (a crop box inside the
image and a target size), drawn from a seeded RNG so every run and every
library sees the same values. The scaled set replicates the 5k images by hard
link under new names: the bytes are identical, so near-duplicate search finds
real duplicate groups, and no extra disk is used.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"
ARCHIVE = DATA / "val2017.zip"
IMAGES = DATA / "val2017"
URLS = (
    "http://images.cocodataset.org/zips/val2017.zip",
    "https://s3.amazonaws.com/fast-ai-coco/val2017.zip",
)
#: From cocodataset.org's download page.
MD5 = "442b8da7639aecaf257c1dceb8ba8c80"
SHA256 = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
N_IMAGES = 5000
SEED = 20261013


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    if not ARCHIVE.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        part = ARCHIVE.with_suffix(".part")
        for url in URLS:
            try:
                print(f"downloading {url}")
                with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                break
            except OSError as e:
                print(f"  failed: {e}")
        else:
            raise SystemExit("no download source reachable")
        part.rename(ARCHIVE)
    for algo, want in (("md5", MD5), ("sha256", SHA256)):
        got = _digest(ARCHIVE, algo)
        if got != want:
            raise SystemExit(f"{ARCHIVE}: {algo} {got} != {want}; delete it and retry")
    print(f"{ARCHIVE.name}: md5 and sha256 verified")


def extract() -> list[Path]:
    if not IMAGES.exists():
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(DATA)
    files = sorted(IMAGES.glob("*.jpg"))
    if len(files) != N_IMAGES:
        raise SystemExit(f"expected {N_IMAGES} images in {IMAGES}, found {len(files)}")
    return files


def workload4_params(width: np.ndarray, height: np.ndarray, seed: int) -> dict:
    """A crop box inside each image (50-100% of each side) and a target size."""
    rng = np.random.default_rng(seed)
    n = len(width)
    crop_h = (height * rng.uniform(0.5, 1.0, n)).astype(np.int64).clip(1)
    crop_w = (width * rng.uniform(0.5, 1.0, n)).astype(np.int64).clip(1)
    top = (rng.uniform(0, 1, n) * (height - crop_h + 1)).astype(np.int64)
    left = (rng.uniform(0, 1, n) * (width - crop_w + 1)).astype(np.int64)
    return {
        "crop_top": top,
        "crop_left": left,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "target_h": rng.integers(64, 321, n),
        "target_w": rng.integers(64, 321, n),
    }


def manifest(files: list[Path]) -> pl.DataFrame:
    dims = []
    for f in files:
        with Image.open(f) as im:
            dims.append((im.width, im.height, len(im.getbands())))
    w, h, c = (np.array(x) for x in zip(*dims, strict=True))
    return pl.DataFrame(
        {
            "path": [str(f) for f in files],
            "width": w,
            "height": h,
            "channels": c,
            **workload4_params(w, h, SEED),
        }
    ).with_row_index("id")


def scaled(base: pl.DataFrame, n: int) -> pl.DataFrame:
    """``n`` rows cycling over ``base``, each a hard link under a new name."""
    out = DATA / f"val2017-x{n}"
    out.mkdir(exist_ok=True)
    reps = -(-n // base.height)
    rows = pl.concat([base] * reps).head(n).drop("id").with_row_index("id")
    paths = []
    for i, src in zip(rows["id"], rows["path"], strict=True):
        dst = out / f"{i:07d}.jpg"
        if not dst.exists():
            dst.hardlink_to(src)
        paths.append(str(dst))
    return rows.with_columns(path=pl.Series(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scale", type=int, default=0, help="also build an n-image set"
    )
    args = parser.parse_args()
    download()
    base_path = DATA / "coco-5k.parquet"
    if base_path.exists():
        base = pl.read_parquet(base_path)
    else:
        base = manifest(extract())
        base.write_parquet(base_path)
    print(f"{base_path.name}: {base.height} images")
    if args.scale:
        big = scaled(base, args.scale)
        big_path = DATA / f"coco-{args.scale // 1000}k.parquet"
        big.write_parquet(big_path)
        print(f"{big_path.name}: {big.height} images")


if __name__ == "__main__":
    main()
