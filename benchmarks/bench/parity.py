"""How far each implementation's output may be from the PIL reference, and why.

``compare`` reduces two normalised outputs (``workloads.to_common``) to a few
numbers. ``check`` holds those numbers to a tolerance. Every tolerance below
says where its slack comes from. A metric a tolerance names but ``compare``
did not produce is a failure, never a pass.

Sources of difference, which are expected:

- **JPEG decoding.** PIL, OpenCV and torchvision use libjpeg-turbo with
  different IDCT and upsampling settings, and polars-cv and Daft use pure-Rust
  decoders. The same file decodes a few levels apart at chroma edges.
- **Bilinear resize.** PIL, torchvision (``antialias=True``), polars-cv and
  Daft widen the filter when downscaling, so the result is area-weighted.
  OpenCV's ``INTER_LINEAR`` samples the 2x2 nearest source pixels at any
  scale, so on a 640 to 224 downscale it aliases. Its tolerance is wider for
  that reason, not because of a bug.
- **Grayscale rounding.** Everyone uses BT.601 weights (0.299, 0.587, 0.114),
  but they round at different points.
- **Thumbnail decode.** PIL's ``thumbnail`` and polars-cv's ``thumbnail``
  both decode a JPEG at a reduced scale, but each picks a different scale
  before resizing.
- **Perceptual hash.** imagehash, OpenCV's ``img_hash``, Daft and polars-cv
  each implement pHash (a DCT of a downscaled grayscale image), but they
  differ in the downscaling and the median, so the bits differ. Exact
  duplicates still collide in every implementation, so the duplicate groups
  must match exactly.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _gray_if_ref_gray(ref: np.ndarray, out: np.ndarray) -> np.ndarray:
    """A 1-channel reference against a 3-channel copy of the same gray pixels."""
    if ref.shape[-1] == 1 and out.shape[-1] == 3:
        return out[..., :1]
    return out


def _decode(b: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(b)) as im:
        a = np.asarray(im)
    return a[..., None] if a.ndim == 2 else a


def compare(workload: str, ref, out) -> dict:
    if workload == "w1":
        if ref.shape != out.shape:
            return {"shape_mismatch": 1}
        d = np.abs(ref.astype(np.int16) - out.astype(np.int16))
        return {
            "shape_mismatch": 0,
            "mean_abs": float(d.mean()),
            "p99_abs": float(np.percentile(d, 99)),
            "max_abs": int(d.max()),
        }
    if workload == "w2":
        if len(ref) != len(out):
            return {"count_mismatch": 1}
        dims, means = [], []
        for r, o in zip(ref, out, strict=True):
            ra, oa = _decode(r), _decode(o)
            oa = _gray_if_ref_gray(ra, oa)
            dims.append(
                max(abs(ra.shape[0] - oa.shape[0]), abs(ra.shape[1] - oa.shape[1]))
            )
            if ra.shape[:2] != oa.shape[:2]:
                mode = "L" if oa.shape[-1] == 1 else "RGB"
                img = Image.fromarray(oa.squeeze(-1) if mode == "L" else oa, mode)
                oa = np.asarray(
                    img.resize((ra.shape[1], ra.shape[0]), Image.Resampling.BILINEAR)
                )
                oa = oa[..., None] if oa.ndim == 2 else oa
            if ra.shape != oa.shape:
                return {"count_mismatch": 0, "channel_mismatch": 1}
            means.append(
                float(np.abs(ra.astype(np.int16) - oa.astype(np.int16)).mean())
            )
        return {
            "count_mismatch": 0,
            "channel_mismatch": 0,
            "dims_max": int(max(dims)),
            "mean_abs": float(np.mean(means)),
            "worst_image_mean_abs": float(max(means)),
        }
    if workload == "w3":
        (rf, rg), (of, og) = ref, out
        if rf["id"].to_list() != of["id"].to_list():
            return {"meta_mismatch": len(rf)}
        meta = sum(
            r != o
            for r, o in zip(
                rf.select("width", "height", "channels").rows(),
                of.select("width", "height", "channels").rows(),
                strict=True,
            )
        )
        ham = [
            bin(a ^ b).count("1") for a, b in zip(rf["hash"], of["hash"], strict=True)
        ]
        return {
            "meta_mismatch": int(meta),
            "hash_mean_bits": float(np.mean(ham)),
            "hash_max_bits": int(max(ham)),
            "groups_equal": int(rg == og),
            "groups": len(og),
        }
    if workload == "w4":
        if len(ref) != len(out):
            return {"shape_mismatch": len(ref)}
        bad, means, maxes = 0, [], []
        for r, o in zip(ref, out, strict=True):
            o = _gray_if_ref_gray(r, o)
            if r.shape != o.shape:
                bad += 1
                continue
            d = np.abs(r.astype(np.int16) - o.astype(np.int16))
            means.append(float(d.mean()))
            maxes.append(int(d.max()))
        if bad:
            return {"shape_mismatch": bad}
        return {
            "shape_mismatch": 0,
            "mean_abs": float(np.mean(means)),
            "worst_image_mean_abs": float(max(means)),
            "max_abs": int(max(maxes)),
        }
    raise ValueError(workload)


#: The sentinel tolerance: every metric zero and the groups equal.
EXACT: dict = {}

#: Lower-is-better metrics, except ``groups_equal`` (must be at least the
#: value given, i.e. 1).
_AT_LEAST = {"groups_equal"}


def check(diff: dict, tol: dict) -> list[str]:
    if tol is EXACT:
        return [
            f"{k}={v}"
            for k, v in diff.items()
            if (k in _AT_LEAST and v != 1)
            or (k not in _AT_LEAST and k != "groups" and v != 0)
        ]
    failures = []
    for key, limit in tol.items():
        if key not in diff:
            failures.append(f"{key} missing")
        elif key in _AT_LEAST:
            if diff[key] < limit:
                failures.append(f"{key}={diff[key]} < {limit}")
        elif diff[key] > limit:
            failures.append(f"{key}={diff[key]} > {limit}")
    return failures


# --- Tolerances -------------------------------------------------------------------
# Each value sits just above what the 40-image sample measured (results/
# parity.json), so a regression shows up, and the comment says where the gap
# comes from. A wrong operation fails anyway: a wrong size or channel count is
# a shape mismatch, and a flip or a missing grayscale moves mean_abs by tens of
# levels, far above any bound here.

_SAME_DECODER = "different JPEG decoder; same area-weighted bilinear and BT.601 gray"
_INTER_LINEAR = (
    "OpenCV INTER_LINEAR does not antialias when downscaling, so edges alias"
)
_PHASH = (
    "a different pHash algorithm and bit layout, so bits are not compared; "
    "exact duplicates must still form the same groups"
)

TOLERANCES: dict[tuple[str, str], dict] = {}
WHY: dict[tuple[str, str], str] = {}


def _tol(libs: tuple[str, ...], workload: str, why: str, **limits) -> None:
    for lib in libs:
        TOLERANCES[(lib, workload)] = limits or EXACT
        WHY[(lib, workload)] = why


_PCV = ("polars-cv-eager", "polars-cv-streaming")
_CV = ("opencv", "opencv-mp")
_REF = ("pil", "pil-mp", "hf-datasets")
for _w in ("w1", "w2", "w3", "w4"):
    _tol(_REF, _w, "the reference code (PIL, imagehash) itself")

_tol(_PCV, "w1", _SAME_DECODER, shape_mismatch=0, mean_abs=0.5, p99_abs=2, max_abs=4)
_tol(
    _PCV,
    "w2",
    "a different JPEG encoder: both outputs are lossy, re-decoded",
    count_mismatch=0,
    channel_mismatch=0,
    dims_max=0,
    mean_abs=3.0,
    worst_image_mean_abs=6.0,
)
_tol(_PCV, "w3", _PHASH, meta_mismatch=0, groups_equal=1)
_tol(
    _PCV,
    "w4",
    _SAME_DECODER,
    shape_mismatch=0,
    mean_abs=0.5,
    worst_image_mean_abs=1.0,
    max_abs=4,
)

_tol(_CV, "w1", _INTER_LINEAR, shape_mismatch=0, mean_abs=5.0, p99_abs=40)
_tol(
    _CV,
    "w2",
    _INTER_LINEAR + "; libjpeg-turbo encoder with OpenCV's settings",
    count_mismatch=0,
    channel_mismatch=0,
    dims_max=0,
    mean_abs=7.0,
    worst_image_mean_abs=15.0,
)
_tol(_CV, "w3", _PHASH, meta_mismatch=0, groups_equal=1)
_tol(
    _CV, "w4", _INTER_LINEAR, shape_mismatch=0, mean_abs=6.0, worst_image_mean_abs=15.0
)

_tol(
    ("torchvision",),
    "w1",
    "torchvision's antialiased bilinear rounds differently",
    shape_mismatch=0,
    mean_abs=1.0,
    p99_abs=2,
    max_abs=4,
)
_tol(
    ("torchvision",),
    "w2",
    "libjpeg-turbo on both sides; resize rounding",
    count_mismatch=0,
    channel_mismatch=0,
    dims_max=0,
    mean_abs=0.5,
    worst_image_mean_abs=1.0,
)
_tol(
    ("torchvision",),
    "w4",
    "torchvision's antialiased bilinear rounds differently",
    shape_mismatch=0,
    mean_abs=0.5,
    worst_image_mean_abs=0.5,
    max_abs=2,
)

_tol(
    ("daft",),
    "w1",
    "Daft's resize (Rust image crate) uses a different filter support",
    shape_mismatch=0,
    mean_abs=1.5,
    p99_abs=10,
    max_abs=30,
)
_tol(
    ("daft",),
    "w3",
    "Daft's phash follows imagehash; the DCT input resize differs slightly",
    meta_mismatch=0,
    groups_equal=1,
    hash_mean_bits=1.0,
    hash_max_bits=4,
)


def write_report(report: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "parity.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing.update(report)
    path.write_text(json.dumps(dict(sorted(existing.items())), indent=1) + "\n")
    return path
