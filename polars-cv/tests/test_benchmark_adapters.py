"""The benchmark adapters compute the same operation on both sides.

A benchmark that times OpenCV doing one thing and polars-cv another measures
nothing, and the suite's output validator then reports the gap as a
correctness failure. Several single-op benchmarks did exactly that: ``resize``
ran ``INTER_AREA`` against bilinear, ``sharpen`` an unsharp mask against a 3x3
kernel, ``erode``/``dilate`` thresholded on one side only, ``adjust_contrast``
and ``adjust_brightness`` truncated to u8 against a float result, and
``rotate`` truncated its expanded size and so rotated 90° non-square images
into the wrong shape.

For each reference library (OpenCV, Pillow) every single-op benchmark is
listed in ``TOLERANCE`` with the largest per-pixel difference allowed between
that library's adapter and each polars-cv adapter, and why it is not 0 — or in
``UNSUPPORTED``, when the library has no call that computes the op, where its
adapter must raise ``NotImplementedError`` rather than time something else.
An op missing from both fails.

Ops that begin with a grayscale conversion run on a gray image stored as RGB,
so that conversion is exact on both sides and the op itself is what is
compared; ``grayscale`` is compared on colour, at its own tolerance.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from benchmarks.frameworks import get_adapter
from benchmarks.scenarios.single_ops import get_single_op_benchmarks
from PIL import Image

from tests.conftest import plugin_required

cv2 = pytest.importorskip("cv2")

pytestmark = plugin_required

H, W = 96, 128

_EXACT = dict.fromkeys(
    ["flip_horizontal", "flip_vertical", "crop_center", "invert", "pad", "threshold"]
    + ["erode", "dilate"],
    0.0,
)

#: Largest allowed |reference - polars-cv| per pixel, in the op's output units.
TOLERANCE: dict[str, dict[str, float]] = {
    "opencv": {
        **_EXACT,
        "rotate_90": 0,  # a lattice rotation is a permutation on both sides
        "histogram_equalize": 0,
        "canny": 0,
        "sobel_x": 0,
        "sharpen": 0,
        "normalize": 1e-6,
        "adjust_contrast": 1e-3,  # f32 arithmetic in both
        "adjust_brightness": 1e-3,
        # OpenCV converts with 14-bit fixed-point weights, polars-cv with 8-bit.
        "grayscale": 1,
        # OpenCV's u8 Gaussian runs a fixed-point kernel.
        "blur": 1,
        # Inside the rotated content; its rim is `RIM_TOLERANCE`'s.
        "rotate_45": 1,
        # polars-cv's bilinear antialiases a downscale (as Pillow does); OpenCV's
        # INTER_LINEAR does not, and OpenCV has no antialiased bilinear. On
        # this smooth image the two differ by a few levels; against Pillow the
        # same resize agrees to 1.
        "resize": 4,
    },
    "pillow": {
        **_EXACT,
        "rotate_90": 0,
        "normalize": 1e-6,
        "adjust_contrast": 1e-3,
        "adjust_brightness": 1e-3,
        "resize": 1,
        # Pillow's "L" conversion rounds ITU-R 601 weights over 1000.
        "grayscale": 1,
        # Pillow approximates a Gaussian with repeated box blurs.
        "blur": 2,
        # Pillow's equalize builds its lookup table with a different rounding
        # rule from OpenCV's (which polars-cv matches exactly).
        "histogram_equalize": 2,
        "rotate_45": 1,
    },
}

#: Ops a reference library has no call for: its adapter raises.
UNSUPPORTED: dict[str, set[str]] = {
    "opencv": set(),
    "pillow": {"canny", "sobel_x", "sharpen"},
}

#: Larger tolerance on the rim of a resampled op's content, where the
#: libraries treat the zero border differently: OpenCV 4.x's fixed-point
#: warpAffine blends with it (up to 4 levels here), Pillow does not blend at
#: all (a rim pixel is sampled or filled, so any value can differ).
RIM_TOLERANCE: dict[str, dict[str, float]] = {
    "opencv": {"rotate_45": 4},
    "pillow": {"rotate_45": 255},
}

_GRAY_FIRST = {"threshold", "erode", "dilate", "histogram_equalize", "canny", "sobel_x"}


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _smooth_colour() -> np.ndarray:
    rng = np.random.default_rng(5)
    small = rng.integers(0, 256, (H // 8, W // 8, 3)).astype(np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


COLOUR = _smooth_colour()
GRAY_AS_RGB = np.repeat(cv2.cvtColor(COLOUR, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)

BENCHMARKS = {b.name: b for b in get_single_op_benchmarks(H, W)}


def _run(adapter_name: str, name: str) -> np.ndarray:
    adapter = get_adapter(adapter_name)
    img = GRAY_AS_RGB if name in _GRAY_FIRST else COLOUR
    out = adapter.run_pipeline_batch([_png(img)], [BENCHMARKS[name].params])[0]
    arr = np.asarray(adapter.to_numpy(out))
    return arr[..., 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr


def _content_rim(name: str) -> np.ndarray:
    """Pixels within one pixel of where the op's output stops being image.

    The OpenCV adapter runs the op on an all-white image; wherever that is not
    white in a pixel's 3x3 neighbourhood, the pixel blends with the border.
    """
    adapter = get_adapter("opencv")
    white = np.full_like(COLOUR, 255)
    out = adapter.run_pipeline_batch([_png(white)], [BENCHMARKS[name].params])[0]
    full = np.asarray(adapter.to_numpy(out)).min(axis=2) == 255
    inner = cv2.erode(full.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return np.repeat(~inner[..., None], 3, axis=2)


REFERENCES = sorted(TOLERANCE)
_COMPARED = [(ref, name) for ref in REFERENCES for name in sorted(TOLERANCE[ref])]


@pytest.mark.parametrize("ref", REFERENCES)
def test_every_single_op_benchmark_is_accounted_for(ref: str) -> None:
    assert set(TOLERANCE[ref]).isdisjoint(UNSUPPORTED[ref])
    assert set(TOLERANCE[ref]) | UNSUPPORTED[ref] == set(BENCHMARKS)


@pytest.mark.parametrize(
    ("ref", "name"), [(r, n) for r in REFERENCES for n in sorted(UNSUPPORTED[r])]
)
def test_an_unsupported_op_raises_rather_than_timing_another(
    ref: str, name: str
) -> None:
    with pytest.raises(NotImplementedError):
        _run(ref, name)


@pytest.mark.parametrize("engine", ["polars-cv-eager", "polars-cv-streaming"])
@pytest.mark.parametrize(("ref", "name"), _COMPARED)
def test_the_reference_and_polars_cv_compute_the_same_op(
    ref: str, name: str, engine: str
) -> None:
    expected = _run(ref, name)
    got = _run(engine, name)
    assert got.shape == expected.shape, f"{name}: {got.shape} vs {expected.shape}"
    diff = np.abs(got.astype(np.float64) - expected.astype(np.float64))
    if name in RIM_TOLERANCE[ref]:
        rim = _content_rim(name)
        assert diff[rim].max() <= RIM_TOLERANCE[ref][name], (
            f"{name}: rim {diff[rim].max()}"
        )
        diff = diff[~rim]
    tol = TOLERANCE[ref][name]
    assert diff.max() <= tol, (
        f"{ref} vs {engine}, {name}: max |diff| {diff.max()} > {tol} "
        f"({(diff > tol).mean():.1%} of values)"
    )


@pytest.mark.parametrize("ref", REFERENCES)
def test_rotating_a_non_square_image_by_90_swaps_its_sides(ref: str) -> None:
    assert _run(ref, "rotate_90").shape[:2] == (W, H)
