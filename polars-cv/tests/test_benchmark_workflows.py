"""Each library's implementation of a multi-branch workflow computes what the
polars-cv one does.

The workflows in ``benchmarks/scenarios/workflows.py`` are written once per
library, in that library's idiom, so nothing but this test holds them to the
same result. For every (library, workflow) pair ``TOLERANCE`` gives the
largest allowed difference per output, and why it is not 0, or the pair is in
``UNSUPPORTED`` and the library has no implementation to time. A workflow
missing from both fails.
"""

from __future__ import annotations

import numpy as np
import pytest
from benchmarks.frameworks import get_adapter
from benchmarks.scenarios.workflows import WORKFLOWS, run_library, run_polars_cv
from benchmarks.utils.data_gen import generate_image_set

from tests.conftest import plugin_required

pytest.importorskip("cv2")

pytestmark = plugin_required

LIBRARIES = ["opencv", "pillow", "pyvips"]

#: Per workflow, per output: the largest allowed |library - polars-cv|; for
#: ``total_area`` a fraction of polars-cv's value.
#: The letterbox agrees to 1-2 levels before min-max scaling (its single-op
#: tolerance), plus f32 rounding in the scaling.
TENSOR = 2.01 / 255

#: The mean grey level: grayscale weights round differently by 1 level on
#: some pixels; under a mask, those near the threshold also flip.
MEAN = 0.1

TOLERANCE: dict[str, dict[str, dict[str, float]]] = {
    "opencv": {
        "multi_output_etl": {
            "tensor": TENSOR,
            # INTER_AREA (OpenCV's antialiased downscale) is a box filter;
            # polars-cv's antialiased bilinear is a triangle.
            "thumb": 3,
            "mean": MEAN,
        },
        # OpenCV blurs in u8 (rounded) before the weighted sum.
        "unsharp_mask": {"image": 1},
        "masked_stats": {"mean": MEAN, "max": 0},
        "mask_to_contours": {
            "count": 0,
            # OpenCV traces pixel centres, polars-cv pixel edges (its area is
            # the pixel count), so every contour loses about half its
            # perimeter in area.
            "total_area": 0.1,
        },
    },
    "pillow": {
        "multi_output_etl": {"tensor": TENSOR, "thumb": 1, "mean": MEAN},
        # Pillow's Gaussian is three box blurs.
        "unsharp_mask": {"image": 3},
        "masked_stats": {"mean": MEAN, "max": 0},
    },
    "pyvips": {
        "multi_output_etl": {
            "tensor": TENSOR,
            # libvips extends the edge on a shrink where polars-cv
            # renormalises its clipped kernel.
            "thumb": 1,
            "mean": MEAN,
        },
        # libvips's default u8 Gaussian is fixed point (see the single-op
        # `blur` row in test_benchmark_adapters.py).
        "unsharp_mask": {"image": 4},
        "masked_stats": {"mean": MEAN, "max": 0},
    },
}

#: Workflows a library has no call for, so it has no implementation.
UNSUPPORTED: dict[str, set[str]] = {
    "opencv": set(),
    # Neither traces contours.
    "pillow": {"mask_to_contours"},
    "pyvips": {"mask_to_contours"},
}

#: Image outputs compared away from a rim this wide: the blur's radius,
#: ``ceil(3 sigma)``, inside which each library extends the border its own way.
INTERIOR = {"unsharp_mask": 6}

_PAIRS = [(lib, wf) for lib in LIBRARIES for wf in sorted(TOLERANCE[lib])]


def _images(height: int = 96, width: int = 128) -> list[bytes]:
    """Smooth images (as the single-op parity test uses), so the workflow is
    what is compared rather than how two filters treat pixel noise: OpenCV
    has no antialiased bilinear, and on noise a box filter and a triangle
    differ by tens of levels."""
    return generate_image_set(
        count=4, height=height, width=width, channels=3, pattern="blobs"
    ).image_bytes


@pytest.mark.parametrize("lib", LIBRARIES)
def test_every_workflow_is_accounted_for(lib: str) -> None:
    supported = {name for name, wf in WORKFLOWS.items() if lib in wf.libraries}
    assert supported == set(TOLERANCE[lib])
    assert set(TOLERANCE[lib]) | UNSUPPORTED[lib] == set(WORKFLOWS)


@pytest.mark.parametrize("streaming", [False, True], ids=["eager", "streaming"])
@pytest.mark.parametrize(("lib", "name"), _PAIRS)
def test_the_library_and_polars_cv_compute_the_same_workflow(
    lib: str, name: str, streaming: bool
) -> None:
    workflow = WORKFLOWS[name]
    images = _images()
    expected = run_polars_cv(workflow, images, streaming=streaming)
    got = run_library(workflow, get_adapter(lib), images)
    assert len(got) == len(expected)
    for row, (g, e) in enumerate(zip(got, expected, strict=True)):
        assert set(g) == set(e) == set(TOLERANCE[lib][name]), row
        for key, tol in TOLERANCE[lib][name].items():
            ga = np.asarray(g[key], dtype=np.float64)
            ea = np.asarray(e[key], dtype=np.float64)
            assert ga.shape == ea.shape, f"{name}/{key}: {ga.shape} vs {ea.shape}"
            diff = np.abs(ga - ea)
            rim = INTERIOR.get(name, 0)
            if rim and diff.ndim >= 2:
                diff = diff[rim:-rim, rim:-rim]
            if key == "total_area":
                diff = diff / max(float(ea), 1.0)
            assert diff.max() <= tol, (
                f"{lib} {name} row {row} {key}: max |diff| {diff.max()} > {tol}"
            )


def test_mask_to_contours_finds_many_regions() -> None:
    """The workflow's input must give the tracer real work: a gradient
    thresholds to one region, which would time almost nothing."""
    workflow = WORKFLOWS["mask_to_contours"]
    assert workflow.pattern == "blobs"
    images = _images(256, 256)
    counts = [r["count"] for r in run_polars_cv(workflow, images)]
    assert min(counts) >= 5, counts
