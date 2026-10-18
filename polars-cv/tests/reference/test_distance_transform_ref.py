"""Reference tests for ``distance_transform`` (CR-93).

Unsigned: each non-zero pixel's Euclidean distance to the nearest zero pixel,
centre to centre, as ``scipy.ndimage.distance_transform_edt`` and
``cv2.distanceTransform(DIST_L2, DIST_MASK_PRECISE)`` compute it. scipy works
in f64, so its distances rounded to f32 are the correctly rounded truth and
equality with it is exact. OpenCV's are not always correctly rounded (its
``sqrt(61)`` is one ulp above the f32 nearest), so it is held to one ulp.

Signed: negative inside (non-zero), positive outside, ``d - 0.5`` from the
centre distance ``d`` to the nearest pixel of the other class, which is the
distance to the pixel-edge boundary ``extract_contours`` traces. It is checked
against the vector ``.point.signed_distance_to_contour`` on that traced
outline: exact wherever the nearest boundary point lies on an edge, within
``(sqrt(2) - 1) / 2`` where it is a corner.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import polars as pl
import pytest
from scipy import ndimage

from polars_cv import Pipeline, numpy_from_column
from polars_cv.geometry.schemas import CONTOUR_SCHEMA
from tests.conftest import plugin_required

pytestmark = plugin_required

_DTYPES = (
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float32,
    np.float64,
)

_POLARS = {
    np.dtype(np.uint8): pl.UInt8,
    np.dtype(np.int8): pl.Int8,
    np.dtype(np.uint16): pl.UInt16,
    np.dtype(np.int16): pl.Int16,
    np.dtype(np.uint32): pl.UInt32,
    np.dtype(np.int32): pl.Int32,
    np.dtype(np.uint64): pl.UInt64,
    np.dtype(np.int64): pl.Int64,
    np.dtype(np.float32): pl.Float32,
    np.dtype(np.float64): pl.Float64,
}


def _run(image: np.ndarray, *, signed: bool = False) -> np.ndarray:
    """``distance_transform`` of one ``[H, W]`` or ``[H, W, 1]`` array."""
    df = pl.DataFrame(
        {"m": [image]},
        schema={"m": pl.Array(_POLARS[image.dtype], image.shape)},
    )
    pipe = Pipeline().source("array").distance_transform(signed=signed)
    out = df.select(pl.col("m").cv.pipe(pipe).sink("numpy"))
    (result,) = numpy_from_column(out.to_series(), copy=True)
    assert result is not None
    return result


def _scipy_unsigned(mask: np.ndarray) -> np.ndarray:
    return ndimage.distance_transform_edt(mask != 0).astype(np.float32)


def _scipy_signed(mask: np.ndarray) -> np.ndarray:
    inside = mask != 0
    d_in = ndimage.distance_transform_edt(inside)
    d_out = ndimage.distance_transform_edt(~inside)
    return np.where(inside, -(d_in - 0.5), d_out - 0.5).astype(np.float32)


def _masks() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(88)
    cases = [
        (f"noise{d}", (rng.random((37, 53)) < d).astype(np.uint8) * 255)
        for d in (0.05, 0.5, 0.95)
    ]
    single = np.ones((9, 11), np.uint8)
    single[4, 5] = 0
    corner = np.ones((6, 7), np.uint8)
    corner[0, 0] = 0
    rect = np.zeros((20, 30), np.uint8)
    rect[4:15, 6:25] = 1
    row = np.array([[0, 5, 7, 1, 1, 0, 3]], np.uint8)
    column = row.T.copy()
    return [
        *cases,
        ("single zero", single),
        ("corner zero", corner),
        ("rectangle", rect),
        ("one row", row),
        ("one column", column),
    ]


@pytest.mark.parametrize(
    ("name", "mask"), _masks(), ids=lambda v: v if isinstance(v, str) else ""
)
@pytest.mark.parity(
    ops=["distance_transform"],
    against="SciPy `scipy.ndimage.distance_transform_edt`; OpenCV `cv2.distanceTransform` (DIST_L2, precise)",
    tolerance="SciPy: exact (f32); OpenCV: within 1 ulp",
)
def test_unsigned_matches_scipy_and_opencv(name: str, mask: np.ndarray) -> None:
    got = _run(mask)
    assert got.dtype == np.float32
    assert got.shape == mask.shape
    np.testing.assert_array_equal(got, _scipy_unsigned(mask))
    cv = cv2.distanceTransform(
        (mask != 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    np.testing.assert_array_max_ulp(got, cv, maxulp=1)


@pytest.mark.parametrize(
    ("name", "mask"), _masks(), ids=lambda v: v if isinstance(v, str) else ""
)
@pytest.mark.parity(
    ops=["distance_transform"],
    against="SciPy `scipy.ndimage.distance_transform_edt` composition (d_out - 0.5 outside, -(d_in - 0.5) inside)",
    tolerance="exact (f32)",
)
def test_signed_matches_the_scipy_composition(name: str, mask: np.ndarray) -> None:
    if (mask != 0).all() or (mask == 0).all():
        pytest.skip("one class only: covered by the infinity tests")
    np.testing.assert_array_equal(_run(mask, signed=True), _scipy_signed(mask))


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parity(
    ops=["distance_transform"],
    against="SciPy `scipy.ndimage.distance_transform_edt`, all 10 numeric dtypes, unsigned and signed",
    tolerance="exact (f32)",
)
def test_foreground_is_every_non_zero_value_in_every_dtype(dtype) -> None:
    """Negative values are foreground too, and NaN (non-zero) is."""
    values = np.array([[0, 1, 2, 0], [3, 0, 4, 5], [6, 7, 0, 8]], dtype=np.float64)
    if np.dtype(dtype).kind == "i":
        values[1, 2] = -4
    if np.dtype(dtype).kind == "f":
        values[2, 3] = np.nan
    image = values.astype(dtype)
    np.testing.assert_array_equal(_run(image), _scipy_unsigned(image))
    np.testing.assert_array_equal(_run(image, signed=True), _scipy_signed(image))


@pytest.mark.parity(
    ops=["distance_transform"],
    against="SciPy `scipy.ndimage.distance_transform_edt`, [H, W, 1] input",
    tolerance="exact (f32)",
)
def test_a_channel_axis_of_one_is_kept() -> None:
    mask = (np.random.default_rng(1).random((12, 9, 1)) < 0.6).astype(np.uint8)
    got = _run(mask)
    assert got.shape == (12, 9, 1)
    np.testing.assert_array_equal(got[:, :, 0], _scipy_unsigned(mask[:, :, 0]))


@pytest.mark.parity(
    ops=["distance_transform"],
    against="Definition: with no zero pixel the distance is +inf (-inf signed)",
    tolerance="exact",
)
def test_no_background_pixel_is_infinitely_far() -> None:
    """scipy measures to a phantom corner and OpenCV returns 2**64 here;
    the distance to a zero pixel that does not exist is +inf."""
    ones = np.full((4, 5), 7, np.uint8)
    assert np.isposinf(_run(ones)).all()
    assert np.isneginf(_run(ones, signed=True)).all()


@pytest.mark.parity(
    ops=["distance_transform"],
    against="Definition: all-zero image is 0 unsigned and +inf signed",
    tolerance="exact",
)
def test_an_all_background_image_is_zero_unsigned_and_inf_signed() -> None:
    zeros = np.zeros((3, 6), np.uint8)
    np.testing.assert_array_equal(_run(zeros), np.zeros((3, 6), np.float32))
    assert np.isposinf(_run(zeros, signed=True)).all()


def _traced_rectangle() -> tuple[np.ndarray, dict]:
    mask = np.zeros((24, 31), np.uint8)
    mask[5:17, 7:22] = 255
    df = pl.DataFrame({"m": [mask]}, schema={"m": pl.Array(pl.UInt8, mask.shape)})
    traced = df.select(
        pl.col("m")
        .cv.pipe(Pipeline().source("array").extract_contours(mode="external"))
        .sink("native")
    ).item()
    assert len(traced) == 1
    return mask, traced[0]


@pytest.mark.parity(
    ops=["distance_transform", "extract_contours", "point.signed_distance_to_contour"],
    against="Cross-check: vector `.point.signed_distance_to_contour` on the `extract_contours` outline",
    tolerance='"atol 1e-5 on edges; corner overshoot <= (sqrt 2 - 1)/2"',
)
def test_signed_agrees_with_the_vector_distance_to_the_traced_outline() -> None:
    mask, contour = _traced_rectangle()
    got = _run(mask, signed=True)
    h, w = mask.shape
    ys, xs = np.mgrid[0:h, 0:w]
    points = pl.DataFrame({"x": xs.ravel() + 0.5, "y": ys.ravel() + 0.5})
    vector = (
        points.select(
            pl.struct("x", "y").point.signed_distance_to_contour(
                pl.lit(contour, dtype=CONTOUR_SCHEMA)
            )
        )
        .to_series()
        .to_numpy()
        .reshape(h, w)
    )
    # Every inside pixel, and every outside pixel level with a side, has its
    # nearest boundary point on an edge: there the two agree exactly.
    inside = mask != 0
    rows = np.zeros_like(inside)
    rows[5:17, :] = True
    cols = np.zeros_like(inside)
    cols[:, 7:22] = True
    on_an_edge = inside | rows | cols
    np.testing.assert_allclose(got[on_an_edge], vector[on_an_edge], rtol=0, atol=1e-5)
    # Nearest to a corner, the pixel-grid distance overshoots by at most
    # (sqrt(2) - 1) / 2, and never undershoots.
    excess = got[~on_an_edge].astype(np.float64) - vector[~on_an_edge]
    assert excess.min() >= -1e-5
    assert excess.max() <= (math.sqrt(2) - 1) / 2 + 1e-5


@pytest.mark.parity(
    ops=["distance_transform", "grayscale"],
    against="Definition: the plan publishes f32 output of the input rank",
    tolerance="exact",
)
def test_the_plan_publishes_f32_and_the_input_rank() -> None:
    pipe = Pipeline().source("image_bytes").grayscale().distance_transform()
    assert pipe.output_dtype() == "f32"
    df = pl.DataFrame(
        {"m": [np.zeros((3, 4), np.uint8)]}, schema={"m": pl.Array(pl.UInt8, (3, 4))}
    )
    out = df.select(
        pl.col("m")
        .cv.pipe(Pipeline().source("array").distance_transform())
        .sink("array", shape=[3, 4])
    )
    assert out.schema["m"] == pl.Array(pl.Float32, (3, 4))
