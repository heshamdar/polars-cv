"""Reference tests for the disk footprint of ``erode``/``dilate``/
``morphology_gradient``/``morphology_open``/``morphology_close`` (CR-94).

``footprint="disk"`` takes the minimum (or maximum) over the pixels within
Euclidean distance ``r = ksize // 2`` of the centre, edges replicated, as
``footprint="square"`` does over the ``ksize x ksize`` square. Three
independent oracles:

* a NumPy sliding window masked by the disk (any dtype, NaN included);
* ``scipy.ndimage.grey_erosion`` / ``grey_dilation`` with that footprint and
  ``mode="nearest"`` (u8: scipy computes through float64);
* on a binary mask, the distance transform: a pixel is set after a disk
  dilation iff a set pixel lies within ``r``, and after a disk erosion iff no
  unset pixel does. Exact at the border too, because replicating an edge never
  brings a pixel closer than its unclamped offset.

``iterations=n`` applies the disk ``n`` times. Unlike squares, two disks do
not compose into one larger disk, so this is not a single radius-``n r`` pass.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import ndimage

from polars_cv import Pipeline, numpy_from_column
from tests.conftest import plugin_required

pytestmark = plugin_required

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


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return yy * yy + xx * xx <= r * r


def _window(plane: np.ndarray, r: int, reduce) -> np.ndarray:
    """``reduce`` over the disk of radius ``r`` around each pixel, edges
    replicated; NumPy's reductions propagate NaN."""
    padded = np.pad(plane, r, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (2 * r + 1, 2 * r + 1))
    return reduce(windows[..., _disk(r)], axis=-1).astype(plane.dtype)


def _ref(plane: np.ndarray, ksize: int, iterations: int, op: str) -> np.ndarray:
    r = ksize // 2
    out = plane
    for _ in range(iterations):
        out = _window(out, r, np.min if op == "erode" else np.max)
    return out


def _run(image: np.ndarray, build) -> np.ndarray:
    df = pl.DataFrame(
        {"m": [image]}, schema={"m": pl.Array(_POLARS[image.dtype], image.shape)}
    )
    out = df.select(
        pl.col("m").cv.pipe(build(Pipeline().source("array"))).sink("numpy")
    )
    (result,) = numpy_from_column(out.to_series(), copy=True)
    assert result is not None
    return result


def _image(dtype, shape=(23, 31), seed=94) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return (rng.random(shape) * 200 - 100).astype(dtype)
    info = np.iinfo(dtype)
    lo, hi = max(info.min, -1000), min(info.max, 1000)
    return rng.integers(lo, hi, size=shape, endpoint=True).astype(dtype)


@pytest.mark.parametrize("dtype", list(_POLARS), ids=lambda d: d.name)
@pytest.mark.parametrize("op", ["erode", "dilate"])
@pytest.mark.parametrize(
    ("ksize", "iterations"), [(1, 1), (3, 1), (5, 1), (7, 2), (9, 1)]
)
@pytest.mark.parity(
    ops=["erode", "dilate"],
    against="NumPy sliding window `np.min`/`np.max` over the disk, edges replicated",
    tolerance="exact",
)
def test_disk_matches_the_masked_window(dtype, op, ksize, iterations) -> None:
    image = _image(dtype)
    got = _run(
        image,
        lambda p: getattr(p, op)(ksize=ksize, iterations=iterations, footprint="disk"),
    )
    np.testing.assert_array_equal(got, _ref(image, ksize, iterations, op))


@pytest.mark.parametrize("ksize", [3, 5, 7, 11])
@pytest.mark.parity(
    ops=["erode", "dilate"],
    against="SciPy `scipy.ndimage.grey_erosion`/`grey_dilation`, disk footprint, nearest",
    tolerance="exact",
)
def test_disk_matches_scipy(ksize: int) -> None:
    image = _image(np.uint8, (40, 37), seed=ksize)
    foot = _disk(ksize // 2)
    for op, scipy_op in (
        ("erode", ndimage.grey_erosion),
        ("dilate", ndimage.grey_dilation),
    ):
        got = _run(
            image, lambda p, op=op: getattr(p, op)(ksize=ksize, footprint="disk")
        )
        np.testing.assert_array_equal(
            got, scipy_op(image, footprint=foot, mode="nearest"), err_msg=op
        )


@pytest.mark.parametrize("density", [0.02, 0.2, 0.6, 0.97])
@pytest.mark.parametrize("ksize", [3, 5, 9, 15])
@pytest.mark.parity(
    ops=["erode", "dilate"],
    against="SciPy `scipy.ndimage.distance_transform_edt` thresholded at the disk radius",
    tolerance="exact",
)
def test_disk_on_a_mask_is_the_distance_transform_thresholded(density, ksize) -> None:
    rng = np.random.default_rng(int(density * 100) + ksize)
    mask = (rng.random((29, 34)) < density).astype(np.uint8) * 255
    r = ksize // 2
    dilated = _run(mask, lambda p: p.dilate(ksize=ksize, footprint="disk"))
    eroded = _run(mask, lambda p: p.erode(ksize=ksize, footprint="disk"))
    to_set = ndimage.distance_transform_edt(mask == 0)  # to the nearest set pixel
    to_unset = ndimage.distance_transform_edt(mask != 0)  # to the nearest unset one
    if (mask != 0).any():
        np.testing.assert_array_equal(dilated != 0, to_set <= r)
    if (mask == 0).any():
        np.testing.assert_array_equal(eroded != 0, to_unset > r)


@pytest.mark.parity(
    ops=["erode"],
    against="NumPy sliding window `np.min` over the disk (NaN propagates)",
    tolerance="exact (NaN mask)",
)
def test_disk_nan_is_the_window_result() -> None:
    image = _image(np.float32, (9, 9))
    image[4, 4] = np.nan
    got = _run(image, lambda p: p.erode(ksize=5, footprint="disk"))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(_ref(image, 5, 1, "erode")))
    # The disk of radius 2 does not reach the square's corners.
    assert not np.isnan(got[2, 2]) and np.isnan(got[2, 4]) and np.isnan(got[3, 3])


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.int16, np.float64], ids=lambda d: np.dtype(d).name
)
@pytest.mark.parity(
    ops=["morphology_gradient", "morphology_open", "morphology_close"],
    against="NumPy disk window erode/dilate, composed per the definition of each op",
    tolerance="exact",
)
def test_gradient_open_and_close_take_the_disk(dtype) -> None:
    image = _image(dtype)
    k = 5
    ero = _ref(image, k, 1, "erode")
    dil = _ref(image, k, 1, "dilate")
    if np.dtype(dtype).kind == "f":
        gradient = np.maximum(dil - ero, 0)
    else:
        gradient = np.clip(dil.astype(np.int64) - ero, 0, np.iinfo(dtype).max).astype(
            dtype
        )
    np.testing.assert_array_equal(
        _run(image, lambda p: p.morphology_gradient(ksize=k, footprint="disk")),
        gradient,
    )
    np.testing.assert_array_equal(
        _run(image, lambda p: p.morphology_open(ksize=k, footprint="disk")),
        _ref(ero, k, 1, "dilate"),
    )
    np.testing.assert_array_equal(
        _run(image, lambda p: p.morphology_close(ksize=k, footprint="disk")),
        _ref(dil, k, 1, "erode"),
    )


@pytest.mark.parity(
    ops=["erode", "dilate", "morphology_gradient"],
    against='Definition: `footprint="square"` is the default footprint',
    tolerance="exact",
)
def test_square_is_the_default_and_unchanged() -> None:
    image = _image(np.uint8)
    for op in ("erode", "dilate", "morphology_gradient"):
        default = _run(image, lambda p, op=op: getattr(p, op)(ksize=5))
        square = _run(
            image, lambda p, op=op: getattr(p, op)(ksize=5, footprint="square")
        )
        np.testing.assert_array_equal(default, square, err_msg=op)


@pytest.mark.parity(
    ops=["erode"],
    against="Definition: a footprint other than disk or square is refused",
    tolerance="exact (raises)",
)
def test_an_unknown_footprint_is_refused() -> None:
    with pytest.raises(Exception, match="footprint|disk|square"):
        Pipeline().source("image_bytes").grayscale().erode(ksize=3, footprint="cross")
