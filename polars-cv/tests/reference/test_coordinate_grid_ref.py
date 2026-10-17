"""Reference tests for ``coordinate_grid(normalized=)`` (CR-100).

The f32 ``[H, W, 2]`` grid of each pixel's centre ``(x + 0.5, y + 0.5)`` (the
centres ``rasterize`` samples), divided by ``W`` and ``H`` when ``normalized``,
for a buffer of any dtype and channel count: it reads the input's height and
width, never its values. Coordinates are in the frame of the buffer it
follows, so a crop before it moves the origin and one after it slices the
grid. Oracle: ``np.meshgrid``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_column
from tests._plan_view import planned
from tests.conftest import plugin_required

pytestmark = plugin_required

_POLARS = {np.dtype(np.uint8): pl.UInt8, np.dtype(np.float32): pl.Float32}


def _oracle(h: int, w: int, normalized: bool = False) -> np.ndarray:
    xs, ys = np.arange(w) + 0.5, np.arange(h) + 0.5
    if normalized:
        xs, ys = xs / w, ys / h
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1).astype(np.float32)


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


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize(
    "shape", [(5, 7), (5, 7, 1), (6, 4, 3), (1, 9, 4), (8, 1)], ids=str
)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_matches_meshgrid(shape, dtype, normalized: bool) -> None:
    image = np.random.default_rng(0).integers(0, 255, shape).astype(dtype)
    got = _run(image, lambda p: p.coordinate_grid(normalized=normalized))
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, _oracle(shape[0], shape[1], normalized))


def test_the_default_is_pixel_units() -> None:
    image = np.zeros((3, 4), np.uint8)
    np.testing.assert_array_equal(
        _run(image, lambda p: p.coordinate_grid()), _oracle(3, 4)
    )


def test_values_do_not_matter_and_a_view_is_read_where_it_lies() -> None:
    image = np.random.default_rng(1).integers(0, 255, (6, 5, 3)).astype(np.uint8)
    flipped = _run(image, lambda p: p.flip_h().coordinate_grid())
    np.testing.assert_array_equal(flipped, _oracle(6, 5))


def test_coordinates_are_in_the_frame_of_the_buffer_they_follow() -> None:
    image = np.zeros((10, 12), np.uint8)
    # A crop first: the grid is the crop's own, origin at its corner.
    before = _run(
        image, lambda p: p.crop(top=2, left=3, height=4, width=5).coordinate_grid()
    )
    np.testing.assert_array_equal(before, _oracle(4, 5))
    # A crop after: the full grid, sliced.
    after = _run(
        image, lambda p: p.coordinate_grid().crop(top=2, left=3, height=4, width=5)
    )
    np.testing.assert_array_equal(after, _oracle(10, 12)[2:6, 3:8])


def test_the_plan_knows_the_shape_and_dtype() -> None:
    pipe = Pipeline().source("image_bytes").resize(height=5, width=7).coordinate_grid()
    assert planned(pipe).dims == (5, 7, 2)
    assert pipe.output_dtype() == "f32"


def test_a_per_row_normalized_flag() -> None:
    image = np.zeros((2, 4), np.uint8)
    df = pl.DataFrame(
        {"m": [image, image], "n": [False, True]},
        schema={"m": pl.Array(pl.UInt8, (2, 4)), "n": pl.Boolean},
    )
    pipe = Pipeline().source("array").coordinate_grid(normalized=pl.col("n"))
    out = numpy_from_column(
        df.select(pl.col("m").cv.pipe(pipe).sink("numpy")).to_series(), copy=True
    )
    np.testing.assert_array_equal(out[0], _oracle(2, 4))
    np.testing.assert_array_equal(out[1], _oracle(2, 4, normalized=True))


def test_a_field_over_positions_composes_from_it() -> None:
    """What it is for: a field over pixel positions, from ordinary ops."""
    image = np.zeros((9, 9), np.uint8)
    pipe = Pipeline().source("array").coordinate_grid().add_constant(-4.5).square()
    got = _run(image, lambda _p: pipe)
    np.testing.assert_allclose(got, (_oracle(9, 9) - 4.5) ** 2, rtol=1e-6)
