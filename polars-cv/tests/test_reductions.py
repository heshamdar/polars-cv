"""The reductions, against NumPy: one ``Reduction`` authority for every reducer.

``label_reduce`` (a contour region's pixels), ``reduce_masked`` (the pixels
a mask selects) and ``statistics()`` all reduce with the same definitions:
``sum``, ``mean``, ``min``, ``max``, ``std`` (population), ``median``,
``percentile`` (NumPy's linear interpolation, with ``q``) and ``count``. Over
no pixels each is NaN, except ``sum`` and ``count`` (0).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

from .conftest import plugin_required

pytestmark = plugin_required

H, W = 8, 10
IMG = np.random.default_rng(7).integers(0, 100, (H, W)).astype(np.float64)

#: Reduction name → NumPy over the selected values.
NUMPY = {
    "sum": np.sum,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "std": np.std,
    "median": np.median,
    "count": lambda v: float(v.size),
}


def _square(x0: float, y0: float, x1: float, y1: float) -> dict:
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return {
        "exterior": [{"x": x, "y": y} for x, y in pts],
        "holes": [],
        "is_closed": True,
    }


#: Two regions, as bboxes: their pixels are plain slices of the image.
REGIONS = [_square(2, 1, 6, 5), _square(0, 4, 3, 8)]
SLICES = [IMG[1:5, 2:6], IMG[4:8, 0:3]]


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"a": pl.Series("a", IMG[np.newaxis]), "c": [REGIONS]},
        schema_overrides={"c": CONTOUR_SET_SCHEMA},
    )


class TestLabelReduce:
    @pytest.mark.parametrize("reduction", sorted(NUMPY))
    def test_every_reduction_matches_numpy(self, reduction: str) -> None:
        pipe = (
            Pipeline()
            .source("array")
            .label_reduce(pl.col("c"), reduction=reduction, region_mode="bbox")
        )
        got = _frame().select(pl.col("a").cv.pipe(pipe).sink("native")).item()
        np.testing.assert_allclose(got, [NUMPY[reduction](s) for s in SLICES])

    @pytest.mark.parametrize("q", [0.0, 10.0, 62.5, 100.0])
    def test_percentile_matches_numpy(self, q: float) -> None:
        pipe = (
            Pipeline()
            .source("array")
            .label_reduce(pl.col("c"), reduction="percentile", q=q, region_mode="bbox")
        )
        got = _frame().select(pl.col("a").cv.pipe(pipe).sink("native")).item()
        np.testing.assert_allclose(got, [np.percentile(s, q) for s in SLICES])

    def test_the_contour_accessor_takes_the_same_reductions(self) -> None:
        got = (
            _frame()
            .select(
                pl.col("c").contour.label_reduce(
                    pl.col("a"), reduction="percentile", q=90.0, region_mode="bbox"
                )
            )
            .item()
        )
        np.testing.assert_allclose(got, [np.percentile(s, 90.0) for s in SLICES])

    def test_percentile_needs_q(self) -> None:
        with pytest.raises(ValueError, match="q"):
            Pipeline().source("array").label_reduce(pl.col("c"), reduction="percentile")

    def test_q_without_percentile_is_refused(self) -> None:
        with pytest.raises(ValueError, match="q"):
            Pipeline().source("array").label_reduce(
                pl.col("c"), reduction="mean", q=5.0
            )


def _reduce(pipe: Pipeline) -> np.ndarray:
    out = _frame().select(pl.col("a").cv.pipe(pipe).sink("numpy")).item()
    from polars_cv import numpy_from_struct

    return numpy_from_struct(out)


class TestAxes:
    @pytest.mark.parametrize("axis", [0, 1])
    def test_reduce_sum_along_an_axis(self, axis: int) -> None:
        got = _reduce(Pipeline().source("array").reduce_sum(axis=axis))
        np.testing.assert_allclose(got, IMG.sum(axis=axis))

    @pytest.mark.parametrize(("q", "axis"), [(10.0, 0), (75.0, 1), (100.0, 1)])
    def test_reduce_percentile_along_an_axis(self, q: float, axis: int) -> None:
        got = _reduce(Pipeline().source("array").reduce_percentile(q, axis=axis))
        np.testing.assert_allclose(got, np.percentile(IMG, q, axis=axis))

    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_reduce_median(self, axis: int | None) -> None:
        pipe = Pipeline().source("array").reduce_median(axis=axis)
        if axis is None:
            got = _frame().select(pl.col("a").cv.pipe(pipe).sink("native")).item()
        else:
            got = _reduce(pipe)
        np.testing.assert_allclose(got, np.median(IMG, axis=axis))


RNG = np.random.default_rng(11)
MASK = np.where(RNG.random((H, W)) < 0.5, 255, 0).astype(np.uint8)
MASK[:, 3] = 0  # a column with no masked pixel
RGB = RNG.integers(0, 100, (H, W, 3)).astype(np.float64)


def _masked_frame(mask: np.ndarray = MASK, img: np.ndarray = IMG) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": pl.Series("a", img[np.newaxis]),
            "m": pl.Series("m", mask[np.newaxis]),
        }
    )


def _masked(reduction: str, axis: int | None = None, q: float | None = None, **frame):
    img = pl.col("a").cv.pipe(Pipeline().source("array"))
    mask = pl.col("m").cv.pipe(Pipeline().source("array"))
    expr = img.reduce_masked(mask, reduction=reduction, q=q, axis=axis)
    out = _masked_frame(**frame).select(
        expr.sink("native" if axis is None else "numpy")
    )
    if axis is None:
        return out.item()
    from polars_cv import numpy_from_struct

    return numpy_from_struct(out.item())


def _numpy_masked(
    values: np.ndarray, keep: np.ndarray, reduction: str, q=None
) -> float:
    selected = values[keep]
    if reduction == "count":
        return float(selected.size)
    if reduction == "sum":
        return float(selected.sum())
    if selected.size == 0:
        return float("nan")
    if reduction == "percentile":
        return float(np.percentile(selected, q))
    return float(NUMPY[reduction](selected))


class TestMasked:
    @pytest.mark.parametrize("reduction", sorted(NUMPY))
    def test_global_matches_numpy(self, reduction: str) -> None:
        got = _masked(reduction)
        np.testing.assert_allclose(got, _numpy_masked(IMG, MASK > 0, reduction))

    @pytest.mark.parametrize("reduction", [*sorted(NUMPY), "percentile"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_along_an_axis_matches_numpy(self, reduction: str, axis: int) -> None:
        q = 75.0 if reduction == "percentile" else None
        got = _masked(reduction, axis=axis, q=q)
        keep = MASK > 0
        lines = range(W) if axis == 0 else range(H)
        want = [
            _numpy_masked(
                IMG[:, i] if axis == 0 else IMG[i],
                keep[:, i] if axis == 0 else keep[i],
                reduction,
                q,
            )
            for i in lines
        ]
        np.testing.assert_allclose(got, want)

    def test_a_line_with_no_masked_pixel_is_nan_and_counts_zero(self) -> None:
        assert np.isnan(_masked("median", axis=0)[3])
        assert _masked("count", axis=0)[3] == 0
        assert _masked("sum", axis=0)[3] == 0

    def test_a_float_mask_selects_its_positive_pixels(self) -> None:
        soft = np.where(MASK > 0, 0.25, 0.0)
        got = _masked("mean", mask=soft)
        np.testing.assert_allclose(got, IMG[MASK > 0].mean())

    def test_channels_are_reduced_together_or_kept_along_an_axis(self) -> None:
        keep = MASK > 0
        np.testing.assert_allclose(_masked("max", img=RGB), RGB[keep].max())
        got = _masked("mean", axis=1, img=RGB)
        assert got.shape == (H, 3)
        np.testing.assert_allclose(
            got[0], RGB[0][keep[0]].mean(axis=0) if keep[0].any() else np.nan
        )

    def test_another_mask_dtype_is_refused(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError, match="mask"):
            _masked("mean", mask=MASK.astype(np.int32))

    def test_a_mask_of_another_size_is_refused(self) -> None:
        with pytest.raises(pl.exceptions.ComputeError):
            _masked("mean", mask=MASK[:, :4])

    def test_an_axis_past_the_image_plane_is_refused(self) -> None:
        with pytest.raises(ValueError, match="axis"):
            _masked("mean", axis=2, img=RGB)
