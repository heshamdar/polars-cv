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
