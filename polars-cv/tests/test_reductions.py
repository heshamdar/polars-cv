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

    @pytest.mark.parametrize("reduction", sorted(NUMPY))
    def test_a_region_with_no_pixels_is_nan_or_zero(self, reduction: str) -> None:
        """Over no pixels, as every reduction: NaN, but 0 for a sum or a
        count. A region outside the image, and one with no points."""
        empty = {"exterior": [], "holes": [], "is_closed": True}
        frame = pl.DataFrame(
            {
                "a": pl.Series("a", IMG[np.newaxis]),
                "c": [[_square(90, 90, 94, 94), empty]],
            },
            schema_overrides={"c": CONTOUR_SET_SCHEMA},
        )
        pipe = Pipeline().source("array").label_reduce(pl.col("c"), reduction=reduction)
        got = frame.select(pl.col("a").cv.pipe(pipe).sink("native")).item()
        want = 0.0 if reduction in ("sum", "count") else np.nan
        np.testing.assert_array_equal(got, [want, want])

    @pytest.mark.parametrize("q", [0.0, 10.0, 62.5, 100.0])
    def test_percentile_matches_numpy(self, q: float) -> None:
        pipe = (
            Pipeline()
            .source("array")
            .label_reduce(pl.col("c"), reduction=f"p{q:g}", region_mode="bbox")
        )
        got = _frame().select(pl.col("a").cv.pipe(pipe).sink("native")).item()
        np.testing.assert_allclose(got, [np.percentile(s, q) for s in SLICES])

    def test_the_contour_accessor_takes_the_same_reductions(self) -> None:
        got = (
            _frame()
            .select(
                pl.col("c").contour.label_reduce(
                    pl.col("a"), reduction="p90", region_mode="bbox"
                )
            )
            .item()
        )
        np.testing.assert_allclose(got, [np.percentile(s, 90.0) for s in SLICES])

    @pytest.mark.parametrize(
        ("reduction", "match"),
        [
            ("percentile", r"unknown reduction 'percentile'.*p90"),
            ("p150", r"p150.*outside \[0, 100\]"),
            ("p9.999", "two decimal places"),
            ("pmax", "unknown reduction 'pmax'"),
        ],
    )
    def test_a_percentile_is_spelled_pNN(self, reduction: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Pipeline().source("array").label_reduce(pl.col("c"), reduction=reduction)


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

    def test_reduce_percentile_refuses_q_outside_0_100(self) -> None:
        """As every percentile here does; it used to clamp to the max."""
        with pytest.raises(ValueError, match=r"outside \[0, 100\]"):
            Pipeline().source("array").reduce_percentile(150.0)
        pipe = Pipeline().source("array").reduce_percentile(pl.col("q"))
        frame = _frame().with_columns(q=pl.lit(150.0))
        with pytest.raises(pl.exceptions.ComputeError, match=r"outside \[0, 100\]"):
            frame.select(pl.col("a").cv.pipe(pipe).sink("native"))

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


def _masked(reduction: str, axis: int | None = None, **frame):
    img = pl.col("a").cv.pipe(Pipeline().source("array"))
    mask = pl.col("m").cv.pipe(Pipeline().source("array"))
    expr = img.reduce_masked(mask, reduction=reduction, axis=axis)
    out = _masked_frame(**frame).select(
        expr.sink("native" if axis is None else "numpy")
    )
    if axis is None:
        return out.item()
    from polars_cv import numpy_from_struct

    return numpy_from_struct(out.item())


def _numpy_masked(values: np.ndarray, keep: np.ndarray, reduction: str) -> float:
    selected = values[keep]
    if reduction == "count":
        return float(selected.size)
    if reduction == "sum":
        return float(selected.sum())
    if selected.size == 0:
        return float("nan")
    if reduction.startswith("p") and reduction[1:].replace(".", "").isdigit():
        return float(np.percentile(selected, float(reduction[1:])))
    return float(NUMPY[reduction](selected))


class TestMasked:
    @pytest.mark.parametrize("reduction", sorted(NUMPY))
    def test_global_matches_numpy(self, reduction: str) -> None:
        got = _masked(reduction)
        np.testing.assert_allclose(got, _numpy_masked(IMG, MASK > 0, reduction))

    @pytest.mark.parametrize("reduction", [*sorted(NUMPY), "p75"])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_along_an_axis_matches_numpy(self, reduction: str, axis: int) -> None:
        got = _masked(reduction, axis=axis)
        keep = MASK > 0
        lines = range(W) if axis == 0 else range(H)
        want = [
            _numpy_masked(
                IMG[:, i] if axis == 0 else IMG[i],
                keep[:, i] if axis == 0 else keep[i],
                reduction,
            )
            for i in lines
        ]
        np.testing.assert_allclose(got, want)

    def test_a_per_row_reduction_may_mix_percentiles_and_statistics(self) -> None:
        """The percentile is part of the reduction's value, so a column of
        them is one parameter; a separate ``q`` could only fit one kind."""
        frame = pl.concat([_masked_frame(), _masked_frame()]).with_columns(
            r=pl.Series(["p90", "mean"])
        )
        img = pl.col("a").cv.pipe(Pipeline().source("array"))
        mask = pl.col("m").cv.pipe(Pipeline().source("array"))
        got = frame.select(
            img.reduce_masked(mask, reduction=pl.col("r")).sink("native")
        ).to_series()
        np.testing.assert_allclose(
            got.to_list(),
            [_numpy_masked(IMG, MASK > 0, "p90"), _numpy_masked(IMG, MASK > 0, "mean")],
        )

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


class TestStatistics:
    def _img(self):
        return pl.col("a").cv.pipe(Pipeline().source("array"))

    def _mask(self):
        return pl.col("m").cv.pipe(Pipeline().source("array"))

    def test_masked_statistics_match_numpy(self) -> None:
        include = ["mean", "median", "p10", "p90", "count", "std"]
        got = (
            _masked_frame()
            .select(self._img().statistics(include=include, mask=self._mask()))
            .item()
        )
        sel = IMG[MASK > 0]
        want = {
            "mean": sel.mean(),
            "median": np.median(sel),
            "p10": np.percentile(sel, 10),
            "p90": np.percentile(sel, 90),
            "count": float(sel.size),
            "std": sel.std(),
        }
        assert set(got) == set(include)
        for name, value in want.items():
            np.testing.assert_allclose(got[name], value, err_msg=name)

    def test_unmasked_median_and_percentiles(self) -> None:
        got = (
            _masked_frame()
            .select(self._img().statistics(include=["median", "p25", "p62.5"]))
            .item()
        )
        np.testing.assert_allclose(got["median"], np.median(IMG))
        np.testing.assert_allclose(got["p25"], np.percentile(IMG, 25))
        np.testing.assert_allclose(got["p62.5"], np.percentile(IMG, 62.5))

    def test_count_without_a_mask_counts_every_value(self) -> None:
        got = _frame().select(self._img().statistics(include=["count"])).item()
        assert got["count"] == IMG.size

    @pytest.mark.parametrize("name", ["p101", "p", "px", "mode"])
    def test_an_unknown_statistic_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match=f"reduction '{name}'"):
            self._img().statistics(include=[name])


class TestRegionStats:
    def test_several_maps_reduced_per_region(self) -> None:
        from polars_cv import region_stats

        heat = pl.col("a").cv.pipe(Pipeline().source("array"))
        other = pl.col("b").cv.pipe(Pipeline().source("array"))
        frame = _frame().with_columns(b=pl.Series("b", (IMG * 2)[np.newaxis]))
        out = frame.select(
            stats=region_stats(
                pl.col("c"),
                {
                    "heat_max": (heat, "max"),
                    "heat_p90": (heat, "p90"),
                    "other_min": (other, "min"),
                    "area": (heat, "count"),
                },
                region_mode="bbox",
            )
        )
        got = out.item()
        assert set(got) == {"heat_max", "heat_p90", "other_min", "area"}
        np.testing.assert_allclose(got["heat_max"], [s.max() for s in SLICES])
        np.testing.assert_allclose(
            got["heat_p90"], [np.percentile(s, 90) for s in SLICES]
        )
        np.testing.assert_allclose(got["other_min"], [2 * s.min() for s in SLICES])
        np.testing.assert_allclose(got["area"], [s.size for s in SLICES])
        # One row per region, as the struct of equal-length lists explodes.
        rows = out.unnest("stats").explode(pl.all())
        assert rows.height == len(REGIONS)


class TestNormalizeWindows:
    """``normalize(method="percentile" | "range")``: a window mapped to [0, 1],
    clipped outside it."""

    def _run(self, pipe: Pipeline, **columns) -> np.ndarray:
        from polars_cv import numpy_from_struct

        frame = _frame().with_columns(**{k: pl.lit(v) for k, v in columns.items()})
        return numpy_from_struct(
            frame.select(pl.col("a").cv.pipe(pipe).sink("numpy")).item()
        )

    @pytest.mark.parametrize(("low", "high"), [(2.0, 98.0), (0.0, 100.0), (25.0, 30.0)])
    def test_percentile_window(self, low: float, high: float) -> None:
        got = self._run(
            Pipeline()
            .source("array")
            .normalize(method="percentile", low=low, high=high)
        )
        lo, hi = np.percentile(IMG, [low, high])
        np.testing.assert_allclose(
            got, np.clip((IMG - lo) / (hi - lo), 0, 1), atol=1e-6
        )
        assert got.dtype == np.float32

    def test_a_nan_pixel_makes_the_percentile_window_nan(self) -> None:
        """As ``np.percentile`` and ``minmax``: an unknown window is NaN, not
        the empty window's zeros."""
        from polars_cv import numpy_from_struct

        img = np.arange(16, dtype=np.float32).reshape(4, 4)
        img[1, 2] = np.nan
        frame = pl.DataFrame({"a": pl.Series("a", img[np.newaxis])})
        for kwargs in (
            {"method": "percentile", "low": 2.0, "high": 98.0},
            {"method": "minmax"},
        ):
            pipe = Pipeline().source("array").normalize(**kwargs)
            out = frame.select(pl.col("a").cv.pipe(pipe).sink("numpy")).item()
            assert np.isnan(numpy_from_struct(out)).all(), kwargs

    def test_range_window_per_row(self) -> None:
        pipe = (
            Pipeline()
            .source("array")
            .normalize(method="range", low=pl.col("lo"), high=pl.col("hi"))
        )
        got = self._run(pipe, lo=20.0, hi=60.0)
        np.testing.assert_allclose(got, np.clip((IMG - 20) / 40, 0, 1), atol=1e-6)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"method": "percentile", "low": 2.0}, "low.*high"),
            ({"method": "range"}, "low.*high"),
            ({"method": "minmax", "low": 1.0, "high": 2.0}, "low"),
            ({"method": "percentile", "low": -1.0, "high": 50.0}, r"\[0, 100\]"),
            ({"method": "range", "low": 5.0, "high": 5.0}, "below"),
            (
                {
                    "method": "percentile",
                    "low": 2.0,
                    "high": 98.0,
                    "mean": [0.5],
                    "std": [1.0],
                },
                "preset",
            ),
        ],
    )
    def test_refusals(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Pipeline().source("array").normalize(**kwargs)
