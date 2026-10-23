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


def test_numpy_covers_every_statistic() -> None:
    """Every sweep below parametrizes over ``NUMPY``: a statistic added to
    the engine's ``Statistic`` and not here would go untested everywhere."""
    from polars_cv._types import Statistic

    assert set(NUMPY) == {s.value for s in Statistic}


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
        # +0, as NumPy's sum of nothing (it was -0, Rust's `Sum` identity).
        assert not np.signbit(np.asarray(got, np.float64)).any(), got

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
            ("pmax", "unknown reduction 'pmax'"),
            # The digits of a percentile only: no sign, exponent or space.
            ("p+50", "unknown reduction 'p\\+50'"),
            ("p5e1", "unknown reduction 'p5e1'"),
            ("p 50", "unknown reduction 'p 50'"),
            ("p50.", "unknown reduction 'p50.'"),
            ("pinf", "unknown reduction 'pinf'"),
        ],
    )
    def test_a_percentile_is_spelled_pNN(self, reduction: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Pipeline().source("array").label_reduce(pl.col("c"), reduction=reduction)


def _reduce(pipe: Pipeline) -> np.ndarray:
    out = _frame().select(pl.col("a").cv.pipe(pipe).sink("numpy")).item()
    from polars_cv import numpy_from_struct

    return numpy_from_struct(out)


#: A statistic's dedicated op, as ``reduce`` must compute it.
DEDICATED = {
    "sum": lambda p, axis: p.reduce_sum(axis=axis),
    "mean": lambda p, axis: p.reduce_mean(axis=axis),
    "min": lambda p, axis: p.reduce_min(axis=axis),
    "max": lambda p, axis: p.reduce_max(axis=axis),
    "std": lambda p, axis: p.reduce_std(axis=axis),
    "median": lambda p, axis: p.reduce_median(axis=axis),
    "p90": lambda p, axis: p.reduce_percentile(90.0, axis=axis),
}


def _reduced(arr: np.ndarray, pipe: Pipeline) -> np.ndarray:
    dtype = {np.uint8: pl.UInt8, np.float32: pl.Float32}[arr.dtype.type]
    frame = pl.DataFrame(
        {"a": pl.Series("a", arr[np.newaxis], dtype=pl.Array(dtype, arr.shape))}
    )
    sink = "native" if pipe.current_domain() == "scalar" else "list"
    out = frame.select(pl.col("a").cv.pipe(pipe).sink(sink)).item()
    return np.asarray(out.to_list() if isinstance(out, pl.Series) else out, np.float64)


class TestReduce:
    """``reduce(reduction, axis=)``: any reduction, globally or along an
    axis, computed by the same kernel as the statistic's dedicated op."""

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("reduction", [*sorted(NUMPY), "p90"])
    def test_matches_numpy(self, reduction: str, axis: int | None) -> None:
        pipe = Pipeline().source("array").reduce(reduction, axis=axis)
        got = _reduced(IMG.astype(np.float32), pipe)
        x = IMG.astype(np.float32).astype(np.float64)
        if reduction == "count":
            want = x.size if axis is None else np.full(x.shape[1 - axis], x.shape[axis])
        elif reduction == "p90":
            want = np.percentile(x, 90, axis=axis)
        else:
            want = NUMPY[reduction](x, axis=axis)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize("reduction", sorted(DEDICATED))
    def test_is_the_dedicated_ops_result(self, reduction: str, dtype, axis) -> None:
        """Bit for bit, NaN included: one implementation per statistic."""
        arr = IMG.astype(dtype)
        if dtype is np.float32:
            arr[2, 3] = np.nan
        base = Pipeline().source("array")
        got = _reduced(arr, base.reduce(reduction, axis=axis))
        want = _reduced(arr, DEDICATED[reduction](base, axis))
        np.testing.assert_array_equal(got, want)


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
    """``normalize_percentile`` / ``normalize_range``: a window mapped to
    [0, 1], clipped outside it. Each its own op, so ``low`` and ``high`` are
    required fields rather than ``normalize`` fields read by two methods."""

    def _run(self, pipe: Pipeline, **columns) -> np.ndarray:
        from polars_cv import numpy_from_struct

        frame = _frame().with_columns(**{k: pl.lit(v) for k, v in columns.items()})
        return numpy_from_struct(
            frame.select(pl.col("a").cv.pipe(pipe).sink("numpy")).item()
        )

    @pytest.mark.parametrize(("low", "high"), [(2.0, 98.0), (0.0, 100.0), (25.0, 30.0)])
    def test_percentile_window(self, low: float, high: float) -> None:
        got = self._run(
            Pipeline().source("array").normalize_percentile(low=low, high=high)
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
        for pipe in (
            Pipeline().source("array").normalize_percentile(low=2.0, high=98.0),
            Pipeline().source("array").normalize(method="minmax"),
        ):
            out = frame.select(pl.col("a").cv.pipe(pipe).sink("numpy")).item()
            assert np.isnan(numpy_from_struct(out)).all(), pipe

    def test_range_window_per_row(self) -> None:
        pipe = (
            Pipeline()
            .source("array")
            .normalize_range(low=pl.col("lo"), high=pl.col("hi"))
        )
        got = self._run(pipe, lo=20.0, hi=60.0)
        np.testing.assert_allclose(got, np.clip((IMG - 20) / 40, 0, 1), atol=1e-6)

    @pytest.mark.parametrize(
        ("op", "low", "high", "match"),
        [
            ("normalize_percentile", -1.0, 50.0, r"\[0, 100\]"),
            ("normalize_percentile", 50.0, 150.0, r"\[0, 100\]"),
            ("normalize_percentile", 60.0, 40.0, "below"),
            ("normalize_range", 5.0, 5.0, "below"),
        ],
    )
    def test_refusals(self, op: str, low: float, high: float, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            getattr(Pipeline().source("array"), op)(low=low, high=high)


class TestOnePercentileRule:
    """Every percentile is the one ``Percent``: ``reduce("pNN")``,
    ``reduce_percentile(q)``, ``rank_filter(q)`` and ``normalize_percentile``'s
    edges take the same values and refuse the same ones, in the same words."""

    BUILD = {
        "reduce": lambda p, q: p.reduce(f"p{q}"),
        "reduce_percentile": lambda p, q: p.reduce_percentile(q),
        "rank_filter": lambda p, q: p.rank_filter(height=3, width=3, q=q),
        "normalize_percentile": lambda p, q: p.normalize_percentile(low=0.0, high=q),
    }

    @pytest.mark.parametrize("op", sorted(BUILD))
    @pytest.mark.parametrize("q", [100.5, 250])
    def test_outside_0_100_is_refused_alike(self, op: str, q: float) -> None:
        with pytest.raises(ValueError, match=rf"{q} is outside \[0, 100\]"):
            self.BUILD[op](Pipeline().source("array"), q)

    @pytest.mark.parametrize("op", sorted(BUILD))
    def test_any_decimals_are_taken_alike(self, op: str) -> None:
        self.BUILD[op](Pipeline().source("array"), 33.333)

    def test_a_percentile_with_decimals_matches_numpy(self) -> None:
        got = _reduced(
            IMG.astype(np.float32), Pipeline().source("array").reduce("p33.333")
        )
        np.testing.assert_allclose(got, np.percentile(IMG, 33.333), rtol=1e-12)

    def test_a_per_row_q_outside_0_100_names_its_op_and_field(self) -> None:
        frame = _frame().with_columns(q=pl.lit(150.0))
        pipe = Pipeline().source("array").reduce_percentile(pl.col("q"))
        with pytest.raises(
            pl.exceptions.ComputeError,
            match=r"reduce_percentile: q: .*150 is outside \[0, 100\]",
        ):
            frame.select(pl.col("a").cv.pipe(pipe).sink("native"))
