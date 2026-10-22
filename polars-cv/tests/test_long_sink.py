"""The ``long`` sink: a buffer as one struct per pixel, against NumPy.

``sink("long", names=[...])`` turns an ``[H, W]`` or ``[H, W, C]`` buffer into
``List[Struct{row, col, <names>...}]``, one name per channel, in row-major
order. ``where=<name>`` keeps the pixels whose named channel is > 0 and drops
that channel from the struct. ``channel_merge`` is how several maps become the
channels.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline

from .conftest import plugin_required

pytestmark = plugin_required

RNG = np.random.default_rng(4)
HEAT = RNG.random((5, 7)).astype(np.float32)
SCORE = RNG.random((5, 7)).astype(np.float32)
MASK = (RNG.random((5, 7)) > 0.6).astype(np.float32)


def _frame(**maps: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({k: pl.Series(k, v[np.newaxis]) for k, v in maps.items()})


def _src(col: str):
    return pl.col(col).cv.pipe(Pipeline().source("array"))


def _want(maps: dict[str, np.ndarray], where: str | None = None) -> pl.DataFrame:
    rows, cols = np.indices(next(iter(maps.values())).shape)
    keep = np.ones(rows.shape, bool) if where is None else maps[where] > 0
    out = {"row": rows[keep].astype(np.uint32), "col": cols[keep].astype(np.uint32)}
    out |= {k: v[keep] for k, v in maps.items() if k != where}
    return pl.DataFrame(out)


def _long(frame: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    return frame.select(expr.alias("p")).explode("p").unnest("p")


class TestLongSink:
    def test_a_single_map_is_one_row_per_pixel(self) -> None:
        got = _long(_frame(h=HEAT), _src("h").sink("long", names=["heat"]))
        assert got.equals(_want({"heat": HEAT}))

    def test_merged_maps_name_each_channel(self) -> None:
        expr = (
            _src("h")
            .channel_merge(_src("s"), _src("m"))
            .sink("long", names=["heat", "score", "mask"])
        )
        got = _long(_frame(h=HEAT, s=SCORE, m=MASK), expr)
        assert got.equals(_want({"heat": HEAT, "score": SCORE, "mask": MASK}))

    def test_where_keeps_the_positive_pixels_and_drops_its_channel(self) -> None:
        expr = (
            _src("h")
            .channel_merge(_src("s"), _src("m"))
            .sink("long", names=["heat", "score", "mask"], where="mask")
        )
        got = _long(_frame(h=HEAT, s=SCORE, m=MASK), expr)
        want = _want({"heat": HEAT, "score": SCORE, "mask": MASK}, where="mask")
        assert got.equals(want)
        assert 0 < got.height < HEAT.size

    def test_the_schema_is_published_before_any_data(self) -> None:
        expr = _src("h").channel_merge(_src("m")).sink("long", names=["heat", "mask"])
        schema = _frame(h=HEAT, m=MASK).lazy().select(expr.alias("p")).collect_schema()
        assert schema["p"] == pl.List(
            pl.Struct(
                {
                    "row": pl.UInt32,
                    "col": pl.UInt32,
                    "heat": pl.Float32,
                    "mask": pl.Float32,
                }
            )
        )

    def test_the_dtype_is_kept(self) -> None:
        img = RNG.integers(0, 256, (3, 4), dtype=np.uint8)
        got = _long(_frame(i=img), _src("i").sink("long", names=["v"]))
        assert got.equals(_want({"v": img}))

    def test_an_empty_selection_is_an_empty_list_and_a_null_row_is_null(self) -> None:
        frame = pl.DataFrame(
            {
                "m": pl.Series(
                    "m",
                    [np.zeros((2, 2), np.float32), None],
                    dtype=pl.Array(pl.Float32, (2, 2)),
                )
            }
        )
        expr = (
            _src("m")
            .channel_merge(_src("m"))
            .sink("long", names=["v", "keep"], where="keep")
        )
        got = frame.select(expr.alias("p"))["p"]
        assert got.to_list() == [[], None]


class TestLongSinkRefusals:
    def test_the_names_must_cover_every_channel(self) -> None:
        # The channel count is the column's, so the query's plan refuses it.
        expr = _src("h").channel_merge(_src("m")).sink("long", names=["heat"])
        with pytest.raises(pl.exceptions.ComputeError, match="2 channels.*1 name"):
            _frame(h=HEAT, m=MASK).select(expr)

    def test_where_must_be_one_of_the_names(self) -> None:
        with pytest.raises(ValueError, match="'where'.*'mask'.*not one of"):
            _src("h").sink("long", names=["heat"], where="mask")

    @pytest.mark.parametrize("names", [["row"], ["a", "a"], []])
    def test_names_must_be_distinct_and_not_row_or_col(self, names: list[str]) -> None:
        with pytest.raises(ValueError, match="names"):
            _src("h").sink("long", names=names)

    def test_a_buffer_of_another_rank_is_refused(self) -> None:
        expr = Pipeline().source("array").reshape([35])
        with pytest.raises(ValueError, match=r"\[H, W\] or \[H, W, C\]"):
            pl.col("h").cv.pipe(expr).sink("long", names=["v"])

    def test_names_are_required(self) -> None:
        with pytest.raises(ValueError, match="names"):
            _src("h").sink("long")


def test_channel_merge_of_mixed_dtypes_says_to_cast() -> None:
    expr = _src("h").channel_merge(
        pl.col("m").cv.pipe(Pipeline().source("array").cast("u8"))
    )
    with pytest.raises(pl.exceptions.ComputeError, match="cast them to one"):
        _frame(h=HEAT, m=MASK).select(expr.sink("numpy"))
