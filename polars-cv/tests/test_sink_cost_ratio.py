"""The typed tensor sinks cost about what the zero-copy numpy sink costs.

``sink("list")`` at rank >= 2, and ``sink("array")`` with any null row, used to
build one ``AnyValue`` per *element*: on 64 rows of 64x64x3 u8 they ran ~34x
and ~40x slower than ``sink("numpy")`` (CR-33). They are now built straight
into Arrow from one flat values buffer.

A wall-clock test is only robust when the margin dwarfs the noise, so this
pins a *ratio* against the numpy sink over the same decode, best of several
runs, with a bound (5x) far below the old 34-40x and far above the new ~1.2x.
It is in the slow lane because it times real work.
"""

from __future__ import annotations

import time

import polars as pl
import pytest

from polars_cv import Pipeline

from .conftest import make_image_png, plugin_required

ROWS = 64
BOUND = 5.0


def _best_seconds(frame: pl.DataFrame, expr: pl.Expr, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        frame.lazy().select(o=expr).collect()
        best = min(best, time.perf_counter() - start)
    return best


@plugin_required
@pytest.mark.slow
class TestTensorSinkCost:
    @pytest.fixture(scope="class")
    def frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        images = [make_image_png(64, 64, 3, seed=i) for i in range(ROWS)]
        full = pl.DataFrame({"img": images})
        with_null = pl.DataFrame(
            {"img": images[:-1] + [None]}, schema={"img": pl.Binary}
        )
        return full, with_null

    @pytest.fixture(scope="class")
    def pipe(self) -> Pipeline:
        return Pipeline().source("image_bytes", dtype="u8")

    @pytest.mark.parametrize(
        ("label", "sink", "kwargs", "nulls"),
        [
            ("list, rank 3", "list", {}, False),
            ("list, rank 3, one null row", "list", {}, True),
            ("array, one null row", "array", {"shape": [64, 64, 3]}, True),
        ],
    )
    def test_within_bound_of_numpy_sink(
        self,
        frames: tuple[pl.DataFrame, pl.DataFrame],
        pipe: Pipeline,
        label: str,
        sink: str,
        kwargs: dict,
        nulls: bool,
    ) -> None:
        frame = frames[1] if nulls else frames[0]
        col = pl.col("img").cv
        numpy = _best_seconds(frame, col.pipe(pipe).sink("numpy"))
        typed = _best_seconds(frame, col.pipe(pipe).sink(sink, **kwargs))
        ratio = typed / numpy
        assert ratio < BOUND, (
            f"{label} sink took {ratio:.1f}x the numpy sink (bound {BOUND}x): "
            "a per-element slow path is back"
        )
