"""One plugin call runs its rows in parallel and returns them as if in order.

A call splits its rows into ranges that run on the plugin's thread pool
(CR-32). These pin what that must not change, on a frame large enough to be
split and held in a single chunk (the in-memory engine hands such a frame to
the plugin in one call): row order, which error ``on_error="raise"`` reports,
and which rows the null policies null.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct

from .conftest import plugin_required

N = 300
BAD_EARLY, BAD_LATE = 120, 250


def _blobs() -> list[bytes]:
    """N distinct f32 blobs, via the plugin's own blob sink."""
    arrays = [np.full((3, 2), i, dtype=np.float32) for i in range(N)]
    df = pl.DataFrame({"a": pl.Series("a", np.stack(arrays))})
    return df.select(b=pl.col("a").cv.pipe(Pipeline().source("array")).sink("blob"))[
        "b"
    ].to_list()


def _frame(*, corrupt: bool) -> pl.DataFrame:
    blobs = _blobs()
    if corrupt:
        # Two different failures, so the reported one identifies the row.
        blobs[BAD_EARLY] = b"XXXX" + blobs[BAD_EARLY][4:]  # bad magic
        blobs[BAD_LATE] = b"VIEW"  # too short for a header
    df = pl.DataFrame({"b": blobs}, schema={"b": pl.Binary}).rechunk()
    assert df["b"].n_chunks() == 1
    return df


def _sums(series: pl.Series) -> list[float | None]:
    return [
        None if row is None else float(np.sum(numpy_from_struct(row)))
        for row in series.to_list()
    ]


@plugin_required
class TestParallelRows:
    def test_rows_come_back_in_order(self) -> None:
        pipe = Pipeline().source("blob", dtype="f32").scale(2.0)
        out = _frame(corrupt=False).select(o=pl.col("b").cv.pipe(pipe).sink("numpy"))
        assert _sums(out["o"]) == [12.0 * i for i in range(N)]

    def test_raise_reports_the_earliest_failing_row(self) -> None:
        pipe = Pipeline().source("blob", dtype="f32").scale(2.0)
        with pytest.raises(pl.exceptions.ComputeError) as excinfo:
            _frame(corrupt=True).select(o=pl.col("b").cv.pipe(pipe).sink("numpy"))
        message = str(excinfo.value)
        assert "magic" in message
        assert "too short" not in message

    def test_null_policies_null_exactly_the_failing_rows(self) -> None:
        pipe = (
            Pipeline()
            .source("blob", dtype="f32")
            .scale(2.0)
            .on_error("null_with_message")
        )
        out = _frame(corrupt=True).select(o=pl.col("b").cv.pipe(pipe).sink("numpy"))
        rows = out.unnest("o")
        expected = [None if i in (BAD_EARLY, BAD_LATE) else 12.0 * i for i in range(N)]
        assert _sums(rows["_output"]) == expected
        errors = rows["_error"].to_list()
        assert [i for i, e in enumerate(errors) if e is not None] == [
            BAD_EARLY,
            BAD_LATE,
        ]
        assert "magic" in errors[BAD_EARLY]
        assert "too short" in errors[BAD_LATE]
