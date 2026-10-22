"""``flip(when=)``: a per-row condition on a flip.

Orienting every view the same way (flip the right-side views so the chest
wall is on the left) needed a branch in polars after sinking. The shape is
the same flipped or not, so the condition may come from a column.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_cv import Pipeline, numpy_from_struct

from .conftest import plugin_required

pytestmark = plugin_required

IMG = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)


def _run(pipe: Pipeline, views: list[str]) -> list[np.ndarray]:
    frame = pl.DataFrame(
        {"a": pl.Series("a", np.stack([IMG] * len(views))), "view": views}
    )
    out = frame.select(pl.col("a").cv.pipe(pipe).sink("numpy")).to_series()
    return [numpy_from_struct(row) for row in out]


def test_a_column_decides_each_rows_flip() -> None:
    pipe = Pipeline().source("array").flip_h(when=pl.col("view").str.starts_with("R"))
    got = _run(pipe, ["RMLO", "LMLO", "RCC"])
    np.testing.assert_array_equal(got[0], IMG[:, ::-1])
    np.testing.assert_array_equal(got[1], IMG)
    np.testing.assert_array_equal(got[2], IMG[:, ::-1])


def test_a_literal_false_leaves_the_image() -> None:
    (got,) = _run(Pipeline().source("array").flip(axes=[0], when=False), ["x"])
    np.testing.assert_array_equal(got, IMG)


def test_flip_v_and_the_default_flip() -> None:
    (got,) = _run(Pipeline().source("array").flip_v(when=True), ["x"])
    np.testing.assert_array_equal(got, IMG[::-1])
    (got,) = _run(Pipeline().source("array").flip(axes=[0, 1]), ["x"])
    np.testing.assert_array_equal(got, IMG[::-1, ::-1])
