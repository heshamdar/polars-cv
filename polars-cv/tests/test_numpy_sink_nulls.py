"""A null row in the numpy/torch/ndarray sinks is a null value (CR-39).

It used to be a struct whose five fields were all null, so ``is_null()`` was
``False`` and ``drop_nulls()``/``null_count()`` could not see it — unlike every
other sink, which publishes a null row as a null.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct

from .conftest import make_test_png, plugin_required


@plugin_required
@pytest.mark.parametrize("sink", ["numpy", "torch", "ndarray"])
class TestTensorStructNullRows:
    @staticmethod
    def _frame() -> pl.DataFrame:
        return pl.DataFrame(
            {"img": [make_test_png(4, 4), None, b"not an image"]},
            schema={"img": pl.Binary},
        )

    @staticmethod
    def _expr(sink: str) -> pl.Expr:
        pipe = Pipeline().source("image_bytes", dtype="u8").on_error("null")
        return pl.col("img").cv.pipe(pipe).sink(sink)

    @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
    def test_null_input_and_failed_row_are_null(self, sink: str, engine: str) -> None:
        out = self._frame().lazy().select(o=self._expr(sink)).collect(engine=engine)  # ty: ignore[invalid-argument-type]
        assert out["o"].is_null().to_list() == [False, True, True]
        assert out["o"].null_count() == 2

    def test_drop_nulls_keeps_only_real_rows(self, sink: str) -> None:
        out = self._frame().select(o=self._expr(sink)).drop_nulls()
        assert out.height == 1
        assert numpy_from_struct(out["o"][0]).shape == (4, 4, 3)


def test_numpy_from_struct_rejects_a_null_row_clearly() -> None:
    with pytest.raises(ValueError, match="null"):
        numpy_from_struct(None)  # ty: ignore[invalid-argument-type]
