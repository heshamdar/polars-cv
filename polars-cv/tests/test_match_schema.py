"""The matchers' shared tail reads metrics' own correspondence contract.

``matched_table`` takes any producer of a ``MATCH_SCHEMA`` struct -- the
polars-cv matchers (``.contour.correspond`` / ``.bbox.correspond``) are two;
a plain Polars column is another -- and refuses any other struct before
data moves, so a producer that drifts fails rather than being misread.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.metrics._matching._table import matched_table
from polars_cv.metrics._types import MATCH_SCHEMA


def _prepared(match_dtype: pl.DataType, match_value: dict) -> pl.LazyFrame:
    """One image: two predictions, one ground truth, and a precomputed match."""
    return pl.LazyFrame(
        {
            "image_id": ["a"],
            "weight": [1.0],
            "scores": [[0.9, 0.4]],
            "gts": [[1]],
            "m": [match_value],
        },
        schema={
            "image_id": pl.String,
            "weight": pl.Float64,
            "scores": pl.List(pl.Float64),
            "gts": pl.List(pl.Int64),
            "m": match_dtype,
        },
    )


def _table(prepared: pl.LazyFrame):
    return matched_table(
        prepared,
        match=lambda _t: pl.col("m"),
        thresholds=(0.5,),
        scores_col="scores",
        gt_col="gts",
        class_col=None,
        group_col=None,
    )


def test_a_plain_polars_producer_is_matched() -> None:
    """No polars-cv in the producer: the tail is generic."""
    match = {"right_idx": [0, None], "overlap": [0.8, 0.0], "duplicate": [False, False]}
    det = (
        _table(_prepared(MATCH_SCHEMA, match))
        .detections.sort("score", descending=True)
        .collect()
    )
    assert det["is_tp"].to_list() == [True, False]
    assert det["iou"].to_list() == [0.8, 0.0]


@pytest.mark.parametrize(
    "dtype, value",
    [
        (
            pl.Struct(
                {
                    "pred_idx": pl.List(pl.UInt32),
                    "overlap": pl.List(pl.Float64),
                    "duplicate": pl.List(pl.Boolean),
                }
            ),
            {"pred_idx": [0, None], "overlap": [0.8, 0.0], "duplicate": [False, False]},
        ),
        (
            pl.Struct(
                {
                    "right_idx": pl.List(pl.Int64),
                    "overlap": pl.List(pl.Float64),
                    "duplicate": pl.List(pl.Boolean),
                }
            ),
            {
                "right_idx": [0, None],
                "overlap": [0.8, 0.0],
                "duplicate": [False, False],
            },
        ),
        (
            pl.Struct(
                {"right_idx": pl.List(pl.UInt32), "overlap": pl.List(pl.Float64)}
            ),
            {"right_idx": [0, None], "overlap": [0.8, 0.0]},
        ),
    ],
    ids=["renamed field", "wrong index dtype", "missing field"],
)
def test_a_producer_off_the_contract_is_refused(
    dtype: pl.DataType, value: dict
) -> None:
    with pytest.raises(TypeError, match="MATCH_SCHEMA"):
        _table(_prepared(dtype, value))
