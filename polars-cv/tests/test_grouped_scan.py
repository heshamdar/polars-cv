"""The grouped-scan and exact-sum authority (``metrics/_grouped_scan.py``).

Each output is pinned against the ``.over()`` form it replaces, on randomised
grouped data with ties, nulls and null group keys, and each plan is checked for
in-memory fallbacks. Integer scans must match exactly. A float cumulative sum is
exact in fixed point and rounded once, so it matches the float ``.over`` form to
a relative tolerance set by that form's own rounding, not by this one.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_cv.metrics._grouped_scan import (
    CumMax,
    CumSum,
    IsFirst,
    IsLast,
    Lag,
    RowIndex,
    exact_sums,
    grouped_scan,
)
from tests._streaming_guard import in_memory_nodes


def _frame(seed: int, n: int = 400) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 7, n)
    return pl.DataFrame(
        {
            # Group 6 becomes the null group: `.over` treats null as a key.
            "g": pl.Series([None if v == 6 else int(v) for v in g], dtype=pl.Int64),
            "h": rng.integers(0, 2, n).astype(str),
            "s": rng.integers(0, 40, n).astype(float),  # ties on purpose
            "t": rng.permutation(n),  # breaks the ties: a total order
            "k": rng.integers(0, 5, n),
            "w": rng.lognormal(0.0, 2.0, n),
            "p": rng.random(n),
        }
    )


def _reference(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    over = (lambda e: e.over(keys)) if keys else (lambda e: e)
    return df.sort(
        *keys, "s", "t", descending=[False] * len(keys) + [True, False]
    ).with_columns(
        ck=over(pl.col("k").cum_sum()),
        cw=over(pl.col("w").cum_sum()),
        mx=over(pl.col("p").reverse().cum_max().reverse()),
        fmx=over(pl.col("p").cum_max()),
        lag=over(pl.col("p").shift(1)),
        idx=over(pl.int_range(pl.len(), dtype=pl.Int64)),
        first=over(pl.int_range(pl.len()) == 0),
        last=over(pl.int_range(pl.len()) == pl.len() - 1),
    )


def _scan(df: pl.DataFrame, keys: list[str]) -> pl.LazyFrame:
    return grouped_scan(
        df.lazy(),
        keys,
        by=["s", "t"],
        descending=[True, False],
        ck=CumSum("k"),
        cw=CumSum("w"),
        mx=CumMax("p", reverse=True),
        fmx=CumMax("p"),
        lag=Lag("p"),
        idx=RowIndex(),
        first=IsFirst(),
        last=IsLast(),
    )


@pytest.mark.parametrize("keys", [[], ["g"], ["g", "h"]], ids=str)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grouped_scan_matches_over(keys: list[str], seed: int) -> None:
    df = _frame(seed)
    want = _reference(df, keys)
    got = _scan(df, keys).collect(engine="streaming")
    exact = [
        *["g", "h", "s", "t", "k", "w", "p"],
        *["ck", "mx", "fmx", "lag", "idx", "first", "last"],
    ]
    assert_frame_equal(got.select(exact), want.select(exact), check_dtypes=False)
    np.testing.assert_allclose(got["cw"], want["cw"], rtol=1e-12)


@pytest.mark.parametrize("keys", [[], ["g", "h"]], ids=str)
def test_grouped_scan_stays_streaming(keys: list[str]) -> None:
    assert in_memory_nodes(_scan(_frame(0), keys)) == []


def test_grouped_scan_keeps_nulls_where_over_does() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 2],
            "s": [3.0, 2.0, 1.0, 1.0],
            "t": [0, 1, 2, 3],
            "k": [1, None, 2, 5],
            "w": [0.5, None, 0.25, 1.0],
            "p": [0.2, None, 0.9, 0.1],
        }
    )
    got = grouped_scan(
        df.lazy(),
        ["g"],
        by=["s", "t"],
        descending=[True, False],
        ck=CumSum("k"),
        cw=CumSum("w"),
        mx=CumMax("p", reverse=True),
    ).collect()
    want = df.sort("g", "s", "t", descending=[False, True, False]).with_columns(
        ck=pl.col("k").cum_sum().over("g"),
        cw=pl.col("w").cum_sum().over("g"),
        mx=pl.col("p").reverse().cum_max().reverse().over("g"),
    )
    assert_frame_equal(got.select(want.columns), want, check_dtypes=False)


def test_float_cum_sum_resets_exactly_after_huge_groups() -> None:
    """A float running sum minus its group offset would cancel catastrophically
    here; the fixed-point one does not."""
    df = pl.DataFrame(
        {
            "g": [0, 0, 1, 1],
            "s": [2.0, 1.0, 2.0, 1.0],
            "t": [0, 1, 2, 3],
            "w": [1e12, 1e12, 0.1, 0.2],
        }
    )
    got = grouped_scan(
        df.lazy(), ["g"], by=["s", "t"], descending=[True, False], c=CumSum("w")
    ).collect()
    np.testing.assert_allclose(
        got["c"].to_list(), [1e12, 2e12, 0.1, 0.1 + 0.2], rtol=1e-15
    )


# -- exact sums -------------------------------------------------------------


def test_exact_sums_match_the_true_sum() -> None:
    df = _frame(3)
    got = (
        exact_sums(df.lazy(), ["g"], total=pl.col("w"))
        .collect()
        .sort("g", nulls_last=True)
    )
    want = df.group_by("g").agg(total=pl.col("w").sum()).sort("g", nulls_last=True)
    np.testing.assert_allclose(got["total"], want["total"], rtol=1e-14)


def test_exact_sums_are_independent_of_chunking() -> None:
    """The point of the fixed-point sum: float addition is not associative, so
    a chunked float sum's last bits depend on the chunking. This one's do not."""
    rng = np.random.default_rng(7)
    values = rng.lognormal(0.0, 6.0, 5000)
    whole = pl.DataFrame({"g": [0] * 5000, "w": values})
    chunked = pl.concat([whole.slice(i, 37) for i in range(0, 5000, 37)], rechunk=False)
    assert chunked.n_chunks() > 1
    float_sums = {
        whole.select(pl.col("w").sum()).item(),
        chunked.select(pl.col("w").sum()).item(),
        whole.reverse().select(pl.col("w").sum()).item(),
    }
    exact = {
        exact_sums(frame.lazy(), ["g"], total=pl.col("w")).collect().item(0, "total")
        for frame in (
            whole,
            chunked,
            whole.reverse(),
            whole.sample(fraction=1.0, shuffle=True, seed=1),
        )
    }
    assert len(float_sums) > 1, "fixture no longer shows float sums diverging"
    assert len(exact) == 1


def test_exact_sums_without_keys_and_with_nulls() -> None:
    lf = pl.LazyFrame({"w": [0.5, None, 0.25]})
    out = exact_sums(lf, [], total=pl.col("w")).collect()
    assert out.item(0, "total") == 0.75
    empty = exact_sums(
        pl.LazyFrame({"w": [None]}, schema={"w": pl.Float64}), [], total=pl.col("w")
    ).collect()
    assert empty.item(0, "total") == 0.0


def test_exact_sums_stay_streaming() -> None:
    assert (
        in_memory_nodes(exact_sums(_frame(0).lazy(), ["g", "h"], total=pl.col("w")))
        == []
    )


_SPECIALS = pl.DataFrame(
    {
        "g": [0, 0, 1, 1, 2, 2, 3, 3, 4],
        "s": [5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 5.0, 4.0, 1.0],
        "t": list(range(9)),
        "w": [
            1.0,
            float("nan"),
            float("inf"),
            2.0,
            float("inf"),
            float("-inf"),
            float("-inf"),
            0.5,
            0.25,
        ],
    }
)


def test_exact_sums_combine_nan_and_inf_like_float_addition() -> None:
    got = exact_sums(_SPECIALS.lazy(), ["g"], total=pl.col("w")).collect().sort("g")
    want = _SPECIALS.group_by("g").agg(total=pl.col("w").sum()).sort("g")
    assert_frame_equal(got, want)


def test_grouped_cum_sum_carries_nan_and_inf_within_the_group_only() -> None:
    got = grouped_scan(
        _SPECIALS.lazy(), ["g"], by=["s", "t"], descending=[True, False], c=CumSum("w")
    ).collect()
    want = _SPECIALS.sort("g", "s", "t", descending=[False, True, False]).with_columns(
        c=pl.col("w").cum_sum().over("g")
    )
    assert_frame_equal(got, want)
