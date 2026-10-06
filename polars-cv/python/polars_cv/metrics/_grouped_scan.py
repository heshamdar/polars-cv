"""Grouped ordered scans and exact sums: the metrics' one way to write either.

**The grouped scan** (:func:`grouped_scan`) sorts once by ``(*keys, *by)``
and computes each scan (running sum, running max, lag, row index, first/last
flag) as a window ``.over(keys)`` on the sorted frame. Since polars 2.0 both
the sort and the windows are native streaming nodes
(``tests/test_grouped_scan.py`` holds it to no fallback at all). A sort inside
an ``agg`` still falls back to the in-memory engine, so per-group scans are
written here rather than there.

**Exact sums** (:func:`exact_sums`) convert each value to Int128 fixed point
and sum those integers natively in a group-by. The scale is a power of two
chosen from the frame's largest finite magnitude and its row count, and the
conversion rounds once. Integer addition is associative, so a sum does not
depend on how the rows were chunked, ordered or split across threads. A float
``sum`` does: its last bits change from run to run, which made bootstrap
bounds irreproducible.

Error: a term keeps at least ``125 − log2(rows)`` bits below the frame's
largest magnitude (absolute error ≤ ``max·rows·2⁻¹²⁵`` per term), and the sum
is rounded to Float64 once. Both are below Float64's own resolution for any
frame polars can hold. NaN and ±inf are counted apart and combined as float
addition would combine them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

#: Bits of Int128 headroom above the largest magnitude (and the row count).
_FIXED_BITS = 125
#: Keeps ``2 ** k`` a finite, normal Float64.
_MAX_EXP = 1000


@dataclass(frozen=True)
class CumSum:
    """The group's running sum of ``col`` (nulls stay null, as in ``.over``)."""

    col: str


@dataclass(frozen=True)
class CumMax:
    """The group's running max of ``col``; ``reverse`` runs it from the end."""

    col: str
    reverse: bool = False


@dataclass(frozen=True)
class Lag:
    """``col`` from ``n`` rows earlier in the group (null across a group start)."""

    col: str
    n: int = 1


@dataclass(frozen=True)
class RowIndex:
    """The row's 0-based position within its group (Int64)."""


@dataclass(frozen=True)
class IsFirst:
    """Whether the row is its group's first (in the scan order)."""


@dataclass(frozen=True)
class IsLast:
    """Whether the row is its group's last (in the scan order)."""


Scan = CumSum | CumMax | Lag | RowIndex | IsFirst | IsLast


def grouped_scan(
    lf: pl.LazyFrame,
    keys: Sequence[str],
    *,
    by: Sequence[str],
    descending: Sequence[bool],
    nulls_last: bool = False,
    **out: Scan,
) -> pl.LazyFrame:
    """``lf`` sorted by ``(*keys, *by)``, with each ``out`` scan per group.

    Each scan is a window over ``keys`` on the sorted frame, so null group
    keys form a group of their own, as in ``.over``. The order ``by`` gives
    within a group should be total; ties would make the result depend on
    which tied row the sort put first.

    Args:
        lf: The frame.
        keys: The group columns; empty scans the whole frame as one group.
        by: The within-group order.
        descending: One flag per ``by`` column.
        nulls_last: Passed to the sort.
        **out: Output column name → the scan that fills it.

    Returns:
        The sorted frame with the ``out`` columns appended.
    """
    keys, by = list(keys), list(by)
    if len(descending) != len(by):
        raise ValueError("`descending` needs one flag per `by` column")
    ordered = lf.sort(
        *keys,
        *by,
        descending=[False] * len(keys) + list(descending),
        nulls_last=nulls_last,
        maintain_order=True,
    )

    def per_group(expr: pl.Expr) -> pl.Expr:
        return expr.over(keys) if keys else expr

    position = pl.int_range(pl.len(), dtype=pl.Int64)
    exprs: list[pl.Expr] = []
    for name, scan in out.items():
        if isinstance(scan, CumSum):
            expr = pl.col(scan.col).cum_sum()
        elif isinstance(scan, CumMax):
            expr = pl.col(scan.col).cum_max(reverse=scan.reverse)
        elif isinstance(scan, Lag):
            expr = pl.col(scan.col).shift(scan.n)
        elif isinstance(scan, RowIndex):
            expr = position
        elif isinstance(scan, IsFirst):
            expr = position == 0
        elif isinstance(scan, IsLast):
            expr = position == pl.len() - 1
        else:  # pragma: no cover - Scan is closed
            raise TypeError(f"not a scan: {scan!r}")
        exprs.append(per_group(expr).alias(name))
    return ordered.with_columns(exprs)


_SPECIAL = ("nan", "pinf", "ninf")


@dataclass(frozen=True)
class _Fixed:
    """A float column split into an exact finite part and its specials.

    ``value`` is the finite part in fixed point (``0`` for a non-finite value,
    null for null); ``specials`` count NaN, +inf and -inf (``0``/``1`` per row),
    so a sum or running sum of each is an exact integer too and
    :func:`_combine` rebuilds what a float sum would give.
    """

    value: pl.Expr
    scale: pl.Expr
    specials: dict[str, pl.Expr]


def _fixed(expr: pl.Expr) -> _Fixed:
    """``expr`` in Int128 fixed point, with the frame-wide scale it used.

    The scale is ``2 ** k`` with ``k = 125 − ⌈log2(rows+1)⌉ − ⌈log2 max|x|⌉``
    over the finite values, so a sum of every row cannot overflow.
    Multiplying by a power of two is exact, so the only rounding is to the
    nearest integer.
    """
    finite = pl.when(expr.is_finite()).then(expr)
    magnitude = finite.abs().max()
    k = (
        pl.when(magnitude > 0)
        .then(
            pl.lit(_FIXED_BITS)
            - (pl.len() + 1).cast(pl.Float64).log(2).ceil()
            - magnitude.log(2).ceil()
        )
        .otherwise(0.0)
        .clip(-_MAX_EXP, _MAX_EXP)
    )
    scale = pl.lit(2.0).pow(k)
    value = (
        pl.when(expr.is_finite())
        .then((expr * scale).round())
        .when(expr.is_not_null())
        .then(0.0)
        .cast(pl.Int128)
    )
    specials = {
        "nan": expr.is_nan(),
        "pinf": expr == float("inf"),
        "ninf": expr == float("-inf"),
    }
    return _Fixed(
        value=value,
        scale=scale,
        specials={k: v.fill_null(False).cast(pl.Int64) for k, v in specials.items()},
    )


def _combine(finite: pl.Expr, nan: pl.Expr, pinf: pl.Expr, ninf: pl.Expr) -> pl.Expr:
    """The finite sum plus whether NaN, +inf or -inf were added, combined as
    float addition would combine them (``inf + -inf`` is NaN)."""
    return (
        pl.when(nan | (pinf & ninf))
        .then(float("nan"))
        .when(pinf)
        .then(float("inf"))
        .when(ninf)
        .then(float("-inf"))
        .otherwise(finite)
    )


def exact_sums(
    lf: pl.LazyFrame,
    keys: Sequence[str],
    *,
    extra: Sequence[pl.Expr] = (),
    **sums: pl.Expr,
) -> pl.LazyFrame:
    """``[*keys, *sums, *extra]``: each ``sums`` expression summed exactly per group.

    The sum is order-independent, so it is reproducible bit for bit (see the
    module docstring). Nulls are skipped, and a group whose values are all
    null sums to ``0.0``. ``extra`` holds further aggregations for the same
    group-by; they must be native in streaming themselves (``count``,
    ``any``, ``len`` …). With no ``keys`` the result is one row.
    """
    keys = list(keys)
    fixed = {name: _fixed(e) for name, e in sums.items()}
    framed = lf.with_columns(
        *(f.value.alias(f"_xs_{n}") for n, f in fixed.items()),
        *(f.scale.alias(f"_xsc_{n}") for n, f in fixed.items()),
        *(
            e.alias(f"_xs{k}_{n}")
            for n, f in fixed.items()
            for k, e in f.specials.items()
        ),
    )
    aggs = [
        *(pl.col(f"_xs_{n}").sum() for n in fixed),
        *(pl.col(f"_xsc_{n}").first() for n in fixed),
        *(pl.col(f"_xs{k}_{n}").sum() for n in fixed for k in _SPECIAL),
        *extra,
    ]
    agged = framed.group_by(keys).agg(aggs) if keys else framed.select(aggs)

    def total(n: str) -> pl.Expr:
        finite = pl.col(f"_xs_{n}").cast(pl.Float64) / pl.col(f"_xsc_{n}").fill_null(
            1.0
        )
        nan, pinf, ninf = (pl.col(f"_xs{k}_{n}") > 0 for k in _SPECIAL)
        return _combine(finite, nan, pinf, ninf).alias(n)

    return agged.with_columns(total(n) for n in fixed).select(
        *keys, *fixed, *(e.meta.output_name() for e in extra)
    )


def exact_mean(
    lf: pl.LazyFrame,
    keys: Sequence[str],
    col: str,
    out: str,
    *,
    extra: Sequence[pl.Expr] = (),
) -> pl.LazyFrame:
    """``col``'s exact mean per group as ``out`` (null for no non-null values).

    ``extra`` passes further aggregations through to :func:`exact_sums`.
    """
    summed = exact_sums(
        lf, keys, extra=[pl.col(col).count().alias("_xm_n"), *extra], _xm_s=pl.col(col)
    )
    return summed.with_columns(
        pl.when(pl.col("_xm_n") > 0).then(pl.col("_xm_s") / pl.col("_xm_n")).alias(out)
    ).drop("_xm_s", "_xm_n")


__all__ = [
    "CumMax",
    "CumSum",
    "IsFirst",
    "IsLast",
    "Lag",
    "RowIndex",
    "exact_mean",
    "exact_sums",
    "grouped_scan",
]
