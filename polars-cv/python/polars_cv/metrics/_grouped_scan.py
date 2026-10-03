"""Grouped ordered scans and exact sums that stay on the streaming engine.

The metrics need two things polars' streaming engine does not run natively:

* **a scan restarted per group**: ``cum_sum().over(keys)``,
  ``cum_max().over(keys)``, ``shift().over(keys)`` and
  ``int_range(pl.len()).over(keys)``;
* **a deterministic sum inside a group-by**. The old ``sort().cum_sum()``
  inside ``agg`` was deterministic but not native.

The streaming engine runs each of these through an ``in-memory-map`` node,
which collects its whole input first. That defeats streaming for exactly the
frames that grow largest: bootstrap replicates are ``n_bootstrap ×
detections`` rows. This module is the one way the metrics express either
operation; ``tests/test_streaming_plans.py`` rejects a new fallback.

**The grouped scan** sorts once by ``(*keys, *by)``, so every group is a
contiguous run. Each scan then runs over the *whole* frame and is reset at
group starts with arithmetic that is exact:

* A cumulative sum subtracts, from the frame-wide running sum, its value just
  before the group starts. Done in floats, that subtraction cancels
  catastrophically once earlier groups are large, so floats are summed in
  Int128 fixed point (:func:`_fixed`). Integers are summed as they are,
  which is exact.
* A cumulative max encodes each value's dense rank under its group index, so a
  later group can never win an earlier group's maximum. The rank is then
  decoded back to the value. Ranking orders floats exactly, so the selected
  value is bit-identical to ``.over``'s.
* A lag nulls the rows whose predecessor belongs to another group.

Every step here is a native streaming node: ``sort``, ``with-row-index``,
``cum_sum``, ``forward_fill``, ``shift``, ``cum_max``, a ``rank`` and one
equi-join. ``sort``, ``rank`` and a reverse ``cum_max`` buffer their column;
none of them hands the plan to the in-memory engine.

**Exact sums** (:func:`exact_sums`) convert each value to Int128 fixed point.
The scale is a power of two chosen from the frame's largest magnitude and row
count, and the conversion rounds once. Integer addition is associative, so a
sum does not depend on how the rows were chunked, ordered or split across
threads. A float ``sum`` does: its last bits change from run to run, which
made bootstrap bounds irreproducible.

Error: a term keeps at least ``125 − log2(rows)`` bits below the frame's
largest magnitude (absolute error ≤ ``max·rows·2⁻¹²⁵`` per term), and the sum
is rounded to Float64 once. Both are below Float64's own resolution for any
frame polars can hold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

_I = "_gs_i"
_GROUP = "_gs_g"
_NEW = "_gs_new"

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

    Equivalent to sorting the same way and writing ``scan.over(keys)``,
    including null group keys, which form a group of their own. The order
    ``by`` gives within a group should be total; ties would make the result
    depend on which tied row the sort put first, as with ``.over``.

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
    schema = lf.collect_schema()
    names = list(schema.names())
    ordered = lf.sort(
        *keys,
        *by,
        descending=[False] * len(keys) + list(descending),
        nulls_last=nulls_last,
        maintain_order=True,
    ).with_row_index(_I)

    starts = pl.col(_I) == 0
    for k in keys:
        starts = starts | pl.col(k).ne_missing(pl.col(k).shift(1))
    framed = ordered.with_columns(starts.alias(_NEW)).with_columns(
        (pl.col(_NEW).cast(pl.Int64).cum_sum() - 1).alias(_GROUP)
    )

    fixed = {
        scan.col: _fixed(pl.col(scan.col))
        for scan in out.values()
        if isinstance(scan, CumSum) and schema[scan.col].is_float()
    }
    framed = framed.with_columns(
        *(v.value.alias(f"_gs_fx_{c}") for c, v in fixed.items()),
        *(v.scale.alias(f"_gs_sc_{c}") for c, v in fixed.items()),
        *(
            e.alias(f"_gs_{k}_{c}")
            for c, v in fixed.items()
            for k, e in v.specials.items()
        ),
    )

    exprs: list[pl.Expr] = []
    lookups: list[tuple[str, str]] = []
    for name, scan in out.items():
        if isinstance(scan, CumSum):
            exprs.append(_cum_sum(scan.col, scan.col in fixed).alias(name))
        elif isinstance(scan, CumMax):
            exprs.append(_cum_max_rank(scan).alias(f"_gs_r_{name}"))
            lookups.append((name, scan.col))
        elif isinstance(scan, Lag):
            same = pl.col(_GROUP) == pl.col(_GROUP).shift(scan.n)
            exprs.append(pl.when(same).then(pl.col(scan.col).shift(scan.n)).alias(name))
        elif isinstance(scan, RowIndex):
            first = pl.when(pl.col(_NEW)).then(pl.col(_I)).forward_fill()
            exprs.append((pl.col(_I) - first).cast(pl.Int64).alias(name))
        elif isinstance(scan, IsFirst):
            exprs.append(pl.col(_NEW).alias(name))
        elif isinstance(scan, IsLast):
            last = pl.col(_GROUP).ne_missing(pl.col(_GROUP).shift(-1))
            exprs.append(last.alias(name))
        else:  # pragma: no cover - Scan is closed
            raise TypeError(f"not a scan: {scan!r}")
    framed = framed.with_columns(exprs)

    for name, col in lookups:
        # Rank → value. One value per rank (dense), so the join neither drops
        # nor duplicates rows, and `maintain_order` keeps the scan order.
        values = framed.select(
            pl.col(col).rank("dense").cast(pl.Int64).alias(f"_gs_r_{name}"),
            pl.col(col).alias(name),
        ).unique(subset=f"_gs_r_{name}")
        framed = framed.join(
            values, on=f"_gs_r_{name}", how="left", maintain_order="left"
        )
    return framed.select(*names, *out)


def _restarted(x: pl.Expr) -> pl.Expr:
    """The running sum of integer ``x`` restarted at each group start.

    The frame-wide running sum minus its value just before the group began:
    integer arithmetic, so the subtraction is exact.
    """
    run = x.cum_sum()
    return run - pl.when(pl.col(_NEW)).then(run - x).forward_fill()


def _cum_sum(col: str, fixed: bool) -> pl.Expr:
    """``col``'s running sum within its group (null where ``col`` is null)."""
    if fixed:
        finite = _restarted(pl.col(f"_gs_fx_{col}").fill_null(0)).cast(
            pl.Float64
        ) / pl.col(f"_gs_sc_{col}")
        nan, pinf, ninf = (_restarted(pl.col(f"_gs_{k}_{col}")) > 0 for k in _SPECIAL)
        within = _combine(finite, nan, pinf, ninf)
    else:
        within = _restarted(pl.col(col).fill_null(0))
    return pl.when(pl.col(col).is_not_null()).then(within)


def _cum_max_rank(scan: CumMax) -> pl.Expr:
    """The dense rank of the group's running max of ``scan.col``.

    The rank sits in the low 64 bits; the group index above it is subtracted
    (a reverse scan meets later groups first) or added (a forward scan meets
    earlier ones first), so the other groups always lose.
    """
    rank = pl.col(scan.col).rank("dense").cast(pl.Int128)
    shift = pl.col(_GROUP).cast(pl.Int128) * (1 << 64)
    if scan.reverse:
        return ((rank - shift).cum_max(reverse=True) + shift).cast(pl.Int64)
    return ((rank + shift).cum_max() - shift).cast(pl.Int64)


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
