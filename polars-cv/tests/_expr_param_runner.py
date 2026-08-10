"""Execution harness for per-row (expression-valued) operation parameters.

Three independent claims are made about a parameter that accepts a
``pl.Expr``, and each needs its own check because none implies the others:

1. **The expression resolves to that row's value.** Compared against the same
   pipeline built with the row's value as a *literal*. An off-by-one in
   ``row_idx`` produces rows that differ from each other but not from each
   other's literals, so "the rows differ" does not cover this.
2. **The value reaches the kernel.** Compared across rows: distinct inputs
   must give distinct outputs. Claim 1 cannot see a parameter the kernel
   ignores entirely — it would be ignored identically on both paths, and the
   comparison would pass. This is the check that caught expression arguments
   being *accepted and dropped* in the geometry namespaces.
3. **A row's result does not depend on its neighbours.** The same pipeline is
   run over the batch and over each row on its own. This is the claim that
   morsel boundaries, broadcasting and the compiled-graph cache can break, and
   it is the only one available for a parameter with no literal spelling.

The comparison is exact. Both legs run the same kernels on the same bytes; a
tolerance here would mask precisely the small resolution errors (a truncated
``f32``, a value read from the wrong row of a nearly-constant column) that the
harness exists to find.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import polars as pl

from polars_cv import Pipeline

#: Sink per output domain. ``list`` keeps a buffer's nesting and dtype without
#: fixing its shape (an expression-valued ``resize`` has no plan-time shape, so
#: the ``array`` sink cannot be used), and ``native`` is the domain-appropriate
#: Polars type for everything else.
SINK_FOR_DOMAIN: dict[str, str] = {
    "buffer": "list",
    "contour": "native",
    "scalar": "native",
    "vector": "native",
}


def sink_for(pipe: Pipeline) -> str:
    """The sink that renders *pipe*'s output as comparable Polars values."""
    domain = pipe.current_domain()
    try:
        return SINK_FOR_DOMAIN[domain]
    except KeyError:  # pragma: no cover - a new domain must choose a sink
        msg = f"no sink registered for the {domain!r} domain"
        raise AssertionError(msg) from None


def run(
    df: pl.DataFrame,
    input_column: str,
    pipe: Pipeline,
    *,
    engine: str = "in-memory",
    name: str = "out",
    sink: str | None = None,
) -> list:
    """Execute *pipe* over *df* and return the output column as Python values.

    *sink* defaults to :func:`sink_for`. Naming one is for callers whose
    pipeline cannot use the domain default — the ``list`` sink needs a
    concrete element dtype, which an image source without ``dtype=`` does not
    have.
    """
    expr = pl.col(input_column).cv.pipe(pipe).sink(sink or sink_for(pipe))
    return df.lazy().with_columns(**{name: expr}).collect(engine=engine)[name].to_list()


def assert_matches_per_row_literals(
    df: pl.DataFrame,
    *,
    input_column: str,
    param_column: str,
    build: Callable[[Any], Pipeline],
    values: Sequence[Any],
    label: str = "",
    sink: str | None = None,
) -> list:
    """Claim 1: each row equals the pipeline built with that row's literal.

    The literal leg runs on a one-row frame, so a row that matches has also
    been shown not to depend on its neighbours (claim 3) for the literal path.

    Args:
        df: Frame carrying *input_column* and *param_column*, one row per value.
        input_column: The column the pipeline reads.
        param_column: The column holding the per-row parameter values.
        build: Builds the pipeline from one parameter value (literal or
            expression).
        values: The per-row values, in row order.
        label: Prefix for assertion messages.
        sink: Sink override, as for :func:`run`.

    Returns:
        The dynamic (expression-driven) output, one entry per row.
    """
    assert df.height == len(values), "one row per parameter value"
    dynamic = run(df, input_column, build(pl.col(param_column)), sink=sink)

    for i, value in enumerate(values):
        expected = run(df.slice(i, 1), input_column, build(value), sink=sink)
        assert dynamic[i] == expected[0], (
            f"{label}row {i} with {param_column}={value!r} does not match the "
            f"pipeline built with that value as a literal"
        )
    return dynamic


def assert_rows_are_independent(
    df: pl.DataFrame,
    *,
    input_column: str,
    pipe: Pipeline,
    label: str = "",
) -> list:
    """Claim 3: a row's result is the same alone as it is in the batch."""
    batched = run(df, input_column, pipe)
    for i in range(df.height):
        alone = run(df.slice(i, 1), input_column, pipe)
        assert batched[i] == alone[0], (
            f"{label}row {i} differs between a batched run and a one-row run"
        )
    return batched


def assert_values_vary(outputs: Sequence[Any], *, label: str = "") -> None:
    """Claim 2: distinct parameter values produced distinct outputs."""
    rendered = [repr(value) for value in outputs]
    assert len(set(rendered)) == len(rendered), (
        f"{label}two rows with different parameter values produced the same "
        f"output — the value never reached the kernel"
    )
