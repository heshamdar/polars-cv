"""Graphs with two branches: binary ops against their references.

Every lazy-only binary op (``add``, ``blend``, ``bitwise_and``, …) joins two
branches, each its own short chain, and a tail may follow the join. The
references are checked stage by stage, as chains are:

* each branch is executed on its own, and the op's reference applied to
  those *actual* branch outputs must match the joined graph's output, at the
  op's own tolerance;
* the tail is checked against its reference applied to the actual join.

And the graph must not depend on how it is executed: every engine, the
optimizer on and off, and — when both branches read one column — the same
graph over two identical columns. One source feeding two chains is where
common-subexpression elimination shares their prefix; the second column is
the same computation with nothing to share.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, event, note
from hypothesis import strategies as st

from tests.conftest import plugin_required
from tests.parity.framework import known
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.cases import batches
from tests.parity.framework.checks import ParityFailure, same_output
from tests.parity.framework.images import DTYPES, ImageSpec, render
from tests.parity.framework.oracle import BINARY, OPS
from tests.parity.framework.run import (
    ENGINES,
    OPTIMIZATION,
    Axes,
    BinaryCase,
    PlanRefused,
    Step,
    execute,
    execute_binary,
)
from tests.parity.framework.tolerance import compare

pytestmark = plugin_required

#: Branch ops that keep a buffer's shape and dtype, so the two branches stay
#: joinable whatever each draws. Morphology needs a single channel.
_SHAPE_PRESERVING = ("flip", "flip_h", "flip_v", "invert", "blur")
_SINGLE_CHANNEL_ONLY = ("erode", "dilate", "morphology_open", "morphology_close")

#: Tail ops: pointwise or data movement with a reference on any dtype.
_TAILS = ("flip", "invert", "abs", "cast")


def _branch(data: st.DataObject, x: np.ndarray, label: str) -> list[Step]:
    names = list(_SHAPE_PRESERVING)
    if x.shape[2] == 1:
        names += list(_SINGLE_CHANNEL_ONLY)
    n = data.draw(st.integers(0, 2), label=f"{label} length")
    steps = []
    for i in range(n):
        spec = OPS[data.draw(st.sampled_from(names), label=f"{label} op {i}")]
        params = data.draw(
            st.composite(lambda d: spec.params(d, x))(), label=f"{label} args {i}"
        )
        steps.append(Step(spec.method, params))
    return steps


def _branch_output(images: tuple, steps: list[Step]) -> list:
    return execute(list(images), steps, Axes()).rows


@property_lanes(weight=2, data=st.data())
def test_binary_graph_matches_references(data: st.DataObject) -> None:
    """Two branches, a join, a tail: each stage against its reference."""
    method = data.draw(st.sampled_from(sorted(BINARY)), label="method")
    spec = BINARY[method]
    dtypes = [d for d, dt in DTYPES.items() if spec.accepts(np.zeros((1, 1, 1), dt))]
    batch = data.draw(
        st.sampled_from(dtypes).flatmap(
            lambda d: batches(
                dtypes=(d,), heterogeneous=False, nulls=False, max_side=16
            )
        ),
        label="left batch",
    )
    shared = data.draw(st.booleans(), label="shared source")
    left = batch.images
    if shared:
        right = left
    else:
        # The same image specs under a fresh seed: identical shape and dtype,
        # different pixels (and possibly a different content kind).
        salt = data.draw(st.integers(1, 2**16), label="right seed")
        content = data.draw(
            st.sampled_from(["noise", "edges", "constant", "extremes"]),
            label="right content",
        )
        right = tuple(
            None
            if s is None
            else render(
                ImageSpec(**{**s.__dict__, "seed": s.seed ^ salt, "content": content})
            )
            for s in batch.specs
        )
    sample = batch.present[0]
    case = BinaryCase(
        left=left,
        right=right,
        left_steps=tuple(_branch(data, sample, "left")),
        right_steps=tuple(_branch(data, sample, "right")),
        method=method,
        shared=shared,
    )
    note(f"graph: {case.describe()}")

    try:
        joined = execute_binary(case)
    except PlanRefused as exc:
        # Both branches keep shape and dtype, so the join must be accepted.
        raise ParityFailure(
            f"planner refused a joinable graph: {exc}\n{case.describe()}"
        ) from exc
    left_out = _branch_output(case.left, list(case.left_steps))
    right_out = _branch_output(case.right, list(case.right_steps))
    # Either operand can carry the defect (a small left and a wide right).
    divergence = next(
        (
            d
            for operand in (*left_out, *right_out)
            if operand is not None
            and (d := known.step_divergence(Step(method), operand)) is not None
        ),
        None,
    )
    if divergence is not None:
        event(f"known divergence: {divergence.key}")
        return
    for row, (a, b, out) in enumerate(zip(left_out, right_out, joined.rows)):
        if not spec.ref_accepts(a):
            event(f"{method}: no reference for this input")
            continue
        expected = spec.ref(a, b)
        mismatch = compare(out, expected, spec.tol(a))
        if mismatch is not None:
            raise ParityFailure(
                f"row {row}: {method} of the two branches disagrees with its "
                f"reference: {mismatch}\n{case.describe()}"
            )

    # The tail, against its reference on the actual join.
    tail_spec = OPS[data.draw(st.sampled_from(_TAILS), label="tail op")]
    tail_args = data.draw(
        st.composite(lambda d: tail_spec.params(d, joined.rows[0]))(), label="tail args"
    )
    with_tail = BinaryCase(
        **{**case.__dict__, "tail": (Step(tail_spec.method, tail_args),)}
    )
    tailed = execute_binary(with_tail)
    for row, (x, out) in enumerate(zip(joined.rows, tailed.rows)):
        if not tail_spec.has_reference(x, tail_args):
            event(f"{tail_spec.method}: no reference for this input")
            continue
        mismatch = compare(
            out, tail_spec.ref(x, dict(tail_args)), tail_spec.tolerance(x, tail_args)
        )
        if mismatch is not None:
            raise ParityFailure(
                f"row {row}: tail {tail_spec.method} after the join disagrees with "
                f"its reference: {mismatch}\n{with_tail.describe()}"
            )


@property_lanes(weight=2, data=st.data())
def test_binary_graph_is_execution_invariant(data: st.DataObject) -> None:
    """Same graph, every engine, optimizer on/off, shared or duplicated column."""
    method = data.draw(st.sampled_from(sorted(BINARY)), label="method")
    spec = BINARY[method]
    dtypes = [d for d, dt in DTYPES.items() if spec.accepts(np.zeros((1, 1, 1), dt))]
    batch = data.draw(
        st.sampled_from(dtypes).flatmap(
            lambda d: batches(
                dtypes=(d,), heterogeneous=False, nulls=False, max_side=16
            )
        ),
        label="batch",
    )
    sample = batch.present[0]
    case = BinaryCase(
        left=batch.images,
        right=batch.images,
        left_steps=tuple(_branch(data, sample, "left")),
        right_steps=tuple(_branch(data, sample, "right")),
        method=method,
        shared=True,
    )
    note(f"graph: {case.describe()}")
    try:
        base = execute_binary(case)
    except PlanRefused:
        assume(False)
    variants = [
        (f"engine={e}", lambda e=e: execute_binary(case, engine=e)) for e in ENGINES
    ]
    variants += [
        (f"optimize={v}", lambda v=v: execute_binary(case, optimize=v))
        for v in OPTIMIZATION
        if v != "all"
    ]
    duplicated = BinaryCase(**{**case.__dict__, "shared": False})
    variants.append(("two identical columns", lambda: execute_binary(duplicated)))
    for label, run in variants:
        other = run()
        for row, (a, b) in enumerate(zip(base.rows, other.rows)):
            difference = same_output(a, b)
            if difference is not None:
                raise ParityFailure(
                    f"row {row}: {label} differs from the baseline: {difference}\n"
                    f"{case.describe()}"
                )
