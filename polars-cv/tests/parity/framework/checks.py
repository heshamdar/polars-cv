"""The comparisons, shared by every parity test module.

Three checks, one per kind of evidence:

* :func:`check_step` — one op against its reference, row by row.
* :func:`ChainChecker` — a chain against its references, two ways: each step
  against the reference applied to the engine's *own* previous output (so the
  incoming error is zero and the step's own tolerance applies), and the whole
  chain against the composed references with the tolerance propagated
  (:func:`tolerance.propagate`).
* :func:`check_invariant` — the same case along several axes must produce
  byte-identical output. No reference is involved, so this covers every op,
  including those the reference table cannot model.

Every failure message ends in a repro: the images' description, the chain as
Python, and the axes, so a shrunk Hypothesis example can be pasted into a
regression test directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from hypothesis import event as _hyp_event
from hypothesis import note as _hyp_note
from hypothesis.control import currently_in_test_context

from tests.parity.framework import known
from tests.parity.framework.oracle import OPS, OpSpec
from tests.parity.framework.run import (
    Axes,
    NotApplicable,
    PlanRefused,
    Result,
    Step,
    execute,
    render_chain,
)
from tests.parity.framework.tolerance import EXACT, Tol, compare, propagate


def event(value: str) -> None:
    """``hypothesis.event``, or nothing when called outside a Hypothesis test
    (the framework fixtures call the checks directly)."""
    if currently_in_test_context():
        _hyp_event(value)


def note(value: str) -> None:
    """``hypothesis.note``, or nothing outside a Hypothesis test."""
    if currently_in_test_context():
        _hyp_note(value)


class ParityFailure(AssertionError):
    """A parity check failed. The message carries the repro."""


def _describe(images: Sequence[Any]) -> str:
    parts = []
    for im in images:
        if im is None:
            parts.append("None")
        else:
            parts.append(f"{im.dtype}{list(im.shape)}")
    return "[" + ", ".join(parts) + "]"


def repro(images: Sequence[Any], steps: Sequence[Step], axes: Axes) -> str:
    """A one-paragraph description of a case, for failure messages."""
    return (
        f"images={_describe(images)}\n"
        f"chain=Pipeline().source(...){render_chain(steps)}\n"
        f"axes={axes}"
    )


def _fail(
    message: str, images: Sequence[Any], steps: Sequence[Step], axes: Axes
) -> None:
    raise ParityFailure(f"{message}\n--- repro ---\n{repro(images, steps, axes)}")


def _check_schema(
    result: Result, images: Sequence[Any], steps: Sequence[Step], axes: Axes
) -> None:
    if result.planned != result.realized:
        _fail(
            f"planned dtype {result.planned!r} != executed {result.realized!r}",
            images,
            steps,
            axes,
        )


def _check_nulls(
    result: Result, images: Sequence[Any], steps: Sequence[Step], axes: Axes
) -> None:
    for i, (image, out) in enumerate(zip(images, result.rows)):
        if (image is None) != (out is None):
            _fail(
                f"row {i}: input is {'null' if image is None else 'present'} but "
                f"output is {'null' if out is None else 'present'}",
                images,
                steps,
                axes,
            )


def reference_inputs(images: Sequence[Any], axes: Axes) -> list[Any]:
    """What the engine actually decoded: the images themselves, unless the
    source is lossy, in which case the engine's own identity decode."""
    from tests.parity.framework.io import SOURCES

    if SOURCES[axes.source].lossless:
        return list(images)
    decoded = execute(images, [], axes.but(sink="numpy", params="literal"))
    return decoded.rows


# ---------------------------------------------------------------------------
# One step against its reference
# ---------------------------------------------------------------------------


def check_step(
    images: Sequence[Any],
    step: Step,
    axes: Axes = Axes(),
    *,
    spec: OpSpec | None = None,
) -> Result | None:
    """Execute one step and compare every row with its reference.

    Rows the reference does not model (:meth:`OpSpec.has_reference`) or that
    hit a known divergence are skipped and counted with ``event``. *spec*
    overrides the table's entry (the framework fixtures use it to watch a
    wrong reference get caught).
    """
    spec = spec if spec is not None else OPS[step.method]
    note(f"step: {repro(images, [step], axes)}")
    for row, x in enumerate(images):
        if x is None:
            continue
        divergence = known.append_divergence([], step.literal_for_row(row), [x])
        if divergence is not None:
            event(f"known divergence: {divergence.key}")
            return None
    if spec.domain_out != "buffer":
        axes = axes.but(sink="native")
    divergence = known.axes_divergence(axes, images, [], [step])
    if divergence is not None and divergence.avoid:
        event(f"known divergence: {divergence.key}")
        return None
    result = execute(images, [step], axes)
    _check_schema(result, images, [step], axes)
    _check_nulls(result, images, [step], axes)
    divergence = known.axes_divergence(axes, images, output_shapes(result), [step])
    if divergence is not None:
        event(f"known divergence: {divergence.key}")
        return result
    inputs = reference_inputs(images, axes)
    for row, (x, out) in enumerate(zip(inputs, result.rows)):
        if x is None:
            continue
        literal = step.literal_for_row(row)
        if not spec.has_reference(x, literal.params):
            event(f"{step.method}: no reference for this input")
            continue
        divergence = known.step_divergence(literal, x)
        if divergence is not None:
            event(f"known divergence: {divergence.key}")
            continue
        expected = spec.ref(x, dict(literal.params))
        mismatch = compare(out, expected, spec.tolerance(x, dict(literal.params)))
        if mismatch is not None:
            _fail(
                f"row {row}: {step!r} disagrees with its reference: {mismatch}",
                images,
                [step],
                axes,
            )
    return result


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------


@dataclass
class ChainChecker:
    """Grows a chain one step at a time, checking each step as it goes.

    The chain generator calls :meth:`push` after drawing each step; the
    checker executes the prefix (as one fused pipeline, from the source),
    compares it to the reference two ways, and makes the result the input the
    generator draws the next step's parameters from.

    Attributes:
        image: The source image (one row).
        axes: How every prefix is executed.
        steps: The chain so far.
        actual: The engine's output after the last step.
        expected: The composed references' output (``None`` once a step has
            no reference for its input).
        tol: The propagated end-to-end tolerance (``None``: unbounded).
    """

    image: np.ndarray
    axes: Axes = field(default_factory=Axes)
    steps: list[Step] = field(default_factory=list)
    actual: Any = None
    expected: Any = None
    tol: Tol | None = EXACT
    checked_steps: int = 0

    def __post_init__(self) -> None:
        start = reference_inputs([self.image], self.axes)[0]
        self.actual = start
        self.expected = start

    @property
    def domain(self) -> str:
        return "buffer" if not self.steps else OPS[self.steps[-1].method].domain_out

    def admits(self, step: Step) -> bool:
        """Whether *step* can be appended (no known divergence makes it raise)."""
        divergence = known.append_divergence(self.steps, step, [self.actual])
        if divergence is not None:
            event(f"known divergence: {divergence.key}")
            return False
        return True

    def push(self, step: Step) -> bool:
        """Append *step*, execute the new prefix and check it.

        Returns ``False`` (and appends nothing) when the planner refuses the
        step — a plan-time fact the generator does not model.
        """
        spec = OPS[step.method]
        previous = self.actual
        axes = (
            self.axes if spec.domain_out == "buffer" else self.axes.but(sink="native")
        )
        try:
            result = execute([self.image], [*self.steps, step], axes)
        except PlanRefused:
            event(f"planner refused {step.method} mid-chain")
            return False
        self.steps.append(step)
        note(f"chain so far: {render_chain(self.steps)}")
        _check_schema(result, [self.image], self.steps, axes)
        self.actual = result.rows[0]

        # 1. This step against its reference, fed the engine's own previous
        #    output: exact input, so the step's own tolerance applies.
        params = dict(step.params)
        modelled = spec.has_reference(previous, params)
        divergence = known.step_divergence(step, previous) if modelled else None
        if divergence is not None:
            event(f"known divergence: {divergence.key}")
        elif modelled:
            mismatch = compare(
                self.actual,
                spec.ref(previous, params),
                spec.tolerance(previous, params),
            )
            if mismatch is not None:
                _fail(
                    f"step {len(self.steps)} {step!r} disagrees with its reference "
                    f"on the engine's own previous output: {mismatch}",
                    [self.image],
                    self.steps,
                    self.axes,
                )
            self.checked_steps += 1
        else:
            event(f"{step.method}: no reference for this input")

        # 2. The whole chain against the composed references.
        self._advance_end_to_end(spec, params, modelled and divergence is None)
        return True

    def _advance_end_to_end(self, spec: OpSpec, params: dict, modelled: bool) -> None:
        if self.expected is None or self.tol is None:
            return
        if not modelled or not spec.has_reference(self.expected, params):
            self.expected, self.tol = None, None
            return
        own = spec.tolerance(self.expected, params)
        expected = spec.ref(self.expected, params)
        self.tol = propagate(
            self.tol,
            own,
            spec.gain_for(params),
            kind=spec.kind,
            integer_out=np.asarray(expected).dtype.kind in "iu",
        )
        self.expected = expected
        if self.tol is None:
            event("end-to-end bound lost (discontinuous or sparse)")
            self.expected = None
            return
        mismatch = compare(self.actual, self.expected, self.tol)
        if mismatch is not None:
            _fail(
                f"after {len(self.steps)} steps the chain drifted from the composed "
                f"references beyond the propagated bound: {mismatch}",
                [self.image],
                self.steps,
                self.axes,
            )


# ---------------------------------------------------------------------------
# Invariance
# ---------------------------------------------------------------------------


def same_output(a: Any, b: Any) -> str | None:
    """``None`` if two decoded outputs are identical, else how they differ.

    Arrays must agree in dtype, shape and every element (NaN equal to NaN);
    anything else (contour structures) by equality.
    """
    if a is None or b is None:
        return None if a is None and b is None else f"{a!r} vs {b!r}"
    if (
        isinstance(a, np.ndarray)
        or isinstance(b, np.ndarray)
        or isinstance(a, np.generic)
    ):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype:
            return f"dtype {a.dtype} vs {b.dtype}"
        if a.shape != b.shape:
            return f"shape {a.shape} vs {b.shape}"
        if a.dtype.kind == "f":
            differs = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        else:
            differs = a != b
        if np.any(differs):
            where = tuple(np.argwhere(differs)[0])
            count = int(np.count_nonzero(differs))
            return f"{count} elements differ; first at {where}: {a[where]!r} vs {b[where]!r}"
        return None
    return None if a == b else f"{a!r} vs {b!r}"


def output_shapes(result: Result) -> list[tuple[int, ...]]:
    """The non-null rows' output shapes (for sinks that need them)."""
    return [
        np.shape(r) for r in result.rows if r is not None and isinstance(r, np.ndarray)
    ]


def check_invariant(
    images: Sequence[Any],
    steps: Sequence[Step],
    baseline: Axes,
    variants: Sequence[Axes],
) -> Result:
    """Execute along *baseline* and every *variant*; all must agree exactly."""
    note(f"invariance case: {repro(images, steps, baseline)}")
    base = execute(images, steps, baseline)
    _check_schema(base, images, steps, baseline)
    _check_nulls(base, images, steps, baseline)
    shapes = output_shapes(base)
    for axes in variants:
        divergence = known.axes_divergence(axes, images, shapes, steps)
        if divergence is not None:
            event(f"known divergence: {divergence.key}")
            continue
        try:
            other = execute(images, steps, axes, shapes=shapes)
        except PlanRefused:
            event("variant refused by the planner")
            continue
        except NotApplicable:
            event("variant not expressible")
            continue
        _check_schema(other, images, steps, axes)
        for row, (a, b) in enumerate(zip(base.rows, other.rows)):
            difference = same_output(a, b)
            if difference is not None:
                _fail(
                    f"row {row}: output along {axes} differs from the baseline "
                    f"{baseline}: {difference}",
                    images,
                    steps,
                    axes,
                )
    return base
