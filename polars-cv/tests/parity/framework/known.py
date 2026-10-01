"""Verified divergences this suite has found and not yet fixed.

The parity sweeps are generative, so a known bug cannot simply be left
failing: every run would rediscover it, and the per-push lane would be red
until it is fixed. Nor can it be absorbed into a tolerance or dropped from the
generator quietly — that is how a sweep ends up covering less than it claims.

So each divergence is recorded here once, with:

* a **predicate** saying which cases it covers, which the checks consult to
  *skip that one comparison* (reported through ``hypothesis.event``, so the
  statistics show how often it fires). The case is still generated and still
  executed along every other axis; only the comparison the bug breaks is
  withheld.
* a **repro**: the minimal case, asserting the *correct* behaviour. It runs as
  a strict ``xfail`` in ``oracle/test_parity_known_divergences.py``, so the day the
  bug is fixed it XPASSes, the suite goes red, and the entry (predicate and
  all) has to be deleted — which puts the fixed path back under the sweeps.

An entry must be a *confirmed* defect: reproduced against the running engine
with an independent decoder or reference. Suspicions do not go here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

if TYPE_CHECKING:
    from tests.parity.framework.run import Axes, Step


@dataclass(frozen=True)
class Divergence:
    """One confirmed, unfixed defect.

    Attributes:
        key: Stable identifier (used as the xfail test id).
        summary: What is wrong, and what "fixed" looks like.
        affects_axes: ``(axes, images, output_shapes, steps) -> bool`` for a
            defect of an execution axis (a source, sink, frame layout,
            parameter style), possibly only for some ops.
        affects_step: ``(step, input) -> bool`` for a defect of one op on one
            kind of input.
        affects_chain: ``(steps) -> bool`` for a defect that depends on what
            came before: called with the chain up to and including the step
            about to be appended.
        repro: Runs the minimal case and asserts the correct behaviour.
        avoid: Whether the defect cannot be carried forward: it raises, or
            it leaves a state no later step is meant to handle (a zero-sized
            image). Such a case is never executed or appended to a chain; the
            checks skip it and count it with ``event``.
        raises: The exception the repro fails with today: ``AssertionError``
            for a wrong answer, the engine's error type for one it refuses.
        match: A fragment of that exception's message. Together with
            *raises* it pins the failure to the defect, so a repro broken for
            another reason (a renamed helper, a changed fixture) is reported
            rather than read as "still reproduces" (:func:`still_reproduces`).
    """

    key: str
    summary: str
    repro: Callable[[], None]
    affects_axes: (
        Callable[[Axes, Sequence[Any], Sequence[tuple[int, ...]], Sequence[Step]], bool]
        | None
    ) = None
    affects_step: Callable[[Step, np.ndarray], bool] | None = None
    affects_chain: Callable[[Sequence[Step]], bool] | None = None
    avoid: bool = False
    raises: type[BaseException] = AssertionError
    match: str = ""


def still_reproduces(divergence: Divergence) -> None:
    """Run *divergence*'s repro and require it to fail *for its defect*.

    Raises ``AssertionError`` when the repro passes (the defect is fixed:
    delete the entry, which puts the path back under the sweeps) or fails
    some other way (the repro is broken, not the engine).
    """
    try:
        divergence.repro()
    except divergence.raises as exc:
        if divergence.match not in str(exc):
            msg = (
                f"{divergence.key}: the repro raised {type(exc).__name__} but "
                f"not with {divergence.match!r}: {str(exc)[:300]}"
            )
            raise AssertionError(msg) from exc
        return
    except Exception as exc:
        msg = (
            f"{divergence.key}: the repro failed with {type(exc).__name__}, not "
            f"{divergence.raises.__name__}: {str(exc)[:300]}"
        )
        raise AssertionError(msg) from exc
    msg = (
        f"{divergence.key}: the repro passes. The defect is fixed: delete its "
        "entry (predicate included) so the sweeps cover the path again."
    )
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Repros
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

DIVERGENCES: tuple[Divergence, ...] = ()


def _first(matches: Sequence[Divergence]) -> Divergence | None:
    """The match a caller must act on: an ``avoid`` entry if any matches.

    Several entries can cover one case (a large i32 image resized to zero
    width was both a value-only precision entry and ``derived-extent-zero``).
    Returning the first in registry order would let a value-only entry mask
    one whose case must not be executed at all.
    """
    for divergence in matches:
        if divergence.avoid:
            return divergence
    return matches[0] if matches else None


def axes_divergence(
    axes: Axes,
    images: Sequence[Any],
    shapes: Sequence[tuple[int, ...]],
    steps: Sequence[Step] = (),
) -> Divergence | None:
    """The known divergence executing *steps* along *axes* would hit, if any."""
    return _first(
        [
            d
            for d in DIVERGENCES
            if d.affects_axes is not None
            and d.affects_axes(axes, images, shapes, steps)
        ]
    )


def step_divergence(step: Step, x: np.ndarray) -> Divergence | None:
    """The known divergence *step* on input *x* would hit, if any."""
    return _first(
        [
            d
            for d in DIVERGENCES
            if d.affects_step is not None and d.affects_step(step, x)
        ]
    )


def chain_divergence(steps: Sequence[Step]) -> Divergence | None:
    """The known divergence the last of *steps* would hit, given the rest."""
    return _first(
        [
            d
            for d in DIVERGENCES
            if d.affects_chain is not None and d.affects_chain(steps)
        ]
    )


def append_divergence(
    steps: Sequence[Step], step: Step, inputs: Sequence[np.ndarray]
) -> Divergence | None:
    """The divergence to avoid when appending *step* after *steps*, if any.

    *inputs* are the arrays the step would read (one per non-null row).
    Only ``avoid`` entries count: a value-only divergence is withheld at
    comparison time, not by refusing to build the case.
    """
    candidates = [chain_divergence([*steps, step])]
    candidates += [step_divergence(step, x) for x in inputs]
    for divergence in candidates:
        if divergence is not None and divergence.avoid:
            return divergence
    return None
