"""Chains of ops against composed references.

A single-op sweep cannot see what only happens in combination: a fused
kernel, an op reading its input as a strided view it left in place, an
in-place write when nothing else reads the buffer, a plan whose facts after
one op mislead the next. So Hypothesis grows a chain one op at a time, each
drawn from the ops whose contract admits the *engine's actual output so far*
(its arguments drawn for that output too), and every prefix is executed as
one fused pipeline from the source.

Each step is checked twice (``framework/checks.ChainChecker``):

1. against its reference applied to the engine's own previous output — the
   incoming error is zero by construction, so the step's own tolerance
   applies however long the chain; and
2. the whole chain against the composed references, with the tolerance
   carried through each step's gain (``framework/tolerance.propagate``),
   until a discontinuous step makes the bound meaningless.
"""

from __future__ import annotations

from hypothesis import event
from hypothesis import strategies as st

from tests.conftest import plugin_required
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.cases import draw_step, lossless_sources
from tests.parity.framework.checks import ChainChecker
from tests.parity.framework.images import image_specs
from tests.parity.framework.run import ENGINES, OPTIMIZATION, Axes

pytestmark = plugin_required

#: The dtypes chains start from. u8 is what images are; f32 is what they
#: become; the others reach every op's integer and wide paths.
_CHAIN_DTYPES = ("u8", "u8", "f32", "u16", "i16", "f64")

MAX_STEPS = 5


@property_lanes(weight=8, data=st.data())
def test_chain_matches_composed_references(data: st.DataObject) -> None:
    """A random chain, every step and the whole against the references."""
    image = data.draw(
        image_specs(dtypes=_CHAIN_DTYPES, max_side=24), label="image"
    ).render()
    axes = Axes(
        source=data.draw(st.sampled_from(lossless_sources([image])), label="source"),
        engine=data.draw(st.sampled_from(ENGINES), label="engine"),
        optimize=data.draw(st.sampled_from(OPTIMIZATION), label="optimize"),
    )
    checker = ChainChecker(image, axes)
    length = data.draw(st.integers(1, MAX_STEPS), label="length")
    for index in range(length):
        last = index == length - 1
        step = draw_step(
            data, checker.actual, allow_terminal=last, label=f"step {index}"
        )
        if not checker.admits(step) or not checker.push(step):
            continue
        if checker.domain != "buffer":
            break
    event(f"steps checked against a reference: {checker.checked_steps}")
    event(f"end-to-end bound held to the end: {checker.tol is not None}")
