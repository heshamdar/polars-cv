"""Every op against an independent reference, across the input space.

For each op with a reference (``framework/oracle.py``), Hypothesis draws a
batch of inputs the op's contract admits — dtype, size, channel count, pixel
content, null rows, rows of different sizes — and arguments valid for them,
executes through a randomly chosen source and engine, and compares every row
with the reference at the op's declared tolerance.

Arguments sometimes vary per row. Those ride as expression columns
(``PerRow``), and each row is compared with the reference evaluated at *its*
value — the per-row resolution checked against ground truth rather than
against the literal path (which ``test_expression_op_params.py`` does).
"""

from __future__ import annotations

import pytest
from hypothesis import strategies as st

from tests.conftest import plugin_required
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.cases import batches_for, steps_for
from tests.parity.framework.checks import check_step
from tests.parity.framework.io import sources_for
from tests.parity.framework.oracle import OPS
from tests.parity.framework.run import ENGINES, Axes

pytestmark = plugin_required

REFERENCED = sorted(
    m for m, s in OPS.items() if s.ref is not None and s.domain_in == "buffer"
)


@property_lanes(
    weight=1,
    marks=(pytest.mark.parametrize("method", REFERENCED),),
    data=st.data(),
)
def test_op_matches_its_reference(method: str, data: st.DataObject) -> None:
    """One op, one batch, every row against the reference."""
    spec = OPS[method]
    batch = data.draw(batches_for(spec, reference=True), label="batch")
    per_row = data.draw(st.booleans(), label="per-row arguments")
    step = data.draw(steps_for(spec, batch, per_row=per_row), label="step")

    # A lossy source is fine here: the reference is fed what the engine
    # decoded (checks.reference_inputs).
    sources = sources_for(batch.present)
    axes = Axes(
        source=data.draw(st.sampled_from(sources), label="source"),
        engine=data.draw(st.sampled_from(ENGINES), label="engine"),
        params="column" if step.varies else "literal",
    )
    check_step(list(batch.images), step, axes)
