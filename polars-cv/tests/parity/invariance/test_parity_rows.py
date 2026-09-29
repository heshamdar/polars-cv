"""A row's answer depends on that row alone.

Two claims about batches, neither implied by the other or by the axis sweep:

* **per-row arguments resolve to that row's value** — a step whose arguments
  vary per row (expression columns) must give, on every row, exactly what
  the same step with that row's values as *literals* gives on that row alone;
* **a row's result does not depend on its neighbours** — every row of a batch
  (with null rows mixed in, split across chunks, under each engine) equals
  the same row executed alone.

``test_expression_op_params.py`` checks the first on one fixed input per
parameter; here the inputs, the ops and the per-row values are all drawn.
"""

from __future__ import annotations

import inspect

from hypothesis import assume, event
from hypothesis import strategies as st

from polars_cv import Pipeline
from tests.conftest import plugin_required
from tests.parity.framework import known
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.cases import (
    baseline_axes,
    batches,
    batches_for,
    candidate_specs,
    steps_for,
)
from tests.parity.framework.checks import ParityFailure, same_output
from tests.parity.framework.oracle import OPS
from tests.parity.framework.run import (
    ENGINES,
    PlanRefused,
    execute,
    expression_eligible,
)

pytestmark = plugin_required


def _eligible_methods() -> list[str]:
    """Buffer ops with at least one per-row parameter, by the live signatures."""
    return sorted(
        spec.method
        for spec in OPS.values()
        if spec.domain_in == "buffer"
        and any(
            expression_eligible(spec.method, name)
            for name in inspect.signature(getattr(Pipeline, spec.method)).parameters
        )
    )


PER_ROW = _eligible_methods()


@property_lanes(weight=2, data=st.data())
def test_per_row_arguments_match_literals(data: st.DataObject) -> None:
    """Each row with a PerRow argument equals that row alone with a literal."""
    spec = OPS[data.draw(st.sampled_from(PER_ROW), label="method")]
    batch = data.draw(
        batches_for(spec, reference=False, heterogeneous=False, nulls=True),
        label="batch",
    )
    assume(len(batch.present) > 1)
    step = data.draw(steps_for(spec, batch, per_row=True), label="step")
    assume(step.varies)
    for row, x in enumerate(batch.images):
        if x is not None and known.append_divergence(
            [], step.literal_for_row(row), [x]
        ):
            event("known divergence")
            return
    axes = baseline_axes(batch.images, spec.domain_out).but(
        params=data.draw(st.sampled_from(["column", "derived"]), label="style"),
        engine=data.draw(st.sampled_from(ENGINES), label="engine"),
    )
    divergence = known.axes_divergence(axes, batch.images, [], [step])
    if divergence is not None:
        event(f"known divergence: {divergence.key}")
        return
    try:
        together = execute(batch.images, [step], axes)
    except PlanRefused as exc:
        raise ParityFailure(f"planner refused a per-row {step!r}: {exc}") from exc
    for row, x in enumerate(batch.images):
        if x is None:
            continue
        alone = execute(
            [x], [step.literal_for_row(row)], baseline_axes([x], spec.domain_out)
        )
        difference = same_output(together.rows[row], alone.rows[0])
        if difference is not None:
            raise ParityFailure(
                f"row {row}: {step!r} with per-row expressions differs from the "
                f"row's literal {step.literal_for_row(row)!r} alone: {difference}"
            )


@property_lanes(weight=2, data=st.data())
def test_a_row_does_not_depend_on_its_neighbours(data: st.DataObject) -> None:
    """Every row of a batch equals that row executed alone."""
    batch = data.draw(batches(max_rows=6, max_side=16), label="batch")
    assume(len(batch.images) > 1)
    proxy = batch.proxy()
    specs = candidate_specs(proxy, allow_terminal=True, require_ref=False)
    if not all(im.shape == batch.present[0].shape for im in batch.present):
        specs = [s for s in specs if not s.uniform_rows]
    spec = data.draw(st.sampled_from(specs), label="op")
    step = data.draw(steps_for(spec, batch, per_row=False), label="step")
    if known.append_divergence([], step, batch.present):
        event("known divergence")
        return
    domain = spec.domain_out
    axes = baseline_axes(batch.images, domain).but(
        engine=data.draw(st.sampled_from(ENGINES), label="engine"),
        chunked=data.draw(st.booleans(), label="chunked"),
    )
    divergence = known.axes_divergence(axes, batch.images, [], [step])
    if divergence is not None:
        event(f"known divergence: {divergence.key}")
        return
    try:
        together = execute(batch.images, [step], axes)
    except PlanRefused:
        event(f"planner refused {step.method}")
        return
    for row, x in enumerate(batch.images):
        if x is None:
            if together.rows[row] is not None:
                raise ParityFailure(
                    f"row {row}: a null row produced {together.rows[row]!r}"
                )
            continue
        alone = execute([x], [step], baseline_axes([x], domain))
        difference = same_output(together.rows[row], alone.rows[0])
        if difference is not None:
            raise ParityFailure(
                f"row {row}: {step!r} in a batch of {len(batch.images)} "
                f"({axes}) differs from the row alone: {difference}"
            )
