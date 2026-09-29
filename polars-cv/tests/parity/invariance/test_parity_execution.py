"""The answer does not depend on how the question is asked.

For a drawn batch and chain (any dtype, any op the contract admits, including
the ops no reference models: hashes, contour extraction and the contour ops
after it), the output must be byte-identical across every execution axis
(``framework/run.Axes``):

* **source** — every lossless source that can carry the rows (PNG, TIFF, WebP,
  a file path, ``list``, ``array``, ``raw``, ``blob``, ``auto``);
* **sink** — every exact sink that can carry the output (``numpy``,
  ``ndarray``, ``list``, ``array``, ``blob``, the lossless image codecs);
* **engine** — eager, lazy in-memory, lazy streaming;
* **parameters** — literals, ``pl.col``, ``pl.lit``, a computed expression;
* **composition** — one pipeline, split with ``.pipe()``, every prefix aliased
  in one multi-output graph, or every step materialized through ``blob``
  (nothing can fuse across a materialization);
* **optimizer** — every pass on, every pass off;
* **frame layout** — one chunk or several (a morsel boundary mid-column).

Each axis is swept one at a time from the baseline, plus one random
combination per example. No reference is involved, so a disagreement is
always the engine disagreeing with itself.
"""

from __future__ import annotations

from hypothesis import event
from hypothesis import strategies as st

from tests.conftest import plugin_required
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.cases import (
    axes_variants,
    baseline_axes,
    batches,
    draw_batch_chain,
)
from tests.parity.framework.checks import check_invariant, output_shapes

pytestmark = plugin_required


@property_lanes(weight=4, data=st.data())
def test_every_execution_axis_agrees(data: st.DataObject) -> None:
    """One case, executed along every applicable axis, one answer."""
    batch = data.draw(batches(max_side=20), label="batch")
    steps, base = draw_batch_chain(data, batch, max_steps=4)
    variants = data.draw(
        axes_variants(
            batch.images,
            steps,
            output_domain=base.info.domain,
            output_dtype=base.info.dtype,
            output_shapes=output_shapes(base),
        ),
        label="variants",
    )
    event(f"output domain: {base.info.domain}")
    event(f"chain length: {len(steps)}")
    baseline = baseline_axes(batch.images, base.info.domain)
    if steps and any(s.varies for s in steps):
        baseline = baseline.but(params="column")
    check_invariant(batch.images, steps, baseline, variants)
