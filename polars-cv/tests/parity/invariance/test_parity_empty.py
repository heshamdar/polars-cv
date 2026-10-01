"""An empty image is refused cleanly or processed: never a panic.

Every buffer op in the registry (``framework/oracle.OPS``) runs on images with
a zero height or width, with arguments drawn for a non-empty image of the same
dtype and channels. The op may refuse (its contract: a resample has nothing to
sample, an ordering reduction nothing to order) or produce an output; it may
not panic the engine. This is the empty-image slice of the generator's state
space, swept separately because "refused" is a correct answer here and a
failure everywhere else.

The images ride the ``raw`` source: an ``Array`` column with a zero-size
dimension cannot cross the plugin boundary at all (polars' FFI import drops
its rows), which the planner refuses cleanly (``test_a_zero_width_array_is
_refused``), and an empty ``list`` row reads as null by design.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import strategies as st

from polars_cv import Pipeline
from tests.conftest import plugin_required
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.images import DTYPES
from tests.parity.framework.oracle import OPS
from tests.parity.framework.run import Axes, NotApplicable, PlanRefused, Step, execute

pytestmark = plugin_required

BUFFER_OPS = sorted(m for m, s in OPS.items() if s.domain_in == "buffer")


@property_lanes(
    weight=0.5,
    marks=(pytest.mark.parametrize("method", BUFFER_OPS),),
    data=st.data(),
)
def test_an_empty_image_is_refused_or_processed(
    method: str, data: st.DataObject
) -> None:
    spec = OPS[method]
    dtype = data.draw(st.sampled_from(list(spec.ref_dtypes or DTYPES)), label="dtype")
    channels = data.draw(st.sampled_from([1, 2, 3, 4]), label="channels")
    proxy = np.zeros((5, 6, channels), DTYPES[dtype])
    if not spec.accepts(proxy):
        return
    params = data.draw(st.composite(lambda d: spec.params(d, proxy))(), label="params")
    empty_axis = data.draw(st.sampled_from([0, 1]), label="empty axis")
    shape = (0, 6, channels) if empty_axis == 0 else (5, 0, channels)
    image = np.zeros(shape, DTYPES[dtype])
    try:
        execute([image], [Step(method, params)], Axes(source="raw"))
    except (PlanRefused, NotApplicable, ValueError):
        pass  # refused before any data moved
    except pl.exceptions.ComputeError as exc:
        assert "panicked" not in str(exc), f"{method}{params} on {shape}: {exc}"


def test_a_zero_width_array_is_refused() -> None:
    """A zero-size ``Array`` dimension is refused at planning, naming the
    column; the FFI import of one panicked the plugin."""
    empty = np.zeros((0, 3, 1), np.float32)
    df = pl.DataFrame({"a": pl.Series([empty], dtype=pl.Array(pl.Float32, (0, 3, 1)))})
    query = pl.col("a").cv.pipe(Pipeline().source("array")).sink("numpy")
    with pytest.raises(pl.exceptions.ComputeError, match="zero-size dimension"):
        df.select(query)
