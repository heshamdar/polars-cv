"""The sink contract: what the planner promises, execution must deliver.

Two functions decide what a sink produces, and they are two halves of one
contract:

* ``dtype_for_output`` (``graph/decode.rs``) — the Polars **dtype**, from
  ``(expected_domain, sink format)``;
* ``encode_node_output`` (``graph/encode.rs``) — the **value**.

The second used to key on the runtime ``NodeOutput`` *variant* instead, which
is a different fact: a domain can arrive in more than one representation. A
perceptual hash is a ``vector``-domain output riding as a ``Buffer`` (a 1-D
``u8`` buffer), while ``extract_shape`` produces a real ``Vector``. Wherever
those diverged the two halves disagreed, always the same way round — the plan
promised something execution then refused:

* ``perceptual_hash().sink("native")`` planned ``List(UInt8)`` and failed with
  "Buffer outputs require explicit format";
* ``extract_shape().sink("array", shape=[3])`` planned ``Array(Float64, 3)``
  and failed with "Unsupported sink format: array";
* the pairs that worked did so by the two dispatches happening to agree.

So the invariant here is not a list of blessed pairs — a list would need
updating for every new sink and would be one more thing to drift. It is the
relationship: **if a pair survives planning, execution must produce exactly the
dtype planning promised.** A pair rejected at plan time is fine, whatever the
reason; that is the planner doing its job, before any data moves.

``test_sink_matrix_is_complete`` keeps the sweep honest by pinning the format
axis to ``SinkFormat`` and the domain axis to the pipeline domains, so a new
sink format or domain cannot join without a case here.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline
from polars_cv._types import SinkFormat

from .conftest import plugin_required


@pytest.fixture(scope="module")
def rgb_df() -> pl.DataFrame:
    """A single 8x8x3 PNG row — small, and non-square in the channel axis."""
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buf, format="PNG")
    return pl.DataFrame({"img": [buf.getvalue()]})


def _base() -> Pipeline:
    """A buffer pipeline with fully known shape and a concrete dtype."""
    return (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=8, width=8, channels=3)
        .cast("u8")
    )


#: One pipeline per *representation* a domain can reach the sink in, with the
#: shape its ``array`` sink needs.
#:
#: ``vector`` appears twice on purpose. That a vector can arrive as either a
#: ``Vector`` or a ``Buffer`` is precisely what the two dispatches disagreed
#: about, so a sweep carrying only one of them would have passed throughout.
_PIPELINES: dict[str, tuple[str, "callable", list[int] | None]] = {
    "buffer": ("buffer", _base, [8, 8, 3]),
    "contour": (
        "contour",
        lambda: _base().grayscale().threshold(128).extract_contours(),
        None,
    ),
    "scalar": ("scalar", lambda: _base().reduce_sum(), None),
    "vector-as-vector": ("vector", lambda: _base().extract_shape(), [3]),
    "vector-as-buffer": (
        "vector",
        lambda: _base().perceptual_hash(hash_size=8),
        [1],
    ),
}


def test_sink_matrix_is_complete() -> None:
    """Both axes of the sweep are pinned to the real vocabularies.

    Without this the matrix would silently shrink as sinks are added — the
    failure mode of every hand-maintained table in this repo.
    """
    covered_domains = {domain for domain, _, _ in _PIPELINES.values()}
    assert covered_domains == {"buffer", "contour", "scalar", "vector"}, (
        f"pipeline domains not covered by the sink sweep: {covered_domains}"
    )
    # The format axis is generated from SinkFormat directly (see the
    # parametrize below), so this asserts the enum itself is non-trivial and
    # that nothing has quietly emptied it.
    assert len(set(SinkFormat)) >= 10, (
        "SinkFormat shrank; the sweep's format axis comes from it"
    )


@plugin_required
@pytest.mark.parametrize("sink", sorted(m.value for m in SinkFormat))
@pytest.mark.parametrize("case", sorted(_PIPELINES))
def test_planned_sink_dtype_is_what_execution_produces(rgb_df, case, sink) -> None:
    """A pair that survives planning must execute to the dtype it planned.

    Rejection at plan time is an acceptable outcome for any pair — that is the
    planner refusing a combination before any data moves, and it is what makes
    ``.sink("png")`` on a scalar a build error rather than a runtime surprise.
    What must never happen is planning succeeding and execution then failing or
    producing something else: that is the contract the whole lazy API rests on.
    """
    _domain, build, array_shape = _PIPELINES[case]
    pipe = build()

    kwargs = {}
    if sink == "array":
        if array_shape is None:
            pytest.skip(
                f"{case} has no fixed shape, so an array sink cannot be asked for"
            )
        kwargs["shape"] = array_shape

    try:
        expr = pl.col("img").cv.pipe(pipe).sink(sink, **kwargs)
    except (ValueError, TypeError):
        return  # Rejected while building the expression — before planning.

    lf = rgb_df.lazy().select(out=expr)
    try:
        planned = lf.collect_schema()["out"]
    except Exception:
        return  # Rejected at plan time.

    # Past this point the planner has published a dtype, and execution is
    # obliged to match it.
    produced = lf.collect()["out"].dtype
    assert produced == planned, (
        f"{case} + '{sink}' sink: planner promised {planned} but execution "
        f"produced {produced}"
    )


@plugin_required
@pytest.mark.parametrize(
    ("case", "sink", "kwargs"),
    [
        # The three pairs that were broken, pinned individually so a
        # regression names the case rather than only a matrix cell.
        ("vector-as-buffer", "native", {}),
        ("vector-as-vector", "array", {"shape": [3]}),
        ("vector-as-buffer", "array", {"shape": [1]}),
    ],
)
def test_vector_sinks_execute(rgb_df, case, sink, kwargs) -> None:
    """The vector-domain pairs that planned one thing and executed another.

    Each of these planned successfully and then failed at ``collect()``. They
    are separate from the matrix sweep because the matrix would also pass if
    these pairs started being *rejected* at plan time — which would be a
    regression in reach, not a fix.
    """
    _domain, build, _shape = _PIPELINES[case]
    expr = pl.col("img").cv.pipe(build()).sink(sink, **kwargs)
    lf = rgb_df.lazy().select(out=expr)

    planned = lf.collect_schema()["out"]
    out = lf.collect()["out"]
    assert out.dtype == planned
    assert out.null_count() == 0, "the row produced no value"


@plugin_required
def test_vector_representations_agree_at_the_sink(rgb_df) -> None:
    """Both vector representations encode the same way for the same sink.

    The planner declares one ``vector`` domain; that a hash rides as a buffer
    and ``extract_shape`` as a vector is an executor detail. If the two ever
    encode differently for the same sink, the domain has stopped being one
    thing and the planner's single ``expected_domain`` is a lie again.
    """
    for sink in ("native", "list"):
        dtypes = set()
        for case in ("vector-as-vector", "vector-as-buffer"):
            _domain, build, _shape = _PIPELINES[case]
            lf = rgb_df.lazy().select(out=pl.col("img").cv.pipe(build()).sink(sink))
            assert lf.collect()["out"].dtype == lf.collect_schema()["out"]
            dtypes.add(str(lf.collect_schema()["out"].base_type()))
        assert dtypes == {"List"}, (
            f"the two vector representations disagree on the '{sink}' sink "
            f"container: {dtypes}"
        )
