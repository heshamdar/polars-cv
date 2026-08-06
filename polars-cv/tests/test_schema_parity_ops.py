"""Plan == exec for every chainable operation, across every sink.

The op axis is not a list written here. It is ``tests/_op_cases.py``'s
``OP_CASES``, which ``test_append_contract.test_op_case_table_is_complete``
pins to ``polars_cv.lazy._chainable_pipeline_ops()``. So an operation cannot
join the library without also getting a plan-vs-exec cell: the completeness
assertion fails first.

Before this file, roughly forty of the seventy-one chainable ops had no
plan-vs-exec test at any point — every reduction (including the ``ForceI64``
argmax/argmin and the ``ReduceByOne`` axis forms), all four morphology ops,
every colour conversion, ``canny``, ``equalize_histogram``, ``letterbox``,
``pad_to_size``, the five ``resize_*`` variants, and all five histogram output
modes. They were covered at the *hint* level (does the planner compute the
right shape?) but never against the data the planner was making promises about.

Each test runs one op against every ``SinkFormat``. Rejection at build or plan
time is an acceptable outcome for any cell — that is the planner refusing a
combination before data moves. What must never happen is planning succeeding
and execution then producing something else, which is what
:func:`plan_or_reject` asserts.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import SinkFormat
from tests._op_cases import (
    BUFFER,
    COLOR_SPACES,
    CONTOUR,
    EXTRA_CASES,
    HISTOGRAM_OUTPUTS,
    IMAGE_ENCODER_SINKS,
    OP_CASES,
    SINGLE_CHANNEL_OPS,
    comparable_ops,
)
from tests._schema_parity import (
    ParityResult,
    assert_not_vacuous,
    encodable_by_image_codec,
    frame,
    plan_or_reject,
    rows_for,
)
from tests.conftest import make_image_png, make_rect_png, plugin_required

#: ``OP_CASES``' arguments are written against a 100x200x3 image (``crop``
#: takes 50x50, ``reshape`` takes exactly 60000 elements, ``resize_max`` 120).
#: Non-square so an H/W swap cannot cancel itself out.
HEIGHT, WIDTH, CHANNELS = 100, 200, 3

SINKS: tuple[str, ...] = tuple(sorted(m.value for m in SinkFormat))

#: Row layouts this sweep runs. The full set — including the 64-row
#: morsel-crossing and heterogeneous patterns — lives in
#: ``test_schema_parity_nulls.py``; running all nine here would multiply a
#: 71x10 matrix by nine for little extra signal, since what varies per op is
#: the sink, not the null layout.
PATTERNS: tuple[str, ...] = ("single", "all_null", "null_first")


def _base(domain: str, *, single_channel: bool = False) -> Pipeline:
    """A pipeline in *domain* with fully known shape hints and a concrete dtype.

    ``cast("u8")`` matters: an ``image_bytes`` source carries dtype ``"auto"``,
    and the typed ``list``/``array`` sinks refuse to guess an element dtype, so
    without it every typed cell would reject and the sweep would say nothing
    about them.

    *single_channel* prepends ``grayscale()`` for the ops in
    ``SINGLE_CHANNEL_OPS``, whose kernels reject a three-channel buffer at
    execution time.
    """
    pipe = (
        Pipeline()
        .source("image_bytes")
        .assert_shape(height=HEIGHT, width=WIDTH, channels=CHANNELS)
        .cast("u8")
    )
    if domain == CONTOUR:
        return pipe.grayscale().threshold(128).extract_contours()
    if single_channel:
        return pipe.grayscale()
    return pipe


def _images(domain: str) -> list[bytes]:
    """Three distinct sample images for the row patterns to arrange."""
    factory = make_rect_png if domain == CONTOUR else make_image_png
    if domain == CONTOUR:
        return [make_rect_png(HEIGHT, WIDTH, CHANNELS) for _ in range(3)]
    return [
        factory(HEIGHT, WIDTH, CHANNELS, seed=seed)  # type: ignore[call-arg]
        for seed in (1, 2, 3)
    ]


def _sweep_sinks(pipe: Pipeline, domain: str, label: str) -> None:
    """Run *pipe* into every sink, under every row pattern, and check parity."""
    images = _images(domain)
    results: dict[tuple[str, str], ParityResult] = {}

    # The image-encoder sinks carry two preconditions the planner knows and
    # does not check — the buffer's dtype (jpeg/webp are 8-bit) and its rank (a
    # 1-D buffer is not an image) — so a pipeline that violates either plans as
    # Binary and then dies inside the encoder. That is one finding with one
    # cause; it is pinned once, in test_schema_parity_encoder_sinks.py, rather
    # than smeared across every op that promotes to f32 or drops a dimension.
    # u8 at rank 2 or 3 satisfies all four encoders.
    encodable = encodable_by_image_codec(pipe)
    sinks = [s for s in SINKS if encodable or s not in IMAGE_ENCODER_SINKS]

    for pattern in PATTERNS:
        df = frame(rows_for(pattern, images), pl.Binary)
        for sink in sinks:
            results[(pattern, sink)] = plan_or_reject(
                df, lambda s=sink: pl.col("img").cv.pipe(pipe).sink(s)
            )

    assert_not_vacuous(results, f"{label}: every sink x row-pattern cell")


@plugin_required
@pytest.mark.parametrize("op", comparable_ops())
def test_every_op_plans_what_it_executes(op: str) -> None:
    """One op, every sink, every null layout: the planned dtype is produced.

    The op itself is built outside the sweep on purpose. ``OP_CASES``' argument
    sets are known-good (the eager/lazy parity sweep calls them), so a builder
    raising here is a real failure — only the *sink* is allowed to reject.
    """
    domain, kwargs = OP_CASES[op]
    base = _base(domain, single_channel=op in SINGLE_CHANNEL_OPS)
    pipe = getattr(base, op)(**kwargs)
    _sweep_sinks(pipe, domain, op)


@plugin_required
@pytest.mark.parametrize(
    ("op", "domain", "kwargs"),
    EXTRA_CASES,
    ids=[f"{o}-{'-'.join(sorted(k))}" for o, _, k in EXTRA_CASES],
)
def test_op_branches_plan_what_they_execute(op: str, domain: str, kwargs: dict) -> None:
    """The branches a single case misses: rotate's fast path, axis reductions,
    rank-changing reshapes, offset-only crop, struct-encoded histogram."""
    base = _base(domain, single_channel=op in SINGLE_CHANNEL_OPS)
    pipe = getattr(base, op)(**kwargs)
    _sweep_sinks(pipe, domain, f"{op}-{'-'.join(sorted(kwargs))}")


@plugin_required
@pytest.mark.parametrize("output", HISTOGRAM_OUTPUTS)
def test_histogram_output_modes_plan_what_they_execute(output: str) -> None:
    """Each histogram mode lands in a different (rank, dtype, domain) corner.

    ``counts`` is ForceU64 rank 1, ``normalized``/``edges`` ForceF64 rank 1,
    ``buckets`` a rank-2 struct-encoded vector, ``quantized`` a ForceU32 buffer
    that stays in the buffer domain. None of the five had a plan-vs-exec test.
    """
    pipe = _base(BUFFER).grayscale().histogram(bins=8, output=output)
    _sweep_sinks(pipe, BUFFER, f"histogram-{output}")


@plugin_required
@pytest.mark.parametrize("to_space", COLOR_SPACES)
def test_color_conversions_plan_what_they_execute(to_space: str) -> None:
    """``convert_color`` is the ``StripProcessRestore`` channel rule's only user.

    Its output channel count comes from the target space, and Lab additionally
    forces the dtype to f32 — two schema effects from one parameter, neither
    previously compared against data.
    """
    pipe = _base(BUFFER).convert_color(from_space="rgb", to_space=to_space)
    _sweep_sinks(pipe, BUFFER, f"convert_color-{to_space}")


@plugin_required
def test_the_op_axis_is_the_shared_table() -> None:
    """This sweep must read the completeness-guarded table, not a local list.

    ``test_op_case_table_is_complete`` pins ``OP_CASES`` to the real chainable
    op list. That guarantee only reaches this file if this file is actually
    driven by the same object — a local copy would silently stop growing.
    """
    from tests import _op_cases

    assert OP_CASES is _op_cases.OP_CASES
    swept = set(comparable_ops())
    exempt = {name for name, case in OP_CASES.items() if case is None}
    assert swept | exempt == set(OP_CASES)
    assert len(swept) > 60, (
        f"only {len(swept)} ops in the sweep — the table shrank or the filter "
        f"stopped matching"
    )
