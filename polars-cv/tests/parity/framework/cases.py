"""Hypothesis strategies for whole cases: rows, steps, chains and axes.

The reference table (:mod:`oracle`) says what one call looks like on one
input. This module assembles cases from it:

* :func:`batches` — the rows of a frame: one image or several, the same shape
  or different shapes (same dtype and channels), with null rows mixed in.
* :func:`steps_for` — one op's arguments for a batch, optionally varying per
  row (a :class:`PerRow` value, only for expression-eligible parameters).
* :func:`draw_step` — the next op of a chain, drawn from the ops whose
  contract admits the current input; used interactively with ``st.data()``
  so each step's arguments can depend on the engine's actual output so far.
* :func:`axes_variants` — the executions an invariance check compares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from hypothesis import event
from hypothesis import strategies as st

from tests.parity.framework.images import DTYPES, ImageSpec, image_specs, render
from tests.parity.framework.io import SINKS, SOURCES, sinks_for, sources_for
from tests.parity.framework.oracle import OPS, OpSpec
from tests.parity.framework.run import (
    COMPOSITIONS,
    ENGINES,
    OPTIMIZATION,
    PARAM_STYLES,
    Axes,
    PerRow,
    Step,
    expression_eligible,
)


@dataclass(frozen=True)
class Batch:
    """The image column of a case: rendered rows and how they were made."""

    specs: tuple[ImageSpec | None, ...]
    images: tuple[np.ndarray | None, ...]

    @property
    def present(self) -> list[np.ndarray]:
        return [im for im in self.images if im is not None]

    @property
    def homogeneous(self) -> bool:
        return len({im.shape for im in self.present}) == 1

    def proxy(self) -> np.ndarray:
        """An image as small as every row in each spatial axis.

        Arguments drawn for it are valid for every row (a crop inside it is
        inside each row), so one literal set can serve a heterogeneous batch.
        """
        rows = self.present
        h = min(im.shape[0] for im in rows)
        w = min(im.shape[1] for im in rows)
        return rows[int(np.argmin([im.shape[0] * im.shape[1] for im in rows]))][:h, :w]

    def __repr__(self) -> str:
        return f"Batch({list(self.specs)!r})"


def batches(
    *,
    dtypes: Sequence[str] = tuple(DTYPES),
    channels: Sequence[int] = (1, 2, 3, 4),
    max_rows: int = 4,
    nulls: bool = True,
    heterogeneous: bool = True,
    max_side: int = 32,
) -> st.SearchStrategy[Batch]:
    """Draw a :class:`Batch`: 1..*max_rows* rows, optionally with nulls."""

    @st.composite
    def build(draw: st.DrawFn) -> Batch:
        first = draw(
            image_specs(
                dtypes=tuple(dtypes), channels=tuple(channels), max_side=max_side
            ),
            label="first image",
        )
        n = draw(st.integers(1, max_rows), label="rows")
        vary = heterogeneous and n > 1 and draw(st.booleans(), label="heterogeneous")
        specs: list[ImageSpec | None] = [first]
        for i in range(1, n):
            seed = draw(st.integers(0, 2**32 - 1), label=f"seed {i}")
            spec = ImageSpec(**{**first.__dict__, "seed": seed})
            if vary:
                spec = ImageSpec(
                    **{
                        **spec.__dict__,
                        "height": draw(st.integers(1, max_side), label=f"height {i}"),
                        "width": draw(st.integers(1, max_side), label=f"width {i}"),
                    }
                )
            specs.append(spec)
        if nulls and n > 1:
            null_rows = draw(
                st.sets(st.integers(0, n - 1), max_size=n - 1), label="null rows"
            )
            specs = [None if i in null_rows else s for i, s in enumerate(specs)]
        images = tuple(None if s is None else render(s) for s in specs)
        return Batch(specs=tuple(specs), images=images)

    return build()


def batches_for(
    spec: OpSpec, *, reference: bool, **kwargs: Any
) -> st.SearchStrategy[Batch]:
    """Batches of inputs *spec*'s contract admits.

    With *reference*, the dtypes are narrowed to those the reference models
    (:attr:`OpSpec.ref_dtypes`) so examples are not spent on skipped rows.
    """
    dtypes = list(spec.ref_dtypes or DTYPES) if reference else list(DTYPES)
    admitted = [
        (d, c)
        for d in dtypes
        for c in (1, 2, 3, 4)
        if spec.accepts(np.zeros((3, 3, c), dtype=DTYPES[d]))
    ]
    if not admitted:
        msg = f"{spec.method}: no dtype/channel combination is admitted"
        raise ValueError(msg)
    if spec.uniform_rows:
        kwargs["heterogeneous"] = False
    # The dtype/channel probe above is 3x3; a contract that also bounds the
    # size (perceptual_hash's 2x2 minimum, contour extraction's maximum) is
    # checked on the rows actually drawn.
    return (
        st.sampled_from(admitted)
        .flatmap(lambda dc: batches(dtypes=(dc[0],), channels=(dc[1],), **kwargs))
        .filter(lambda b: all(spec.accepts(im) for im in b.present))
    )


def draw_params(draw: st.DrawFn, spec: OpSpec, x: np.ndarray) -> dict[str, Any]:
    """Arguments for *spec* that are valid on input *x*."""
    return spec.params(draw, x)


def _merge_per_row(method: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Fold per-row argument dicts into one step, or ``None`` if impossible.

    A parameter that differs between rows becomes a :class:`PerRow` — which
    only an expression-eligible parameter can be. If an ineligible one
    differs, the rows cannot share a step.
    """
    keys = rows[0].keys()
    if any(r.keys() != keys for r in rows):
        return None
    merged: dict[str, Any] = {}
    for key in keys:
        values = [r[key] for r in rows]
        lengths = {len(v) for v in values if isinstance(v, (list, tuple))}
        if len(lengths) > 1:
            return None  # a list's length is structural: it cannot vary
        if all(v == values[0] for v in values):
            merged[key] = values[0]
        elif expression_eligible(method, key) is not None:
            merged[key] = PerRow(tuple(values))
        else:
            return None
    return merged


def steps_for(spec: OpSpec, batch: Batch, *, per_row: bool) -> st.SearchStrategy[Step]:
    """One step of *spec* for *batch*.

    With *per_row* (and a homogeneous batch), each row draws its own
    arguments and those that differ ride as :class:`PerRow` expressions;
    otherwise one literal set, drawn on the batch's :meth:`Batch.proxy`,
    serves every row.
    """

    @st.composite
    def build(draw: st.DrawFn) -> Step:
        rows = batch.present
        if per_row and batch.homogeneous and len(batch.images) > 1:
            dicts = [
                draw(
                    st.composite(lambda d, im=im: draw_params(d, spec, im))(),
                    label="row args",
                )
                for im in batch.images
                if im is not None
            ]
            # Null rows take the first row's arguments; they are never read.
            expanded = []
            it = iter(dicts)
            for im in batch.images:
                expanded.append(next(it) if im is not None else dicts[0])
            merged = _merge_per_row(spec.method, expanded)
            if merged is not None:
                return Step(spec.method, merged)
        proxy = batch.proxy() if len(rows) > 1 else rows[0]
        return Step(
            spec.method,
            draw(st.composite(lambda d: draw_params(d, spec, proxy))(), label="args"),
        )

    return build()


def candidate_specs(
    x: Any,
    *,
    domain: str = "buffer",
    allow_terminal: bool,
    require_ref: bool,
    exclude: Sequence[str] = (),
) -> list[OpSpec]:
    """The ops whose contract admits input *x* (in *domain*)."""
    out = []
    for spec in OPS.values():
        if spec.method in exclude or spec.domain_in != domain:
            continue
        if spec.terminal and not allow_terminal:
            continue
        if require_ref and spec.ref is None:
            continue
        if domain == "buffer" and not spec.accepts(x):
            continue
        out.append(spec)
    return out


def draw_step(
    data: st.DataObject,
    x: Any,
    *,
    domain: str = "buffer",
    allow_terminal: bool = False,
    require_ref: bool = True,
    exclude: Sequence[str] = (),
    label: str = "step",
) -> Step:
    """Draw the next op of a chain and its arguments for input *x*."""
    candidates = candidate_specs(
        x,
        domain=domain,
        allow_terminal=allow_terminal,
        require_ref=require_ref,
        exclude=exclude,
    )
    spec = data.draw(st.sampled_from(candidates), label=f"{label} op")
    params = data.draw(
        st.composite(lambda d: draw_params(d, spec, x))(), label=f"{label} args"
    )
    return Step(spec.method, params)


def draw_batch_chain(
    data: st.DataObject,
    batch: Batch,
    *,
    max_steps: int,
    require_ref: bool = False,
) -> tuple[list[Step], Any]:
    """Grow a chain for a whole batch, executing each prefix to draw the next.

    Arguments are drawn on the smallest current output in each axis (as
    :meth:`Batch.proxy` does for inputs), so one literal set is valid for
    every row. Ops that restate a row's whole shape are only drawn while
    every row has the same shape. The chain may leave the buffer domain
    (a reduction, a hash, contour extraction followed by contour ops).

    Returns:
        The steps, and the baseline execution of the full chain.
    """
    from tests.parity.framework import known
    from tests.parity.framework.run import PlanRefused, execute

    axes = baseline_axes(batch.images)
    steps: list[Step] = []
    result = execute(batch.images, steps, axes)
    domain = "buffer"
    length = data.draw(st.integers(1, max_steps), label="length")
    for index in range(length):
        outputs = [r for r in result.rows if r is not None]
        if domain == "buffer":
            uniform = len({o.shape for o in outputs}) == 1
            proxy = _smallest(outputs)
            exclude = (
                [] if uniform else [s.method for s in OPS.values() if s.uniform_rows]
            )
        else:
            proxy, exclude = None, []
        step = draw_step(
            data,
            proxy,
            domain=domain,
            allow_terminal=True,
            require_ref=require_ref,
            exclude=exclude,
            label=f"step {index}",
        )
        if domain == "buffer":
            divergence = known.append_divergence(steps, step, outputs)
            if divergence is not None:
                event(f"known divergence: {divergence.key}")
                continue
        spec = OPS[step.method]
        sink = "numpy" if spec.domain_out == "buffer" else "native"
        try:
            result = execute(batch.images, [*steps, step], axes.but(sink=sink))
        except PlanRefused:
            # A plan-time fact this generator does not model (a rank the
            # planner cannot know after a blob round trip, a channel limit)
            # ruled the op out. The single-op suites keep the contract
            # prefilters honest; here the step is simply not taken.
            event(f"planner refused {step.method} mid-chain")
            continue
        steps.append(step)
        domain = spec.domain_out
        if domain in ("scalar", "vector"):
            break
    return steps, result


def _smallest(outputs: Sequence[np.ndarray]) -> np.ndarray:
    """An output no larger than any other in each spatial axis."""
    if len(outputs) == 1 or outputs[0].ndim < 2:
        return outputs[0]
    h = min(o.shape[0] for o in outputs)
    w = min(o.shape[1] for o in outputs)
    return outputs[0][:h, :w]


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


def baseline_axes(images: Sequence[Any], domain: str = "buffer") -> Axes:
    """The reference execution for *images*: a source carrying them exactly.

    ``array`` when every row has one shape (the most direct route), ``list``
    otherwise (the only lossless source for mixed shapes and any dtype); the
    ``numpy`` sink for a buffer, ``native`` for anything else.
    """
    present = [im for im in images if im is not None]
    homogeneous = len({im.shape for im in present}) == 1
    return Axes(
        source="array" if homogeneous else "list",
        sink="numpy" if domain == "buffer" else "native",
    )


def lossless_sources(images: Sequence[Any]) -> list[str]:
    """Every lossless source that can carry *images*, bar any a known
    divergence rules out (it would raise, not disagree)."""
    from tests.parity.framework import known

    present = [im for im in images if im is not None]
    usable = []
    for name in sources_for(present):
        divergence = known.axes_divergence(Axes(source=name), images, [])
        if SOURCES[name].lossless and not (divergence is not None and divergence.avoid):
            usable.append(name)
    return usable


def axes_variants(
    images: Sequence[Any],
    steps: Sequence[Step],
    *,
    output_domain: str,
    output_dtype: str,
    output_shapes: Sequence[tuple[int, ...]],
) -> st.SearchStrategy[list[Axes]]:
    """One variant per axis value that applies to this case.

    Every axis is swept one at a time from the baseline (so a failure names
    the axis that caused it), then one fully random combination is added so
    interactions between axes are sampled too.
    """
    from tests.parity.framework.io import OutputInfo

    base = baseline_axes(images, output_domain)
    info = OutputInfo(
        domain=output_domain, dtype=output_dtype, shapes=tuple(output_shapes)
    )
    has_expr = any(
        expression_eligible(s.method, k) is not None for s in steps for k in s.params
    )
    varies = any(s.varies for s in steps)
    sources = lossless_sources(images)
    sinks = [s for s in sinks_for(info) if SINKS[s].exact]
    styles = [p for p in PARAM_STYLES if not (varies and p in ("literal", "lit"))]
    compositions = _compositions(steps)
    if varies:
        base = base.but(params="column")

    @st.composite
    def build(draw: st.DrawFn) -> list[Axes]:
        variants = [base.but(source=s) for s in sources if s != base.source]
        variants += [base.but(sink=s) for s in sinks if s != base.sink]
        variants += [base.but(engine=e) for e in ENGINES if e != base.engine]
        if has_expr:
            variants += [base.but(params=p) for p in styles if p != base.params]
        variants += [
            base.but(composition=c) for c in compositions if c != base.composition
        ]
        # Every pass off, then a few single-pass settings per example (each
        # pass alone, each pass removed): across a lane's examples every pass
        # is reached both ways without paying for all of them every time.
        variants.append(base.but(optimize="none"))
        per_pass = [v for v in OPTIMIZATION if ":" in v]
        chosen = draw(
            st.lists(st.sampled_from(per_pass), min_size=1, max_size=4, unique=True),
            label="optimizer settings",
        )
        variants += [base.but(optimize=v) for v in chosen]
        if len(images) > 1:
            variants.append(base.but(chunked=True))
        combo = Axes(
            source=draw(st.sampled_from(sources), label="source"),
            sink=draw(st.sampled_from(sinks), label="sink"),
            engine=draw(st.sampled_from(ENGINES), label="engine"),
            params=draw(st.sampled_from(styles), label="params")
            if has_expr
            else base.params,
            composition=draw(st.sampled_from(compositions), label="composition"),
            optimize=draw(st.sampled_from(OPTIMIZATION), label="optimize"),
            chunked=len(images) > 1 and draw(st.booleans(), label="chunked"),
        )
        if combo.composition == "aliased" and SINKS[combo.sink].kwargs(info):
            # A multi-output sink shares its keyword arguments across every
            # output, so a sink that needs its own (array's shape=) cannot be
            # one output among several blobs.
            combo = combo.but(composition="chain")
        variants.append(combo)
        return variants

    return build()


def _compositions(steps: Sequence[Step]) -> list[str]:
    """The compositions a chain can be written in.

    ``materialized`` and ``aliased`` round-trip every intermediate through a
    ``blob`` sink, which only a buffer can take, so a chain whose
    non-final step leaves the buffer domain can only be written as one
    pipeline (or split with ``.pipe``).
    """
    out = ["chain", "continuation"]
    intermediate_domains = {OPS[s.method].domain_out for s in steps[:-1]}
    if intermediate_domains <= {"buffer"} and steps:
        out += ["aliased", "materialized"]
    return [c for c in COMPOSITIONS if c in out]
