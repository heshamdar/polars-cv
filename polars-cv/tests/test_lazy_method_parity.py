"""``Pipeline`` and ``LazyPipelineExpr`` expose the same chainable methods, with the same signatures."""

from __future__ import annotations

import re

import pytest

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


def test_lazy_pipeline_method_parity():
    """Every chainable Pipeline method must exist on LazyPipelineExpr with an
    identical parameter list.

    The lazy forwarders are generated from Pipeline (``_lazy_forwarders.py``,
    by ``gen_ops.py``), so parity holds while that file is current; this guards
    the generation and any hand-written lazy method against drift."""
    import inspect

    from polars_cv.lazy import (
        PIPELINE_ONLY_METHODS,
        LazyPipelineExpr,
        _chainable_pipeline_ops,
    )
    from polars_cv.pipeline import Pipeline

    # The single source of truth for the eager/lazy asymmetry lives in lazy.py.
    pipeline_methods = {
        name
        for name in dir(Pipeline)
        if not name.startswith("_") and callable(getattr(Pipeline, name))
    }
    assert set(_chainable_pipeline_ops()) == pipeline_methods - PIPELINE_ONLY_METHODS

    missing = sorted(
        n for n in _chainable_pipeline_ops() if not hasattr(LazyPipelineExpr, n)
    )
    assert not missing, f"LazyPipelineExpr is missing Pipeline methods: {missing}"

    for name in _chainable_pipeline_ops():
        p_sig = inspect.signature(getattr(Pipeline, name))
        l_sig = inspect.signature(getattr(LazyPipelineExpr, name))
        # Compare parameter names, kinds and defaults (skip `self`); string
        # defaults compare equal to their str-enum counterparts.
        p_params = [
            (p.name, p.kind, p.default) for p in list(p_sig.parameters.values())[1:]
        ]
        l_params = [
            (p.name, p.kind, p.default) for p in list(l_sig.parameters.values())[1:]
        ]
        assert p_params == l_params, (
            f"Signature drift on '{name}':\n  Pipeline: {p_params}\n  Lazy:     {l_params}"
        )


def test_source_modifiers_are_not_generated_lazy_forwarders():
    """A Pipeline method that sets the source must be Pipeline-only.

    ``gen_ops.py`` generates a lazy forwarder for every chainable Pipeline op,
    running it on a *sourceless* continuation
    (``_continuation()`` returns a pipeline whose plan has no source). A
    source-modifier (``source`` today) would therefore unconditionally raise
    "requires a source" as a lazy method — a latent, always-failing forwarder.
    Such methods must live in ``PIPELINE_ONLY_METHODS`` so no forwarder is
    generated. This catches any future source-mutating builder that forgets the
    exclusion."""
    import inspect

    from polars_cv.lazy import PIPELINE_ONLY_METHODS
    from polars_cv.pipeline import Pipeline

    # A source is set only through `Plan.with_source`.
    assigns_source = re.compile(r"\.with_source\(")
    offenders = []
    for name in dir(Pipeline):
        if name.startswith("_"):
            continue
        method = getattr(Pipeline, name)
        if not callable(method):
            continue
        try:
            src = inspect.getsource(method)
        except (OSError, TypeError):
            continue
        if assigns_source.search(src) and name not in PIPELINE_ONLY_METHODS:
            offenders.append(name)
    assert not offenders, (
        "source-mutating Pipeline methods must be in PIPELINE_ONLY_METHODS "
        f"(else they generate always-failing lazy forwarders): {offenders}"
    )


def test_explicit_lazy_methods_take_a_lazy_operand():
    """A hand-written lazy method is only justified when its Pipeline
    counterpart takes a ``LazyPipelineExpr`` operand (a binary op, mask/merge).

    Ordinary ops must be generated, not hand-mirrored: the ``label_reduce``
    regression (a redundant explicit copy whose docstring drifted) is exactly
    what this guards against. For every method explicitly defined on
    ``LazyPipelineExpr`` that ALSO exists as a chainable Pipeline op, the
    Pipeline signature must reference ``LazyPipelineExpr`` in a parameter
    annotation (the bespoke-lazy exception); otherwise it should be deleted and
    left to the generator."""
    import inspect

    from polars_cv.lazy import LazyPipelineExpr, _chainable_pipeline_ops
    from polars_cv.pipeline import Pipeline

    chainable = set(_chainable_pipeline_ops())
    offenders = []
    for name, member in vars(LazyPipelineExpr).items():
        if name.startswith("_") or not callable(member):
            continue
        if name not in chainable:
            continue  # lazy-only method (merge_pipe, statistics_lazy, …) — fine
        p_sig = inspect.signature(getattr(Pipeline, name))
        takes_lazy_operand = any(
            "LazyPipelineExpr" in str(p.annotation) for p in p_sig.parameters.values()
        )
        if not takes_lazy_operand:
            offenders.append(name)
    assert not offenders, (
        "explicit LazyPipelineExpr methods without a LazyPipelineExpr operand "
        f"should be deleted and generated instead: {offenders}"
    )
