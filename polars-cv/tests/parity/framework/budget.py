"""How many examples a parity property runs, and in which lane.

Every property in ``tests/parity`` is declared once with :func:`property_lanes`
and runs twice:

* **fast** — collected under its own name, in the per-push lane. A small,
  *derandomized* budget: the same examples every run, so a green
  ``scripts/verify.sh`` is reproducible and CI cannot flake on a new draw.
* **deep** — the same body registered again as ``<name>_deep``, marked
  ``slow`` (the weekly lane), with a budget :data:`DEEP_FACTOR` times larger
  and a fresh random seed each run. This is the lane that *searches*; a
  failure it finds is saved to the example database and replayed first
  next time, and belongs in the fast lane as an ``@example`` once understood.

A test weight scales both budgets by what one example costs (a chain draws
and executes several pipelines per example, a single-op check one).

Two environment knobs, for local exploration only:

* ``POLARS_CV_PARITY_SCALE`` — multiply every budget (e.g. ``20`` for a long
  local hunt).
* ``POLARS_CV_PARITY_RANDOM=1`` — let the fast lane draw fresh examples too.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import pytest
from hypothesis import HealthCheck, given, settings

#: Examples per fast-lane property at weight 1.
FAST_EXAMPLES = 20

#: How many times more examples the deep (slow-lane) twin runs.
DEEP_FACTOR = 25

_SUPPRESSED = (HealthCheck.too_slow, HealthCheck.data_too_large)


def _scale() -> float:
    raw = os.environ.get("POLARS_CV_PARITY_SCALE", "1")
    try:
        value = float(raw)
    except ValueError:
        msg = f"POLARS_CV_PARITY_SCALE must be a number, got {raw!r}"
        raise ValueError(msg) from None
    if value <= 0:
        msg = f"POLARS_CV_PARITY_SCALE must be positive, got {raw!r}"
        raise ValueError(msg)
    return value


def examples(weight: float, *, deep: bool) -> int:
    """The example budget for a property of *weight* in one lane."""
    base = FAST_EXAMPLES * weight * _scale()
    if deep:
        base *= DEEP_FACTOR
    return max(1, round(base))


def lane_settings(weight: float, *, deep: bool) -> settings:
    """Hypothesis settings for one lane of a property of *weight*.

    ``deadline`` is off: the first call of a pipeline shape compiles its graph
    and the debug extension is slow, neither of which is what a deadline is
    for. ``print_blob`` makes every failure reproducible with
    ``@reproduce_failure``.
    """
    derandomize = not deep and os.environ.get("POLARS_CV_PARITY_RANDOM") != "1"
    return settings(
        max_examples=examples(weight, deep=deep),
        deadline=None,
        derandomize=derandomize,
        print_blob=True,
        suppress_health_check=_SUPPRESSED,
    )


def property_lanes(
    weight: float = 1.0,
    *,
    marks: tuple[pytest.MarkDecorator, ...] = (),
    **strategies: Any,
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Declare a property once; register its fast and deep lanes.

    Use as a decorator on a module-level test function, passing the
    strategies ``given`` would take::

        @property_lanes(weight=2, case=single_op_cases())
        def test_op_matches_reference(case): ...

    The decorated name is the fast test. A second test, ``<name>_deep``, is
    written into the defining module's namespace (where pytest collects it)
    and marked ``slow``. Both wrap the *same* body, so the two lanes can never
    drift apart — each is ``given`` applied separately, which is what gives
    them independent settings. *marks* (a ``parametrize``, say) apply to both.
    """

    def decorate(body: Callable[..., None]) -> Callable[..., None]:
        fast = lane_settings(weight, deep=False)(given(**strategies)(body))
        deep = lane_settings(weight, deep=True)(given(**strategies)(body))
        for mark in marks:
            fast = mark(fast)
            deep = mark(deep)
        deep = pytest.mark.slow(deep)
        deep.__name__ = f"{body.__name__}_deep"
        deep.__qualname__ = deep.__name__
        module_globals = body.__globals__
        if deep.__name__ in module_globals:
            msg = f"{deep.__name__} is already defined in {body.__module__}"
            raise RuntimeError(msg)
        module_globals[deep.__name__] = deep
        return fast

    return decorate
