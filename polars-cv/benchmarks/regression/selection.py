"""Which cases a suite run executes: ``scenario[:glob]`` selectors.

A selector names a scenario (every case in it) or a scenario and an
``fnmatch`` glob over its case names — the ``operation`` name each result
carries, so what ``compare`` prints is what ``--select`` takes::

    pipelines                       every pipeline
    single_ops:rotate_*             rotate_90, rotate_45
    targeted:geom_*,e2e             the geometry cases, and every e2e workflow

Each scenario's case list is read from the scenario itself (its
``get_*_benchmarks()`` or ``CASES``), never restated here. ``zero_copy`` and
``remote`` run as a unit with their own fixed matrix, so they take no glob.

Nothing here degrades silently: an unknown scenario, a glob matching no case,
or a glob on a scenario that runs as a whole is a ``ValueError`` — each of
those would otherwise run less than asked and compare as "no regressions".
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmarks.regression.config import ALL_SCENARIOS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

SCENARIOS: tuple[str, ...] = ALL_SCENARIOS


def _single_ops() -> list[str]:
    from benchmarks.scenarios.single_ops import get_single_op_benchmarks

    return [b.name for b in get_single_op_benchmarks()]


def _pipelines() -> list[str]:
    from benchmarks.scenarios.pipelines import get_pipeline_benchmarks

    return [b.name for b in get_pipeline_benchmarks()]


def _e2e() -> list[str]:
    from benchmarks.scenarios.e2e_workflow import get_e2e_workflows

    # Results are named `e2e_<workflow>` (see run_e2e_workflow_polars).
    return [f"e2e_{w.name}" for w in get_e2e_workflows()]


def _targeted() -> list[str]:
    from benchmarks.scenarios.targeted import case_names

    return case_names()


# None: the scenario runs as a whole (its own fixed matrix, no case filter).
_CASE_LISTS: Mapping[str, Callable[[], list[str]] | None] = {
    "single_ops": _single_ops,
    "pipelines": _pipelines,
    "e2e": _e2e,
    "targeted": _targeted,
    "zero_copy": None,
    "remote": None,
}
if set(_CASE_LISTS) != set(SCENARIOS):
    msg = f"case lists {sorted(_CASE_LISTS)} != scenarios {sorted(SCENARIOS)}"
    raise RuntimeError(msg)


def case_names(scenario: str) -> list[str] | None:
    """The scenario's case names, or None if it runs as a whole."""
    lister = _CASE_LISTS[scenario]
    return None if lister is None else lister()


@dataclass(frozen=True)
class Selection:
    """Scenario -> selected case names (``None`` = every case), in suite order."""

    _entries: tuple[tuple[str, frozenset[str] | None], ...] = ()

    def scenarios(self) -> tuple[str, ...]:
        return tuple(s for s, _ in self._entries)

    def cases(self, scenario: str) -> frozenset[str] | None:
        """The selected cases of ``scenario`` (None = all of them)."""
        for s, cases in self._entries:
            if s == scenario:
                return cases
        msg = f"scenario {scenario!r} is not selected"
        raise KeyError(msg)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __or__(self, other: Selection) -> Selection:
        merged: dict[str, frozenset[str] | None] = dict(self._entries)
        for s, cases in other._entries:
            if s in merged:
                prev = merged[s]
                merged[s] = None if prev is None or cases is None else prev | cases
            else:
                merged[s] = cases
        return _ordered(merged)

    def render(self) -> str:
        """The selector string that parses back to this selection."""
        parts: list[str] = []
        for s, cases in self._entries:
            parts += [s] if cases is None else [f"{s}:{c}" for c in sorted(cases)]
        return ",".join(parts)


def _ordered(entries: Mapping[str, frozenset[str] | None]) -> Selection:
    return Selection(tuple((s, entries[s]) for s in SCENARIOS if s in entries))


def _one(selector: str) -> Selection:
    scenario, _, glob = selector.partition(":")
    if scenario not in _CASE_LISTS:
        msg = f"unknown scenario {scenario!r} in {selector!r}; valid: {SCENARIOS}"
        raise ValueError(msg)
    if not glob:
        return Selection(((scenario, None),))
    names = case_names(scenario)
    if names is None:
        msg = (
            f"{selector!r}: scenario {scenario!r} runs as a whole (its own "
            f"fixed matrix); select it without a glob"
        )
        raise ValueError(msg)
    matched = frozenset(fnmatch.filter(names, glob))
    if not matched:
        msg = f"{selector!r} matches no case; {scenario} has: {sorted(names)}"
        raise ValueError(msg)
    return Selection(((scenario, matched),))


def parse(spec: str) -> Selection:
    """Parse comma-separated selectors into one ``Selection``."""
    sel = Selection()
    for selector in (s.strip() for s in spec.split(",")):
        if selector:
            sel = sel | _one(selector)
    return sel
