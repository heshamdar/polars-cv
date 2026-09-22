"""The timing-authority ratchet's logic, separated so it can be tested itself.

``test_timing_authority`` runs this over ``benchmarks/``. ``test_timing_
authority_fixtures`` runs it over committed known-bad and known-good snippets.
Both import from here, so the fixtures exercise the code that actually guards
the tree — a fixture testing a copy would prove nothing about the guard.

What it guards and why
----------------------

``benchmarks/utils/timing.py`` is the single authority for *how long something
took* and *what number gets reported for it*. Before it existed the suite had
six structurally-identical timing loops (``single_ops``, ``pipelines``,
``e2e_workflow``, two GPU variants, ``remote_source._timed``), fourteen further
un-repeated spans in ``zero_copy_ingestion``, thirty in
``inference_pipeline_comparison``, and two more methodologies in
``plugin_overhead`` / ``batch_throughput`` — which reported *min and median*
while every scenario reported the *mean*. One suite, two answers to "what is
the number", and nothing that could notice.

Two rules, not one
------------------

Banning the clock is not enough. ``perf_counter`` bans a hand-rolled *timer*;
it does nothing about a hand-rolled *statistic*, and the four scenario loops
each divided by ``benchmark_iterations`` independently — four chances to get
the reduction wrong, in four places. So ``BenchmarkResult`` may only be
constructed by the authority (:data:`R5`). You cannot produce a result record
without having gone through the code that computed the number in it. That is
the repo's "make the sequence unskippable" rule rather than a checklist of
things a new scenario must remember.

Detection is on the resolved *binding*, not the text, so ``import time as _t``
followed by ``_t.perf_counter()`` is caught. Text-matching this would also
flag every docstring and help string that mentions ``perf_counter`` — the
false positive that has to stay fixtured, because it is the one that makes a
maintainer add a blanket exemption.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

# The one module allowed to read a clock and to build a result record. Relative
# to the `polars-cv/` package directory, POSIX separators.
AUTHORITY = "benchmarks/utils/timing.py"

# `time.sleep` is not a measurement. `remote_source.py`'s loopback server uses
# it to inject synthetic latency (`--latency-ms`), which is the thing being
# measured rather than the measuring.
ALLOWED_TIME_ATTRS = frozenset({"sleep"})

# Any of these, reached through `getattr(x, "...")`, is a clock by another name.
_CLOCK_ATTR_PREFIXES = ("perf_", "monotonic", "process_time", "thread_time")

# The result record. Constructing one outside the authority means computing the
# statistic outside the authority.
RESULT_RECORD = "BenchmarkResult"


@dataclass(frozen=True)
class Offender:
    """One rule violation, located well enough to fix without searching."""

    path: str
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


class _Visitor(ast.NodeVisitor):
    """Resolve module bindings, then report uses of them.

    Two passes are not needed: Python requires the import to precede the use
    textually within a module for the name to be bound at runtime, and an
    import inside a function still appears before its own uses.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.offenders: list[Offender] = []
        # Local names currently bound to the `time` module.
        self._time_aliases: set[str] = set()
        # Local names currently bound to `BenchmarkResult`.
        self._result_aliases: set[str] = {RESULT_RECORD}

    def _add(self, node: ast.AST, rule: str, detail: str) -> None:
        self.offenders.append(
            Offender(self.path, getattr(node, "lineno", 0), rule, detail)
        )

    # -- binding ---------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "time":
                self._time_aliases.add(alias.asname or "time")
            elif alias.name == "timeit":
                self._add(
                    node,
                    "R3",
                    f"`import timeit{' as ' + alias.asname if alias.asname else ''}`"
                    f" — use benchmarks.utils.timing instead",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "time":
            for alias in node.names:
                if alias.name not in ALLOWED_TIME_ATTRS:
                    self._add(
                        node,
                        "R2",
                        f"`from time import {alias.name}` — "
                        f"use benchmarks.utils.timing instead",
                    )
        elif node.module == "timeit":
            self._add(node, "R3", "`from timeit import ...` — use timing.measure")
        else:
            # `from benchmarks.frameworks import BenchmarkResult as _R` still
            # binds the record's constructor; track the alias so R5 sees it.
            for alias in node.names:
                if alias.name == RESULT_RECORD:
                    self._result_aliases.add(alias.asname or alias.name)
        self.generic_visit(node)

    # -- use -------------------------------------------------------------

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self._time_aliases
            and node.attr not in ALLOWED_TIME_ATTRS
        ):
            self._add(
                node,
                "R1",
                f"`{node.value.id}.{node.attr}` — the clock lives in "
                f"benchmarks.utils.timing",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self._check_getattr_clock(node)
        self._check_result_record(node)
        self.generic_visit(node)

    def _check_getattr_clock(self, node: ast.Call) -> None:
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            return
        if len(node.args) < 2:
            return
        name = node.args[1]
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            if name.value.startswith(_CLOCK_ATTR_PREFIXES):
                self._add(
                    node,
                    "R4",
                    f'`getattr(..., "{name.value}")` — reaching a clock by '
                    f"string does not make it a different clock",
                )

    def _check_result_record(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            return
        if name in self._result_aliases and name != "":
            self._add(
                node,
                "R5",
                f"`{name}(...)` — only benchmarks.utils.timing.to_result may "
                f"build a result record, so the statistic has one definition",
            )


def timing_offenders(source: str, *, path: str) -> list[Offender]:
    """Report every timing-authority violation in one module's source.

    ``path`` is used for reporting and for the authority exemption, so it must
    be the module's path relative to ``polars-cv/`` with POSIX separators.

    A syntax error is reported as an offender rather than raised: a checker
    that dies on one bad file stops checking the rest, which is the silent-green
    failure this guard exists to prevent.
    """
    if path == AUTHORITY:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - fixtures are valid Python
        return [Offender(path, exc.lineno or 0, "PARSE", f"cannot parse: {exc.msg}")]
    visitor = _Visitor(path)
    visitor.visit(tree)
    return sorted(visitor.offenders, key=lambda o: (o.line, o.rule))
