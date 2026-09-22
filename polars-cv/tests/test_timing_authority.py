"""`benchmarks/utils/timing.py` is the only thing in `benchmarks/` that times.

This runs `_timing_authority.timing_offenders` over the benchmark tree. The
fixtures for the checker itself live in `test_timing_authority_fixtures.py`;
this file is the part that points it at the repo.

Discovery goes through :func:`tests._discovery.benchmark_modules`, which raises
rather than returning an empty set — so this guard cannot pass by finding
nothing, which is the failure mode it would otherwise share with every other
source scan here.

Structural: it reads source and needs no compiled extension, so pre-commit can
run it.
"""

from __future__ import annotations

import pytest

from tests._discovery import REPO_ROOT, benchmark_modules
from tests._timing_authority import AUTHORITY, Offender, timing_offenders

pytestmark = pytest.mark.structural

PACKAGE_ROOT = REPO_ROOT / "polars-cv"


def _scan() -> list[Offender]:
    offenders: list[Offender] = []
    for module in benchmark_modules():
        rel = module.relative_to(PACKAGE_ROOT).as_posix()
        offenders.extend(timing_offenders(module.read_text(), path=rel))
    return offenders


def test_the_authority_exists_where_the_ratchet_exempts_it() -> None:
    """The exempt path must name a real module.

    If `AUTHORITY` drifts from the file's actual location the exemption stops
    matching, and the authority starts reporting *itself* — which reads as a
    broken tree rather than as a broken constant, and gets fixed by exempting
    something wider.
    """
    assert (PACKAGE_ROOT / AUTHORITY).is_file(), (
        f"the timing authority {AUTHORITY} does not exist at that path, so the "
        f"ratchet's exemption matches nothing."
    )


def test_timing_authority_is_the_only_clock() -> None:
    """No module under `benchmarks/` may read a clock or build a result record.

    Both halves matter. Banning the clock stops a hand-rolled timer; banning
    `BenchmarkResult(...)` stops a hand-rolled *statistic*, which is what
    actually diverged — four scenarios each reduced their samples with their own
    arithmetic while two other modules used a different statistic entirely.
    """
    offenders = _scan()
    assert not offenders, (
        f"{len(offenders)} timing-authority violation(s); route these through "
        f"`benchmarks.utils.timing` (`measure`/`measure_phase`/`to_result`):\n"
        + "\n".join(f"  {o}" for o in offenders)
    )
