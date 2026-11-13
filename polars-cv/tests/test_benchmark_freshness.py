"""CR-149: the published cross-library benchmarks must measure a recent release.

A known gap, pinned as `test_known_gaps.py` pins its others (strict xfail,
only an ``AssertionError`` is the gap, so the refresh turns the suite red and
the marker goes). It lives here rather than there because it reads repository
files through ``tests._discovery``, which needs a module-level lane, and that
module's gaps each choose their own.
"""

from __future__ import annotations

import pytest

from tests._discovery import requires_checkout

pytestmark = [pytest.mark.structural, requires_checkout]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CR-149: the published benchmarks report 0.38.0",
)
def test_the_published_benchmarks_measure_a_recent_release() -> None:
    """The cross-library results (``benchmarks/results/coco-*.json``, which
    the README's table and figures are made from) each record the polars-cv
    they measured. Fixed when every one is re-measured on a release within
    one minor of the checkout's version (the plan runs that with the release
    that adopts the ``tensor`` sink)."""
    import json

    import polars_cv
    from tests._discovery import benchmark_results

    # The checkout's version (`test_version_consistency.py` holds it to
    # `pyproject.toml`).
    current = polars_cv.__version__

    def minor(version: str) -> tuple[int, int]:
        major, minor_, *_ = version.split(".")
        return int(major), int(minor_)

    # Raises `EmptyDiscovery`, not an AssertionError, when there are none, so
    # a missing results directory never reads as the gap.
    results = benchmark_results()
    stale = []
    for path in results:
        env = json.loads(path.read_text())["environment"]["libraries"]
        measured = env["polars-cv"]
        (m_major, m_minor), (c_major, c_minor) = minor(measured), minor(current)
        if m_major != c_major or c_minor - m_minor > 1:
            stale.append(f"{path.name} measured {measured}")
    assert not stale, f"the checkout is {current}; " + ", ".join(stale)
