"""Known-bad and known-good snippets for the timing-authority ratchet.

`test_timing_authority.py` points the checker at the repo. This file points it
at committed snippets, so the checker's *own* behaviour is pinned rather than
inferred from "the tree is currently clean" — a checker that silently stopped
matching would keep that test green forever.

Both files import `timing_offenders` from `tests._timing_authority`, so these
fixtures exercise the code that actually guards the tree.

The known-good half is the important half. A text scan for `perf_counter` would
flag every docstring, comment and help string that names it, and the fix a
maintainer reaches for under that pressure is a blanket exemption — which ends
the guard. Those three cases are fixtured so the AST-based resolution that
avoids them cannot be traded away for a regex later.
"""

from __future__ import annotations

import pytest

from tests._timing_authority import AUTHORITY, timing_offenders

pytestmark = pytest.mark.structural

SCENARIO = "benchmarks/scenarios/_fixture.py"


# Each entry: (id, source, expected rule). The rule is asserted, not just the
# count, because a checker that rejects the right line for the wrong reason has
# a blind spot the next edit will find.
KNOWN_BAD: list[tuple[str, str, str]] = [
    (
        "plain_perf_counter",
        """
import time

def run(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0
""",
        "R1",
    ),
    (
        "time_monotonic",
        """
import time

def run(fn):
    t0 = time.monotonic()
    fn()
    return time.monotonic() - t0
""",
        "R1",
    ),
    (
        "time_time",
        """
import time

def run(fn):
    start = time.time()
    fn()
    return time.time() - start
""",
        "R1",
    ),
    (
        "time_process_time",
        """
import time

def run(fn):
    start = time.process_time()
    fn()
    return time.process_time() - start
""",
        "R1",
    ),
    (
        "aliased_time_module",
        """
import time as _t

def run(fn):
    t0 = _t.perf_counter()
    fn()
    return _t.perf_counter() - t0
""",
        "R1",
    ),
    (
        "from_time_import",
        """
from time import perf_counter

def run(fn):
    t0 = perf_counter()
    fn()
    return perf_counter() - t0
""",
        "R2",
    ),
    (
        "from_time_import_aliased",
        """
from time import perf_counter as pc

def run(fn):
    t0 = pc()
    fn()
    return pc() - t0
""",
        "R2",
    ),
    (
        "import_timeit",
        """
import timeit

def run(fn):
    return timeit.timeit(fn, number=10)
""",
        "R3",
    ),
    (
        "from_timeit_import",
        """
from timeit import default_timer

def run(fn):
    t0 = default_timer()
    fn()
    return default_timer() - t0
""",
        "R3",
    ),
    (
        "getattr_clock",
        """
import contextlib

def run(fn, clock):
    now = getattr(clock, "perf_counter")
    t0 = now()
    fn()
    return now() - t0
""",
        "R4",
    ),
    (
        "hand_rolled_result_record",
        """
from benchmarks.frameworks import BenchmarkResult

def make(avg_time, count):
    return BenchmarkResult(
        framework="polars-cv-eager",
        operation="resize",
        image_count=count,
        image_size=(256, 256),
        total_time_seconds=avg_time,
        throughput_images_per_second=count / avg_time,
        latency_ms_per_image=(avg_time / count) * 1000,
        peak_memory_mb=0.0,
    )
""",
        "R5",
    ),
    (
        "aliased_result_record",
        """
from benchmarks.frameworks import BenchmarkResult as _R

def make(avg_time, count):
    return _R(framework="x", operation="y", image_count=count,
              image_size=(1, 1), total_time_seconds=avg_time,
              throughput_images_per_second=1.0, latency_ms_per_image=1.0,
              peak_memory_mb=0.0)
""",
        "R5",
    ),
    (
        "qualified_result_record",
        """
from benchmarks import frameworks

def make(avg_time):
    return frameworks.BenchmarkResult(total_time_seconds=avg_time)
""",
        "R5",
    ),
]


# The false positives. Each of these is something a text scan would flag, and
# flagging any of them is what gets a guard exempted into uselessness.
KNOWN_GOOD: list[tuple[str, str]] = [
    (
        "docstring_naming_the_clock",
        '''
def run(fn):
    """Time *fn*.

    Deliberately does not call time.perf_counter itself — the authority in
    benchmarks.utils.timing owns the clock, see its module docstring.
    """
    from benchmarks.utils.timing import measure

    return measure(fn, warmup_fn=fn, warmup=1, iterations=3, label="x")
''',
    ),
    (
        "comment_naming_the_clock",
        """
from benchmarks.utils.timing import measure

# Previously this used time.perf_counter() directly; see CHANGELOG 0.29.0 for
# why every span moved behind `measure`.
def run(fn):
    return measure(fn, warmup_fn=fn, warmup=1, iterations=3, label="x")
""",
    ),
    (
        "string_literal_naming_the_clock",
        """
import argparse

def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--clock",
        default="time.perf_counter",
        help="ignored; retained so old invocations do not break",
    )
    return p
""",
    ),
    (
        "time_sleep_is_not_a_measurement",
        """
import time

def serve_with_latency(handler, latency_ms):
    # The injected delay is the thing being measured, not the measuring.
    time.sleep(latency_ms / 1000.0)
    return handler()
""",
    ),
    (
        "from_time_import_sleep",
        """
from time import sleep

def serve_with_latency(handler, latency_ms):
    sleep(latency_ms / 1000.0)
    return handler()
""",
    ),
    (
        "routes_through_the_authority",
        """
from benchmarks.utils.timing import measure, to_result

def run(adapter, images, ops, count, size):
    stats = measure(
        lambda: adapter.run_pipeline_on_decoded(images, ops),
        warmup_fn=lambda: adapter.run_pipeline_on_decoded(images, ops),
        warmup=3,
        iterations=10,
        label="resize",
    )
    return to_result(
        stats,
        framework=adapter.name,
        engine=adapter.engine,
        operation="resize",
        image_count=count,
        image_size=size,
        memory=None,
    )
""",
    ),
    (
        "unrelated_attribute_named_like_a_clock",
        """
class Server:
    def __init__(self):
        self.monotonic_request_id = 0

def bump(server):
    server.monotonic_request_id += 1
    return server.monotonic_request_id
""",
    ),
    (
        "getattr_on_something_that_is_not_a_clock",
        """
def read(config):
    return getattr(config, "peak_memory_mb", None)
""",
    ),
]


@pytest.mark.parametrize(
    ("source", "rule"),
    [(src, rule) for _, src, rule in KNOWN_BAD],
    ids=[name for name, _, _ in KNOWN_BAD],
)
def test_known_bad_is_rejected(source: str, rule: str) -> None:
    offenders = timing_offenders(source, path=SCENARIO)
    assert offenders, (
        "this snippet bypasses the timing authority and the checker did not "
        "notice:\n" + source
    )
    rules = {o.rule for o in offenders}
    assert rule in rules, (
        f"expected rule {rule}, got {sorted(rules)} — the snippet is rejected, "
        f"but for the wrong reason, so the rule it is meant to pin is untested."
    )


@pytest.mark.parametrize(
    "source",
    [src for _, src in KNOWN_GOOD],
    ids=[name for name, _ in KNOWN_GOOD],
)
def test_known_good_is_accepted(source: str) -> None:
    offenders = timing_offenders(source, path=SCENARIO)
    assert not offenders, (
        "this snippet is legitimate and the checker flagged it; a false "
        "positive here is what gets the guard exempted into uselessness:\n"
        + "\n".join(f"  {o}" for o in offenders)
        + "\n"
        + source
    )


def test_the_authority_itself_is_exempt() -> None:
    """The one module that may read a clock must not be reported for doing so."""
    source = KNOWN_BAD[0][1]
    assert timing_offenders(source, path=SCENARIO), "fixture no longer offends"
    assert not timing_offenders(source, path=AUTHORITY)
