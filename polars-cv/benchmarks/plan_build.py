"""Time how long it takes to *build* pipelines (no execution).

The typed-op migration (``TYPED_OPS_PLAN.md``) changes what every builder call
does: today each append runs several FFI round-trips over JSON; afterwards it is
one typed push into a Rust plan. The regression harness measures execution
throughput, not this, so this script is the baseline and the check for it.

Reports the best of ``--repeats`` runs, in microseconds per append, for a mix of
cheap ops, per-row expression params and a lazy continuation.

Usage::

    python -m benchmarks.plan_build [--appends 100] [--repeats 7]
"""

from __future__ import annotations

import argparse
import json
import time

import polars as pl

from polars_cv import Pipeline


def _chain(appends: int) -> None:
    pipe = Pipeline().source("image_bytes")
    for i in range(appends):
        pipe = pipe.scale(1.0) if i % 2 else pipe.clamp(0.0, 255.0)


def _mixed(appends: int) -> None:
    pipe = Pipeline().source("image_bytes")
    for _ in range(appends // 4):
        pipe = (
            pipe.resize(height=pl.col("h"), width=64)
            .blur(1.0)
            .crop(top=0, left=0, height=32, width=32)
            .grayscale()
        )


def _lazy(appends: int) -> None:
    node = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
    for _ in range(appends // 2):
        node = node.pipe(Pipeline().scale(2.0).clamp(0.0, 255.0))


SCENARIOS = {"chain": _chain, "mixed": _mixed, "lazy_continuation": _lazy}


def measure(appends: int, repeats: int) -> dict[str, float]:
    results = {}
    for name, build in SCENARIOS.items():
        best = min(_timed(build, appends) for _ in range(repeats))
        results[name] = round(best / appends * 1e6, 1)
    return results


def _timed(build, appends: int) -> float:
    start = time.perf_counter()
    build(appends)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appends", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps({"us_per_append": measure(args.appends, args.repeats)}))


if __name__ == "__main__":
    main()
