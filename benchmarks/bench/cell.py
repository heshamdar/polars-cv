"""One benchmark cell, run in its own interpreter by ``run.py``.

    python -m bench.cell '{"library": ..., "workload": ..., "source": ...,
                           "threads": ..., "dataset": ..., "runs": 5, "warmup": 1}'

Prints one JSON object as its last line of stdout.
"""

from __future__ import annotations

import json
import sys
import time

import polars as pl

from bench import harness, workloads
from bench.data import DATA


def summarize_output(workload: str, out) -> dict:
    """A few facts about the result, to show the job did the work."""
    if workload == "w3":
        _, dupes = workloads.to_common("w3", out)
        return {"duplicate_groups": len(dupes)}
    if workload == "w1":
        return {"shape": list(getattr(out, "shape", ()))}
    return {"items": len(out)}


def main() -> None:
    cfg = json.loads(sys.argv[1])
    manifest = pl.read_parquet(DATA / f"{cfg['dataset']}.parquet")
    result = dict(cfg, n_images=manifest.height)
    t0 = time.perf_counter()
    try:
        job = workloads.prepare(
            cfg["library"], cfg["workload"], cfg["source"], manifest, cfg["threads"]
        )
    except workloads.NotApplicable as e:
        print(json.dumps(dict(result, status="n/a", reason=str(e))))
        return
    result["prepare_s"] = time.perf_counter() - t0
    del manifest

    first: dict = {}

    def inspect(out) -> None:
        first.update(summarize_output(cfg["workload"], out))

    timing = harness.measure(
        job, runs=cfg["runs"], warmup=cfg["warmup"], inspect=inspect
    )
    median = timing.summary["median_s"]
    result.update(
        status="ok",
        times_s=timing.times_s,
        **timing.summary,
        images_per_s=result["n_images"] / median,
        peak_pss_bytes=timing.peak_pss_bytes,
        baseline_pss_bytes=timing.baseline_pss_bytes,
        added_pss_bytes=timing.peak_pss_bytes - timing.baseline_pss_bytes,
        output=first,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
