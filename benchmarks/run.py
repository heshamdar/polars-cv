"""Run benchmark cells and write one results JSON.

    uv run python run.py                                  # everything, coco-5k
    uv run python run.py --workload w1 --library polars-cv-eager,pil-mp --threads 4
    uv run python run.py --dataset coco-50k --source path --threads max

Each cell (workload x library x threads x source) runs in a fresh interpreter
with every thread-count variable pinned (``harness.THREAD_ENV``). Cells a
library cannot run are recorded as ``n/a`` with the reason, so a results file
lists every combination that was asked for.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

from bench import harness
from bench.workloads import LIBRARIES, WORKLOADS, cpu_count

ROOT = Path(__file__).resolve().parent


def _list(value: str, choices: tuple[str, ...]) -> list[str]:
    if value == "all":
        return list(choices)
    items = value.split(",")
    unknown = [x for x in items if x not in choices]
    if unknown:
        raise SystemExit(f"unknown {unknown}; choose from {choices}")
    return items


def _threads(value: str) -> list[int]:
    return [cpu_count() if t == "max" else int(t) for t in value.split(",")]


def _kill_tree(proc: psutil.Process) -> None:
    for p in [*proc.children(recursive=True), proc]:
        with contextlib.suppress(psutil.Error):
            p.kill()


def run_cell(cfg: dict, timeout: float, mem_limit: float | None = None) -> dict:
    """Run one cell in a fresh interpreter.

    With ``mem_limit`` (bytes), the cell's process tree PSS is polled every
    second and the whole tree killed past it, recorded as ``memory-limit``:
    a cell that would exhaust the machine fails alone instead of the kernel's
    OOM killer choosing a victim.
    """
    env = harness.pinned_env(cfg["threads"])
    cmd = [sys.executable, "-m", "bench.cell", json.dumps(cfg)]
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        child = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=out, stderr=err, text=True
        )
        tree = psutil.Process(child.pid)
        start, peak = time.monotonic(), 0
        while child.poll() is None:
            if time.monotonic() - start > timeout:
                _kill_tree(tree)
                child.wait()
                return dict(cfg, status="timeout", reason=f"exceeded {timeout:.0f}s")
            if mem_limit is not None:
                peak = max(peak, harness.tree_pss_bytes(tree))
                if peak > mem_limit:
                    _kill_tree(tree)
                    child.wait()
                    return dict(
                        cfg,
                        status="memory-limit",
                        reason=f"process tree PSS passed {mem_limit / 2**30:.1f} GiB",
                    )
            time.sleep(1.0 if mem_limit is not None else 0.2)
        out.seek(0)
        err.seek(0)
        proc = subprocess.CompletedProcess(
            cmd, child.returncode, out.read(), err.read()
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
        return dict(
            cfg, status="error", returncode=proc.returncode, reason="\n".join(tail)
        )
    return json.loads(lines[-1])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--workload", default="all")
    p.add_argument("--library", default="all")
    p.add_argument("--threads", default="max", help="comma list, e.g. 1,max; 'max' = all cores")
    p.add_argument("--source", default="bytes,path")
    p.add_argument("--dataset", default="coco-5k")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument(
        "--mem-limit-gib",
        type=float,
        default=None,
        help="kill a cell whose process tree PSS passes this (recorded as memory-limit)",
    )
    p.add_argument("--out", type=Path, default=None, help="results JSON path")
    args = p.parse_args()

    cells = [
        {
            "workload": w,
            "library": lib,
            "threads": t,
            "source": src,
            "dataset": args.dataset,
            "runs": args.runs,
            "warmup": args.warmup,
        }
        for w, lib, t, src in itertools.product(
            _list(args.workload, WORKLOADS),
            _list(args.library, LIBRARIES),
            sorted(set(_threads(args.threads))),
            _list(args.source, ("bytes", "path")),
        )
    ]
    out = (
        args.out
        or ROOT / "results" / f"{args.dataset}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    )
    data = {
        "environment": harness.environment(),
        "args": vars(args) | {"out": str(out)},
        "cells": [],
    }
    for i, cfg in enumerate(cells, 1):
        limit = args.mem_limit_gib * 2**30 if args.mem_limit_gib else None
        res = run_cell(cfg, args.timeout, limit)
        data["cells"].append(res)
        harness.dump(out, data)  # after every cell, so a long run is never lost
        tag = f"{cfg['workload']} {cfg['library']:<20} t={cfg['threads']} {cfg['source']:<5}"
        if res["status"] == "ok":
            msg = f"{res['images_per_s']:8.1f} img/s  median {res['median_s']:.2f}s  IQR {res['iqr_s']:.2f}s  +{res['added_pss_bytes'] / 2**20:.0f} MiB"
        else:
            msg = f"{res['status']}: {res.get('reason', '')[:100]}"
        print(f"[{i}/{len(cells)}] {tag} {msg}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
