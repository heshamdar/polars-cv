"""Timing, memory sampling, thread pinning and environment capture.

Each benchmark cell (workload x library x threads x source x dataset) runs in
its own fresh interpreter, so thread-count environment variables are set before
any library is imported, and no allocator state or cache carries over from the
previous cell.

Memory is the peak **PSS** (proportional set size) summed over the process and
every descendant, sampled every ``SAMPLE_S`` seconds from
``/proc/<pid>/smaps_rollup``. PSS divides a shared page between the processes
mapping it, so a ``fork``-ed worker pool is not charged for the parent's pages
once per worker, as summing RSS would. Off Linux it falls back to RSS.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_S = 0.05

#: Thread-count variables pinned for every cell. Each library reads one or more.
THREAD_ENV = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "DAFT_NUM_THREADS",
    "VIPS_CONCURRENCY",
)


def pinned_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    for k in THREAD_ENV:
        env[k] = str(threads)
    # Tokenizers/HF datasets: no surprise parallelism or progress bars.
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["DAFT_PROGRESS_BAR"] = "0"
    return env


def _pss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except OSError:
        pass
    try:
        return psutil.Process(pid).memory_info().rss // 1024
    except psutil.Error:
        return 0


def tree_pss_bytes(proc: psutil.Process | None = None) -> int:
    """PSS of a process and all its descendants, in bytes."""
    proc = proc or psutil.Process()
    try:
        pids = [proc.pid] + [c.pid for c in proc.children(recursive=True)]
    except psutil.Error:
        pids = [proc.pid]
    return 1024 * sum(_pss_kb(p) for p in pids)


class PeakSampler:
    """Track the peak tree PSS on a background thread while active."""

    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._proc = psutil.Process()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, tree_pss_bytes(self._proc))
            self._stop.wait(SAMPLE_S)

    def __enter__(self) -> PeakSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, tree_pss_bytes(self._proc))


@dataclass
class Timing:
    times_s: list[float]
    peak_pss_bytes: int
    baseline_pss_bytes: int
    summary: dict = field(init=False)

    def __post_init__(self) -> None:
        self.summary = summarize(self.times_s)


def summarize(times: list[float]) -> dict:
    q = (
        statistics.quantiles(times, n=4, method="inclusive")
        if len(times) > 1
        else times * 3
    )
    return {
        "median_s": statistics.median(times),
        "q1_s": q[0],
        "q3_s": q[2],
        "iqr_s": q[2] - q[0],
        "min_s": min(times),
        "max_s": max(times),
    }


def measure(
    fn: Callable[[], object],
    *,
    runs: int,
    warmup: int,
    inspect: Callable[[object], None] | None = None,
) -> Timing:
    """``warmup`` untimed calls, then ``runs`` timed ones under the sampler.

    The result of each call is dropped before the next starts, so a run's
    peak never includes the previous run's output. ``inspect`` sees the first
    call's output (a warmup call when there is one).
    """
    import gc

    gc.collect()
    baseline = tree_pss_bytes()
    for i in range(warmup):
        out = fn()
        if i == 0 and inspect is not None:
            inspect(out)
            inspect = None
        del out
        gc.collect()
    times = []
    with PeakSampler() as sampler:
        for _ in range(runs):
            t0 = time.perf_counter()
            out = fn()
            times.append(time.perf_counter() - t0)
            if inspect is not None:
                inspect(out)
                inspect = None
            del out
            gc.collect()
    return Timing(times, sampler.peak, baseline)


# --- Environment capture ----------------------------------------------------

LIBS = (
    "polars-cv",
    "polars",
    "numpy",
    "pillow",
    "opencv-contrib-python-headless",
    "torch",
    "torchvision",
    "daft",
    "datasets",
    "imagehash",
    "pyarrow",
    "psutil",
)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _locked_versions() -> dict[str, str]:
    lock = ROOT / "uv.lock"
    if not lock.exists():
        return {}
    data = tomllib.loads(lock.read_text())
    return {p["name"]: p.get("version", "") for p in data.get("package", [])}


def environment() -> dict:
    installed = {}
    for name in LIBS:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    locked = _locked_versions()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = ""
    mem = psutil.virtual_memory().total
    return {
        "cpu": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "affinity_cpus": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "memory_bytes": mem,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "libraries": installed,
        "uv_lock": {k: locked[k] for k in sorted(locked) if k in LIBS},
        "git_commit": commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
