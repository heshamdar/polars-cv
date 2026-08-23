"""
Parallelism probe: is the Daft comparison in `README.md` a fair fight?

Every scenario in the harness hands a framework one in-memory batch, which for
Daft means a DataFrame built by `from_pydict`. Two things could make the
headline numbers unfair to Daft, and this script checks both:

1. **Partitioning.** If that batch lands as a single partition with no
   intra-partition parallelism, the tables would be pitting a one-core Daft
   against a four-core polars-cv. Sweeping `into_partitions` shows whether
   partitioning is a lever that was withheld.

2. **Core utilization.** Measuring CPU time against wall time gives the average
   number of cores each engine actually kept busy, which turns "engine A is 5x
   faster" into "engine A is 5x faster *and* here is how much of that came from
   using more cores".

Run::

    PYTHONPATH=. uv run --no-sync python \
        benchmarks/reports/2026-08-23-daft-comparison/parallelism_probe.py
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from benchmarks.frameworks import OperationParams, OperationType, get_adapter
from benchmarks.utils.data_gen import generate_image_set

#: Partition counts to sweep. 4 is this machine's core count.
PARTITIONS: tuple[int, ...] = (1, 2, 4, 8)

IMAGE_COUNT = 100
IMAGE_SIZE = 256
WARMUP = 3
ITERATIONS = 10

RESIZE = OperationParams(operation=OperationType.RESIZE, height=224, width=224)
BLUR = OperationParams(operation=OperationType.BLUR, sigma=2.0)


def _cpu_seconds() -> float:
    """
    Total CPU time consumed by this process and its threads.

    Returns:
        User + system CPU seconds.
    """
    times = os.times()
    return times.user + times.system


def _measure(fn: Callable[[], Any]) -> tuple[float, float]:
    """
    Time a callable, reporting throughput and how many cores it kept busy.

    Args:
        fn: Zero-argument callable performing one full batch.

    Returns:
        ``(images_per_second, average_cores_used)``.
    """
    for _ in range(WARMUP):
        fn()
    cpu_start = _cpu_seconds()
    wall_start = time.perf_counter()
    for _ in range(ITERATIONS):
        fn()
    wall = time.perf_counter() - wall_start
    cpu = _cpu_seconds() - cpu_start
    return IMAGE_COUNT * ITERATIONS / wall, cpu / wall


def probe_partitioning(image_set: Any) -> None:
    """
    Sweep partition counts for a native and a UDF operation.

    Args:
        image_set: Generated benchmark images.
    """
    print("PARTITION SWEEP")
    print("-" * 70)
    cases = [
        ("resize 224 (Daft native expression)", "daft", RESIZE),
        ("blur sigma=2 (Daft batch UDF)", "daft-udf", BLUR),
    ]
    for label, adapter_name, params in cases:
        adapter = get_adapter(adapter_name)
        decoded = adapter.prepare_decoded_images(image_set.image_bytes)
        print(f"  {label}")
        baseline = None
        for count in PARTITIONS:
            frame = decoded.into_partitions(count).collect()
            rate, _ = _measure(
                lambda f=frame: adapter.run_pipeline_on_decoded(f, [params])
            )
            baseline = baseline or rate
            print(
                f"     {count} partition(s): {rate:8.1f} img/s "
                f"({rate / baseline:.2f}x vs 1)"
            )
        print()


def probe_core_utilization(image_set: Any) -> None:
    """
    Report throughput and average cores used, per framework and operation.

    Args:
        image_set: Generated benchmark images.
    """
    print("CORE UTILIZATION")
    print("-" * 70)
    print(f"  {'framework':<22s}{'operation':<14s}{'img/s':>10s}{'cores used':>13s}")
    frameworks = [
        "polars-cv-eager",
        "polars-cv-streaming",
        "daft",
        "daft-udf",
        "opencv",
    ]
    for name in frameworks:
        adapter = get_adapter(name)
        if not adapter.is_available():
            continue
        decoded = adapter.prepare_decoded_images(image_set.image_bytes)
        for op_label, params in (("resize", RESIZE), ("blur", BLUR)):
            try:
                rate, cores = _measure(
                    lambda p=params: adapter.run_pipeline_on_decoded(decoded, [p])
                )
            except NotImplementedError:
                print(f"  {name:<22s}{op_label:<14s}{'n/a':>10s}{'':>13s}")
                continue
            print(f"  {name:<22s}{op_label:<14s}{rate:>10.1f}{cores:>13.2f}")
    print()


def main() -> int:
    """
    Run both probes.

    Returns:
        Process exit code.
    """
    import daft

    print(f"cores: {os.cpu_count()}   daft: {daft.__version__}")
    print(f"{IMAGE_COUNT} x {IMAGE_SIZE}^2 RGB images, {ITERATIONS} timed iterations\n")

    image_set = generate_image_set(
        count=IMAGE_COUNT,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        channels=3,
        pattern="gradient",
    )
    probe_partitioning(image_set)
    probe_core_utilization(image_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
