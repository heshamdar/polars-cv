"""
Is Daft's UDF path optimal? Measuring it properly rather than assuming.

Daft's design position is that it does not need a native expression for every
image operation, because a UDF is a first-class, efficiently-executed way to
run one. The `daft-udf` adapter in the main benchmark takes the shape a user
gets by writing an expression chain — **one UDF per operation** — and that is
the pessimal shape if UDF entry costs anything. Reporting it as "Daft's UDF
performance" without checking the fused alternative would be unfair.

This script separates the two questions that "is the UDF path optimal?"
actually contains:

1. **What does one UDF round trip cost, independent of image work?**
   `identity` measures a UDF that returns its input untouched, so the entire
   number is marshalling: Rust column -> Python objects -> Rust column. Divided
   by row count it gives a per-row entry cost that any real UDF pays on top of
   its kernel.

2. **Does fusing recover it?** `chained` applies the heavy pipeline as six
   UDFs, one per operation; `fused` applies the same six operations inside a
   single UDF. The difference is what per-op chaining costs and what an
   optimizing Daft user would get back.

Two access strategies are compared too, since `to_pylist()` materializes Python
objects and `to_arrow()` may not.

Reference points are the plain OpenCV loop (same kernels, no engine) and
polars-cv streaming (same kernels' worth of work, never leaving Rust).

Run::

    PYTHONPATH=. uv run --no-sync python \
        benchmarks/reports/2026-08-23-daft-comparison/udf_path_probe.py
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import numpy as np
from benchmarks.frameworks import OperationParams, get_adapter
from benchmarks.scenarios.pipelines import get_pipeline_benchmarks
from benchmarks.utils.data_gen import generate_image_set

IMAGE_COUNT = 100
IMAGE_SIZE = 256
WARMUP = 3
ITERATIONS = 10


def _measure(fn: Callable[[], Any]) -> tuple[float, float]:
    """
    Time a callable, reporting throughput and average cores kept busy.

    Args:
        fn: Zero-argument callable performing one full batch.

    Returns:
        ``(images_per_second, average_cores_used)``.
    """
    for _ in range(WARMUP):
        fn()
    times = os.times()
    cpu_start = times.user + times.system
    wall_start = time.perf_counter()
    for _ in range(ITERATIONS):
        fn()
    wall = time.perf_counter() - wall_start
    times = os.times()
    cpu = (times.user + times.system) - cpu_start
    return IMAGE_COUNT * ITERATIONS / wall, cpu / wall


def _as_image(arr: "np.ndarray") -> "np.ndarray":
    """
    Give a 2-D result the trailing axis Daft's image dtype requires.

    Args:
        arr: Operation output.

    Returns:
        An array with at least three dimensions.
    """
    return arr[:, :, np.newaxis] if arr.ndim == 2 else arr


def _for_opencv(arr: "np.ndarray") -> "np.ndarray":
    """
    Drop a redundant single-channel axis before handing an array to OpenCV.

    Args:
        arr: Array from a Daft image column.

    Returns:
        The array in the form OpenCV's grayscale checks expect.
    """
    return arr[:, :, 0] if arr.ndim == 3 and arr.shape[2] == 1 else arr


def main() -> int:  # noqa: PLR0915 - a probe script; the sequence is the point
    """
    Run every UDF-shape experiment and print a comparison table.

    Returns:
        Process exit code.
    """
    import daft

    print(f"cores: {os.cpu_count()}   daft: {daft.__version__}")
    print(f"{IMAGE_COUNT} x {IMAGE_SIZE}^2 RGB, {ITERATIONS} timed iterations\n")

    image_set = generate_image_set(
        count=IMAGE_COUNT,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        channels=3,
        pattern="gradient",
    )

    daft_adapter = get_adapter("daft-udf")
    decoded = daft_adapter.prepare_decoded_images(image_set.image_bytes)
    cv = get_adapter("opencv")

    image_dtype = daft.DataType.image()

    # ---------------------------------------------------------------- 1. entry
    @daft.func.batch(return_dtype=image_dtype)
    def identity_pylist(series: Any) -> list["np.ndarray"]:
        return list(series.to_pylist())

    @daft.func.batch(return_dtype=image_dtype)
    def identity_arrow(series: Any) -> list["np.ndarray"]:
        # to_arrow() avoids building a Python list of ndarrays; the column still
        # has to be rebuilt from whatever comes back.
        series.to_arrow()
        return list(series.to_pylist())

    @daft.func.batch(return_dtype=image_dtype)
    def identity_passthrough(series: Any) -> Any:
        # The cheapest thing a batch UDF can possibly do: hand the Series
        # straight back, never materializing a Python object. This is the floor
        # on UDF entry, and no real operation can beat it.
        return series

    print("1. Cost of a UDF round trip with no image work at all")
    print("-" * 72)
    for label, udf in (
        ("return Series as-is", identity_passthrough),
        ("to_pylist()", identity_pylist),
        ("to_arrow()", identity_arrow),
    ):
        rate, cores = _measure(
            lambda u=udf: decoded.select(u(daft.col("images")).alias("o")).collect()
        )
        print(
            f"   identity UDF, {label:<20s} {rate:9.1f} img/s "
            f"({1e6 / rate:7.1f} us/img, {cores:.2f} cores)"
        )
    print()

    # ------------------------------------------------------- 2. chained v fused
    heavy = next(
        b
        for b in get_pipeline_benchmarks(IMAGE_SIZE, IMAGE_SIZE)
        if b.name == "heavy_pipeline"
    )
    ops: list[OperationParams] = heavy.operations
    op_names = " -> ".join(o.operation.name.lower() for o in ops)

    print(f"2. Heavy pipeline ({len(ops)} ops): {op_names}")
    print("-" * 72)

    # (a) One UDF per operation — what an expression chain gives you, and what
    #     the main benchmark's `daft-udf` adapter measures.
    def chained() -> Any:
        return daft_adapter.run_pipeline_on_decoded(decoded, ops)

    # (b) All operations inside a single UDF — what an optimizing user writes.
    probe = cv.load_from_bytes(image_set.image_bytes[0])
    for op in ops:
        probe = _as_image(cv.apply_operation(_for_opencv(probe), op))
    fused_dtype = (
        image_dtype
        if probe.dtype == np.uint8
        else daft.DataType.tensor(daft.DataType.from_numpy_dtype(probe.dtype))
    )

    @daft.func.batch(return_dtype=fused_dtype)
    def fused_udf(series: Any) -> list["np.ndarray"]:
        out = []
        for row in series.to_pylist():
            arr = np.asarray(row)
            for op in ops:
                arr = _as_image(cv.apply_operation(_for_opencv(arr), op))
            out.append(arr)
        return out

    def fused() -> Any:
        return decoded.select(fused_udf(daft.col("images")).alias("o")).collect()

    results: list[tuple[str, float, float]] = []
    for label, fn in (
        ("daft, one UDF per op (chained)", chained),
        ("daft, single fused UDF", fused),
    ):
        rate, cores = _measure(fn)
        results.append((label, rate, cores))

    # Reference points.
    cv_decoded = cv.prepare_decoded_images(image_set.image_bytes)
    rate, cores = _measure(lambda: cv.run_pipeline_on_decoded(cv_decoded, ops))
    results.append(("opencv, plain Python loop", rate, cores))

    for name in ("polars-cv-eager", "polars-cv-streaming"):
        adapter = get_adapter(name)
        frame = adapter.prepare_decoded_images(image_set.image_bytes)
        rate, cores = _measure(
            lambda a=adapter, f=frame: a.run_pipeline_on_decoded(f, ops)
        )
        results.append((name, rate, cores))

    best_daft = max(r for _, r, _ in results[:2])
    for label, rate, cores in results:
        print(f"   {label:<34s} {rate:9.1f} img/s  ({cores:.2f} cores)")
    print()
    chained_rate, fused_rate = results[0][1], results[1][1]
    print(f"   fusing 6 UDFs into 1: {fused_rate / chained_rate:.2f}x")
    for label, rate, _ in results[2:]:
        print(f"   best Daft UDF shape vs {label:<22s} {best_daft / rate:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
