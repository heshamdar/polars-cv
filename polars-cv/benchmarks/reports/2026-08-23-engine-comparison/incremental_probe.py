"""
Pixeltable on its own terms: the incremental computed-column model.

Every scenario in the harness re-runs its pipeline on every iteration, which is
exactly what Pixeltable is built never to do. Its pitch is that you declare a
*computed column*, it materializes once, and every later read is a lookup —
and that when new rows arrive only the new rows are computed. Judging it purely
on recompute throughput measures the one thing its design deliberately avoids,
so this script measures the model instead.

It also insists on a fair counterfactual. "A database that caches beats an
engine that recomputes" is not a finding; anyone using polars-cv for a workload
they read repeatedly would cache too, by writing results to Parquet. So the
comparison here is:

    Pixeltable : add_computed_column once, then read the column
    polars-cv  : compute once and write Parquet, then read the Parquet back

with the ad-hoc recompute cost for both as the baseline, and a break-even
calculation showing how many reads it takes for materializing to pay for
itself.

The last section measures the part polars-cv genuinely has no answer for:
appending rows to a table that already has a computed column, where only the
new rows are computed.

Run::

    PYTHONPATH=. uv run --no-sync python \
        benchmarks/reports/2026-08-23-daft-comparison/incremental_probe.py
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROWS = 200
IMAGE_SIZE = 256
TARGET = 224
READ_ITERATIONS = 5


def _rate(fn: Callable[[], Any], rows: int, iterations: int = READ_ITERATIONS) -> float:
    """
    Time a callable and return throughput in rows/second.

    Args:
        fn: Zero-argument callable performing one full pass.
        rows: Rows processed per pass.
        iterations: Number of timed repetitions.

    Returns:
        Rows per second.
    """
    fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    return rows * iterations / (time.perf_counter() - start)


def _write_images(root: Path, count: int, size: int) -> list[str]:
    """
    Write a deterministic set of PNG files.

    Args:
        root: Directory to write into.
        count: Number of images.
        size: Square image side length.

    Returns:
        The written paths.
    """
    from PIL import Image

    rng = np.random.default_rng(0)
    paths = []
    for index in range(count):
        arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        path = root / f"{index}.png"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    return paths


def probe_pixeltable(paths: list[str]) -> dict[str, float]:
    """
    Measure Pixeltable's ad-hoc, materialize and cached-read costs.

    Args:
        paths: Image file paths to load into a table.

    Returns:
        Named throughput figures in rows/second.
    """
    import pixeltable as pxt

    pxt.init()
    pxt.create_dir("incprobe", if_exists="ignore")
    table = pxt.create_table("incprobe.t", {"img": pxt.Image}, if_exists="replace")
    table.insert({"img": path} for path in paths)

    out: dict[str, float] = {}
    out["decode only"] = _rate(lambda: table.select(o=table.img).collect(), len(paths))
    out["ad-hoc recompute"] = _rate(
        lambda: table.select(o=table.img.resize((TARGET, TARGET))).collect(), len(paths)
    )

    start = time.perf_counter()
    table.add_computed_column(small=table.img.resize((TARGET, TARGET)))
    out["materialize (one-time)"] = len(paths) / (time.perf_counter() - start)

    out["cached read"] = _rate(
        lambda: table.select(o=table.small).collect(), len(paths)
    )

    # Incremental append: only the new rows should be computed.
    extra = paths[: max(1, len(paths) // 10)]
    start = time.perf_counter()
    table.insert({"img": path} for path in extra)
    elapsed = time.perf_counter() - start
    out["incremental append (rows/s)"] = len(extra) / elapsed
    out["_appended"] = float(len(extra))
    out["_append_seconds"] = elapsed
    return out


def probe_polars_cv(paths: list[str], workdir: Path) -> dict[str, float]:
    """
    Measure the same workflow in polars-cv, including a Parquet cache.

    Args:
        paths: Image file paths.
        workdir: Directory for the Parquet cache.

    Returns:
        Named throughput figures in rows/second.
    """
    import polars as pl

    import polars_cv.expressions  # noqa: F401
    from polars_cv import Pipeline

    pipe = (
        Pipeline()
        .source("file_path")
        .resize(height=TARGET, width=TARGET, filter="bilinear")
    )
    frame = pl.DataFrame({"path": paths})

    def recompute() -> Any:
        return (
            frame.lazy()
            .with_columns(small=pl.col("path").cv.pipe(pipe).sink("blob"))
            .collect(engine="streaming")
        )

    out: dict[str, float] = {}
    out["ad-hoc recompute"] = _rate(recompute, len(paths))

    cache = workdir / "cache.parquet"
    start = time.perf_counter()
    recompute().write_parquet(cache)
    out["materialize (one-time)"] = len(paths) / (time.perf_counter() - start)

    out["cached read"] = _rate(
        lambda: pl.read_parquet(cache).select("small"), len(paths)
    )
    return out


def main() -> int:
    """
    Run both probes and print the comparison.

    Returns:
        Process exit code.
    """
    workdir = Path(tempfile.mkdtemp())
    try:
        images = workdir / "images"
        images.mkdir()
        paths = _write_images(images, ROWS, IMAGE_SIZE)
        print(f"{ROWS} x {IMAGE_SIZE}^2 PNGs, resize to {TARGET}^2\n")

        pxt_out = probe_pixeltable(paths)
        pcv_out = probe_polars_cv(paths, workdir)

        print(f"{'':<28s}{'pixeltable':>16s}{'polars-cv':>16s}")
        print("-" * 60)
        for key in ("ad-hoc recompute", "materialize (one-time)", "cached read"):
            left = pxt_out.get(key)
            right = pcv_out.get(key)
            print(
                f"{key:<28s}{left:>13.1f}/s{right:>13.1f}/s"
                if left and right
                else f"{key:<28s}{left:>13.1f}/s{'—':>16s}"
            )
        print(f"{'decode only':<28s}{pxt_out['decode only']:>13.1f}/s{'—':>16s}")
        print()

        # Break-even: reads needed before materializing beats recomputing.
        for name, data in (("pixeltable", pxt_out), ("polars-cv", pcv_out)):
            recompute = data["ad-hoc recompute"]
            build = data["materialize (one-time)"]
            cached = data["cached read"]
            if cached <= recompute:
                print(f"{name}: caching never pays off (cached read is not faster)")
                continue
            # n * (1/recompute) = 1/build + n * (1/cached)
            denominator = (1 / recompute) - (1 / cached)
            breakeven = (1 / build) / denominator if denominator > 0 else float("inf")
            print(
                f"{name}: materializing pays for itself after "
                f"{breakeven:.1f} reads "
                f"(cached read is {cached / recompute:.2f}x the recompute rate)"
            )
        print()
        appended = int(pxt_out["_appended"])
        print(
            f"pixeltable incremental append: {appended} rows added to a "
            f"{ROWS}-row table with a computed column took "
            f"{pxt_out['_append_seconds']:.3f}s "
            f"({pxt_out['incremental append (rows/s)']:.1f} rows/s) — only the "
            "new rows are computed."
        )
        full_rebuild = (ROWS + appended) / pxt_out["materialize (one-time)"]
        print(
            f"  recomputing the whole {ROWS + appended}-row column would cost "
            f"~{full_rebuild:.2f}s, so the append is ~"
            f"{full_rebuild / pxt_out['_append_seconds']:.1f}x cheaper."
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
