"""
Ingestion micro-benchmark: what each source costs per row.

Every case decodes the same u8 pixels from a different column layout and
passes them straight to a ``blob`` sink, so the differences between cases are
the source decode (the performance plan's Phase 6). The fixed-size ``array``
source reads rows in place and is the reference.

Run directly:

    uv run python benchmarks/ingestion_overhead.py [--rows 20000] [--size 64]

Cases:
  - array:           Array[Array[u8]] column, read in place
  - list u8:         List[List[u8]], the same values
  - list f32 flat:   List[f32], one row = size*size values
  - list i64 -> u8:  List[List[i64]] decoded as u8 (a converting copy)
  - raw aligned:     raw u8 rows whose lengths keep every row 8-byte aligned
  - raw odd:         raw u8 rows of odd length, so most rows start unaligned
  - blob aligned:    VIEW blobs of 8-byte multiple length
  - blob odd:        VIEW blobs of odd length (u8 payloads need no alignment)

The ``odd`` cases use (size-1)x(size-1) images, so the row lengths are odd.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import polars as pl

from polars_cv import Pipeline


def _blobs(images: np.ndarray) -> pl.Series:
    """One VIEW blob per image, encoded by the plugin itself."""
    arr = pl.Series("a", images)
    return (
        pl.DataFrame({"a": arr})
        .select(
            pl.col("a").cv.pipe(Pipeline().source("array", dtype="u8")).sink("blob")
        )
        .to_series()
    )


def _columns(rows: int, size: int) -> dict[str, tuple[pl.Series, Pipeline]]:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(rows, size, size), dtype=np.uint8)
    odd = images[:, : size - 1, : size - 1].copy()
    array = pl.Series("c", images)
    return {
        "array": (array, Pipeline().source("array", dtype="u8")),
        "list u8": (
            array.cast(pl.List(pl.List(pl.UInt8))),
            Pipeline().source("list", dtype="u8"),
        ),
        "list f32 flat": (
            pl.Series("c", images.reshape(rows, -1).astype(np.float32)).cast(
                pl.List(pl.Float32)
            ),
            Pipeline().source("list", dtype="f32"),
        ),
        "list i64 -> u8": (
            array.cast(pl.List(pl.List(pl.Int64))),
            Pipeline().source("list", dtype="u8"),
        ),
        "raw aligned": (
            pl.Series("c", [im.tobytes() for im in images], dtype=pl.Binary),
            Pipeline().source("raw", dtype="u8"),
        ),
        "raw odd": (
            pl.Series("c", [im.tobytes() for im in odd], dtype=pl.Binary),
            Pipeline().source("raw", dtype="u8"),
        ),
        "blob aligned": (_blobs(images), Pipeline().source("blob", dtype="u8")),
        "blob odd": (_blobs(odd), Pipeline().source("blob", dtype="u8")),
    }


def _time(fn, repeat: int) -> tuple[float, float]:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--size", type=int, default=64, help="square u8 image side")
    parser.add_argument(
        "--only", default=None, help="run only cases whose name contains this"
    )
    args = parser.parse_args()

    print(f"rows={args.rows} image={args.size}x{args.size} u8 repeat={args.repeat}")
    print(
        f"{'case':<16} {'mode':<10} {'best (s)':>10} {'median (s)':>11} {'µs/row':>8}"
    )
    for name, (column, pipe) in _columns(args.rows, args.size).items():
        if args.only and args.only not in name:
            continue
        df = pl.DataFrame({"c": column})
        expr = pl.col("c").cv.pipe(pipe).sink("blob")

        def eager(df=df, expr=expr):
            df.with_columns(out=expr)

        def streaming(df=df, expr=expr):
            df.lazy().with_columns(out=expr).collect(engine="streaming")

        for mode, fn in (("eager", eager), ("streaming", streaming)):
            best, median = _time(fn, args.repeat)
            print(
                f"{name:<16} {mode:<10} {best:>10.3f} {median:>11.3f} "
                f"{median / args.rows * 1e6:>8.2f}"
            )


if __name__ == "__main__":
    main()
