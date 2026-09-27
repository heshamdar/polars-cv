"""Targeted benchmarks for the paths the regression suite does not cover.

Uses only APIs present at both the base commit and the branch head, so the
same script measures both release builds. Prints one JSON object:
{case: best_seconds}. Run with POLARS_MAX_THREADS set.
"""

from __future__ import annotations

import io
import json
import sys
import time

import numpy as np
import polars as pl
from PIL import Image

import polars_cv  # noqa: F401
from polars_cv import Pipeline

RNG = np.random.default_rng(0)
N_IMG = 200
H = W = 512
N_GEOM = 100_000
REPEATS = 5


def best(fn) -> float:
    fn()  # warm-up (and graph compile)
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def encoded(fmt: str) -> list[bytes]:
    out = []
    for _ in range(N_IMG):
        arr = RNG.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        # Smooth images compress like photos rather than noise.
        arr = (arr // 64 * 64).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(
            buf, format=fmt, quality=90
        ) if fmt == "JPEG" else Image.fromarray(arr).save(buf, format=fmt)
        out.append(buf.getvalue())
    return out


def main() -> None:
    results: dict[str, float] = {}
    png = pl.DataFrame({"img": encoded("PNG")})
    jpg = pl.DataFrame({"img": encoded("JPEG")})
    src = Pipeline().source("image_bytes")

    def run(df: pl.DataFrame, expr: pl.Expr) -> None:
        df.select(expr)

    # Image decode (numpy sink is zero-copy, so this is the decode path).
    results["decode_png_to_numpy"] = best(
        lambda: run(png, pl.col("img").cv.pipe(src).sink("numpy"))
    )
    results["decode_jpeg_to_numpy"] = best(
        lambda: run(jpg, pl.col("img").cv.pipe(src).sink("numpy"))
    )

    # Image encode: decode + re-encode; the decode half is measured above.
    results["png_to_png"] = best(
        lambda: run(png, pl.col("img").cv.pipe(src).sink("png"))
    )
    results["jpeg_to_jpeg"] = best(
        lambda: run(jpg, pl.col("img").cv.pipe(src).sink("jpeg"))
    )

    # Tensor sinks.
    results["png_to_array_u8"] = best(
        lambda: run(
            png,
            pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes", dtype="u8"))
            .sink("array", shape=[H, W, 3]),
        )
    )
    results["png_to_list_u8"] = best(
        lambda: run(
            png,
            pl.col("img")
            .cv.pipe(Pipeline().source("image_bytes", dtype="u8"))
            .sink("list"),
        )
    )
    results["png_transpose_to_array_u8"] = best(
        lambda: run(
            png,
            pl.col("img")
            .cv.pipe(
                Pipeline().source("image_bytes", dtype="u8").transpose(axes=[1, 0, 2])
            )
            .sink("array", shape=[W, H, 3]),
        )
    )

    # Blob source passthrough and a fused f32 op.
    blobs = png.select(
        pl.col("img")
        .cv.pipe(Pipeline().source("image_bytes").cast("f32"))
        .sink("blob")
        .alias("b")
    )
    results["blob_f32_to_numpy"] = best(
        lambda: run(blobs, pl.col("b").cv.pipe(Pipeline().source("blob")).sink("numpy"))
    )
    results["blob_f32_scale_to_numpy"] = best(
        lambda: run(
            blobs,
            pl.col("b").cv.pipe(Pipeline().source("blob").scale(2.0)).sink("numpy"),
        )
    )

    # Geometry.
    ring = [
        {"x": float(np.cos(t) * 10 + 10), "y": float(np.sin(t) * 10 + 10)}
        for t in np.linspace(0, 6.2, 40)
    ]
    contour = {"exterior": ring, "holes": [], "is_closed": True}
    geo = pl.DataFrame(
        {
            "c": [contour] * N_GEOM,
            "d": [contour] * N_GEOM,
            "p": [{"x": 10.0, "y": 10.0}] * N_GEOM,
            "q": [{"x": 3.0, "y": 4.0}] * N_GEOM,
        }
    )
    results["contour_area"] = best(lambda: geo.select(pl.col("c").contour.area()))
    results["contour_translate"] = best(
        lambda: geo.select(pl.col("c").contour.translate(1.0, 2.0))
    )
    results["contour_iou"] = best(
        lambda: geo.select(pl.col("c").contour.iou(pl.col("d")))
    )
    results["contour_contains_point"] = best(
        lambda: geo.select(pl.col("c").contour.contains_point(pl.col("p")))
    )
    results["point_translate"] = best(
        lambda: geo.select(pl.col("p").point.translate(1.0, 2.0))
    )
    results["point_distance"] = best(
        lambda: geo.select(pl.col("p").point.distance(pl.col("q")))
    )
    results["point_distance_to_contour"] = best(
        lambda: geo.select(pl.col("p").point.distance_to_contour(pl.col("c")))
    )
    small = geo.head(20_000)
    results["contour_source_rasterize"] = best(
        lambda: small.select(
            pl.col("c")
            .cv.pipe(Pipeline().source("contour").rasterize(width=32, height=32))
            .sink("numpy")
        )
    )

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
