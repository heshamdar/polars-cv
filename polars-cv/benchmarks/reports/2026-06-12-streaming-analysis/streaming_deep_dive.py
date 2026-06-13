#!/usr/bin/env python
"""Streaming-engine-focused experiments for polars-cv.

Run with: POLARS_MAX_THREADS=N [POLARS_IDEAL_MORSEL_SIZE=M] python streaming_deep_dive.py <label>

Experiments:
  1. thread/morsel scaling of streaming vs eager on a realistic pipeline
  2. fixed per-row overhead (tiny images, trivial op)
  3. graph-JSON / compile overhead sensitivity (pipeline length on tiny images)
  4. decode vs op vs encode share (image_bytes vs blob source; blob vs numpy sink)
"""

from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
import polars as pl
from PIL import Image

import polars_cv.expressions  # noqa: F401
from polars_cv import Pipeline

LABEL = sys.argv[1] if len(sys.argv) > 1 else "default"
REPS = int(os.environ.get("DD_REPS", "5"))


def make_pngs(n: int, h: int, w: int) -> list[bytes]:
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    png = buf.getvalue()
    return [png] * n


def to_blob_df(pngs: list[bytes]) -> pl.DataFrame:
    pipe = Pipeline().source("image_bytes")
    df = pl.DataFrame({"images": pngs})
    res = df.with_columns(blob=pl.col("images").cv.pipe(pipe).sink("blob"))
    return pl.DataFrame({"images": res["blob"]})


def bench(df: pl.DataFrame, expr: pl.Expr, streaming: bool, reps: int = REPS) -> float:
    times = []
    for _ in range(reps + 1):
        t = time.perf_counter()
        if streaming:
            df.lazy().with_columns(out=expr).collect(engine="streaming")
        else:
            df.with_columns(out=expr)
        times.append(time.perf_counter() - t)
    return min(times[1:])  # drop warmup, take best


results = []


def record(exp: str, name: str, n: int, secs: float) -> None:
    results.append(
        {
            "label": LABEL,
            "experiment": exp,
            "case": name,
            "rows": n,
            "seconds": secs,
            "rows_per_sec": n / secs,
            "threads": pl.thread_pool_size(),
            "morsel_size": os.environ.get("POLARS_IDEAL_MORSEL_SIZE", "default"),
        }
    )
    print(
        f"[{LABEL}] {exp}/{name}: rows={n} t={secs * 1000:.1f}ms "
        f"({n / secs:.0f} rows/s)",
        flush=True,
    )


# ---- Experiment 1: realistic pipeline, eager vs streaming ----
N = 1000
pngs = make_pngs(N, 256, 256)
blob_df = to_blob_df(pngs)
imagenet = (
    Pipeline()
    .source("blob")
    .resize(height=256, width=256, filter="bilinear")
    .crop(top=16, left=16, height=224, width=224)
    .normalize(method="minmax")
)
expr = pl.col("images").cv.pipe(imagenet).sink("blob")
record("imagenet_blob_1000", "eager", N, bench(blob_df, expr, False))
record("imagenet_blob_1000", "streaming", N, bench(blob_df, expr, True))

# ---- Experiment 2: fixed per-row overhead (8x8 images, flip) ----
N2 = 20000
tiny = make_pngs(1, 8, 8) * N2
tiny_df = to_blob_df(tiny)
flip = Pipeline().source("blob").flip_h()
expr2 = pl.col("images").cv.pipe(flip).sink("blob")
record("tiny_8x8_flip_20000", "eager", N2, bench(tiny_df, expr2, False))
record("tiny_8x8_flip_20000", "streaming", N2, bench(tiny_df, expr2, True))

# ---- Experiment 3: graph length sensitivity on tiny images ----
long_pipe = Pipeline().source("blob")
for _ in range(10):
    long_pipe = long_pipe.flip_h().flip_v()
expr3 = pl.col("images").cv.pipe(long_pipe).sink("blob")
record("tiny_8x8_20ops_20000", "eager", N2, bench(tiny_df, expr3, False))
record("tiny_8x8_20ops_20000", "streaming", N2, bench(tiny_df, expr3, True))

# ---- Experiment 4: decode/encode share at 256px ----
N4 = 500
pngs4 = make_pngs(N4, 256, 256)
png_df = pl.DataFrame({"images": pngs4})
blob_df4 = to_blob_df(pngs4)
rsz_png = (
    Pipeline().source("image_bytes").resize(height=224, width=224, filter="bilinear")
)
rsz_blob = Pipeline().source("blob").resize(height=224, width=224, filter="bilinear")
record(
    "decode_share_resize_500",
    "png_source_streaming",
    N4,
    bench(png_df, pl.col("images").cv.pipe(rsz_png).sink("blob"), True),
)
record(
    "decode_share_resize_500",
    "blob_source_streaming",
    N4,
    bench(blob_df4, pl.col("images").cv.pipe(rsz_blob).sink("blob"), True),
)
record(
    "sink_share_resize_500",
    "numpy_sink_streaming",
    N4,
    bench(blob_df4, pl.col("images").cv.pipe(rsz_blob).sink("numpy"), True),
)
# identity (no op): pure decode+encode round trip
ident = Pipeline().source("blob")
record(
    "identity_blob_500",
    "streaming",
    N4,
    bench(blob_df4, pl.col("images").cv.pipe(ident).sink("blob"), True),
)

out_path = f"/home/user/bench-results/deep_dive_{LABEL}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=1)
print(f"wrote {out_path}")
