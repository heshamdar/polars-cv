import io
import json
import sys
import time

import numpy as np
import polars as pl
from PIL import Image

import polars_cv  # noqa
from polars_cv import Pipeline

rng = np.random.default_rng(0)


def png(n):
    b = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (n, n, 3), dtype=np.uint8)).save(
        b, format="PNG"
    )
    return b.getvalue()


big = pl.DataFrame({"img": [png(512) for _ in range(200)]})
blobs = big.select(
    pl.col("img")
    .cv.pipe(Pipeline().source("image_bytes").cast("f32"))
    .sink("blob")
    .alias("b")
)
small = pl.DataFrame({"img": [png(256) for _ in range(100)]})
cases = {
    "blob_f32_scale_numpy": (
        blobs,
        pl.col("b").cv.pipe(Pipeline().source("blob").scale(2.0)).sink("numpy"),
        False,
    ),
    "numpy_output_to_list": (
        small,
        pl.col("img").cv.pipe(Pipeline().source("image_bytes")).sink("numpy"),
        True,
    ),
}
out = {}
for name, (df, e, to_list) in cases.items():
    run = (
        (lambda: df.select(e).to_series().to_list())
        if to_list
        else (lambda: df.select(e))
    )
    run()
    ts = []
    for _ in range(10):
        t = time.perf_counter()
        run()
        ts.append(time.perf_counter() - t)
    out[name] = round(sorted(ts)[5] * 1e3, 1)
json.dump(out, sys.stdout)
