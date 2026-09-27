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


def png():
    b = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(
        b, format="PNG"
    )
    return b.getvalue()


df = pl.DataFrame({"img": [png() for _ in range(300)]})
cases = {
    "decode_only_numpy": (Pipeline().source("image_bytes"), "numpy"),
    "invert_blob": (Pipeline().source("image_bytes").invert(), "blob"),
    "normalize_blob": (Pipeline().source("image_bytes").normalize(), "blob"),
    "threshold_blob": (
        Pipeline().source("image_bytes").grayscale().threshold(128),
        "blob",
    ),
}
out = {}
for name, (p, sink) in cases.items():
    e = pl.col("img").cv.pipe(p).sink(sink)
    df.select(e)
    ts = []
    for _ in range(10):
        t = time.perf_counter()
        df.select(e)
        ts.append(time.perf_counter() - t)
    out[name] = sorted(ts)[len(ts) // 2]
json.dump(out, sys.stdout)
