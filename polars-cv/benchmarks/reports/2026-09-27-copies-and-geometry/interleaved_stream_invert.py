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


lf = pl.DataFrame({"img": [png() for _ in range(300)]}).lazy()
e = pl.col("img").cv.pipe(Pipeline().source("image_bytes").invert()).sink("blob")


def run():
    return lf.select(e).collect(engine="streaming")


run()
ts = []
for _ in range(15):
    t = time.perf_counter()
    run()
    ts.append(time.perf_counter() - t)
json.dump({"streaming_invert_ms": round(sorted(ts)[7] * 1e3, 1)}, sys.stdout)
