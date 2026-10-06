# polars 2.0.0: streaming sort/group_by under an unreachable OOC budget can spin
# forever in polars_ooc::MemoryManager::do_spill. Run with:
#   POLARS_OOC_MEMORY_BUDGET_MB=1 POLARS_OOC_SPILL_MIN_BYTES=0 python ooc_livelock_repro.py
# Hangs in roughly 1 run in 4-6 on a 4-core machine; otherwise prints OK.

import numpy as np
import polars as pl

n, side = 300, 64
rng = np.random.default_rng(0)
pixels = [rng.integers(0, 255, side * side * 3, dtype=np.uint8) for _ in range(n)]
df = pl.DataFrame(
    {
        "key": [i % 17 for i in range(n)],
        "blob": [p.tobytes() for p in pixels],
        "rec": [{"data": p.tobytes(), "shape": [side, side, 3]} for p in pixels],
        "arr": pl.Series(np.stack(pixels).reshape(n, side, side, 3)),
    }
)
lf = df.lazy()
for q in (
    lf.sort("key", maintain_order=True),
    lf.group_by("key").agg(pl.all().first()).sort("key"),
):
    assert q.collect(engine="streaming").equals(q.collect(engine="in-memory"))
print("OK")
