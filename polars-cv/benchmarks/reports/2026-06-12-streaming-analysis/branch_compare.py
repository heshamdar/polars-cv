#!/usr/bin/env python
"""Build the main-vs-branch comparison table (streaming mode), with OpenCV ref."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/user/bench-results")
from analyze import load  # noqa: E402

D = Path("/home/user/bench-results")


def collect(files: list[str]) -> dict:
    rows: dict = defaultdict(dict)
    for f in files:
        p = D / f
        if not p.exists():
            continue
        for r in load(p):
            size = (
                tuple(r["image_size"])
                if isinstance(r["image_size"], list)
                else r["image_size"]
            )
            rows[(r["operation"], r["image_count"], size)][r["framework"]] = r[
                "throughput_images_per_second"
            ]
    return rows


main_rows = collect(
    [
        "main_single_ops.json",
        "main_pipelines.json",
        "main_pipelines_1000.json",
        "main_e2e.json",
    ]
)
branch_rows = collect(
    [
        "branch_single_ops.json",
        "branch_pipelines.json",
        "branch_pipelines_1000.json",
        "branch_e2e.json",
    ]
)

print(
    "| operation | n | size | opencv | main stream | branch stream "
    "| branch/main | branch stream/ocv |"
)
print("|---|---|---|---|---|---|---|---|")
gains = []
ocv_ratios = []
for k in sorted(branch_rows):
    op, n, size = k
    if n != 100 and not op.startswith("e2e") and n != 1000:
        continue
    m = main_rows.get(k, {})
    b = branch_rows[k]
    ocv = m.get("opencv")
    ms = m.get("polars-cv-streaming")
    bs = b.get("polars-cv-streaming")
    if not (ms and bs):
        continue
    gain = bs / ms
    gains.append((gain, op))
    line = f"| {op} | {n} | {size[0]} | "
    line += f"{ocv:,.0f} | " if ocv else "— | "
    line += f"{ms:,.0f} | {bs:,.0f} | **{gain:.2f}×** | "
    if ocv:
        ocv_ratios.append(bs / ocv)
        line += f"{bs / ocv:.2f}× |"
    else:
        line += "— |"
    print(line)

g = math.prod(x for x, _ in gains) ** (1 / len(gains))
print(f"\nGeomean branch/main (streaming): {g:.2f}x over {len(gains)} cells")
if ocv_ratios:
    go = math.prod(ocv_ratios) ** (1 / len(ocv_ratios))
    print(f"Geomean branch-streaming/opencv: {go:.2f}x")
biggest = sorted(gains, reverse=True)[:8]
print("Largest gains:", ", ".join(f"{op} {x:.2f}x" for x, op in biggest))
worst = sorted(gains)[:5]
print("Regressions/flat:", ", ".join(f"{op} {x:.2f}x" for x, op in worst))
