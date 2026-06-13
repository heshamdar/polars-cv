#!/usr/bin/env python
"""Parse benchmark JSON outputs and build comparison tables.

For each (operation, image_count, image_size) cell, shows throughput per
framework and the ratio of each polars-cv mode vs the OpenCV baseline.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("/home/user/bench-results")


def load(path: Path) -> list[dict]:
    text = path.read_text()
    # Strip any leading non-JSON banner lines.
    start = text.find("[")
    start_obj = text.find("{")
    if start == -1 or (start_obj != -1 and start_obj < start):
        start = start_obj
    if start == -1:
        return []
    payload = json.loads(text[start:])
    if isinstance(payload, dict):
        # ResultsCollector.to_json may wrap in {"results": [...]}
        payload = payload.get("results", [])
    return payload


def key(r: dict) -> tuple:
    size = r["image_size"]
    if isinstance(size, (list, tuple)):
        size = tuple(size)
    return (r["operation"], r["image_count"], size)


def report(paths: list[Path], baseline: str = "opencv") -> str:
    rows: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for p in paths:
        if not p.exists():
            continue
        for r in load(p):
            rows[key(r)][r["framework"]] = r

    frameworks = sorted({fw for cell in rows.values() for fw in cell})
    out = []
    header = (
        ["operation", "count", "size"]
        + [f"{fw} img/s" for fw in frameworks]
        + [
            "stream/ocv",
            "eager/ocv",
            "stream/eager",
        ]
    )
    out.append("\t".join(header))
    for k in sorted(rows):
        cell = rows[k]
        line = [k[0], str(k[1]), str(k[2])]
        for fw in frameworks:
            r = cell.get(fw)
            line.append(f"{r['throughput_images_per_second']:.1f}" if r else "-")

        def ratio(a: str, b: str) -> str:
            ra, rb = cell.get(a), cell.get(b)
            if not ra or not rb:
                return "-"
            return f"{ra['throughput_images_per_second'] / rb['throughput_images_per_second']:.2f}"

        line.append(ratio("polars-cv-streaming", baseline))
        line.append(ratio("polars-cv-eager", baseline))
        line.append(ratio("polars-cv-streaming", "polars-cv-eager"))
        out.append("\t".join(line))
    return "\n".join(out)


if __name__ == "__main__":
    paths = [Path(a) for a in sys.argv[1:]] or sorted(RESULTS_DIR.glob("main_*.json"))
    print(report(paths))
