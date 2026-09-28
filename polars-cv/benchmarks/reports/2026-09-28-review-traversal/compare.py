"""Compare criterion medians across interleaved base/head rounds.

usage: compare.py DIR  (files [prefix-]rN-<side>.txt; a side starting with
'base' is the base, any other the head)
Prints per case: median-of-rounds base, head, speedup (base/head).
"""

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

UNITS = {"ns": 1e-9, "µs": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}
TIME = re.compile(r"time:\s+\[\S+ \S+ (\S+) (\S+) \S+ \S+\]")


def parse(path: Path) -> dict[str, float]:
    out = {}
    name = None
    for line in path.read_text().splitlines():
        m = TIME.search(line)
        head = line.split("time:")[0].strip()
        if head and not head.startswith(("Benchmarking", "Found", "Warning", "change")):
            if "/" in head and " " not in head:
                name = head
        if m and name:
            out[name] = float(m.group(1)) * UNITS[m.group(2)]
            name = None
    return out


def main() -> None:
    d = Path(sys.argv[1])
    sides: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(d.glob("*r[0-9]*-*.txt")):
        side = "base" if f.stem.rsplit("-", 1)[1].startswith("base") else "head"
        for case, t in parse(f).items():
            sides[side][case].append(t)
    cases = sorted(set(sides["base"]) & set(sides["head"]))
    print(f"{'case':<40} {'base':>10} {'head':>10} {'speedup':>8}")

    def fmt(s: float) -> str:
        return f"{s * 1e3:.3f}ms" if s >= 1e-3 else f"{s * 1e6:.1f}µs"

    for c in cases:
        b = statistics.median(sides["base"][c])
        h = statistics.median(sides["head"][c])
        print(f"{c:<40} {fmt(b):>10} {fmt(h):>10} {b / h:>7.2f}x")


if __name__ == "__main__":
    main()
