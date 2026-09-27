"""Head vs a band of base runs: a change counts only outside every base run.

Usage: band.py head.json base_a.json [base_b.json ...]
"""

import json
import sys


def load(path):
    return {
        (
            r["framework"],
            r["operation"],
            tuple(r["image_size"])
            if isinstance(r["image_size"], list)
            else r["image_size"],
            r["image_count"],
        ): r["throughput_images_per_second"]
        for r in json.load(open(path))
    }


head = load(sys.argv[1])
bases = [load(p) for p in sys.argv[2:]]
MARGIN = 0.07
rows = []
for key, h in sorted(head.items()):
    vals = [b[key] for b in bases if key in b]
    lo, hi = min(vals), max(vals)
    mid = sum(vals) / len(vals)
    if h > hi * (1 + MARGIN):
        verdict = "IMPROVED"
    elif h < lo * (1 - MARGIN):
        verdict = "REGRESSED"
    else:
        verdict = "within noise"
    rows.append((verdict, key, (h / mid - 1) * 100, (hi / lo - 1) * 100))
for verdict in ("REGRESSED", "IMPROVED", "within noise"):
    sel = [r for r in rows if r[0] == verdict]
    print(f"\n{verdict}: {len(sel)}")
    if verdict != "within noise":
        for _, (fw, op, size, n), delta, spread in sel:
            print(
                f"  {fw:20} {op:32} {delta:+7.1f}% vs base mean  (base spread {spread:.1f}%)"
            )
