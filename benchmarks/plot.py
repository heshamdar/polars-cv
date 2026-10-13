"""Charts and tables from results JSON.

    uv run python plot.py results/coco-5k.json [results/coco-50k.json]

Writes SVGs next to the results and prints markdown tables (paste them into
README.md). Static SVG, so values are labelled on the bars rather than shown
on hover. polars-cv is drawn in blue and every other library in gray: the
chart answers "where does polars-cv stand", and a hue per library would need
more distinguishable colours than a bar chart can carry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.workloads import LIBRARIES, WORKLOADS

TITLES = {
    "w1": "w1: decode → resize 224² → gray → numpy",
    "w2": "w2: thumbnail 256 → JPEG",
    "w3": "w3: size + pHash → duplicate groups",
    "w4": "w4: per-row crop + resize",
}
HIGHLIGHT = "#2a78d6"  # categorical slot 1
OTHER = "#a3a29c"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
    }
)


def load(paths: list[Path]) -> tuple[list[dict], dict]:
    cells, env = [], {}
    for p in paths:
        data = json.loads(p.read_text())
        env = env or data["environment"]
        cells += data["cells"]
    return cells, env


def pick(cells, **kw) -> dict[str, dict]:
    """Library -> cell for the cells matching every keyword."""
    return {c["library"]: c for c in cells if all(c.get(k) == v for k, v in kw.items())}


def _bars(ax, rows: dict[str, dict], value, label, order=LIBRARIES) -> None:
    libs = [lib for lib in order if lib in rows]
    ys = range(len(libs))[::-1]
    vmax = max(
        (value(rows[lib]) for lib in libs if rows[lib]["status"] == "ok"), default=1
    )
    for y, lib in zip(ys, libs, strict=True):
        c = rows[lib]
        if c["status"] != "ok":
            ax.text(
                0,
                y,
                f"  {c['status']}",
                va="center",
                ha="left",
                color=MUTED,
                fontsize=8,
            )
            continue
        v = value(c)
        color = HIGHLIGHT if lib.startswith("polars-cv") else OTHER
        if value is _mib:
            # Baseline (interpreter, imports, loaded inputs) light; what the
            # job added on top of it solid.
            base = max(0.0, c["baseline_pss_bytes"] / 2**20)
            tint = "#bcd5f2" if lib.startswith("polars-cv") else "#dad9d4"
            ax.barh(y, base, height=0.72, color=tint, edgecolor="white", linewidth=1)
            ax.barh(
                y,
                max(0.0, v - base),
                left=base,
                height=0.72,
                color=color,
                edgecolor="white",
                linewidth=1,
            )
        else:
            ax.barh(y, v, height=0.72, color=color, edgecolor="white", linewidth=1)
        if "q1_s" in c and value is _ips:
            lo, hi = c["n_images"] / c["q3_s"], c["n_images"] / c["q1_s"]
            ax.plot([lo, hi], [y, y], color=INK, linewidth=0.8)
        ax.text(
            v + vmax * 0.015, y, label(v), va="center", ha="left", color=INK, fontsize=8
        )
    ax.set_yticks(list(ys), libs)
    ax.set_xlim(0, vmax * 1.28)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _ips(c: dict) -> float:
    return c["images_per_s"]


def _mib(c: dict) -> float:
    return c["peak_pss_bytes"] / 2**20


def throughput_grid(cells, dataset, source, threads_list, out: Path) -> None:
    workloads = [w for w in WORKLOADS if any(c["workload"] == w for c in cells)]
    fig, axes = plt.subplots(
        len(workloads),
        len(threads_list),
        figsize=(5.4 * len(threads_list), 2.6 * len(workloads)),
        squeeze=False,
    )
    for i, w in enumerate(workloads):
        for j, t in enumerate(threads_list):
            ax = axes[i][j]
            rows = pick(cells, dataset=dataset, workload=w, source=source, threads=t)
            rows = {
                k: v
                for k, v in rows.items()
                if v["status"] != "n/a"
                or "single-process" not in v.get("reason", "")
                and "one process" not in v.get("reason", "")
            }
            _bars(ax, rows, _ips, lambda v: f"{v:,.0f}")
            ax.set_title(
                f"{TITLES[w]}  ·  {t} thread{'s' * (t > 1)}",
                loc="left",
                fontsize=9,
                color=INK,
            )
            if i == len(workloads) - 1:
                ax.set_xlabel("images / second (median of 5; line = IQR)")
    fig.suptitle(
        f"Throughput, {dataset}, {source} source",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def memory_grid(cells, dataset, source, threads, out: Path) -> None:
    workloads = [w for w in WORKLOADS if any(c["workload"] == w for c in cells)]
    fig, axes = plt.subplots(
        1, len(workloads), figsize=(4.0 * len(workloads), 3.2), squeeze=False
    )
    for ax, w in zip(axes[0], workloads, strict=True):
        rows = pick(cells, dataset=dataset, workload=w, source=source, threads=threads)
        rows = {
            k: v
            for k, v in rows.items()
            if v["status"] == "ok" or k not in ("pil", "opencv")
        }
        _bars(ax, rows, _mib, lambda v: f"{v:,.0f}")
        ax.set_title(TITLES[w], loc="left", fontsize=9, color=INK)
        ax.set_xlabel("peak PSS, MiB (light: before the job; solid: added)")
    fig.suptitle(
        f"Peak memory, {dataset}, {source} source, {threads} threads",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def summary(cells, dataset, source, threads, out: Path) -> None:
    """The README chart: every workload at all cores, file-path source."""
    fig, grid = plt.subplots(2, 2, figsize=(10.5, 6.2), squeeze=False)
    axes = [list(grid.flat)]
    for ax, w in zip(axes[0], WORKLOADS, strict=True):
        rows = pick(cells, dataset=dataset, workload=w, source=source, threads=threads)
        rows = {k: v for k, v in rows.items() if k not in ("pil", "opencv")}
        _bars(ax, rows, _ips, lambda v: f"{v:,.0f}")
        ax.set_title(TITLES[w], loc="left", fontsize=9, color=INK)
        ax.set_xlabel("images / second")
    fig.suptitle(
        f"COCO val2017 ({dataset[5:]}), {threads} threads, from file paths — higher is better",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _loop_or_pool_only(c: dict | None) -> bool:
    """An n/a that only says this thread count belongs to the sibling row."""
    return (
        c is not None
        and c["status"] == "n/a"
        and ("single-process" in c["reason"] or "one process" in c["reason"])
    )


def table(cells, dataset, workload, source) -> str:
    threads = sorted({c["threads"] for c in cells if c["dataset"] == dataset})
    head = (
        "| library | "
        + " | ".join(f"{t} thr img/s (IQR) | {t} thr peak MiB" for t in threads)
        + " |"
    )
    lines = [head, "|---|" + "---:|---:|" * len(threads)]
    for lib in LIBRARIES:
        found = [
            pick(
                cells, dataset=dataset, workload=workload, source=source, threads=t
            ).get(lib)
            for t in threads
        ]
        if all(c is None or _loop_or_pool_only(c) for c in found):
            continue
        cols = []
        for c in found:
            if c is None or _loop_or_pool_only(c):
                cols += ["—", "—"]
            elif c["status"] == "ok":
                lo, hi = c["n_images"] / c["q3_s"], c["n_images"] / c["q1_s"]
                cols += [
                    f"{c['images_per_s']:,.0f} ({lo:,.0f}–{hi:,.0f})",
                    f"{_mib(c):,.0f}",
                ]
            else:
                cols += [c["status"], "—"]
        lines.append(f"| {lib} | " + " | ".join(cols) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    p.add_argument("--png", action="store_true", help="also write PNG previews")
    p.add_argument(
        "--all-threads",
        action="store_true",
        help="also chart the 1-thread cells (default: all cores only)",
    )
    args = p.parse_args()
    if args.png:
        _orig = matplotlib.figure.Figure.savefig

        def _both(self, fname, *a, **k):
            _orig(self, fname, *a, **k)
            _orig(self, Path(fname).with_suffix(".png"), *a, dpi=70, **k)

        matplotlib.figure.Figure.savefig = _both
    cells, _ = load(args.results)
    if not args.all_threads:
        top = {
            d: max(c["threads"] for c in cells if c["dataset"] == d)
            for d in {c["dataset"] for c in cells}
        }
        cells = [c for c in cells if c["threads"] == top[c["dataset"]]]
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in sorted({c["dataset"] for c in cells}):
        dc = [c for c in cells if c["dataset"] == dataset]
        threads = sorted({c["threads"] for c in dc})
        nmax = max(threads)
        for source in sorted({c["source"] for c in dc}):
            throughput_grid(
                dc,
                dataset,
                source,
                threads,
                args.out / f"throughput-{dataset}-{source}.svg",
            )
            for w in WORKLOADS:
                if any(c["workload"] == w for c in dc):
                    print(f"\n#### {dataset}, {w}, {source} source\n")
                    print(table(dc, dataset, w, source))
        src = "path" if any(c["source"] == "path" for c in dc) else dc[0]["source"]
        memory_grid(dc, dataset, src, nmax, args.out / f"memory-{dataset}.svg")
        if dataset == "coco-5k":
            summary(dc, dataset, src, nmax, args.out / "summary.svg")
    print(f"\nwrote SVGs to {args.out}")


if __name__ == "__main__":
    main()
