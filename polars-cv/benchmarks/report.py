"""
The comparison report: a saved benchmark run rendered as one HTML page.

``run_benchmarks --save-json`` writes a run (``{"meta": ..., "results": [...]}``);
:func:`build_report_data` turns it into everything the page shows, and
:func:`render_html` embeds that in ``report_template.html``. All arithmetic
lives here, where ``tests/test_benchmark_report.py`` can check it; the page's
script only draws.

Usage:
    python -m benchmarks.report run.json [more.json ...] -o report.html

Several runs (say, small images at a large count and large images at a small
one) merge into one report; each keeps its own configurations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

BASELINE = "polars-cv-eager"

#: Display order; frameworks not listed follow alphabetically.
FRAMEWORK_ORDER = [
    "polars-cv-eager",
    "polars-cv-streaming",
    "opencv",
    "pillow",
    "pyvips",
    "torchvision-cpu",
    "torchvision-cuda",
    "torchvision-mps",
]

SCENARIOS = [
    {
        "key": "single_ops",
        "title": "Single operations",
        "blurb": (
            "One operation on images already decoded into each library's own "
            "format, so decoding is not timed."
        ),
    },
    {
        "key": "pipelines",
        "title": "Pipelines",
        "blurb": (
            "Chains of operations on pre-decoded images: what fusion and "
            "avoiding intermediate copies are worth."
        ),
    },
    {
        "key": "workflows",
        "title": "Multi-branch workflows",
        "blurb": (
            "Graphs that branch and rejoin, from encoded PNG bytes (decode "
            "included), each written in its library's own idiom."
        ),
    },
    {
        "key": "e2e",
        "title": "End to end",
        "blurb": "Files on disk to processed arrays in memory.",
    },
]

_LIBRARIES = {
    "polars": "polars",
    "polars-cv": "polars-cv",
    "numpy": "numpy",
    "opencv": "opencv-python",
    "pillow": "pillow",
    "pyvips": "pyvips",
    "torch": "torch",
    "torchvision": "torchvision",
}


def _framework_label(result: dict[str, Any]) -> str:
    mode = result.get("gpu_mode")
    return f"{result['framework']} ({mode})" if mode else str(result["framework"])


def _ordered(frameworks: set[str]) -> list[str]:
    known = [f for f in FRAMEWORK_ORDER if f in frameworks]
    return known + sorted(frameworks - set(known))


def _descriptions() -> dict[str, str]:
    """Operation name -> description, from the scenario definitions."""
    from benchmarks.scenarios.e2e_workflow import get_e2e_workflows
    from benchmarks.scenarios.pipelines import get_pipeline_benchmarks
    from benchmarks.scenarios.single_ops import get_single_op_benchmarks
    from benchmarks.scenarios.workflows import WORKFLOWS

    out = {b.name: b.description for b in get_single_op_benchmarks()}
    out |= {b.name: b.description for b in get_pipeline_benchmarks()}
    out |= {f"e2e_{w.name}": w.description for w in get_e2e_workflows()}
    out |= {f"workflow_{w.name}": w.description for w in WORKFLOWS.values()}
    return out


def _summary(framework: str, pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """polars-cv's speedup over ``framework`` from (baseline, other)
    throughputs: the geometric mean of their ratios, and the spread."""
    ratios = [b / o for b, o in pairs]
    return {
        "framework": framework,
        "geomean": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
        "cases": len(ratios),
        "wins": sum(r > 1 for r in ratios),
        "min": min(ratios),
        "max": max(ratios),
    }


def build_report_data(run: dict[str, Any]) -> dict[str, Any]:
    """
    Everything the report page shows, computed from a saved run.

    For each configuration (image size x count), and each scenario, one row
    per operation with every framework's cell: its throughput, latency and
    peak memory, and ``ratio``, the baseline's throughput over its own (above
    1: polars-cv is faster), or ``None`` for a framework that did not run the
    op. ``speedups`` summarises each framework against the baseline over the
    cases both ran with a positive throughput.

    Args:
        run: ``{"meta": ..., "results": [...]}`` as ``run_benchmarks
            --save-json`` writes it.

    Returns:
        The report data, JSON-serialisable.
    """
    meta = run.get("meta", {})
    results = run["results"]
    frameworks = _ordered(
        set(meta.get("frameworks", [])) | {_framework_label(r) for r in results}
    )

    # (size, count) -> scenario -> op -> framework -> result
    grid: dict[tuple[int, int], dict[str, dict[str, dict[str, dict]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    op_order: dict[str, list[str]] = defaultdict(list)
    for r in results:
        size = int(r["image_size"][0])
        key = (size, int(r["image_count"]))
        scenario, op = r.get("scenario", "single_ops"), r["operation"]
        grid[key][scenario].setdefault(op, {})[_framework_label(r)] = r
        if op not in op_order[scenario]:
            op_order[scenario].append(op)

    configs = [
        {"key": f"{size}px · {count} images", "size": size, "count": count}
        for size, count in sorted(grid)
    ]
    tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
    speedups: dict[str, list[dict[str, Any]]] = {}
    scenario_speedups: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for config in configs:
        by_scenario = grid[(config["size"], config["count"])]
        tables[config["key"]] = {}
        pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
        scenario_speedups[config["key"]] = {}
        for scenario in [s["key"] for s in SCENARIOS] + sorted(
            set(by_scenario) - {s["key"] for s in SCENARIOS}
        ):
            if scenario not in by_scenario:
                continue
            rows = []
            scenario_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for op in op_order[scenario]:
                ran = by_scenario[scenario].get(op)
                if not ran:
                    continue
                base = ran.get(BASELINE)
                base_tp = base["throughput_images_per_second"] if base else 0.0
                cells: dict[str, dict[str, Any] | None] = {}
                for fw in frameworks:
                    r = ran.get(fw)
                    if r is None:
                        cells[fw] = None
                        continue
                    tp = r["throughput_images_per_second"]
                    ok = tp > 0 and base_tp > 0
                    cells[fw] = {
                        "throughput": tp,
                        "latency_ms": r["latency_ms_per_image"],
                        "memory_mb": r["peak_memory_mb"],
                        "ratio": base_tp / tp if ok else None,
                    }
                    if ok and fw != BASELINE:
                        pairs[fw].append((base_tp, tp))
                        scenario_pairs[fw].append((base_tp, tp))
                fastest = max(
                    (fw for fw in frameworks if cells[fw]),
                    key=lambda fw: cells[fw]["throughput"],  # type: ignore[index]
                )
                rows.append({"op": op, "cells": cells, "fastest": fastest})
            tables[config["key"]][scenario] = rows
            scenario_speedups[config["key"]][scenario] = [
                _summary(fw, scenario_pairs[fw])
                for fw in frameworks
                if scenario_pairs[fw]
            ]
        speedups[config["key"]] = [
            _summary(fw, pairs[fw]) for fw in frameworks if pairs[fw]
        ]

    present = {s for by in grid.values() for s in by}
    return {
        "meta": meta,
        "baseline": BASELINE,
        "frameworks": frameworks,
        "scenarios": [s for s in SCENARIOS if s["key"] in present]
        + [
            {"key": s, "title": s, "blurb": ""}
            for s in sorted(present - {s["key"] for s in SCENARIOS})
        ],
        "descriptions": _descriptions(),
        "configs": configs,
        "tables": tables,
        "speedups": speedups,
        "scenario_speedups": scenario_speedups,
    }


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def collect_run_metadata(
    *,
    counts: list[int],
    sizes: list[tuple[int, int]],
    warmup: int,
    iterations: int,
    frameworks: list[str],
) -> dict[str, Any]:
    """What the run measured, on what, with which library versions."""
    versions = {name: _version(dist) for name, dist in _LIBRARIES.items()}
    try:
        import pyvips

        versions["libvips"] = ".".join(str(pyvips.version(i)) for i in range(3))
    except (ImportError, OSError):
        versions["libvips"] = None
    status = _git("status", "--porcelain")
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(status),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "versions": {k: v for k, v in versions.items() if v},
        "counts": counts,
        "sizes": [list(s) for s in sizes],
        "warmup": warmup,
        "iterations": iterations,
        "frameworks": frameworks,
    }


_TEMPLATE = Path(__file__).with_name("report_template.html")


def render_html(data: dict[str, Any], *, standalone: bool = True) -> str:
    """
    The report page for :func:`build_report_data`'s output.

    Args:
        data: Report data.
        standalone: Wrap in a full HTML document (for a file opened locally);
            ``False`` gives the bare page for hosts that supply the document.

    Returns:
        HTML text.
    """
    # Every "<" escaped, so no value can close the script element early.
    payload = json.dumps(data).replace("<", "\\u003c")
    page = _TEMPLATE.read_text().replace("__REPORT_DATA__", payload)
    if not standalone:
        return page
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"</head>\n<body>\n{page}\n</body>\n</html>\n"
    )


def merge_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """One run from several: their results together, and the first run's
    metadata with every run's sizes, counts and frameworks.

    Raises:
        ValueError: If two runs measured the same configuration, which would
            put two results in one cell.
    """
    seen: dict[tuple[int, int], int] = {}
    for i, run in enumerate(runs):
        for size, count in {
            (int(r["image_size"][0]), int(r["image_count"])) for r in run["results"]
        }:
            if seen.setdefault((size, count), i) != i:
                first = seen[(size, count)]
                msg = f"runs {first} and {i} both measured {size}px x {count}"
                raise ValueError(msg)
    meta = dict(runs[0].get("meta", {}))
    for key in ("sizes", "counts", "frameworks"):
        merged: list[Any] = []
        for run in runs:
            for v in run.get("meta", {}).get(key, []):
                if v not in merged:
                    merged.append(v)
        meta[key] = merged
    return {"meta": meta, "results": [r for run in runs for r in run["results"]]}


def main(argv: list[str] | None = None) -> int:
    """Render a saved run as an HTML report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "runs", nargs="+", help="JSON written by run_benchmarks --save-json"
    )
    parser.add_argument("-o", "--output", required=True, help="HTML file to write")
    parser.add_argument(
        "--fragment",
        action="store_true",
        help="Write the page without the document wrapper",
    )
    args = parser.parse_args(argv)
    run = merge_runs([json.loads(Path(p).read_text()) for p in args.runs])
    data = build_report_data(run)
    Path(args.output).write_text(render_html(data, standalone=not args.fragment))
    print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
