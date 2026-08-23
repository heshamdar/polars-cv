"""
Turn the raw benchmark JSON in this directory into the tables in `README.md`.

Run it after `run_benchmarks.py` has produced `single_ops.json`,
`pipelines.json` and `e2e.json`::

    uv run --no-sync python benchmarks/reports/2026-08-23-daft-comparison/analyze.py > tables.md

Ratios are always *throughput* ratios (higher = the first framework is
faster), and the geometric mean is used to aggregate them so that a 4x win and
a 4x loss cancel instead of averaging to 2.1x.
"""

from __future__ import annotations

import json
import math
import pathlib
from collections import defaultdict
from typing import Any

HERE = pathlib.Path(__file__).parent

#: Column order for every table. Names are shortened for width.
FRAMEWORKS: list[str] = [
    "polars-cv-eager",
    "polars-cv-streaming",
    "daft",
    "daft-udf",
    "opencv",
    "pillow",
]

SHORT = {
    "polars-cv-eager": "pcv-eager",
    "polars-cv-streaming": "pcv-stream",
    "daft": "daft",
    "daft-udf": "daft-udf",
    "opencv": "opencv",
    "pillow": "pillow",
}

#: The three operations Daft can run with its own image expressions. These are
#: the only cells where "engine vs engine" means anything.
NATIVE_OVERLAP = ("resize", "grayscale", "crop_center")


def load(name: str) -> list[dict[str, Any]]:
    """
    Load one raw results file.

    Args:
        name: File stem, e.g. ``"single_ops"``.

    Returns:
        The list of result records, or an empty list if the file is absent.
    """
    path = HERE / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def index(records: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, float]]:
    """
    Index results by ``(operation, image size)`` then framework.

    Args:
        records: Raw benchmark records.

    Returns:
        Nested mapping to throughput in images/second.
    """
    table: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for rec in records:
        key = (rec["operation"], rec["image_size"][0])
        table[key][rec["framework"]] = rec["throughput_images_per_second"]
    return table


def geomean(values: list[float]) -> float:
    """
    Geometric mean of a list of ratios.

    Args:
        values: Positive ratios.

    Returns:
        The geometric mean, or ``nan`` when there is nothing to average.
    """
    clean = [v for v in values if v > 0]
    if not clean:
        return math.nan
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def throughput_table(title: str, records: list[dict[str, Any]]) -> str:
    """
    Render a throughput table (images/second) for one scenario.

    Args:
        title: Section heading.
        records: Raw benchmark records.

    Returns:
        A markdown section.
    """
    table = index(records)
    if not table:
        return f"### {title}\n\n_No results._\n"

    header = "| op | size | " + " | ".join(SHORT[f] for f in FRAMEWORKS) + " |"
    rule = "|---|---:|" + "---:|" * len(FRAMEWORKS)
    lines = [
        f"### {title}",
        "",
        "Throughput, images/second (higher is better).",
        "",
        header,
        rule,
    ]

    for op, size in sorted(table, key=lambda k: (k[1], k[0])):
        cells = []
        for framework in FRAMEWORKS:
            value = table[(op, size)].get(framework)
            cells.append(f"{value:,.0f}" if value else "—")
        lines.append(f"| {op} | {size} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def ratio_table(records: list[dict[str, Any]]) -> str:
    """
    Render the head-to-head ratios on the operations Daft runs natively.

    Args:
        records: Single-op benchmark records.

    Returns:
        A markdown section.
    """
    table = index(records)
    lines = [
        "### Head-to-head on Daft's native operations",
        "",
        "Throughput ratios on the three single ops Daft implements with its own",
        "expressions — the only cells where both engines run their own kernels on",
        "comparable work.",
        "",
        "| op | size | pcv-eager ÷ daft | pcv-stream ÷ daft | daft ÷ opencv |",
        "|---|---:|---:|---:|---:|",
    ]
    collected: dict[str, list[float]] = defaultdict(list)
    for op in NATIVE_OVERLAP:
        for size in sorted({k[1] for k in table}):
            cell = table.get((op, size))
            if not cell or "daft" not in cell:
                continue
            daft = cell["daft"]
            eager = cell.get("polars-cv-eager", math.nan) / daft
            stream = cell.get("polars-cv-streaming", math.nan) / daft
            ocv = daft / cell.get("opencv", math.nan)
            collected["eager"].append(eager)
            collected["stream"].append(stream)
            collected["ocv"].append(ocv)
            lines.append(
                f"| {op} | {size} | {eager:.2f}x | {stream:.2f}x | {ocv:.2f}x |"
            )
    lines.append(
        f"| **geomean** | | **{geomean(collected['eager']):.2f}x** | "
        f"**{geomean(collected['stream']):.2f}x** | "
        f"**{geomean(collected['ocv']):.2f}x** |"
    )
    return "\n".join(lines) + "\n"


def udf_overhead_table(records: list[dict[str, Any]]) -> str:
    """
    Compare `daft-udf` against the OpenCV kernels it calls.

    Both run the same OpenCV code, so the ratio is what Daft's batch-UDF
    machinery costs (or saves, if its parallelism outweighs the marshalling).

    Args:
        records: Single-op benchmark records.

    Returns:
        A markdown section.
    """
    table = index(records)
    lines = [
        "### What Daft's batch-UDF path costs",
        "",
        "`daft-udf` calls the very same OpenCV kernels as the `opencv` adapter, so",
        "this ratio isolates Daft's UDF machinery against a plain single-threaded",
        "Python loop. Below 1.00x, the dataframe engine is losing on ops it has to",
        "hand back to Python.",
        "",
        "| op | size | daft-udf ÷ opencv | pcv-stream ÷ opencv |",
        "|---|---:|---:|---:|",
    ]
    udf_ratios: list[float] = []
    pcv_ratios: list[float] = []
    for op, size in sorted(table, key=lambda k: (k[1], k[0])):
        cell = table[(op, size)]
        if "daft-udf" not in cell or "opencv" not in cell:
            continue
        udf = cell["daft-udf"] / cell["opencv"]
        udf_ratios.append(udf)
        pcv_text = "—"
        if "polars-cv-streaming" in cell:
            pcv = cell["polars-cv-streaming"] / cell["opencv"]
            pcv_ratios.append(pcv)
            pcv_text = f"{pcv:.2f}x"
        lines.append(f"| {op} | {size} | {udf:.2f}x | {pcv_text} |")
    lines.append(
        f"| **geomean** | | **{geomean(udf_ratios):.2f}x** | "
        f"**{geomean(pcv_ratios):.2f}x** |"
    )
    return "\n".join(lines) + "\n"


def coverage_note(records: list[dict[str, Any]]) -> str:
    """
    Report how many benchmark cells each framework actually produced.

    Args:
        records: Single-op benchmark records.

    Returns:
        A markdown section.
    """
    counts: dict[str, int] = defaultdict(int)
    for rec in records:
        counts[rec["framework"]] += 1
    total = max(counts.values()) if counts else 0
    lines = [
        "### Benchmark coverage",
        "",
        f"Cells completed out of {total} (20 single ops x 2 image sizes).",
        "",
        "| framework | cells | coverage |",
        "|---|---:|---:|",
    ]
    for framework in FRAMEWORKS:
        got = counts.get(framework, 0)
        pct = (got / total * 100) if total else 0.0
        lines.append(f"| {SHORT[framework]} | {got} | {pct:.0f}% |")
    return "\n".join(lines) + "\n"


def main() -> int:
    """
    Print every table.

    Returns:
        Process exit code.
    """
    single = load("single_ops")
    print("# Daft vs polars-cv — benchmark tables\n")
    print(coverage_note(single))
    print(ratio_table(single))
    print(udf_overhead_table(single))
    print(throughput_table("Single operations", single))
    print(throughput_table("Multi-operation pipelines", load("pipelines")))
    print(throughput_table("End-to-end file workflows", load("e2e")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
