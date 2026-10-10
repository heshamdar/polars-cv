"""The comparison report summarises a run without inventing or dropping cases.

``benchmarks.report`` turns the saved results of ``run_benchmarks`` into the
data the HTML page renders. Every number the page shows is computed here, so
these tests pin the aggregation: a speedup is a geometric mean over the cases
both frameworks ran, an op a framework did not run is "unsupported" rather
than slow, and the embedded data survives the page it is embedded in.
"""

from __future__ import annotations

import json
import math
import re

import pytest
from benchmarks.report import build_report_data, merge_runs, render_html


def _result(
    framework: str,
    operation: str,
    throughput: float,
    *,
    scenario: str = "single_ops",
    size: int = 256,
    count: int = 100,
) -> dict[str, object]:
    return {
        "framework": framework,
        "operation": operation,
        "image_count": count,
        "image_size": [size, size],
        "total_time_seconds": count / throughput,
        "throughput_images_per_second": throughput,
        "latency_ms_per_image": 1000 / throughput,
        "peak_memory_mb": 10.0,
        "gpu_mode": None,
        "scenario": scenario,
    }


def _run(results: list[dict[str, object]], frameworks: list[str]) -> dict:
    return {
        "meta": {"frameworks": frameworks, "sizes": [[256, 256]], "counts": [100]},
        "results": results,
    }


FRAMEWORKS = ["opencv", "polars-cv-eager", "pyvips"]


def test_speedup_is_the_geometric_mean_over_shared_cases() -> None:
    run = _run(
        [
            _result("polars-cv-eager", "resize", 100),
            _result("polars-cv-eager", "blur", 100),
            _result("polars-cv-eager", "canny", 100),
            _result("opencv", "resize", 50),  # polars-cv 2x faster
            _result("opencv", "blur", 200),  # polars-cv 2x slower
            _result("pyvips", "resize", 25),  # 4x; pyvips has no blur or canny
        ],
        FRAMEWORKS,
    )
    data = build_report_data(run)
    (config,) = data["configs"]
    summary = {s["framework"]: s for s in data["speedups"][config["key"]]}
    assert summary["opencv"]["geomean"] == pytest.approx(1.0)
    assert summary["opencv"]["cases"] == 2
    assert summary["opencv"]["wins"] == 1
    assert summary["pyvips"]["geomean"] == pytest.approx(4.0)
    assert summary["pyvips"]["cases"] == 1
    assert "polars-cv-eager" not in summary  # the baseline is not compared to itself


def test_an_op_a_framework_did_not_run_is_unsupported_not_slow() -> None:
    run = _run(
        [
            _result("polars-cv-eager", "canny", 100),
            _result("opencv", "canny", 80),
        ],
        FRAMEWORKS,
    )
    data = build_report_data(run)
    (config,) = data["configs"]
    row = next(
        r for r in data["tables"][config["key"]]["single_ops"] if r["op"] == "canny"
    )
    assert row["cells"]["pyvips"] is None
    assert row["cells"]["opencv"]["ratio"] == pytest.approx(100 / 80)
    assert row["cells"]["polars-cv-eager"]["ratio"] == pytest.approx(1.0)
    assert row["fastest"] == "polars-cv-eager"


def test_frameworks_are_listed_in_a_fixed_order_with_polars_cv_first() -> None:
    run = _run(
        [_result(f, "resize", 10) for f in ["pyvips", "opencv", "polars-cv-eager"]],
        FRAMEWORKS,
    )
    assert build_report_data(run)["frameworks"] == [
        "polars-cv-eager",
        "opencv",
        "pyvips",
    ]


def test_configs_are_one_per_size_and_count() -> None:
    run = _run(
        [
            _result("polars-cv-eager", "resize", 10, size=s, count=c)
            for s in (512, 256)
            for c in (10, 100)
        ],
        ["polars-cv-eager"],
    )
    keys = [(c["size"], c["count"]) for c in build_report_data(run)["configs"]]
    assert keys == [(256, 10), (256, 100), (512, 10), (512, 100)]


def test_a_missing_baseline_leaves_ratios_undefined() -> None:
    run = _run([_result("opencv", "resize", 10)], FRAMEWORKS)
    data = build_report_data(run)
    (config,) = data["configs"]
    (row,) = data["tables"][config["key"]]["single_ops"]
    assert row["cells"]["opencv"]["ratio"] is None
    assert data["speedups"][config["key"]] == []


def test_the_page_embeds_the_data_it_was_given() -> None:
    run = _run(
        [
            _result("polars-cv-eager", "resize</script><b>", 100),
            _result("opencv", "resize</script><b>", 50),
        ],
        FRAMEWORKS,
    )
    data = build_report_data(run)
    html = render_html(data)
    assert re.search(r"<title>[^<]+</title>", html)
    match = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match, "the report data block is missing or was cut short"
    assert json.loads(match.group(1)) == json.loads(json.dumps(data))


def test_speedup_ignores_non_positive_throughput() -> None:
    """A zero throughput would make the geometric mean log(0)."""
    run = _run(
        [
            _result("polars-cv-eager", "resize", 100),
            _result("opencv", "resize", 50),
        ],
        FRAMEWORKS,
    )
    run["results"].append(
        {**_result("opencv", "blur", 1), "throughput_images_per_second": 0.0}
    )
    run["results"].append(_result("polars-cv-eager", "blur", 100))
    data = build_report_data(run)
    (config,) = data["configs"]
    (opencv,) = [
        s for s in data["speedups"][config["key"]] if s["framework"] == "opencv"
    ]
    assert opencv["cases"] == 1
    assert math.isclose(opencv["geomean"], 2.0)


def test_runs_merge_into_one_report_with_every_configuration() -> None:
    small = _run(
        [_result("polars-cv-eager", "resize", 100, size=256, count=200)], FRAMEWORKS
    )
    large = _run([_result("opencv", "resize", 5, size=1024, count=50)], ["opencv"])
    merged = merge_runs([small, large])
    assert len(merged["results"]) == 2
    keys = [(c["size"], c["count"]) for c in build_report_data(merged)["configs"]]
    assert keys == [(256, 200), (1024, 50)]


def test_two_runs_of_the_same_configuration_do_not_merge() -> None:
    a = _run([_result("opencv", "resize", 5)], FRAMEWORKS)
    with pytest.raises(ValueError, match="both measured"):
        merge_runs([a, a])
