"""The regression harness runs only what it is asked to, and refuses to lie.

``benchmarks/regression`` is how a change proves it made something faster. Two
things make that slow or wrong, and these pin both:

- **Running everything, every time.** ``--select scenario[:glob]`` runs a
  subset of cases, and ``--changed REF`` derives that subset from the files a
  change touches (``relevance.RULES``). A selector that matches nothing is an
  error, never an empty run that compares as "no regressions"; a changed source
  file no rule covers is an error, never a silently skipped benchmark.
- **Measuring a debug build.** A debug extension measures nothing useful, and
  the numbers it gives look plausible. The suite refuses one unless told it is
  a harness smoke test, and ``compare`` refuses such results.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from benchmarks.regression import relevance, selection
from benchmarks.regression.compare import main as compare_main
from benchmarks.regression.config import SuiteConfig
from benchmarks.regression.run_suite import main as run_suite_main
from benchmarks.regression.run_suite import run_suite

from tests.conftest import plugin_required

_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Selection: scenario[:glob], resolved against each scenario's own case list
# ---------------------------------------------------------------------------


def test_a_bare_scenario_selects_all_of_its_cases() -> None:
    sel = selection.parse("single_ops")
    assert sel.scenarios() == ("single_ops",)
    assert sel.cases("single_ops") is None  # None = every case


def test_a_glob_selects_the_matching_cases_by_their_result_name() -> None:
    sel = selection.parse("single_ops:rotate_*,pipelines:medium_pipeline")
    assert sel.cases("single_ops") == frozenset({"rotate_90", "rotate_45"})
    assert sel.cases("pipelines") == frozenset({"medium_pipeline"})


def test_e2e_cases_are_named_as_their_results_are() -> None:
    # compare prints `e2e_basic_preprocess`; the selector must take that name.
    sel = selection.parse("e2e:e2e_basic_*")
    assert sel.cases("e2e") == frozenset({"e2e_basic_preprocess"})


def test_a_glob_matching_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="matches no case"):
        selection.parse("single_ops:rotat_*")


def test_an_unknown_scenario_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        selection.parse("single_opz")


def test_a_scenario_without_a_case_list_refuses_a_glob() -> None:
    # zero_copy and remote run as a unit; a glob there would be ignored.
    with pytest.raises(ValueError, match="runs as a whole"):
        selection.parse("remote:remote_http_*")


def test_selections_merge_and_render_round_trip() -> None:
    sel = selection.parse("single_ops:invert,single_ops:blur,targeted")
    assert sel.cases("single_ops") == frozenset({"invert", "blur"})
    assert selection.parse(sel.render()) == sel


def test_a_whole_scenario_absorbs_its_globs() -> None:
    sel = selection.parse("single_ops:invert,single_ops")
    assert sel.cases("single_ops") is None


# ---------------------------------------------------------------------------
# Thread requirements: code a selection measures only on a parallel pool
# ---------------------------------------------------------------------------


def test_a_threads_marker_is_carried_by_the_selection() -> None:
    sel = selection.parse("pipelines,@threads=4")
    assert sel.min_threads == 4
    assert sel.scenarios() == ("pipelines",)
    assert selection.parse(sel.render()) == sel


def test_merged_selections_keep_the_larger_thread_requirement() -> None:
    sel = selection.parse("@threads=2,single_ops:invert") | selection.parse(
        "@threads=4,pipelines"
    )
    assert sel.min_threads == 4


def test_a_selection_without_a_marker_needs_one_thread() -> None:
    assert selection.parse("single_ops:invert").min_threads == 1


@pytest.mark.parametrize("bad", ["@threads=0", "@threads=x", "@threads", "@cores=4"])
def test_a_malformed_marker_is_an_error(bad: str) -> None:
    with pytest.raises(ValueError, match="marker"):
        selection.parse(f"pipelines,{bad}")


def test_parallel_only_cases_carry_their_own_requirement() -> None:
    # The split_ cases time how a call spreads over the pool: on one thread
    # there is nothing to spread, and their numbers would measure nothing.
    assert selection.parse("targeted:split_*").min_threads >= 2
    assert selection.parse("targeted").min_threads >= 2
    assert selection.parse("targeted:geom_*").min_threads == 1


# ---------------------------------------------------------------------------
# Relevance: changed files -> selection
# ---------------------------------------------------------------------------


def test_a_geometry_change_selects_the_geometry_cases_only() -> None:
    sel, unbenchmarked = relevance.select_for(["polars-cv/src/geom_columns.rs"])
    assert sel.scenarios() == ("targeted",)
    cases = sel.cases("targeted")
    assert cases is not None
    assert cases and all(c.startswith("geom_") for c in cases)
    assert unbenchmarked == []


def test_a_kernel_change_selects_the_op_scenarios() -> None:
    sel, _ = relevance.select_for(["view-buffer/src/ops/filter.rs"])
    assert {"single_ops", "pipelines"} <= set(sel.scenarios())


def test_a_dependency_change_selects_everything() -> None:
    sel, _ = relevance.select_for(["Cargo.lock"])
    assert set(sel.scenarios()) == set(selection.SCENARIOS)
    assert all(sel.cases(s) is None for s in sel.scenarios())


def test_non_code_changes_select_nothing() -> None:
    sel, unbenchmarked = relevance.select_for(
        ["CHANGELOG.md", "polars-cv/tests/test_x.py", "polars-cv/src/AGENTS.md"]
    )
    assert sel.scenarios() == ()
    assert unbenchmarked == []


def test_code_no_scenario_measures_is_reported_not_hidden() -> None:
    sel, unbenchmarked = relevance.select_for(["polars-cv/src/image_metadata.rs"])
    assert sel.scenarios() == ()
    assert unbenchmarked == ["polars-cv/src/image_metadata.rs"]


def test_a_code_file_no_rule_covers_is_an_error() -> None:
    with pytest.raises(ValueError, match="no relevance rule"):
        relevance.select_for(["polars-cv/src/brand_new_module.rs"])


def test_every_tracked_code_file_is_covered_by_a_rule() -> None:
    files = subprocess.run(
        ["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    code = [f for f in files if relevance.is_code(f)]
    # The classifier must see the crates, or this checks nothing.
    assert any(f.startswith("view-buffer/src/") for f in code)
    assert any(f.startswith("polars-cv/python/polars_cv/") for f in code)
    uncovered = [f for f in code if relevance.rule_for(f) is None]
    assert uncovered == [], f"add a relevance.RULES entry for: {uncovered}"


def test_every_rule_matches_a_tracked_file() -> None:
    # A rule whose file moved matches nothing, and the moved file falls to
    # whatever broader rule covers it, selecting the wrong cases silently.
    files = subprocess.run(
        ["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    dead = [
        pattern
        for pattern, _spec in relevance.RULES
        if not any(relevance._regex(pattern).fullmatch(f) for f in files)
    ]
    assert dead == [], f"relevance.RULES entries matching no tracked file: {dead}"


def test_a_row_split_change_selects_every_split_path_on_a_parallel_pool() -> None:
    # Every call's rows and every tensor sink's fill phase run through the
    # splitter, and it does nothing on one thread: PR #124 regressed the list
    # sink and streaming sobel_x at 4 threads, which a 1-thread run of the
    # cases it selected could not see.
    sel, _ = relevance.select_for(["polars-cv/src/row_split.rs"])
    assert sel.min_threads >= 2
    for scenario in ("single_ops", "pipelines", "e2e"):
        assert sel.cases(scenario) is None
    targeted = sel.cases("targeted")
    assert targeted is not None
    for prefix in ("sink_", "split_", "geom_"):
        assert any(c.startswith(prefix) for c in targeted), prefix


def test_every_rule_selector_resolves() -> None:
    # A case renamed out from under a rule would otherwise select nothing.
    for _pattern, spec in relevance.RULES:
        if spec:
            selection.parse(spec)  # raises on a dead selector


def test_changed_files_include_uncommitted_work(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    (tmp_path / "a.rs").write_text("")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "a"], cwd=tmp_path, check=True)
    (tmp_path / "b.rs").write_text("")  # untracked
    (tmp_path / "a.rs").write_text("x")  # modified
    subprocess.run(["git", "add", "b.rs"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "b"], cwd=tmp_path, check=True)
    (tmp_path / "c.rs").write_text("")
    assert relevance.changed_files("HEAD~1", cwd=tmp_path) == ["a.rs", "b.rs", "c.rs"]


# ---------------------------------------------------------------------------
# The suite runs exactly the selection, and refuses a debug build
# ---------------------------------------------------------------------------


def _tiny(sel: str) -> SuiteConfig:
    return SuiteConfig(
        image_counts=[4],
        image_sizes=[(32, 32)],
        warmup_iterations=0,
        benchmark_iterations=1,
        suite_repeats=1,
        selection=selection.parse(sel),
    )


@plugin_required
def test_the_suite_runs_exactly_the_selected_cases() -> None:
    cfg = _tiny("single_ops:invert,pipelines:light_pipeline,targeted:geom_point_*")
    results = run_suite(cfg, allow_debug_build=True)
    ops = {r.operation for r in results}
    assert ops == {
        "invert",
        "light_pipeline",
        "geom_point_translate",
        "geom_point_distance",
        "geom_point_distance_to_contour",
    }


@plugin_required
def test_every_targeted_case_runs() -> None:
    results = run_suite(_tiny("targeted"), allow_debug_build=True)
    assert {r.operation for r in results} == set(selection.case_names("targeted"))
    assert all(r.throughput_images_per_second > 0 for r in results)


@plugin_required
def test_the_suite_refuses_a_debug_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.regression import run_suite as module

    # Pinned to "debug" so this holds whatever profile the suite runs against.
    monkeypatch.setattr(module, "is_debug_build", lambda: True)
    with pytest.raises(SystemExit, match="--profile benchmark"):
        run_suite_main(
            ["--out", str(tmp_path / "r.json"), "--select", "single_ops:invert"]
        )


def test_a_thread_requirement_is_met_or_refused() -> None:
    from benchmarks.regression.run_suite import check_threads

    check_threads(4, selection.parse("pipelines,@threads=4"))
    check_threads(8, selection.parse("pipelines,@threads=4"))
    check_threads(1, selection.parse("pipelines"))
    with pytest.raises(SystemExit, match="needs --threads >= 4"):
        check_threads(2, selection.parse("pipelines,@threads=4"))


def test_the_cli_refuses_fewer_threads_than_the_selection_needs(
    tmp_path: Path,
) -> None:
    # Refused before anything is pinned or run.
    with pytest.raises(SystemExit, match="needs --threads >= 4"):
        run_suite_main(
            [
                "--out",
                str(tmp_path / "r.json"),
                "--select",
                "pipelines:light_pipeline,@threads=4",
                "--threads",
                "1",
            ]
        )


def test_the_suite_refuses_a_pool_smaller_than_the_selection_needs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The backstop for a programmatic caller: the pool the cases will actually
    # run on, not an argument, decides.
    from benchmarks.regression import run_suite as module

    monkeypatch.setattr(module, "pool_threads", lambda: 1)
    with pytest.raises(SystemExit, match="1-thread pool"):
        run_suite(_tiny("targeted:split_*"), allow_debug_build=True)


@plugin_required
def test_compare_refuses_debug_build_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.regression import run_suite as module

    import polars_cv._lib as lib

    monkeypatch.setattr(module, "is_debug_build", lambda: True)
    out = tmp_path / "r.json"
    args = ["--out", str(out), "--select", "single_ops:invert", "--allow-debug-build"]
    args += ["--counts", "4", "--sizes", "32", "--repeats", "1", "--warmup", "0"]
    args += ["--iterations", "1", "--quiet"]
    assert run_suite_main(args) == 0
    meta = json.loads(out.with_suffix(".json.meta.json").read_text())
    assert meta["debug_build"] is True
    assert meta["build_profile"] == lib.__build_profile__
    assert meta["selection"] == "single_ops:invert"
    with pytest.raises(SystemExit, match="debug build"):
        compare_main([str(out), str(out)])


def test_the_relevance_cli_prints_a_reusable_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(
        [*git, "commit", "-q", "--allow-empty", "-m", "a"], cwd=tmp_path, check=True
    )
    (tmp_path / "polars-cv" / "src").mkdir(parents=True)
    (tmp_path / "polars-cv" / "src" / "contour.rs").write_text("")
    monkeypatch.chdir(tmp_path)
    assert relevance.main(["main"]) == 0
    printed = capsys.readouterr().out.strip()
    assert selection.parse(printed) == selection.parse("targeted:geom_*")


def test_a_deleted_or_renamed_file_is_not_a_changed_file(tmp_path: Path) -> None:
    # Its old path needs no rule: what replaced it (or its callers) is in the diff.
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    (tmp_path / "old.rs").write_text("fn a() {}\n" * 20)
    (tmp_path / "gone.rs").write_text("")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "a"], cwd=tmp_path, check=True)
    subprocess.run(["git", "mv", "old.rs", "new.rs"], cwd=tmp_path, check=True)
    subprocess.run(["git", "rm", "-q", "gone.rs"], cwd=tmp_path, check=True)
    subprocess.run([*git, "commit", "-qm", "b"], cwd=tmp_path, check=True)
    assert relevance.changed_files("HEAD~1", cwd=tmp_path) == ["new.rs"]


@plugin_required
def test_the_extension_names_the_profile_it_was_built_with() -> None:
    import polars_cv._lib as lib

    # Cargo's profile directory: `debug` for `maturin develop`, else the
    # `--profile`/`--release` name. Only the dev profile keeps debug assertions.
    assert lib.__build_profile__ in {"debug", "release", "benchmark"}
    assert lib.__debug_assertions__ is (lib.__build_profile__ == "debug")


def _results(
    tmp_path: Path, name: str, profile: str | None, *, threads: int = 1
) -> Path:
    row = {
        "framework": "polars-cv-eager",
        "operation": "invert",
        "image_count": 4,
        "image_size": [32, 32],
        "total_time_seconds": 1.0,
        "throughput_images_per_second": 4.0,
        "latency_ms_per_image": 250.0,
        "peak_memory_mb": 1.0,
    }
    path = tmp_path / name
    path.write_text(json.dumps([row]))
    if profile is not None:
        meta = {"debug_build": False, "build_profile": profile, "num_threads": threads}
        Path(f"{path}.meta.json").write_text(json.dumps(meta))
    return path


def test_compare_refuses_results_from_different_build_profiles(tmp_path: Path) -> None:
    # Thin- and fat-LTO builds differ by several percent per case: comparing
    # across them reports the profile, not the change.
    base = _results(tmp_path, "base.json", "release")
    head = _results(tmp_path, "head.json", "benchmark")
    with pytest.raises(SystemExit, match="build profile"):
        compare_main([str(base), str(head)])


def test_compare_refuses_results_run_on_different_thread_counts(
    tmp_path: Path,
) -> None:
    # A 1-thread base against a 4-thread head reports the pool, not the change.
    base = _results(tmp_path, "base.json", "benchmark", threads=1)
    head = _results(tmp_path, "head.json", "benchmark", threads=4)
    with pytest.raises(SystemExit, match="thread"):
        compare_main([str(base), str(head)])


def test_compare_accepts_results_from_the_same_profile(tmp_path: Path) -> None:
    base = _results(tmp_path, "base.json", "benchmark")
    head = _results(tmp_path, "head.json", "benchmark")
    assert compare_main([str(base), str(head)]) == 0
