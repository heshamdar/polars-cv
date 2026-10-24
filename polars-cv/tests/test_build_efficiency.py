"""Guards for the development build configuration.

These pin the "one canonical build" invariants that keep the dev loop fast and
the container's disk from filling. Each exists because the opposite has already
cost real time here: a `uv sync` that built the whole polars stack at release
LTO only to have `maturin develop` overwrite it, and a SessionStart hook that
ran that same build *synchronously*, so the session appeared to hang after
initialization (see `git log` for the "SessionStart hook hang" fix).

All are ``structural``: they read repository files, need no compiled extension,
and so run in the pre-commit lane alongside the other shape guards.

These are source scans — the weakest guard kind (`CLAUDE.md`: prefer compiler
exhaustiveness > runtime assertion > source scanning). They are used here only
because the properties live in hand-written CI YAML, a shell hook and cargo
config, which the first two cannot express. The CI scan reuses the
fixture-backed parser in ``test_sanitation`` rather than a second regex.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tests._discovery import discovered, workflow_files
from tests.test_sanitation import _ci_run_commands

pytestmark = pytest.mark.structural

_ROOT = Path(__file__).resolve().parents[2]


def _uv_syncs(ci_yaml: str) -> list[str]:
    """Every `uv sync ...` command in a CI workflow, via the shared parser."""
    return [c for c in _ci_run_commands(ci_yaml) if c.split()[:2] == ["uv", "sync"]]


# ---------------------------------------------------------------------------
# CI must never trigger the release-LTO project build
# ---------------------------------------------------------------------------


def test_ci_uv_sync_never_installs_the_project() -> None:
    """No workflow may let `uv sync` build+install the project.

    `uv sync` without `--no-install-project` compiles and installs polars-cv
    through its maturin backend at the workspace release profile (fat LTO) — a
    multi-minute build of the whole polars stack. Every job that needs the
    extension builds it explicitly with `maturin develop`; every job that does
    not (docs) needs no extension at all. So the implicit release build is
    always waste, in every workflow.
    """
    seen_any = False
    offenders: dict[str, list[str]] = {}
    for wf in workflow_files():
        syncs = _uv_syncs(wf.read_text())
        seen_any = seen_any or bool(syncs)
        bad = [c for c in syncs if "--no-install-project" not in c]
        if bad:
            offenders[wf.name] = bad

    assert seen_any, "no `uv sync` found in any workflow — the parser drifted"
    assert not offenders, (
        "these workflow `uv sync` invocations build+install polars-cv at the "
        f"release profile, work that is always discarded or unneeded: {offenders}. "
        "Add `--no-install-project`."
    )


def test_uv_sync_guard_would_catch_a_regression() -> None:
    """Watch the guard fail for its reason: a project-installing sync is flagged.

    Without this, a broken predicate (e.g. matching the wrong tokens) would let
    the guard above pass while watching nothing.
    """
    bad_ci = """
    jobs:
      test:
        steps:
          - run: uv sync --group dev
    """
    offenders = [c for c in _uv_syncs(bad_ci) if "--no-install-project" not in c]
    assert offenders == ["uv sync --group dev"], (
        "the predicate did not flag a project-installing `uv sync` — the CI "
        "guard would be watching nothing"
    )


def test_uv_sync_guard_accepts_the_fixed_form() -> None:
    good_ci = """
    jobs:
      test:
        steps:
          - run: uv sync --group dev --no-install-project
    """
    offenders = [c for c in _uv_syncs(good_ci) if "--no-install-project" not in c]
    assert offenders == [], "the guard rejects the corrected form — a false positive"


# ---------------------------------------------------------------------------
# The SessionStart hook: async, debug-only, self-reclaiming
# ---------------------------------------------------------------------------


def _hook() -> str:
    return (_ROOT / ".claude" / "hooks" / "session-start.sh").read_text()


def test_session_start_hook_is_async() -> None:
    assert '"async": true' in _hook(), (
        "the SessionStart hook must emit the async control line so the harness "
        "backgrounds the build; run synchronously, a cold compile of the polars "
        "stack blocks the session prompt for minutes"
    )


def test_session_start_hook_emits_async_line_before_building() -> None:
    """The async echo must run before the first *command* (comments don't count)."""
    lines = _hook().splitlines()
    async_line = next(
        (i for i, ln in enumerate(lines) if '"async": true' in ln and "echo" in ln),
        None,
    )
    assert async_line is not None, "no `echo '{...\"async\"...}'` line in the hook"

    work = ("uv sync", "uv run", "maturin", "cargo clean", "rustup", "pre-commit")
    first_command = next(
        (
            i
            for i, ln in enumerate(lines)
            if not ln.lstrip().startswith("#")
            and any(tok in ln for tok in work)
            and "echo" not in ln
        ),
        None,
    )
    assert first_command is not None, "the hook runs no build/setup commands at all"
    assert async_line < first_command, (
        "the async control line must be printed before any build/setup command, "
        "or the harness never sees it and runs the hook synchronously"
    )


def test_session_start_hook_builds_nothing() -> None:
    """The session builds the extension itself (``maturin develop``, debug);
    a build in the hook raced it. Nor may the sync build the project."""
    hook = _hook()
    assert "--no-install-project" in hook, (
        "the hook must skip the release-LTO project build (`--no-install-project`)"
    )
    commands = [
        ln for ln in hook.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    builds = [ln for ln in commands if "maturin" in ln and "echo" not in ln]
    assert builds == [], f"the hook must not build the extension: {builds}"


def test_session_start_hook_provisions_every_verify_lane() -> None:
    """The hook installs what ``scripts/verify.sh`` needs beyond dev deps.

    verify.sh treats a missing tool as a FAIL (no false green), so a hook that
    skips one leaves every fresh web session unable to pass verify.sh on a
    correct tree: that is how the mkdocs lane (docs dependency group) and the
    cargo-deny lane (a separately installed binary) failed in every session.
    """
    hook = _hook()
    assert "--group docs" in hook, (
        "the hook must sync the docs dependency group, or verify.sh's "
        "`mkdocs build --strict` lane fails with a missing module"
    )
    assert "cargo install --locked cargo-deny" in hook, (
        "the hook must install cargo-deny, or verify.sh's `cargo deny` lane fails "
        "with `no such command: deny`"
    )
    assert "cargo install --locked cargo-machete" in hook, (
        "the hook must install cargo-machete, or verify.sh's `cargo machete` lane "
        "fails with `no such command: machete`"
    )


def test_session_start_hook_exports_the_pyo3_env_to_the_session() -> None:
    """Session shells get maturin's PyO3 env, so ad-hoc cargo does not thrash.

    A bare ``cargo test`` in a session shell otherwise lacks the variables
    ``maturin develop`` sets, and the next ``maturin develop`` rebuilds the
    polars stack. The values come from ``scripts/with-pyo3-env.sh`` (the one
    authority) and reach every later Bash call through ``$CLAUDE_ENV_FILE``.
    """
    hook = _hook()
    assert "scripts/with-pyo3-env.sh" in hook, (
        "the hook must derive the PyO3 env from scripts/with-pyo3-env.sh, not "
        "restate it"
    )
    assert "CLAUDE_ENV_FILE" in hook, (
        "the hook must persist the PyO3 env for the session via $CLAUDE_ENV_FILE"
    )


# ---------------------------------------------------------------------------
# The knobs that make `maturin develop` fast and small
# ---------------------------------------------------------------------------


def test_linux_dev_build_links_with_lld() -> None:
    cfg = (_ROOT / ".cargo" / "config.toml").read_text()
    assert "-fuse-ld=lld" in cfg, (
        "the x86_64-linux dev build should link the plugin cdylib with lld — "
        "linking the whole polars stack with the default linker is the slowest "
        "single step of `maturin develop`"
    )


def test_dev_profile_is_tuned_for_iteration() -> None:
    cargo = (_ROOT / "Cargo.toml").read_text()
    assert "[profile.dev]" in cargo, "no [profile.dev] — dev build uses cargo defaults"
    assert "debug = 1" in cargo, (
        "the dev profile should carry line-table debuginfo (`debug = 1`): full "
        "`debug = 2` DWARF for the polars stack is the bulk of target/debug"
    )


def test_uv_never_builds_the_project() -> None:
    """`uv run` must not build the extension (CR-43).

    `--no-install-project` protects only the commands that pass it; every
    `uv run` re-syncs the project and, for a maturin-backed package, compiles
    it at the release profile (fat LTO) — observed from the documented
    `uv run pytest`. `package = false` makes uv treat the project as virtual,
    so no invocation can build it and `maturin develop` stays the one build.
    """
    config = tomllib.loads((_ROOT / "polars-cv" / "pyproject.toml").read_text())
    assert config.get("tool", {}).get("uv", {}).get("package") is False, (
        "polars-cv/pyproject.toml needs `[tool.uv] package = false`: without it "
        "`uv run` builds the extension at release LTO"
    )


# ---------------------------------------------------------------------------
# The benchmark build: optimised like release, linked in a fraction of the time
# ---------------------------------------------------------------------------


def test_benchmark_profile_is_release_with_thin_lto() -> None:
    """``[profile.benchmark]`` is release, minus the fat-LTO link.

    A release build of this stack takes 10-18 minutes here, almost all of it
    the single-codegen-unit fat-LTO link; a benchmark only needs the kernels
    optimised. It inherits release so ``panic = "unwind"`` (which the plugin's
    ``catch_unwind`` relies on) and ``opt-level = 3`` cannot drift apart.
    """
    profiles = tomllib.loads((_ROOT / "Cargo.toml").read_text())["profile"]
    bench = profiles.get("benchmark")
    assert bench is not None, "no [profile.benchmark] in the workspace Cargo.toml"
    assert bench.get("inherits") == "release", bench
    assert bench.get("lto") == "thin", bench
    assert bench.get("codegen-units", 16) > 1, bench


def test_benchmark_artifacts_are_reclaimed_like_release() -> None:
    """``target/benchmark`` is as large as ``target/release``; both get cleared."""
    for script in (
        _ROOT / "scripts" / "dev-clean.sh",
        _ROOT / ".claude" / "hooks" / "session-start.sh",
    ):
        text = script.read_text()
        assert "cargo clean --profile benchmark" in text, (
            f"{script.relative_to(_ROOT)} must reclaim target/benchmark"
        )


# ---------------------------------------------------------------------------
# What keeps target/debug from growing without bound
# ---------------------------------------------------------------------------


def _workspace() -> dict[str, dict]:
    """Each workspace member's package name -> its parsed Cargo.toml."""
    root = tomllib.loads((_ROOT / "Cargo.toml").read_text())
    members = {}
    for member in root["workspace"]["members"]:
        manifest = tomllib.loads((_ROOT / member / "Cargo.toml").read_text())
        members[manifest["package"]["name"]] = manifest
    return members


def _sidecar_leaking_members(cargo: dict, members: list[str]) -> list[str]:
    """The workspace members whose dev build writes split-debuginfo sidecars.

    With ``split-debuginfo = "unpacked"``, each incremental rebuild of a crate
    writes its ``.dwo`` files under new names, and cargo never deletes the old
    ones; only the workspace's own crates rebuild incrementally, so each must
    override it to ``"off"``. A dev profile that is not ``unpacked`` leaks
    nothing.
    """
    dev = cargo.get("profile", {}).get("dev", {})
    if dev.get("split-debuginfo") != "unpacked":
        return []
    overrides = dev.get("package", {})
    return [
        m
        for m in members
        if overrides.get(m, {}).get("split-debuginfo", "unpacked") != "off"
    ]


def test_sidecar_rule_fixtures() -> None:
    """The rule itself, on known-bad and known-good profiles."""
    unpacked = {"profile": {"dev": {"split-debuginfo": "unpacked"}}}
    assert _sidecar_leaking_members(unpacked, ["a", "b"]) == ["a", "b"]
    half = {
        "profile": {
            "dev": {
                "split-debuginfo": "unpacked",
                "package": {"a": {"split-debuginfo": "off"}},
            }
        }
    }
    assert _sidecar_leaking_members(half, ["a", "b"]) == ["b"]
    assert _sidecar_leaking_members({"profile": {"dev": {}}}, ["a"]) == []


def test_workspace_crates_write_no_debuginfo_sidecars() -> None:
    """Every workspace crate keeps its debuginfo inline in the dev build.

    A session of edits left ~79,000 stale ``.dwo`` files (about 12 GB) in
    ``target/debug/deps``, all from this workspace's crates: the dependencies
    keep their sidecars, which they write once.
    """
    cargo = tomllib.loads((_ROOT / "Cargo.toml").read_text())
    leaking = _sidecar_leaking_members(cargo, sorted(_workspace()))
    assert not leaking, (
        f"workspace members {leaking} write split-debuginfo sidecars that pile up "
        'with every rebuild: add `[profile.dev.package.<name>] split-debuginfo = "off"`'
    )


def test_rust_tests_use_the_plugins_polars() -> None:
    """``polars-cv`` restates no ``polars`` for its tests.

    The tests then exercise the polars features the plugin ships with, not a
    superset (the dev-dependency enabled polars' defaults and ``lazy``).
    """
    dev = _workspace()["polars-cv"].get("dev-dependencies", {})
    assert "polars" not in dev, (
        "polars-cv's Rust tests must use the plugin's own `polars` dependency, "
        "not a dev-dependency with a different feature set"
    )


def test_view_buffer_integration_tests_are_one_binary() -> None:
    """``view-buffer/tests`` is one test crate, and compiles every file in it.

    Each top-level ``tests/*.rs`` is its own executable linking all of
    view-buffer: 31 of them were ~4 GB of ``target/debug``, each relinked on
    every change. In the one crate (``tests/it``), a file ``main.rs`` does not
    declare is compiled by nothing, so it is checked too.
    """
    tests = _ROOT / "view-buffer" / "tests"
    entries = discovered(sorted(tests.iterdir()), "view-buffer/tests entries")
    loose = [p.name for p in entries if p.suffix == ".rs"]
    assert not loose, (
        f"view-buffer/tests/{loose} would each be a separate test binary: add "
        "them as modules of tests/it/main.rs instead"
    )
    main = (tests / "it" / "main.rs").read_text()
    declared = set(re.findall(r"^mod (\w+);", main, re.M))
    modules = discovered(
        sorted(p for p in (tests / "it").iterdir() if p.suffix == ".rs"),
        "tests/it modules",
    )
    files = {p.stem for p in modules} - {"main"}
    assert files == declared, (
        f"undeclared in tests/it/main.rs (never compiled): {sorted(files - declared)}; "
        f"declared without a file: {sorted(declared - files)}"
    )


def test_session_start_hook_cleans_target_when_the_toolchain_changes() -> None:
    """A new rustc orphans every artifact in ``target/``; the hook reclaims it.

    The check must follow ``rustup update`` (which is what changes the
    toolchain) and precede the dependency sync.
    """
    hook = _hook()
    assert "rustc -vV" in hook, (
        "the hook must compare the compiler target/ was built with"
    )
    stamp = hook.index("RUSTC_STAMP=")
    sync = hook.index("uv sync --group")
    assert hook.index("rustup update stable") < stamp < sync, (
        "the toolchain check must run after `rustup update` and before the sync"
    )
    check = hook[stamp:sync]
    assert "cargo clean" in check, "a changed toolchain must clean target/"
