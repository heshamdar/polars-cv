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

import sys
from pathlib import Path

import pytest

from tests._discovery import workflow_files
from tests.test_sanitation import _ci_run_commands

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: `tomllib` is 3.11+ stdlib; `tomli` is its exact predecessor.
    import tomli as tomllib

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


def test_session_start_hook_builds_debug_only() -> None:
    hook = _hook()
    assert "--no-install-project" in hook, (
        "the hook must skip the release-LTO project build (`--no-install-project`)"
    )
    assert "maturin develop" in hook, "the hook must build the debug extension"
    assert "maturin develop --release" not in hook, (
        "the SessionStart hook must build the canonical debug extension, never a "
        "`maturin develop --release`"
    )
    assert "maturin build --release" not in hook, (
        "the SessionStart hook must not produce release wheels"
    )


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
