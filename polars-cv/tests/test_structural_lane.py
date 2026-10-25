"""The structural lane: pre-commit's ``-m structural`` selection must stay meaningful."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._discovery import (
    suite_modules,
)

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


def _module_marks(source: str) -> set[str]:
    """Mark names a module applies to itself via a **module-level** ``pytestmark``.

    Only ``tree.body`` is inspected, not ``ast.walk``. A ``pytestmark`` nested
    in a class body binds as a class attribute, which pytest applies to that
    class alone — walking the whole tree counted it as a module declaration,
    so moving one into a class silently cut the lane from 651 tests to 547
    while this and the lane-size guard both reported it healthy.
    """
    marks: set[str] = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.Attribute):
                marks.add(sub.attr)
    return marks


def _imports_discovery(source: str) -> bool:
    """True when a module imports ``tests._discovery`` in any form.

    Parsed rather than substring-matched. The substring ``"_discovery import"``
    caught only ``from ._discovery import …``; the equally ordinary
    ``from tests import _discovery as _d`` sailed past it, which made the
    "derived, so a new guard cannot join without choosing" claim untrue of a
    form the suite already permits.
    """
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if "_discovery" in (node.module or ""):
                return True
            if any(alias.name == "_discovery" for alias in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any("_discovery" in alias.name for alias in node.names):
                return True
    return False


def test_every_source_scanning_module_declares_its_lane() -> None:
    """A module that reads repository files must say which lane it is in.

    ``-m structural`` is what pre-commit runs, so a guard outside it is a guard
    the hook does not enforce. The set is derived rather than listed: reading
    repository files means going through ``_discovery`` (that is
    ``test_scans_go_through_discovery``'s whole job), so importing it is the
    signal, and a new source-scanning guard cannot join without choosing.

    ``slow`` is the other valid answer — ``test_examples.py`` discovers files
    and then spends a minute running them, which is not a pre-commit check.

    Limit worth stating: this covers modules that scan *sets* of files. A guard
    reading one named file (``test_param_strictness.py`` opening ``execute.rs``)
    has no reason to import ``_discovery`` and so is not caught here; those are
    marked by hand.
    """
    offenders = []
    for module in suite_modules():
        source = module.read_text()
        if not _imports_discovery(source):
            continue
        marks = _module_marks(source)
        if not ({"structural", "slow"} & marks):
            offenders.append(module.name)

    assert not offenders, (
        f"these modules scan repository files but declare no lane: {offenders}. "
        f"Add `pytestmark = pytest.mark.structural` (a fast shape check that "
        f"pre-commit should run) or `pytest.mark.slow` (something heavier)."
    )


#: Guard modules that must always be in the pre-commit lane.
#:
#: A bare count was too weak to be worth having: the threshold sat at 8 against
#: an actual 9, so dropping the marker from a hand-marked module left the lane
#: 104 tests smaller and this test still green. Naming the load-bearing modules
#: makes that specific loss fail. It is not the *whole* lane — a new guard
#: module is covered by the derived check above when it scans files. A module
#: that declares the marker but scans no files is covered by *neither*, so it
#: belongs here explicitly; two did, and were unpinned.
_CORE_STRUCTURAL_MODULES = frozenset(
    {
        "test_append_contract.py",
        "test_param_applicability.py",
        "test_param_strictness.py",
        "test_doc_vocabularies.py",
        "test_version_consistency.py",
        # The guards' guards. These carry the committed known-bad/known-good
        # fixtures that prove `_dtype_ratchet.dispatch_offenders` and
        # `_doc_tables` still detect what they claim to, so a ratchet that
        # silently stopped matching would be caught here and nowhere else.
        # They fell through *both* lane mechanisms: not named here, and not
        # covered by the derived check either, because that one keys on a
        # module scanning files via `_discovery` and these two scan fixtures.
        "test_dtype_ratchet_fixtures.py",
        "test_doc_table_fixtures.py",
        # The parity framework's coverage ratchets (every op, source, sink and
        # dtype has a parity entry or a reason) and the fixtures that prove its
        # comparators, tolerance propagation and divergence predicates still
        # reject what they claim to. Neither scans files.
        "test_parity_ratchets.py",
        "test_parity_framework_fixtures.py",
    }
)


def test_the_structural_lane_is_not_empty() -> None:
    """``-m structural`` must select the guards the pre-commit hook exists for.

    The lane is the hook's entire contents, so anything that quietly empties it
    — a renamed marker, a dropped ``pyproject.toml`` declaration, a
    ``pytestmark`` that moved into a class — leaves the hook passing in
    milliseconds while checking nothing.
    """
    marked = {
        m.name for m in suite_modules() if "structural" in _module_marks(m.read_text())
    }
    missing = sorted(_CORE_STRUCTURAL_MODULES - marked)
    assert not missing, (
        f"these guard modules are no longer in the structural lane: {missing}. "
        f"The pre-commit hook runs `-m structural`, so they are not enforced "
        f"before a commit any more."
    )

    markers = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    assert "structural:" in markers, (
        "the `structural` marker is not declared in pyproject.toml, so "
        "`-m structural` warns and the hook's selection is unenforced."
    )


def test_the_precommit_hook_runs_the_structural_lane() -> None:
    """The hook every guard above assumes must actually exist, and select it.

    ``test_the_structural_lane_is_not_empty`` justifies itself by "the lane is
    the pre-commit hook's entire contents" — but nothing read the hook. Its
    entry could be edited or deleted and no test would notice, which would make
    that justification, and the marker on ten modules, describe nothing.
    """
    # `parents[2]`, the git root, is where `pre-commit` actually looks: it
    # resolves the config relative to the repository top level, and installs
    # its hook into that repo's `.git/hooks`. The file used to live one level
    # down in `polars-cv/`, so the hook this test validated was not the one
    # `pre-commit` would load -- and every "the pre-commit hook enforces this"
    # justification in `tests/AGENTS.md` and ten module docstrings rested on
    # it. The config's own contents give the game away: `cargo fmt --all` and
    # `uv run --directory polars-cv` are both written for the root.
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / ".pre-commit-config.yaml"
    assert config_path.is_file(), (
        f"no .pre-commit-config.yaml at the git root ({repo_root}). "
        "pre-commit resolves its config from there, so a config anywhere else "
        "is never loaded and the structural lane does not run before a commit."
    )
    config = config_path.read_text()
    assert "structural-checks" in config, (
        "the structural-checks hook is gone from .pre-commit-config.yaml; the "
        "`structural` marker no longer feeds anything that runs before a commit."
    )
    assert '-m "structural and not slow"' in config, (
        'the structural-checks hook no longer selects `-m "structural and not '
        'slow"`, so the lane it runs is not the lane these guards describe.'
    )
    assert "--no-sync" in config, (
        "the structural-checks hook lost `--no-sync`, so every commit will "
        "re-sync the project and rebuild the Rust extension at --profile "
        "release — minutes per commit."
    )


def test_the_precommit_hook_runs_the_ci_python_linters() -> None:
    """The pre-commit hook must run the same ruff lint/format CI's lint job runs.

    CI's ``lint`` job fails on ``uvx ruff check`` and ``uvx ruff format
    --check``, and those are exactly the failures a commit-time hook can cheaply
    prevent. The hook used to run ruff through the ``ruff-pre-commit`` mirror,
    which pins its *own* ruff at a fixed rev — and that rev drifted from CI's
    unpinned ``uvx ruff`` (the mirror sat at v0.9.2 while ``uvx`` resolved
    0.15.x), so code the hook formatted locally still failed CI's format check.
    Two ruff versions is the second authority this repo's ``CLAUDE.md`` forbids:
    the fix is to run ruff through the same ``uvx ruff`` CI uses, so there is
    one ruff and passing the hook means passing CI's lint job.

    Read the required checks out of ``ci.yml`` so this guard cannot pin the hook
    to a check CI has stopped running.
    """
    repo_root = Path(__file__).resolve().parents[2]
    config = (repo_root / ".pre-commit-config.yaml").read_text()
    ci = (repo_root / ".github" / "workflows" / "ci.yml").read_text()

    for fragment in ("ruff check", "ruff format --check"):
        assert fragment in ci, (
            f"CI's lint job no longer runs `{fragment}`; update this guard "
            "rather than leaving it pinning the hook to a check that is gone."
        )
        assert fragment in config, (
            f"the pre-commit hooks do not run `{fragment}`, so a commit can "
            "pass the hook and still fail CI's lint job on it — the frequent "
            "lint failure this hook exists to prevent."
        )

    # One ruff, not two: the hook must invoke the same `uvx ruff` CI resolves,
    # never a second pinned ruff via the mirror that drifts from it. Key on the
    # mirror's *repo path*, which is only present when it is actually configured
    # as a `- repo:` entry — a prose mention of the name in a comment must not
    # trip this (the comment above the hook explains why the mirror was dropped).
    assert "astral-sh/ruff-pre-commit" not in config, (
        "the ruff-pre-commit mirror pins its own ruff version, which drifts "
        "from CI's `uvx ruff` and reintroduces the two-ruff skew this hook was "
        "changed to remove; run ruff through `uvx` in a local hook instead."
    )
    assert "uvx ruff" in config, (
        "the ruff hooks must invoke `uvx ruff`, the same resolution CI's lint "
        "job uses, so the hook and CI cannot lint with different ruff versions."
    )
