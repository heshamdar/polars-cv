"""The verification entry point (``scripts/verify.sh``) must not drift from CI."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


def test_verify_script_covers_every_ci_check() -> None:
    """``scripts/verify.sh`` must run every check CI runs.

    The script exists so a full verification is one command with per-check exit
    codes, rather than a set of commands whose output someone reads and
    summarises — reading a filtered view is what produced false "all green"
    reports here more than once. That only holds if the script stays complete,
    so the two are pinned together: a check added to CI and not to the script
    would leave the local run passing while CI fails.
    """
    root = Path(__file__).resolve().parents[2]
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    # Read only the lines that *invoke* a check, never the raw file: a whole-
    # file search is satisfied by a comment naming the check, so commenting out
    # a `run_check` line while leaving any mention of it was undetectable. Its
    # sibling `test_no_ci_check_is_missing_from_the_verify_script` was fixed
    # this way and this one was not, which left one live hole -- commenting out
    # the structural lane passed *both* guards.
    script = _verify_script_checks((root / "scripts" / "verify.sh").read_text())
    assert script.strip(), (
        "no run_check lines found in verify.sh -- the scan is broken, and "
        "would otherwise report every check as missing (or, if inverted, none)"
    )

    # (substring identifying the check in CI, substring identifying it in the
    # script). Matched loosely on purpose: flags legitimately differ (CI adds
    # -x/-v, the script adds -q), and pinning exact command lines would make
    # this fail on formatting rather than on a missing check.
    required = [
        ("cargo fmt --all -- --check", "cargo fmt --all -- --check"),
        ("cargo clippy", "cargo clippy"),
        ("cargo test -p view-buffer", "cargo test -p view-buffer"),
        ("maturin develop", "maturin develop"),
        ('-m "not network and not slow"', '-m "not network and not slow"'),
        ('-m "slow and not network"', '-m "slow and not network"'),
        ('-m "structural and not slow"', '-m "structural and not slow"'),
        ("ruff check", "ruff check"),
        ("ruff format --check", "ruff format --check"),
        ("ty check", "ty check"),
        ("cargo deny", "cargo deny"),
        (
            "cargo check -p view-buffer --no-default-features",
            "cargo check -p view-buffer --no-default-features",
        ),
        ("mkdocs build --strict", "mkdocs build --strict"),
    ]
    missing = [
        ci_frag
        for ci_frag, script_frag in required
        if ci_frag in ci and script_frag not in script
    ]
    assert not missing, (
        f"scripts/verify.sh is missing checks that CI runs: {missing}. "
        f"A local 'PASS' would not mean CI passes."
    )

    stale = [ci_frag for ci_frag, _ in required if ci_frag not in ci]
    assert not stale, (
        f"this test expects CI to run checks it no longer does: {stale}. "
        f"Update the list rather than leaving it asserting nothing."
    )


#: Shell constructs in a `run:` block that invoke nothing checkable.
_CI_SHELL_NOISE = ("#", "if", "then", "else", "elif", "fi", "echo", "source", "cd")

#: How each command CI runs is classified: the value is a token that must
#: appear in `scripts/verify.sh`, or ``None`` when the line is setup rather
#: than a check.
#:
#: A key matches when its words appear as an ordered subsequence of the
#: command's tokens, starting at the first — so `uv run pytest` matches
#: `uv run --no-sync pytest tests/ -q`, and flags between the words do not
#: need enumerating. The longest match wins.
#:
#: A command line matching **no** entry fails the test. That is what makes this
#: more than a second hand-written list: adding a new checker to CI cannot be
#: silent, because the new command is unclassified until someone says which of
#: the two it is.
_CI_COMMAND_CLASSIFICATION: "dict[str, str | None]" = {
    "cargo fmt": "cargo fmt",
    "cargo clippy": "cargo clippy",
    "cargo test": "cargo test",
    "cargo deny": "cargo deny",
    "cargo check": "cargo check -p view-buffer",
    "maturin develop": "maturin develop",
    "pytest": "pytest",
    "uvx ruff check": "ruff check",
    "uvx ruff format": "ruff format",
    "uvx ty": "ty check",
    "uv run mkdocs": "mkdocs build",
    "uv run pytest": "pytest",
    # Setup: installs and environment, nothing a code change can break.
    "uv sync": None,
    "uv venv": None,
    "uv pip": None,
    "source": None,
    # Deliberately NOT here: `uv run python`. Exempting it classified every
    # `uv run python scripts/check_*.py` as setup, so a whole family of
    # checkers could join CI unnoticed. A Python script that gates the build
    # is a check, and belongs above with the token verify.sh must contain.
}


def _matches_command(key_words: "list[str]", tokens: "list[str]") -> bool:
    """True when *key_words* appear in order among *tokens*, starting at the first.

    Anchoring on the first token keeps `cargo test` from matching a line that
    merely mentions cargo later; allowing gaps after that lets flags sit between
    the words (`uv run --no-sync pytest`) without every combination needing its
    own entry.
    """
    if not tokens or not key_words or tokens[0] != key_words[0]:
        return False
    remaining = iter(tokens[1:])
    return all(word in remaining for word in key_words[1:])


def _normalise_command(line: str) -> str:
    """Strip leading ``VAR=value`` assignments from one shell command.

    ``PYTHONPATH=python uv run …`` is a ``uv run`` invocation; classifying it
    by the variable name would make every env-prefixed command look like a
    tool nobody has heard of.
    """
    while re.match(r"^[A-Za-z_][A-Za-z0-9_]*=\S*\s+\S", line):
        line = line.split(None, 1)[1]
    return line.strip()


def _ci_run_commands(ci_yaml: str) -> list[str]:
    """Every command CI runs, from both ``run:`` forms.

    Read textually rather than by parsing the YAML: the ``run:`` bodies are
    plain shell, and what matters is which executables they invoke.

    Three things this has to get right, each of which it once got wrong and
    each of which hid a real check:

    * **Both spellings of ``run:``.** The block form ``run: |`` and the
      single-line form ``run: cargo test -p view-buffer``. Reading only the
      block form left ``cargo test -p view-buffer`` — a check `ci.yml` has run
      all along — invisible to the guard that claims nothing can hide.
    * **Chained commands.** ``cd polars-cv && deno lint`` is two commands, and
      classifying the line by its first token files the whole thing under
      ``cd``, which is shell noise. Every ``&&``/``;``/``||`` segment is
      classified on its own, so a checker cannot be smuggled in behind a
      ``cd``.
    * **Continuations.** A command split over lines with a trailing ``\\`` is
      one command; joining them keeps the tool name and its flags together.
    """
    lines: list[str] = []
    in_run = False
    run_indent = 0

    for raw in ci_yaml.splitlines():
        # The `- run:` form is legal YAML and just as capable of hiding a
        # check, so both spellings are read even though the workflows here
        # only use the own-line one today.
        block = re.match(r"^(\s*)(?:-\s+)?run:\s*\|", raw)
        if block:
            in_run = True
            run_indent = len(block.group(1))
            continue
        single = re.match(r"^\s*(?:-\s+)?run:\s*(?!\|)(\S.*)$", raw)
        if single:
            in_run = False
            lines.append(single.group(1).strip())
            continue
        if not in_run or not raw.strip():
            continue
        if len(raw) - len(raw.lstrip()) <= run_indent:
            in_run = False
            continue
        lines.append(raw.strip())

    # Re-join backslash continuations before splitting on operators.
    joined: list[str] = []
    for line in lines:
        if joined and joined[-1].endswith("\\"):
            joined[-1] = joined[-1][:-1].rstrip() + " " + line
        else:
            joined.append(line)

    commands: list[str] = []
    for line in joined:
        # Drop comments before splitting on operators. A comment reading
        # "Only the slow-marked tests; still skip network ones." otherwise
        # yields "still skip network ones." as a command nobody can classify.
        line = re.sub(r"(^|\s)#.*$", "", line).strip()
        if not line:
            continue
        for segment in re.split(r"&&|\|\||;", line):
            command = _normalise_command(segment)
            if not command:
                continue
            if command.split()[0] in _CI_SHELL_NOISE:
                continue
            commands.append(command)
    return commands


def _verify_script_checks(script: str) -> str:
    """Only the lines of ``verify.sh`` that actually invoke a check.

    A whole-file substring search is satisfied by a *comment* naming the check
    — so commenting out a `run_check` line, while leaving any mention of it,
    was undetectable. Checks are declared one way, `run_check "<label>" <cmd>`,
    so read those lines and nothing else.
    """
    kept: list[str] = []
    continuation = False
    for raw in script.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continuation = False
            continue
        if line.startswith("run_check") or continuation:
            kept.append(line)
            continuation = line.endswith("\\")
            continue
        continuation = False
    return "\n".join(kept)


def test_no_ci_check_is_missing_from_the_verify_script() -> None:
    """A check *added* to CI must be added to ``verify.sh`` too.

    ``test_verify_script_covers_every_ci_check`` above pins a hand-written list
    both ways — a check in that list must be in CI, and must be in the script.
    What neither direction catches is a check added to CI that nobody adds to
    the list: it is simply never mentioned, and the local run keeps reporting
    PASS while CI fails.

    This closes that direction by reading every command in ``ci.yml``'s
    ``run:`` blocks and requiring each to be *classified* — as a check that
    ``verify.sh`` must also run, or as setup. An unclassified command fails, so
    a checker added to CI cannot pass unnoticed the way an unlisted one could.
    """
    root = Path(__file__).resolve().parents[2]
    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    script = _verify_script_checks((root / "scripts" / "verify.sh").read_text())

    commands = _ci_run_commands(ci)
    assert commands, (
        "no commands were found in ci.yml's run: blocks. Either the workflow "
        "was restructured or the extraction has rotted; this test is checking "
        "nothing."
    )

    unclassified: list[str] = []
    missing: list[str] = []
    for command in commands:
        tokens = command.split()
        matches = [
            key
            for key in _CI_COMMAND_CLASSIFICATION
            if _matches_command(key.split(), tokens)
        ]
        if not matches:
            unclassified.append(command)
            continue
        token = _CI_COMMAND_CLASSIFICATION[max(matches, key=lambda k: len(k.split()))]
        if token is not None and token not in script:
            missing.append(command)

    assert not unclassified, (
        f"ci.yml runs commands this test cannot classify: {sorted(set(unclassified))}. "
        f"Add each to _CI_COMMAND_CLASSIFICATION — with the token verify.sh must "
        f"contain if it is a check, or None if it is setup. Leaving it out is how "
        f"a check gets into CI and never into the local script."
    )
    assert not missing, (
        f"ci.yml runs {sorted(set(missing))}, which scripts/verify.sh does not. "
        f"A local 'PASS' would not mean CI passes. Add the check to the script."
    )
