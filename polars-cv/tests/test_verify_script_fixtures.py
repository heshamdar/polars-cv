"""Fixtures for the CI/verify.sh parity helpers in :mod:`tests.test_sanitation`.

`test_no_ci_check_is_missing_from_the_verify_script` parses two hand-written
formats — GitHub Actions YAML and a bash script — with regexes. Every branch in
those helpers exists because a real check once hid from the guard: a block-form
``run: |`` hid ``cargo test -p view-buffer``, a ``cd x && lint`` chain filed a
checker under ``cd``, a comment satisfied a whole-file substring search for a
``run_check`` that had been commented out.

Those are exactly the branches a future edit can silently drop, and the guard
gives no sign when it does — it just classifies fewer commands and keeps
passing. `CLAUDE.md`: a guard with non-trivial logic gets committed fixtures,
both known-bad snippets it must reject and known-good ones it must not.

The helpers are imported from the test module rather than a ``_``-prefixed
support module because they are coupled to its classification tables
(``_CI_COMMAND_CLASSIFICATION``, ``_CI_SHELL_NOISE``); splitting them would
move those tables away from the test that reads them.
"""

from __future__ import annotations

import pytest

from tests.test_sanitation import (
    _ci_run_commands,
    _matches_command,
    _normalise_command,
    _verify_script_checks,
)

# ---------------------------------------------------------------------------
# _ci_run_commands: every command CI runs must be visible to the classifier
# ---------------------------------------------------------------------------

_CI_CASES = {
    "block form run: | with several commands": (
        """
        jobs:
          test:
            steps:
              - name: A step
                run: |
                  cargo test -p view-buffer --all-features
                  cargo test -p polars-cv
        """,
        {"cargo test -p view-buffer --all-features", "cargo test -p polars-cv"},
    ),
    "single-line run:": (
        """
                run: cargo fmt --all -- --check
        """,
        {"cargo fmt --all -- --check"},
    ),
    "chained commands are split, cd dropped as noise": (
        """
                run: cd polars-cv && uvx ruff check python tests
        """,
        {"uvx ruff check python tests"},
    ),
    "backslash continuation is rejoined": (
        """
              - name: A step
                run: |
                  uv run pytest tests/ \\
                    -m "not network and not slow"
        """,
        {'uv run pytest tests/ -m "not network and not slow"'},
    ),
    "trailing comment is not a command": (
        """
              - name: A step
                run: |
                  cargo clippy  # Only the slow-marked tests; still skip network ones.
        """,
        {"cargo clippy"},
    ),
    "env-prefixed command classifies by the tool": (
        """
                run: PYTHONPATH=python uv run pytest tests/
        """,
        {"uv run pytest tests/"},
    ),
    "the inline dash form is read too": (
        """
              - run: cargo deny check
        """,
        {"cargo deny check"},
    ),
    "a block ends when indentation returns": (
        """
              - name: A step
                run: |
                  cargo fmt --all -- --check
              - name: Something else
                uses: actions/checkout@v4
        """,
        {"cargo fmt --all -- --check"},
    ),
}


@pytest.mark.parametrize("label", sorted(_CI_CASES))
def test_ci_run_commands_extracts_the_expected_commands(label: str) -> None:
    yaml_text, expected = _CI_CASES[label]
    assert set(_ci_run_commands(yaml_text)) == expected, (
        f"extraction drifted for {label!r} — a command CI runs would become "
        f"invisible to the parity guard"
    )


def test_ci_run_commands_finds_nothing_in_a_workflow_without_run_blocks() -> None:
    """The empty case must stay distinguishable from a broken regex.

    The caller asserts non-emptiness against the real workflow; this pins that
    an genuinely run-free workflow yields ``[]`` rather than raising, so the
    caller's assertion is the thing that fires.
    """
    assert _ci_run_commands("jobs:\n  test:\n    steps:\n      - uses: x@v1\n") == []


# ---------------------------------------------------------------------------
# _verify_script_checks: only real invocations count, never comments
# ---------------------------------------------------------------------------


def test_a_commented_out_check_is_not_counted() -> None:
    """The bug this helper exists for: a comment satisfying a substring search."""
    script = """
    # run_check "cargo test view-buffer" cargo test -p view-buffer
    run_check "cargo fmt --check" cargo fmt --all -- --check
    """
    kept = _verify_script_checks(script)
    assert "cargo fmt" in kept
    assert "view-buffer" not in kept, (
        "a commented-out run_check was counted as a live check, which is "
        "exactly how a disabled check kept reporting PASS"
    )


def test_a_continued_run_check_keeps_its_later_lines() -> None:
    script = 'run_check "pytest fast" uv run pytest tests/ \\\n  -m "not network"\n'
    kept = _verify_script_checks(script)
    assert "not network" in kept, (
        "a check's flags were dropped, so a substring test for them would "
        "wrongly report the check missing"
    )


def test_a_comment_interrupts_a_continuation() -> None:
    """A comment after a continued line must not be absorbed into the check."""
    script = 'run_check "a" cmd \\\n# run_check "b" other\nunrelated_line\n'
    kept = _verify_script_checks(script)
    assert "other" not in kept


# ---------------------------------------------------------------------------
# _matches_command: anchored on the first token, gaps allowed after it
# ---------------------------------------------------------------------------

_MATCH_CASES = [
    (["cargo", "test"], ["cargo", "test", "-p", "polars-cv"], True),
    (["uv", "run", "pytest"], ["uv", "run", "--no-sync", "pytest", "tests/"], True),
    # must NOT match: `cargo` is not the first token, it is merely mentioned
    (["cargo", "test"], ["echo", "cargo", "test"], False),
    (["cargo", "test"], ["cargo", "fmt", "--all"], False),
    ([], ["cargo"], False),
    (["cargo"], [], False),
]


@pytest.mark.parametrize(("key", "tokens", "expected"), _MATCH_CASES)
def test_matches_command(key: list[str], tokens: list[str], expected: bool) -> None:
    assert _matches_command(key, tokens) is expected


def test_normalise_command_strips_only_leading_assignments() -> None:
    assert _normalise_command("A=1 B=2 cargo test") == "cargo test"
    # a bare assignment is not a command prefix — nothing follows it to run
    assert _normalise_command("A=1") == "A=1"
