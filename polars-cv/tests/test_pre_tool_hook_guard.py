"""The PreToolUse hook blocks the Bash commands CLAUDE.md forbids, and only those.

``.claude/hooks/pre_tool_guard.py`` is the one place these rules live. The
known-bad fixtures must be refused with their rule named; the known-good ones
must pass. A refusal is exit 2 with the rule id on stderr. The rule id is
checked as well as the exit code, because a missing script also exits 2 and
would otherwise look like a block.

The hook must also be registered for Bash in ``.claude/settings.json``, or none
of this runs; ``test_the_hook_is_registered_for_bash`` pins that.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests._discovery import REPO_ROOT

pytestmark = pytest.mark.structural

HOOK = REPO_ROOT / ".claude" / "hooks" / "pre_tool_guard.py"

BAD = [
    ("git commit --no-verify -m wip", "no-verify"),
    ("git commit -n -m wip", "no-verify"),
    ("git push --no-verify origin main", "no-verify"),
    ("git push --force origin feature", "force-push"),
    ("git push -f origin feature", "force-push"),
    ("git push --force-with-lease origin feature", "force-push"),
    ("git push origin +HEAD:main", "force-push"),
    ("git reset --hard HEAD~1", "reset-hard"),
    ("git clean -fd", "clean-force"),
    ("git clean -f", "clean-force"),
    ("git checkout .", "discard-changes"),
    ("git checkout -- .", "discard-changes"),
    ("git restore .", "discard-changes"),
    ("git branch -D old-branch", "branch-force-delete"),
    ("rm -rf build", "recursive-delete"),
    ("rm -r build", "recursive-delete"),
    ("cd polars-cv && git reset --hard", "reset-hard"),
    ("pytest -q; git push --force", "force-push"),
    ("make test | tee log && git clean -fdx", "clean-force"),
    ("bash -c 'git reset --hard'", "reset-hard"),
    ("sudo rm -rf /tmp/scratch", "recursive-delete"),
    ("FOO=1 git push -f origin feature", "force-push"),
    ("git -C polars-cv reset --hard", "reset-hard"),
]

GOOD = [
    "git status",
    "git log --grep='reset --hard'",
    "git log --oneline -5",
    "git commit -m 'fix: explain why --no-verify is banned'",
    "git push -u origin ccr-d58fac73-vojcra",
    "git checkout -b new-branch",
    "git checkout main",
    "git restore --staged notes/file.md",
    "git branch -d merged-branch",
    "git diff HEAD~1",
    "rm notes/scratch.txt",
    "uv run pytest -q",
    "cargo test -p view-buffer",
    "scripts/verify.sh",
    "echo 'git reset --hard' > notes.txt",
    "git -C polars-cv status",
]


def _run(payload: object) -> subprocess.CompletedProcess[str]:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=body,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _bash(command: str) -> dict[str, object]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
    }


@pytest.mark.parametrize(("command", "rule"), BAD)
def test_a_forbidden_command_is_refused_with_its_rule(command: str, rule: str) -> None:
    result = _run(_bash(command))
    assert result.returncode == 2, result.stderr
    assert f"[{rule}]" in result.stderr, result.stderr


@pytest.mark.parametrize("command", GOOD)
def test_an_allowed_command_passes(command: str) -> None:
    result = _run(_bash(command))
    assert result.returncode == 0, result.stderr


def test_a_non_bash_tool_is_not_inspected() -> None:
    payload = {"tool_name": "Edit", "tool_input": {"file_path": "x.py"}}
    assert _run(payload).returncode == 0


def test_unreadable_input_fails_closed() -> None:
    result = _run("not json at all")
    assert result.returncode == 2
    assert "pre_tool_guard" in result.stderr


def test_the_hook_is_registered_for_bash() -> None:
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text())
    entries = settings["hooks"]["PreToolUse"]
    bash = [e for e in entries if e.get("matcher") == "Bash"]
    assert bash, "no PreToolUse hook matches Bash in .claude/settings.json"
    commands = [h["command"] for e in bash for h in e["hooks"]]
    assert any("pre_tool_guard.py" in c for c in commands), commands
    handlers = [h for e in bash for h in e["hooks"]]
    assert all(h.get("onFailure") == "block" for h in handlers), (
        "a crashed or timed-out guard must block, not pass: set onFailure to block"
    )
