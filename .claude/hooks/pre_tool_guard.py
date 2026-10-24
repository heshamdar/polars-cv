#!/usr/bin/env python3
"""PreToolUse guard for Bash: refuse the commands CLAUDE.md forbids.

Claude Code sends the tool call as JSON on stdin. A refusal is exit 2 with the
rule id on stderr, which Claude receives as the tool's error. Any other exit
code lets the call proceed, so every internal failure here exits 2 instead:
an unreadable input is refused, not waved through.

What it understands: each shell command is split on ``; && || | &`` with a
real shell lexer, so quoted text is never mistaken for a command. Leading
``VAR=value`` assignments and ``sudo``/``env``/``command``/``nohup``/``time``
wrappers are skipped, ``git -C``/``git -c`` are skipped, and ``bash -c``/``sh
-c`` strings are checked recursively.

What it does not: command substitution ``$(...)`` and backticks are not parsed,
nor ``eval``, ``xargs`` or scripts a command runs. It is a backstop against
honest mistakes, not a sandbox. The rules are the ones in ``polars-cv/tests/
test_pre_tool_hook_guard.py``; change a rule there and here together.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys

WRAPPERS = frozenset({"sudo", "env", "command", "nohup", "time", "exec"})
SHELLS = frozenset({"bash", "sh", "zsh"})
SEPARATORS = frozenset(";&|()")
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def _segments(command: str) -> list[list[str]]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    out: list[list[str]] = []
    current: list[str] = []
    for token in lexer:
        if token and set(token) <= SEPARATORS:
            if current:
                out.append(current)
            current = []
        else:
            current.append(token)
    if current:
        out.append(current)
    return out


def _strip_wrappers(words: list[str]) -> list[str]:
    while words and (ASSIGNMENT.match(words[0]) or words[0] in WRAPPERS):
        words = words[1:]
    return words


def _git_subcommand(args: list[str]) -> tuple[str, list[str]]:
    """The git subcommand and the words after it, skipping ``-C dir``/``-c k=v``."""
    i = 0
    while i < len(args) and args[i] in {"-C", "-c"}:
        i += 2
    if i >= len(args):
        return "", []
    return args[i], args[i + 1 :]


def _short_flags(words: list[str]) -> set[str]:
    return {w[1:] for w in words if w.startswith("-") and not w.startswith("--")}


def _check_git(rest: list[str], sub: str) -> str | None:
    if sub == "commit" and ("--no-verify" in rest or "-n" in rest):
        return "no-verify"
    if sub == "push":
        if "--no-verify" in rest:
            return "no-verify"
        if any(w in {"--force", "-f"} or w.startswith("--force") for w in rest):
            return "force-push"
        if any(w.startswith("+") for w in rest):
            return "force-push"
        if any("f" in flags for flags in _short_flags(rest)):
            return "force-push"
    if sub == "reset" and "--hard" in rest:
        return "reset-hard"
    if sub == "clean" and (
        any("f" in flags for flags in _short_flags(rest)) or "--force" in rest
    ):
        return "clean-force"
    if sub in {"checkout", "restore"} and "." in rest:
        if sub == "restore" and "--staged" in rest and "--worktree" not in rest:
            return None
        return "discard-changes"
    if sub == "branch" and ("-D" in rest or ("--delete" in rest and "--force" in rest)):
        return "branch-force-delete"
    return None


def _violation(words: list[str]) -> str | None:
    words = _strip_wrappers(words)
    if not words:
        return None
    program = os.path.basename(words[0])
    args = words[1:]
    if program == "git":
        sub, rest = _git_subcommand(args)
        return _check_git(rest, sub)
    if program == "rm":
        if any(c in "rR" for flags in _short_flags(args) for c in flags) or (
            "--recursive" in args
        ):
            return "recursive-delete"
        return None
    if program in SHELLS and "-c" in args:
        nested = args[args.index("-c") + 1 :]
        if nested:
            return _check(nested[0])
    return None


def _check(command: str) -> str | None:
    for words in _segments(command):
        rule = _violation(words)
        if rule:
            return rule
    return None


REASONS = {
    "no-verify": "skips the pre-commit gate, which CLAUDE.md requires to pass",
    "force-push": "rewrites published history",
    "reset-hard": "discards uncommitted work",
    "clean-force": "deletes untracked files",
    "discard-changes": "discards working-tree changes",
    "branch-force-delete": "deletes a branch even when it is not merged",
    "recursive-delete": "deletes a directory tree",
}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except (ValueError, OSError) as exc:
        print(
            f"pre_tool_guard: cannot read the hook input ({exc}); refusing",
            file=sys.stderr,
        )
        return 2
    if not isinstance(payload, dict):
        print(
            "pre_tool_guard: hook input is not a JSON object; refusing", file=sys.stderr
        )
        return 2
    if payload.get("tool_name") != "Bash":
        return 0
    tool_input = payload.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str):
        print(
            "pre_tool_guard: Bash input has no command string; refusing",
            file=sys.stderr,
        )
        return 2
    try:
        rule = _check(command)
    except ValueError as exc:
        print(
            f"pre_tool_guard: cannot parse the command ({exc}); refusing",
            file=sys.stderr,
        )
        return 2
    if rule is None:
        return 0
    print(
        f"Blocked by pre_tool_guard [{rule}]: this {REASONS[rule]}. "
        "Ask the user to run it themselves.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - any unforeseen failure must block, not pass
        print(f"pre_tool_guard: internal error ({exc!r}); refusing", file=sys.stderr)
        sys.exit(2)
