"""Which benchmark cases a change can move: changed files -> ``Selection``.

``RULES`` maps repository paths to the selectors (``selection`` syntax) whose
cases execute that code. As in ``CODEOWNERS``, the **last** matching rule wins,
so a file rule (``graph/encode.rs``) overrides its directory's (``graph/**``)
and must name everything the file moves. A change selects the union over its
files. An empty selector records that no scenario measures the file.

It is a map, so it can be wrong in one direction the harness cannot detect:
naming too few cases for a file. It cannot be wrong silently in the others:

- a changed code file no rule matches is an error (``select_for``), and a guard
  test holds every tracked code file to a rule. Crate-root files are listed
  one by one, so a new module there must be placed deliberately;
- a selector naming a case that no longer exists fails to parse, and a guard
  test parses every rule.

When unsure, name more — a benchmark that runs needlessly costs minutes, one
that is skipped costs a regression.

Patterns: ``*`` matches within one path segment, ``**`` across segments.
"""

from __future__ import annotations

import re
import subprocess
from functools import cache
from typing import TYPE_CHECKING

from benchmarks.regression.selection import SCENARIOS, Selection, parse

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_ALL = ",".join(SCENARIOS)
_OPS = "single_ops,pipelines,e2e"
_GEOM = "targeted:geom_*"
_NONE = ""  # no scenario measures this code

# Product code: what a benchmark executes. Tests, docs and the harness itself
# are not in it, so they select nothing.
_CODE_ROOTS = (
    "view-buffer/src/",
    "polars-cv/src/",
    "polars-cv-macros/src/",
    "polars-cv/python/polars_cv/",
)
_CODE_SUFFIXES = (".rs", ".py")
# Build inputs that change what every kernel compiles to.
_BUILD_FILES = (
    "Cargo.toml",
    "Cargo.lock",
    "*/Cargo.toml",
    ".cargo/config.toml",
    "*/.cargo/config.toml",
    "rust-toolchain.toml",
    "polars-cv/build.rs",
    "polars-cv/pyproject.toml",
    "polars-cv/uv.lock",
)

RULES: tuple[tuple[str, str], ...] = (
    *((f, _ALL) for f in _BUILD_FILES),
    # The derive macros generate every op's wire form, resolution and catalogue.
    ("polars-cv-macros/src/**", _ALL),
    # --- view-buffer, the engine ---
    ("view-buffer/src/ops/**", _OPS),
    ("view-buffer/src/execution/**", _OPS),
    ("view-buffer/src/expr.rs", _OPS),
    ("view-buffer/src/lib.rs", _OPS),
    ("view-buffer/src/mode.rs", _OPS),
    ("view-buffer/src/naming.rs", _OPS),
    ("view-buffer/src/core/**", f"{_OPS},zero_copy,targeted"),
    ("view-buffer/src/geometry/**", _GEOM),
    ("view-buffer/src/interop/**", "zero_copy,targeted:sink_*,targeted:blob_*"),
    ("view-buffer/src/interop/image.rs", f"{_OPS},targeted:codec_*"),
    ("view-buffer/src/protocol.rs", "zero_copy,targeted:blob_*"),
    # --- polars-cv, the plugin ---
    ("polars-cv/src/lib.rs", _OPS),
    ("polars-cv/src/naming.rs", _OPS),
    ("polars-cv/src/params.rs", _OPS),
    ("polars-cv/src/passes.rs", _OPS),
    ("polars-cv/src/plan.rs", _OPS),
    ("polars-cv/src/ops/**", _OPS),
    ("polars-cv/src/graph/**", _OPS),
    (
        "polars-cv/src/graph/encode.rs",
        f"{_OPS},zero_copy,targeted:sink_*,targeted:codec_*",
    ),
    ("polars-cv/src/graph/sink_kind.rs", f"{_OPS},zero_copy,targeted:sink_*"),
    (
        "polars-cv/src/graph/decode.rs",
        f"{_OPS},zero_copy,targeted:codec_*,targeted:blob_*",
    ),
    ("polars-cv/src/formats/**", f"{_OPS},zero_copy,targeted"),
    ("polars-cv/src/execute.rs", f"{_OPS},targeted:codec_*"),
    ("polars-cv/src/output.rs", f"{_OPS},zero_copy,targeted:blob_*"),
    ("polars-cv/src/ext_types.rs", _NONE),  # tagged (`ndarray`) outputs only
    ("polars-cv/src/row_split.rs", f"pipelines,{_GEOM}"),
    ("polars-cv/src/geom_*.rs", _GEOM),
    ("polars-cv/src/contour.rs", _GEOM),
    ("polars-cv/src/point.rs", _GEOM),
    ("polars-cv/src/fetch.rs", "remote"),
    ("polars-cv/src/cloud.rs", "remote"),
    ("polars-cv/src/cloud_auth.rs", "remote"),
    ("polars-cv/src/read_bytes.rs", "remote"),
    ("polars-cv/src/image_metadata.rs", _NONE),
    # The global allocator: every allocation any case makes (CR-60).
    ("polars-cv/src/allocator.rs", _ALL),
    ("polars-cv/src/test_alloc.rs", _NONE),  # test-only allocator
    # --- the Python builder: plan, serialize and dispatch run on every call ---
    ("polars-cv/python/polars_cv/*.py", _OPS),
    ("polars-cv/python/polars_cv/display.py", _NONE),
    ("polars-cv/python/polars_cv/_graph_viz.py", _NONE),
    ("polars-cv/python/polars_cv/geometry/**", _GEOM),
    ("polars-cv/python/polars_cv/metrics/**", _NONE),  # pure polars; unbenchmarked
)


@cache
def _regex(pattern: str) -> re.Pattern[str]:
    out = []
    for part in re.split(r"(\*\*|\*|\?)", pattern):
        out.append({"**": ".*", "*": "[^/]*", "?": "[^/]"}.get(part, re.escape(part)))
    return re.compile("".join(out))


def is_code(path: str) -> bool:
    """Whether ``path`` (repo-relative) is code a benchmark can execute."""
    if any(_regex(p).fullmatch(path) for p in _BUILD_FILES):
        return True
    return path.startswith(_CODE_ROOTS) and path.endswith(_CODE_SUFFIXES)


def rule_for(path: str) -> str | None:
    """The selector spec of the last rule matching ``path`` (None: no rule)."""
    spec = None
    for pattern, rule_spec in RULES:
        if _regex(pattern).fullmatch(path):
            spec = rule_spec
    return spec


def select_for(paths: Iterable[str]) -> tuple[Selection, list[str]]:
    """The cases ``paths`` can move, and the code among them nothing measures.

    Raises ``ValueError`` for a code file no rule matches: a new module must be
    mapped (even to ``""``) before a change to it can be benchmarked.
    """
    sel = Selection()
    unbenchmarked: list[str] = []
    unmapped: list[str] = []
    for path in paths:
        if not is_code(path):
            continue
        spec = rule_for(path)
        if spec is None:
            unmapped.append(path)
        elif not spec:
            unbenchmarked.append(path)
        else:
            sel = sel | parse(spec)
    if unmapped:
        msg = (
            f"no relevance rule covers {unmapped}; add one to "
            f'benchmarks/regression/relevance.py RULES ("" if no scenario '
            f"measures it)"
        )
        raise ValueError(msg)
    return sel, unbenchmarked


def _git(args: list[str], cwd: Path | str | None) -> list[str]:
    out = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return out.stdout.split()


def changed_files(ref: str, *, cwd: Path | str | None = None) -> list[str]:
    """Every file present now that differs from ``ref``'s merge base.

    Includes uncommitted edits and untracked files, so a change can be
    benchmarked before it is committed. Paths are repository-relative.
    """
    base = _git(["merge-base", ref, "HEAD"], cwd)[0]
    # Deletions (and the old side of renames) are left out: the code that
    # replaced them, or their callers, is in the diff and carries the rule.
    diffed = _git(["diff", "--name-only", "--diff-filter=d", base], cwd)
    untracked = _git(["ls-files", "--others", "--exclude-standard", "--full-name"], cwd)
    return sorted({*diffed, *untracked})


def main(argv: list[str] | None = None) -> int:
    """Print the selection the changes since ``REF`` can move.

    The output is a ``--select`` argument, so both sides of a comparison run
    the same cases even after the base is checked out (where ``--changed``
    would see no change)::

        SEL=$(python -m benchmarks.regression.relevance origin/main)
        python -m benchmarks.regression.run_suite --select "$SEL" --out head.json
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("ref", help="the base to diff against (its merge base)")
    args = parser.parse_args(argv)
    try:
        sel, unbenchmarked = select_for(changed_files(args.ref))
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if unbenchmarked:
        print(f"not measured by any scenario: {unbenchmarked}", file=sys.stderr)
    if not sel:
        print("no benchmarked code changed", file=sys.stderr)
    print(sel.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
