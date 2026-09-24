#!/usr/bin/env python
"""Fail if a symbol a phase deleted appears anywhere in the tracked sources.

The typed-op migration (``TYPED_OPS_PLAN.md``) ends every phase by deleting
what it replaced. Compiler errors catch a deleted Rust item that is still
*called*; they do not catch a comment, a doc page, a Python string or a test
that still names it, and those are how a removed mechanism quietly comes back
or keeps being documented. This is the phase exit gate for that: each entry
names a symbol, the reason it is gone, and (rarely) the files that may still
mention it on purpose.

Matching is whole-word (``\\b``), over files tracked by git, so build output,
virtualenvs and untracked scratch files are never read. History files are
exempt: the changelog, the review ledger and the plan itself record removals
by name.

Limits: this is a textual scan (see CLAUDE.md, "prefer compiler exhaustiveness
> runtime assertion > source scanning"). It is used only for what the compiler
cannot see — references in prose, strings and other languages — and each
symbol must be distinctive enough that a whole-word match means the symbol.

Usage::

    python scripts/check_removed_symbols.py      # exit 1 on any hit
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Files that record removals by name and are therefore never scanned.
HISTORY_FILES: frozenset[str] = frozenset(
    {
        "CHANGELOG.md",
        "CODE_REVIEW_FINDINGS.md",
        "TYPED_OPS_PLAN.md",
        "polars-cv/docs/changelog.md",
        "polars-cv/scripts/check_removed_symbols.py",
        "polars-cv/tests/test_removed_symbols.py",
    }
)


@dataclass(frozen=True)
class Removed:
    """One deleted symbol: what it was, why it is gone, where it may remain."""

    symbol: str
    reason: str
    allowed_in: frozenset[str] = field(default_factory=frozenset)


#: Every removed symbol, grouped by the change that removed it. A phase of the
#: typed-op plan appends its list here as its exit gate.
REMOVED: tuple[Removed, ...] = (
    # CR-32: a call runs its rows in parallel, so the single-thread warning's
    # advice ("ran on one thread", use streaming) became false.
    Removed("engine_warning", "CR-32: the single-thread engine warning was deleted"),
    Removed("CallGuard", "CR-32: the engine warning's per-call tracker"),
    Removed(
        "POLARS_CV_SILENCE_ENGINE_WARNING",
        "CR-32: the engine warning it silenced is gone",
        # The tombstone clears it so a leftover setting cannot mask a warning.
        allowed_in=frozenset({"polars-cv/tests/test_removed_surfaces.py"}),
    ),
    Removed(
        "POLARS_CV_ENGINE_WARN_SECONDS",
        "CR-32: the engine warning it tuned is gone",
        # The tombstone sets the old trigger threshold to prove nothing prints.
        allowed_in=frozenset({"polars-cv/tests/test_removed_surfaces.py"}),
    ),
    # Typed-op P1: expression parameters cross the boundary as positional
    # `{"$slot": n}` indices assigned by `SlotTable`; no text, no names.
    Removed("expr_key", "P1: display-text expression identity (CR-31)"),
    Removed("_EXPR_KEYS", "P1: the process-wide expr_key registry"),
    Removed(
        "expr_column_names",
        "P1: the name list that bound expression params to input columns",
        # The tombstone sends it to prove the kwargs struct rejects it.
        allowed_in=frozenset({"polars-cv/tests/test_removed_surfaces.py"}),
    ),
    Removed("name_to_slot", "P1: Rust name->slot binding"),
    Removed("shape_pipeline", "P1: replaced by the `shape_node` id"),
    Removed("op_probe_json", "P1: probe re-serialization; `op_json` is serde"),
    Removed("param_probe_json", "P1: probe re-serialization; `op_json` is serde"),
    Removed("_build_column_bindings", "P1: bindings come from the SlotTable"),
    Removed("_get_ordered_columns", "P1: input columns come from the SlotTable"),
)


@dataclass(frozen=True)
class Hit:
    symbol: str
    path: str
    line: int
    reason: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: `{self.symbol}` ({self.reason})"


def find_hits(removed: tuple[Removed, ...], files: dict[str, str]) -> list[Hit]:
    """Every whole-word occurrence of a removed symbol outside its allowances.

    ``files`` maps repo-relative paths to their text. History files are skipped.
    """
    if not files:
        msg = "no files to scan: an empty scan would pass vacuously"
        raise ValueError(msg)
    patterns = [(r, re.compile(rf"\b{re.escape(r.symbol)}\b")) for r in removed]
    hits: list[Hit] = []
    for path, text in sorted(files.items()):
        if path in HISTORY_FILES:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for entry, pattern in patterns:
                if path not in entry.allowed_in and pattern.search(line):
                    hits.append(Hit(entry.symbol, path, number, entry.reason))
    return hits


def tracked_text_files(root: Path = REPO_ROOT) -> dict[str, str]:
    """Repo-relative path -> text for every tracked, UTF-8 decodable file."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
    ).stdout.split(b"\0")
    files: dict[str, str] = {}
    for raw in listed:
        if not raw:
            continue
        rel = raw.decode()
        try:
            files[rel] = (root / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary, or deleted in the working tree
    # The scan must reach both languages, or a broken listing reads as clean.
    for suffix in (".rs", ".py", ".md"):
        if not any(p.endswith(suffix) for p in files):
            msg = f"file listing found no {suffix} files under {root}"
            raise ValueError(msg)
    return files


def main() -> int:
    hits = find_hits(REMOVED, tracked_text_files())
    for hit in hits:
        print(hit)
    if hits:
        print(f"\n{len(hits)} reference(s) to removed symbols.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
