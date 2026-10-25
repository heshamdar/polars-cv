"""The guard index: the canonical-paths table in the root ``AGENTS.md``.

Each row of that table names a fact, its one authority, and the mechanism that
rejects a second declaration — the guards. The table is the index of guards, so
it is checked both ways (``test_guard_index.py``):

- **Every guard it names exists.** A backticked ``test_*`` name, a ``Test*``
  class or a cited test file (``*.py`` / ``*.rs``) that resolves to nothing is
  a guard the table claims and the suite does not have — renamed or deleted.
  A bare snake_case name of three words or more
  (``catalog_matches_the_committed_file``, a Rust test) must appear somewhere
  in the Rust sources or the suite; that is all that is checked of it, since
  the column also cites the helpers and attributes a guard relies on
  (``ext_from_params``, ``deny_unknown_fields``), which are not tests.
- **Every source-scanning test file is in it.** A ``test_*.py`` that parses,
  tokenizes or reads source (``ast``, ``tokenize``, ``inspect.getsource``, or a
  ``_discovery`` file accessor) is a guard by construction; the table must cite
  the file or one of its tests, or the rule it holds has no visible guard.

Limits: the parse reads backticked tokens of the table's third column and
nothing else, so a guard cited only in prose elsewhere does not count; a
scanner is recognised by its imports, so one that reads files through a helper
not listed in :data:`SCANNING_MARKERS` is missed.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass

#: The heading of the section whose table is the index.
SECTION = "## Canonical Paths"

#: Imports and calls that make a test module a source scanner.
SCANNING_MARKERS: tuple[str, ...] = (
    "import ast",
    "import tokenize",
    "getsource(",
    "rust_sources(",
    "package_modules(",
    "suite_files(",
    "suite_modules(",
    "doc_pages(",
    "doc_page(",
    "repo_file(",
    "rust_src_dir(",
    "example_files(",
    "workflow_files(",
)

_BACKTICKED = re.compile(r"`([^`]+)`")
_RUST_TEST = re.compile(
    r"#\[test\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)"
)
_WORD = re.compile(r"\b[A-Za-z_]\w*\b")
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+){2,}$")


@dataclass(frozen=True)
class Suite:
    """What the guards can resolve against."""

    python_tests: frozenset[str]
    """Every ``test_*`` function and ``Test*`` class under ``tests/``."""
    rust_tests: frozenset[str]
    """Every ``#[test]`` function in the repository's Rust sources."""
    words: frozenset[str]
    """Every identifier-shaped word in the Rust sources and the test suite."""
    files: tuple[str, ...]
    """Every test, package, script and Rust source path, ``/``-separated, for
    suffix lookup."""


def guard_column(agents_md: str) -> list[str]:
    """The third cell of every row of the index table, in order.

    Raises:
        ValueError: If the section or its table is missing or has no rows —
            an index that parsed to nothing would vouch for nothing.
    """
    text = "\n" + agents_md
    start = text.find(f"\n{SECTION}\n")
    if start < 0:
        raise ValueError(f"no {SECTION!r} section")
    rows: list[str] = []
    in_table = False
    for line in text[start + len(SECTION) + 2 :].splitlines():
        if line.startswith("## "):
            break
        if line.startswith("|"):
            in_table = True
            cells = [c.strip() for c in _split_row(line)]
            if len(cells) < 3 or set(cells[0]) <= {"-", " "} or cells[0] == "Fact":
                continue
            rows.append(cells[2])
        elif in_table:
            break
    if not rows:
        raise ValueError(f"the {SECTION!r} table has no rows")
    return rows


def _split_row(line: str) -> list[str]:
    """A table row's cells: split on ``|`` outside backticks."""
    cells, cell, ticked = [], [], False
    for ch in line.strip().strip("|"):
        if ch == "`":
            ticked = not ticked
        if ch == "|" and not ticked:
            cells.append("".join(cell))
            cell = []
        else:
            cell.append(ch)
    cells.append("".join(cell))
    return cells


def cited(cells: Iterable[str]) -> list[str]:
    """Every backticked token in the cells, a ``file::name`` split in two."""
    tokens: list[str] = []
    for cell in cells:
        for token in _BACKTICKED.findall(cell):
            tokens.extend(part for part in token.split("::") if part)
    return tokens


def _is_file(token: str) -> bool:
    return token.endswith((".py", ".rs")) and " " not in token and "*" not in token


def unresolved(tokens: Iterable[str], suite: Suite) -> list[str]:
    """The cited guards that name nothing in the suite."""
    missing = []
    for token in tokens:
        if _is_file(token):
            if not any(f == token or f.endswith("/" + token) for f in suite.files):
                missing.append(token)
        elif token.startswith(("test_", "Test")) and token.isidentifier():
            if token not in suite.python_tests and token not in suite.rust_tests:
                missing.append(token)
        elif _SNAKE.match(token) and token not in suite.words:
            # A snake name of three words or more that appears nowhere: a Rust
            # guard renamed away. One that exists is not checked further — the
            # column also cites the helpers and attributes a guard relies on.
            missing.append(token)
    return missing


def python_test_names(source: str) -> set[str]:
    """``test_*`` functions and ``Test*`` classes defined in a module."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.add(node.name)
    return names


def rust_test_names(source: str) -> set[str]:
    """``#[test]`` functions defined in a Rust file."""
    return set(_RUST_TEST.findall(source))


def words(source: str) -> set[str]:
    """Every identifier-shaped word in a source file."""
    return set(_WORD.findall(source))


def scans_source(source: str) -> bool:
    """Whether a test module reads source to check it."""
    return any(marker in source for marker in SCANNING_MARKERS)


def uncited_scanners(scanners: dict[str, str], tokens: Iterable[str]) -> list[str]:
    """The scanning test files neither cited by name nor by one of their tests.

    Args:
        scanners: Each scanning test file's name, to its source.
        tokens: What the index cites.
    """
    cited_tokens = set(tokens)
    missing = []
    for name, source in sorted(scanners.items()):
        if name in cited_tokens or any(name.endswith("/" + t) for t in cited_tokens):
            continue
        if python_test_names(source) & cited_tokens:
            continue
        missing.append(name)
    return missing
