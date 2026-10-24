"""Checks for the guard-test registry (see ``test_guard_registry``).

A guard module is any ``tests/test_*`` file whose name contains ``guard`` or
``ratchet``. Helper modules (leading underscore) are never guards. A guard
"has tests" when its AST defines at least one function named ``test_*``,
including methods in classes. This counts definitions, not collected or
passing tests: it catches a guard that was emptied or gutted, not one that
fails, which the test run already reports.

Callers pass the files to check, normally ``tests._discovery.suite_files()``,
so the set is never found by a direct glob (see ``_discovery``'s docstring).
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from fnmatch import fnmatch
from pathlib import Path

GUARD_PATTERNS = ("test_*guard*.py", "test_*ratchet*.py")


def _is_guard(path: Path) -> bool:
    return any(fnmatch(path.name, pattern) for pattern in GUARD_PATTERNS)


def guard_modules(files: Iterable[Path]) -> set[str]:
    """File names of every guard test module among ``files``."""
    return {path.name for path in files if _is_guard(path)}


def unregistered(files: Iterable[Path], registry: Iterable[str]) -> set[str]:
    """Guard modules among ``files`` that the registry does not name."""
    return guard_modules(files) - set(registry)


def count_tests(path: Path) -> int:
    """Number of ``test_*`` function definitions in the file at ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )


def empty_or_missing(files: Iterable[Path], registry: Iterable[str]) -> list[str]:
    """Registered guards that are not among ``files`` or define no tests."""
    by_name = {path.name: path for path in files}
    bad = []
    for name in registry:
        path = by_name.get(name)
        if path is None or count_tests(path) == 0:
            bad.append(name)
    return bad
