"""The import-layering guard's logic, separated so it can be tested itself.

``test_import_layering`` runs this over ``python/polars_cv``;
``test_import_layering_fixtures`` runs it over committed known-bad and
known-good packages. Both import from here, so the fixtures exercise the code
that guards the tree.

What it enforces, on the *static top-level import graph*:

(a) **No import cycle.** No strongly connected component of more than one
    module. A cycle makes import order load-bearing: whichever module is
    imported first decides which of its siblings sees a half-initialised peer.
(b) **No module inside the package imports the package root.** ``from polars_cv
    import X``, ``import polars_cv`` and a root-level module's ``from . import
    X`` all name ``polars_cv/__init__.py``, which re-exports the whole library,
    so every such edge closes a cycle the moment the root imports that module
    (and the root always eventually does). ``from polars_cv._x import y`` and
    ``from ._x import y`` name the submodule and are fine. ``__init__.py`` is
    the root and is exempt, as is the ``__init__`` of a subpackage importing
    its own siblings.

The graph has one edge per import statement run at import time: module-level
statements, including those inside ``try``/``if``/``with`` blocks and class
bodies, but not inside a function (those run on a call, and are how
``pipeline`` and ``lazy``, which construct each other, stay out of the graph)
and not inside an ``if TYPE_CHECKING:`` body (its ``else`` is kept).
``from pkg import name`` is an edge to ``pkg``, plus one to ``pkg.name`` when
that is a module.

Limits, stated so nobody mistakes this for more than it is:

* It is a source scan. ``importlib.import_module``, ``__import__``, a module
  ``__getattr__`` and ``exec`` are invisible to it.
* An ``if``/``try`` branch is assumed to run whichever way its condition goes
  (``sys.version_info`` checks, ``try/except ImportError``), so it can
  over-report a cycle that one branch would never take, never under-report.
* A parent package's ``__init__`` running before a submodule is not an edge
  (modelling it would make every submodule depend on the root). Rule (b) is
  what keeps that implicit edge from closing a cycle.
* Third-party and standard-library imports are not in the graph.

The subprocess test beside it (``import polars_cv.<mod>`` in a fresh
interpreter) is the dynamic check of the property this scan approximates.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from pathlib import Path

ROOT = "polars_cv"


def read_modules(
    paths: Iterable[Path], package_dir: Path, root: str = ROOT
) -> dict[str, tuple[str, bool]]:
    """Name -> (source, is_package) for ``paths``, files under ``package_dir``.

    The caller supplies the paths (``tests._discovery.package_modules()``, which
    refuses to return an empty set), so the scan cannot pass on an empty glob.
    """
    modules: dict[str, tuple[str, bool]] = {}
    for path in sorted(paths):
        parts = list(path.relative_to(package_dir).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        name = ".".join([root, *parts])
        modules[name] = (path.read_text(encoding="utf-8"), is_package)
    return modules


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _toplevel_imports(body: list[ast.stmt]) -> list[ast.Import | ast.ImportFrom]:
    """Imports that run when the module is imported."""
    found: list[ast.Import | ast.ImportFrom] = []
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            found.append(stmt)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        elif isinstance(stmt, ast.If):
            if not _is_type_checking(stmt.test):
                found += _toplevel_imports(stmt.body)
            found += _toplevel_imports(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                found += _toplevel_imports(block)
            for handler in stmt.handlers:
                found += _toplevel_imports(handler.body)
        elif isinstance(stmt, (ast.With, ast.AsyncWith, ast.For, ast.While)):
            found += _toplevel_imports(stmt.body)
            found += _toplevel_imports(getattr(stmt, "orelse", []))
        elif isinstance(stmt, ast.ClassDef):
            found += _toplevel_imports(stmt.body)
    return found


def _base(module: str, is_package: bool, node: ast.ImportFrom) -> str | None:
    """The absolute module a ``from ... import`` names, or None if it leaves."""
    if node.level == 0:
        return node.module
    package = module.split(".") if is_package else module.split(".")[:-1]
    keep = len(package) - (node.level - 1)
    if keep < 1:
        return None
    return ".".join([*package[:keep], *(node.module.split(".") if node.module else [])])


def analyse(
    modules: Mapping[str, tuple[str, bool]], root: str = ROOT
) -> tuple[dict[str, set[str]], list[str]]:
    """The import graph, and the rule-(b) violations found while building it."""
    graph: dict[str, set[str]] = {name: set() for name in modules}
    root_imports: list[str] = []

    def add(src: str, dst: str) -> None:
        if dst in modules and dst != src:
            graph[src].add(dst)

    for name, (source, is_package) in modules.items():
        for node in _toplevel_imports(ast.parse(source).body):
            names_root = False
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(name, alias.name)
                    names_root |= alias.name == root
            else:
                base = _base(name, is_package, node)
                if base is None:
                    continue
                add(name, base)
                names_root |= base == root
                for alias in node.names:
                    add(name, f"{base}.{alias.name}")
            if names_root and name != root:
                root_imports.append(
                    f"{name}:{node.lineno}: imports the package root "
                    f"(`{ast.unparse(node)}`); import the submodule instead"
                )
    return graph, sorted(root_imports)


def cycles(graph: Mapping[str, set[str]]) -> list[list[str]]:
    """Strongly connected components of more than one module (Tarjan)."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found: list[list[str]] = []
    counter = [0]

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(graph[v]):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == v:
                    break
            if len(component) > 1:
                found.append(sorted(component))

    for v in sorted(graph):
        if v not in index:
            visit(v)
    return sorted(found)


def violations(modules: Mapping[str, tuple[str, bool]], root: str = ROOT) -> list[str]:
    """Every layering violation in ``modules``, as printable lines."""
    graph, root_imports = analyse(modules, root)
    out = [f"import cycle: {' <-> '.join(c)}" for c in cycles(graph)]
    return out + root_imports
