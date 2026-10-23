"""The in-memory pin checker, separated so its fixtures test the code that
guards the tree (``tests/test_engine_pin_guard.py``).

Limits: it reads string constants, so an engine name built at run time
(``"in-" + "memory"``) or passed through a variable defined in another
function is not seen. A function's scope includes the functions and classes
nested in it, and a parameter default counts as part of the function.
"""

from __future__ import annotations

import ast
from pathlib import Path

_IN_MEMORY = "in-memory"
_STREAMING = "streaming"
_AFFINITY_VAR = "POLARS_ENGINE_AFFINITY"


def _strings(node: ast.AST) -> set[str]:
    """The string constants under *node*, docstrings excluded."""
    docstrings = {
        id(body[0].value)
        for n in ast.walk(node)
        if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        )
        and (body := n.body)
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    }
    return {
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
    }


def _affinity_changes(node: ast.AST) -> list[ast.AST]:
    """Calls and assignments under *node* that change the engine affinity."""
    hits: list[ast.AST] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func = n.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name == "set_engine_affinity":
                hits.append(n)
            elif any(k.arg == "engine_affinity" for k in n.keywords):
                hits.append(n)
            elif name == "setenv" and any(
                isinstance(a, ast.Constant) and a.value == _AFFINITY_VAR for a in n.args
            ):
                hits.append(n)
        elif isinstance(n, ast.Assign):
            for target in n.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == _AFFINITY_VAR
                ):
                    hits.append(n)
    return hits


def engine_pin_offenders(
    source: str, name: str, *, path: Path | None = None
) -> list[str]:
    """``file:line: why`` for each pin of the in-memory engine in *source*.

    *name* is the file's name: ``conftest.py`` may set the lane's default at
    module level (outside any function).
    """
    tree = ast.parse(source)
    where = str(path) if path else name
    found: list[str] = []
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    in_function = {id(c) for f in functions for c in ast.walk(f)}
    for hit in _affinity_changes(tree):
        if name == "conftest.py" and id(hit) not in in_function:
            continue
        found.append(f"{where}:{hit.lineno}: changes the engine affinity")
    # Outermost functions only: a nested one is part of its parent's scope.
    nested = {id(c) for f in functions for c in ast.walk(f) if c is not f}
    for func in functions:
        if id(func) in nested:
            continue
        strings = _strings(func)
        if _IN_MEMORY in strings and _STREAMING not in strings:
            found.append(
                f"{where}:{func.lineno}: {func.name}() runs the in-memory engine "
                "without the streaming one"
            )
    return found
