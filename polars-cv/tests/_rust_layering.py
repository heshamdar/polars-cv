"""Module-level cycles in the Rust crates (``view-buffer/src``, ``polars-cv/src``).

Rust accepts module cycles, so the compiler never reports them. This scanner
reads the source text instead and runs the same strongly-connected-component
check as ``tests/_import_layering.py``.

What it reads: every ``crate::``, ``super::`` and ``self::`` path in a file,
with ``//`` comments and ``#[cfg(test)]`` items removed, resolved to the longest
module it starts with.

What it does not count: a ``mod x;`` declaration, and any edge between a module
and its own ancestor or descendant (``ops::histogram`` naming ``ops``). Rust
modules form a tree, and a child reaching for its parent's items is the normal
idiom; the layering problem is between sibling or cross-branch modules, such as
``kernels::histogram`` importing ``ops::histogram``.

Limits: a path reached through a ``use`` alias, a macro or a string is not seen,
and a brace or ``//`` inside a string literal can confuse the stripper. The
crate root may import modules but is never a target, since most modules name
root items and the root re-exports them. A grouped ``use crate::ops::{a, b}``
is an edge to ``ops``, not to ``ops::a``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from tests._import_layering import cycles

_PATH = re.compile(r"\b(crate|super|self)((?:::\w+)+)")
_CFG_TEST = re.compile(r"#\[cfg\(test\)\]")
_LINE_COMMENT = re.compile(r"//[^\n]*")


def module_path(src: Path, file: Path) -> str:
    """The module a file defines: ``""`` for the crate root, ``ops::util`` for ``ops/util.rs``."""
    parts = list(file.relative_to(src).with_suffix("").parts)
    if parts[-1] == "mod" or (len(parts) == 1 and parts[0] in ("lib", "main")):
        parts = parts[:-1]
    return "::".join(parts)


def strip_test_code(text: str) -> str:
    """``text`` without line comments or ``#[cfg(test)]`` items."""
    text = _LINE_COMMENT.sub("", text)
    while match := _CFG_TEST.search(text):
        brace = text.find("{", match.end())
        semi = text.find(";", match.end())
        if brace == -1 or (semi != -1 and semi < brace):
            cut = len(text) if semi == -1 else semi + 1
        else:
            depth = 0
            cut = len(text)
            for i in range(brace, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        cut = i + 1
                        break
        text = text[: match.start()] + text[cut:]
    return text


def _absolute(kind: str, segments: list[str], current: list[str]) -> list[str]:
    """The absolute module path a ``crate``/``super``/``self`` path names."""
    if kind == "crate":
        base: list[str] = []
    elif kind == "super":
        base = current[:-1]
    else:
        base = list(current)
    for segment in segments:
        if segment == "super":
            if base:
                base.pop()
        elif segment != "self":
            base.append(segment)
    return base


def _target(absolute: list[str], modules: set[str]) -> str | None:
    """The longest known module ``absolute`` starts with, if any."""
    for n in range(len(absolute), 0, -1):
        candidate = "::".join(absolute[:n])
        if candidate in modules:
            return candidate
    return None


def _nested(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are the same module or one contains the other."""
    return a == b or a.startswith(f"{b}::") or b.startswith(f"{a}::")


def _crate_of(file: Path, roots: Mapping[str, Path]) -> str | None:
    """The crate whose ``src/`` directory holds ``file``, if any."""
    for crate, src in roots.items():
        if file.is_relative_to(src):
            return crate
    return None


def scan(roots: Mapping[str, Path], files: Iterable[Path]) -> dict[str, set[str]]:
    """The module graph of each crate, as one graph with crate-prefixed names.

    ``roots`` maps a crate name to its ``src/`` directory; ``files`` is every
    ``.rs`` file to read (files outside every root are ignored). A crate's root
    is named by the crate alone (``view-buffer``); a module is
    ``view-buffer::ops::util``. Edges never cross crates.
    """
    placed = []
    modules: dict[str, set[str]] = {crate: set() for crate in roots}
    for file in files:
        crate = _crate_of(file, roots)
        if crate is None:
            continue
        mod = module_path(roots[crate], file)
        placed.append((crate, mod, file))
        if mod:
            modules[crate].add(mod)
    graph: dict[str, set[str]] = {}
    for crate, mod, file in placed:
        name = crate if mod == "" else f"{crate}::{mod}"
        edges = graph.setdefault(name, set())
        text = strip_test_code(file.read_text(encoding="utf-8"))
        current = mod.split("::") if mod else []
        for match in _PATH.finditer(text):
            segments = match.group(2).split("::")[1:]
            target = _target(
                _absolute(match.group(1), segments, current), modules[crate]
            )
            if target is not None and not _nested(target, mod):
                edges.add(f"{crate}::{target}")
    return graph


def rust_cycles(roots: Mapping[str, Path], files: Iterable[Path]) -> list[str]:
    """Each module cycle as ``a <-> b``, members sorted, cycles sorted."""
    return [" <-> ".join(component) for component in cycles(scan(roots, files))]
