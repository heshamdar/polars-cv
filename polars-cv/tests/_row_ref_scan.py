"""The row-reference scan, separated so its fixtures test the code that guards
the tree (``tests/test_row_ref_guard.py``).

An error naming a row by hand (``"... (row {})"``, ``"at row {}"``) presents
a batch index as the frame's row under the streaming engine; the one
rendering is ``polars-cv/src/row_ref.rs``. This reads every
``polars_err!``/``polars_bail!``/``polars_ensure!``/``format!`` invocation and
reports a string literal in it that formats a row number.

Limits: a source scan. It sees only those four macros (an ``assert!`` message
in a test is not an error a user reads), finds a macro's end by balancing
parentheses outside string literals, and does not follow a format string held
in a variable.
"""

from __future__ import annotations

import re

_MACRO = re.compile(r"\b(polars_err|polars_bail|polars_ensure|format)!\s*\(")
#: A string literal; an escape may be a `\\`-newline continuation.
_STRING = re.compile(r'"(?:\\[\s\S]|[^"\\])*"')
#: "row" followed by a format placeholder: `row {}`, `row {i}`, `row {:?}`.
_ROW_PLACEHOLDER = re.compile(r"\brow \{[^}]*\}")


def _invocation(source: str, start: int) -> str:
    """The text of the macro invocation whose `(` is at *start*."""
    depth, i = 0, start
    while i < len(source):
        if source[i] == '"':
            m = _STRING.match(source, i)
            i = m.end() if m else i + 1
            continue
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    return source[start:]


def row_ref_offenders(source: str, where: str) -> list[str]:
    """``where:line`` for each error macro formatting a row number itself."""
    found = []
    for m in _MACRO.finditer(source):
        text = _invocation(source, m.end() - 1)
        if any(_ROW_PLACEHOLDER.search(s) for s in _STRING.findall(text)):
            line = source.count("\n", 0, m.start()) + 1
            found.append(f"{where}:{line}: {m.group(1)}! names a row by hand")
    return found
