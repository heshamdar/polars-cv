"""Reading vocabularies out of the Markdown documentation.

The user guide restates several vocabularies the code already owns — the domain
list, what ``source("auto")`` resolves each column dtype to, the operations a
pipeline offers. Those tables are hand-maintained, and nothing compared them to
anything, so a renamed method or a retired domain left the docs asserting
something untrue for as long as nobody noticed.

The parsing here is deliberately small, and it is the *only* copy: the guards in
``test_doc_vocabularies.py`` call these functions, and so do the fixtures in
``test_doc_table_fixtures.py``. A fixture exercising a reimplementation would
prove nothing about the guard.

Limits, stated because a source scan has them:

* :func:`pipe_tables` understands GitHub-style pipe tables only, and treats a
  run of consecutive ``|``-prefixed lines as one table. It does not handle pipes
  inside inline code, of which the docs have none.
* :func:`fenced_python_method_calls` reads ``.name(`` tokens textually. It
  cannot tell a ``Pipeline`` method from a Polars one, which is why its caller
  resolves each name against real classes rather than trusting the extraction.
"""

from __future__ import annotations

import re

#: A fenced block opening with ```python (or ```py), captured to its closing
#: fence. Non-greedy so consecutive blocks do not merge into one.
_PYTHON_FENCE = re.compile(r"^```(?:python|py)\s*$(.*?)^```\s*$", re.M | re.S)

#: A `code span` occupying a whole table cell, e.g. "`buffer`".
_CELL_CODE = re.compile(r"^`([^`]+)`$")

#: A method call in source text: ``.name(``.
#:
#: The lookbehind excludes only a preceding dot (``..name(``). It must *not*
#: exclude a preceding word character: ``df.with_columns(`` is the ordinary
#: shape of a method call, and an earlier ``(?<![\w.])`` here matched only
#: calls chained off a closing paren — silently missing most of the page.
#: ``test_chained_calls_are_all_found`` pins that.
#:
#: A float needs no special case: the name must start with ``[a-zA-Z_]``, so
#: ``0.5`` cannot match however it is followed.
_METHOD_CALL = re.compile(r"(?<!\.)\.([a-zA-Z_]\w*)\s*\(")


def pipe_tables(markdown: str) -> list[list[list[str]]]:
    """Every pipe table in *markdown*, as rows of trimmed cells.

    The header separator (``|---|---|``) is dropped; the header row survives as
    row 0, because the guards identify a table by its heading.

    Args:
        markdown: The page's full text.

    Returns:
        One entry per table: a list of rows, each a list of cell strings.
    """
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            # The separator row carries only dashes and colons.
            if all(set(c) <= set("-: ") and c for c in cells):
                continue
            current.append(cells)
            continue
        if current:
            tables.append(current)
            current = []

    if current:
        tables.append(current)
    return tables


def table_with_header(markdown: str, *header: str) -> list[list[str]] | None:
    """The first table whose header row starts with *header*, minus that row.

    Identifying a table by its column names rather than its position means
    inserting a paragraph above it does not silently point the guard at a
    different table.

    Args:
        markdown: The page's full text.
        header: The leading header cells to match, exactly.

    Returns:
        The table's data rows, or ``None`` when no table matches.
    """
    for table in pipe_tables(markdown):
        if not table:
            continue
        if tuple(table[0][: len(header)]) == header:
            return table[1:]
    return None


def cell_code(cell: str) -> str | None:
    """The code span filling *cell*, or ``None`` if the cell is not one.

    Used to read a vocabulary term out of a table cell: the docs write these as
    ``` `buffer` ```, and prose cells alongside them must not be mistaken for
    terms.
    """
    match = _CELL_CODE.match(cell.strip())
    return match.group(1) if match else None


def fenced_python_method_calls(markdown: str) -> set[str]:
    """Every ``.name`` called inside a ```python fence.

    Only fenced Python is read: prose mentions a method without calling it, and
    the guard's job is to check the code a reader would copy.
    """
    names: set[str] = set()
    for block in _PYTHON_FENCE.findall(markdown):
        names.update(_METHOD_CALL.findall(block))
    return names
