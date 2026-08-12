"""Fixtures for the Markdown readers in ``_doc_tables``.

The guards in ``test_doc_vocabularies.py`` rest entirely on this parsing. A
reader that silently stopped matching would take those guards green with it —
the failure mode this repo has shipped more than once — so the known-bad and
known-good cases are pinned here.

These call the same functions the guards call. A fixture exercising a
reimplementation would prove nothing about the guard.
"""

from __future__ import annotations

import pytest

from ._doc_tables import (
    cell_code,
    fenced_python_method_calls,
    pipe_tables,
    table_with_header,
)

#: Every test here is a structural guard: it checks the *shape* of the
#: codebase rather than the behaviour of a pipeline, so it needs no compiled
#: extension and runs in milliseconds. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural

# ---------------------------------------------------------------------------
# pipe_tables
# ---------------------------------------------------------------------------

_ONE_TABLE = """\
# Heading

| Domain | Description |
|--------|-------------|
| `buffer` | Image data |
| `scalar` | Single number |
"""

_TWO_TABLES = """\
| A | B |
|---|---|
| 1 | 2 |

Some prose between them.

| C | D |
| :-- | --: |
| 3 | 4 |
"""


def test_reads_a_single_table_without_its_separator() -> None:
    tables = pipe_tables(_ONE_TABLE)
    assert tables == [
        [
            ["Domain", "Description"],
            ["`buffer`", "Image data"],
            ["`scalar`", "Single number"],
        ]
    ]


def test_prose_between_tables_splits_them() -> None:
    """Two tables must not merge into one.

    Merging would let a guard reading "the first table" silently pick up rows
    from a second, unrelated one.
    """
    tables = pipe_tables(_TWO_TABLES)
    assert len(tables) == 2
    assert tables[0][0] == ["A", "B"]
    assert tables[1][0] == ["C", "D"]


def test_alignment_colons_are_still_a_separator() -> None:
    """``| :-- | --: |`` is a separator, not a data row."""
    rows = pipe_tables(_TWO_TABLES)[1]
    assert ["", ""] not in rows
    assert rows == [["C", "D"], ["3", "4"]]


def test_a_page_with_no_tables_yields_none() -> None:
    assert pipe_tables("# Just prose\n\nNo tables here.\n") == []


# ---------------------------------------------------------------------------
# table_with_header
# ---------------------------------------------------------------------------


def test_table_is_found_by_header_not_position() -> None:
    """A table inserted above the wanted one must not shift the answer.

    This is the property that makes the guards robust to editing: they name the
    columns they want rather than counting tables.
    """
    rows = table_with_header(_TWO_TABLES, "C", "D")
    assert rows == [["3", "4"]]


def test_unmatched_header_returns_none_rather_than_the_first_table() -> None:
    """The dangerous failure is answering with the *wrong* table."""
    assert table_with_header(_TWO_TABLES, "Domain", "Description") is None


# ---------------------------------------------------------------------------
# cell_code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("`buffer`", "buffer"),
        ("  `file_path`  ", "file_path"),
        # Prose is not a term, even when it contains a code span: reading
        # "Binary carrying the VIEW protocol magic" as a vocabulary entry is
        # how a guard starts reporting nonsense.
        ("`Binary` otherwise", None),
        ("Image/array data", None),
        ("", None),
    ],
)
def test_cell_code_reads_only_a_whole_cell_code_span(
    cell: str, expected: str | None
) -> None:
    assert cell_code(cell) == expected


# ---------------------------------------------------------------------------
# fenced_python_method_calls
# ---------------------------------------------------------------------------

_FENCES = """\
Prose calling `.not_extracted()` outside a fence.

```python
pipe = Pipeline().source("image_bytes").resize(height=1, width=2)
```

```py
df.with_columns(x=pl.col("a").cv.pipe(pipe).sink("numpy"))
```

```bash
some_command --flag
```

```
plain fence .ignored()
```
"""


def test_only_python_fences_are_read() -> None:
    found = fenced_python_method_calls(_FENCES)
    assert {"source", "resize", "with_columns", "col", "pipe", "sink"} <= found
    # A bash fence and an unlabelled fence are not Python.
    assert "some_command" not in found
    assert "ignored" not in found
    # Prose outside any fence is not code a reader copies.
    assert "not_extracted" not in found


def test_consecutive_fences_do_not_merge() -> None:
    """Non-greedy matching: the bash block must not be swallowed into the py one."""
    assert "flag" not in fenced_python_method_calls(_FENCES)


def test_a_float_is_not_a_method_call() -> None:
    """``0.5(`` must not read as a call to ``5``."""
    text = "```python\nx = fn(0.5) * 1.25\n```\n"
    assert fenced_python_method_calls(text) == set()


def test_chained_calls_are_all_found() -> None:
    text = "```python\na.b().c().d()\n```\n"
    assert fenced_python_method_calls(text) == {"b", "c", "d"}


def test_a_page_with_no_python_yields_nothing() -> None:
    """The empty answer the guards must treat as failure, not success."""
    assert fenced_python_method_calls("# Prose only\n") == set()
