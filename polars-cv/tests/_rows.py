"""How an error names the row it is about, and the one assertion for it.

A plugin sees the rows polars passes in one call: the whole frame in memory,
a batch under the streaming engine. It cannot know a batch's offset in the
frame, so an error names the row within its batch and says how many rows that
batch has: ``(row 11 of 12 in this batch)``. When the batch is the whole
frame the number is the frame's row.
"""

from __future__ import annotations

import re

#: A row reference, as ``polars-cv/src/row_ref.rs`` renders it.
ROW = re.compile(r"\(row (\d+) of (\d+) in this batch\)")


def assert_names_row(message: str, row: int, frame_rows: int) -> None:
    """*message* names one row, consistently with the frame's *row* failing.

    On every engine the row is within its batch and the batch within the
    frame; a batch of the whole frame names the frame's own row.
    """
    hits = [(int(i), int(n)) for i, n in ROW.findall(message)]
    assert len(hits) == 1, f"expected one row reference in {message!r}"
    ((index, batch),) = hits
    assert 0 <= index < batch <= frame_rows, message
    if batch == frame_rows:
        assert index == row, message
