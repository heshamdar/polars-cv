"""Rows are named only through ``polars-cv/src/row_ref.rs``.

Under the streaming engine a plugin call is one batch and cannot know where
it starts in the frame; an error that wrote ``(row {})`` itself named a batch
index as if it were the frame's row. The checker is
:func:`tests._row_ref_scan.row_ref_offenders`, run here over committed
fixtures and over the plugin's source.
"""

from __future__ import annotations

import pytest

from tests._discovery import rust_sources, rust_src_dir
from tests._row_ref_scan import row_ref_offenders

pytestmark = pytest.mark.structural


@pytest.mark.parametrize(
    "src",
    [
        'polars_err!(ComputeError: "{} (row {})", e, row)',
        'polars_bail!(ComputeError: "x must be > 0 at row {}", row);',
        'format!("order repeats entry {} in row {}; ...", index, row)',
        'return Err(format!("row {row_idx} is past the end"));',
        'polars_ensure!(ok, ComputeError: "row {} has shape {:?}", i, s);',
        # Across lines, with a nested call before the string.
        'polars_bail!(ComputeError:\n    "{}: got {} (row {})", f(a, b), v, i);',
    ],
)
def test_a_hand_rolled_row_is_rejected(src: str) -> None:
    assert row_ref_offenders(src, "x.rs"), src


@pytest.mark.parametrize(
    "src",
    [
        "crate::row_ref::at_row(msg, row_idx, state.rows)",
        'polars_err!(ComputeError: "{}", at_row(e, i, len))',
        # Test assertions are not errors a user reads.
        'assert_eq!(got, want, "row {i}");',
        # "row" without a placeholder, and a placeholder without "row".
        'format!("a row holds {} values", n)',
        'polars_bail!(ComputeError: "the arrow {} is bad", x)',
        # A `\`-newline continuation is part of its string, not its end.
        # (Misread, the quotes paired off wrongly and the code after it,
        # `= row {`, read as a string.)
        'polars_err!(ComputeError: "no \\\n row", d);\nif let Some(v) = row { return; }\nf("x");',
        # A parenthesis inside a string does not end the macro early.
        'format!("(", x); polars_err!(ComputeError: "fine {}", v)',
    ],
)
def test_the_one_rendering_and_unrelated_text_are_accepted(src: str) -> None:
    assert row_ref_offenders(src, "x.rs") == [], src


def test_rows_are_named_only_through_row_ref() -> None:
    src = rust_src_dir()
    found = [
        hit
        for path in rust_sources()
        if path.is_relative_to(src) and path.name != "row_ref.rs"
        for hit in row_ref_offenders(path.read_text(), str(path.relative_to(src)))
    ]
    assert not found, "errors naming a row by hand:\n  " + "\n  ".join(found)
