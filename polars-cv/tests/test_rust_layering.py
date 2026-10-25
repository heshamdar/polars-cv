"""The Rust crates' module cycles may not grow, and a fixed one must leave the baseline.

Rust accepts module cycles, so ``cargo`` never reports them. ``tests/_rust_layering.py``
reads the source and finds them. The cycles that exist today are recorded in
``tests/rust_layering_baseline.txt``; this test fails when that set changes in
either direction. A new cycle is a layering regression. A cycle that disappeared
must also leave the file, so the baseline only ever shrinks. See
``tests/test_rust_layering_fixtures.py`` for the cases the scanner must reject
and accept.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._discovery import REPO_ROOT, rust_sources, rust_src_dir
from tests._rust_layering import rust_cycles

pytestmark = pytest.mark.structural

ROOTS = {
    "view-buffer": REPO_ROOT / "view-buffer" / "src",
    "polars-cv": rust_src_dir(),
}
BASELINE = Path(__file__).with_name("rust_layering_baseline.txt")


def _baseline() -> set[str]:
    return {
        line
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }


def test_rust_module_cycles_match_the_baseline() -> None:
    found = set(rust_cycles(ROOTS, rust_sources()))
    known = _baseline()
    new = sorted(found - known)
    gone = sorted(known - found)
    assert not new and not gone, (
        "Rust module cycles changed. A new cycle is a layering regression: move the "
        "shared item into a module both sides can import. A cycle that no longer "
        "exists must be deleted from tests/rust_layering_baseline.txt.\n"
        + "".join(f"  new:  {cycle}\n" for cycle in new)
        + "".join(f"  gone: {cycle}\n" for cycle in gone)
    )
