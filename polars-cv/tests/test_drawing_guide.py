"""Every example in the drawing guide runs (CR-101).

``docs/user-guide/drawing.md`` promises that each example runs as written:
each fenced ``python`` block is executed here on its own, in a fresh
namespace, so an API change that breaks the guide fails this test rather than
a reader.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import plugin_required

GUIDE = Path(__file__).resolve().parent.parent / "docs" / "user-guide" / "drawing.md"
BLOCKS = re.findall(r"^```python\n(.*?)^```", GUIDE.read_text(), re.S | re.M)


def test_the_guide_has_examples() -> None:
    assert len(BLOCKS) >= 5, f"found {len(BLOCKS)} python blocks in {GUIDE}"


@plugin_required
@pytest.mark.parametrize("index", range(len(BLOCKS)))
def test_each_example_runs(index: int) -> None:
    exec(compile(BLOCKS[index], f"{GUIDE.name}[{index}]", "exec"), {})
