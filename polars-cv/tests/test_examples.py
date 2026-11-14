"""Run every example script, because nothing did.

``polars-cv/examples/`` holds fourteen files that no test, workflow or lint
step touched: ``ruff``'s ``extend-exclude`` skips the directory, and CI never
executed them. An example calling a renamed method, or passing a parameter that
was removed, would have stayed broken indefinitely while reading as
documentation — the removed ``rasterize(anti_alias=)`` is exactly the kind of
surface an example can outlive.

Marked ``slow``: one subprocess per example, each decoding images and rendering
plots, is a minute of wall clock rather than the milliseconds the fast lane is
made of. It runs in the same scheduled lane as the out-of-core tests.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ._discovery import doc_pages, example_files
from .conftest import plugin_required

if TYPE_CHECKING:
    pass


#: Scripts needing arguments beyond their defaults. Empty: every example runs
#: bare today. An example that grows a *required* argument belongs here rather
#: than being dropped from the sweep.
_EXTRA_ARGS: dict[str, list[str]] = {}

#: The guard below is an ordinary structural check and belongs in the
#: pre-commit lane. Only the runner itself is slow, and it carries its own
#: `slow` mark — a module-level `slow` here deselected the guard from
#: `-m "structural and not slow"` along with it.
pytestmark = pytest.mark.structural


@plugin_required
@pytest.mark.slow
@pytest.mark.parametrize(
    "script",
    example_files(),
    ids=lambda p: p.stem,  # type: ignore[misc]
)
def test_examples_run(script: "Path") -> None:
    """Each example must exit 0.

    Run as a subprocess rather than imported: they are scripts with
    ``__main__`` blocks and module-level state, and one raising ``SystemExit``
    or leaving a matplotlib figure open should not affect the next.

    ``MPLBACKEND=Agg`` keeps the plotting examples from wanting a display.
    Output images land in ``examples/outputs/``, which is where running an
    example puts them and which ``.gitignore`` already covers — so this asserts
    the real behaviour rather than a redirected version of it.
    """
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(script), *_EXTRA_ARGS.get(script.name, [])],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}.\n"
        f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
    )


_QUOTE = re.compile(r'--8<-- "examples/([\w.]+):(\w+)"')


def test_every_section_a_guide_quotes_is_in_its_example() -> None:
    """A guide's code is a marked section of an example, not a copy of it.

    ``docs/user-guide/augmentation.md`` and "Choosing a sink" quote
    ``examples/14_augmentation.py`` and ``15_choosing_a_sink.py`` with
    ``--8<-- "examples/<file>:<section>"``, so the runner above is what keeps
    the guide's code working. ``mkdocs build --strict`` fails on a missing
    file but renders a missing *section* as an empty block, so a renamed
    marker is checked here.
    """
    by_name = {p.name: p.read_text() for p in example_files()}
    quoted = [
        (page.name, *hit)
        for page in doc_pages()
        for hit in _QUOTE.findall(page.read_text())
    ]
    assert quoted, "no guide quotes an example: the pattern is stale"
    missing = [
        (page, name, section)
        for page, name, section in quoted
        if name not in by_name
        or f"--8<-- [start:{section}]" not in by_name[name]
        or f"--8<-- [end:{section}]" not in by_name[name]
    ]
    assert not missing, f"guides quote sections that do not exist: {missing}"
