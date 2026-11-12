"""Run every example script, because nothing did.

``polars-cv/examples/`` holds fourteen files that no test, workflow or lint
step touched: ``ruff``'s ``extend-exclude`` skips the directory, and CI never
executed them. An example calling a renamed method, or passing a parameter that
was removed, would have stayed broken indefinitely while reading as
documentation — the removed ``rasterize(anti_alias=)`` is exactly the kind of
surface an example can outlive.

Marked ``slow``: thirteen subprocesses, each decoding images and rendering
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

from ._discovery import (
    _NON_SCRIPT_EXAMPLES,
    doc_pages,
    example_files,
    example_scripts,
)
from .conftest import plugin_required

if TYPE_CHECKING:
    pass


#: Scripts needing arguments beyond their defaults. Empty: every example runs
#: bare today, and ``06_detection_metrics.py``'s ``argparse`` options all carry
#: defaults. An example that grows a *required* argument belongs here rather
#: than being dropped from the sweep.
_EXTRA_ARGS: dict[str, list[str]] = {}

#: The two allowlist guards below are ordinary structural checks and belong in
#: the pre-commit lane. Only the runner itself is slow, and it carries its own
#: `slow` mark — a module-level `slow` here deselected the guards from
#: `-m "structural and not slow"` along with it.
pytestmark = pytest.mark.structural


@plugin_required
@pytest.mark.slow
@pytest.mark.parametrize(
    "script",
    example_scripts(),
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


def test_non_script_exemptions_are_real_and_needed() -> None:
    """``_NON_SCRIPT_EXAMPLES`` may not hide a runnable example.

    It is an allowlist that removes files from the sweep, so it is the way this
    runner gets quietly narrowed — add an entry and a broken example stops
    being checked, with nothing to say so. ``_DISCOVERY_EXEMPT`` in
    ``test_discovery_scope.py`` carries the same risk and the same guard.

    Two directions:

    * an entry naming a file that does not exist is stale, and would silently
      cover a *future* file of that name;
    * an entry naming a file with a ``__main__`` block is excusing something
      runnable, which is what the exemption is not for.
    """
    present = {p.name for p in example_files()}

    stale = sorted(name for name in _NON_SCRIPT_EXAMPLES if name not in present)
    assert not stale, (
        f"_NON_SCRIPT_EXAMPLES names files that do not exist: {stale}. Remove "
        f"them; a stale entry silently exempts a future file of the same name."
    )

    by_name = {p.name: p for p in example_files()}
    runnable = sorted(
        name for name in _NON_SCRIPT_EXAMPLES if "__main__" in by_name[name].read_text()
    )
    assert not runnable, (
        f"these are exempted as non-scripts but have a __main__ block: "
        f"{runnable}. They are runnable, so the sweep should run them."
    )


def test_every_example_is_either_run_or_exempted() -> None:
    """No example may fall out of the sweep unaccounted for.

    ``example_scripts()`` is the swept set and ``_NON_SCRIPT_EXAMPLES`` is the
    excused set; together they must be every ``.py`` in ``examples/``. Without
    this, a discovery change that quietly stopped matching some files would
    shrink the sweep while every remaining case still passed.
    """
    present = {p.name for p in example_files()}
    swept = {p.name for p in example_scripts()}

    unaccounted = sorted(present - swept - set(_NON_SCRIPT_EXAMPLES))
    assert not unaccounted, (
        f"these examples are neither run nor exempted: {unaccounted}. Either "
        f"the sweep should run them or they belong in _NON_SCRIPT_EXAMPLES "
        f"with a reason."
    )
    assert swept, "example_scripts() found nothing; the sweep is checking nothing"


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
