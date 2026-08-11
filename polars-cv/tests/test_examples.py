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
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from ._discovery import example_scripts
from .conftest import plugin_required

if TYPE_CHECKING:
    from pathlib import Path


#: Scripts needing arguments beyond their defaults. Empty: every example runs
#: bare today, and ``06_detection_metrics.py``'s ``argparse`` options all carry
#: defaults. An example that grows a *required* argument belongs here rather
#: than being dropped from the sweep.
_EXTRA_ARGS: dict[str, list[str]] = {}


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
