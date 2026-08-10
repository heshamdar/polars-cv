"""One way for a generator to format what it emits.

Every generator in this directory writes a checked-in Python file that a
regenerate-and-diff test compares byte for byte against a fresh render, so the
render has to be formatted exactly the way `ruff format` would leave the
committed file — otherwise the diff test reports "out of date" for a file that
is perfectly current.

That makes formatting part of the generated artefact, not a cosmetic step, and
it has two consequences this module exists to enforce:

- **One resolution strategy.** `gen_lazy_stub.py` looked ruff up with
  ``shutil.which`` (it is a dev dependency, so it is on the venv's path);
  `gen_dtype_names.py` shelled out to ``uvx ruff`` instead, which resolves
  differently and can want the network. Two spellings of one fact, and the
  second one silently produced a *different* answer wherever ``uvx`` could not
  run.
- **No silent fallback.** Returning the unformatted source when ruff is
  missing does not degrade gracefully: it makes the diff test fail claiming the
  file is stale, and then makes the fix its message recommends — rerun the
  generator — write an unformatted file that ``ruff format --check`` rejects.
  A missing formatter is reported as a missing formatter.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class RuffUnavailable(RuntimeError):
    """`ruff` could not be run, so the generated text cannot be formatted.

    Raised rather than falling back to unformatted output: the caller's diff
    test would otherwise report a current file as stale, which sends the reader
    to the wrong problem.
    """


def _ruff_executable() -> str:
    """Locate ruff: the active environment first, then this interpreter's.

    ``shutil.which`` finds it in an activated venv or on ``PATH``; the second
    candidate covers ``python scripts/... `` run against a venv interpreter
    directly, without activation.
    """
    found = shutil.which("ruff")
    if found:
        return found
    beside_python = Path(sys.executable).parent / "ruff"
    if beside_python.exists():
        return str(beside_python)
    msg = (
        "ruff is not available, so generated output cannot be formatted the "
        "way the committed file is. It is a dev dependency: run "
        "`uv sync --group dev`, or run this generator through "
        "`uv run --directory polars-cv`."
    )
    raise RuffUnavailable(msg)


def ruff_format(text: str, *, filename: str) -> str:
    """Format *text* as ruff would format a file called *filename*.

    Args:
        text: The generated source.
        filename: The name ruff should assume, which selects its rules — a
            ``.pyi`` stub is formatted differently from a ``.py`` module.

    Returns:
        The formatted source.

    Raises:
        RuffUnavailable: If ruff cannot be found or exits non-zero. Both are
            reported rather than swallowed, so a formatting problem never
            reaches the caller disguised as a stale file.
    """
    ruff = _ruff_executable()
    try:
        done = subprocess.run(
            [ruff, "format", "--stdin-filename", filename, "-"],
            input=text,
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError as e:
        msg = f"could not run {ruff}: {e}"
        raise RuffUnavailable(msg) from e
    except subprocess.CalledProcessError as e:
        msg = f"{ruff} failed to format the generated {filename}: {e.stderr}"
        raise RuffUnavailable(msg) from e
    return done.stdout
