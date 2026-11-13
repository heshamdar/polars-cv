"""Every sink a message, docstring or guide names is a sink that exists.

CR-81 replaced the `numpy`, `torch` and `ndarray` sinks with `tensor`, and
three user-facing errors went on telling users to "use a `numpy` sink", which
the wire then refused as an unknown format. The sink names come from the
catalogue (`SinkFormat`, generated from Rust's `Sink`), so this reads them
from there rather than from a list of its own.

A source scan, because the property is about prose: it finds a sink named as
``sink("x")`` (also escaped inside a Rust string) and as `` `x` sink`` or
`` `x`/`y` sinks``. It does not find a sink named in plain words ("the x
sink") or in any other spelling, and it skips `tests/` (which names unknown
sinks on purpose) and the migration guides (which name removed ones). The
fixtures below hold both patterns to what they must and must not report.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from polars_cv._ops_generated import SinkFormat
from tests._discovery import (
    REPO_ROOT,
    doc_pages,
    example_files,
    package_modules,
    repo_file,
    requires_checkout,
    rust_sources,
)

pytestmark = [pytest.mark.structural, requires_checkout]

ROOT = REPO_ROOT
SINKS = frozenset(s.value for s in SinkFormat)

_CALL = re.compile(r"""sink\(\\?["'](\w+)\\?["']""")
_NAMED = re.compile(r"((?:`\w+`/)*`\w+`) sinks?\b(?!-)")


def unknown_sinks(text: str) -> list[tuple[int, str]]:
    """``(line, name)`` for each sink *text* names that the catalogue lacks."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        names = [m.group(1) for m in _CALL.finditer(line)]
        for m in _NAMED.finditer(line):
            names += re.findall(r"`(\w+)`", m.group(1))
        found += [(number, n) for n in names if n not in SINKS]
    return found


def _scanned() -> list[Path]:
    pages = [p for p in doc_pages() if not p.name.startswith("migration-")]
    return [
        *rust_sources(),
        *package_modules(),
        *pages,
        *example_files(),
        repo_file("polars-cv/README.md"),
    ]


def test_the_scan_covers_the_files_that_name_sinks() -> None:
    names = {p.name for p in _scanned()}
    assert {"image.rs", "encode.rs", "decode.rs", "lazy.py", "README.md"} <= names


def test_every_named_sink_exists() -> None:
    offenders = [
        f"{path.relative_to(ROOT)}:{line}: {name}"
        for path in _scanned()
        for line, name in unknown_sinks(path.read_text())
    ]
    assert not offenders, "names a sink the catalogue does not have:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize(
    "bad",
    [
        # The three messages CR-81 left behind, as they read.
        'a `numpy`/`list`/`array` sink for non-image data.",',
        '(e.g. `source(\\"image_bytes\\")`), or use a `numpy` sink, \\',
        '"expected the tensor struct, as sink(\\"numpy\\"), sink(\\"ndarray\\")',
        "df.select(expr.sink('torch'))",
        "the `ndarray` sinks write",
    ],
)
def test_a_removed_sink_is_reported(bad: str) -> None:
    assert unknown_sinks(bad)


@pytest.mark.parametrize(
    "good",
    [
        'a `tensor`/`list`/`array` sink for non-image data.",',
        'expr.sink("tensor", dtype="f16")',
        'sink(\\"png\\")',
        "the `numpy` package",  # not a sink
        "`numpy_output_dtype` sink-side",  # an identifier, not `x` sink
    ],
)
def test_a_real_sink_or_other_prose_is_not_reported(good: str) -> None:
    assert not unknown_sinks(good)
