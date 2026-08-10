"""File discovery for the source-scanning guards: a set that cannot be empty.

Every structural guard in this suite works by finding a set of files and then
asserting something about their contents. That shape has one failure mode, and
this repo has shipped it: **the find returns nothing, the assertion holds
vacuously, and the guard reads as coverage forever.** Two live instances
motivated this module — ``_test_files()`` and ``_PACKAGE_MODULES`` both went
green on an empty glob, and the second is what the entire ``_push_op`` append
contract rests on.

So discovery happens here and nowhere else, and every accessor either returns a
non-empty set or raises.

The subtler failure is the *skip*. ``_rust_src_dir()`` used to answer "the Rust
sources are not here" by returning ``None``, and five guards skipped on it —
which is correct for an installed wheel and catastrophic for a moved package
layout, because the two are indistinguishable from the inside. The fix is to
key the skip on an *independent* fact: :func:`in_checkout` asks whether the
version manifests are present, and only then does :func:`rust_src_dir` insist
the crate sources exist. A layout change now fails; a wheel install still
skips.

Callers must not glob for themselves. ``test_scans_go_through_discovery`` in
``test_sanitation.py`` walks the test suite's AST and fails on a direct
``glob``/``rglob`` outside this module and the files listed in its
``_DISCOVERY_EXEMPT``, each of which is a place where finding nothing is a
meaningful answer rather than a broken scan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeVar

import pytest

import polars_cv

T = TypeVar("T")

#: Repository root: ``polars-cv/tests/_discovery.py`` -> ``../../``.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Files whose presence *defines* "running from a source checkout". Shared with
#: ``test_version_consistency.py``, which needs the same fact and used to
#: compute it separately.
CHECKOUT_MARKERS: tuple[str, ...] = (
    "polars-cv/pyproject.toml",
    "polars-cv/Cargo.toml",
    "view-buffer/Cargo.toml",
)

#: Directories that hold build output or vendored code rather than sources. A
#: scan that walks these reports offenders in code this repo does not own —
#: the dtype ratchet was reading a Rust template inside ``.venv`` before this.
_EXCLUDED_DIRS: tuple[str, ...] = ("/target/", "/.venv/", "/node_modules/")


class EmptyDiscovery(AssertionError):
    """A discovery that found nothing.

    An ``AssertionError`` rather than a bespoke exception so it reports as an
    ordinary test failure: the point is that the guard *fails*, loudly, in the
    same way it would if it had found a real offender.
    """


def discovered(items: Iterable[T], what: str) -> list[T]:
    """Return *items* as a list, refusing to hand back an empty one.

    Args:
        items: The result of a search.
        what: What was being searched for, named in the failure.

    Returns:
        The items, guaranteed non-empty.

    Raises:
        EmptyDiscovery: If the search found nothing, which means the guard
            downstream of it would have passed while checking nothing.
    """
    collected = list(items)
    if not collected:
        msg = (
            f"found no {what} — the guard reading this would have passed "
            f"vacuously. Discovery is broken, not the code under test."
        )
        raise EmptyDiscovery(msg)
    return collected


def in_checkout() -> bool:
    """True when the repository layout these guards read is actually present.

    Determined from the version manifests, deliberately *not* from anything a
    guard then goes on to scan: a fact used to excuse a missing scan must not
    be the same fact the scan is looking for.
    """
    return all((REPO_ROOT / marker).is_file() for marker in CHECKOUT_MARKERS)


#: Skip marker for guards that read repository files. The one legitimate reason
#: to skip a source scan is that there are no sources — an installed wheel.
requires_checkout = pytest.mark.skipif(
    not in_checkout(),
    reason="repository sources are not available outside a source checkout",
)


def rust_src_dir() -> Path:
    """The plugin crate's ``src/`` directory.

    Raises:
        EmptyDiscovery: In a checkout where the directory is missing. That is a
            layout change, and the five guards that read it must fail rather
            than skip — skipping is what let them all switch off at once.

    Note:
        Outside a checkout this still raises; pair it with
        :data:`requires_checkout` so the wheel case skips before calling.
    """
    src = REPO_ROOT / "polars-cv" / "src"
    if not (src / "lib.rs").exists():
        raise EmptyDiscovery(
            f"no Rust crate sources at {src} — if the layout moved, the scans "
            f"that read it are no longer scanning anything"
        )
    return src


def rust_sources() -> list[Path]:
    """Every ``.rs`` file this repository owns, build output excluded."""
    return discovered(
        sorted(
            path
            for path in REPO_ROOT.glob("**/*.rs")
            if not any(marker in str(path) for marker in _EXCLUDED_DIRS)
        ),
        "Rust sources",
    )


def package_modules() -> list[Path]:
    """Every ``.py`` module in the installed ``polars_cv`` package.

    The append-contract guard scans all of them: its first version read only
    ``pipeline.py``, and both real ``_ops`` mutations outside it sailed through.
    """
    return discovered(
        sorted(Path(polars_cv.__file__).parent.rglob("*.py")), "package modules"
    )


def suite_files() -> list[Path]:
    """Every ``.py`` file under ``tests/``, this module and ``conftest`` included.

    The unfiltered set. Guards that police the suite itself — "does anything
    glob directly?" — must see every file, including the ones
    :func:`suite_modules` filters out, or the exclusion becomes a blind spot.
    """
    return discovered(
        sorted(Path(__file__).resolve().parent.rglob("*.py")), "test suite files"
    )


#: Files :func:`suite_modules` holds back, and why. These are the files the
#: conformance guards are *looking for* — the shared fixtures live in the first
#: and the guards themselves in the second — so including them would make every
#: such guard report itself.
SUITE_MODULE_EXCLUSIONS: dict[str, str] = {
    "conftest.py": "holds the shared fixtures the conformance guards look for",
    "test_sanitation.py": "holds the guards, so it names every pattern they ban",
}


def suite_modules() -> list[Path]:
    """Test modules that the conformance guards scan for local re-definitions."""
    return discovered(
        [p for p in suite_files() if p.name not in SUITE_MODULE_EXCLUSIONS],
        "test modules",
    )
