"""
Guards against the two ways a reported version can be wrong.

1. **Drift.** The version is recorded in four hand-edited places that must agree.
   `CONTRIBUTING.md` lists them and notes that nothing checks them; these tests are
   that check.
2. **Staleness.** The install is editable, so the Python sources under test are
   always the working tree's — but the compiled extension is not. After a `git
   pull` that touches Rust, `_lib.abi3.so` keeps its build-time version until
   `maturin develop` is re-run, and the suite silently exercises old Rust against
   new Python. Comparing `__version__` against the extension and the installed
   distribution turns that into a test failure instead of a wrong answer.
"""

import sys
from pathlib import Path

import pytest
import tomllib

import polars_cv
from tests._discovery import CHECKOUT_MARKERS, requires_checkout
from tests.conftest import plugin_required

#: Every test here is a structural guard: it checks the *shape* of the codebase
#: -- registries, authorities, removed surfaces, documented vocabularies --
#: rather than the numerical behaviour of a pipeline. `-m structural` is the
#: lane pre-commit runs; see `tests/AGENTS.md`. Note that the lane as a whole
#: does need the compiled extension: many structural facts are only observable
#: through the FFI, and those tests fail rather than skip without it.
pytestmark = pytest.mark.structural

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every file that records the version, and the TOML path to it within that file.
# The keys are the same manifests `_discovery.CHECKOUT_MARKERS` uses to decide
# whether a source checkout is present — asserted below rather than restated, so
# a manifest added to one list cannot go missing from the other.
VERSION_SOURCES = {
    "polars-cv/pyproject.toml": ("project", "version"),
    "polars-cv/Cargo.toml": ("package", "version"),
    "view-buffer/Cargo.toml": ("package", "version"),
}


def test_version_sources_are_the_checkout_markers() -> None:
    """The two lists naming the manifests are one fact, not two.

    ``_discovery.CHECKOUT_MARKERS`` decides whether the repository is present;
    ``VERSION_SOURCES`` decides which manifests carry the version. They have
    always held the same three paths, and a fourth manifest added to one and
    not the other would either go unversioned or make every source-scanning
    guard skip. Pinned in both directions rather than kept in step by hand.
    """
    assert set(VERSION_SOURCES) == set(CHECKOUT_MARKERS)


def _declared_version(relative_path: str, keys: tuple[str, ...]) -> str:
    document = tomllib.loads((REPO_ROOT / relative_path).read_text())
    for key in keys:
        document = document[key]
    assert isinstance(document, str)
    return document


@requires_checkout
@pytest.mark.parametrize(("relative_path", "keys"), list(VERSION_SOURCES.items()))
def test_declared_version_matches_dunder_version(
    relative_path: str, keys: tuple[str, ...]
) -> None:
    """Each manifest agrees with `polars_cv.__version__`."""
    declared = _declared_version(relative_path, keys)
    assert declared == polars_cv.__version__, (
        f"{relative_path} declares {declared!r} but "
        f"polars_cv.__version__ is {polars_cv.__version__!r}. "
        "See the release checklist in CONTRIBUTING.md — all of them must agree."
    )


@requires_checkout
def test_crates_are_versioned_together() -> None:
    """`polars-cv` and `view-buffer` are released as a pair."""
    plugin = _declared_version("polars-cv/Cargo.toml", ("package", "version"))
    engine = _declared_version("view-buffer/Cargo.toml", ("package", "version"))
    assert plugin == engine


def test_build_info_reports_every_channel() -> None:
    """`build_info()` surfaces all three versions, not just the one it can't get wrong."""
    info = polars_cv.build_info()
    assert set(info) == {
        "version",
        "plugin_version",
        "dist_version",
        # The pair that can actually detect staleness within a release cycle;
        # the versions above cannot. See
        # `test_compiled_plugin_matches_the_rust_sources`.
        "plugin_source_hash",
        "source_hash",
    }
    # `version` is read straight off the module, so asserting it matches
    # `__version__` proves nothing. The other two are what can disagree.
    assert info["plugin_version"] is not None or info["dist_version"] is not None, (
        "build_info() found neither a compiled plugin nor installed distribution "
        "metadata — it cannot detect staleness in this environment"
    )


@plugin_required
def test_compiled_plugin_is_not_stale() -> None:
    """
    The compiled extension was built from this Python source.

    A mismatch means `_lib.abi3.so` predates the checkout. The Python sources being
    imported are the working tree's either way (the install is editable), so the
    suite would go on exercising old Rust against new Python and the failures would
    point anywhere but here. Rebuild with `maturin develop`.
    """
    info = polars_cv.build_info()
    assert info["plugin_version"] == info["version"], (
        f"compiled plugin is {info['plugin_version']!r} but the Python source is "
        f"{info['version']!r} — the installed package is stale, "
        "re-run `maturin develop`"
    )


@requires_checkout
@plugin_required
def test_compiled_plugin_matches_the_rust_sources() -> None:
    """The extension was built from the Rust now on disk.

    This is the check that actually detects staleness.
    ``test_compiled_plugin_is_not_stale`` above compares release *versions*,
    and both sides read the same ``Cargo.toml`` literal — so they agree
    throughout a release cycle, which is exactly the window in which Rust gets
    edited without a rebuild. It can only ever fire across a version bump.

    ``build.rs`` bakes a hash of both crates' sources into the extension and
    ``build_info()`` recomputes it from the working tree, so any edit to a
    ``.rs`` file, a crate manifest, or the lockfile makes them differ until
    ``maturin develop`` is re-run. That matters because 53% of the suite is
    gated on a ``.so`` merely existing, of any age — a stale one does not skip
    those tests, it runs them against old Rust and reports pass.
    """
    info = polars_cv.build_info()
    assert info["source_hash"] is not None, (
        "the working tree's Rust sources could not be hashed, so this guard "
        "checked nothing -- `_source_hash_from_tree` found no crate manifest"
    )
    assert info["plugin_source_hash"] is not None, (
        "the compiled extension carries no __source_hash__; it predates "
        "build.rs. Re-run `maturin develop`."
    )
    assert info["plugin_source_hash"] == info["source_hash"], (
        f"the compiled extension was built from different Rust sources than "
        f"the ones in this checkout (built {info['plugin_source_hash']}, "
        f"working tree {info['source_hash']}) — re-run `maturin develop`, or "
        f"the suite will exercise the old extension and report pass."
    )


def test_installed_distribution_is_not_stale() -> None:
    """The installed distribution metadata matches the imported source."""
    info = polars_cv.build_info()

    if info["dist_version"] is None:
        pytest.skip("polars-cv is not installed as a distribution")

    assert info["dist_version"] == info["version"], (
        f"installed distribution is {info['dist_version']!r} but the imported "
        f"source is {info['version']!r} — re-run `maturin develop`"
    )


@requires_checkout
def test_imported_package_is_the_source_tree() -> None:
    """
    `polars_cv` resolves to this checkout's source, not to a copy of it.

    This project installs editable — `.venv` carries a `.pth` pointing at
    `polars-cv/python`, and `maturin develop` drops `_lib.abi3.so` into that same
    directory — so the `.py` files under test are always the ones in the working
    tree. If that ever stops holding, edits to the Python source silently stop
    affecting test runs, and the failure looks like a mysteriously unfixable bug.
    """
    imported = Path(polars_cv.__file__).resolve().parent
    source = (REPO_ROOT / "polars-cv" / "python" / "polars_cv").resolve()

    assert imported == source, (
        f"polars_cv is imported from {imported}, not the source tree at {source}. "
        f"Editing the Python sources will not affect this run. "
        f"sys.path[0] is {sys.path[0]!r}"
    )
