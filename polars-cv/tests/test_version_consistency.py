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
from tests.conftest import plugin_required

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every file that records the version, and the TOML path to it within that file.
VERSION_SOURCES = {
    "polars-cv/pyproject.toml": ("project", "version"),
    "polars-cv/Cargo.toml": ("package", "version"),
    "view-buffer/Cargo.toml": ("package", "version"),
}


def _declared_version(relative_path: str, keys: tuple[str, ...]) -> str:
    document = tomllib.loads((REPO_ROOT / relative_path).read_text())
    for key in keys:
        document = document[key]
    assert isinstance(document, str)
    return document


def _running_from_checkout() -> bool:
    """True when the repository layout these tests read is actually present."""
    return all((REPO_ROOT / path).is_file() for path in VERSION_SOURCES)


requires_checkout = pytest.mark.skipif(
    not _running_from_checkout(),
    reason="version manifests are not available outside a source checkout",
)


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
    assert set(info) == {"version", "plugin_version", "dist_version"}
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

    A mismatch means the installed package predates the checkout: `maturin develop`
    copies the Python sources in, so the `.py` files being imported are also stale
    and no amount of editing the source will affect what runs. Rebuild with
    `maturin develop --release`.
    """
    info = polars_cv.build_info()
    assert info["plugin_version"] == info["version"], (
        f"compiled plugin is {info['plugin_version']!r} but the Python source is "
        f"{info['version']!r} — the installed package is stale, "
        "re-run `maturin develop --release`"
    )


def test_installed_distribution_is_not_stale() -> None:
    """The installed distribution metadata matches the imported source."""
    info = polars_cv.build_info()

    if info["dist_version"] is None:
        pytest.skip("polars-cv is not installed as a distribution")

    assert info["dist_version"] == info["version"], (
        f"installed distribution is {info['dist_version']!r} but the imported "
        f"source is {info['version']!r} — re-run `maturin develop --release`"
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
