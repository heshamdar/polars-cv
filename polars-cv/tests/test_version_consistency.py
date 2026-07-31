"""
Guards against the two ways a reported version can be wrong.

1. **Drift.** The version is recorded in four hand-edited places that must agree.
   `CONTRIBUTING.md` lists them and notes that nothing checks them; these tests are
   that check.
2. **Staleness.** `maturin develop` installs a *copy* of the Python sources next to
   the compiled extension, so after a `git pull` an unrebuilt environment keeps
   reporting the old version — and, more importantly, keeps running the old code.
   Comparing `__version__` against the extension and the installed distribution
   turns that into a test failure instead of a silent wrong answer.
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


def test_build_info_reports_the_python_version() -> None:
    info = polars_cv.build_info()
    assert info["version"] == polars_cv.__version__
    assert set(info) == {"version", "plugin_version", "dist_version"}


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


def test_imported_package_is_the_one_under_test() -> None:
    """
    `polars_cv` resolves to this checkout, or to an install built from it.

    Catches the case where a site-packages copy shadows the source tree entirely.
    """
    if not _running_from_checkout():
        pytest.skip("not running from a source checkout")

    imported = Path(polars_cv.__file__).resolve().parent
    source = REPO_ROOT / "polars-cv" / "python" / "polars_cv"

    if imported == source:
        return

    # Installed elsewhere (a copy): the version check above is what guards it, but
    # make the situation visible when it fails.
    assert polars_cv.__version__ == _declared_version(
        "polars-cv/pyproject.toml", ("project", "version")
    ), (
        f"polars_cv is imported from {imported} (not {source}) and its version "
        f"does not match this checkout — the install is stale. "
        f"sys.path[0] is {sys.path[0]!r}"
    )
