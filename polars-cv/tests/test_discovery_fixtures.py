"""Fixtures for the discovery mechanism, which is what stops vacuous guards.

``tests/AGENTS.md`` requires a guard with non-trivial logic to carry committed
known-bad and known-good inputs, exercising *the same helper the real guard
calls* — a fixture testing a copy proves nothing about the guard. Everything
here imports from :mod:`tests._discovery` for that reason.

The property under test is narrow and load-bearing: an empty result must raise,
because every structural guard in this suite is "find some files, then assert
something about them", and a find that returns nothing makes the assertion hold
for free. Two guards in this repo shipped that way.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._discovery import (
    CHECKOUT_MARKERS,
    SUITE_MODULE_EXCLUSIONS,
    EmptyDiscovery,
    discovered,
    in_checkout,
    package_modules,
    rust_sources,
    rust_src_dir,
    suite_files,
    suite_modules,
)


class TestDiscoveredRefusesNothing:
    """The known-bad and known-good inputs to :func:`discovered`."""

    @pytest.mark.parametrize("empty", [[], (), set(), iter(())])
    def test_an_empty_result_raises(self, empty: object) -> None:
        """Every empty spelling raises, not just the literal list."""
        with pytest.raises(EmptyDiscovery, match="would have passed"):
            discovered(empty, "widgets")  # type: ignore[arg-type]

    def test_the_failure_names_what_was_missing(self) -> None:
        """A discovery failure must say what it was looking for.

        Otherwise the failure reads as a bug in the code under test rather
        than in the scan, which is how a broken scan gets 'fixed' by weakening
        the guard that depends on it.
        """
        with pytest.raises(EmptyDiscovery, match="found no Rust sources"):
            discovered([], "Rust sources")

    def test_a_non_empty_result_passes_through_unchanged(self) -> None:
        """The known-good case: real input is returned as a list, in order."""
        assert discovered(iter([3, 1, 2]), "numbers") == [3, 1, 2]

    def test_a_single_item_is_enough(self) -> None:
        """Non-empty is the bar, not some threshold.

        A count threshold is what made an earlier ratchet in this suite
        self-concealing — damage that grew past the threshold started passing.
        """
        assert discovered([0], "numbers") == [0]


class TestEveryAccessorFindsSomething:
    """Each accessor returns real files here, so the guards are not skipping."""

    def test_this_is_a_source_checkout(self) -> None:
        """If this fails, every source-scanning guard below is skipping."""
        assert in_checkout(), (
            f"expected the manifests {CHECKOUT_MARKERS} to be present; the "
            f"source-scan guards all skip without them"
        )

    def test_rust_src_dir_resolves(self) -> None:
        assert (rust_src_dir() / "lib.rs").is_file()

    def test_rust_sources_excludes_build_output(self) -> None:
        """Build output and vendored crates are not this repo's code.

        The dtype ratchet was reading a Rust template inside ``.venv`` before
        the exclusion existed — an offender reported in code nobody here can
        fix teaches the next reader to ignore the guard.
        """
        paths = rust_sources()
        assert paths
        assert not [p for p in paths if "/target/" in str(p) or "/.venv/" in str(p)]

    def test_package_modules_finds_the_package(self) -> None:
        names = {p.name for p in package_modules()}
        assert {"pipeline.py", "lazy.py", "_types.py"} <= names

    def test_suite_files_includes_the_files_suite_modules_holds_back(self) -> None:
        """The unfiltered set is genuinely unfiltered.

        The guards that police the suite itself read ``suite_files``; if the
        exclusions leaked into it, those two files would become a blind spot.
        """
        everything = {p.name for p in suite_files()}
        filtered = {p.name for p in suite_modules()}
        assert set(SUITE_MODULE_EXCLUSIONS) <= everything
        assert not set(SUITE_MODULE_EXCLUSIONS) & filtered
        assert filtered < everything

    def test_every_exclusion_carries_a_reason(self) -> None:
        blank = [
            name for name, why in SUITE_MODULE_EXCLUSIONS.items() if not why.strip()
        ]
        assert not blank, f"exclusions without a reason: {blank}"


class TestRustSrcDirFailsRatherThanSkips:
    """A moved layout must fail, because a skip switches five guards off."""

    def test_a_missing_crate_directory_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Point discovery at an empty tree: it must raise, not return None.

        This is the regression the module exists for. The previous
        ``_rust_src_dir()`` answered a missing layout with ``None``, and five
        guards read that as "installed wheel, nothing to check" — so a package
        move would have silently disabled all five at once while the suite
        stayed green.
        """
        import tests._discovery as discovery

        monkeypatch.setattr(discovery, "REPO_ROOT", tmp_path)
        with pytest.raises(EmptyDiscovery, match="no Rust crate sources"):
            discovery.rust_src_dir()

    def test_an_empty_tree_makes_rust_sources_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import tests._discovery as discovery

        monkeypatch.setattr(discovery, "REPO_ROOT", tmp_path)
        with pytest.raises(EmptyDiscovery, match="found no Rust sources"):
            discovery.rust_sources()
