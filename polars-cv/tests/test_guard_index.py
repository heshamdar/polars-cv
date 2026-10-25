"""The canonical-paths table is the index of guards, checked both ways (#189).

``_guard_index.py`` states what is parsed and its limits; the fixtures below
pin the parser on known-bad and known-good snippets.
"""

from __future__ import annotations

import pytest

from tests import _guard_index as gi
from tests._discovery import (
    package_modules,
    repo_file,
    requires_checkout,
    rust_sources,
    script_files,
    suite_files,
)

pytestmark = pytest.mark.structural


def _suite() -> gi.Suite:
    python_tests: set[str] = set()
    words: set[str] = set()
    files: list[str] = []
    for path in suite_files():
        source = path.read_text()
        python_tests |= gi.python_test_names(source)
        words |= gi.words(source)
        files.append(path.as_posix())
    rust_tests: set[str] = set()
    for path in rust_sources():
        source = path.read_text()
        rust_tests |= gi.rust_test_names(source)
        words |= gi.words(source)
        files.append(path.as_posix())
    files += [path.as_posix() for path in [*package_modules(), *script_files()]]
    return gi.Suite(
        python_tests=frozenset(python_tests),
        rust_tests=frozenset(rust_tests),
        words=frozenset(words),
        files=tuple(files),
    )


def _index() -> list[str]:
    return gi.cited(gi.guard_column(repo_file("AGENTS.md").read_text()))


@requires_checkout
def test_every_guard_the_index_names_exists() -> None:
    missing = gi.unresolved(_index(), _suite())
    assert not missing, (
        "the canonical-paths table in AGENTS.md names guards that do not "
        f"exist (renamed or deleted?): {missing}"
    )


@requires_checkout
def test_every_source_scanning_test_file_is_in_the_index() -> None:
    scanners = {
        path.name: source
        for path in suite_files()
        if path.name.startswith("test_") and gi.scans_source(source := path.read_text())
    }
    assert scanners, "found no source-scanning test files: the marker list is stale"
    missing = gi.uncited_scanners(scanners, _index())
    assert not missing, (
        "these test files read source to enforce a rule, but the canonical-"
        "paths table in AGENTS.md cites neither the file nor any of its tests; "
        f"add the rule's row (or cite the guard in its row): {missing}"
    )


# --- Fixtures: the parser itself ----------------------------------------------

_TABLE = """\
# Doc

## Canonical Paths

Intro.

| Fact | Single authority | Rejection mechanism |
|------|------------------|---------------------|
| A | `x` | `test_present` and `tests/test_file.py::test_in_file` |
| B | `y \\| z` | `rust_guard_that_exists`, `deny_unknown_fields` |

## Next
| ignored | row | `test_not_in_the_index` |
"""

_FIXTURE_SUITE = gi.Suite(
    python_tests=frozenset({"test_present", "test_in_file"}),
    rust_tests=frozenset({"rust_guard_that_exists"}),
    words=frozenset({"rust_guard_that_exists", "deny_unknown_fields"}),
    files=("/r/polars-cv/tests/test_file.py",),
)


def test_the_parser_reads_only_the_sections_third_column() -> None:
    tokens = gi.cited(gi.guard_column(_TABLE))
    assert tokens == [
        "test_present",
        "tests/test_file.py",
        "test_in_file",
        "rust_guard_that_exists",
        "deny_unknown_fields",
    ]


def test_a_table_that_parses_to_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="no rows"):
        gi.guard_column("## Canonical Paths\n\nNo table here.\n\n## Next\n")
    with pytest.raises(ValueError, match="section"):
        gi.guard_column("# Nothing\n")


def test_the_fixture_index_resolves() -> None:
    assert gi.unresolved(gi.cited(gi.guard_column(_TABLE)), _FIXTURE_SUITE) == []


@pytest.mark.parametrize(
    "token",
    ["test_renamed_away", "TestRenamedAway", "test_gone.py", "rust_guard_renamed_away"],
)
def test_a_guard_that_names_nothing_is_reported(token: str) -> None:
    assert gi.unresolved([token], _FIXTURE_SUITE) == [token]


def test_a_cited_helper_or_short_name_is_not_a_guard_claim() -> None:
    # `deny_unknown_fields` exists as a word; two-word names and code are skipped.
    assert (
        gi.unresolved(["deny_unknown_fields", "map_new", "Plan.push"], _FIXTURE_SUITE)
        == []
    )


def test_a_scanner_is_cited_by_file_or_by_one_of_its_tests() -> None:
    scanner = "import ast\n\ndef test_holds_the_rule():\n    pass\n"
    assert gi.scans_source(scanner)
    assert not gi.scans_source("def test_plain():\n    assert 1\n")
    scanners = {"test_a.py": scanner, "test_b.py": scanner.replace("rule", "other")}
    assert gi.uncited_scanners(scanners, ["test_a.py"]) == ["test_b.py"]
    assert (
        gi.uncited_scanners(scanners, ["test_holds_the_rule", "test_holds_the_other"])
        == []
    )
    assert gi.uncited_scanners(scanners, []) == ["test_a.py", "test_b.py"]
