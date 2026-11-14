"""``CHANGELOG.md`` stays a short, user-facing record (rules: ``_changelog_style``).

The fixtures pin the checker on known-good and known-bad snippets; the last
test holds the real file to it. The fixtures call the same function the guard
calls, so a checker that stopped matching would fail here first.
"""

from __future__ import annotations

import pytest

from ._changelog_style import MAX_BULLET_WORDS, problems, releases
from ._discovery import repo_file

pytestmark = pytest.mark.structural

_GOOD = """\
# Changelog

Preamble prose is not checked, however long it runs on and on and on.

## [Unreleased]

### Added

- `reduce(statistic, axis=)` takes any statistic or percentile.

## [0.41.0] — 2026-10-23

Reductions and filters. Upgrading from 0.40: see the
[migration guide](https://example.org/migration-0.41/).

### Breaking changes

- `normalize(method="preset")` is now `normalize_preset(mean=, std=)`;
  results are unchanged.

### Fixed

- `rotate(expand=True)` clipped the corners of odd-sized images.

---

[Unreleased]: https://example.org/compare/v0.41.0...HEAD
A footer of link references and notes is not checked, however long it runs.
"""


def test_a_well_formed_changelog_passes() -> None:
    assert problems(_GOOD) == []
    assert releases(_GOOD) == ["## [Unreleased]", "## [0.41.0] — 2026-10-23"]


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n### Added\n\n- "
            + "word " * (MAX_BULLET_WORDS + 1),
            "a bullet of",
            id="long-bullet",
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n### Added\n\n- short start\n"
            + "  continued " * MAX_BULLET_WORDS,
            "a bullet of",
            id="long-bullet-over-continuation-lines",
        ),
        pytest.param("## 1.0.0\n", "release heading", id="bad-heading"),
        pytest.param(
            "## [1.0.0] - 2026-01-01\n", "release heading", id="hyphen-not-em-dash"
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n### Internal\n\n- a refactor\n",
            "unknown section",
            id="unknown-section",
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n### Fixed\n\n- a\n\n### Added\n\n- b\n",
            "out of order",
            id="sections-out-of-order",
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n### Added\n\n- a\n\n### Added\n\n- b\n",
            "out of order",
            id="repeated-section",
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n### Fixed\n\n- the CR-76 box\n",
            "ledger id CR-76",
            id="ledger-id",
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n" + "summary " * 91 + "\n\n### Added\n\n- a\n",
            "a summary of 91 words",
            id="long-summary",
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n### Added\n\n- a\n\nA design note.\n",
            "prose outside a bullet",
            id="prose-in-section",
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n- a\n",
            "before the release's first section",
            id="bullet-without-section",
        ),
        pytest.param(
            "## [1.0.0] — 2026-01-01\n\n### Added\n\n- a\n\n---\n\n## [0.9.0] — 2025-12-01\n",
            "below the footer rule",
            id="release-hidden-below-a-rule",
        ),
    ],
)
def test_a_departure_is_reported(snippet: str, expected: str) -> None:
    found = problems(snippet)
    assert any(expected in p for p in found), found


def test_the_changelog_follows_its_style() -> None:
    text = repo_file("CHANGELOG.md").read_text(encoding="utf-8")
    # The check must have something to check: every release is a heading.
    headings = releases(text)
    assert headings[0] == "## [Unreleased]", headings[:1]
    assert len(headings) >= 30, f"only {len(headings)} release headings parsed"
    found = problems(text)
    assert not found, (
        f"CHANGELOG.md departs from its style in {len(found)} place(s) "
        "(rules: tests/_changelog_style.py; guidance: root AGENTS.md, 'Changelog'):\n"
        + "\n".join(found[:40])
    )
