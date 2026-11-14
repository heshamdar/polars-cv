"""The shape a ``CHANGELOG.md`` entry may take, checked by ``test_changelog_style.py``.

The changelog is read by users deciding whether and how to upgrade. It grew to
400 KB of design notes once, because nothing bounded an entry; these rules are
that bound. They are the authority: the prose rule in the root ``AGENTS.md``
("Changelog") states the same limits and points here.

What is checked, between the file's preamble (everything above the first
``## `` heading) and its footer (everything below a ``---`` rule: the
release link references), in which a release heading is an error:

- a release heading is ``## [Unreleased]`` or ``## [X.Y.Z] — YYYY-MM-DD``;
- a release's summary (prose between its heading and its first section) is
  at most :data:`MAX_SUMMARY_WORDS` words;
- a section is one of :data:`SECTIONS`, each at most once and in that order;
- every bullet, with its continuation lines, is at most
  :data:`MAX_BULLET_WORDS` words, and a section holds nothing but bullets;
- no in-repo ledger id (``CR-12``, ``FU-3``) appears: open work lives in
  GitHub issues, and a ledger id resolves to nothing.

Limits: words are whitespace-separated tokens, so a long code span counts as
one word. A bullet ends at a blank line or at the next bullet, so a bullet
continued after a blank line is reported as prose outside a bullet.
"""

from __future__ import annotations

import re

#: Sections a release may have, in the order they must appear.
SECTIONS: tuple[str, ...] = (
    "Breaking changes",
    "Security",
    "Added",
    "Changed",
    "Fixed",
    "Performance",
    "Removed",
)

#: Upper bound on one bullet, continuation lines included.
MAX_BULLET_WORDS = 60

#: Upper bound on a release's summary paragraph(s), migration link included.
MAX_SUMMARY_WORDS = 90

_RELEASE = re.compile(r"^## \[(?:Unreleased|\d+\.\d+\.\d+)\](?: — \d{4}-\d{2}-\d{2})?$")
_LEDGER_ID = re.compile(r"\b(?:CR|FU)-\d+\b")


def releases(text: str) -> list[str]:
    """Every release heading (``## ...`` line) in ``text``, in order."""
    return [line for line in text.splitlines() if line.startswith("## ")]


def problems(text: str) -> list[str]:
    """Every way ``text`` departs from the changelog's shape, one line each."""
    found: list[str] = []
    release: str | None = None
    section: str | None = None
    section_index = -1
    summary_words = 0
    bullet: list[str] = []
    footer = False

    def where() -> str:
        return f"{release or '(preamble)'} / {section or '(summary)'}"

    def flush() -> None:
        if bullet:
            words = len(" ".join(bullet).split())
            if words > MAX_BULLET_WORDS:
                found.append(
                    f"{where()}: a bullet of {words} words (max {MAX_BULLET_WORDS}): "
                    f"{bullet[0][:70]!r}"
                )
            bullet.clear()

    def close_summary() -> None:
        if release is not None and summary_words > MAX_SUMMARY_WORDS:
            found.append(
                f"{release}: a summary of {summary_words} words (max {MAX_SUMMARY_WORDS})"
            )

    for line in text.splitlines():
        if footer:
            if line.startswith("## "):
                found.append(f"{line!r}: a release heading below the footer rule (---)")
            continue
        if line == "---" and release is not None:
            footer = True
            continue
        if line.startswith("## "):
            flush()
            if section is None:
                close_summary()
            release, section, section_index, summary_words = line, None, -1, 0
            if not _RELEASE.match(line):
                found.append(
                    f"{line!r}: a release heading is '## [Unreleased]' or "
                    "'## [X.Y.Z] — YYYY-MM-DD'"
                )
            continue
        if release is None:
            continue  # the preamble
        for match in _LEDGER_ID.finditer(line):
            found.append(
                f"{where()}: ledger id {match.group()} (link a GitHub issue, or drop it)"
            )
        if line.startswith("### "):
            flush()
            if section is None:
                close_summary()
            name = line[4:].strip()
            section = name
            if name not in SECTIONS:
                found.append(
                    f"{where()}: unknown section (allowed: {', '.join(SECTIONS)})"
                )
                continue
            index = SECTIONS.index(name)
            if index <= section_index:
                found.append(
                    f"{where()}: repeated or out of order (order: {', '.join(SECTIONS)})"
                )
            section_index = max(section_index, index)
        elif line.startswith("- "):
            flush()
            if section is None:
                found.append(f"{where()}: a bullet before the release's first section")
            bullet.append(line)
        elif line.startswith("  ") and bullet:
            bullet.append(line)
        elif not line.strip():
            flush()
        elif section is None:
            summary_words += len(line.split())
        else:
            flush()
            found.append(f"{where()}: prose outside a bullet: {line[:70]!r}")
    flush()
    if section is None:
        close_summary()
    return found
