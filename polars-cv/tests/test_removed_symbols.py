"""The removed-symbol gate: fixtures for its logic, then the real repository.

``scripts/check_removed_symbols.py`` is the exit gate each typed-op phase
extends (``TYPED_OPS_PLAN.md``). A textual scanner earns committed fixtures
(CLAUDE.md, "Guards must be watched failing"): what it must reject, what it
must not, and that an empty scan is an error rather than a pass.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.structural

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_removed_symbols.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_removed_symbols", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


crs = _module()
GONE = crs.Removed("OldThing", "fixture: it was deleted")
ALLOWED = crs.Removed(
    "OLD_ENV", "fixture", allowed_in=frozenset({"tests/tombstone.py"})
)


class TestFixtures:
    def test_a_reference_is_reported_with_its_location(self) -> None:
        hits = crs.find_hits((GONE,), {"src/a.rs": "fn f() {}\nlet x = OldThing;\n"})
        assert [(h.path, h.line, h.symbol) for h in hits] == [
            ("src/a.rs", 2, "OldThing")
        ]

    def test_a_mention_in_prose_or_a_string_is_a_reference(self) -> None:
        files = {"docs/x.md": "Use `OldThing` here.", "p.py": 'name = "OldThing"'}
        assert len(crs.find_hits((GONE,), files)) == 2

    def test_matching_is_whole_word(self) -> None:
        files = {"a.rs": "OldThingy; MyOldThing; old_thing"}
        assert crs.find_hits((GONE,), files) == []

    def test_history_files_are_not_scanned(self) -> None:
        files = {name: "OldThing" for name in crs.HISTORY_FILES}
        assert crs.find_hits((GONE,), files) == []

    def test_an_allowance_covers_only_its_own_files(self) -> None:
        files = {"tests/tombstone.py": "OLD_ENV=1", "src/lib.rs": "OLD_ENV"}
        hits = crs.find_hits((ALLOWED,), files)
        assert [h.path for h in hits] == ["src/lib.rs"]

    def test_an_empty_scan_is_an_error_not_a_pass(self) -> None:
        with pytest.raises(ValueError, match="vacuously"):
            crs.find_hits((GONE,), {})


def test_the_repository_holds_no_removed_symbol() -> None:
    hits = crs.find_hits(crs.REMOVED, crs.tracked_text_files())
    assert not hits, "\n".join(str(h) for h in hits)


def test_every_allowance_names_a_tracked_file() -> None:
    """A stale allowance (the file moved or was deleted) would silently widen."""
    tracked = crs.tracked_text_files()
    for entry in crs.REMOVED:
        for path in entry.allowed_in:
            assert path in tracked, f"{entry.symbol}: allowed_in {path} is not tracked"
