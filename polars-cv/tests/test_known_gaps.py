"""Executable specifications for defects that are verified but not yet fixed.

Every test in this module is ``xfail(strict=True)``. Each one:

* describes a defect that has been **confirmed against running code or source**,
  not a suspicion;
* asserts the behaviour the codebase *should* have, so it fails today; and
* names, in its docstring, what "fixed" looks like.

``strict=True`` is the point. When someone lands the fix, the test XPASSes and
the suite goes **red** — which is the signal to delete the marker and let the
test join the suite proper. A backlog that lives in prose is a backlog that
rots; this one cannot silently become stale in either direction. It is the same
reasoning as `AGENTS.md`'s "The Single-Authority Refactor: What Was Done, What
Is Left" section, made executable.

These are the items recorded as deferred in that review: the ones where a fix is
a design change rather than a correction, and so wants its own commit. Adding a
test here is not a way to avoid fixing something — it is a way to stop the
knowledge evaporating between sessions.

Do **not** put a flaky or environment-dependent test here. `xfail` marks "known
broken", never "sometimes fails".
"""

from __future__ import annotations

import pytest

# Each gap carries its own lane: a source scan is `structural` (pre-commit runs
# it with no compiled extension), a runtime one `plugin_required`. No gap is
# open: the planned-size defects closed in PLANNER_SIZES_PLAN.md S1 and S2.


def _gap(reason: str) -> pytest.MarkDecorator:
    """Mark a verified, unfixed defect. Strict, so a fix fails the suite.

    Only an ``AssertionError`` is the expected failure: every gap states the
    defect as an ``assert``, so a gap that breaks for any other reason (a
    renamed helper, a fixture that no longer builds) fails the suite rather
    than reading as the defect.
    """
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)
