"""The divergences the parity sweeps found and have not fixed, as repros.

Each entry of ``framework/known.DIVERGENCES`` runs here: its repro asserts the
*correct* behaviour, so it fails today, and ``known.still_reproduces`` requires
it to fail with the entry's own ``raises`` and ``match``. When the defect is
fixed the repro passes and this test fails, so the entry has to be deleted —
which also deletes the predicate that was withholding the comparison from the
sweeps, putting the fixed path back under test. A repro that fails for another
reason (a renamed helper, a changed fixture) fails here too, rather than
reading as the defect. This is the mechanism of ``tests/test_known_gaps.py``
(a strict ``xfail`` pinned to ``AssertionError``), pinned to the message as
well, and keyed to the registry the sweeps consult rather than to a second
list.
"""

from __future__ import annotations

import pytest

from tests.conftest import plugin_required
from tests.parity.framework.known import DIVERGENCES, Divergence, still_reproduces


@plugin_required
@pytest.mark.parametrize("divergence", [pytest.param(d, id=d.key) for d in DIVERGENCES])
def test_known_divergence_still_reproduces(divergence: Divergence) -> None:
    """The repro asserts the fixed behaviour; it must still fail, for its
    defect."""
    still_reproduces(divergence)


def test_every_divergence_is_described_and_scoped() -> None:
    """An entry must say what is wrong, what fixed looks like, and what it
    covers — a registry entry that covers nothing withholds nothing and would
    never be noticed going stale."""
    keys = [d.key for d in DIVERGENCES]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
    for d in DIVERGENCES:
        assert "Fixed" in d.summary, f"{d.key}: say what 'fixed' looks like"
        assert d.match, f"{d.key}: pin the failure with a message fragment"
        scopes = (d.affects_axes, d.affects_step, d.affects_chain)
        assert any(s is not None for s in scopes), f"{d.key}: no predicate"
