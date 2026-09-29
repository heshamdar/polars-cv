"""The divergences the parity sweeps found and have not fixed, as repros.

Each entry of ``framework/known.DIVERGENCES`` runs here as a strict ``xfail``:
its repro asserts the *correct* behaviour, so it fails today. When the defect
is fixed the repro passes, ``strict=True`` turns the XPASS into a failure,
and the entry has to be deleted — which also deletes the predicate that was
withholding the comparison from the sweeps, putting the fixed path back under
test. This is the same mechanism as ``tests/test_known_gaps.py``, keyed to the
registry the sweeps consult rather than to a second list.
"""

from __future__ import annotations

import pytest

from tests.conftest import plugin_required
from tests.parity.framework.known import DIVERGENCES, Divergence


@plugin_required
@pytest.mark.parametrize(
    "divergence",
    [
        pytest.param(
            d,
            id=d.key,
            marks=pytest.mark.xfail(strict=True, reason=d.summary),
        )
        for d in DIVERGENCES
    ],
)
def test_known_divergence_still_reproduces(divergence: Divergence) -> None:
    """The repro asserts the fixed behaviour; it must still fail."""
    divergence.repro()


def test_every_divergence_is_described_and_scoped() -> None:
    """An entry must say what is wrong, what fixed looks like, and what it
    covers — a registry entry that covers nothing withholds nothing and would
    never be noticed going stale."""
    keys = [d.key for d in DIVERGENCES]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
    for d in DIVERGENCES:
        assert "Fixed" in d.summary, f"{d.key}: say what 'fixed' looks like"
        scopes = (d.affects_axes, d.affects_step, d.affects_chain)
        assert any(s is not None for s in scopes), f"{d.key}: no predicate"
