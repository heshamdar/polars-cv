"""Error models: how far an output may sit from its reference, and why.

A single ``atol`` cannot check a chain. ``grayscale`` agrees with OpenCV to
±1, and that is a fine per-op tolerance — but feed it to ``threshold`` and a
pixel one level either side of the cut moves by 255. So every reference entry
carries two numbers, not one:

* its own error (:class:`Tol`) — how far the op's output may sit from the
  reference *on the same input*; and
* its **gain** — how much it may amplify an error already present in its
  input (1 for resampling and data movement, ``|factor|`` for ``scale``,
  ``inf`` for anything discontinuous: ``threshold``, ``canny``, ``round``).

:func:`propagate` combines them along a chain. The chain suites use the result
for the *end-to-end* comparison. The primary check does not need it: each
step is compared against the reference applied to the engine's own previous
output, where the incoming error is zero by construction (see
``oracle/test_oracle_chains.py``).

The propagation rules are deliberately conservative bounds, not estimates. A
bound that is too loose still catches a wrong kernel (those differ by far more
than a few levels); one that is too tight fails on correct code and gets
loosened by whoever hits it. Its arithmetic is pinned by committed fixtures in
``meta/test_parity_framework_fixtures.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np


def _premultiplied(x: np.ndarray) -> np.ndarray:
    """Colour times alpha over the dtype's maximum; alpha as it is.

    For ``[H, W, 2|4]`` (gray+alpha, RGBA). The representation a resampler
    with alpha works in, and the only one where a bound on it is dense: back
    in straight colour each side's rounding is multiplied by ``MAX / alpha``.
    """
    if x.ndim != 3 or x.shape[2] not in (2, 4):
        msg = f"premultiplied space needs [H, W, 2|4] with alpha last, got {x.shape}"
        raise ValueError(msg)
    top = float(np.iinfo(x.dtype).max) if x.dtype.kind in "iu" else 1.0
    out = x.astype(np.float64)
    out[..., :-1] *= out[..., -1:] / top
    return out


#: The representations a bound may be stated in, by name (see ``Tol.space``).
SPACES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "premultiplied": _premultiplied,
}


@dataclass(frozen=True)
class Tol:
    """Allowed deviation of an output from its reference.

    Every element must satisfy ``|a - e| <= atol + rtol * |e|``, except that a
    fraction ``frac`` of the elements may instead reach ``frac_atol``. The
    sparse allowance is for kernels whose disagreement is confined to a few
    pixels (an interpolation boundary, a rotation's anti-aliased edge): a
    dense ``atol`` large enough to cover those would hide a kernel that is
    wrong everywhere by that much.

    ``space`` names the representation the bound holds in (one of
    :data:`SPACES`; ``None``: the output as it is). :func:`compare` maps both
    sides into it first. A translucent resample is compared premultiplied,
    where it is well-conditioned, rather than given a straight-colour
    allowance loose enough to hide a wrong kernel.
    """

    atol: float = 0.0
    rtol: float = 0.0
    frac: float = 0.0
    frac_atol: float = 0.0
    space: str | None = None

    def __post_init__(self) -> None:
        if self.space is not None and self.space not in SPACES:
            msg = f"unknown comparison space {self.space!r}; known: {sorted(SPACES)}"
            raise ValueError(msg)
        if min(self.atol, self.rtol, self.frac, self.frac_atol) < 0:
            msg = f"tolerances are non-negative: {self}"
            raise ValueError(msg)
        if self.frac > 1:
            msg = f"frac is a fraction of elements: {self}"
            raise ValueError(msg)
        if self.frac and self.frac_atol < self.atol:
            msg = f"the sparse bound cannot be tighter than the dense one: {self}"
            raise ValueError(msg)

    @property
    def is_exact(self) -> bool:
        """True when nothing may differ at all."""
        return self == EXACT

    def __repr__(self) -> str:
        if self.is_exact:
            return "EXACT"
        parts = [f"atol={self.atol:g}"]
        if self.rtol:
            parts.append(f"rtol={self.rtol:g}")
        if self.frac:
            parts.append(f"frac={self.frac:g}@{self.frac_atol:g}")
        if self.space:
            parts.append(f"in {self.space} space")
        return f"Tol({', '.join(parts)})"


EXACT = Tol()


def lsb(k: float = 1) -> Tol:
    """Within *k* units of the last place of an integer output."""
    return Tol(atol=k)


def close(atol: float = 0.0, rtol: float = 1e-6) -> Tol:
    """A float output: an absolute floor and a relative tolerance."""
    return Tol(atol=atol, rtol=rtol)


def sparse(atol: float, frac: float, frac_atol: float) -> Tol:
    """Dense *atol*, with up to *frac* of the elements allowed *frac_atol*."""
    return Tol(atol=atol, frac=frac, frac_atol=frac_atol)


#: How an op moves values, which decides how an incoming error propagates.
#:
#: * ``movement`` — copies values to new positions (crop, flip, pad, channel
#:   selection). Exact, and adds no rounding of its own.
#: * ``pointwise`` — each output element depends on the same input element
#:   only, so a sparse error stays exactly as sparse.
#: * ``spatial`` — each output mixes a neighbourhood (resample, filter), which
#:   can spread a sparse error, or (through a crop of the result) concentrate
#:   it, so a sparse incoming error becomes unbounded.
#: * ``global`` — depends on whole-image statistics (histogram equalization,
#:   min/max normalization); any incoming error is unbounded.
KINDS = ("movement", "pointwise", "spatial", "global")


def propagate(
    incoming: Tol | None,
    own: Tol,
    gain: float,
    *,
    kind: str,
    integer_out: bool,
) -> Tol | None:
    """The tolerance after one step, or ``None`` when no bound holds.

    Args:
        incoming: The tolerance on the step's input (``None``: unbounded).
        own: The step's error on an exact input.
        gain: The step's Lipschitz bound (``math.inf`` if discontinuous).
        kind: One of :data:`KINDS`.
        integer_out: Whether the step's output dtype is an integer. Rounding
            an inexact value can move it one further level than the input
            error alone (``|round(x) - round(y)| <= |x - y| + 1``).

    Rules, in order:

    1. An unbounded input stays unbounded.
    2. An exact input carries only the step's own error.
    3. A discontinuous step (infinite gain), or a ``global`` one, turns any
       inexact input unbounded.
    4. A sparse input survives only ``movement``/``pointwise`` steps.
    5. A bound stated in another space (``Tol.space``) does not compose
       with errors in the output's own: over an exact input it stands as
       the step's own, otherwise no bound holds.
    6. Otherwise errors add: ``gain * incoming + own``, plus one level of
       re-rounding for an integer output that is not pure data movement.
    """
    if kind not in KINDS:
        msg = f"unknown step kind {kind!r}; expected one of {KINDS}"
        raise ValueError(msg)
    if incoming is None:
        return None
    if incoming.is_exact:
        return own
    if math.isinf(gain) or kind == "global":
        return None
    if incoming.frac and kind not in ("movement", "pointwise"):
        return None
    if incoming.space or own.space:
        return None

    requant = 1.0 if integer_out and kind != "movement" else 0.0
    atol = gain * incoming.atol + own.atol + requant
    rtol = incoming.rtol + own.rtol
    frac = min(1.0, incoming.frac + own.frac)
    if frac:
        frac_atol = (
            gain * max(incoming.frac_atol, incoming.atol)
            + max(own.frac_atol, own.atol)
            + requant
        )
    else:
        frac_atol = 0.0
    return Tol(atol=atol, rtol=rtol, frac=frac, frac_atol=frac_atol)


@dataclass(frozen=True)
class Mismatch:
    """Why an output did not match its reference."""

    reason: str

    def __str__(self) -> str:
        return self.reason


def compare(actual: np.ndarray, expected: np.ndarray, tol: Tol) -> Mismatch | None:
    """``None`` if *actual* matches *expected* within *tol*, else the reason.

    Shape and dtype must match exactly — a reference declares its output
    dtype independently of the planner, so a dtype disagreement is a finding,
    not something to tolerate. NaNs must sit in the same places, and
    infinities must match exactly.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return Mismatch(f"shape {actual.shape} != reference {expected.shape}")
    if actual.dtype != expected.dtype:
        return Mismatch(f"dtype {actual.dtype} != reference {expected.dtype}")
    if actual.size == 0:
        return None
    if tol.space is not None:
        to_space = SPACES[tol.space]
        actual, expected = to_space(actual), to_space(expected)

    a = actual.astype(np.float64) if actual.dtype.kind != "O" else actual
    e = expected.astype(np.float64)
    if actual.dtype.kind in "iu" and actual.dtype.itemsize == 8:
        # float64 cannot hold every 64-bit integer; compare those exactly
        # through Python ints when the tolerance is zero.
        if tol.is_exact:
            if np.array_equal(actual, expected):
                return None
            where = np.argwhere(actual != expected)[0]
            return Mismatch(
                f"{np.count_nonzero(actual != expected)} of {actual.size} elements "
                f"differ; first at {tuple(where)}: {actual[tuple(where)]} vs "
                f"{expected[tuple(where)]}"
            )

    nan_a, nan_e = np.isnan(a), np.isnan(e)
    if not np.array_equal(nan_a, nan_e):
        where = tuple(np.argwhere(nan_a != nan_e)[0])
        return Mismatch(
            f"NaN placement differs at {where}: {a[where]} vs reference {e[where]}"
        )
    inf_a, inf_e = np.isinf(a), np.isinf(e)
    if not np.array_equal(np.where(inf_a, a, 0), np.where(inf_e, e, 0)):
        where = tuple(np.argwhere((inf_a | inf_e) & (a != e))[0])
        return Mismatch(
            f"infinity differs at {where}: {a[where]} vs reference {e[where]}"
        )

    finite = ~(nan_a | inf_a | inf_e)
    with np.errstate(invalid="ignore", over="ignore"):
        diff = np.where(finite, np.abs(a - e), 0.0)
    allowed = tol.atol + tol.rtol * np.abs(np.where(finite, e, 0.0))
    over = diff > allowed
    n_over = int(np.count_nonzero(over))
    if n_over == 0:
        return None

    worst = tuple(np.unravel_index(int(np.argmax(diff - allowed)), diff.shape))
    detail = (
        f"worst at {worst}: {a[worst]!r} vs reference {e[worst]!r} "
        f"(|diff| {diff[worst]:g}, allowed {allowed[worst]:g})"
    )
    # Rounded up: a sparse allowance means "a few elements", and on a tiny
    # image that is at least one.
    budget = math.ceil(tol.frac * diff.size)
    if n_over > budget:
        return Mismatch(
            f"{n_over} of {diff.size} elements exceed {tol} "
            f"(sparse budget {budget}); {detail}"
        )
    sparse_allowed = tol.frac_atol + tol.rtol * np.abs(np.where(finite, e, 0.0))
    beyond = diff > sparse_allowed
    if np.any(beyond):
        worst = tuple(np.argwhere(beyond)[0])
        return Mismatch(
            f"{int(np.count_nonzero(beyond))} elements exceed even the sparse bound "
            f"{tol.frac_atol:g}; first at {worst}: {a[worst]!r} vs reference "
            f"{e[worst]!r}"
        )
    return None


def widen(tol: Tol, **changes: float) -> Tol:
    """A copy of *tol* with some fields replaced (for readable table entries)."""
    return replace(tol, **changes)
