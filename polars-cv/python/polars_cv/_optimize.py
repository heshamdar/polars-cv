"""Plan-time optimization control for polars-cv pipelines.

Optimization is an explicit *phase*, not something that leaks into other
stages. Construction builds the logical graph and does nothing else;
serialization emits whatever graph it is handed and does nothing else; the one
:meth:`PipelineGraph.optimize` call between them rewrites the logical graph into
an equivalent physical graph.

This module owns the **single authority** for which logical (Tier-1) passes
exist — :data:`LOGICAL_PASSES` — and the control surface that toggles them,
:class:`OptFlags`. Per-row engine lowering (scalar fusion, cast elimination,
flip/transpose algebra) lives in the Rust engine and runs on the *already
optimized* graph; it is not part of this tier and not toggled here.

Every pass is output-preserving, but not every pass is *bit*-exact: CSE shares
an identical computation, so toggling it cannot change a single byte, whereas
affine fusion replaces a run of warps with one composed warp — mathematically
the same transform, but one interpolation pass instead of several, so the pixels
can differ. :attr:`PassSpec.bit_exact` records that distinction so the
differential-equivalence guard tests each pass against the right standard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

#: Environment variable supplying the global default when ``.sink()`` is called
#: without an explicit ``opt_flags=``. Mirrors Polars' per-query-flags + config
#: model. See :meth:`OptFlags.from_env` for the accepted grammar.
OPT_ENV_VAR = "POLARS_CV_OPTIMIZATIONS"


@dataclass(frozen=True)
class PassSpec:
    """One logical optimization pass.

    Attributes:
        name: Stable identifier; also the :class:`OptFlags` field name. The
            flag↔pass parity guard rejects a pass without a matching field or a
            field without a pass, so this table and ``OptFlags`` cannot drift.
        summary: One-line description for ``explain`` output and docs.
        bit_exact: Whether toggling the pass is guaranteed byte-identical. When
            ``False`` the pass preserves the transform but may round pixels
            differently, so the equivalence guard checks it within tolerance
            rather than for byte-equality.
    """

    name: str
    summary: str
    bit_exact: bool


#: The single authority: which logical optimizations exist. Adding a pass here
#: and adding the matching :class:`OptFlags` field is one act — see the parity
#: guard in ``tests/test_optimize.py``.
LOGICAL_PASSES: tuple[PassSpec, ...] = (
    PassSpec(
        name="common_subexpression_elimination",
        summary=(
            "Share a common leading op run across sibling pipelines that read "
            "the same source column into one upstream node."
        ),
        bit_exact=True,
    ),
    PassSpec(
        name="affine_fusion",
        summary=(
            "Collapse a run of static affine ops (warp_affine and non-90° "
            "rotate) into a single composed warp_affine."
        ),
        bit_exact=False,
    ),
)

#: Pass names in declaration order.
PASS_NAMES: tuple[str, ...] = tuple(p.name for p in LOGICAL_PASSES)


@dataclass(frozen=True)
class OptFlags:
    """Which logical optimization passes run, one boolean per pass.

    Construct directly (``OptFlags(affine_fusion=False)``), with the
    :meth:`all`/:meth:`none` shorthands, or from the environment via
    :meth:`from_env`. Defaults are all-on, matching the always-on behaviour that
    predated the explicit phase.
    """

    common_subexpression_elimination: bool = True
    affine_fusion: bool = True

    @classmethod
    def all(cls) -> "OptFlags":
        """Every pass enabled."""
        return cls(**{name: True for name in PASS_NAMES})

    @classmethod
    def none(cls) -> "OptFlags":
        """Every pass disabled — serialize the logical graph verbatim."""
        return cls(**{name: False for name in PASS_NAMES})

    def enabled(self, name: str) -> bool:
        """Is the named pass on? Raises ``KeyError`` for an unknown pass."""
        if name not in PASS_NAMES:
            msg = f"Unknown optimization pass: {name!r}. Known passes: {list(PASS_NAMES)}."
            raise KeyError(msg)
        return bool(getattr(self, name))

    @classmethod
    def from_env(cls) -> "OptFlags":
        """Read the global default from :data:`OPT_ENV_VAR`.

        An unset or blank value means all-on (the omitted-keyword default).
        Otherwise the value is parsed by :meth:`parse`.
        """
        raw = os.environ.get(OPT_ENV_VAR)
        if raw is None or not raw.strip():
            return cls.all()
        return cls.parse(raw)

    @classmethod
    def parse(cls, raw: str) -> "OptFlags":
        """Parse a flag string into :class:`OptFlags`.

        Grammar (comma-separated, whitespace-insensitive):

        - ``all`` / ``none`` as the first token sets the base; otherwise the
          base is all-off.
        - a bare pass name turns that pass on; ``-name`` turns it off.

        Examples: ``"all"``, ``"none"``, ``"affine_fusion"`` (only that one on),
        ``"all,-affine_fusion"`` (all but that one).

        An unknown pass name raises ``ValueError`` rather than being silently
        ignored — an unrecognised flag is a bug, not a no-op.
        """
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if tokens and tokens[0] in ("all", "none"):
            enabled = {name: tokens[0] == "all" for name in PASS_NAMES}
            tokens = tokens[1:]
        else:
            enabled = {name: False for name in PASS_NAMES}
        for tok in tokens:
            off = tok.startswith("-")
            name = tok[1:] if off else tok
            if name not in PASS_NAMES:
                msg = (
                    f"Unknown optimization pass: {name!r} (in {raw!r}). "
                    f"Known passes: {list(PASS_NAMES)}."
                )
                raise ValueError(msg)
            enabled[name] = not off
        return cls(**enabled)


def resolve_opt_flags(value: "OptFlags | bool | None") -> OptFlags:
    """Coerce a public ``opt_flags=`` argument into :class:`OptFlags`.

    ``None`` defers to :meth:`OptFlags.from_env`; ``True``/``False`` are the
    all/none shorthands; an :class:`OptFlags` passes through.
    """
    if value is None:
        return OptFlags.from_env()
    if value is True:
        return OptFlags.all()
    if value is False:
        return OptFlags.none()
    if isinstance(value, OptFlags):
        return value
    msg = f"opt_flags must be an OptFlags, bool, or None; got {type(value).__name__}."
    raise TypeError(msg)


def _field_names() -> tuple[str, ...]:
    """OptFlags boolean field names — used by the parity guard."""
    return tuple(f.name for f in fields(OptFlags))
