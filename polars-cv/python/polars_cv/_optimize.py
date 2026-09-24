"""Plan-time optimization control for polars-cv pipelines.

Optimization is an explicit *phase*, not something that leaks into other
stages. Construction builds the logical graph and does nothing else;
serialization emits whatever graph it is handed and does nothing else; the one
:meth:`PipelineGraph.optimize` call between them rewrites the logical graph into
an equivalent physical graph.

This module owns the **single authority** for which optimizations exist —
:data:`OPTIMIZATION_PASSES` — and the control surface that toggles them,
:class:`OptFlags`. It spans both tiers: the logical (Tier-1) passes applied by
``PipelineGraph.optimize``, and the per-row engine-lowering rewrites (scalar fusion, cast
elimination, flip/transpose algebra) that run in the Rust engine on the
*already optimized* graph. Each pass carries a ``tier`` saying how it is
toggled; engine flags ride to Rust in the graph's ``opt`` object. Mandatory
correctness lowering (materialization, the f64 fusion exclusion) is not an
optimization and carries no toggle.

Every pass is output-preserving, and every pass is also **byte-identical** when
toggled: each rewrites or shares a computation without changing a single output
byte. :attr:`PassSpec.bit_exact` records this (it is ``True`` for every current
pass) so the differential-equivalence guard can hold each pass to byte-equality;
the field exists so a future pass that only preserves output approximately (an
interpolation-fusing pass, say) must declare itself and be tested within a
tolerance rather than silently. An earlier ``affine_fusion`` pass was such a
pass and was removed precisely because it changed pixels — see the CHANGELOG.
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
    tier: str


#: The single authority: which optimizations exist, across both tiers. Adding a
#: pass here and adding the matching :class:`OptFlags` field is one act — see the
#: parity guard in ``tests/test_optimize.py``.
#:
#: ``tier`` says *how* a pass is toggled:
#:
#: - ``"logical"`` passes are applied by ``PipelineGraph.optimize`` (they
#:   rewrite the logical graph before serialization): CSE in Python, which holds
#:   the expression identities, and the node-scope ones in Rust (``node_pass``,
#:   ``src/passes.rs``), whose ``LogicalPass`` names them all. When their flag
#:   is off the pass simply is not applied.
#: - ``"engine"`` passes are the per-row lowering rewrites in the Rust engine
#:   (``ViewExpr::optimize_with``). Their flag is *serialized* into the graph's
#:   ``opt`` object and gates the matching field of the Rust ``OptConfig`` — the
#:   engine field name equals the pass name, so they cannot drift.
#:
#: Every pass is output-preserving and, at present, byte-identical when toggled;
#: ``bit_exact`` records that so the equivalence guard holds each to byte-equality
#: (and a future approximate pass must declare ``bit_exact=False``).
#:
#: Mandatory correctness lowering (contiguity materialization, stride-preserving
#: views, the f64 fusion exclusion) is deliberately *not* here: it is not
#: optional, so it carries no toggle.
OPTIMIZATION_PASSES: tuple[PassSpec, ...] = (
    PassSpec(
        name="common_subexpression_elimination",
        summary=(
            "Share a common leading op run across sibling pipelines that read "
            "the same source column into one upstream node."
        ),
        bit_exact=True,
        tier="logical",
    ),
    PassSpec(
        name="identity_elimination",
        summary=(
            "Delete no-op operations — a zero pad, a same-dtype cast, a "
            "full-frame crop — that preserve their input byte for byte."
        ),
        bit_exact=True,
        tier="logical",
    ),
    PassSpec(
        name="spatial_window_pushdown",
        summary=(
            "Hoist a spatial window (a crop/ROI) earlier past ops it commutes "
            "with, so upstream ops process fewer pixels."
        ),
        bit_exact=True,
        tier="logical",
    ),
    PassSpec(
        name="cast_chain_collapse",
        summary=(
            "Drop a redundant intermediate cast from a cast chain when the "
            "intermediate dtype losslessly holds the input and dropping it keeps "
            "the final cast's conversion (a narrowing intermediate quantizes, and "
            "a float between an integer input and an integer target saturates, "
            "so both are kept)."
        ),
        bit_exact=True,
        tier="engine",
    ),
    PassSpec(
        name="cast_identity",
        summary="Drop a cast whose target dtype already equals its input dtype.",
        bit_exact=True,
        tier="engine",
    ),
    PassSpec(
        name="view_flip_involution",
        summary="Cancel two adjacent flips over the same axes (flip∘flip = id).",
        bit_exact=True,
        tier="engine",
    ),
    PassSpec(
        name="view_transpose_merge",
        summary="Merge two adjacent transposes into one (or into the identity).",
        bit_exact=True,
        tier="engine",
    ),
    PassSpec(
        name="scalar_fusion",
        summary=(
            "Fuse adjacent scalar/compute ops into one kernel (f64 chains stay "
            "unfused — a mandatory precision guard, not this toggle)."
        ),
        bit_exact=True,
        tier="engine",
    ),
)

#: Pass names in declaration order (both tiers).
PASS_NAMES: tuple[str, ...] = tuple(p.name for p in OPTIMIZATION_PASSES)

#: Engine-tier pass names — the fields of the Rust ``OptConfig`` serialized into
#: the graph's ``opt`` object. Their names must equal the Rust field names.
ENGINE_PASS_NAMES: tuple[str, ...] = tuple(
    p.name for p in OPTIMIZATION_PASSES if p.tier == "engine"
)

#: Logical-tier pass names — applied by ``PipelineGraph.optimize``.
LOGICAL_PASS_NAMES: tuple[str, ...] = tuple(
    p.name for p in OPTIMIZATION_PASSES if p.tier == "logical"
)


@dataclass(frozen=True)
class OptFlags:
    """Which optimization passes run, one boolean per pass (both tiers).

    Construct directly (``OptFlags(scalar_fusion=False)``), with the
    :meth:`all`/:meth:`none` shorthands, or from the environment via
    :meth:`from_env`. Defaults are all-on, matching the always-on behaviour that
    predated the explicit phase.
    """

    # Logical tier (applied by PipelineGraph.optimize).
    common_subexpression_elimination: bool = True
    identity_elimination: bool = True
    spatial_window_pushdown: bool = True
    # Engine tier (serialized to the graph's `opt` object; gates Rust OptConfig).
    cast_chain_collapse: bool = True
    cast_identity: bool = True
    view_flip_involution: bool = True
    view_transpose_merge: bool = True
    scalar_fusion: bool = True

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

    def engine_opt(self) -> "dict[str, bool]":
        """The engine-tier toggles, keyed by their Rust ``OptConfig`` field name.

        Serialized as the graph's ``opt`` object; the key set equals
        :data:`ENGINE_PASS_NAMES`, which equals the Rust ``OptConfig`` fields.
        """
        return {name: bool(getattr(self, name)) for name in ENGINE_PASS_NAMES}

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

        Examples: ``"all"``, ``"none"``, ``"scalar_fusion"`` (only that one on),
        ``"all,-scalar_fusion"`` (all but that one).

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
