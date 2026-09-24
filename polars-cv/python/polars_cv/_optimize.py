"""Plan-time optimization control for polars-cv pipelines.

Optimization is an explicit *phase*, not something that leaks into other
stages. Construction builds the logical graph and does nothing else;
serialization emits whatever graph it is handed and does nothing else; the one
:meth:`PipelineGraph.optimize` call between them rewrites the logical graph into
an equivalent physical graph.

Which optimizations exist is declared in Rust — the logical passes by
``LogicalPass`` (``src/passes.rs``), the engine's by ``engine_passes!``
(view-buffer, which also declares ``OptConfig``) — and reaches Python as the
generated ``PASS_CATALOG``, from which this module builds
:data:`OPTIMIZATION_PASSES` and the control surface that toggles them,
:class:`OptFlags`. It spans both tiers: the logical (Tier-1) passes applied by
``PipelineGraph.optimize``, and the per-row engine-lowering rewrites (scalar fusion, cast
elimination, flip/transpose algebra) that run in the Rust engine on the
*already optimized* graph. Each pass carries a ``tier`` saying how it is
toggled; engine flags ride to Rust in the graph's ``opt`` object. Mandatory
correctness lowering (materialization, the f64 fusion exclusion) is not an
optimization and carries no toggle.

Every pass is output-preserving and **byte-identical** when toggled: each
rewrites or shares a computation without changing a single output byte, and the
differential-equivalence guard holds every registered pass to byte-equality. An
earlier ``affine_fusion`` pass only preserved output approximately and was
removed for changing pixels — see the CHANGELOG.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from polars_cv._ops_generated import PASS_CATALOG, _OptFlagFields

#: Environment variable supplying the global default when ``.sink()`` is called
#: without an explicit ``opt_flags=``. Mirrors Polars' per-query-flags + config
#: model. See :meth:`OptFlags.from_env` for the accepted grammar.
OPT_ENV_VAR = "POLARS_CV_OPTIMIZATIONS"


@dataclass(frozen=True)
class PassSpec:
    """One optimization pass, as the Rust pass catalogue describes it.

    Attributes:
        name: Stable identifier; also the :class:`OptFlags` field name (both
            are generated from the catalogue).
        summary: One-line description for ``explain`` output and docs.
        tier: ``"logical"`` — applied by ``PipelineGraph.optimize``, rewriting
            the logical graph before serialization (CSE in Python, which holds
            the expression identities; the node-scope passes in Rust,
            ``node_pass``); or ``"engine"`` — a per-row lowering rewrite in the
            Rust engine, whose flag is serialized into the graph's ``opt``
            object and gates the matching ``OptConfig`` field. Mandatory
            correctness lowering (materialization, the f64 fusion exclusion) is
            not an optimization and has no pass.
    """

    name: str
    summary: str
    tier: str


#: Every optimization pass, in the order they apply.
OPTIMIZATION_PASSES: tuple[PassSpec, ...] = tuple(
    PassSpec(name=name, summary=summary, tier=tier)
    for name, tier, summary in PASS_CATALOG
)

#: Pass names in declaration order (both tiers).
PASS_NAMES: tuple[str, ...] = tuple(p.name for p in OPTIMIZATION_PASSES)

#: Engine-tier pass names — the fields of the Rust ``OptConfig`` serialized into
#: the graph's ``opt`` object (``engine_passes!`` declares both, and
#: ``OptConfig`` refuses an unknown key).
ENGINE_PASS_NAMES: tuple[str, ...] = tuple(
    p.name for p in OPTIMIZATION_PASSES if p.tier == "engine"
)

#: Logical-tier pass names — applied by ``PipelineGraph.optimize``.
LOGICAL_PASS_NAMES: tuple[str, ...] = tuple(
    p.name for p in OPTIMIZATION_PASSES if p.tier == "logical"
)


@dataclass(frozen=True)
class OptFlags(_OptFlagFields):
    """Which optimization passes run, one boolean per pass (both tiers).

    The fields are generated from the pass catalogue, one per pass. Construct
    directly (``OptFlags(scalar_fusion=False)``), with the :meth:`all`/:meth:`none`
    shorthands, or from the environment via :meth:`from_env`. Defaults are
    all-on, matching the always-on behaviour that predated the explicit phase.
    """

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
