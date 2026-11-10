"""The metrics boundary guard's logic, separated so it can be tested itself.

``polars_cv.metrics`` is a *downstream* layer: built on polars-cv the way
scikit-learn is built on NumPy, and headed for a package of its own in which
CV metrics are one offering among several. This scanner holds the boundary
that move needs, so the move is mechanical when it comes.
``test_metrics_boundary`` runs it over ``python/polars_cv``;
``test_metrics_boundary_fixtures`` runs it over committed known-bad and
known-good packages.

The tiers are the layout, not a list. Under the downstream package:

* :data:`CV_PACKAGE` (``polars_cv.metrics._cv``) is the **CV tier**: the
  matchers and entry points that read images and geometry.
* the package's own ``__init__`` is the **facade**, which re-exports both.
* every other module is **generic**: the part that needs Polars only.

So a new module is generic unless it is created inside ``_cv/``, and a
generic module cannot sit in a package whose ``__init__`` loads polars-cv.

Rules, over **every** import statement (module level, function level and
``if TYPE_CHECKING:`` alike — a deferred or type-only import is still a
dependency the split package would have to satisfy):

(a) **Core never imports metrics.** No module outside :data:`DOWNSTREAM`
    imports one inside it. ``import polars_cv`` must not load the metrics.
(b) **Generic metrics do not depend on polars-cv.** A generic module imports
    nothing from polars-cv outside the metrics package, imports neither the
    CV tier nor the facade (which loads it), and uses no polars-cv expression
    accessor (``expr.cv.…``, ``expr.contour.…`` — every namespace polars-cv
    registers with ``register_expr_namespace``, read from those calls).
(c) **The CV tier uses polars-cv's public API only**: an absolute
    ``from <surface> import name`` where the surface is in
    :data:`PUBLIC_SURFACES` and ``name`` is in that surface's ``__all__``
    (read from its source). No private module, no relative import climbing
    out of the metrics package, no bare ``import polars_cv``. The facade is
    held to the same rule.

``test_the_generic_tier_runs_without_polars_cv`` is the run-time form of (b):
it copies the generic modules out and runs them with ``polars_cv``
unimportable, which also sees what a scan cannot.

Limits, stated so nobody mistakes this for more than it is:

* It is a source scan. ``importlib.import_module``, ``__import__``,
  ``getattr(expr, "contour")`` and ``exec`` are invisible to it (the run-time
  test above covers the generic tier's).
* Accessor detection matches the attribute name: a non-polars object with an
  attribute called ``contour`` (or any registered namespace) whose own
  attribute is read would be reported too.
"""

from __future__ import annotations

import ast
from collections.abc import Collection, Mapping

from tests._import_layering import DOWNSTREAM as _CONSUMERS
from tests._import_layering import _base

ROOT = "polars_cv"

#: The downstream package: everything under it is metrics, everything else core.
#: Read from the layering guard, which exempts it from its root-import rule.
(DOWNSTREAM,) = _CONSUMERS

#: polars-cv's public import surfaces: each one's ``__all__`` is what an
#: adapter may import from it. The root, and the documented
#: ``from polars_cv.geometry import bbox_from_coords``.
PUBLIC_SURFACES: tuple[str, ...] = ("polars_cv", "polars_cv.geometry")

#: The CV tier: the one package under :data:`DOWNSTREAM` that may use
#: polars-cv. Everything else there, bar the facade, is generic (rule b).
CV_PACKAGE = f"{DOWNSTREAM}._cv"

_REGISTRAR = "register_expr_namespace"


def is_downstream(module: str, downstream: str = DOWNSTREAM) -> bool:
    """Whether ``module`` is the downstream package or inside it."""
    return module == downstream or module.startswith(downstream + ".")


def _within(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def generic_modules(
    modules: Collection[str],
    cv: str = CV_PACKAGE,
    downstream: str = DOWNSTREAM,
) -> list[str]:
    """The metrics modules held to rule (b): the part that needs Polars only.

    Raises:
        AssertionError: ``cv`` is not a package among ``modules`` (renamed or
            moved): every module would then read as generic, or none.
    """
    if cv not in modules or not is_downstream(cv, downstream):
        raise AssertionError(f"the CV tier {cv} is not a package under {downstream}")
    return sorted(
        m
        for m in modules
        if is_downstream(m, downstream) and m != downstream and not _within(m, cv)
    )


def _in_core(module: str, root: str, downstream: str) -> bool:
    return (module == root or module.startswith(root + ".")) and not is_downstream(
        module, downstream
    )


def public_names(modules: Mapping[str, tuple[str, bool]], surface: str) -> set[str]:
    """The names in ``surface``'s literal ``__all__``.

    Raises:
        AssertionError: ``surface`` is not among ``modules`` or has no literal
            ``__all__`` list — an empty answer would let nothing through, or
            (read as "no restriction") everything.
    """
    if surface not in modules:
        raise AssertionError(f"public surface {surface} is not a module")
    for node in ast.parse(modules[surface][0]).body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            value = ast.literal_eval(node.value)  # type: ignore[arg-type]
            if value:
                return set(value)
    raise AssertionError(f"public surface {surface} has no literal __all__")


def accessor_names(
    modules: Mapping[str, tuple[str, bool]],
    root: str = ROOT,
    downstream: str = DOWNSTREAM,
) -> set[str]:
    """The expression namespaces core registers (``register_expr_namespace("x")``).

    Raises:
        AssertionError: None found — rule (b)'s accessor check would then match
            nothing and pass forever.
    """
    found: set[str] = set()
    for name, (source, _) in modules.items():
        if not _in_core(name, root, downstream):
            continue
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.Call) and node.args):
                continue
            func = node.func
            called = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            arg = node.args[0]
            if (
                called == _REGISTRAR
                and isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
            ):
                found.add(arg.value)
    if not found:
        raise AssertionError(f"no {_REGISTRAR}(...) call found in core")
    return found


def _imports(
    name: str, is_package: bool, source: str
) -> list[tuple[ast.Import | ast.ImportFrom, str | None]]:
    """Every import statement in ``source`` with the module it names (None if it leaves)."""
    out: list[tuple[ast.Import | ast.ImportFrom, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node, alias.name))
        elif isinstance(node, ast.ImportFrom):
            out.append((node, _base(name, is_package, node)))
    return out


def _targets(
    node: ast.Import | ast.ImportFrom, base: str, modules: Collection[str]
) -> list[str]:
    """``base`` plus each ``base.name`` that is a module (``from pkg import mod``)."""
    if isinstance(node, ast.Import):
        return [base]
    return [
        base,
        *(f"{base}.{a.name}" for a in node.names if f"{base}.{a.name}" in modules),
    ]


def violations(
    modules: Mapping[str, tuple[str, bool]],
    cv: str = CV_PACKAGE,
    surfaces: Collection[str] = PUBLIC_SURFACES,
    root: str = ROOT,
    downstream: str = DOWNSTREAM,
) -> list[str]:
    """Every boundary violation in ``modules``, as printable lines."""
    out: list[str] = []
    generic_tier = set(generic_modules(modules, cv, downstream))
    public = {s: public_names(modules, s) for s in surfaces}
    accessors = accessor_names(modules, root, downstream)

    for name, (source, is_package) in sorted(modules.items()):
        here = is_downstream(name, downstream)
        generic = name in generic_tier
        for node, base in _imports(name, is_package, source):
            if base is None:
                continue
            where = f"{name}:{node.lineno}: `{ast.unparse(node)}`"
            targets = _targets(node, base, modules)
            if not here:
                if any(is_downstream(t, downstream) for t in targets):
                    out.append(f"{where} — core imports {downstream}")
                continue
            if not _in_core(base, root, downstream):
                # An import within metrics: the facade loads the CV tier, so
                # a generic module may name neither.
                if generic and any(
                    t == downstream or _within(t, cv)
                    for t in targets
                    if is_downstream(t, downstream)
                ):
                    out.append(
                        f"{where} — a generic metrics module imports the CV tier"
                    )
                continue
            if generic:
                out.append(f"{where} — a generic metrics module depends on polars-cv")
            elif not isinstance(node, ast.ImportFrom) or node.level != 0:
                out.append(
                    f"{where} — import polars-cv absolutely, as "
                    f"`from <one of {sorted(surfaces)}> import name`"
                )
            elif base not in public:
                out.append(
                    f"{where} — {base} is not a public surface {sorted(surfaces)}"
                )
            else:
                private = sorted(
                    a.name for a in node.names if a.name not in public[base]
                )
                if private:
                    out.append(f"{where} — not in {base}.__all__: {', '.join(private)}")
        if generic:
            for node in ast.walk(ast.parse(source)):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr in accessors
                ):
                    out.append(
                        f"{name}:{node.lineno}: `{ast.unparse(node)}` — a generic "
                        f"metrics module uses polars-cv's .{node.value.attr} accessor"
                    )
    return out
