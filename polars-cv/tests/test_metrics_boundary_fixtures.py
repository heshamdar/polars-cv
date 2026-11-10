"""Tests for the metrics boundary scanner itself.

Each rule is pinned by committed snippets it must reject and ones it must
accept, run through the same ``violations`` the repo-wide test runs.
"""

from __future__ import annotations

import pytest

from tests._metrics_boundary import accessor_names, public_names, violations

pytestmark = pytest.mark.structural

#: The adapters of the synthetic package: one module and the facade.
ADAPTERS = frozenset({"polars_cv.metrics", "polars_cv.metrics.cv"})


def pkg(**sources: str) -> dict[str, tuple[str, bool]]:
    """A synthetic ``polars_cv`` with a public root, ``geometry`` and ``metrics``.

    The root exports ``Pipeline``; ``geometry`` exports ``to_box``; a private
    ``_types`` holds ``Alias``; ``expr`` registers the ``contour`` namespace.
    ``metrics`` is a package whose adapter is ``metrics.cv`` (``ADAPTERS``).
    Keyword ``m__x`` is ``polars_cv.metrics.x``; ``core__x`` is ``polars_cv.x``.
    """
    modules: dict[str, tuple[str, bool]] = {
        "polars_cv": (
            "from .pipeline import Pipeline\n__all__ = ['Pipeline']\n",
            True,
        ),
        "polars_cv.pipeline": ("class Pipeline: ...\n", False),
        "polars_cv._types": ("Alias = int\n", False),
        "polars_cv.geometry": ("def to_box(): ...\n__all__ = ['to_box']\n", True),
        "polars_cv.expr": (
            "import polars as pl\n"
            "@pl.api.register_expr_namespace('contour')\n"
            "class C: ...\n",
            False,
        ),
        "polars_cv.metrics": ("from .cv import f\n", True),
        "polars_cv.metrics.cv": ("def f(): ...\n", False),
        "polars_cv.metrics.stats": ("import polars as pl\n", False),
    }
    for key, src in sources.items():
        prefix, _, rest = key.partition("__")
        full = {"m": "polars_cv.metrics.", "core": "polars_cv."}[prefix] + rest
        modules[full] = (src, modules.get(full, ("", False))[1])
    return modules


BAD = {
    "core imports metrics at module level": pkg(
        core__other="from polars_cv.metrics import f\n"
    ),
    "core imports metrics inside a function": pkg(
        core__pipeline="def g():\n    from .metrics import f\n"
    ),
    "core root imports the metrics subpackage by name": pkg(
        core__pipeline="from polars_cv import metrics\n"
    ),
    "generic module imports the root": pkg(m__stats="from polars_cv import Pipeline\n"),
    "generic module imports a public surface": pkg(
        m__stats="from polars_cv.geometry import to_box\n"
    ),
    "generic module imports polars-cv only for type checking": pkg(
        m__stats=(
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from polars_cv import Pipeline\n"
        )
    ),
    "generic module imports a CV adapter": pkg(m__stats="from .cv import f\n"),
    "generic module imports a CV adapter in a function": pkg(
        m__stats="def g():\n    from polars_cv.metrics.cv import f\n"
    ),
    "generic module uses a registered accessor": pkg(
        m__stats="import polars as pl\nx = pl.col('a').contour.area()\n"
    ),
    "adapter imports a private module": pkg(
        m__cv="from polars_cv._types import Alias\n"
    ),
    "adapter imports a public module that is not a surface": pkg(
        m__cv="from polars_cv.pipeline import Pipeline\n"
    ),
    "adapter imports a name missing from __all__": pkg(
        m__cv="from polars_cv import Hidden\n"
    ),
    "adapter climbs out relatively, even to the root": pkg(
        m__cv="from .. import Pipeline\n"
    ),
    "adapter imports the root bare": pkg(m__cv="import polars_cv\n"),
    "adapter list names a missing module": pkg(),
}

GOOD = {
    "the baseline package": pkg(),
    "generic module imports a generic sibling": pkg(m__more="from .stats import pl\n"),
    "generic module uses polars' own namespaces": pkg(
        m__stats="import polars as pl\nx = pl.col('a').list.len().struct.field('b')\n"
    ),
    "adapter imports the root's public names": pkg(
        m__cv="from polars_cv import Pipeline\n"
    ),
    "adapter imports a public surface's names": pkg(
        m__cv="from polars_cv.geometry import to_box\n"
    ),
    "adapter uses a registered accessor": pkg(
        m__cv="import polars as pl\nx = pl.col('a').contour.area()\n"
    ),
    "adapter imports a generic module": pkg(m__cv="from .stats import pl\n"),
    "core imports a third-party package": pkg(core__other="import numpy\n"),
}

#: The rule each BAD case must be reported under: a fragment of the message,
#: required in *every* line reported, so a case cannot pass because some other
#: rule happened to fire.
REASON = {
    "core imports metrics at module level": "core imports",
    "core imports metrics inside a function": "core imports",
    "core root imports the metrics subpackage by name": "core imports",
    "generic module imports the root": "depends on polars-cv",
    "generic module imports a public surface": "depends on polars-cv",
    "generic module imports polars-cv only for type checking": "depends on polars-cv",
    "generic module imports a CV adapter": "imports a CV adapter",
    "generic module imports a CV adapter in a function": "imports a CV adapter",
    "generic module uses a registered accessor": "accessor",
    "adapter imports a private module": "is not a public surface",
    "adapter imports a public module that is not a surface": "is not a public surface",
    "adapter imports a name missing from __all__": "not in polars_cv.__all__",
    "adapter climbs out relatively, even to the root": "import polars-cv absolutely",
    "adapter imports the root bare": "import polars-cv absolutely",
    "adapter list names a missing module": "CV_ADAPTERS names",
}

#: ``violations`` takes the adapter set; this BAD case lists a module the
#: package does not have.
_ADAPTERS_FOR = {
    "adapter list names a missing module": ADAPTERS | {"polars_cv.metrics.nope"}
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_scanner_rejects(name: str) -> None:
    found = violations(BAD[name], adapters=_ADAPTERS_FOR.get(name, ADAPTERS))
    assert found, f"scanner accepted known-bad layout: {name}"
    other = [line for line in found if REASON[name] not in line]
    assert not other, (
        f"{name}: expected only {REASON[name]!r} violations, also got: {other}"
    )


def test_every_bad_case_names_its_reason() -> None:
    assert set(REASON) == set(BAD)


@pytest.mark.parametrize("name", sorted(GOOD))
def test_scanner_accepts(name: str) -> None:
    found = violations(GOOD[name], adapters=ADAPTERS)
    assert not found, f"scanner rejected known-good layout {name}: {found}"


def test_surface_names_come_from_dunder_all() -> None:
    assert public_names(pkg(), "polars_cv") == {"Pipeline"}


@pytest.mark.parametrize(
    "modules, surface",
    [
        (pkg(), "polars_cv.nowhere"),
        (pkg(core__geometry="def to_box(): ...\n"), "polars_cv.geometry"),
    ],
    ids=["missing surface", "surface without __all__"],
)
def test_a_surface_it_cannot_read_is_an_error(
    modules: dict[str, tuple[str, bool]], surface: str
) -> None:
    with pytest.raises(AssertionError):
        public_names(modules, surface)


def test_accessors_come_from_the_registrations() -> None:
    assert accessor_names(pkg()) == {"contour"}


def test_no_registered_accessor_is_an_error() -> None:
    """Rule (b)'s accessor check must not pass by matching nothing."""
    with pytest.raises(AssertionError):
        accessor_names(pkg(core__expr="X = 1\n"))
