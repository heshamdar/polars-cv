"""Tests for the metrics boundary scanner itself.

Each rule is pinned by committed snippets it must reject and ones it must
accept, run through the same ``violations`` the repo-wide test runs.
"""

from __future__ import annotations

import pytest

from tests._metrics_boundary import (
    accessor_names,
    generic_modules,
    public_names,
    violations,
)

pytestmark = pytest.mark.structural


def pkg(**sources: str) -> dict[str, tuple[str, bool]]:
    """A synthetic ``polars_cv`` with a public root, ``geometry`` and ``metrics``.

    The root exports ``Pipeline``; ``geometry`` exports ``to_box``; a private
    ``_types`` holds ``Alias``; ``expr`` registers the ``contour`` namespace.
    ``metrics`` is a package whose CV tier is the package ``metrics._cv`` and
    whose one generic module is ``metrics.stats``. Keyword ``m__x`` is
    ``polars_cv.metrics.x`` and ``core__x`` is ``polars_cv.x``; a further
    ``__`` is a dot (``m___cv__inner`` is ``polars_cv.metrics._cv.inner``).
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
        "polars_cv.metrics": ("from ._cv import f\n", True),
        "polars_cv.metrics._cv": ("def f(): ...\n", True),
        "polars_cv.metrics.stats": ("import polars as pl\n", False),
    }
    for key, src in sources.items():
        prefix, _, rest = key.partition("__")
        full = {"m": "polars_cv.metrics.", "core": "polars_cv."}[prefix]
        full += rest.replace("__", ".")
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
    "generic module imports the CV tier": pkg(m__stats="from ._cv import f\n"),
    "generic module imports the CV tier in a function": pkg(
        m__stats="def g():\n    from polars_cv.metrics._cv import f\n"
    ),
    "generic module imports a module inside the CV tier": pkg(
        m___cv__inner="g = 1\n", m__stats="from ._cv.inner import g\n"
    ),
    "generic module imports the facade": pkg(m__stats="from . import f\n"),
    "generic module uses a registered accessor": pkg(
        m__stats="import polars as pl\nx = pl.col('a').contour.area()\n"
    ),
    "generic module binds a registered accessor to a name": pkg(
        m__stats="import polars as pl\nns = pl.col('a').contour\nx = ns.area()\n"
    ),
    "adapter reads a private attribute of a public class": pkg(
        m___cv="from polars_cv import Pipeline\nx = Pipeline._hidden\n"
    ),
    "adapter reads a private attribute through a call chain": pkg(
        m___cv="from polars_cv import Pipeline as P\nx = P().source('auto')._hidden\n"
    ),
    "adapter imports a private module": pkg(
        m___cv="from polars_cv._types import Alias\n"
    ),
    "adapter imports a public module that is not a surface": pkg(
        m___cv="from polars_cv.pipeline import Pipeline\n"
    ),
    "adapter imports a name missing from __all__": pkg(
        m___cv="from polars_cv import Hidden\n"
    ),
    "adapter climbs out relatively, even to the root": pkg(
        m___cv="from ... import Pipeline\n"
    ),
    "adapter imports the root bare": pkg(m___cv="import polars_cv\n"),
}

GOOD = {
    "the baseline package": pkg(),
    "generic module imports a generic sibling": pkg(m__more="from .stats import pl\n"),
    "generic module uses polars' own namespaces": pkg(
        m__stats="import polars as pl\nx = pl.col('a').list.len().struct.field('b')\n"
    ),
    "adapter imports the root's public names": pkg(
        m___cv="from polars_cv import Pipeline\n"
    ),
    "adapter imports a public surface's names": pkg(
        m___cv="from polars_cv.geometry import to_box\n"
    ),
    "adapter reads its own private attributes and dunders": pkg(
        m___cv=(
            "from polars_cv import Pipeline\n"
            "class H:\n    _x = 1\n"
            "y = H()._x\nz = Pipeline.__name__\n"
        )
    ),
    "adapter uses a registered accessor": pkg(
        m___cv="import polars as pl\nx = pl.col('a').contour.area()\n"
    ),
    "adapter imports a generic module": pkg(m___cv="from ..stats import pl\n"),
    "module inside the CV tier imports the public API": pkg(
        m___cv__inner="from polars_cv import Pipeline\n"
    ),
    "generic subpackage with generic modules": pkg(
        m__sub="from .a import x\n", m__sub__a="x = 1\n"
    ),
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
    "generic module imports the CV tier": "imports the CV tier",
    "generic module imports the CV tier in a function": "imports the CV tier",
    "generic module imports a module inside the CV tier": "imports the CV tier",
    "generic module imports the facade": "imports the CV tier",
    "generic module uses a registered accessor": "accessor",
    "generic module binds a registered accessor to a name": "accessor",
    "adapter reads a private attribute of a public class": "private attribute",
    "adapter reads a private attribute through a call chain": "private attribute",
    "adapter imports a private module": "is not a public surface",
    "adapter imports a public module that is not a surface": "is not a public surface",
    "adapter imports a name missing from __all__": "not in polars_cv.__all__",
    "adapter climbs out relatively, even to the root": "import polars-cv absolutely",
    "adapter imports the root bare": "import polars-cv absolutely",
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_scanner_rejects(name: str) -> None:
    found = violations(BAD[name])
    assert found, f"scanner accepted known-bad layout: {name}"
    other = [line for line in found if REASON[name] not in line]
    assert not other, (
        f"{name}: expected only {REASON[name]!r} violations, also got: {other}"
    )


def test_every_bad_case_names_its_reason() -> None:
    assert set(REASON) == set(BAD)


@pytest.mark.parametrize("name", sorted(GOOD))
def test_scanner_accepts(name: str) -> None:
    found = violations(GOOD[name])
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


def test_the_tiers_come_from_the_layout() -> None:
    """Generic is everything under metrics but the facade and the CV package."""
    modules = pkg(m___cv__inner="g = 1\n", m__sub="", m__sub__a="x = 1\n")
    modules["polars_cv.metrics.sub"] = ("", True)
    assert generic_modules(modules) == [
        "polars_cv.metrics.stats",
        "polars_cv.metrics.sub",
        "polars_cv.metrics.sub.a",
    ]


def test_a_missing_cv_package_is_an_error() -> None:
    """Were ``_cv`` renamed, every module would read as generic, or none."""
    modules = {k: v for k, v in pkg().items() if k != "polars_cv.metrics._cv"}
    with pytest.raises(AssertionError, match="CV tier"):
        violations(modules)
