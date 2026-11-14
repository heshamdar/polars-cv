"""Tests for the import-layering scanner itself.

A source scanner that silently matches nothing reads as green forever, so each
rule is pinned by a committed snippet it must reject and one it must accept.
These run the same ``violations`` the repo-wide test runs.
"""

from __future__ import annotations

import pytest

from tests._import_layering import violations

pytestmark = pytest.mark.structural


def pkg(**sources: str) -> dict[str, tuple[str, bool]]:
    """A synthetic ``polars_cv`` package: ``a`` -> ``polars_cv.a`` etc.

    ``root`` is ``__init__``; a double underscore is a dot (``sub__x``).
    """
    modules = {"polars_cv": (sources.pop("root", ""), True)}
    for key, src in sources.items():
        is_package = key.endswith("_pkg")
        key = key.removesuffix("_pkg")
        modules["polars_cv." + key.replace("__", ".")] = (src, is_package)
    return modules


BAD = {
    "from-root import of a submodule (the original entry edge)": pkg(
        root="from . import a\n",
        a="from polars_cv import b\n",
        b="X = 1\n",
    ),
    "from-root import of a name": pkg(
        root="from .a import f\n", a="from polars_cv import f\n"
    ),
    "plain import of the root": pkg(root="from .a import f\n", a="import polars_cv\n"),
    "relative root import in a root-level module": pkg(
        root="from .a import f\n", a="from . import b\n", b=""
    ),
    "two-module cycle": pkg(root="", a="from ._b import y\n", _b="from .a import x\n"),
    "three-module cycle through relative imports": pkg(
        a="from .b import y\n", b="from .c import z\n", c="from .a import x\n"
    ),
    "cycle hidden in a try block": pkg(
        a="try:\n    from .b import y\nexcept ImportError:\n    y = None\n",
        b="from .a import x\n",
    ),
    "cycle in a class body": pkg(
        a="class K:\n    from .b import y\n", b="from .a import x\n"
    ),
    "cycle across a subpackage": pkg(
        a="from .sub import m\n",
        sub_pkg="from .m import f\n",
        sub__m="from polars_cv.a import x\n",
    ),
    "type-checking else branch still runs": pkg(
        a="from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    pass\nelse:\n    from .b import y\n",
        b="from .a import x\n",
    ),
}

GOOD = {
    "submodule import by absolute path": pkg(
        a="from polars_cv._plugin import call\n", _plugin="def call(): ...\n"
    ),
    "relative submodule import": pkg(
        a="from ._plugin import call\n", _plugin="def call(): ...\n"
    ),
    "root re-exporting its submodules": pkg(
        root="from .a import f\nfrom .b import g\n", a="f = 1\n", b="g = 2\n"
    ),
    "cycle only under TYPE_CHECKING": pkg(
        a="from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .b import Y\n",
        b="from .a import x\n",
    ),
    "cycle only inside a function (pipeline <-> lazy)": pkg(
        a="def f():\n    from .b import y\n", b="from .a import f\n"
    ),
    "cycle only inside a method": pkg(
        a="class K:\n    def m(self):\n        from polars_cv import b\n",
        b="from .a import K\n",
    ),
    "function-level import of the root": pkg(
        a="def f():\n    import polars_cv\n    from polars_cv import b\n", b=""
    ),
    "subpackage init importing its siblings": pkg(
        sub_pkg="from .m import f\n", sub__m="from polars_cv._t import T\n", _t=""
    ),
    "relative import climbing out of a subpackage": pkg(
        sub__m="from .._t import T\n", _t=""
    ),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_scanner_rejects(name: str) -> None:
    assert violations(BAD[name]), f"scanner accepted known-bad layout: {name}"


@pytest.mark.parametrize("name", sorted(GOOD))
def test_scanner_accepts(name: str) -> None:
    found = violations(GOOD[name])
    assert not found, f"scanner rejected known-good layout {name}: {found}"
