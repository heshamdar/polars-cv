"""Discovery: no guard may find its own files (every scan goes through ``tests/_discovery.py``)."""

from __future__ import annotations

import ast

import pytest

from tests._discovery import (
    requires_checkout,
    suite_files,
)

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


#: Where a direct filesystem walk is legitimate, with the reason. Every entry
#: is a place where finding *nothing* is a meaningful answer rather than a
#: broken scan, which is exactly the property `_discovery` refuses to allow.
_DISCOVERY_EXEMPT: dict[str, str] = {
    "_discovery.py": "the discovery module itself",
    "conftest.py": (
        "plugin detection: an empty result means the extension is not built, "
        "which is the answer `plugin_required` needs rather than a failure"
    ),
    "test_streaming_ooc.py": (
        "asserts a spill directory stays empty, so an empty walk is the "
        "assertion rather than a broken scan"
    ),
}


@requires_checkout
def test_scans_go_through_discovery() -> None:
    """File discovery happens in ``tests/_discovery.py`` and nowhere else.

    A guard that finds its own files can find none of them, and then it passes
    while checking nothing — the failure mode `tests/AGENTS.md` calls worse
    than no guard, and one this suite had shipped twice: ``_test_files()`` and
    ``_PACKAGE_MODULES`` were both a bare ``rglob`` whose empty result was
    indistinguishable from a clean bill of health.

    Routing every scan through :mod:`tests._discovery` makes that impossible by
    construction rather than by remembering to assert non-emptiness at each
    site. This test is what stops the next scan from being written the old way,
    and it is deliberately a *mechanism* check (may you glob?) rather than a
    list of the scans that exist.
    """
    offenders: list[str] = []
    for module in suite_files():
        if module.name in _DISCOVERY_EXEMPT:
            continue
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in {"glob", "rglob"}:
                offenders.append(f"{module.name}:{node.lineno}")

    assert not offenders, (
        "these sites walk the filesystem directly instead of calling "
        f"tests._discovery: {offenders}. A scan that finds nothing passes "
        "vacuously; _discovery raises instead. If an empty result is genuinely "
        "the answer, add the file to _DISCOVERY_EXEMPT with the reason."
    )


@requires_checkout
def test_discovery_exemptions_are_real_files() -> None:
    """An exemption for a file that no longer exists is a stale exemption.

    Without this the allowlist could quietly grow to cover renamed files,
    re-opening the hole it documents closing.
    """
    names = {module.name for module in suite_files()}
    stale = sorted(name for name in _DISCOVERY_EXEMPT if name not in names)
    assert not stale, f"exemptions for files that do not exist: {stale}"
