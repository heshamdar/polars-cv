"""``polars_cv._plugin.call`` is the only way into the compiled extension.

It is what pins polars to the extension file Python imports and hands the
plugin storage instead of extension types (see ``polars_cv._plugin``). A second
``register_plugin_function`` call site would silently skip both, so the package
is scanned for one, and the scan's reach is pinned by fixtures in both
directions.
"""

from __future__ import annotations

import ast

import pytest

from ._discovery import package_modules

pytestmark = pytest.mark.structural


def _plugin_references(source: str) -> list[int]:
    """Lines that reach ``register_plugin_function`` by any spelling.

    Catches the import (``from polars.plugins import register_plugin_function``)
    and attribute access (``pl.plugins.register_plugin_function``,
    ``plugins.register_plugin_function``). Limits: a ``getattr`` with a computed
    string would evade it; nothing in this codebase does that, and the fixtures
    below pin what it does catch.
    """
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and any(
            a.name == "register_plugin_function" for a in node.names
        ):
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.Attribute) and node.attr == "register_plugin_function"
        ):
            lines.append(node.lineno)
    return sorted(lines)


@pytest.mark.parametrize(
    "snippet",
    [
        "from polars.plugins import register_plugin_function",
        "from polars.plugins import register_plugin_function as rpf",
        "import polars as pl\npl.plugins.register_plugin_function(plugin_path='x')",
        "from polars import plugins\nplugins.register_plugin_function()",
    ],
)
def test_reference_scan_catches_bypasses(snippet: str) -> None:
    assert _plugin_references(snippet), snippet


@pytest.mark.parametrize(
    "snippet",
    [
        "from polars_cv import _plugin\n_plugin.call('area', args=[])",
        "x = 'register_plugin_function'",
    ],
)
def test_reference_scan_ignores_the_sanctioned_path(snippet: str) -> None:
    assert not _plugin_references(snippet), snippet


def test_only_the_plugin_module_registers_plugin_functions() -> None:
    """``_plugin.call`` is the only way into the compiled extension.

    It is what pins polars to the file Python imports and strips extension
    tags before the plugin sees them; a second call site would skip both.
    """
    offenders = {
        path.name: lines
        for path in package_modules()
        if path.name != "_plugin.py" and (lines := _plugin_references(path.read_text()))
    }
    assert offenders == {}, (
        f"register_plugin_function reached outside polars_cv._plugin: {offenders}"
    )
