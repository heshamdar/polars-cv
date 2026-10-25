"""Test-suite conformance: shared fixtures live in ``conftest.py`` only."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests._discovery import (
    suite_modules,
)

#: A structural guard: it checks the *shape* of the codebase rather than the
#: numerical behaviour of a pipeline. `-m structural` is the lane pre-commit
#: runs; see `tests/AGENTS.md`.
pytestmark = pytest.mark.structural


def test_no_local_plugin_available_definitions() -> None:
    """Plugin availability is checked in exactly one place (conftest.py).

    Import ``plugin_required`` from ``tests.conftest`` instead of redefining
    ``_plugin_available`` per file."""
    offenders = [
        str(p.name) for p in suite_modules() if "def _plugin_available" in p.read_text()
    ]
    assert not offenders, f"local _plugin_available definitions in: {offenders}"


#: The PNG-construction fixtures `conftest.py` owns. A test module that
#: redefines one shadows the shared fixture, which is how eleven modules came
#: to carry a copy of `encode_png` that had dropped conftest's
#: `except ImportError: pytest.skip("PIL/Pillow required")` arm -- so without
#: Pillow they errored where the suite means to skip.
_CONFTEST_PNG_FACTORIES = ("create_test_png", "encode_png")

#: conftest-owned image *data* fixtures (as opposed to the factory functions
#: above). Overriding one per-module is a legitimate pytest pattern *when it
#: delegates to a guarded factory* -- e.g. `test_statistical_reductions.py`
#: takes `encode_png` and feeds it a test-specific array. What is banned is a
#: redefinition that builds the PNG itself via Pillow, bypassing conftest's
#: `except ImportError: pytest.skip` -- `test_typed_nodes.py` carried one, so on
#: a Pillow-less machine it raised at collection where the suite means to skip.
_CONFTEST_IMAGE_DATA_FIXTURES = ("sample_image_bytes",)


def test_no_local_png_factories() -> None:
    """PNG construction fixtures live in conftest.py only; test files must not
    define their own.

    Checks **every** name in `_CONFTEST_PNG_FACTORIES`. It used to grep for
    `create_test_png` alone while its own docstring named two, and the
    unchecked half had drifted into eleven copies -- a guard that reads as
    coverage and enforces half of it.
    """
    conftest = Path(__file__).parent / "conftest.py"
    conftest_src = conftest.read_text()
    for name in _CONFTEST_PNG_FACTORIES:
        assert f"def {name}" in conftest_src, (
            f"probe is broken: conftest.py no longer defines {name!r}; "
            "update _CONFTEST_PNG_FACTORIES rather than leaving it scanning "
            "for a name that cannot be found."
        )

    offenders: dict[str, list[str]] = {}
    for module in suite_modules():
        source = module.read_text()
        for name in _CONFTEST_PNG_FACTORIES:
            if f"def {name}" in source:
                offenders.setdefault(name, []).append(str(module.name))

    assert not offenders, (
        f"these modules redefine a conftest PNG factory: {offenders}. "
        "Use the shared fixture -- a local copy shadows it, and the copies "
        "have already dropped its Pillow-missing skip."
    )


def _image_data_fixture_offenders(source: str, names: tuple[str, ...]) -> list[str]:
    """Names in ``names`` that ``source`` defines as a fixture building a PNG via
    direct Pillow calls (a body that references the ``Image`` / ``PIL`` name)
    rather than delegating to a guarded conftest factory.

    A delegating override -- ``def sample_image_bytes(encode_png): return
    encode_png(arr)`` -- keeps the shared ``except ImportError: pytest.skip`` and
    is fine; its body never names ``Image``. Limit: an override that imported
    ``Image`` only for a type annotation would false-positive, but none do.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - suite modules parse
        return []
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in names
        ):
            if any(
                isinstance(inner, ast.Name) and inner.id in ("Image", "PIL")
                for inner in ast.walk(node)
            ):
                offenders.append(node.name)
    return offenders


def test_no_unguarded_local_image_data_fixtures() -> None:
    """A per-module override of a conftest image-*data* fixture must delegate to
    a guarded factory (``create_test_png``/``encode_png``), never build the PNG
    itself via Pillow.

    ``test_no_local_png_factories`` bans redefining the factory *functions*; this
    is its sibling for the *data* fixtures they feed. Building the PNG inline
    drops conftest's ``except ImportError: pytest.skip`` and raises at collection
    on a Pillow-less machine -- ``test_typed_nodes.py`` carried exactly such a
    ``sample_image_bytes``. The detector is watched both ways below so it cannot
    silently match nothing.
    """
    bad = (
        "@pytest.fixture\n"
        "def sample_image_bytes():\n"
        "    buf = BytesIO()\n"
        "    Image.fromarray(arr).save(buf, format='PNG')\n"
        "    return buf.getvalue()\n"
    )
    good = (
        "@pytest.fixture\n"
        "def sample_image_bytes(encode_png):\n"
        "    return encode_png(arr)\n"
    )
    assert _image_data_fixture_offenders(bad, _CONFTEST_IMAGE_DATA_FIXTURES) == [
        "sample_image_bytes"
    ], "detector must flag a direct-Pillow override"
    assert _image_data_fixture_offenders(good, _CONFTEST_IMAGE_DATA_FIXTURES) == [], (
        "detector must not flag a factory-delegating override"
    )

    offenders: dict[str, list[str]] = {}
    for module in suite_modules():
        for name in _image_data_fixture_offenders(
            module.read_text(), _CONFTEST_IMAGE_DATA_FIXTURES
        ):
            offenders.setdefault(name, []).append(str(module.name))

    assert not offenders, (
        f"these modules override a conftest image-data fixture with a direct-"
        f"Pillow build, bypassing the shared skip: {offenders}. Take `encode_png`"
        " (or `create_test_png`) and feed it your array instead."
    )
