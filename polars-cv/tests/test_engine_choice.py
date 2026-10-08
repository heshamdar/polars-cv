"""The package never chooses polars' engine; polars' default does.

Since polars 2.0 a lazy ``collect()`` resolves ``engine="auto"`` to the
streaming engine, and ``pl.Config.set_engine_affinity`` lets a user choose
otherwise (in-memory while debugging, say). The metrics once passed
``engine="streaming"`` at sixteen call sites: sixteen restatements of the
default, each overriding the user's affinity. Now a ``collect``/``collect_all``
in ``python/polars_cv`` either passes no engine or forwards one its own caller
chose (``DetectionTable.collect(engine=...)``). A literal engine is the
package choosing, and is refused.

The streaming coverage this gave up nothing of: ``test_streaming_plans.py``
holds every metrics plan to the streaming engine, whatever engine a caller
then collects it with.

Limits: a source scan (the property is about call sites, which no runtime
check sees). It finds ``engine=<literal>`` keywords in any call. An engine
smuggled in another way (``**{"engine": ...}``, a module constant) is not
caught. The fixtures below hold the scanner to what it claims.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import polars_cv
from tests._discovery import package_modules

pytestmark = pytest.mark.structural

PACKAGE = Path(polars_cv.__file__).parent


def literal_engines(source: str) -> list[int]:
    """Lines in ``source`` that choose an engine: a call passing ``engine=`` a
    literal, or a parameter named ``engine`` defaulting to anything but
    ``"auto"`` (polars' own default, which defers to the user's affinity)."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "engine" and isinstance(kw.value, ast.Constant):
                    lines.append(kw.value.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = [*args.posonlyargs, *args.args]
            defaults = dict(
                zip(positional[len(positional) - len(args.defaults) :], args.defaults)
            )
            defaults.update(
                (a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d
            )
            for arg, default in defaults.items():
                if arg.arg == "engine" and not (
                    isinstance(default, ast.Constant) and default.value == "auto"
                ):
                    lines.append(default.lineno)
    return lines


@pytest.mark.parametrize(
    "source",
    [
        'lf.collect(engine="streaming")',
        'pl.collect_all(frames, engine="in-memory")',
        'x = f(lf).sink_parquet("p", engine="streaming")',
        "lf.collect(engine=None)",
        'def collect(self, engine: str = "streaming"): ...',
        'def run(lf, *, engine="in-memory"): ...',
    ],
)
def test_the_scanner_refuses_a_chosen_engine(source: str) -> None:
    assert literal_engines(source) == [1]


@pytest.mark.parametrize(
    "source",
    [
        "lf.collect()",
        "pl.collect_all(frames)",
        "def collect(self, engine='auto'):\n    return pl.collect_all(fs, engine=engine)",
        'def run(lf, *, engine: str = "auto"): ...',
        "def run(lf, other=1, eng=None): ...",
        '# lf.collect(engine="streaming") in a comment',
        '"""Under ``engine="streaming"`` a bytes column ..."""',
    ],
)
def test_the_scanner_accepts_the_default_and_forwarding(source: str) -> None:
    assert literal_engines(source) == []


def test_the_package_never_chooses_an_engine() -> None:
    offenders = [
        f"{path.relative_to(PACKAGE)}:{line}"
        for path in package_modules()
        for line in literal_engines(path.read_text())
    ]
    assert offenders == [], (
        "these calls choose polars' engine; leave it to polars' default (or "
        "forward the caller's choice):\n  " + "\n  ".join(offenders)
    )
