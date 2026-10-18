"""Parity markers: what each reference test holds polars-cv to, and the report.

Every test under ``tests/reference/`` declares the comparison it makes::

    pytestmark = pytest.mark.parity(
        ops=["blur"],
        against="OpenCV cv2.GaussianBlur",
        tolerance=f"mean |diff| < {MEAN_TOL}, max |diff| <= {MAX_TOL} levels",
    )

on its module (``pytestmark``), class or function; the closest one wins. The
declaration is what puts a test in the parity report, and collection refuses a
reference test without one, so a comparison cannot be added that the report
does not show. ``ops`` are checked against the public API: a bare name is a
``Pipeline`` or ``LazyPipelineExpr`` method, ``contour.``/``point.``/``bbox.``/
``cv.`` name an expression-namespace method and ``metrics.`` a function of
``polars_cv.metrics``. A name none of them has is refused, so a renamed op
cannot leave the report describing one that no longer exists.

A declared test must also run polars-cv: one whose setup and body never enter
the ``polars_cv`` package fails, however its assertions went. A test that
checks a reference library against itself (``cv2.findContours`` against
NumPy, with no polars-cv output in sight) reads as coverage in the report
while comparing nothing; this is what refuses it. :class:`Reach` watches for
the first ``polars_cv`` frame with ``sys.setprofile`` and stops watching
there, so it costs nothing once the test has called in.

``--parity-report PATH`` writes one entry per declared comparison, with how
many of its tests passed, failed or were skipped (under ``--collect-only``:
how many were collected). ``scripts/parity_report.py`` runs the suite with it
and renders the docs page and the README summary from the result;
``test_parity_report.py`` holds the committed report to what is collected.

Limit: ``tolerance`` is prose. Build it from the constants the test asserts
with (an f-string, as above) so the two cannot drift; nothing can check prose
that restates a number instead.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
_PROP = "parity"
_FIELDS = ("ops", "against", "tolerance")
_OUTCOMES = ("passed", "failed", "skipped")

#: What ``against`` may start with: the independent implementation a test
#: compares with. ``Python`` is a reimplementation written in the test,
#: ``Cross-check`` a second, independent polars-cv path to the same quantity,
#: ``Definition`` a written contract and ``Hand-computed`` values worked out
#: by hand. The report groups by this word, so a free-form one is refused.
REFERENCES = (
    "OpenCV", "SciPy", "NumPy", "PIL", "imagehash", "matplotlib", "pycocotools",
    "Python", "Cross-check", "Definition", "Hand-computed",
)  # fmt: skip


def addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--parity-report",
        metavar="PATH",
        default=None,
        help="write each declared reference comparison and its outcomes to PATH",
    )


def _known_op(name: str) -> bool:
    import polars as pl

    import polars_cv.metrics
    from polars_cv import LazyPipelineExpr, Pipeline

    prefix, _, member = name.rpartition(".")
    if not prefix:
        return hasattr(Pipeline, name) or hasattr(LazyPipelineExpr, name)
    if prefix == "metrics":
        return hasattr(polars_cv.metrics, member)
    if prefix in ("contour", "point", "bbox", "cv"):
        return hasattr(getattr(pl.col("x"), prefix), member)
    return False


def declaration(item: pytest.Item) -> dict[str, Any]:
    """The item's parity declaration, validated; raises on a missing or bad one."""
    mark = item.get_closest_marker(_PROP)
    if mark is None:
        raise pytest.UsageError(
            f"{item.nodeid}: a reference test must declare what it compares "
            "with @pytest.mark.parity(ops=[...], against=..., tolerance=...)"
        )
    if mark.args or set(mark.kwargs) != set(_FIELDS):
        raise pytest.UsageError(
            f"{item.nodeid}: parity takes exactly the keywords {_FIELDS}, "
            f"got args={mark.args} kwargs={sorted(mark.kwargs)}"
        )
    ops, against, tolerance = (mark.kwargs[f] for f in _FIELDS)
    if not (
        isinstance(ops, (list, tuple)) and ops and all(isinstance(o, str) for o in ops)
    ):
        raise pytest.UsageError(
            f"{item.nodeid}: parity ops must be a non-empty list of names"
        )
    unknown = [o for o in ops if not _known_op(o)]
    if unknown:
        raise pytest.UsageError(f"{item.nodeid}: parity names no such op: {unknown}")
    for field, value in (("against", against), ("tolerance", tolerance)):
        if not (isinstance(value, str) and value.strip()):
            raise pytest.UsageError(
                f"{item.nodeid}: parity {field} must be non-empty text"
            )
    if against.split(" ", 1)[0].rstrip(":") not in REFERENCES:
        raise pytest.UsageError(
            f"{item.nodeid}: parity against must start with one of {REFERENCES}, "
            f"got {against!r}"
        )
    file = Path(str(item.path)).resolve().relative_to(REFERENCE_DIR.parent.parent)
    return {
        "file": file.as_posix(),
        "ops": list(ops),
        "against": against,
        "tolerance": tolerance,
    }


def _key(decl: dict[str, Any]) -> str:
    return json.dumps(decl, sort_keys=True)


class Recorder:
    """Collects declarations and outcomes; on the xdist controller too.

    Workers attach each item's declaration to ``user_properties``, which xdist
    ships back on every report, so the controller (which collects nothing)
    still knows which comparison a report belongs to.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.collected: dict[str, int] = defaultdict(int)
        self.outcome: dict[str, tuple[str, str]] = {}

    def collection(self, items: list[pytest.Item]) -> None:
        for item in items:
            if REFERENCE_DIR in Path(str(item.path)).resolve().parents:
                self.collected[_key(declaration(item))] += 1

    def report(self, rep: pytest.TestReport) -> None:
        props = dict(rep.user_properties)
        if _PROP not in props:
            return
        key, prev = props[_PROP], self.outcome.get(rep.nodeid, (None, ""))[1]
        if rep.failed:
            now = "failed"
        elif rep.skipped:
            now = "skipped"
        elif rep.when == "call":
            now = "passed"
        else:
            return
        rank = {"": 0, "passed": 1, "skipped": 2, "failed": 3}
        if rank[now] > rank[prev]:
            self.outcome[rep.nodeid] = (key, now)

    def write(self) -> None:
        groups: dict[str, dict[str, Any]] = {}
        for key, n in self.collected.items():
            groups[key] = {**json.loads(key), "tests": n}
        for key, outcome in self.outcome.values():
            g = groups.setdefault(key, {**json.loads(key), "tests": 0})
            g[outcome] = g.get(outcome, 0) + 1
        for g in groups.values():
            for o in _OUTCOMES:
                g.setdefault(o, 0)
            g["tests"] = max(g["tests"], sum(g[o] for o in _OUTCOMES))
        rows = sorted(
            groups.values(), key=lambda g: (g["file"], g["ops"], g["against"])
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rows, indent=1) + "\n")


def tag(items: list[pytest.Item]) -> None:
    """Refuse an undeclared reference test; tag each declared one for reports."""
    for item in items:
        if REFERENCE_DIR in Path(str(item.path)).resolve().parents:
            item.user_properties.append((_PROP, _key(declaration(item))))


_PACKAGE = str(Path(__import__("polars_cv").__file__).resolve().parent)


class Reach:
    """Whether the current thread has entered the ``polars_cv`` package.

    Profiles Python calls (a native call into the extension is entered through
    a ``polars_cv`` Python frame: the builder, ``_plugin.call``, a namespace),
    and unhooks itself at the first one. Limit: work that only a fixture
    shared across tests did (module scope) is not seen by later tests; give
    each test its own call into polars-cv.
    """

    def __init__(self) -> None:
        self.reached = False
        self._prev = None

    def _profile(self, frame, event, arg):  # noqa: ANN001, ANN202
        if event == "call" and frame.f_code.co_filename.startswith(_PACKAGE):
            self.reached = True
            sys.setprofile(self._prev)

    def start(self) -> None:
        self._prev = sys.getprofile()
        sys.setprofile(self._profile)

    def stop(self) -> None:
        if not self.reached:
            sys.setprofile(self._prev)


def is_declared(item: pytest.Item) -> bool:
    return REFERENCE_DIR in Path(str(item.path)).resolve().parents


def unreached(item: pytest.Item) -> str:
    return (
        f"{item.nodeid} declares a parity comparison but never ran polars-cv: "
        "compare polars-cv's output with the reference, or move the test out "
        "of tests/reference/"
    )
