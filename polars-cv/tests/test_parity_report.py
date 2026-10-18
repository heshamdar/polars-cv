"""The parity report stays true to the tests it is built from.

``tests/_parity.py`` makes every reference test declare its comparison and
run polars-cv; ``scripts/parity_report.py`` renders the README section and
``docs/user-guide/parity.md`` from a run. These guards hold the committed
report to both ends:

- the committed declarations are exactly what the suite collects now, so a
  test added, removed or re-declared without regenerating the report fails;
- the README block and the docs page are exactly what the committed JSON
  renders to, so neither can be hand-edited into saying something else.

And the declaration check and :class:`~tests._parity.Reach` are pinned on
known-bad and known-good fixtures, since a checker that matches nothing reads
as green forever.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests import _parity

pytestmark = pytest.mark.structural

_PKG = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "parity_report", _PKG / "scripts" / "parity_report.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _key(g: dict) -> tuple:
    return (g["file"], tuple(g["ops"]), g["against"], g["tolerance"], g["tests"])


def test_committed_report_matches_the_collected_declarations(tmp_path: Path) -> None:
    out = tmp_path / "collected.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "tests/reference", "--collect-only",
            "-q", "-p", "no:xdist", "-p", "no:cacheprovider", f"--parity-report={out}",
        ],
        cwd=_PKG, capture_output=True, text=True,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    collected = {_key(g) for g in json.loads(out.read_text())}
    committed = {_key(g) for g in _script().load()[0]["groups"]}
    assert collected == committed, (
        "tests/reference/ declarations differ from docs/assets/parity/reference.json; "
        "regenerate with `uv run --no-sync --with pycocotools python "
        f"scripts/parity_report.py`.\nonly collected: {sorted(collected - committed)}"
        f"\nonly committed: {sorted(committed - collected)}"
    )


def test_committed_report_records_a_clean_run() -> None:
    ref, _ = _script().load()
    for g in ref["groups"]:
        assert g["failed"] == g["skipped"] == 0 and g["passed"] == g["tests"], g


def test_readme_and_page_are_rendered_from_the_committed_json() -> None:
    script = _script()
    page, readme = script.render(*script.load())
    assert script.PAGE.read_text() == page, "docs/user-guide/parity.md is stale"
    assert script.README.read_text() == readme, "README.md parity block is stale"


# --- Fixtures for the declaration check ---------------------------------------


def _mark(**kwargs):
    return SimpleNamespace(args=(), kwargs=kwargs)


def _item(mark, name: str = "test_x.py") -> SimpleNamespace:
    return SimpleNamespace(
        nodeid=f"tests/reference/{name}::t",
        path=_parity.REFERENCE_DIR / name,
        get_closest_marker=lambda _name: mark,
    )


GOOD = {"ops": ["blur"], "against": "OpenCV `cv2.GaussianBlur`", "tolerance": "exact"}


def test_a_good_declaration_is_accepted() -> None:
    decl = _parity.declaration(_item(_mark(**GOOD)))
    assert decl == {"file": "tests/reference/test_x.py", **GOOD}


@pytest.mark.parametrize(
    ("mark", "message"),
    [
        (None, "must declare"),
        (_mark(**{**GOOD, "ops": ["no_such_op"]}), "no such op"),
        (_mark(**{**GOOD, "ops": ["contour.no_such_accessor"]}), "no such op"),
        (_mark(**{**GOOD, "ops": []}), "non-empty list"),
        (_mark(**{**GOOD, "tolerance": " "}), "tolerance must be"),
        (_mark(ops=["blur"], against="OpenCV"), "exactly the keywords"),
        (_mark(**{**GOOD, "against": "Some library"}), "must start with"),
        (_mark(**GOOD, extra=1), "exactly the keywords"),
    ],
)
def test_a_bad_declaration_is_refused(mark, message: str) -> None:
    with pytest.raises(pytest.UsageError, match=message):
        _parity.declaration(_item(mark))


@pytest.mark.parametrize(
    "op", ["resize", "contour.area", "cv.image_info", "metrics.evaluate_detections"]
)
def test_each_op_spelling_resolves(op: str) -> None:
    _parity.declaration(_item(_mark(**{**GOOD, "ops": [op]})))


# --- Fixtures for Reach -------------------------------------------------------


def _watch(fn) -> bool:
    reach = _parity.Reach()
    reach.start()
    try:
        fn()
    finally:
        reach.stop()
    return reach.reached


def test_reach_sees_a_call_into_polars_cv() -> None:
    from polars_cv import Pipeline

    assert _watch(lambda: Pipeline())


def test_reach_does_not_see_numpy_alone() -> None:
    assert not _watch(lambda: np.sort(np.arange(10)).sum())


def test_reach_leaves_the_previous_profiler_in_place() -> None:
    def prior(*_args):  # noqa: ANN202
        return None

    sys.setprofile(prior)
    try:
        _watch(lambda: np.zeros(1))
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(None)
