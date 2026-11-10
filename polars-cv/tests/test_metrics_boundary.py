"""The boundary between polars-cv's core and its downstream metrics layer.

``polars_cv.metrics`` is built on polars-cv's public API the way scikit-learn
is built on NumPy, and its generic part (statistics, curves, bootstrap) on
Polars alone. See ``tests/_metrics_boundary.py`` for the rules and their
limits, and ``tests/test_metrics_boundary_fixtures.py`` for the cases the
scanner must reject and accept.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import polars_cv
from tests._discovery import package_modules
from tests._import_layering import read_modules
from tests._metrics_boundary import DOWNSTREAM, generic_modules, violations

pytestmark = pytest.mark.structural

PACKAGE_DIR = Path(polars_cv.__file__).parent
MODULES = read_modules(package_modules(), PACKAGE_DIR)


def test_metrics_boundary_holds() -> None:
    found = violations(MODULES)
    assert not found, "metrics boundary violated:\n" + "\n".join(found)


def test_importing_polars_cv_does_not_load_metrics() -> None:
    """The run-time form of rule (a), at the entry point users import."""
    code = f"import sys, polars_cv; print({DOWNSTREAM!r} in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False", "import polars_cv loaded the metrics"


#: Run in a fresh interpreter where ``polars_cv`` cannot be imported: every
#: module of the copied generic tier, then a detection evaluation end to end.
_GENERIC_ALONE = """
import importlib, json, sys
sys.modules["polars_cv"] = None  # `import polars_cv` now raises ImportError
sys.path.insert(0, sys.argv[1])
for name in json.loads(sys.argv[2]):
    importlib.import_module(name)
import polars as pl
from generic_metrics._matching._prematched import PreMatchedAdapter
from generic_metrics._metrics import average_precision, froc_auc
detections, images = (pl.DataFrame(d) for d in json.loads(sys.argv[3]))
table = PreMatchedAdapter().match(detections, image_id_col="image", image_meta=images)
print(json.dumps({
    "ap": average_precision(table),
    "froc": froc_auc(table, fp_range=(0.0, 1.0)).collect().row(0),
    "polars_cv": [
        m for m, loaded in sys.modules.items()
        if m.split(".")[0] == "polars_cv" and loaded is not None
    ],
}))
"""

_DETECTIONS = {
    "image": ["a", "a", "a", "b", "b", "c"],
    "score": [0.9, 0.6, 0.2, 0.8, 0.3, 0.7],
    "is_tp": [True, False, True, True, False, False],
}
#: The population, one image of it without a detection.
_IMAGES = {"image_id": ["a", "b", "c", "d"], "n_gts": [2, 1, 1, 0]}


def test_the_generic_tier_runs_without_polars_cv(tmp_path: Path) -> None:
    """The generic tier, copied out on its own, evaluates detections.

    The run-time form of rule (b), and what "the split is mechanical" means:
    the generic modules (the scanner's own answer, not a second list) are
    copied into a package of another name beside nothing else, ``polars_cv``
    is made unimportable, and every module imports and a pre-matched
    evaluation gives the numbers the installed package gives. A source scan
    cannot see ``importlib`` or ``getattr``; this can.
    """
    import polars as pl

    from polars_cv.metrics import PreMatchedAdapter, average_precision, froc_auc

    names = generic_modules(MODULES)
    root = tmp_path / "generic_metrics"
    copied = []
    for name in names:
        is_package = MODULES[name][1]
        parts = name.removeprefix(DOWNSTREAM).lstrip(".").split(".")
        target = root.joinpath(*parts, "__init__.py" if is_package else "")
        target = target if is_package else target.with_suffix(".py")
        for parent in list(target.relative_to(root).parents)[:-1]:
            package = ".".join([DOWNSTREAM, *parent.parts])
            assert package in names, (
                f"generic module {name} lives in the package {package}, which "
                "is not generic: importing it loads polars-cv"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(MODULES[name][0])
        copied.append("generic_metrics." + ".".join(parts))
    # The facade re-exports both tiers and stays behind; the copy's root is empty.
    (root / "__init__.py").write_text("")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _GENERIC_ALONE,
            str(tmp_path),
            json.dumps(copied),
            json.dumps([_DETECTIONS, _IMAGES]),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    got = json.loads(result.stdout)
    assert got["polars_cv"] == [], f"the generic tier loaded {got['polars_cv']}"

    table = PreMatchedAdapter().match(
        pl.DataFrame(_DETECTIONS),
        image_id_col="image",
        image_meta=pl.DataFrame(_IMAGES),
    )
    assert got["ap"] == average_precision(table)
    assert tuple(got["froc"]) == froc_auc(table, fp_range=(0.0, 1.0)).collect().row(0)
