"""The builder call surface is pinned: a change to how a method is called is
deliberate or it is a bug.

Every public method of ``Pipeline`` and ``LazyPipelineExpr`` keeps the
parameter names, kinds and defaults recorded in ``tests/golden/signatures.json``.
The generated methods follow one rule (``gen_ops.positional``: an op's only
required parameter is positional-or-keyword, everything else keyword-only), so
a catalogue change that moves a parameter's kind shows up here as a diff to
review. Pipelines also keep surviving pickle and copy, which today's
plain-Python objects do and a compiled planner object would not by default.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import polars as pl
import pytest

from polars_cv import Pipeline
from tests._plan_view import state_of
from tests.conftest import make_image_png, plugin_required

_PKG = Path(__file__).resolve().parents[1]
FIXTURE = _PKG / "tests" / "golden" / "signatures.json"


def _surface() -> dict:
    spec = importlib.util.spec_from_file_location(
        "gen_signature_snapshot", _PKG / "scripts" / "gen_signature_snapshot.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.surface()


def test_the_call_surface_matches_the_snapshot() -> None:
    want = json.loads(FIXTURE.read_text())
    got = _surface()
    for cls in sorted(set(want) | set(got)):
        w, g = want.get(cls, {}), got.get(cls, {})
        assert sorted(g) == sorted(w), (
            f"{cls}: methods added {sorted(set(g) - set(w))}, "
            f"removed {sorted(set(w) - set(g))}"
        )
        for name in w:
            assert g[name] == w[name], f"{cls}.{name}: {g[name]} != {w[name]}"


def _pipe() -> Pipeline:
    return (
        Pipeline()
        .source("image_bytes", dtype="u8")
        .resize(height=pl.col("h"), width=8)
        .blur(1.0)
    )


def _run(pipe: Pipeline) -> list:
    df = pl.DataFrame({"img": [make_image_png(16, 16, 3, seed=2)] * 2, "h": [6, 9]})
    return df.select(pl.col("img").cv.pipe(pipe).sink("list"))["img"].to_list()


@plugin_required
@pytest.mark.parametrize(
    "roundtrip",
    [
        pytest.param(lambda p: pickle.loads(pickle.dumps(p)), id="pickle"),
        pytest.param(copy.deepcopy, id="deepcopy"),
        pytest.param(copy.copy, id="copy"),
    ],
)
def test_a_pipeline_survives_pickle_and_copy(roundtrip) -> None:
    original = _pipe()
    assert _run(roundtrip(original)) == _run(original)


@plugin_required
def test_a_lazy_expression_survives_pickle_and_deepcopy() -> None:
    lazy = pl.col("img").cv.pipe(_pipe())
    for clone in (pickle.loads(pickle.dumps(lazy)), copy.deepcopy(lazy)):
        df = pl.DataFrame({"img": [make_image_png(16, 16, 3, seed=2)], "h": [6]})
        out = df.select(clone.sink("list"))
        assert out.height == 1


@plugin_required
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: (
                Pipeline().source("list", dtype="u8").assert_shape(dims=[2, 3, 4, 5])
            ),
            id="rank-4",
        ),
        pytest.param(
            lambda: Pipeline().source("list", dtype="u8").assert_shape(channels=3),
            id="unranked-leading",
        ),
        pytest.param(lambda: Pipeline().source("image_bytes"), id="rank-3-unknown"),
    ],
)
def test_a_planned_state_survives_pickle(build) -> None:
    """The planned shape crosses its pickle form whole: rank and every size."""
    state = state_of(build())
    back = pickle.loads(pickle.dumps(state))
    assert back == state
    assert (back.ndim, back.dims) == (state.ndim, state.dims)
