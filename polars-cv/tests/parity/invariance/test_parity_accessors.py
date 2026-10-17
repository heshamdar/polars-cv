"""A ``.contour`` accessor and the pipeline op it is give one answer.

Each op-backed accessor (``geom_fns::OP_ACCESSORS``: the accessor methods
whose function is a pipeline op) is that op's definition, and each runs the
same engine function, but through its own plugin function rather than the
pipeline's executor. So for drawn contour sets — traced regions, their
``boundary()`` lines, a mix, null rows — and the op's drawn arguments, the
accessor and ``Pipeline().source("contour").<op>(...)`` must return the same
rows, refuse the same rows (under ``on_error("raise")``) and null the same
rows (under ``on_error("null")``).

The pipeline returns a measure (``area``, ``centroid``, ``bounding_box``,
...) as one flat vector per row, the accessor as one value or struct per
member; both are compared flattened, in field order.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
from hypothesis import event
from hypothesis import strategies as st

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA
from tests.conftest import plugin_required
from tests.parity.framework.budget import property_lanes
from tests.parity.framework.checks import ParityFailure
from tests.parity.framework.images import sides
from tests.parity.framework.oracle import OPS
from tests.parity.framework.run import Axes, Step, execute

pytestmark = plugin_required


def _op_accessors() -> dict[str, str]:
    """Pipeline method -> accessor method, for every op-backed accessor: a
    ``.contour`` function whose name is an op's (the catalogues' own link)."""
    from polars_cv._lib import geom_catalog, op_catalog

    ops = {op["name"]: op["python"] for op in json.loads(op_catalog())}
    return {
        ops[fn["name"]]: fn["python"]
        for fn in json.loads(geom_catalog())
        if fn["namespace"] == "contour" and fn["name"] in ops
    }


ACCESSORS = _op_accessors()


def test_the_sweep_covers_every_op_backed_accessor() -> None:
    """Guard the guard: the catalogue scan finds the accessors, and each has
    arguments to draw (a reference entry, or none to take)."""
    assert len(ACCESSORS) >= 16, f"only {sorted(ACCESSORS)} found"
    unparametrized = sorted(m for m in ACCESSORS if m not in OPS and m not in _NO_ARGS)
    assert not unparametrized, f"no argument strategy for {unparametrized}"


#: Op-backed accessors with no reference entry (``oracle.EXEMPT``), whose
#: arguments are therefore not drawn there: none to take.
_NO_ARGS = frozenset({"close"})


def _point_list(points: Any) -> list[dict[str, float]]:
    return [{"x": float(p["x"]), "y": float(p["y"])} for p in points]


@st.composite
def _contour_rows(draw: st.DrawFn) -> list[list[dict] | None]:
    """1-3 rows, each a traced region set, its lines, a mix, or null."""
    rows: list[list[dict] | None] = []
    for i in range(draw(st.integers(1, 3), label="rows")):
        kind = draw(
            st.sampled_from(["regions", "lines", "mixed", "null"]), label=f"row {i}"
        )
        if kind == "null":
            rows.append(None)
            continue
        h, w = draw(sides(1, 10)), draw(sides(1, 10))
        rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
        mask = (rng.random((h, w)) < draw(st.floats(0.1, 0.9))).astype(np.uint8) * 255
        steps = [
            Step(
                "extract_contours",
                {"method": draw(st.sampled_from(["none", "simple"]))},
            )
        ]
        if kind == "lines":
            steps.append(Step("boundary"))
        (traced,) = execute([mask[:, :, None]], steps, Axes(sink="native")).rows
        members = [
            {
                **c,
                "exterior": _point_list(c["exterior"]),
                "holes": [_point_list(h) for h in c["holes"]],
            }
            for c in traced
        ]
        if kind == "mixed" and members:
            (lines,) = execute(
                [mask[:, :, None]], [*steps, Step("boundary")], Axes(sink="native")
            ).rows
            members.append(
                {**lines[0], "exterior": _point_list(lines[0]["exterior"]), "holes": []}
            )
        rows.append(members)
    return rows


def _flat(value: Any) -> Any:
    """A measure as flat floats, in field order; anything else unchanged."""
    if isinstance(value, dict):
        return [x for v in value.values() for x in _flat_list(v)]
    return value


def _flat_list(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [x for v in value.values() for x in _flat_list(v)]
    if isinstance(value, list):
        return [x for v in value for x in _flat_list(v)]
    return [float(value)]


def _outcome(run) -> tuple[str, Any]:
    try:
        return "rows", run().to_series().to_list()
    except pl.exceptions.ComputeError as e:
        return "refused", type(e).__name__
    except pl.exceptions.SchemaError as e:
        return "refused", type(e).__name__


@property_lanes(weight=3, rows=_contour_rows(), data=st.data())
def test_an_accessor_answers_as_its_pipeline_op(rows, data) -> None:
    method = data.draw(st.sampled_from(sorted(ACCESSORS)), label="op")
    accessor = ACCESSORS[method]
    params = (
        {}
        if method in _NO_ARGS
        else data.draw(
            st.composite(lambda d: OPS[method].params(d, None))(), label="args"
        )
    )
    on_error = data.draw(st.sampled_from(["raise", "null"]), label="on_error")
    frame = pl.DataFrame({"c": rows}, schema={"c": CONTOUR_SET_SCHEMA})
    pipe = getattr(Pipeline().source("contour").on_error(on_error), method)(**params)
    by_pipeline = _outcome(
        lambda: frame.select(pl.col("c").cv.pipe(pipe).sink("native"))
    )
    namespace = pl.col("c").contour.on_error(on_error)
    by_accessor = _outcome(lambda: frame.select(getattr(namespace, accessor)(**params)))
    event(f"{method}: {by_pipeline[0]}")
    if by_pipeline[0] == "rows" and by_accessor[0] == "rows":
        vector = method in OPS and OPS[method].domain_out == "vector"
        left = [
            _flat_list(r) if vector and r is not None else r for r in by_pipeline[1]
        ]
        right = [
            _flat_list(r) if vector and r is not None else r for r in by_accessor[1]
        ]
        if left == right:
            return
    elif by_pipeline == by_accessor:
        return
    raise ParityFailure(
        f"{method}({params}) on_error={on_error!r}:\n"
        f"  pipeline: {by_pipeline}\n  .contour.{accessor}: {by_accessor}"
    )
