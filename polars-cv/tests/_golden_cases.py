"""The golden behaviour corpus: every op's planned and executed result.

The typed-op migration (``TYPED_OPS_PLAN.md``) rewrites how every operation is
declared, planned and dispatched, and deletes the tests whose only job was to
keep two declarations in sync. This corpus is what makes that safe: it records
what each operation *does* — its planned state, its output schema and its
output values — and ``test_golden_corpus.py`` requires every later commit to
reproduce it.

Three kinds of case, each with a stable id:

- ``op/<method>[...]`` — every row of ``_op_cases.OP_CASES`` and
  ``EXTRA_CASES`` on a 100x200 RGB image (single-channel ops get a
  ``grayscale()`` first, as the schema-parity sweeps do).
- ``expr/<method>.<param>`` — every expression-eligible parameter from
  ``_expr_param_cases.CASES``, driven per row from a column.
- ``reject/<name>`` — inputs that must fail, with the exception class and a
  stable substring of the message. **Before a migration phase deletes a guard
  test, its rejection cases are added here** (the plan's rule for every
  deletion-matrix row).

Recorded values are results only, never internal names, so a renamed method
(plan phase P8) edits a builder here and not the fixture. The fixture stores
exact digests *and* tolerance-comparable samples: the digest is compared only
on the platform the fixture was recorded on, the samples everywhere.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import polars as pl

from polars_cv import Pipeline, numpy_from_struct
from tests._expr_param_cases import CASES as EXPR_CASES
from tests._expr_param_runner import PARAM, input_frame
from tests._op_cases import (
    BUFFER,
    EXTRA_CASES,
    OP_CASES,
    SINGLE_CHANNEL_OPS,
    base_pipeline,
)
from tests._plan_view import planned
from tests.conftest import make_image_png

#: How many evenly spaced numeric leaves a fingerprint keeps for tolerance
#: comparison. Enough to catch a permutation or a shifted window; small enough
#: to keep the fixture reviewable.
SAMPLES = 32


@dataclass(frozen=True)
class GoldenCase:
    """One corpus entry: how to run it, and what it must raise if anything."""

    case_id: str
    run: Callable[[], "Outcome"]
    #: For ``reject/`` cases: a substring the error message must contain.
    expect: str | None = None


Outcome = dict[str, Any]


# --- Fingerprints ------------------------------------------------------------


def _leaves(value: Any, out: list[float], shape: list[str]) -> None:
    """Flatten *value* into numeric leaves, recording its structure."""
    if value is None:
        shape.append("null")
    elif isinstance(value, dict):
        shape.append("{" + ",".join(sorted(value)) + "}")
        for key in sorted(value):
            _leaves(value[key], out, shape)
    elif isinstance(value, (list, tuple)):
        shape.append(f"[{len(value)}]")
        for item in value:
            _leaves(item, out, shape)
    elif isinstance(value, (bytes, bytearray)):
        shape.append(f"b{len(value)}")
        out.extend(value)
    elif isinstance(value, str):
        shape.append(f"s:{value}")
    elif isinstance(value, bool):
        out.append(float(value))
    else:
        out.append(float(value))


def fingerprint(value: Any) -> dict[str, Any]:
    """A platform-comparable summary of one output row.

    ``structure`` and the array ``shape``/``dtype`` must match exactly;
    ``stats`` and ``samples`` are compared with a tolerance; ``sha256`` is an
    exact digest of the leaves, compared only on the recording platform.
    """
    if isinstance(value, dict) and set(value) >= {"data", "dtype", "shape"}:
        arr = numpy_from_struct(value)
        leaves = np.asarray(arr, dtype=np.float64).ravel()
        head = {"array": {"dtype": str(arr.dtype), "shape": list(arr.shape)}}
        structure = "ndarray"
        digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    else:
        flat: list[float] = []
        shape: list[str] = []
        _leaves(value, flat, shape)
        leaves = np.asarray(flat, dtype=np.float64)
        head = {}
        structure = " ".join(shape)
        digest = hashlib.sha256(leaves.tobytes()).hexdigest()
    finite = leaves[np.isfinite(leaves)] if leaves.size else leaves
    idx = np.linspace(0, leaves.size - 1, num=min(SAMPLES, leaves.size)).astype(int)
    return {
        **head,
        "structure": structure,
        "n": int(leaves.size),
        "stats": (
            [float(finite.sum()), float(finite.min()), float(finite.max())]
            if finite.size
            else []
        ),
        "samples": [_json_float(x) for x in leaves[idx]] if leaves.size else [],
        "sha256": digest,
    }


def _json_float(x: float) -> float | str:
    """JSON cannot carry NaN or infinities; spell them."""
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x)


def _execute(df: pl.DataFrame, column: str, pipe: Pipeline, sink: str) -> Outcome:
    """Plan and run *pipe*; the outcome is its plan, schema and row prints."""
    view = planned(pipe)
    expr = pl.col(column).cv.pipe(pipe).sink(sink)
    try:
        out = df.lazy().select(o=expr).collect(engine="in-memory")
    except Exception as exc:  # noqa: BLE001 - the class is the recorded fact
        return {"error": type(exc).__name__}
    return {
        "plan": {
            "domain": view.domain,
            "dtype": view.dtype,
            "ndim": view.ndim,
            "hwc": [view.height, view.width, view.channels],
        },
        "schema": str(out.schema["o"]),
        "rows": [fingerprint(v) for v in out["o"].to_list()],
    }


# --- op/ cases ---------------------------------------------------------------

_IMAGE = "image"


def _image_frame() -> pl.DataFrame:
    return pl.DataFrame({_IMAGE: [make_image_png(100, 200, 3, seed=11)]})


def _sink(pipe: Pipeline) -> str:
    return "numpy" if pipe.current_domain() == BUFFER else "native"


def _op_case(method: str, domain: str, kwargs: dict) -> Callable[[], Outcome]:
    def run() -> Outcome:
        try:
            base = base_pipeline(domain)
            if method in SINGLE_CHANNEL_OPS:
                base = base.grayscale()
            pipe = getattr(base, method)(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return {"error": type(exc).__name__}
        return _execute(_image_frame(), _IMAGE, pipe, _sink(pipe))

    return run


def _kwargs_id(kwargs: dict) -> str:
    return ",".join(f"{k}={kwargs[k]!r}" for k in sorted(kwargs))


def _op_cases() -> list[GoldenCase]:
    cases = [
        GoldenCase(f"op/{method}", _op_case(method, *case))
        for method, case in sorted(OP_CASES.items())
        if case is not None
    ]
    cases += [
        GoldenCase(
            f"op/{method}[{_kwargs_id(kwargs)}]", _op_case(method, domain, kwargs)
        )
        for method, domain, kwargs in EXTRA_CASES
    ]
    return cases


# --- expr/ cases -------------------------------------------------------------


def _expr_case(case: Any) -> Callable[[], Outcome]:
    def run() -> Outcome:
        pipe = case.build(pl.col(PARAM))
        sink = _sink(pipe) if pipe.current_domain() != BUFFER else "numpy"
        return _execute(input_frame(case, case.values), case.column, pipe, sink)

    return run


def _expr_cases() -> list[GoldenCase]:
    return [GoldenCase(f"expr/{c.key}", _expr_case(c)) for c in EXPR_CASES]


# --- reject/ cases -----------------------------------------------------------


def _reject(build: Callable[[], Any], *, run: bool = False) -> Callable[[], Outcome]:
    """An outcome that records the exception *build* (or running it) raises."""

    def outcome() -> Outcome:
        try:
            pipe = build()
            if run:
                _image_frame().select(pl.col(_IMAGE).cv.pipe(pipe).sink(_sink(pipe)))
        except Exception as exc:  # noqa: BLE001
            return {"error": type(exc).__name__, "message": str(exc)}
        return {"error": None}

    return outcome


def _src() -> Pipeline:
    return Pipeline().source("image_bytes")


_REJECTIONS: list[tuple[str, Callable[[], Outcome], str]] = [
    (
        "unknown_enum_value",
        _reject(lambda: _src().resize(height=8, width=8, filter="bogus")),
        "bogus",
    ),
    (
        "structural_param_rejects_expression",
        _reject(lambda: _src().cast(pl.col("d"))),  # type: ignore[arg-type]
        "",
    ),
    (
        "unknown_keyword",
        _reject(lambda: _src().blur(sigma=1.0, radius=3)),  # type: ignore[call-arg]
        "radius",
    ),
    (
        "op_on_the_wrong_domain",
        _reject(lambda: _src().area()),
        "",
    ),
    (
        "negative_crop_offset",
        _reject(lambda: _src().crop(top=-1, left=0, height=4, width=4)),
        "negative",
    ),
    (
        "crop_window_outside_the_image",
        _reject(lambda: _src().crop(top=90, left=0, height=20, width=4), run=True),
        "outside",
    ),
    (
        "singular_affine_matrix",
        _reject(
            lambda: _src().warp_affine(
                matrix=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], output_size=(8, 8)
            ),
            run=True,
        ),
        "singular",
    ),
    (
        "cast_to_an_unknown_dtype",
        _reject(lambda: _src().cast("f128")),
        "f128",
    ),
]


def _reject_cases() -> list[GoldenCase]:
    return [
        GoldenCase(f"reject/{name}", run, expect=expect)
        for name, run, expect in _REJECTIONS
    ]


def golden_cases() -> list[GoldenCase]:
    """Every corpus case, in a stable order, ids unique."""
    cases = _op_cases() + _expr_cases() + _reject_cases()
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        msg = f"duplicate golden case ids: {dupes}"
        raise AssertionError(msg)
    return cases
