"""No operation panics on an input shape the planner could not see (CR-34).

The planner checks shapes when it knows them. A ``blob`` source carries its
shape per row, so the engine meets shapes the plan never saw — and it used to
meet some of them by indexing a missing axis and panicking. Every op now
declares what it accepts (``Op::validate``, a required contract) and the
executor checks it before running the op, so such a row is an ordinary error
that ``on_error`` handles.

The sweep runs every buffer op in ``OP_CASES`` (plus the binary operands)
over a matrix of ranks, channel counts and dtypes. An op may reject an input,
at plan time or as a row error; it may not panic. Caught panics are labelled
"the engine panicked" precisely so this can tell the two apart.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline

from ._op_cases import BUFFER, EXTRA_CASES, OP_CASES
from .conftest import plugin_required

PANIC = "the engine panicked"

SHAPES: list[tuple[int, ...]] = [
    (6,),
    (5, 4),
    (5, 4, 1),
    (5, 4, 2),
    (5, 4, 3),
    (5, 4, 4),
    (5, 4, 5),
    (2, 5, 4, 3),
]
DTYPES = [np.uint8, np.float32]


def _blobs(arrays: list[np.ndarray]) -> pl.Series:
    """Encode arrays as VIEW blobs through the plugin's own blob sink."""
    out = []
    for arr in arrays:
        col = pl.Series("a", arr[np.newaxis])
        if arr.ndim == 1:
            col = col.cast(pl.Array(col.dtype.inner, arr.shape[0]))  # ty: ignore[unresolved-attribute]
        blob = pl.DataFrame({"a": col}).select(
            b=pl.col("a").cv.pipe(Pipeline().source("array")).sink("blob")
        )["b"][0]
        out.append(blob)
    return pl.Series("b", out, dtype=pl.Binary)


@pytest.fixture(scope="module")
def blob_frame() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    arrays = [
        (rng.random(shape) * 255).astype(dtype) for shape in SHAPES for dtype in DTYPES
    ]
    return pl.DataFrame({"b": _blobs(arrays)})


def _cases() -> list[tuple[str, dict]]:
    cases = [
        (op, kw)
        for op, spec in OP_CASES.items()
        if spec
        for dom, kw in [spec]
        if dom == BUFFER
    ]
    cases += [(op, kw) for op, dom, kw in EXTRA_CASES if dom == BUFFER]
    return cases


def _row_errors(frame: pl.DataFrame, expr_for) -> list[str]:  # noqa: ANN001
    """Run one row at a time so each row's outcome is visible."""
    errors = []
    for i in range(frame.height):
        try:
            expr = expr_for()
        except Exception:  # noqa: BLE001 — rejected at plan time: fine
            return []
        try:
            frame.slice(i, 1).select(o=expr)
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
    return errors


@plugin_required
@pytest.mark.parametrize(("op", "kwargs"), _cases(), ids=lambda v: str(v))
def test_no_op_panics_on_any_input_shape(
    blob_frame: pl.DataFrame, op: str, kwargs: dict
) -> None:
    def expr() -> pl.Expr:
        pipe = getattr(Pipeline().source("blob"), op)(**kwargs)
        return (
            pl.col("b")
            .cv.pipe(pipe)
            .sink("blob" if pipe._current_domain == "buffer" else "native")
        )

    panics = [e for e in _row_errors(blob_frame, expr) if PANIC in e]
    assert not panics, f"{op}{kwargs} panicked:\n" + "\n".join(p[:300] for p in panics)


@plugin_required
@pytest.mark.parametrize(
    "op",
    [
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "blend",
        "ratio",
        "bitwise_and",
        "apply_mask",
        "channel_merge",
    ],
)
def test_no_binary_op_panics_on_mismatched_operands(
    blob_frame: pl.DataFrame, op: str
) -> None:
    n = blob_frame.height
    pairs = pl.DataFrame(
        {
            "a": [blob_frame["b"][i] for i in range(n) for _ in range(n)],
            "b": [blob_frame["b"][j] for _ in range(n) for j in range(n)],
        }
    )

    def expr() -> pl.Expr:
        src = Pipeline().source("blob")
        left = pl.col("a").cv.pipe(src)
        right = pl.col("b").cv.pipe(src)
        return getattr(left, op)(right).sink("blob")

    panics = [e for e in _row_errors(pairs, expr) if PANIC in e]
    assert not panics, f"{op} panicked on {len(panics)} operand pairs:\n" + "\n".join(
        sorted({p[:200] for p in panics})
    )
