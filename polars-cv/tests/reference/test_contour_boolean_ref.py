"""Reference tests for the contour boolean ops (CR-99).

``union``, ``intersection``, ``difference`` and ``symmetric_difference`` combine
two contour sets, each read as the union of its regions; the result is a set
of disjoint regions (two shapes can split or merge). Each is a lazy-only
pipeline op (the other operand another pipeline's contour node) and a
``.contour`` accessor (the other operand a contour column), one engine function
behind both.

Oracles, independent of the engine:

* The mask identity: rasterizing ``a op b`` equals the pixelwise ``|``, ``&``,
  ``& ~`` or ``^`` of rasterizing ``a`` and ``b`` (``rasterize`` paints a set's
  union), exactly on integer rectilinear shapes, whose edges put no pixel
  centre in doubt; and the result's members are disjoint (their areas add up
  to its pixels).
* Area identities on float shapes: inclusion-exclusion, and
  ``|a ∩ b| / |a ∪ b|`` equal to ``.contour.iou``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_column
from polars_cv.geometry.schemas import CONTOUR_SCHEMA, CONTOUR_SET_SCHEMA
from tests.conftest import plugin_required

pytestmark = plugin_required

OPS = {
    "union": np.logical_or,
    "intersection": np.logical_and,
    "difference": lambda a, b: a & ~b,
    "symmetric_difference": np.logical_xor,
}


def _pts(points) -> list[dict]:
    return [{"x": float(x), "y": float(y)} for x, y in points]


def _region(points, holes=()) -> dict:
    return {
        "exterior": _pts(points),
        "holes": [_pts(h) for h in holes],
        "is_closed": True,
    }


def _rect(x0, y0, x1, y1) -> list[tuple[float, float]]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _frame(a: list, b: list) -> pl.DataFrame:
    return pl.DataFrame(
        {"a": [a], "b": [b]}, schema={"a": CONTOUR_SET_SCHEMA, "b": CONTOUR_SET_SCHEMA}
    )


def _src(column: str):
    return pl.col(column).cv.pipe(Pipeline().source("contour"))


def _lazy(a: list, b: list, op: str) -> list[dict]:
    """``a op b`` through the pipeline op."""
    expr = getattr(_src("a"), op)(_src("b")).sink("native")
    return _frame(a, b).select(expr).to_series().to_list()[0]


def _accessor(a: list, b: list, op: str) -> list[dict]:
    """``a op b`` through the ``.contour`` accessor."""
    out = _frame(a, b).select(getattr(pl.col("a").contour, op)(pl.col("b"))).item()
    return out.to_list()


def _mask(contours: list, w: int = 44, h: int = 44) -> np.ndarray:
    df = pl.DataFrame({"c": [contours]}, schema={"c": CONTOUR_SET_SCHEMA})
    out = df.select(
        pl.col("c")
        .cv.pipe(Pipeline().source("contour").rasterize(width=w, height=h))
        .sink("numpy")
    )
    (mask,) = numpy_from_column(out.to_series(), copy=True)
    return mask[:, :, 0] > 0


def _area(contours: list) -> float:
    if not contours:
        return 0.0
    df = pl.DataFrame({"c": [contours]}, schema={"c": CONTOUR_SET_SCHEMA})
    pipe = Pipeline().source("contour").area()
    return float(sum(df.select(pl.col("c").cv.pipe(pipe).sink("native")).item()))


def _random_rects(rng, k: int) -> list[dict]:
    out = []
    for _ in range(k):
        x0, y0 = rng.integers(0, 32, 2)
        x1, y1 = x0 + rng.integers(1, 12), y0 + rng.integers(1, 12)
        holes = []
        if x1 - x0 >= 5 and y1 - y0 >= 5 and rng.random() < 0.4:
            holes = [_rect(x0 + 2, y0 + 2, x1 - 2, y1 - 2)]
        out.append(_region(_rect(x0, y0, x1, y1), holes))
    return out


def _star(rng, cx, cy, r, k=9) -> list[tuple[float, float]]:
    angles = np.sort(rng.uniform(0, 2 * math.pi, k))
    radii = rng.uniform(0.45 * r, r, k)
    return list(
        zip(cx + radii * np.cos(angles), cy + radii * np.sin(angles), strict=True)
    )


# ---------------------------------------------------------------------------
# The mask identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("seed", range(6))
def test_each_op_paints_the_pixelwise_op(op: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    a, b = _random_rects(rng, 4), _random_rects(rng, 4)
    result = _lazy(a, b, op)
    assert all(c["is_closed"] for c in result)
    expected = OPS[op](_mask(a), _mask(b))
    np.testing.assert_array_equal(_mask(result), expected)
    # Disjoint members: their areas add up to the painted pixels.
    assert _area(result) == pytest.approx(int(expected.sum()))


# ---------------------------------------------------------------------------
# Area identities on float shapes
# ---------------------------------------------------------------------------

#: The boolean engine (``i_overlay``, under ``geo``) snaps coordinates to an
#: integer grid spanning the shapes, about 2**-30 of their extent, so two
#: routes to one area agree to about 1e-9 relative, not to the last bit.
REL = 1e-7


@pytest.mark.parametrize("seed", range(4))
def test_inclusion_exclusion(seed: int) -> None:
    rng = np.random.default_rng(seed + 10)
    a = [_region(_star(rng, 15.2, 16.1, 10.3)), _region(_star(rng, 22.7, 14.9, 8.1))]
    b = [_region(_star(rng, 19.4, 20.3, 11.6))]
    whole_a = _area(_lazy(a, [], "union"))
    whole_b = _area(_lazy(b, [], "union"))
    union = _area(_lazy(a, b, "union"))
    inter = _area(_lazy(a, b, "intersection"))
    assert inter > 0, "a vacuous fixture: the shapes do not overlap"
    assert union + inter == pytest.approx(whole_a + whole_b, rel=REL)
    assert _area(_lazy(a, b, "difference")) == pytest.approx(whole_a - inter, rel=REL)
    assert _area(_lazy(a, b, "symmetric_difference")) == pytest.approx(
        union - inter, rel=REL
    )


@pytest.mark.parametrize("seed", range(4))
def test_intersection_over_union_is_iou(seed: int) -> None:
    rng = np.random.default_rng(seed + 20)
    a, b = _region(_star(rng, 15.0, 15.0, 10.0)), _region(_star(rng, 18.0, 17.0, 10.0))
    df = pl.DataFrame(
        {"a": [a], "b": [b]}, schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA}
    )
    iou = df.select(pl.col("a").contour.iou(pl.col("b"))).item()
    ratio = _area(_lazy([a], [b], "intersection")) / _area(_lazy([a], [b], "union"))
    assert ratio == pytest.approx(iou, rel=REL)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


RING = _region(_rect(0, 0, 20, 20), [_rect(5, 5, 15, 15)])
ISLAND = _region(_rect(8, 8, 12, 12))
SQUARE = _region(_rect(0, 0, 10, 10))


def test_a_shape_in_a_hole_does_not_intersect() -> None:
    assert _lazy([RING], [ISLAND], "intersection") == []
    assert sorted(_area([c]) for c in _lazy([RING], [ISLAND], "union")) == [16.0, 300.0]


def test_difference_cuts_a_hole() -> None:
    (holed,) = _lazy([_region(_rect(0, 0, 20, 20))], [ISLAND], "difference")
    assert len(holed["holes"]) == 1
    assert _area([holed]) == pytest.approx(400.0 - 16.0)


def test_each_side_is_the_union_of_its_set() -> None:
    """Overlapping members on one side count once: ``a - a`` is empty."""
    overlapping = [_region(_rect(0, 0, 6, 6)), _region(_rect(3, 3, 9, 9))]
    assert _lazy(overlapping, overlapping, "difference") == []
    assert _area(_lazy(overlapping, overlapping, "intersection")) == pytest.approx(63.0)


@pytest.mark.parametrize(
    ("op", "left", "right"),
    [
        ("union", 100.0, 100.0),
        ("intersection", 0.0, 0.0),
        ("difference", 100.0, 0.0),
        ("symmetric_difference", 100.0, 100.0),
    ],
)
def test_an_empty_side(op: str, left: float, right: float) -> None:
    """``s op ∅`` (left) and ``∅ op s`` (right), for a 10 x 10 ``s``."""
    assert _area(_lazy([SQUARE], [], op)) == pytest.approx(left)
    assert _area(_lazy([], [SQUARE], op)) == pytest.approx(right)


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("side", ["left", "right"])
def test_a_line_on_either_side_is_refused(op: str, side: str) -> None:
    line = {"exterior": _pts([(0, 0), (5, 5)]), "holes": [], "is_closed": False}
    a, b = ([line], [SQUARE]) if side == "left" else ([SQUARE], [line])
    for run in (_lazy, _accessor):
        with pytest.raises(pl.exceptions.ComputeError, match="open contour.*no region"):
            run(a, b, op)


# ---------------------------------------------------------------------------
# Surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", sorted(OPS))
def test_the_accessor_is_the_pipeline_op(op: str) -> None:
    rng = np.random.default_rng(3)
    a, b = _random_rects(rng, 3), _random_rects(rng, 3)
    assert _accessor(a, b, op) == _lazy(a, b, op)


def test_single_contour_columns_give_a_set() -> None:
    df = pl.DataFrame(
        {"a": [SQUARE], "b": [_region(_rect(20, 0, 30, 10))]},
        schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
    )
    out = df.select(pl.col("a").contour.union(pl.col("b")))
    assert out.schema["a"] == CONTOUR_SET_SCHEMA
    assert len(out.item()) == 2


def test_a_null_side_nulls_the_row() -> None:
    df = pl.DataFrame(
        {"a": [[SQUARE], None, [SQUARE]], "b": [[SQUARE], [SQUARE], None]},
        schema={"a": CONTOUR_SET_SCHEMA, "b": CONTOUR_SET_SCHEMA},
    )
    out = df.select(pl.col("a").contour.intersection(pl.col("b"))).to_series()
    assert out.null_count() == 2 and out[0] is not None


def test_the_result_chains_into_contour_ops() -> None:
    expr = _src("a").difference(_src("b")).area().sink("native")
    a, b = [_region(_rect(0, 0, 10, 10))], [_region(_rect(5, 0, 15, 10))]
    assert _frame(a, b).select(expr).item().to_list() == pytest.approx([50.0])


def test_a_buffer_operand_is_refused_at_build() -> None:
    image = pl.col("img").cv.pipe(Pipeline().source("image_bytes"))
    with pytest.raises(ValueError, match="contour"):
        _src("a").union(image)
