"""The one-to-one correspondence rule between contour / bbox sets, pinned.

Correspondence is greedy: candidates are walked in a caller-supplied order and
each claims the highest-overlap target not already claimed, provided that
overlap clears the threshold. Four properties decide what comes out, and until
this file existed none of them was tested anywhere:

- the **order** decides who wins a contested target,
- the pairing is **one-to-one** -- a claimed target is gone,
- **ties** resolve to the smallest target index, and
- the threshold is **inclusive** (``>=``).

Two independent implementations are compared throughout, in the idiom of
``test_contour_raster_crosscheck.py``: the engine computes the assignment, and
``_reference`` recomputes it in plain Python from the *engine's own* pairwise
overlap matrix. A fault in either shows up as a disagreement. Plain Python is
fine here -- the no-UDF rule binds shipped plugin code, not test oracles.

Everything below is written in the vocabulary of the correspondence relation
(``order``, ``right_idx``, ``overlap``), never in detection vocabulary. Only
``_run`` and ``_extract`` touch the accessor being tested; they are the whole
of what a rename has to move.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv import BBOX_SCHEMA
from polars_cv.geometry import CONTOUR_SET_SCHEMA
from tests.conftest import plugin_required

# ---------------------------------------------------------------------------
# Geometry fixtures
# ---------------------------------------------------------------------------


def _rect(x: float, y: float, w: float, h: float) -> dict[str, object]:
    """An axis-aligned rectangle as a contour."""
    return {
        "exterior": [
            {"x": x, "y": y},
            {"x": x + w, "y": y},
            {"x": x + w, "y": y + h},
            {"x": x, "y": y + h},
        ],
        "holes": [],
        "is_closed": True,
    }


def _bbox(x: float, y: float, w: float, h: float) -> dict[str, float]:
    """The same rectangle as a bounding box."""
    return {"x": x, "y": y, "width": w, "height": h}


def _l_shape(dx: float, dy: float) -> dict[str, object]:
    """A concave L, area 7500. Ported from view-buffer's pairwise tests."""
    pts = [(0, 0), (100, 0), (100, 50), (50, 50), (50, 100), (0, 100)]
    return {
        "exterior": [{"x": px + dx, "y": py + dy} for px, py in pts],
        "holes": [],
        "is_closed": True,
    }


def _u_shape() -> dict[str, object]:
    """A reflex-notched U. Ported from view-buffer's pairwise tests."""
    pts = [
        (0, 0), (100, 0), (100, 100), (70, 100),
        (70, 40), (30, 40), (30, 100), (0, 100),
    ]
    return {
        "exterior": [{"x": float(px), "y": float(py)} for px, py in pts],
        "holes": [],
        "is_closed": True,
    }


# ---------------------------------------------------------------------------
# The reference implementation
# ---------------------------------------------------------------------------

#: Overlaps closer than this count as equal when picking a target, mirroring
#: the engine. Rectangles clipped two different ways can land a few ULPs apart
#: while being the same area, and without a tolerance the tie rule would be
#: decided by which way the clip rounded.
_TIE_EPS = 1e-12


def _reference(
    matrix: list[list[float]], threshold: float, order: list[int] | None
) -> tuple[list[int | None], list[float]]:
    """Greedy one-to-one assignment over an overlap matrix, in plain Python.

    The independent half of the crosscheck. Walks candidates in *order*; each
    takes the unclaimed target with the highest overlap, ties going to the
    smallest target index, and keeps it only if that overlap is ``>=``
    *threshold*.

    Returns ``(right_idx, overlap)``, both positionally aligned with the left
    set -- ``right_idx[i]`` is ``None`` where candidate ``i`` went unmatched.
    """
    n_left = len(matrix)
    n_right = len(matrix[0]) if n_left else 0
    claimed = [False] * n_right
    right_idx: list[int | None] = [None] * n_left
    overlap = [0.0] * n_left

    for i in order if order is not None else range(n_left):
        best_j, best = None, -1.0
        for j in range(n_right):
            if claimed[j]:
                continue
            candidate = matrix[i][j]
            if candidate > best:
                best, best_j = candidate, j
            elif abs(candidate - best) < _TIE_EPS and best_j is not None and j < best_j:
                # Overlaps that differ only in the last bits count as tied, so
                # a rectangle clipped two ways does not silently pick a winner
                # by floating-point noise.
                best_j = j
        if best_j is not None and best >= threshold:
            claimed[best_j] = True
            right_idx[i] = best_j
            overlap[i] = best
    return right_idx, overlap


# ---------------------------------------------------------------------------
# The adapter -- the ONLY part that a rename of the accessor has to move
# ---------------------------------------------------------------------------


def _order_to_scores(order: list[int], n: int) -> list[float]:
    """Scores whose descending sort reproduces *order*.

    The accessor currently takes confidence scores and derives the walk order
    from them. The tests are written in terms of the order itself, because that
    is the thing the rule actually depends on -- so this converts.
    """
    scores = [0.0] * n
    for rank, idx in enumerate(order):
        scores[idx] = float(n - rank)
    return scores


def _run(
    left: list[object],
    right: list[object],
    *,
    kind: str,
    threshold: float,
    order: list[int] | None,
) -> pl.DataFrame:
    """Apply the correspondence accessor to one row."""
    item = CONTOUR_SET_SCHEMA if kind == "contour" else pl.List(BBOX_SCHEMA)
    data: dict[str, object] = {"left": [left], "right": [right]}
    schema: dict[str, object] = {"left": item, "right": item}
    if order is not None:
        data["order"] = [_order_to_scores(order, len(left))]
        schema["order"] = pl.List(pl.Float64)
    df = pl.DataFrame(data, schema=schema)

    ns = pl.col("left").contour if kind == "contour" else pl.col("left").bbox
    return df.with_columns(
        _c=ns.match_detections(
            pl.col("right"),
            threshold=threshold,
            scores=pl.col("order") if order is not None else None,
        )
    )


def _extract(out: pl.DataFrame) -> tuple[list[int | None], list[float]]:
    """Read ``(right_idx, overlap)`` out of the accessor's struct."""
    cell = out["_c"][0]
    return list(cell["gt_idx"]), list(cell["iou"])


def _matrix(
    left: list[object], right: list[object], *, kind: str
) -> list[list[float]]:
    """The engine's own pairwise overlap matrix, for the crosscheck."""
    item = CONTOUR_SET_SCHEMA if kind == "contour" else pl.List(BBOX_SCHEMA)
    df = pl.DataFrame(
        {"left": [left], "right": [right]}, schema={"left": item, "right": item}
    )
    ns = pl.col("left").contour if kind == "contour" else pl.col("left").bbox
    out = df.with_columns(_m=ns.pairwise_iou(pl.col("right")))
    return [list(row) for row in out["_m"][0]]


def _both(
    left: list[object],
    right: list[object],
    *,
    kind: str,
    threshold: float = 0.5,
    order: list[int] | None = None,
) -> tuple[list[int | None], list[float]]:
    """Run the engine, crosscheck it against ``_reference``, return its answer.

    Every test goes through here, so the two implementations are compared on
    every case the file pins rather than in one dedicated test.
    """
    got = _extract(_run(left, right, kind=kind, threshold=threshold, order=order))
    if left and right:
        want_idx, want_overlap = _reference(
            _matrix(left, right, kind=kind), threshold, order
        )
        assert got[0] == want_idx, (
            f"engine and reference disagree on assignment: "
            f"engine={got[0]} reference={want_idx}"
        )
        assert got[1] == pytest.approx(want_overlap, abs=1e-9), (
            f"engine and reference disagree on overlap: "
            f"engine={got[1]} reference={want_overlap}"
        )
    return got


KINDS = ("contour", "bbox")


def _shapes(kind: str, *rects: tuple[float, float, float, float]) -> list[object]:
    """Build rectangles in whichever representation *kind* names."""
    maker = _rect if kind == "contour" else _bbox
    return [maker(*r) for r in rects]


# ---------------------------------------------------------------------------
# The order decides who wins a contested target
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("kind", KINDS)
class TestOrderDecides:
    """The walk order is what resolves contention, so it must be honoured."""

    #: Two candidates, both clearing the threshold against a single target.
    #: The first has the better overlap (1.0 vs 2/3), so a run that ignores
    #: order entirely still hands the target to candidate 0 -- which is why
    #: the reversed case below is the one that carries the weight.
    LEFT = ((0.0, 0.0, 10.0, 10.0), (2.0, 0.0, 10.0, 10.0))
    RIGHT = ((0.0, 0.0, 10.0, 10.0),)

    def test_first_in_order_claims_the_target(self, kind: str) -> None:
        right_idx, _ = _both(
            _shapes(kind, *self.LEFT),
            _shapes(kind, *self.RIGHT),
            kind=kind,
            order=[0, 1],
        )
        assert right_idx == [0, None]

    def test_reversing_the_order_reverses_the_winner(self, kind: str) -> None:
        """Candidate 1 wins when walked first, despite the worse overlap.

        Without this case the suite passes on an implementation that sorts by
        overlap and ignores the order argument completely.
        """
        right_idx, _ = _both(
            _shapes(kind, *self.LEFT),
            _shapes(kind, *self.RIGHT),
            kind=kind,
            order=[1, 0],
        )
        assert right_idx == [None, 0]


# ---------------------------------------------------------------------------
# One-to-one: a claimed target is gone
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("kind", KINDS)
def test_a_target_can_only_be_claimed_once(kind: str) -> None:
    """Three candidates over threshold against one target yield one match."""
    left = _shapes(
        kind,
        (0.0, 0.0, 10.0, 10.0),
        (1.0, 0.0, 10.0, 10.0),
        (2.0, 0.0, 10.0, 10.0),
    )
    right = _shapes(kind, (0.0, 0.0, 10.0, 10.0))
    right_idx, _ = _both(left, right, kind=kind, order=[0, 1, 2])
    assert right_idx == [0, None, None]
    assert sum(1 for r in right_idx if r is not None) == 1


# ---------------------------------------------------------------------------
# Ties resolve to the smallest target index
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("kind", KINDS)
def test_an_exact_tie_resolves_to_the_smaller_target_index(kind: str) -> None:
    """Two targets at *identical* overlap: the lower index wins.

    A 10x10 candidate against two 10x10 targets offset by 5 on each axis.
    Both intersections are 50 and both unions 150, so the overlap is bit-for-bit
    equal and only the tie rule separates them. Nothing tested this before, in
    Rust or in Python.
    """
    left = _shapes(kind, (0.0, 0.0, 10.0, 10.0))
    right = _shapes(kind, (5.0, 0.0, 10.0, 10.0), (0.0, 5.0, 10.0, 10.0))

    matrix = _matrix(left, right, kind=kind)
    assert abs(matrix[0][0] - matrix[0][1]) < _TIE_EPS, (
        f"fixture is not a tie -- overlaps are {matrix[0]}; the tie rule is "
        f"not under test"
    )

    right_idx, _ = _both(left, right, kind=kind, threshold=0.25)
    assert right_idx == [0]


# ---------------------------------------------------------------------------
# The threshold is inclusive
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("kind", KINDS)
class TestThresholdIsInclusive:
    """A 5x10 target inside a 10x10 candidate: overlap is exactly 0.5.

    Intersection 50, union 100 -- exact in binary floating point, so the
    boundary can be probed without tolerance games.
    """

    LEFT = ((0.0, 0.0, 10.0, 10.0),)
    RIGHT = ((0.0, 0.0, 5.0, 10.0),)

    def test_the_fixture_sits_exactly_on_the_boundary(self, kind: str) -> None:
        assert _matrix(
            _shapes(kind, *self.LEFT), _shapes(kind, *self.RIGHT), kind=kind
        )[0][0] == 0.5

    def test_overlap_equal_to_the_threshold_matches(self, kind: str) -> None:
        right_idx, overlap = _both(
            _shapes(kind, *self.LEFT),
            _shapes(kind, *self.RIGHT),
            kind=kind,
            threshold=0.5,
        )
        assert right_idx == [0]
        assert overlap == [0.5]

    def test_overlap_just_below_the_threshold_does_not(self, kind: str) -> None:
        right_idx, _ = _both(
            _shapes(kind, *self.LEFT),
            _shapes(kind, *self.RIGHT),
            kind=kind,
            threshold=0.5000001,
        )
        assert right_idx == [None]


# ---------------------------------------------------------------------------
# What an unmatched candidate looks like
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("kind", KINDS)
def test_an_unmatched_candidate_is_null_with_zero_overlap(kind: str) -> None:
    """Disjoint shapes: null target index, and overlap 0.0 rather than null."""
    left = _shapes(kind, (0.0, 0.0, 10.0, 10.0))
    right = _shapes(kind, (500.0, 500.0, 10.0, 10.0))
    right_idx, overlap = _both(left, right, kind=kind)
    assert right_idx == [None]
    assert overlap == [0.0]


# ---------------------------------------------------------------------------
# Degenerate rows
# ---------------------------------------------------------------------------


@plugin_required
@pytest.mark.parametrize("kind", KINDS)
class TestDegenerateRows:
    """Empty sides produce empty lists, not errors and not nulls."""

    SQUARE = ((0.0, 0.0, 10.0, 10.0),)

    def test_no_candidates(self, kind: str) -> None:
        right_idx, overlap = _both([], _shapes(kind, *self.SQUARE), kind=kind)
        assert right_idx == []
        assert overlap == []

    def test_no_targets(self, kind: str) -> None:
        right_idx, overlap = _both(_shapes(kind, *self.SQUARE), [], kind=kind)
        assert right_idx == [None]
        assert overlap == [0.0]

    def test_neither_side_has_anything(self, kind: str) -> None:
        right_idx, overlap = _both([], [], kind=kind)
        assert right_idx == []
        assert overlap == []


# ---------------------------------------------------------------------------
# Identity: a set corresponds to itself, elementwise
# ---------------------------------------------------------------------------


@plugin_required
def test_concave_shapes_correspond_to_themselves() -> None:
    """Each shape claims its own index at overlap 1.0.

    Ported up from ``pairwise.rs``'s only assignment test, which is deleted
    along with the code it covers. Concave shapes are the point: an
    intersection routine valid only for convex clips gets this wrong.
    """
    shapes = [_l_shape(0.0, 0.0), _u_shape(), _l_shape(500.0, 500.0)]
    right_idx, overlap = _both(shapes, shapes, kind="contour", threshold=0.5)
    assert right_idx == [0, 1, 2]
    assert overlap == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)


# ---------------------------------------------------------------------------
# The two representations agree
# ---------------------------------------------------------------------------


@plugin_required
def test_contours_and_bboxes_agree_on_axis_aligned_rectangles() -> None:
    """The same rectangles, expressed both ways, correspond identically."""
    rects = [(0.0, 0.0, 10.0, 10.0), (2.0, 0.0, 10.0, 10.0), (40.0, 40.0, 6.0, 6.0)]
    targets = [(0.0, 0.0, 10.0, 10.0), (40.0, 40.0, 6.0, 6.0)]

    as_contour = _both(
        _shapes("contour", *rects), _shapes("contour", *targets),
        kind="contour", order=[0, 1, 2],
    )
    as_bbox = _both(
        _shapes("bbox", *rects), _shapes("bbox", *targets),
        kind="bbox", order=[0, 1, 2],
    )
    assert as_contour[0] == as_bbox[0]
    assert as_contour[1] == pytest.approx(as_bbox[1], abs=1e-9)
