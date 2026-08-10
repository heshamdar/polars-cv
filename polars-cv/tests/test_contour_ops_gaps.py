"""
Tests filling gaps in contour operation coverage.

Covers: area(signed=True), flip+winding verification, simplify edge cases,
scale with different origins, pairwise metrics with partial/no overlap,
contains_point boundary cases, and is_convex on non-convex shapes.
"""

from __future__ import annotations

import polars as pl
import pytest

from polars_cv.geometry import CONTOUR_SCHEMA, POINT_SCHEMA
from tests.conftest import plugin_required

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ccw_square() -> dict:
    """CCW-wound 100×100 square (positive signed area)."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 100.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }


@pytest.fixture
def cw_square() -> dict:
    """CW-wound 100×100 square (negative signed area)."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 0.0, "y": 100.0},
            {"x": 100.0, "y": 100.0},
            {"x": 100.0, "y": 0.0},
        ],
        "holes": [],
        "is_closed": True,
    }


@pytest.fixture
def l_shape() -> dict:
    """L-shaped non-convex contour."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 100.0, "y": 50.0},
            {"x": 50.0, "y": 50.0},
            {"x": 50.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }


@pytest.fixture
def holed_square() -> dict:
    """100×100 square with a 50×50 hole. Net area 10000 - 2500 = 7500."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 100.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [
            [
                {"x": 25.0, "y": 25.0},
                {"x": 25.0, "y": 75.0},
                {"x": 75.0, "y": 75.0},
                {"x": 75.0, "y": 25.0},
            ]
        ],
        "is_closed": True,
    }


@pytest.fixture
def holed_square_ccw_hole(holed_square: dict) -> dict:
    """The same shape with its hole ring wound the other way."""
    return {
        **holed_square,
        "holes": [list(reversed(holed_square["holes"][0]))],
    }


@pytest.fixture
def nested_holes() -> dict:
    """100×100 square whose hole contains a second ring. Region = 10000 - 6400."""
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 100.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [
            [
                {"x": 10.0, "y": 10.0},
                {"x": 10.0, "y": 90.0},
                {"x": 90.0, "y": 90.0},
                {"x": 90.0, "y": 10.0},
            ],
            [
                {"x": 40.0, "y": 40.0},
                {"x": 40.0, "y": 60.0},
                {"x": 60.0, "y": 60.0},
                {"x": 60.0, "y": 40.0},
            ],
        ],
        "is_closed": True,
    }


@pytest.fixture
def overlapping_holes() -> dict:
    """
    100×100 square with two *overlapping* hole rings.

    Hole A is [10,50]² (area 1600, centroid 30,30), hole B is [30,70]² (area 1600,
    centroid 50,50); they share [30,50]² (area 400, centroid 40,40). The union is
    area 2800 about (40,40), leaving a region of area 7200. The arrangement is
    asymmetric on purpose — a symmetric one lands on the right centroid even when
    the holes are subtracted one at a time.
    """
    return {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 100.0, "y": 0.0},
            {"x": 100.0, "y": 100.0},
            {"x": 0.0, "y": 100.0},
        ],
        "holes": [
            [
                {"x": 10.0, "y": 10.0},
                {"x": 50.0, "y": 10.0},
                {"x": 50.0, "y": 50.0},
                {"x": 10.0, "y": 50.0},
            ],
            [
                {"x": 30.0, "y": 30.0},
                {"x": 70.0, "y": 30.0},
                {"x": 70.0, "y": 70.0},
                {"x": 30.0, "y": 70.0},
            ],
        ],
        "is_closed": True,
    }


@pytest.fixture
def overlapping_squares() -> tuple[dict, dict]:
    """Two overlapping squares: (0,0)→(60,60) and (40,40)→(100,100).

    Intersection: (40,40)→(60,60) = 20×20 = 400
    Union: 2×3600 - 400 = 6800
    IoU = 400/6800 ≈ 0.0588
    """
    c1 = {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 60.0, "y": 0.0},
            {"x": 60.0, "y": 60.0},
            {"x": 0.0, "y": 60.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    c2 = {
        "exterior": [
            {"x": 40.0, "y": 40.0},
            {"x": 100.0, "y": 40.0},
            {"x": 100.0, "y": 100.0},
            {"x": 40.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    return c1, c2


@pytest.fixture
def non_overlapping_squares() -> tuple[dict, dict]:
    """Two non-overlapping squares: (0,0)→(40,40) and (60,60)→(100,100)."""
    c1 = {
        "exterior": [
            {"x": 0.0, "y": 0.0},
            {"x": 40.0, "y": 0.0},
            {"x": 40.0, "y": 40.0},
            {"x": 0.0, "y": 40.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    c2 = {
        "exterior": [
            {"x": 60.0, "y": 60.0},
            {"x": 100.0, "y": 60.0},
            {"x": 100.0, "y": 100.0},
            {"x": 60.0, "y": 100.0},
        ],
        "holes": [],
        "is_closed": True,
    }
    return c1, c2


# ---------------------------------------------------------------------------
# area(signed=True)
# ---------------------------------------------------------------------------


@plugin_required
class TestContourAreaSigned:
    """Tests for area(signed=True) which was previously untested."""

    def test_signed_area_ccw_positive(self, ccw_square: dict) -> None:
        """CCW contour should yield positive signed area."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(area=pl.col("contour").contour.area(signed=True))
        assert result["area"][0] > 0
        assert result["area"][0] == pytest.approx(10000.0, rel=0.01)

    def test_signed_area_cw_negative(self, cw_square: dict) -> None:
        """CW contour should yield negative signed area."""
        df = pl.DataFrame({"contour": [cw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(area=pl.col("contour").contour.area(signed=True))
        assert result["area"][0] < 0
        assert result["area"][0] == pytest.approx(-10000.0, rel=0.01)

    def test_unsigned_area_always_positive(self, cw_square: dict) -> None:
        """Default signed=False should always return positive area."""
        df = pl.DataFrame({"contour": [cw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(area=pl.col("contour").contour.area(signed=False))
        assert result["area"][0] > 0

    def test_signed_area_both_windings_same_magnitude(
        self, ccw_square: dict, cw_square: dict
    ) -> None:
        """Signed areas of opposite windings should be equal in magnitude."""
        df = pl.DataFrame(
            {"contour": [ccw_square, cw_square]},
            schema={"contour": CONTOUR_SCHEMA},
        )
        result = df.with_columns(area=pl.col("contour").contour.area(signed=True))
        assert abs(result["area"][0]) == pytest.approx(abs(result["area"][1]), rel=0.01)


# ---------------------------------------------------------------------------
# flip + winding behavioural verification
# ---------------------------------------------------------------------------


@plugin_required
class TestContourFlipWinding:
    """Verify that flip() actually reverses winding direction."""

    def test_flip_reverses_winding(self, ccw_square: dict) -> None:
        """Flipping a CCW contour should produce CW winding."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            original_winding=pl.col("contour").contour.winding(),
            flipped=pl.col("contour").contour.flip(),
        ).with_columns(
            flipped_winding=pl.col("flipped").contour.winding(),
        )
        orig = result["original_winding"][0]
        flipped = result["flipped_winding"][0]
        assert orig != flipped
        assert {orig, flipped} == {"cw", "ccw"}

    def test_double_flip_restores_winding(self, ccw_square: dict) -> None:
        """Flipping twice should restore original winding."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            original_winding=pl.col("contour").contour.winding(),
            double_flipped=pl.col("contour").contour.flip().contour.flip(),
        ).with_columns(
            restored_winding=pl.col("double_flipped").contour.winding(),
        )
        assert result["original_winding"][0] == result["restored_winding"][0]

    def test_flip_preserves_area(self, ccw_square: dict) -> None:
        """Flipping should not change unsigned area."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            orig_area=pl.col("contour").contour.area(),
            flipped_area=pl.col("contour").contour.flip().contour.area(),
        )
        assert result["orig_area"][0] == pytest.approx(
            result["flipped_area"][0], rel=0.01
        )


# ---------------------------------------------------------------------------
# ensure_winding
# ---------------------------------------------------------------------------


@plugin_required
class TestEnsureWinding:
    """Verify ensure_winding() actually corrects winding when needed."""

    def test_ensure_ccw_on_ccw_is_noop(self, ccw_square: dict) -> None:
        """Already-CCW contour should be unchanged by ensure_winding('ccw')."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("ccw"),
        ).with_columns(
            winding=pl.col("ensured").contour.winding(),
        )
        assert result["winding"][0] == "ccw"

    def test_ensure_cw_on_ccw_flips(self, ccw_square: dict) -> None:
        """CCW contour with ensure_winding('cw') should become CW."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("cw"),
        ).with_columns(
            winding=pl.col("ensured").contour.winding(),
        )
        assert result["winding"][0] == "cw"

    def test_ensure_ccw_on_cw_flips(self, cw_square: dict) -> None:
        """CW contour with ensure_winding('ccw') should become CCW."""
        df = pl.DataFrame({"contour": [cw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("ccw"),
        ).with_columns(
            winding=pl.col("ensured").contour.winding(),
        )
        assert result["winding"][0] == "ccw"


@plugin_required
class TestContourEnumParamsAreRejectedNotDefaulted:
    """An unrecognised enum value must fail, not fall back to a default.

    ``ensure_winding`` and ``scale(origin=)`` were the only two user-facing
    string parameters in the plugin that did not read a ``NAMED`` table. Both
    parsed by hand and ended in ``_ => <default>``, so a value the parser did
    not know was answered with a plausible one: ``ensure_winding("CW")``
    returned *counter*-clockwise — the opposite of the request — and
    ``scale(origin="top_left")`` scaled about the centroid. Both silently.

    The values here are the shapes a user actually produces: a capitalisation
    that does not match, and a plausible name from another library.
    """

    def test_a_miscased_winding_is_rejected(self, ccw_square: dict) -> None:
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        with pytest.raises(ValueError, match="Invalid direction 'CW'"):
            df.with_columns(x=pl.col("contour").contour.ensure_winding("CW"))

    def test_the_long_winding_spellings_still_work(self, ccw_square: dict) -> None:
        """The aliases the parser has always accepted are kept, not dropped.

        They live in ``Winding::NAMED`` as aliases, so they are also surfaced
        over ``enum_variants`` and mirrored in ``_types.Winding`` — the
        previous annotation admitted only the short forms while the parser
        took both.
        """
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            ensured=pl.col("contour").contour.ensure_winding("clockwise"),
        ).with_columns(winding=pl.col("ensured").contour.winding())
        assert result["winding"][0] == "cw"

    def test_an_unknown_scale_origin_is_rejected(self, ccw_square: dict) -> None:
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        with pytest.raises(ValueError, match="Invalid origin 'top_left'"):
            df.with_columns(
                x=pl.col("contour").contour.scale(sx=2.0, sy=2.0, origin="top_left")
            )

    def test_the_rejection_lists_what_is_accepted(self, ccw_square: dict) -> None:
        """A rejection that does not say what is valid invites a second guess."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        with pytest.raises(ValueError, match="bbox_center"):
            df.with_columns(
                x=pl.col("contour").contour.scale(sx=2.0, sy=2.0, origin="middle")
            )


# ---------------------------------------------------------------------------
# simplify edge cases
# ---------------------------------------------------------------------------


@plugin_required
class TestContourSimplify:
    """Simplify with various tolerance values."""

    def test_simplify_zero_tolerance_preserves_all_points(
        self, ccw_square: dict
    ) -> None:
        """Zero tolerance should preserve all points."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            simplified=pl.col("contour").contour.simplify(tolerance=0.0),
        )
        # Original has 4 points, simplified with 0 tolerance should keep all
        orig_exterior = result["contour"][0]["exterior"]
        simp_exterior = result["simplified"][0]["exterior"]
        assert len(simp_exterior) == len(orig_exterior)

    def test_simplify_large_tolerance_reduces_points(self, l_shape: dict) -> None:
        """Large tolerance should reduce point count on complex shape."""
        df = pl.DataFrame({"contour": [l_shape]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            simplified=pl.col("contour").contour.simplify(tolerance=100.0),
        )
        simp_exterior = result["simplified"][0]["exterior"]
        # L-shape has 6 points; large tolerance should reduce to fewer
        assert len(simp_exterior) < 6

    def test_simplify_preserves_closure(self, ccw_square: dict) -> None:
        """Simplified contour should remain closed."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            simplified=pl.col("contour").contour.simplify(tolerance=1.0),
        )
        assert result["simplified"][0]["is_closed"] is True


# ---------------------------------------------------------------------------
# scale with different origins
# ---------------------------------------------------------------------------


@plugin_required
class TestContourScaleOrigins:
    """Test scale with all three origin options."""

    def test_scale_origin_default(self, ccw_square: dict) -> None:
        """Scale from origin (0,0) – points should multiply."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            scaled=pl.col("contour").contour.scale(sx=2.0, sy=2.0, origin="origin"),
        )
        ext = result["scaled"][0]["exterior"]
        # Original (100,100) corner should become (200,200)
        xs = [p["x"] for p in ext]
        ys = [p["y"] for p in ext]
        assert max(xs) == pytest.approx(200.0, rel=0.01)
        assert max(ys) == pytest.approx(200.0, rel=0.01)

    def test_scale_origin_centroid(self, ccw_square: dict) -> None:
        """Scale from centroid – centroid should stay fixed."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            orig_centroid=pl.col("contour").contour.centroid(),
            scaled=pl.col("contour").contour.scale(sx=2.0, sy=2.0, origin="centroid"),
        ).with_columns(
            scaled_centroid=pl.col("scaled").contour.centroid(),
        )
        oc = result["orig_centroid"][0]
        sc = result["scaled_centroid"][0]
        assert oc["x"] == pytest.approx(sc["x"], abs=1.0)
        assert oc["y"] == pytest.approx(sc["y"], abs=1.0)

    def test_scale_origin_bbox_center(self, ccw_square: dict) -> None:
        """Scale from bbox_center – bbox center should stay fixed."""
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(
            orig_bbox=pl.col("contour").contour.bounding_box(),
            scaled=pl.col("contour").contour.scale(
                sx=2.0, sy=2.0, origin="bbox_center"
            ),
        ).with_columns(
            scaled_bbox=pl.col("scaled").contour.bounding_box(),
        )
        ob = result["orig_bbox"][0]
        sb = result["scaled_bbox"][0]
        orig_cx = ob["x"] + ob["width"] / 2
        orig_cy = ob["y"] + ob["height"] / 2
        scaled_cx = sb["x"] + sb["width"] / 2
        scaled_cy = sb["y"] + sb["height"] / 2
        assert orig_cx == pytest.approx(scaled_cx, abs=1.0)
        assert orig_cy == pytest.approx(scaled_cy, abs=1.0)


# ---------------------------------------------------------------------------
# Pairwise: IoU, Dice, Hausdorff with overlap scenarios
# ---------------------------------------------------------------------------


@plugin_required
class TestPairwiseOverlapScenarios:
    """Test IoU, Dice, Hausdorff with identical, partial, and no overlap."""

    def test_iou_identical_is_one(self, ccw_square: dict) -> None:
        """IoU of identical contours should be 1.0."""
        df = pl.DataFrame(
            {"a": [ccw_square], "b": [ccw_square]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))
        assert result["iou"][0] == pytest.approx(1.0, abs=0.05)

    def test_iou_no_overlap_is_zero(
        self, non_overlapping_squares: tuple[dict, dict]
    ) -> None:
        """IoU of non-overlapping contours should be 0.0."""
        c1, c2 = non_overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))
        assert result["iou"][0] == pytest.approx(0.0, abs=0.05)

    def test_iou_partial_overlap_in_range(
        self, overlapping_squares: tuple[dict, dict]
    ) -> None:
        """IoU of partially overlapping contours should be between 0 and 1."""
        c1, c2 = overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(iou=pl.col("a").contour.iou(pl.col("b")))
        val = result["iou"][0]
        assert 0.0 < val < 1.0

    def test_dice_identical_is_one(self, ccw_square: dict) -> None:
        """Dice of identical contours should be 1.0."""
        df = pl.DataFrame(
            {"a": [ccw_square], "b": [ccw_square]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(dice=pl.col("a").contour.dice(pl.col("b")))
        assert result["dice"][0] == pytest.approx(1.0, abs=0.05)

    def test_dice_no_overlap_is_zero(
        self, non_overlapping_squares: tuple[dict, dict]
    ) -> None:
        """Dice of non-overlapping contours should be 0.0."""
        c1, c2 = non_overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(dice=pl.col("a").contour.dice(pl.col("b")))
        assert result["dice"][0] == pytest.approx(0.0, abs=0.05)

    def test_hausdorff_identical_is_zero(self, ccw_square: dict) -> None:
        """Hausdorff distance of identical contours should be 0.0."""
        df = pl.DataFrame(
            {"a": [ccw_square], "b": [ccw_square]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(
            dist=pl.col("a").contour.hausdorff_distance(pl.col("b"))
        )
        assert result["dist"][0] == pytest.approx(0.0, abs=0.5)

    def test_hausdorff_non_overlapping_positive(
        self, non_overlapping_squares: tuple[dict, dict]
    ) -> None:
        """Hausdorff distance of separated contours should be positive."""
        c1, c2 = non_overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(
            dist=pl.col("a").contour.hausdorff_distance(pl.col("b"))
        )
        assert result["dist"][0] > 0.0

    def test_hausdorff_is_symmetric(
        self, overlapping_squares: tuple[dict, dict]
    ) -> None:
        """Hausdorff distance should be symmetric: d(A,B) == d(B,A)."""
        c1, c2 = overlapping_squares
        df = pl.DataFrame(
            {"a": [c1], "b": [c2]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        )
        result = df.with_columns(
            d_ab=pl.col("a").contour.hausdorff_distance(pl.col("b")),
            d_ba=pl.col("b").contour.hausdorff_distance(pl.col("a")),
        )
        assert result["d_ab"][0] == pytest.approx(result["d_ba"][0], rel=0.01)


# ---------------------------------------------------------------------------
# Pairwise: winding, concavity and holes must not affect overlap
# ---------------------------------------------------------------------------


def _pair(a: dict, b: dict) -> pl.DataFrame:
    return pl.DataFrame(
        {"a": [a], "b": [b]},
        schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
    )


def _iou(a: dict, b: dict) -> float:
    return _pair(a, b).select(pl.col("a").contour.iou(pl.col("b"))).item()


def _dice(a: dict, b: dict) -> float:
    return _pair(a, b).select(pl.col("a").contour.dice(pl.col("b"))).item()


@plugin_required
class TestOverlapIgnoresWindingAndConcavity:
    """
    A contour scores 1.0 against itself whatever its point order or shape.

    Overlap used to go through a Sutherland-Hodgman clipper, which is only valid
    for a convex, CCW-wound clip polygon — neither of which the schema requires.
    """

    def test_iou_cw_identical_is_one(self, cw_square: dict) -> None:
        assert _iou(cw_square, cw_square) == pytest.approx(1.0, abs=1e-9)

    def test_iou_mixed_winding_identical_is_one(
        self, ccw_square: dict, cw_square: dict
    ) -> None:
        """The same square wound both ways is still the same square."""
        assert _iou(ccw_square, cw_square) == pytest.approx(1.0, abs=1e-9)
        assert _iou(cw_square, ccw_square) == pytest.approx(1.0, abs=1e-9)

    def test_iou_concave_identical_is_one(self, l_shape: dict) -> None:
        assert _iou(l_shape, l_shape) == pytest.approx(1.0, abs=1e-9)

    def test_iou_concave_partial_overlap_is_exact(self, l_shape: dict) -> None:
        """
        `l_shape` offset by (25, 25) overlaps it in three rectangles:
        75×25 + 25×25 + 25×25 = 3125, against a union of 7500 + 7500 - 3125.
        """
        shifted = {
            **l_shape,
            "exterior": [
                {"x": p["x"] + 25.0, "y": p["y"] + 25.0} for p in l_shape["exterior"]
            ],
        }
        assert _iou(l_shape, shifted) == pytest.approx(3125.0 / 11875.0, abs=1e-9)

    def test_iou_is_symmetric(self, ccw_square: dict, l_shape: dict) -> None:
        assert _iou(ccw_square, l_shape) == pytest.approx(
            _iou(l_shape, ccw_square), abs=1e-12
        )

    def test_dice_cw_identical_is_one(self, cw_square: dict) -> None:
        assert _dice(cw_square, cw_square) == pytest.approx(1.0, abs=1e-9)

    def test_dice_concave_identical_is_one(self, l_shape: dict) -> None:
        assert _dice(l_shape, l_shape) == pytest.approx(1.0, abs=1e-9)


@plugin_required
class TestHolesAreStructuralNotDirectional:
    """
    The `holes` field decides hole-ness; ring winding never does.

    These pin the spec as behaviour rather than prose: the same shape described
    with a CW hole and with a CCW hole must be indistinguishable.
    """

    def test_iou_holed_identical_is_one(self, holed_square: dict) -> None:
        assert _iou(holed_square, holed_square) == pytest.approx(1.0, abs=1e-9)

    def test_iou_hole_winding_does_not_matter(
        self, holed_square: dict, holed_square_ccw_hole: dict
    ) -> None:
        assert _iou(holed_square, holed_square_ccw_hole) == pytest.approx(1.0, abs=1e-9)

    def test_iou_holed_vs_solid_accounts_for_the_hole(
        self, holed_square: dict, ccw_square: dict
    ) -> None:
        """Intersection is the holed shape (7500); union is the solid one (10000)."""
        assert _iou(holed_square, ccw_square) == pytest.approx(0.75, abs=1e-9)

    def test_iou_ccw_holed_vs_solid_accounts_for_the_hole(
        self, holed_square_ccw_hole: dict, ccw_square: dict
    ) -> None:
        """
        The load-bearing assertion for hole handling.

        A holed contour matched against *itself* saturates at 1.0 under the final
        clamp no matter what the intersection does, so identity tests cannot detect
        a hole being ignored or double-counted. Only an asymmetric comparison can,
        and it has to run against the CCW-wound hole — the CW one behaves the same
        under fill rules that get this wrong.
        """
        assert _iou(holed_square_ccw_hole, ccw_square) == pytest.approx(0.75, abs=1e-9)

    def test_dice_holed_vs_solid_accounts_for_the_hole(
        self, holed_square: dict, ccw_square: dict
    ) -> None:
        assert _dice(holed_square, ccw_square) == pytest.approx(
            15000.0 / 17500.0, abs=1e-9
        )

    def test_dice_ccw_holed_vs_solid_accounts_for_the_hole(
        self, holed_square_ccw_hole: dict, ccw_square: dict
    ) -> None:
        assert _dice(holed_square_ccw_hole, ccw_square) == pytest.approx(
            15000.0 / 17500.0, abs=1e-9
        )

    def test_nested_holes_are_all_holes(
        self, nested_holes: dict, ccw_square: dict
    ) -> None:
        """
        Every ring in `holes` is a hole, however the rings nest.

        The region is the exterior minus the *union* of the hole rings — the inner
        ring lies in a part already removed, so it changes nothing. 10000 - 6400.
        """
        df = pl.DataFrame(
            {"c": [nested_holes]}, schema={"c": CONTOUR_SCHEMA}
        ).with_columns(area=pl.col("c").contour.area())
        assert df["area"][0] == pytest.approx(3600.0, abs=1e-9)

        assert _iou(nested_holes, nested_holes) == pytest.approx(1.0, abs=1e-9)
        assert _iou(nested_holes, ccw_square) == pytest.approx(0.36, abs=1e-9)
        assert _dice(nested_holes, ccw_square) == pytest.approx(
            7200.0 / 13600.0, abs=1e-9
        )

    def test_nested_hole_interior_is_outside_the_contour(
        self, nested_holes: dict
    ) -> None:
        """A point in the inner ring sits inside a removed region, so it is out."""
        df = pl.DataFrame(
            {"c": [nested_holes], "p": [{"x": 50.0, "y": 50.0}]},
            schema={"c": CONTOUR_SCHEMA, "p": POINT_SCHEMA},
        ).with_columns(inside=pl.col("c").contour.contains_point(pl.col("p")))
        assert not df["inside"][0]

    def test_area_ignores_hole_winding(
        self, holed_square: dict, holed_square_ccw_hole: dict
    ) -> None:
        df = pl.DataFrame(
            {"a": [holed_square], "b": [holed_square_ccw_hole]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        ).with_columns(
            area_a=pl.col("a").contour.area(),
            area_b=pl.col("b").contour.area(),
        )
        assert df["area_a"][0] == pytest.approx(7500.0)
        assert df["area_b"][0] == pytest.approx(7500.0)

    def test_centroid_measures_the_same_region_as_area(
        self, overlapping_holes: dict
    ) -> None:
        """
        The centroid belongs to the region `area` reports, not to a different shape.

        Region moment 10000*50 - 2800*40 = 388000 over an area of 7200. Subtracting
        each hole's moment in turn instead gives 372000/6800 = 54.7059..., the
        centroid of a shape where the shared part was removed twice.
        """
        df = pl.DataFrame(
            {"c": [overlapping_holes]}, schema={"c": CONTOUR_SCHEMA}
        ).with_columns(
            area=pl.col("c").contour.area(),
            centroid=pl.col("c").contour.centroid(),
        )
        assert df["area"][0] == pytest.approx(7200.0, abs=1e-9)
        expected = 388000.0 / 7200.0
        assert df["centroid"][0]["x"] == pytest.approx(expected, abs=1e-6)
        assert df["centroid"][0]["y"] == pytest.approx(expected, abs=1e-6)

    def test_hausdorff_walks_hole_vertices(
        self, holed_square: dict, ccw_square: dict
    ) -> None:
        """
        A hole edge bounds the region, so it counts toward the boundary distance.

        These two share an exterior, so an exterior-only measure would call them
        identical. Each hole corner is 25*sqrt(2) from the nearest corner of the
        solid square.
        """
        df = pl.DataFrame(
            {"a": [holed_square], "b": [ccw_square]},
            schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
        ).with_columns(
            d_ab=pl.col("a").contour.hausdorff_distance(pl.col("b")),
            d_ba=pl.col("b").contour.hausdorff_distance(pl.col("a")),
        )
        expected = 25.0 * 2.0**0.5
        assert df["d_ab"][0] == pytest.approx(expected, abs=1e-9)
        assert df["d_ba"][0] == pytest.approx(expected, abs=1e-9)

    def test_contains_point_ignores_hole_winding(
        self, holed_square: dict, holed_square_ccw_hole: dict
    ) -> None:
        """A point in the hole is outside the contour, however the hole is wound."""
        df = pl.DataFrame(
            {
                "a": [holed_square],
                "b": [holed_square_ccw_hole],
                "in_hole": [{"x": 50.0, "y": 50.0}],
                "in_ring": [{"x": 10.0, "y": 10.0}],
            },
            schema={
                "a": CONTOUR_SCHEMA,
                "b": CONTOUR_SCHEMA,
                "in_hole": POINT_SCHEMA,
                "in_ring": POINT_SCHEMA,
            },
        ).with_columns(
            in_hole_a=pl.col("a").contour.contains_point(pl.col("in_hole")),
            in_hole_b=pl.col("b").contour.contains_point(pl.col("in_hole")),
            in_ring_a=pl.col("a").contour.contains_point(pl.col("in_ring")),
            in_ring_b=pl.col("b").contour.contains_point(pl.col("in_ring")),
        )
        assert not df["in_hole_a"][0]
        assert not df["in_hole_b"][0]
        assert df["in_ring_a"][0]
        assert df["in_ring_b"][0]


# ---------------------------------------------------------------------------
# contains_point
# ---------------------------------------------------------------------------


@plugin_required
class TestContainsPoint:
    """Test point-in-polygon checks."""

    def test_point_inside(self, ccw_square: dict) -> None:
        """A point inside the contour should return True."""
        df = pl.DataFrame(
            {
                "contour": [ccw_square],
                "point": [{"x": 50.0, "y": 50.0}],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            inside=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        assert result["inside"][0] is True

    def test_point_outside(self, ccw_square: dict) -> None:
        """A point outside the contour should return False."""
        df = pl.DataFrame(
            {
                "contour": [ccw_square],
                "point": [{"x": 200.0, "y": 200.0}],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            inside=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        assert result["inside"][0] is False

    def test_point_at_origin_corner(self, ccw_square: dict) -> None:
        """Point exactly at vertex (0,0) – boundary case."""
        df = pl.DataFrame(
            {
                "contour": [ccw_square],
                "point": [{"x": 0.0, "y": 0.0}],
            },
            schema={"contour": CONTOUR_SCHEMA, "point": POINT_SCHEMA},
        )
        result = df.with_columns(
            inside=pl.col("contour").contour.contains_point(pl.col("point"))
        )
        # Boundary behavior is implementation-defined; just verify no crash
        assert result["inside"][0] in (True, False)


# ---------------------------------------------------------------------------
# is_convex
# ---------------------------------------------------------------------------


@plugin_required
class TestIsConvex:
    """Verify is_convex() correctness for convex and non-convex shapes."""

    def test_square_is_convex(self, ccw_square: dict) -> None:
        df = pl.DataFrame({"contour": [ccw_square]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(convex=pl.col("contour").contour.is_convex())
        assert result["convex"][0] is True

    def test_l_shape_is_not_convex(self, l_shape: dict) -> None:
        df = pl.DataFrame({"contour": [l_shape]}, schema={"contour": CONTOUR_SCHEMA})
        result = df.with_columns(convex=pl.col("contour").contour.is_convex())
        assert result["convex"][0] is False
