"""
Cross-checks the analytic contour measures against rasterized ground truth.

`area`, `iou`, `dice` and `centroid` are computed from polygon geometry; a mask
produced by `source("contour", ...)` is computed by an independent scanline
filler. Counting pixels in that mask gives a second, unrelated route to the same
quantities, so the two agreeing is real evidence — a sign error, a mishandled
hole or a winding assumption shows up as a mismatch here even when the polygon
maths is self-consistently wrong.

**The shared convention.** A pixel belongs to a shape when its centre,
`(x + 0.5, y + 0.5)`, lies inside it. For a shape whose vertices are all
integers, no pixel centre ever lands on an edge, so the mask is *exactly* the
area — hence the `RECTILINEAR` cases assert equality to the pixel. Shapes with
diagonal or curved edges cut through pixels, and the error is then bounded by
the number of pixels the boundary crosses, which is why those cases assert a
relative tolerance derived from the perimeter rather than an exact count.

Rasterization is at 512x512 with shapes ~400px across, keeping the boundary a
small fraction of the interior.

The last section closes the loop the other way: contour -> mask -> contour,
through `extract_contours`. That leg has a systematic bias rather than a random
one — see `TestRoundTripThroughExtraction` — and its assertions are written
around the exact size of that bias, not around a tolerance wide enough to hide
it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from polars_cv import CONTOUR_SCHEMA, Pipeline, numpy_from_struct
from tests.conftest import plugin_required

if TYPE_CHECKING:
    from collections.abc import Callable

CANVAS = 512
"""Mask edge length. Large enough that boundary pixels stay a small fraction."""


# ---------------------------------------------------------------------------
# Shape construction
# ---------------------------------------------------------------------------


def _ring(points: list[tuple[float, float]]) -> list[dict]:
    return [{"x": float(x), "y": float(y)} for x, y in points]


def _contour(
    exterior: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> dict:
    return {
        "exterior": _ring(exterior),
        "holes": [_ring(h) for h in (holes or [])],
        "is_closed": True,
    }


def _box(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _box_cw(x: float, y: float, w: float, h: float) -> list[tuple[float, float]]:
    """The same box, wound the other way — hole-ness and area must not notice."""
    return [(x, y), (x, y + h), (x + w, y + h), (x + w, y)]


def _regular_polygon(
    cx: float, cy: float, radius: float, sides: int, phase: float = 0.0
) -> list[tuple[float, float]]:
    return [
        (
            cx + radius * math.cos(phase + 2.0 * math.pi * i / sides),
            cy + radius * math.sin(phase + 2.0 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


# Vertices on integer coordinates and edges on axis lines: every pixel centre
# falls strictly inside or strictly outside, so the mask count is the area.
RECTILINEAR: dict[str, dict] = {
    "square": _contour(_box(50, 50, 400, 400)),
    "square_cw_wound": _contour(_box_cw(50, 50, 400, 400)),
    "wide_rect": _contour(_box(20, 180, 470, 120)),
    "l_shape": _contour(
        [(60, 60), (460, 60), (460, 260), (260, 260), (260, 460), (60, 460)]
    ),
    "u_shape": _contour(
        [
            (60, 60),
            (460, 60),
            (460, 460),
            (360, 460),
            (360, 200),
            (160, 200),
            (160, 460),
            (60, 460),
        ]
    ),
    "plus_shape": _contour(
        [
            (180, 60),
            (330, 60),
            (330, 180),
            (450, 180),
            (450, 330),
            (330, 330),
            (330, 450),
            (180, 450),
            (180, 330),
            (60, 330),
            (60, 180),
            (180, 180),
        ]
    ),
    "holed_square": _contour(_box(50, 50, 400, 400), [_box(150, 150, 200, 200)]),
    "holed_square_cw_hole": _contour(
        _box(50, 50, 400, 400), [_box_cw(150, 150, 200, 200)]
    ),
    "two_disjoint_holes": _contour(
        _box(40, 40, 430, 430),
        [_box(80, 80, 120, 120), _box(280, 280, 120, 120)],
    ),
    # The rings overlap: the shared part must come out once, not twice.
    "overlapping_holes": _contour(
        _box(40, 40, 430, 430),
        [_box(100, 100, 180, 180), _box(220, 220, 180, 180)],
    ),
    # A hole ring inside another hole ring: still a hole, and it lies in a part
    # already removed, so it changes nothing.
    "nested_holes": _contour(
        _box(40, 40, 430, 430),
        [_box(90, 90, 330, 330), _box(200, 200, 100, 100)],
    ),
    "concave_with_hole": _contour(
        [(60, 60), (460, 60), (460, 260), (260, 260), (260, 460), (60, 460)],
        [_box(100, 100, 100, 100)],
    ),
}

# Diagonal and curved edges: the boundary cuts through pixels, so the mask is an
# approximation and the assertions are relative.
CURVED: dict[str, dict] = {
    "triangle": _contour([(60, 440), (450, 440), (255, 70)]),
    "diamond": _contour([(256, 60), (452, 256), (256, 452), (60, 256)]),
    "rotated_square": _contour(_regular_polygon(256, 256, 200, 4, phase=0.3)),
    "hexagon": _contour(_regular_polygon(256, 256, 200, 6)),
    "circle": _contour(_regular_polygon(256, 256, 200, 128)),
    "circle_with_circular_hole": _contour(
        _regular_polygon(256, 256, 200, 128),
        [_regular_polygon(256, 256, 100, 128)],
    ),
    "star": _contour(
        [
            _regular_polygon(256, 256, 200 if i % 2 == 0 else 85, 10)[i]
            for i in range(10)
        ]
    ),
}

ALL_SHAPES = {**RECTILINEAR, **CURVED}


# ---------------------------------------------------------------------------
# The two measurement routes
# ---------------------------------------------------------------------------


def _mask(contour: dict) -> np.ndarray:
    """Rasterize to a boolean mask via the pipeline's scanline filler."""
    frame = pl.DataFrame({"c": [contour]}, schema={"c": CONTOUR_SCHEMA})
    pipe = Pipeline().source(
        "contour", width=CANVAS, height=CANVAS, fill_value=1, background=0
    )
    out = frame.with_columns(m=pl.col("c").cv.pipe(pipe).sink("numpy"))
    return numpy_from_struct(out["m"][0]).reshape(CANVAS, CANVAS).astype(bool)


def _measure(contour: dict, expr) -> object:
    frame = pl.DataFrame({"c": [contour]}, schema={"c": CONTOUR_SCHEMA})
    return frame.select(expr(pl.col("c"))).item()


def _analytic_area(contour: dict) -> float:
    return float(_measure(contour, lambda c: c.contour.area()))


def _analytic_centroid(contour: dict) -> tuple[float, float]:
    point = _measure(contour, lambda c: c.contour.centroid())
    return float(point["x"]), float(point["y"])


def _pair_measure(a: dict, b: dict, expr) -> float:
    frame = pl.DataFrame(
        {"a": [a], "b": [b]},
        schema={"a": CONTOUR_SCHEMA, "b": CONTOUR_SCHEMA},
    )
    return float(frame.select(expr(pl.col("a"), pl.col("b"))).item())


def _perimeter(contour: dict) -> float:
    return float(_measure(contour, lambda c: c.contour.perimeter()))


def _boundary_tolerance(contour: dict, *, slack: float = 2.0) -> float:
    """
    Pixels a one-pixel-wide boundary band can plausibly cover.

    Discretization error is bounded by the number of pixels the boundary passes
    through, which scales with perimeter, not with area. `slack` covers the two
    ends of each scanline span and the corners.
    """
    return slack * _perimeter(contour)


# ---------------------------------------------------------------------------
# Area
# ---------------------------------------------------------------------------


@plugin_required
class TestAreaMatchesRasterizedPixelCount:
    """`area()` counts the same region the rasterizer paints."""

    @pytest.mark.parametrize("name", sorted(RECTILINEAR))
    def test_rectilinear_area_is_exact(self, name: str) -> None:
        """
        Integer vertices on axis-aligned edges put every pixel centre strictly
        inside or strictly outside, so this is an equality, not an estimate.
        """
        contour = RECTILINEAR[name]
        assert int(_mask(contour).sum()) == pytest.approx(
            _analytic_area(contour), abs=0.5
        )

    @pytest.mark.parametrize("name", sorted(CURVED))
    def test_curved_area_is_within_the_boundary_band(self, name: str) -> None:
        contour = CURVED[name]
        analytic = _analytic_area(contour)
        rasterized = int(_mask(contour).sum())
        assert abs(rasterized - analytic) < _boundary_tolerance(contour), (
            f"{name}: analytic {analytic}, rasterized {rasterized}"
        )

    @pytest.mark.parametrize("name", sorted(CURVED))
    def test_curved_area_relative_error_is_small(self, name: str) -> None:
        """A tighter statement than the band: under 1% for shapes this size."""
        contour = CURVED[name]
        analytic = _analytic_area(contour)
        rasterized = int(_mask(contour).sum())
        assert rasterized == pytest.approx(analytic, rel=0.01)

    def test_hole_removes_exactly_its_own_pixels(self) -> None:
        """The mask shrinks by the hole's area, and by nothing else."""
        solid = _contour(_box(50, 50, 400, 400))
        holed = _contour(_box(50, 50, 400, 400), [_box(150, 150, 200, 200)])
        assert int(_mask(solid).sum()) - int(_mask(holed).sum()) == 200 * 200

    def test_overlapping_holes_are_removed_once(self) -> None:
        """
        Two hole rings sharing a region: the shared part is removed once.

        Subtracting each hole in turn would report 430*430 - 180*180 - 180*180;
        the mask cannot double-remove pixels, so it is the arbiter.
        """
        contour = RECTILINEAR["overlapping_holes"]
        union = 180 * 180 + 180 * 180 - 60 * 60
        assert int(_mask(contour).sum()) == 430 * 430 - union
        assert _analytic_area(contour) == pytest.approx(430 * 430 - union, abs=0.5)

    def test_nested_hole_ring_changes_nothing(self) -> None:
        """The inner ring lies in a part the outer hole already removed."""
        contour = RECTILINEAR["nested_holes"]
        assert int(_mask(contour).sum()) == 430 * 430 - 330 * 330
        assert _analytic_area(contour) == pytest.approx(430 * 430 - 330 * 330, abs=0.5)

    @pytest.mark.parametrize(
        ("wound", "reference"),
        [
            ("square_cw_wound", "square"),
            ("holed_square_cw_hole", "holed_square"),
        ],
    )
    def test_winding_changes_neither_route(self, wound: str, reference: str) -> None:
        """Point order is not a hole signal, in the polygon maths or the raster."""
        assert int(_mask(RECTILINEAR[wound]).sum()) == int(
            _mask(RECTILINEAR[reference]).sum()
        )
        assert _analytic_area(RECTILINEAR[wound]) == pytest.approx(
            _analytic_area(RECTILINEAR[reference])
        )


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------


@plugin_required
class TestCentroidMatchesRasterizedCentreOfMass:
    """`centroid()` locates the centre of mass of the painted pixels."""

    @pytest.mark.parametrize("name", sorted(ALL_SHAPES))
    def test_centroid_matches_pixel_centre_of_mass(self, name: str) -> None:
        contour = ALL_SHAPES[name]
        mask = _mask(contour)
        rows, cols = np.nonzero(mask)
        # Pixel (x, y) covers the unit square centred on (x + 0.5, y + 0.5).
        raster_x = float(cols.mean()) + 0.5
        raster_y = float(rows.mean()) + 0.5

        analytic_x, analytic_y = _analytic_centroid(contour)
        assert analytic_x == pytest.approx(raster_x, abs=1.0), name
        assert analytic_y == pytest.approx(raster_y, abs=1.0), name


# ---------------------------------------------------------------------------
# IoU and Dice
# ---------------------------------------------------------------------------


def _overlap_pairs() -> list[tuple[str, dict, dict]]:
    """Pairs spanning identical, nested, partial, touching and disjoint overlap."""
    square = _contour(_box(60, 60, 300, 300))
    return [
        ("identical", square, _contour(_box(60, 60, 300, 300))),
        ("identical_opposite_winding", square, _contour(_box_cw(60, 60, 300, 300))),
        ("half_overlap", square, _contour(_box(210, 60, 300, 300))),
        ("corner_overlap", square, _contour(_box(210, 210, 300, 300))),
        ("contained", square, _contour(_box(110, 110, 100, 100))),
        ("edge_touching", square, _contour(_box(360, 60, 100, 300))),
        ("disjoint", square, _contour(_box(400, 400, 80, 80))),
        (
            "concave_pair",
            RECTILINEAR["l_shape"],
            _contour(
                [(160, 160), (500, 160), (500, 360), (360, 360), (360, 500), (160, 500)]
            ),
        ),
        (
            "holed_vs_solid",
            RECTILINEAR["holed_square"],
            _contour(_box(50, 50, 400, 400)),
        ),
        (
            "holed_vs_holed_offset",
            RECTILINEAR["holed_square"],
            _contour(_box(150, 150, 350, 350), [_box(200, 200, 100, 100)]),
        ),
        (
            "nested_holes_vs_solid",
            RECTILINEAR["nested_holes"],
            _contour(_box(40, 40, 430, 430)),
        ),
        ("circle_pair", CURVED["circle"], _regular_circle_offset()),
    ]


def _regular_circle_offset() -> dict:
    return _contour(_regular_polygon(320, 320, 200, 128))


@plugin_required
class TestOverlapMatchesRasterizedOverlap:
    """`iou()` and `dice()` agree with pixel-counted overlap of the two masks."""

    @pytest.mark.parametrize(
        ("name", "a", "b"),
        _overlap_pairs(),
        ids=[case[0] for case in _overlap_pairs()],
    )
    def test_iou_matches_pixel_iou(self, name: str, a: dict, b: dict) -> None:
        mask_a, mask_b = _mask(a), _mask(b)
        union = int((mask_a | mask_b).sum())
        raster_iou = 0.0 if union == 0 else int((mask_a & mask_b).sum()) / union

        analytic = _pair_measure(a, b, lambda x, y: x.contour.iou(y))
        assert analytic == pytest.approx(raster_iou, abs=0.01), (
            f"{name}: analytic {analytic}, rasterized {raster_iou}"
        )

    @pytest.mark.parametrize(
        ("name", "a", "b"),
        _overlap_pairs(),
        ids=[case[0] for case in _overlap_pairs()],
    )
    def test_dice_matches_pixel_dice(self, name: str, a: dict, b: dict) -> None:
        mask_a, mask_b = _mask(a), _mask(b)
        total = int(mask_a.sum()) + int(mask_b.sum())
        raster_dice = 0.0 if total == 0 else 2.0 * int((mask_a & mask_b).sum()) / total

        analytic = _pair_measure(a, b, lambda x, y: x.contour.dice(y))
        assert analytic == pytest.approx(raster_dice, abs=0.01), (
            f"{name}: analytic {analytic}, rasterized {raster_dice}"
        )

    def test_rectilinear_iou_is_exact(self) -> None:
        """
        With integer vertices there is no discretization error left to hide in,
        so the two routes agree to floating-point noise rather than to a
        tolerance. A hole ignored on one side of the comparison shows up here.
        """
        a = RECTILINEAR["holed_square"]
        b = _contour(_box(50, 50, 400, 400))
        mask_a, mask_b = _mask(a), _mask(b)
        raster_iou = int((mask_a & mask_b).sum()) / int((mask_a | mask_b).sum())

        analytic = _pair_measure(a, b, lambda x, y: x.contour.iou(y))
        assert analytic == pytest.approx(raster_iou, abs=1e-9)
        # 400*400 - 200*200 over 400*400.
        assert analytic == pytest.approx(120000 / 160000, abs=1e-9)


# ---------------------------------------------------------------------------
# contains_point
# ---------------------------------------------------------------------------


@plugin_required
class TestContainsPointMatchesTheMask:
    """`contains_point()` agrees with the mask pixel covering that point."""

    @pytest.mark.parametrize("name", sorted(RECTILINEAR))
    def test_sampled_points_agree(self, name: str) -> None:
        """
        A lattice of pixel centres, checked both ways.

        Only rectilinear shapes are sampled: on a diagonal edge the two routes
        legitimately disagree for the pixels the boundary passes through.
        """
        contour = RECTILINEAR[name]
        mask = _mask(contour)

        centres = [
            (x + 0.5, y + 0.5)
            for y in range(5, CANVAS, 37)
            for x in range(5, CANVAS, 41)
        ]
        frame = pl.DataFrame(
            {
                "c": [contour] * len(centres),
                "p": [{"x": px, "y": py} for px, py in centres],
            },
            schema={
                "c": CONTOUR_SCHEMA,
                "p": pl.Struct({"x": pl.Float64, "y": pl.Float64}),
            },
        ).with_columns(inside=pl.col("c").contour.contains_point(pl.col("p")))

        for (px, py), inside in zip(centres, frame["inside"].to_list()):
            assert bool(mask[int(py), int(px)]) == bool(inside), (
                f"{name}: disagreement at ({px}, {py})"
            )


# ---------------------------------------------------------------------------
# Round trip: contour -> mask -> contour
# ---------------------------------------------------------------------------


def _extract(
    contour: dict, *, mode: str = "external", method: str = "simple"
) -> list[dict]:
    """Rasterize, then trace the mask back into contours."""
    frame = pl.DataFrame({"c": [contour]}, schema={"c": CONTOUR_SCHEMA})
    pipe = (
        Pipeline()
        .source("contour", width=CANVAS, height=CANVAS, fill_value=255, background=0)
        .extract_contours(mode=mode, method=method)
    )
    traced = frame.with_columns(r=pl.col("c").cv.pipe(pipe).sink("native"))["r"][0]
    return list(traced)


def _as_contour(border: dict) -> dict:
    """A traced border on its own, with no holes."""
    return {"exterior": border["exterior"], "holes": [], "is_closed": True}


def _reassemble(borders: list[dict]) -> dict:
    """
    Rebuild one contour from the borders of a region: biggest is the exterior.

    `extract_contours` returns borders as a flat list without hierarchy, so
    putting a holed shape back together is the caller's job. Ranking by area is
    enough for the shapes here, where every hole sits inside a single exterior.
    """
    ranked = sorted(
        borders,
        key=lambda b: _analytic_area(_as_contour(b)),
        reverse=True,
    )
    return {
        "exterior": ranked[0]["exterior"],
        "holes": [b["exterior"] for b in ranked[1:]],
        "is_closed": True,
    }


HOLE_FREE = {name: shape for name, shape in ALL_SHAPES.items() if not shape["holes"]}

# Holes that touch or nest merge into one background region, so the mask has
# fewer holes than the contour was given rings. That is the same union-of-rings
# semantics `area` reports, arrived at through pixels instead of polygon
# arithmetic: `overlapping_holes` and `nested_holes` were each built from two
# rings and enclose one region.
CONNECTED_HOLE_COUNT = {
    "holed_square": 1,
    "holed_square_cw_hole": 1,
    "two_disjoint_holes": 2,
    "overlapping_holes": 1,
    "nested_holes": 1,
    "concave_with_hole": 1,
    "circle_with_circular_hole": 1,
}


@plugin_required
class TestRoundTripThroughExtraction:
    """
    A contour survives being rasterized and traced back.

    The trip is lossy in one specific, predictable way: the tracer reports the
    **centres of the boundary pixels**, so the polygon that comes back is inset
    by half a pixel all round. A `w x h` box returns as `(w-1) x (h-1)`. The
    tests below assert that inset rather than tolerate it — a bug that shifted
    the outline the other way, or collapsed it, would have to survive an exact
    equality to go unnoticed.
    """

    @pytest.mark.parametrize("name", sorted(ALL_SHAPES))
    def test_one_region_yields_one_external_contour(self, name: str) -> None:
        """
        Every shape here is a single connected region, so it traces to one
        border. Before the tracer's sweep was fixed, a filled square came back as
        one degenerate 2x2 contour per row — 399 of them at this canvas size.
        """
        assert len(_extract(ALL_SHAPES[name])) == 1

    def test_disjoint_regions_yield_one_contour_each(
        self, encode_png: "Callable"
    ) -> None:
        """
        Separate regions stay separate. A mask is the only way to state this — a
        contour has one exterior — so this one enters through an image.
        """
        image = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        image[40:160, 40:160] = 255
        image[300:450, 300:450] = 255

        frame = pl.DataFrame({"img": [encode_png(image)]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours(mode="external", method="simple")
        )
        traced = frame.select(out=pl.col("img").cv.pipe(pipe).sink("native"))["out"][0]

        assert len(traced) == 2
        # Each is its own square, not one contour spanning both.
        for border in traced:
            xs = [p["x"] for p in border["exterior"]]
            assert max(xs) - min(xs) < 200

    @pytest.mark.parametrize(
        ("name", "width", "height"),
        [("square", 400, 400), ("wide_rect", 470, 120)],
    )
    def test_box_round_trips_to_an_exact_one_pixel_inset(
        self, name: str, width: int, height: int
    ) -> None:
        """
        The traced outline runs through the centres of the rim pixels, so a box
        filling `w x h` pixels returns bounding `(w-1) x (h-1)`. Exact, not
        approximate — this is the assertion that pins the tracer's convention.
        """
        traced = _as_contour(_extract(RECTILINEAR[name])[0])
        assert _analytic_area(traced) == pytest.approx(
            (width - 1) * (height - 1), abs=1e-9
        )

    def test_box_round_trips_to_four_corners(self) -> None:
        """`method="simple"` drops the collinear rim points, leaving the corners."""
        traced = _extract(RECTILINEAR["square"])[0]
        assert len(traced["exterior"]) == 4

    @pytest.mark.parametrize(
        ("name", "slack"),
        # Half a pixel along each unit of boundary. On an axis-aligned edge that
        # bound is tight. A sloped edge rasterizes to a staircase whose length runs
        # up to sqrt(2) times the edge it approximates, and the inset follows the
        # staircase, so the sloped shapes get that factor and nothing more.
        [(name, 1.0) for name in sorted(RECTILINEAR) if not RECTILINEAR[name]["holes"]]
        + [
            (name, math.sqrt(2.0))
            for name in sorted(CURVED)
            if not CURVED[name]["holes"]
        ],
    )
    def test_round_trip_loses_area_but_never_gains_it(
        self, name: str, slack: float
    ) -> None:
        """
        The inset can only shrink the shape, and only by so much. A trace that
        wandered into the interior, or one that overshot the rim, breaks these
        bounds from one side or the other.
        """
        contour = ALL_SHAPES[name]
        original = _analytic_area(contour)
        traced = _analytic_area(_as_contour(_extract(contour)[0]))
        deficit = original - traced

        assert deficit >= -1.0, f"{name}: round trip gained {-deficit} of area"
        assert deficit <= _perimeter(contour) / 2 * slack + 1.0, (
            f"{name}: lost {deficit}, more than the inset can account for"
        )

    @pytest.mark.parametrize("name", sorted(HOLE_FREE))
    def test_round_trip_iou_is_high(self, name: str) -> None:
        contour = HOLE_FREE[name]
        traced = _as_contour(_extract(contour)[0])
        assert _pair_measure(contour, traced, lambda a, b: a.contour.iou(b)) > 0.98

    @pytest.mark.parametrize("name", sorted(RECTILINEAR))
    def test_traced_outline_lies_inside_the_original(self, name: str) -> None:
        """
        Inset, never outset. Where the trace lies wholly inside the original, the
        intersection *is* the traced area and IoU collapses to the ratio of the two
        areas — so asserting that identity checks containment and pins the size at
        once. An outset or wandering trace drives IoU well below the ratio.

        The tolerance is for reflex corners, where the 8-connected walk cuts the
        inner angle and pokes a pixel or two outside; `plus_shape`, with four of
        them, is the case that needs it.
        """
        contour = RECTILINEAR[name]
        traced = _as_contour(_extract(contour)[0])
        # Compare against the solid outline: `external` mode drops hole borders.
        solid = {"exterior": contour["exterior"], "holes": [], "is_closed": True}

        ratio = _analytic_area(traced) / _analytic_area(solid)
        assert _pair_measure(solid, traced, lambda a, b: a.contour.iou(b)) == (
            pytest.approx(ratio, abs=1e-4)
        )

    @pytest.mark.parametrize("name", sorted(CONNECTED_HOLE_COUNT))
    def test_hole_borders_come_back_as_separate_borders(self, name: str) -> None:
        """
        `mode="all"` reports the exterior plus one border per enclosed background
        region — and rings that overlap or nest enclose *one* region between them,
        which is the union-of-rings semantics `area` uses, reached through pixels.
        """
        borders = _extract(ALL_SHAPES[name], mode="all")
        assert len(borders) == 1 + CONNECTED_HOLE_COUNT[name]

    @pytest.mark.parametrize("name", sorted(CONNECTED_HOLE_COUNT))
    def test_holed_shape_reassembles_to_the_original(self, name: str) -> None:
        """The full trip for a holed shape: rings out, region back."""
        contour = ALL_SHAPES[name]
        rebuilt = _reassemble(_extract(contour, mode="all"))

        assert _pair_measure(contour, rebuilt, lambda a, b: a.contour.iou(b)) > 0.98
        # The inset applies to the exterior and to each hole rim, and they pull in
        # opposite directions, so the net area stays within a perimeter's worth.
        assert _analytic_area(rebuilt) == pytest.approx(
            _analytic_area(contour), abs=_perimeter(contour)
        )

    def test_each_pass_erodes_by_exactly_one_pixel(self) -> None:
        """
        The trip is not idempotent, and shouldn't be claimed to be: every pass
        insets by half a pixel on each side. Pinning the erosion exactly is what
        keeps that a known property rather than a drifting one.
        """
        contour = RECTILINEAR["square"]  # fills 400 x 400 pixels
        for expected_side in (399, 398, 397):
            contour = _as_contour(_extract(contour)[0])
            assert _analytic_area(contour) == pytest.approx(
                expected_side**2, abs=1e-9
            ), f"expected a {expected_side}px square"
