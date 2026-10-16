"""The polar bear: crude primitives, fused and traced into a low-poly outline.

Only the primitives are hand-placed. Everything after is a polars-cv pipeline:
``rasterize`` the overlapping blobs, ``blur`` so neighbours bleed into each
other, ``threshold`` at an iso-level that fuses them into one smooth body (the
metaball trick), then ``extract_contours`` -> ``largest`` -> ``simplify``, whose
Douglas-Peucker tolerance is what gives the silhouette its facets.

Units are pixels at 1x with ``y`` growing downward; the feet rest on y = 141.
"""

from __future__ import annotations

import math

import polars as pl

from polars_cv import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SET_SCHEMA

WORK_SCALE = 2.0  # the sculpting stage runs at this multiple of 1x
CANVAS = (260, 160)  # (width, height) of the bear's own canvas at 1x
SIGMA = 4.5  # blur radius that sets how far the blobs fuse
ISO_LEVEL = 120  # threshold on the blurred u8 mask
TOLERANCE = 2.2  # Douglas-Peucker tolerance: larger is more faceted


def _contour(points: list[tuple[float, float]]) -> dict:
    k = WORK_SCALE
    return {
        "exterior": [{"x": x * k, "y": y * k} for x, y in points],
        "holes": [],
        "is_closed": True,
    }


def _ellipse(cx, cy, rx, ry, rot=0.0, n=72) -> dict:
    a = math.radians(rot)
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        pts.append(
            (
                cx + x * math.cos(a) - y * math.sin(a),
                cy + x * math.sin(a) + y * math.cos(a),
            )
        )
    return _contour(pts)


def _capsule(x0, y0, x1, y1, w, n=24) -> dict:
    """A stadium between two points (a leg)."""
    ang = math.atan2(y1 - y0, x1 - x0)
    r = w / 2
    pts = []
    for cx, cy, start in ((x1, y1, ang - math.pi / 2), (x0, y0, ang + math.pi / 2)):
        for i in range(n + 1):
            t = start + math.pi * i / n
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return _contour(pts)


def sketch() -> list[dict]:
    """A walking polar bear, facing right, from overlapping blobs."""
    return [
        _ellipse(118, 72, 66, 34),  # body
        _ellipse(64, 72, 38, 34),  # rump
        _ellipse(150, 62, 36, 30, -8),  # shoulder hump
        _ellipse(182, 81, 27, 12, 30),  # neck, sloping down
        _ellipse(208, 90, 20, 16, 5),  # head
        _ellipse(229, 97, 15, 8.5, 12),  # snout
        _ellipse(202, 64, 9.5, 12.5, -25),  # ear: large enough to survive the blur
        _capsule(167, 88, 178, 139, 21),  # near front leg
        _capsule(143, 90, 144, 138, 17),  # far front leg
        _capsule(76, 92, 84, 140, 24),  # near back leg
        _capsule(53, 90, 50, 138, 18),  # far back leg
        _ellipse(186, 141, 15, 5),  # front paw
        _ellipse(94, 141, 15, 5),  # back paw
        _ellipse(29, 66, 8, 7),  # tail
    ]


def _canvas() -> tuple[int, int]:
    return int(CANVAS[0] * WORK_SCALE), int(CANVAS[1] * WORK_SCALE)


def _frame(contours: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({"c": [contours]}, schema={"c": CONTOUR_SET_SCHEMA})


def _fused() -> Pipeline:
    w, h = _canvas()
    return (
        Pipeline()
        .source("contour")
        .rasterize(width=w, height=h)
        .blur(SIGMA * WORK_SCALE)
        .threshold(ISO_LEVEL)
    )


def outline() -> list[tuple[float, float]]:
    """The traced, simplified bear exterior, in 1x coordinates."""
    pipe = (
        _fused()
        .extract_contours(mode="external")
        .largest()
        .simplify(TOLERANCE * WORK_SCALE)
    )
    traced = _frame(sketch()).select(pl.col("c").cv.pipe(pipe).sink("native")).item()
    k = WORK_SCALE
    return [(p["x"] / k, p["y"] / k) for p in traced[0]["exterior"]]


def stages() -> dict[str, bytes]:
    """PNGs of the three stages: the sketch, the fused body, the low-poly trace."""
    w, h = _canvas()
    raster = Pipeline().source("contour").rasterize(width=w, height=h)
    low_poly = [_contour(outline())]
    return {
        name: _frame(c).select(pl.col("c").cv.pipe(p).sink("png")).item()
        for name, c, p in (
            ("1-sketch", sketch(), raster),
            ("2-fused", sketch(), _fused()),
            ("3-low-poly", low_poly, raster),
        )
    }
