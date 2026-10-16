"""The "polars-cv" wordmark as contours: a small geometric font of primitives.

Every glyph is a union of closed contours, which ``rasterize`` paints as one
shape, so overlapping primitives need no boolean geometry:

* stems and bars are 4-point rectangles;
* bowls (``o``, ``p``, ``a``) are a circle exterior with a circle hole;
* curves (``c``, ``s``, the shoulder of ``r``) are annular sectors, an outer
  arc joined to the reversed inner arc;
* ``v`` is a single non-convex polygon.

Units are pixels at 1x with ``y`` growing downward; glyphs sit on ``baseline``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

Point = tuple[float, float]
Glyph = Callable[[float, float], tuple[list[list[list[Point]]], float]]

STROKE = 14.0
X_HEIGHT = 70.0
ASCENDER = 100.0
DESCENDER = 30.0
SPACING = 14.0

_R = X_HEIGHT / 2  # bowl outer radius


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[list[Point]]:
    return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]


def _circle(cx: float, cy: float, r: float, n: int = 96) -> list[Point]:
    return [
        (cx + r * math.cos(2 * math.pi * i / n), cy - r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def _ring(cx: float, cy: float, r_out: float, r_in: float) -> list[list[Point]]:
    return [_circle(cx, cy, r_out), _circle(cx, cy, r_in)]


def _arc(
    cx: float, cy: float, r_out: float, r_in: float, t0: float, t1: float, n: int = 64
) -> list[list[Point]]:
    """Annular sector from ``t0`` to ``t1`` degrees (counter-clockwise, y-up)."""
    ts = [math.radians(t0 + (t1 - t0) * i / n) for i in range(n + 1)]
    outer = [(cx + r_out * math.cos(t), cy - r_out * math.sin(t)) for t in ts]
    inner = [(cx + r_in * math.cos(t), cy - r_in * math.sin(t)) for t in reversed(ts)]
    return [outer + inner]


# Each glyph takes the pen position and baseline and returns
# (primitives, advance); a primitive is [exterior, *holes].
def _o(x, b):
    return [_ring(x + _R, b - _R, _R, _R - STROKE)], 2 * _R


def _p(x, b):
    return [
        _ring(x + _R, b - _R, _R, _R - STROKE),
        _rect(x, b - X_HEIGHT, x + STROKE, b + DESCENDER),
    ], 2 * _R


def _a(x, b):
    return [
        _ring(x + _R, b - _R, _R, _R - STROKE),
        _rect(x + 2 * _R - STROKE, b - X_HEIGHT, x + 2 * _R, b),
    ], 2 * _R


def _l(x, b):
    return [_rect(x, b - ASCENDER, x + STROKE, b)], STROKE


def _r(x, b):
    rr = 0.8 * _R  # shoulder radius
    return [
        _rect(x, b - X_HEIGHT, x + STROKE, b),
        _arc(x + rr, b - X_HEIGHT + rr, rr, rr - STROKE, 180, 45),
    ], 1.45 * rr


def _s(x, b):
    rc = (X_HEIGHT - STROKE) / 4  # centreline radius of each stacked half
    h = STROKE / 2
    cx = x + rc + h
    return [
        _arc(cx, b - X_HEIGHT + h + rc, rc + h, rc - h, 20, 270),  # upper hook
        _arc(cx, b - h - rc, rc + h, rc - h, 90, -160),  # lower hook
    ], 2 * (rc + h)


def _hyphen(x, b):
    w = 0.45 * X_HEIGHT
    return [_rect(x, b - _R - STROKE / 2, x + w, b - _R + STROKE / 2)], w


def _c(x, b):
    return [_arc(x + _R, b - _R, _R, _R - STROKE, 45, 315)], 2 * _R - 4


def _v(x, b):
    w = 0.95 * X_HEIGHT
    top, mid = b - X_HEIGHT, x + w / 2
    sw = 1.15 * STROKE  # horizontal width at the top (diagonals read thinner)
    return [
        [
            [
                (x, top),
                (x + sw, top),
                (mid, b - 1.9 * STROKE),
                (x + w - sw, top),
                (x + w, top),
                (mid + sw / 2, b),
                (mid - sw / 2, b),
            ]
        ]
    ], w


GLYPHS: dict[str, Glyph] = {
    "p": _p,
    "o": _o,
    "l": _l,
    "a": _a,
    "r": _r,
    "s": _s,
    "-": _hyphen,
    "c": _c,
    "v": _v,
}


def _contour(rings: list[list[Point]], scale: float) -> dict:
    pts = [[{"x": x * scale, "y": y * scale} for x, y in r] for r in rings]
    return {"exterior": pts[0], "holes": pts[1:], "is_closed": True}


def wordmark(
    text: str, *, x: float, baseline: float, scale: float = 1.0
) -> tuple[list[dict], float]:
    """Lay out ``text`` from pen position ``x``.

    Returns the contour set (in ``CONTOUR_SCHEMA`` form, coordinates multiplied
    by ``scale``) and the pen position after the last glyph's advance.
    """
    contours = []
    for i, ch in enumerate(text):
        if i:
            x += SPACING
        prims, adv = GLYPHS[ch](x, baseline)
        contours.extend(_contour(p, scale) for p in prims)
        x += adv
    return contours, x
