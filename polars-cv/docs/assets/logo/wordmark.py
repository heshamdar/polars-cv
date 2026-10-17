"""The "polars-cv" wordmark as contours: a small geometric font of strokes.

Every glyph is a few centrelines, which ``stroke(width=STROKE, cap="butt")``
turns into letters:

* stems and bars are two-point lines (a butt cap ends them flat);
* bowls (``o``, ``p``, ``a``) are circles, drawn as lines that return to their
  first point (no ends to cap);
* curves (``c``, ``s``, the shoulder of ``r``) are arcs, whose butt caps cut
  them square to the stroke.

``v`` is the exception: a filled polygon, whose flat top and bottom no stroke
gives. A glyph may mix both, so the layout returns its lines (to stroke) and
its regions (to fill) apart: ``stroke`` would outline a region, not fill it.

Units are pixels at 1x with ``y`` growing downward; glyphs sit on ``baseline``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

Point = tuple[float, float]
#: A glyph's contours: a line (open) or a region (closed), as points.
Path = tuple[list[Point], bool]
Glyph = Callable[[float, float], tuple[list[Path], float]]

STROKE = 14.0
X_HEIGHT = 70.0
ASCENDER = 100.0
DESCENDER = 30.0
SPACING = 14.0

_R = X_HEIGHT / 2  # bowl outer radius
_H = STROKE / 2  # a stroke's half-width: centrelines sit this far inside


def _line(*points: Point) -> Path:
    return list(points), False


def _arc(cx: float, cy: float, r: float, t0: float, t1: float, n: int = 64) -> Path:
    """A centreline arc from ``t0`` to ``t1`` degrees (counter-clockwise, y-up)."""
    ts = [math.radians(t0 + (t1 - t0) * i / n) for i in range(n + 1)]
    return [(cx + r * math.cos(t), cy - r * math.sin(t)) for t in ts], False


def _circle(cx: float, cy: float, r: float) -> Path:
    """A circular centreline that returns to its first point — exactly, so it
    is a loop with no ends to cap (360 degrees lands a rounding error away)."""
    points, _ = _arc(cx, cy, r, 0.0, 360.0, n=96)
    return [*points[:-1], points[0]], False


def _bowl(x: float, b: float) -> Path:
    return _circle(x + _R, b - _R, _R - _H)


# Each glyph takes the pen position and baseline and returns
# (contours, advance).
def _o(x, b):
    return [_bowl(x, b)], 2 * _R


def _p(x, b):
    return [_bowl(x, b), _line((x + _H, b - X_HEIGHT), (x + _H, b + DESCENDER))], 2 * _R


def _a(x, b):
    stem = x + 2 * _R - _H
    return [_bowl(x, b), _line((stem, b - X_HEIGHT), (stem, b))], 2 * _R


def _l(x, b):
    return [_line((x + _H, b - ASCENDER), (x + _H, b))], STROKE


def _r(x, b):
    rr = 0.8 * _R  # shoulder radius
    return [
        _line((x + _H, b - X_HEIGHT), (x + _H, b)),
        _arc(x + rr, b - X_HEIGHT + rr, rr - _H, 180, 45),
    ], 1.45 * rr


def _s(x, b):
    rc = (X_HEIGHT - STROKE) / 4  # centreline radius of each stacked half
    cx = x + rc + _H
    return [
        _arc(cx, b - X_HEIGHT + _H + rc, rc, 20, 270),  # upper hook
        _arc(cx, b - _H - rc, rc, 90, -160),  # lower hook
    ], 2 * (rc + _H)


def _hyphen(x, b):
    w = 0.45 * X_HEIGHT
    return [_line((x, b - _R), (x + w, b - _R))], w


def _c(x, b):
    return [_arc(x + _R, b - _R, _R - _H, 45, 315)], 2 * _R - 4


def _v(x, b):
    w = 0.95 * X_HEIGHT
    top, mid = b - X_HEIGHT, x + w / 2
    sw = 1.15 * STROKE  # horizontal width at the top (diagonals read thinner)
    outline = [
        (x, top),
        (x + sw, top),
        (mid, b - 1.9 * STROKE),
        (x + w - sw, top),
        (x + w, top),
        (mid + sw / 2, b),
        (mid - sw / 2, b),
    ]
    return [(outline, True)], w


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


def _contour(path: Path, scale: float) -> dict:
    points, closed = path
    ring = [{"x": x * scale, "y": y * scale} for x, y in points]
    return {"exterior": ring, "holes": [], "is_closed": closed}


def wordmark(
    text: str, *, x: float, baseline: float, scale: float = 1.0
) -> tuple[list[dict], list[dict], float]:
    """Lay out ``text`` from pen position ``x``.

    Returns two contour sets (in ``CONTOUR_SCHEMA`` form, coordinates
    multiplied by ``scale``) — the lines, to stroke at ``STROKE * scale``, and
    the regions, to fill — and the pen position after the last glyph's
    advance.
    """
    lines, regions = [], []
    for i, ch in enumerate(text):
        if i:
            x += SPACING
        paths, adv = GLYPHS[ch](x, baseline)
        for path in paths:
            (regions if path[1] else lines).append(_contour(path, scale))
        x += adv
    return lines, regions, x
