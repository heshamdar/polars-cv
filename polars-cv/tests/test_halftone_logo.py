"""The halftone logo (``docs/assets/logo/halftone_logo.py``) keeps drawing.

Its geometry is a polars query (no plugin needed); its rendering is one
polars-cv graph, checked at the pixels that make it a bear: dots on the face,
empty eyes, an empty background.
"""

from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import pytest

from tests.conftest import plugin_required

LOGO_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "logo"
sys.path.insert(0, str(LOGO_DIR))
import halftone_logo as logo  # noqa: E402


def _outlines():
    return logo.outlines(origin=(0.0, 0.0), size=1.0)


def test_each_layer_is_one_contour_set_row():
    df = _outlines()
    assert df.height == 1
    assert {"head", "muzzle", "feature"} <= set(df.columns)
    assert len(df["head"].item()) == 1
    assert len(df["feature"].item()) == 3  # nose and two eyes


def test_the_head_curve_has_two_ears():
    """The crown (straight up) dips below an ear on each side of it."""
    (head,) = _outlines()["head"].item()
    pts = head["exterior"]
    crown = pts[logo.N // 4]  # theta = pi / 2
    assert math.isclose(crown["x"], logo.PARTS["cx"][0], abs_tol=1e-9)
    left = min(p["y"] for p in pts if p["x"] < crown["x"] - 0.05)
    right = min(p["y"] for p in pts if p["x"] > crown["x"] + 0.05)
    assert left < crown["y"] and right < crown["y"]  # y grows downward


@pytest.fixture
def image():
    pil = pytest.importorskip("PIL.Image")
    return lambda png: pil.open(io.BytesIO(png)).convert("RGB")


def _near(pixel, colour, tol=24):
    return all(abs(a - b) <= tol for a, b in zip(pixel, colour, strict=True))


@plugin_required
def test_mark_renders_a_bear(image):
    """At the user-facing entry point: dots on the face, voids for the eyes.
    The PNG is anti-aliased, so colours are compared with a tolerance."""
    size = 256
    img = image(logo.render_mark(size))
    assert img.size == (size, size)
    background = logo.PALETTE[0]
    assert _near(img.getpixel((2, 2)), background)
    m = logo.MARGIN

    def at(cx, cy):
        return round((m + (1 - 2 * m) * cx) * size), round(
            (m + (1 - 2 * m) * cy) * size
        )

    # An eye is a void in the dots: the background shows through.
    eye = logo.PARTS.filter(logo.PARTS["layer"] == "feature").row(1, named=True)
    assert _near(img.getpixel(at(eye["cx"], eye["cy"])), background)
    # The forehead, between the eyes and the crown, is mostly dots.
    head = logo.PARTS.row(0, named=True)
    y = at(0.5, (eye["cy"] + head["cy"] - head["sy"] * head["r"]) / 2)[1]
    (x0, _), (x1, _) = at(0.4, 0.0), at(0.6, 0.0)
    row = [img.getpixel((x, y))[0] for x in range(x0, x1)]
    mid = (background[0] + logo.PALETTE[logo.FACE][0]) / 2
    assert sum(row) / len(row) > mid


@plugin_required
def test_lockup_renders_at_its_size(image):
    img = image(logo.render_lockup())
    assert img.size == (logo.LOCKUP_WIDTH, logo.LOCKUP_HEIGHT)
    assert tuple(logo.PALETTE[logo.TEXT]) in set(img.getdata())
