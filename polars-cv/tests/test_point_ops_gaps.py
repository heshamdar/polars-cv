"""
Tests filling gaps in point operation coverage.

Covers: point.distance(), point.manhattan_distance(), point.x(), point.y(),
translate/scale correctness verification, and normalize/to_absolute round-trip.
"""

from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest
from polars_cv.geometry import POINT_SCHEMA


def _plugin_available() -> bool:
    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    so_files = list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd"))
    return len(so_files) > 0


plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


# ---------------------------------------------------------------------------
# x() / y() extraction
# ---------------------------------------------------------------------------


class TestPointExtraction:
    """Tests for point.x() and point.y() field extraction."""

    def test_extract_x(self) -> None:
        """point.x() should extract the x coordinate."""
        df = pl.DataFrame(
            {"pt": [{"x": 42.5, "y": 99.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.select(x=pl.col("pt").point.x())
        assert result["x"][0] == pytest.approx(42.5)

    def test_extract_y(self) -> None:
        """point.y() should extract the y coordinate."""
        df = pl.DataFrame(
            {"pt": [{"x": 42.5, "y": 99.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.select(y=pl.col("pt").point.y())
        assert result["y"][0] == pytest.approx(99.0)

    def test_extract_multiple_rows(self) -> None:
        """x/y extraction should work across multiple rows."""
        df = pl.DataFrame(
            {
                "pt": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 10.0, "y": 20.0},
                    {"x": -5.0, "y": 100.0},
                ]
            },
            schema={"pt": POINT_SCHEMA},
        )
        result = df.select(
            x=pl.col("pt").point.x(),
            y=pl.col("pt").point.y(),
        )
        assert result["x"].to_list() == [0.0, 10.0, -5.0]
        assert result["y"].to_list() == [0.0, 20.0, 100.0]


# ---------------------------------------------------------------------------
# distance / manhattan_distance
# ---------------------------------------------------------------------------


@plugin_required
class TestPointDistance:
    """Tests for point distance operations."""

    def test_euclidean_distance_known(self) -> None:
        """Distance between (0,0) and (3,4) should be 5."""
        df = pl.DataFrame(
            {
                "a": [{"x": 0.0, "y": 0.0}],
                "b": [{"x": 3.0, "y": 4.0}],
            },
            schema={"a": POINT_SCHEMA, "b": POINT_SCHEMA},
        )
        result = df.with_columns(dist=pl.col("a").point.distance(pl.col("b")))
        assert result["dist"][0] == pytest.approx(5.0)

    def test_euclidean_distance_same_point_is_zero(self) -> None:
        """Distance from a point to itself should be 0."""
        df = pl.DataFrame(
            {
                "a": [{"x": 50.0, "y": 50.0}],
                "b": [{"x": 50.0, "y": 50.0}],
            },
            schema={"a": POINT_SCHEMA, "b": POINT_SCHEMA},
        )
        result = df.with_columns(dist=pl.col("a").point.distance(pl.col("b")))
        assert result["dist"][0] == pytest.approx(0.0)

    def test_euclidean_distance_is_symmetric(self) -> None:
        """d(a, b) == d(b, a)."""
        df = pl.DataFrame(
            {
                "a": [{"x": 10.0, "y": 20.0}],
                "b": [{"x": 30.0, "y": 50.0}],
            },
            schema={"a": POINT_SCHEMA, "b": POINT_SCHEMA},
        )
        result = df.with_columns(
            d_ab=pl.col("a").point.distance(pl.col("b")),
            d_ba=pl.col("b").point.distance(pl.col("a")),
        )
        assert result["d_ab"][0] == pytest.approx(result["d_ba"][0])

    def test_euclidean_distance_diagonal(self) -> None:
        """Distance of unit diagonal should be sqrt(2)."""
        df = pl.DataFrame(
            {
                "a": [{"x": 0.0, "y": 0.0}],
                "b": [{"x": 1.0, "y": 1.0}],
            },
            schema={"a": POINT_SCHEMA, "b": POINT_SCHEMA},
        )
        result = df.with_columns(dist=pl.col("a").point.distance(pl.col("b")))
        assert result["dist"][0] == pytest.approx(math.sqrt(2))

    def test_manhattan_distance_known(self) -> None:
        """Manhattan distance between (0,0) and (3,4) should be 7."""
        df = pl.DataFrame(
            {
                "a": [{"x": 0.0, "y": 0.0}],
                "b": [{"x": 3.0, "y": 4.0}],
            },
            schema={"a": POINT_SCHEMA, "b": POINT_SCHEMA},
        )
        result = df.with_columns(dist=pl.col("a").point.manhattan_distance(pl.col("b")))
        assert result["dist"][0] == pytest.approx(7.0)

    def test_manhattan_distance_same_point_is_zero(self) -> None:
        """Manhattan distance from a point to itself should be 0."""
        df = pl.DataFrame(
            {
                "a": [{"x": 50.0, "y": 50.0}],
                "b": [{"x": 50.0, "y": 50.0}],
            },
            schema={"a": POINT_SCHEMA, "b": POINT_SCHEMA},
        )
        result = df.with_columns(dist=pl.col("a").point.manhattan_distance(pl.col("b")))
        assert result["dist"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# translate correctness
# ---------------------------------------------------------------------------


@plugin_required
class TestPointTranslate:
    """Verify translate produces correct coordinate changes."""

    def test_translate_positive(self) -> None:
        """translate(dx=10, dy=20) should add to coordinates."""
        df = pl.DataFrame(
            {"pt": [{"x": 5.0, "y": 15.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.with_columns(moved=pl.col("pt").point.translate(dx=10.0, dy=20.0))
        p = result["moved"][0]
        assert p["x"] == pytest.approx(15.0)
        assert p["y"] == pytest.approx(35.0)

    def test_translate_negative(self) -> None:
        """translate(dx=-5, dy=-10) should subtract from coordinates."""
        df = pl.DataFrame(
            {"pt": [{"x": 20.0, "y": 30.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.with_columns(moved=pl.col("pt").point.translate(dx=-5.0, dy=-10.0))
        p = result["moved"][0]
        assert p["x"] == pytest.approx(15.0)
        assert p["y"] == pytest.approx(20.0)

    def test_translate_zero_is_identity(self) -> None:
        """translate(dx=0, dy=0) should be identity."""
        df = pl.DataFrame(
            {"pt": [{"x": 42.0, "y": 99.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.with_columns(moved=pl.col("pt").point.translate(dx=0.0, dy=0.0))
        p = result["moved"][0]
        assert p["x"] == pytest.approx(42.0)
        assert p["y"] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# scale correctness
# ---------------------------------------------------------------------------


@plugin_required
class TestPointScale:
    """Verify scale produces correct coordinate changes."""

    def test_scale_doubles(self) -> None:
        """scale(sx=2, sy=2) should double coordinates."""
        df = pl.DataFrame(
            {"pt": [{"x": 10.0, "y": 20.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.with_columns(scaled=pl.col("pt").point.scale(sx=2.0, sy=2.0))
        p = result["scaled"][0]
        assert p["x"] == pytest.approx(20.0)
        assert p["y"] == pytest.approx(40.0)

    def test_scale_identity(self) -> None:
        """scale(sx=1, sy=1) should be identity."""
        df = pl.DataFrame(
            {"pt": [{"x": 10.0, "y": 20.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.with_columns(scaled=pl.col("pt").point.scale(sx=1.0, sy=1.0))
        p = result["scaled"][0]
        assert p["x"] == pytest.approx(10.0)
        assert p["y"] == pytest.approx(20.0)

    def test_scale_asymmetric(self) -> None:
        """scale(sx=0.5, sy=3.0) should scale axes independently."""
        df = pl.DataFrame(
            {"pt": [{"x": 100.0, "y": 10.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.with_columns(scaled=pl.col("pt").point.scale(sx=0.5, sy=3.0))
        p = result["scaled"][0]
        assert p["x"] == pytest.approx(50.0)
        assert p["y"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# normalize / to_absolute round-trip
# ---------------------------------------------------------------------------


@plugin_required
class TestPointNormalizeRoundTrip:
    """Verify normalize → to_absolute produces original coordinates."""

    def test_round_trip(self) -> None:
        """normalize then to_absolute should recover original point."""
        df = pl.DataFrame(
            {"pt": [{"x": 50.0, "y": 75.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.with_columns(
            restored=pl.col("pt")
            .point.normalize(ref_width=200, ref_height=300)
            .point.to_absolute(ref_width=200, ref_height=300)
        )
        p = result["restored"][0]
        assert p["x"] == pytest.approx(50.0, abs=0.1)
        assert p["y"] == pytest.approx(75.0, abs=0.1)

    def test_normalize_scales_to_unit(self) -> None:
        """After normalize, coordinates should be in [0, 1]."""
        df = pl.DataFrame(
            {"pt": [{"x": 50.0, "y": 75.0}]},
            schema={"pt": POINT_SCHEMA},
        )
        result = df.with_columns(
            normed=pl.col("pt").point.normalize(ref_width=100, ref_height=150)
        )
        p = result["normed"][0]
        assert 0.0 <= p["x"] <= 1.0
        assert 0.0 <= p["y"] <= 1.0
        assert p["x"] == pytest.approx(0.5)
        assert p["y"] == pytest.approx(0.5)
