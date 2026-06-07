"""
Tests that expression parameters (pl.Expr) are properly rejected where unsupported.

Operations in the contour and point namespaces accept literal values but not
pl.Expr arguments. After a bugfix where expression arguments were silently
dropped, these operations now raise TypeError. This test file verifies that
behaviour for every affected method.
"""

from __future__ import annotations

import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Contour namespace – TypeError on pl.Expr parameters
# ---------------------------------------------------------------------------


class TestContourExpressionParams:
    """Verify that contour operations raise TypeError for pl.Expr arguments."""

    @pytest.fixture
    def square_contour(self) -> dict:
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

    # --- normalize ---

    def test_normalize_rejects_expr_ref_width(self, square_contour: dict) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.normalize(ref_width=pl.col("w"), ref_height=100)

    def test_normalize_rejects_expr_ref_height(self, square_contour: dict) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.normalize(ref_width=100, ref_height=pl.col("h"))

    def test_normalize_rejects_both_expr(self, square_contour: dict) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.normalize(ref_width=pl.col("w"), ref_height=pl.col("h"))

    def test_normalize_accepts_literal_ints(self, square_contour: dict) -> None:
        """Sanity check: literal ints should not raise."""
        # Should not raise – just builds the expression, no execution needed
        expr = pl.col("c").contour.normalize(ref_width=100, ref_height=200)
        assert expr is not None

    # --- to_absolute ---

    def test_to_absolute_rejects_expr_ref_width(self) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.to_absolute(ref_width=pl.col("w"), ref_height=100)

    def test_to_absolute_rejects_expr_ref_height(self) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.to_absolute(ref_width=100, ref_height=pl.col("h"))

    def test_to_absolute_accepts_literals(self) -> None:
        expr = pl.col("c").contour.to_absolute(ref_width=640, ref_height=480)
        assert expr is not None

    # --- translate ---

    def test_translate_rejects_expr_dx(self) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.translate(dx=pl.col("dx"), dy=10.0)

    def test_translate_rejects_expr_dy(self) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.translate(dx=10.0, dy=pl.col("dy"))

    def test_translate_accepts_literals(self) -> None:
        expr = pl.col("c").contour.translate(dx=5.0, dy=-3.0)
        assert expr is not None

    # --- scale ---

    def test_scale_rejects_expr_sx(self) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.scale(sx=pl.col("s"), sy=2.0)

    def test_scale_rejects_expr_sy(self) -> None:
        with pytest.raises(TypeError, match="expression parameters"):
            pl.col("c").contour.scale(sx=2.0, sy=pl.col("s"))

    def test_scale_accepts_literals(self) -> None:
        expr = pl.col("c").contour.scale(sx=2.0, sy=0.5)
        assert expr is not None


# ---------------------------------------------------------------------------
# Point namespace – TypeError on pl.Expr parameters
# ---------------------------------------------------------------------------


class TestPointExpressionParams:
    """Verify that point operations raise TypeError for pl.Expr arguments."""

    # --- normalize ---

    def test_normalize_rejects_expr_ref_width(self) -> None:
        with pytest.raises(TypeError, match="pl.Expr"):
            pl.col("p").point.normalize(ref_width=pl.col("w"), ref_height=100)

    def test_normalize_rejects_expr_ref_height(self) -> None:
        with pytest.raises(TypeError, match="pl.Expr"):
            pl.col("p").point.normalize(ref_width=100, ref_height=pl.col("h"))

    def test_normalize_accepts_literals(self) -> None:
        expr = pl.col("p").point.normalize(ref_width=640, ref_height=480)
        assert expr is not None

    # --- to_absolute ---

    def test_to_absolute_rejects_expr_ref_width(self) -> None:
        with pytest.raises(TypeError, match="pl.Expr"):
            pl.col("p").point.to_absolute(ref_width=pl.col("w"), ref_height=100)

    def test_to_absolute_rejects_expr_ref_height(self) -> None:
        with pytest.raises(TypeError, match="pl.Expr"):
            pl.col("p").point.to_absolute(ref_width=100, ref_height=pl.col("h"))

    def test_to_absolute_accepts_literals(self) -> None:
        expr = pl.col("p").point.to_absolute(ref_width=640, ref_height=480)
        assert expr is not None

    # --- translate ---

    def test_translate_rejects_expr_dx(self) -> None:
        with pytest.raises(TypeError, match="pl.Expr"):
            pl.col("p").point.translate(dx=pl.col("dx"), dy=10.0)

    def test_translate_rejects_expr_dy(self) -> None:
        with pytest.raises(TypeError, match="pl.Expr"):
            pl.col("p").point.translate(dx=10.0, dy=pl.col("dy"))

    def test_translate_accepts_literals(self) -> None:
        expr = pl.col("p").point.translate(dx=5.0, dy=-3.0)
        assert expr is not None

    # --- scale ---

    def test_scale_rejects_expr_sx(self) -> None:
        with pytest.raises(TypeError, match="pl.Expr"):
            pl.col("p").point.scale(sx=pl.col("s"), sy=2.0)

    def test_scale_rejects_expr_sy(self) -> None:
        with pytest.raises(TypeError, match="pl.Expr"):
            pl.col("p").point.scale(sx=2.0, sy=pl.col("s"))

    def test_scale_accepts_literals(self) -> None:
        expr = pl.col("p").point.scale(sx=0.5, sy=1.5)
        assert expr is not None
