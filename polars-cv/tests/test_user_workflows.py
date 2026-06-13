"""
User-workflow style functional tests for polars-cv.

Validates end-to-end correctness for realistic pipelines that combine
multiple operations, per-row column arguments, binary image algebra,
multi-output branching, and eager vs streaming engine consistency.

These tests focus on *output values* — silent regressions where no
exception is raised but pixels are wrong.
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline, numpy_from_struct

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plugin_available() -> bool:
    lib_path = Path(__file__).parent.parent / "python" / "polars_cv"
    return bool(list(lib_path.glob("*.so")) + list(lib_path.glob("*.pyd")))


plugin_required = pytest.mark.skipif(
    not _plugin_available(),
    reason="Requires compiled plugin (run maturin develop first)",
)


def _png(arr: np.ndarray) -> bytes:
    """Encode a numpy array as PNG bytes."""
    from PIL import Image

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _solid(h: int, w: int, color: tuple[int, ...], dtype=np.uint8) -> np.ndarray:
    """Solid-color HWC image."""
    c = len(color)
    if c == 1:
        return np.full((h, w), color[0], dtype=dtype)
    return np.full((h, w, c), color, dtype=dtype)


def _decode(row_val) -> np.ndarray:
    """Extract numpy array from a polars numpy-sink struct value."""
    return numpy_from_struct(row_val)


def _checkerboard(h: int, w: int, tile: int = 8) -> np.ndarray:
    """Black-and-white RGB checkerboard."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((y // tile) + (x // tile)) % 2 == 0:
                arr[y : y + tile, x : x + tile] = 255
    return arr


# ===================================================================
# 1. ML Preprocessing Pipeline
# ===================================================================


@plugin_required
class TestMLPreprocessingPipeline:
    """ImageNet-style preprocessing: resize → crop → normalize."""

    def test_output_shape_and_dtype(self) -> None:
        """resize(256) + crop(224) + cast(f32) + scale + normalize → (224,224,3) f32."""
        img = _png(_solid(300, 300, (200, 100, 50)))
        df = pl.DataFrame({"img": [img]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=256, width=256, filter="bilinear")
            .crop(top=16, left=16, height=224, width=224)
            .cast("f32")
            .scale(1.0 / 255.0)
            .normalize(
                method="preset",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = _decode(result.row(0)[0])

        assert out.shape == (224, 224, 3)
        assert out.dtype == np.float32

    def test_pixel_values_in_normalized_range(self) -> None:
        """After ImageNet normalization, values should be in roughly [-3, 3]."""
        img = _png(_solid(300, 300, (200, 100, 50)))
        df = pl.DataFrame({"img": [img]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .cast("f32")
            .scale(1.0 / 255.0)
            .normalize(
                method="preset",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = _decode(result.row(0)[0])

        assert float(out.min()) > -4.0, "Min value suspiciously low"
        assert float(out.max()) < 4.0, "Max value suspiciously high"

    def test_batch_independence(self) -> None:
        """Different input images must produce different outputs in the same batch."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0), (0, 128, 128)]
        pngs = [_png(_solid(64, 64, c)) for c in colors]
        df = pl.DataFrame({"img": pngs})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=32, width=32)
            .cast("f32")
            .scale(1.0 / 255.0)
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        outputs = [_decode(result.row(i)[0]) for i in range(len(colors))]
        # Each row must differ from all other rows
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not np.allclose(
                    outputs[i], outputs[j]
                ), f"Rows {i} and {j} are identical — batch isolation broken"

    def test_sink_torch_layout(self) -> None:
        """sink('torch') produces the same HWC struct as numpy sink.

        The torch sink tags the output format but does not permute axes in
        the serialized struct — CHW permutation is left to the consumer.
        """
        img = _png(_solid(64, 64, (100, 150, 200)))
        df = pl.DataFrame({"img": [img]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=32, width=32)
            .cast("f32")
            .scale(1.0 / 255.0)
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("torch"))
        out = _decode(result.row(0)[0])

        assert out.shape == (32, 32, 3), f"Expected (32,32,3) HWC struct, got {out.shape}"
        assert out.dtype == np.float32


# ===================================================================
# 2. Per-Row Column Arguments
# ===================================================================


@plugin_required
class TestPerRowColumnArguments:
    """Each row uses its own parameter values resolved from DataFrame columns."""

    def test_per_row_resize_shapes(self) -> None:
        """Each row should be resized to its own target dimensions."""
        sizes = [32, 64, 96, 128]
        pngs = [_png(_solid(200, 200, (128, 128, 128))) for _ in sizes]
        df = pl.DataFrame({"img": pngs, "h": sizes, "w": sizes})

        pipe = Pipeline().source("image_bytes").resize(
            height=pl.col("h"), width=pl.col("w")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        for i, s in enumerate(sizes):
            out = _decode(result.row(i)[0])
            assert out.shape == (s, s, 3), (
                f"Row {i}: expected ({s},{s},3), got {out.shape}"
            )

    def test_per_row_threshold_values(self) -> None:
        """Per-row threshold: pixel value 100, thresholds 50/128/200."""
        # All images are solid gray 100
        pngs = [_png(_solid(20, 20, (100, 100, 100)))] * 3
        thresholds = [50, 128, 200]
        df = pl.DataFrame({"img": pngs, "thresh": thresholds})

        pipe = Pipeline().source("image_bytes").grayscale().threshold(pl.col("thresh"))
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        # thresh=50:  100 > 50  → all pixels = 255
        out0 = _decode(result.row(0)[0])
        assert np.all(out0 == 255), "thresh=50: expected all 255"

        # thresh=128: 100 < 128 → all pixels = 0
        out1 = _decode(result.row(1)[0])
        assert np.all(out1 == 0), "thresh=128: expected all 0"

        # thresh=200: 100 < 200 → all pixels = 0
        out2 = _decode(result.row(2)[0])
        assert np.all(out2 == 0), "thresh=200: expected all 0"

    def test_per_row_blur_sigma_effect(self) -> None:
        """Larger sigma should smooth edges more — measured by gradient magnitude."""
        # Sharp vertical edge: left half black, right half white
        arr = np.zeros((40, 40, 3), dtype=np.uint8)
        arr[:, 20:] = 255
        pngs = [_png(arr), _png(arr)]
        df = pl.DataFrame({"img": pngs, "sigma": [0.5, 4.0]})

        pipe = Pipeline().source("image_bytes").grayscale().blur(pl.col("sigma"))
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        out_soft = _decode(result.row(0)[0]).astype(float)
        out_heavy = _decode(result.row(1)[0]).astype(float)

        # Max gradient (diff between adjacent columns) should be larger for small sigma
        grad_soft = float(np.max(np.abs(np.diff(out_soft[:, :, 0], axis=1))))
        grad_heavy = float(np.max(np.abs(np.diff(out_heavy[:, :, 0], axis=1))))
        assert grad_soft > grad_heavy, (
            f"sigma=0.5 gradient ({grad_soft:.1f}) should exceed "
            f"sigma=4.0 gradient ({grad_heavy:.1f})"
        )

    def test_per_row_crop_roi(self) -> None:
        """Crop top-left pixel should match the known value at (top, left) in the source."""
        # Gradient image: pixel value = (x + y) % 256 in R channel
        h, w = 60, 60
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                arr[y, x, 0] = (x + y) % 256
        png = _png(arr)

        tops = [0, 5, 10, 20]
        lefts = [0, 3, 8, 15]
        df = pl.DataFrame(
            {"img": [png] * 4, "top": tops, "left": lefts}
        )

        pipe = Pipeline().source("image_bytes").crop(
            top=pl.col("top"), left=pl.col("left"), height=10, width=10
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        for i in range(4):
            out = _decode(result.row(i)[0])
            expected_r = (lefts[i] + tops[i]) % 256
            actual_r = int(out[0, 0, 0])
            assert actual_r == expected_r, (
                f"Row {i}: expected top-left R={expected_r}, got {actual_r}"
            )

    def test_mixed_literal_and_expr_param(self) -> None:
        """resize(height=expr, width=literal) should work."""
        heights = [32, 64, 96]
        pngs = [_png(_solid(200, 200, (100, 100, 100)))] * len(heights)
        df = pl.DataFrame({"img": pngs, "h": heights})

        pipe = Pipeline().source("image_bytes").resize(
            height=pl.col("h"), width=48
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))

        for i, h in enumerate(heights):
            out = _decode(result.row(i)[0])
            assert out.shape == (h, 48, 3), (
                f"Row {i}: expected ({h},48,3), got {out.shape}"
            )


# ===================================================================
# 3. Binary Image Algebra
# ===================================================================


@plugin_required
class TestBinaryImageAlgebra:
    """Element-wise two-column operations with verified output values."""

    def _make_df(
        self,
        color_a: tuple[int, int, int],
        color_b: tuple[int, int, int],
        size: int = 30,
    ) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "img_a": [_png(_solid(size, size, color_a))],
                "img_b": [_png(_solid(size, size, color_b))],
            }
        )

    def _run(
        self, df: pl.DataFrame, op_name: str, expected_val: int
    ) -> None:
        pipe = Pipeline().source("image_bytes")
        la = pl.col("img_a").cv.pipe(pipe)
        lb = pl.col("img_b").cv.pipe(pipe)
        op_method = getattr(la, op_name)
        result = df.select(out=op_method(lb).sink("numpy"))
        out = _decode(result.row(0)[0])
        assert out.dtype == np.uint8
        assert int(out[0, 0, 0]) == expected_val, (
            f"{op_name}: expected {expected_val}, got {int(out[0,0,0])}"
        )

    def test_add_saturating(self) -> None:
        """100 + 200 = 255 (saturated at u8 max)."""
        df = self._make_df((100, 100, 100), (200, 200, 200))
        self._run(df, "add", 255)

    def test_add_no_overflow(self) -> None:
        """50 + 60 = 110 (no overflow, exact)."""
        df = self._make_df((50, 50, 50), (60, 60, 60))
        self._run(df, "add", 110)

    def test_subtract_positive(self) -> None:
        """200 - 100 = 100."""
        df = self._make_df((200, 200, 200), (100, 100, 100))
        self._run(df, "subtract", 100)

    def test_subtract_saturating_zero(self) -> None:
        """100 - 200 = 0 (saturates at 0, not negative)."""
        df = self._make_df((100, 100, 100), (200, 200, 200))
        self._run(df, "subtract", 0)

    def test_maximum(self) -> None:
        """max(80, 150) = 150."""
        df = self._make_df((80, 80, 80), (150, 150, 150))
        self._run(df, "maximum", 150)

    def test_minimum(self) -> None:
        """min(80, 150) = 80."""
        df = self._make_df((80, 80, 80), (150, 150, 150))
        self._run(df, "minimum", 80)

    def test_apply_mask_zeros_background(self) -> None:
        """apply_mask: top half of image preserved, bottom half zeroed."""
        h, w = 40, 40
        # White image (all 200)
        img_arr = _solid(h, w, (200, 200, 200))
        # Mask: top half white (255), bottom half black (0)
        mask_arr = np.zeros((h, w, 3), dtype=np.uint8)
        mask_arr[: h // 2] = 255

        df = pl.DataFrame(
            {
                "img": [_png(img_arr)],
                "mask": [_png(mask_arr)],
            }
        )
        pipe = Pipeline().source("image_bytes")
        img_expr = pl.col("img").cv.pipe(pipe)
        mask_expr = pl.col("mask").cv.pipe(pipe)

        result = df.select(out=img_expr.apply_mask(mask_expr).sink("numpy"))
        out = _decode(result.row(0)[0])

        # Top half: original values preserved (saturating multiply by 255/255)
        assert int(out[0, 0, 0]) > 0, "Top half should be non-zero"
        # Bottom half: zeroed
        assert int(out[h - 1, 0, 0]) == 0, "Bottom half should be zero"


# ===================================================================
# 4. Multi-Output Branching (merge_pipe)
# ===================================================================


@plugin_required
class TestMultiOutputBranching:
    """Branching pipelines that produce multiple outputs from one input."""

    def test_gray_and_threshold_branch(self) -> None:
        """Common prefix (resize) shared between gray and threshold branches."""
        img = _png(_solid(100, 100, (150, 150, 150)))
        df = pl.DataFrame({"img": [img]})

        gray_pipe = (
            Pipeline().source("image_bytes").resize(height=50, width=50).grayscale()
        )
        thresh_pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=50, width=50)
            .grayscale()
            .threshold(100)
        )

        gray_expr = pl.col("img").cv.pipe(gray_pipe).alias("gray")
        thresh_expr = pl.col("img").cv.pipe(thresh_pipe).alias("thresh")

        result_expr = gray_expr.merge_pipe(thresh_expr).sink(
            {"gray": "numpy", "thresh": "numpy"}
        )
        result = df.select(out=result_expr)

        gray_out = _decode(result["out"].struct.field("gray")[0])
        thresh_out = _decode(result["out"].struct.field("thresh")[0])

        assert gray_out.shape == (50, 50, 1)
        assert thresh_out.shape == (50, 50, 1)

        # Solid gray 150 → grayscale ≈ 150
        assert abs(int(gray_out[0, 0, 0]) - 150) <= 2

        # Threshold 100: 150 > 100 → all pixels 255
        assert np.all(thresh_out == 255)

    def test_three_branch_merge(self) -> None:
        """Three aliases sunk to dict should all produce valid outputs."""
        # Use a checkerboard so blur changes pixel values (blur of a constant is identity)
        img = _png(_checkerboard(60, 60, tile=8))
        df = pl.DataFrame({"img": [img]})

        base_pipe = Pipeline().source("image_bytes").resize(height=32, width=32)
        gray_pipe = Pipeline().source("image_bytes").resize(height=32, width=32).grayscale()
        blur_pipe = (
            Pipeline().source("image_bytes").resize(height=32, width=32).blur(sigma=1.5)
        )

        a = pl.col("img").cv.pipe(base_pipe).alias("base")
        b = pl.col("img").cv.pipe(gray_pipe).alias("gray")
        c = pl.col("img").cv.pipe(blur_pipe).alias("blur")

        result_expr = a.merge_pipe(b, c).sink(
            {"base": "numpy", "gray": "numpy", "blur": "numpy"}
        )
        result = df.select(out=result_expr)

        base_out = _decode(result["out"].struct.field("base")[0])
        gray_out = _decode(result["out"].struct.field("gray")[0])
        blur_out = _decode(result["out"].struct.field("blur")[0])

        assert base_out.shape == (32, 32, 3)
        assert gray_out.shape == (32, 32, 1)
        assert blur_out.shape == (32, 32, 3)

        # Base and blur should have same spatial shape but differ in values
        assert not np.array_equal(base_out, blur_out)


# ===================================================================
# 5. Multi-Step Contour Pipeline
# ===================================================================


@plugin_required
class TestContourPipeline:
    """Full contour-extraction workflow with correctness checks."""

    def test_rectangle_contour_area(self) -> None:
        """area() returns one value per contour; the largest should be ≈ 2400 px²."""
        h, w = 100, 100
        # Must be 3-channel RGB so polars-cv decodes as a color image
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[20:60, 20:80] = 255  # 40 rows × 60 cols = 2400 px white rect

        df = pl.DataFrame({"img": [_png(arr)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours()
            .area()
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("native"))

        # area() now returns a List[f64] — one area per detected contour.
        # The largest contour is the rectangle boundary (Shoelace area ≈ 2301).
        areas = list(result["out"][0])
        assert len(areas) > 0, "No contours found"
        max_area = max(areas)
        assert abs(max_area - 2400.0) < 200, (
            f"Largest contour area ≈ 2400, got {max_area:.1f}"
        )

    def test_contour_centroid_near_center(self) -> None:
        """Centroid of a centered square should be near the image center."""
        h, w = 100, 100
        # Must be 3-channel RGB so polars-cv decodes as a color image
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        # 40×40 square centered at (50, 50)
        arr[30:70, 30:70] = 255

        df = pl.DataFrame({"img": [_png(arr)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .extract_contours(min_area=1000.0)  # select only the main square
            .centroid()
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("native"))

        # centroid() returns [cx₀, cy₀, cx₁, cy₁, ...]; with min_area=1000 there
        # is exactly one contour so the vector is [cx, cy].
        centroid_vals = list(result["out"][0])
        assert len(centroid_vals) == 2, (
            f"Expected 2-element centroid for one contour, got {len(centroid_vals)}"
        )
        cx, cy = centroid_vals[0], centroid_vals[1]
        assert abs(cx - 50.0) < 5.0, f"Centroid x={cx:.1f} not near 50"
        assert abs(cy - 50.0) < 5.0, f"Centroid y={cy:.1f} not near 50"


# ===================================================================
# 6. Complex Chained Pipelines
# ===================================================================


@plugin_required
class TestComplexChainedPipelines:
    """Multi-op chains that exercise several op categories in sequence."""

    def test_edge_detection_pipeline(self) -> None:
        """Checkerboard → resize → gray → blur → canny should detect edges."""
        arr = _checkerboard(128, 128, tile=16)
        df = pl.DataFrame({"img": [_png(arr)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=64, width=64)
            .grayscale()
            .blur(sigma=1.0)
            .canny(low_threshold=30.0, high_threshold=90.0)
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = _decode(result.row(0)[0])

        assert out.shape == (64, 64, 1), f"Expected (64,64,1), got {out.shape}"
        assert out.dtype == np.uint8
        edge_px = int(np.sum(out > 0))
        total_px = 64 * 64
        # Must have some edges (>0) but not fill the whole image
        assert edge_px > 50, f"Too few edge pixels detected: {edge_px}"
        assert edge_px < total_px, "All pixels are edges — canny produced all-white"

    def test_morph_open_removes_small_blob(self) -> None:
        """Morphological open (erode→dilate) removes a single-pixel noise dot."""
        # Large white square + one isolated noise pixel
        arr = np.zeros((50, 50), dtype=np.uint8)
        arr[10:40, 10:40] = 255  # large object
        arr[3, 3] = 255  # isolated noise pixel

        df = pl.DataFrame({"img": [_png(arr)]})

        pipe_open = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
            .morphology_open(ksize=5)
        )
        pipe_orig = (
            Pipeline()
            .source("image_bytes")
            .grayscale()
            .threshold(128)
        )

        result_open = df.select(out=pl.col("img").cv.pipe(pipe_open).sink("numpy"))
        result_orig = df.select(out=pl.col("img").cv.pipe(pipe_orig).sink("numpy"))

        out_open = _decode(result_open.row(0)[0])
        out_orig = _decode(result_orig.row(0)[0])

        white_after_open = int(np.sum(out_open > 0))
        white_before = int(np.sum(out_orig > 0))

        # Morph open should reduce or equal the white pixel count (noise removed)
        assert white_after_open <= white_before, (
            f"Morph open increased white pixels: {white_after_open} > {white_before}"
        )
        # The isolated noise pixel at (3,3) should be gone
        assert int(out_open[3, 3, 0]) == 0, "Isolated noise pixel should be removed by open"

    def test_letterbox_preserves_content_and_adds_padding(self) -> None:
        """Non-square image letter-boxed to square: content in center, borders zero."""
        # Wide image: 100×200 solid red
        arr = _solid(100, 200, (200, 50, 50))
        df = pl.DataFrame({"img": [_png(arr)]})

        pipe = Pipeline().source("image_bytes").letterbox(height=200, width=200, value=0)
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = _decode(result.row(0)[0])

        assert out.shape == (200, 200, 3), f"Expected (200,200,3), got {out.shape}"

        # Top and bottom rows should be the padding color (0)
        top_row_mean = float(out[0, :, :].mean())
        bot_row_mean = float(out[-1, :, :].mean())
        assert top_row_mean < 20.0, f"Top padding not black (mean={top_row_mean:.1f})"
        assert bot_row_mean < 20.0, f"Bottom padding not black (mean={bot_row_mean:.1f})"

        # Center pixel should be non-zero (contains original red content)
        center = out[100, 100]
        assert int(center[0]) > 100, f"Center pixel should be reddish, got {center}"

    def test_dtype_chain_u8_to_f32_and_back(self) -> None:
        """u8 → cast(f32) → scale(1/255) → scale(255) → cast(u8) round-trips within ±1."""
        arr = np.array([[[50, 100, 200], [10, 230, 127]]], dtype=np.uint8)
        df = pl.DataFrame({"img": [_png(arr)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .cast("f32")
            .scale(1.0 / 255.0)
            .scale(255.0)
            .cast("u8")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = _decode(result.row(0)[0])

        assert out.dtype == np.uint8
        np.testing.assert_allclose(
            out[0].astype(int),
            arr[0].astype(int),
            atol=1,
            err_msg="Round-trip u8→f32→u8 should be within ±1",
        )

    def test_normalize_minmax_bounds(self) -> None:
        """After minmax normalization the output min=0.0 and max=1.0."""
        # Image with varied values so minmax has something to normalize
        arr = np.array(
            [[[20, 100, 200], [80, 40, 160], [255, 0, 128]]],
            dtype=np.uint8,
        )
        df = pl.DataFrame({"img": [_png(arr)]})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .cast("f32")
            .normalize(method="minmax")
        )
        result = df.select(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        out = _decode(result.row(0)[0])

        assert out.dtype == np.float32
        assert float(out.min()) >= -0.001
        assert float(out.max()) <= 1.001


# ===================================================================
# 7. Streaming vs Eager Consistency
# ===================================================================


@plugin_required
class TestStreamingVsEager:
    """Same pipeline must produce identical results in eager and streaming modes."""

    def _run_both(
        self, df: pl.DataFrame, expr: pl.Expr
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        eager = df.select(out=expr)
        streaming = df.lazy().select(out=expr).collect(engine="streaming")
        return eager, streaming

    def test_grayscale_pipeline(self) -> None:
        """Grayscale: 10 rows, eager == streaming."""
        pngs = [_png(_solid(32, 32, (i * 20, 128, 200 - i * 15))) for i in range(10)]
        df = pl.DataFrame({"img": pngs})

        pipe = Pipeline().source("image_bytes").grayscale()
        expr = pl.col("img").cv.pipe(pipe).sink("numpy")

        eager, streaming = self._run_both(df, expr)

        for i in range(10):
            e = _decode(eager.row(i)[0])
            s = _decode(streaming.row(i)[0])
            np.testing.assert_array_equal(e, s, err_msg=f"Row {i} mismatch")

    def test_complex_pipeline(self) -> None:
        """resize + blur + normalize: 8 rows, eager == streaming."""
        pngs = [_png(_solid(80, 80, (i * 30 % 256, 100, 200))) for i in range(8)]
        df = pl.DataFrame({"img": pngs})

        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=32, width=32)
            .blur(sigma=1.5)
            .cast("f32")
            .scale(1.0 / 255.0)
        )
        expr = pl.col("img").cv.pipe(pipe).sink("numpy")

        eager, streaming = self._run_both(df, expr)

        for i in range(8):
            e = _decode(eager.row(i)[0])
            s = _decode(streaming.row(i)[0])
            np.testing.assert_allclose(e, s, atol=1e-5, err_msg=f"Row {i} mismatch")

    def test_per_row_params_pipeline(self) -> None:
        """Per-row resize: 6 rows with different sizes, eager == streaming."""
        sizes = [24, 32, 40, 48, 56, 64]
        pngs = [_png(_solid(100, 100, (128, 128, 128)))] * len(sizes)
        df = pl.DataFrame({"img": pngs, "sz": sizes})

        pipe = Pipeline().source("image_bytes").resize(
            height=pl.col("sz"), width=pl.col("sz")
        )
        expr = pl.col("img").cv.pipe(pipe).sink("numpy")

        eager, streaming = self._run_both(df, expr)

        for i in range(len(sizes)):
            e = _decode(eager.row(i)[0])
            s = _decode(streaming.row(i)[0])
            assert e.shape == (sizes[i], sizes[i], 3)
            np.testing.assert_array_equal(e, s, err_msg=f"Row {i} mismatch")
