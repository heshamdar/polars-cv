"""
Feature extraction composition tests for cancer detection use case.

Demonstrates how existing polars-cv primitives compose to compute
diagnostic features: IoU, region proportion, masked pixel statistics,
percentiles, and annotation counts.
"""

import numpy as np
import polars as pl
import pytest
from io import BytesIO
from PIL import Image

from polars_cv.pipeline import Pipeline
from polars_cv.geometry.schemas import CONTOUR_SCHEMA


def encode_png(arr: np.ndarray) -> bytes:
    """Encode numpy array as PNG bytes."""
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_contour(points: list[tuple[float, float]]) -> dict:
    """Create a contour struct compatible with polars-cv."""
    return {
        "exterior": [{"x": x, "y": y} for x, y in points],
        "holes": [],
    }


@pytest.fixture
def synthetic_data():
    """
    Synthetic cancer detection scenario:
    - 100x100 grayscale image
    - GT contour: rectangle covering rows 20-60, cols 30-70
    - Predicted heatmap: higher scores inside GT region
    """
    np.random.seed(42)

    # Base image (tissue-like)
    image = np.random.randint(50, 200, (100, 100), dtype=np.uint8)

    # GT annotation contour (rectangle)
    gt_points = [(30.0, 20.0), (70.0, 20.0), (70.0, 60.0), (30.0, 60.0)]
    gt_contour = make_contour(gt_points)

    # Predicted heatmap (float-like, stored as u8 0-255 representing 0-1)
    heatmap = np.zeros((100, 100), dtype=np.uint8)
    # High prediction inside the lesion area
    heatmap[18:62, 28:72] = 180  # ~0.7 probability
    # Some noise outside
    heatmap[0:10, 0:10] = 50

    return {
        "image_bytes": encode_png(image),
        "heatmap_bytes": encode_png(heatmap),
        "gt_contour": gt_contour,
        "width": 100,
        "height": 100,
        "gt_points": gt_points,
        "image_array": image,
        "heatmap_array": heatmap,
    }


class TestContourIoU:
    """Test IoU/Dice between predicted and annotated contours."""

    def test_contour_iou_identical(self):
        """IoU of a contour with itself should be 1.0."""
        points = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
        contour = make_contour(points)
        df = pl.DataFrame(
            {"gt": [contour], "pred": [contour]},
            schema={
                "gt": CONTOUR_SCHEMA,
                "pred": CONTOUR_SCHEMA,
            },
        )
        result = df.select(iou=pl.col("gt").contour.iou(pl.col("pred")))
        assert abs(result["iou"][0] - 1.0) < 1e-6

    def test_contour_dice_identical(self):
        """Dice of a contour with itself should be 1.0."""
        points = [(10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)]
        contour = make_contour(points)
        df = pl.DataFrame(
            {"gt": [contour], "pred": [contour]},
            schema={
                "gt": CONTOUR_SCHEMA,
                "pred": CONTOUR_SCHEMA,
            },
        )
        result = df.select(dice=pl.col("gt").contour.dice(pl.col("pred")))
        assert abs(result["dice"][0] - 1.0) < 1e-6

    def test_contour_iou_partial_overlap(self):
        """IoU with partial overlap should be between 0 and 1."""
        gt = make_contour([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)])
        pred = make_contour([(25.0, 0.0), (75.0, 0.0), (75.0, 50.0), (25.0, 50.0)])
        df = pl.DataFrame(
            {"gt": [gt], "pred": [pred]},
            schema={"gt": CONTOUR_SCHEMA, "pred": CONTOUR_SCHEMA},
        )
        result = df.select(iou=pl.col("gt").contour.iou(pl.col("pred")))
        iou = result["iou"][0]
        # Intersection = 25*50 = 1250, Union = 50*50 + 50*50 - 1250 = 3750
        expected = 1250.0 / 3750.0
        assert abs(iou - expected) < 0.02  # Allow small geometric error


class TestRegionProportion:
    """Test computing what fraction of the image a contour occupies."""

    def test_region_proportion(self):
        """Area of contour / total image area gives region proportion."""
        # 40x40 rectangle in a 100x100 image = 16%
        contour = make_contour([(30.0, 20.0), (70.0, 20.0), (70.0, 60.0), (30.0, 60.0)])
        df = pl.DataFrame(
            {"contour": [contour], "img_w": [100], "img_h": [100]},
            schema={
                "contour": CONTOUR_SCHEMA,
                "img_w": pl.Int64,
                "img_h": pl.Int64,
            },
        )
        result = df.select(
            proportion=pl.col("contour").contour.area()
            / (pl.col("img_w") * pl.col("img_h"))
        )
        proportion = result["proportion"][0]
        expected = (40 * 40) / (100 * 100)  # 0.16
        assert abs(proportion - expected) < 0.01


class TestMaskedPixelStatistics:
    """Test computing pixel statistics within a masked region."""

    def test_masked_mean_heatmap_score(self, synthetic_data):
        """Mean heatmap value within the GT lesion region."""
        df = pl.DataFrame(
            {
                "heatmap": [synthetic_data["heatmap_bytes"]],
                "gt_contour": [synthetic_data["gt_contour"]],
                "width": [synthetic_data["width"]],
                "height": [synthetic_data["height"]],
            },
            schema={
                "heatmap": pl.Binary,
                "gt_contour": CONTOUR_SCHEMA,
                "width": pl.Int64,
                "height": pl.Int64,
            },
        )

        # Build pipelines: rasterize contour to mask, apply to heatmap, reduce
        mask_pipe = Pipeline().source("contour", width=100, height=100)
        heatmap_pipe = Pipeline().source("image_bytes").grayscale()

        heatmap_node = pl.col("heatmap").cv.pipe(heatmap_pipe)
        mask_node = pl.col("gt_contour").cv.pipe(mask_pipe)
        masked = heatmap_node.apply_mask(mask_node)

        result = df.select(
            mean_score=masked.pipe(Pipeline().reduce_mean()).sink("native"),
        )

        mean_score = result["mean_score"][0]
        # The mean should be > 0 (there are non-zero pixels inside the mask)
        assert mean_score > 0
        # And less than 255
        assert mean_score < 255

    def test_masked_percentile(self, synthetic_data):
        """P95 heatmap value within the GT lesion region."""
        df = pl.DataFrame(
            {
                "heatmap": [synthetic_data["heatmap_bytes"]],
                "gt_contour": [synthetic_data["gt_contour"]],
                "width": [synthetic_data["width"]],
                "height": [synthetic_data["height"]],
            },
            schema={
                "heatmap": pl.Binary,
                "gt_contour": CONTOUR_SCHEMA,
                "width": pl.Int64,
                "height": pl.Int64,
            },
        )

        mask_pipe = Pipeline().source("contour", width=100, height=100)
        heatmap_pipe = Pipeline().source("image_bytes").grayscale()

        heatmap_node = pl.col("heatmap").cv.pipe(heatmap_pipe)
        mask_node = pl.col("gt_contour").cv.pipe(mask_pipe)
        masked = heatmap_node.apply_mask(mask_node)

        result = df.select(
            p95=masked.pipe(Pipeline().reduce_percentile(95.0)).sink("native"),
            p50=masked.pipe(Pipeline().reduce_percentile(50.0)).sink("native"),
            p5=masked.pipe(Pipeline().reduce_percentile(5.0)).sink("native"),
        )

        # P95 >= P50 >= P5
        assert result["p95"][0] >= result["p50"][0]
        assert result["p50"][0] >= result["p5"][0]


class TestPercentileReduction:
    """Test the reduce_percentile primitive directly."""

    def test_percentile_matches_numpy(self):
        """Percentile values should match numpy.percentile."""
        data = np.arange(1, 101, dtype=np.uint8).reshape(10, 10)
        png = encode_png(data)
        df = pl.DataFrame({"image": [png]})
        pipe_base = Pipeline().source("image_bytes").grayscale()

        for q in [0, 25, 50, 75, 100]:
            pipe = pipe_base.reduce_percentile(float(q))
            result = df.select(val=pl.col("image").cv.pipe(pipe).sink("native"))
            expected = np.percentile(data.flatten(), q)
            assert abs(result["val"][0] - expected) < 1e-6, (
                f"q={q}: got {result['val'][0]}, expected {expected}"
            )

    def test_percentile_0_equals_min(self):
        """P0 should equal reduce_min."""
        data = np.array([[5, 10, 15], [20, 25, 30]], dtype=np.uint8)
        png = encode_png(data)
        df = pl.DataFrame({"image": [png]})
        pipe_base = Pipeline().source("image_bytes").grayscale()

        result = df.select(
            p0=pl.col("image").cv.pipe(pipe_base.reduce_percentile(0.0)).sink("native"),
            min_val=pl.col("image").cv.pipe(pipe_base.reduce_min()).sink("native"),
        )
        assert result["p0"][0] == result["min_val"][0]

    def test_percentile_100_equals_max(self):
        """P100 should equal reduce_max."""
        data = np.array([[5, 10, 15], [20, 25, 30]], dtype=np.uint8)
        png = encode_png(data)
        df = pl.DataFrame({"image": [png]})
        pipe_base = Pipeline().source("image_bytes").grayscale()

        result = df.select(
            p100=pl.col("image")
            .cv.pipe(pipe_base.reduce_percentile(100.0))
            .sink("native"),
            max_val=pl.col("image").cv.pipe(pipe_base.reduce_max()).sink("native"),
        )
        assert result["p100"][0] == result["max_val"][0]


class TestAnnotationCounts:
    """Test counting annotations per image using pure Polars."""

    def test_count_per_image(self):
        """Count annotations per image via group_by."""
        c1 = make_contour([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)])
        c2 = make_contour([(20.0, 20.0), (30.0, 20.0), (30.0, 30.0)])
        c3 = make_contour([(50.0, 50.0), (60.0, 50.0), (60.0, 60.0)])

        df = pl.DataFrame(
            {
                "image_id": ["img1", "img1", "img2"],
                "contour": [c1, c2, c3],
            },
            schema={
                "image_id": pl.String,
                "contour": CONTOUR_SCHEMA,
            },
        )

        counts = (
            df.group_by("image_id")
            .agg(
                n_annotations=pl.col("contour").count(),
                total_area=pl.col("contour").contour.area().sum(),
            )
            .sort("image_id")
        )

        assert counts["n_annotations"].to_list() == [2, 1]
        assert all(a > 0 for a in counts["total_area"].to_list())


class TestGlobalPixelStatistics:
    """Test global pixel statistics (not masked)."""

    def test_multi_stat_extraction(self):
        """Extract multiple statistics from a single image in one query."""
        data = np.arange(0, 100, dtype=np.uint8).reshape(10, 10)
        png = encode_png(data)
        df = pl.DataFrame({"image": [png]})
        pipe_base = Pipeline().source("image_bytes").grayscale()

        result = df.select(
            mean=pl.col("image").cv.pipe(pipe_base.reduce_mean()).sink("native"),
            std=pl.col("image").cv.pipe(pipe_base.reduce_std()).sink("native"),
            min_val=pl.col("image").cv.pipe(pipe_base.reduce_min()).sink("native"),
            max_val=pl.col("image").cv.pipe(pipe_base.reduce_max()).sink("native"),
            p25=pl.col("image")
            .cv.pipe(pipe_base.reduce_percentile(25.0))
            .sink("native"),
            p75=pl.col("image")
            .cv.pipe(pipe_base.reduce_percentile(75.0))
            .sink("native"),
        )

        expected = data.flatten().astype(float)
        assert abs(result["mean"][0] - expected.mean()) < 1e-6
        assert abs(result["std"][0] - expected.std()) < 1e-6
        assert result["min_val"][0] == 0.0
        assert result["max_val"][0] == 99.0
        assert abs(result["p25"][0] - np.percentile(expected, 25)) < 1e-6
        assert abs(result["p75"][0] - np.percentile(expected, 75)) < 1e-6
