"""
Tests for binary mask metrics (IoU, Dice), bitwise operations, and reduce_sum.

These tests verify that:
1. reduce_sum() correctly reduces an array to a scalar
2. Bitwise operations (AND, OR, XOR) work correctly on binary masks
3. mask_iou() and mask_dice() compute correct metrics
4. Edge cases (empty masks, full overlap, no overlap) are handled
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, mask_dice, mask_iou

if TYPE_CHECKING:
    pass


def create_mask_bytes(mask_array: list[list[int]]) -> bytes:
    """
    Create PNG bytes from a 2D mask array.

    Args:
        mask_array: 2D list of pixel values (0 or 255).

    Returns:
        PNG bytes representing the mask.
    """
    import numpy as np

    arr = np.array(mask_array, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestReduceSum:
    """Test reduce_sum operation."""

    def test_reduce_sum_basic(self) -> None:
        """Test basic reduce_sum on a simple image."""
        # Create a 4x4 image with known sum
        # 4 pixels at 255 = 1020
        mask = [
            [255, 0, 0, 255],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [255, 0, 0, 255],
        ]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_sum()
        result = df.select(total=pl.col("mask").cv.pipe(pipe).sink("native"))

        # 4 corners * 255 = 1020
        total = result.row(0)[0]
        assert total == 1020.0, f"Expected 1020.0, got {total}"

    def test_reduce_sum_all_zeros(self) -> None:
        """Test reduce_sum on an all-zero image."""
        mask = [[0] * 10 for _ in range(10)]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_sum()
        result = df.select(total=pl.col("mask").cv.pipe(pipe).sink("native"))

        total = result.row(0)[0]
        assert total == 0.0, f"Expected 0.0, got {total}"

    def test_reduce_sum_all_ones(self) -> None:
        """Test reduce_sum on an all-255 image."""
        mask = [[255] * 10 for _ in range(10)]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_sum()
        result = df.select(total=pl.col("mask").cv.pipe(pipe).sink("native"))

        # 100 pixels * 255 = 25500
        total = result.row(0)[0]
        assert total == 25500.0, f"Expected 25500.0, got {total}"


def get_innermost_dtype(dtype: pl.DataType) -> pl.DataType:
    """Extract the innermost dtype from a nested List/Array type."""
    current = dtype
    while hasattr(current, "inner") and current.inner is not None:
        current = current.inner
    return current


class TestListSink:
    """Test that list sink returns proper Polars List types.

    Note: List sink now preserves shape as nested lists. A 2x2x1 grayscale
    image produces List[List[List[UInt8]]] to preserve the [H, W, C] shape.
    """

    def test_list_sink_returns_polars_list(self) -> None:
        """Test that list sink returns a Polars nested List preserving shape."""
        mask = [[255, 0], [0, 255]]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.select(values=pl.col("mask").cv.pipe(pipe).sink("list"))

        # Grayscale produces shape [2, 2, 1] -> nested List[List[List[UInt8]]]
        expected_dtype = pl.List(pl.List(pl.List(pl.UInt8)))
        assert result["values"].dtype == expected_dtype, (
            f"Expected {expected_dtype}, got {result['values'].dtype}"
        )

        # The innermost type should be UInt8
        assert get_innermost_dtype(result["values"].dtype) == pl.UInt8

        # The list should have nested structure matching [2, 2, 1]
        values = result["values"][0].to_list()
        assert isinstance(values, list), f"Expected list, got {type(values)}"
        assert len(values) == 2, f"Expected 2 rows (height), got {len(values)}"
        assert len(values[0]) == 2, f"Expected 2 cols (width), got {len(values[0])}"
        assert len(values[0][0]) == 1, f"Expected 1 channel, got {len(values[0][0])}"

    def test_list_sink_values_correct(self) -> None:
        """Test that list sink returns correct values preserving shape."""
        # 2x2 image with known values
        mask = [[100, 200], [50, 150]]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        pipe = Pipeline().source("image_bytes").grayscale()
        result = df.select(values=pl.col("mask").cv.pipe(pipe).sink("list"))

        values = result["values"][0].to_list()
        # Values should match the original mask with shape [2, 2, 1]
        # Each pixel is wrapped in a list due to the channel dimension
        expected = [[[100], [200]], [[50], [150]]]
        assert values == expected, f"Expected {expected}, got {values}"


class TestArraySink:
    """Test that array sink returns proper Polars Array types."""

    def test_array_sink_returns_polars_array(self) -> None:
        """Test that array sink returns a Polars Array, not List or Binary."""
        mask = [[255, 0], [0, 255]]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        # Grayscale produces shape [2, 2, 1] (H, W, C)
        pipe = (
            Pipeline().source("image_bytes").grayscale().sink("array", shape=[2, 2, 1])
        )
        result = df.select(values=pl.col("mask").cv.pipeline(pipe))

        # The result should be an Array type with nested structure
        assert str(result["values"].dtype).startswith("Array"), (
            f"Expected Array type, got {result['values'].dtype}"
        )

    def test_array_sink_shape_validation_element_count(self) -> None:
        """Test that array sink validates shape matches element count."""
        mask = [[100, 200], [50, 150]]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        # Wrong shape: [3, 3] = 9 elements but image has 4 elements (2x2x1 grayscale)
        pipe = Pipeline().source("image_bytes").grayscale().sink("array", shape=[3, 3])

        # This should fail because element count doesn't match
        with pytest.raises(Exception):
            df.select(values=pl.col("mask").cv.pipeline(pipe))

    def test_array_sink_shape_validation_exact_match(self) -> None:
        """Test that array sink requires exact shape match, not just element count."""
        mask = [[100, 200], [50, 150]]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        # Buffer shape is [2, 2, 1] (grayscale), but we specify [2, 2]
        # Both have 4 elements but different structure
        pipe = Pipeline().source("image_bytes").grayscale().sink("array", shape=[2, 2])

        # This should fail because exact shape doesn't match (use squeeze() first)
        with pytest.raises(Exception):
            df.select(values=pl.col("mask").cv.pipeline(pipe))

    def test_array_sink_values_structure(self) -> None:
        """Test that array sink preserves nested structure."""
        mask = [
            [100, 200, 150, 175],
            [50, 100, 75, 125],
            [25, 50, 37, 62],
            [10, 20, 15, 25],
        ]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame({"mask": [mask_bytes]})

        # Shape [4, 4, 1] for 4x4 grayscale image
        pipe = (
            Pipeline().source("image_bytes").grayscale().sink("array", shape=[4, 4, 1])
        )
        result = df.select(values=pl.col("mask").cv.pipeline(pipe))

        # Check the dtype is correctly nested (grayscale outputs U8)
        expected_dtype = pl.Array(pl.Array(pl.Array(pl.UInt8, 1), 4), 4)
        assert result["values"].dtype == expected_dtype, (
            f"Expected {expected_dtype}, got {result['values'].dtype}"
        )


class TestBitwiseOperations:
    """Test binary mask bitwise operations."""

    def test_bitwise_and_intersection(self) -> None:
        """Test that bitwise_and computes intersection correctly."""
        # Create two overlapping masks (10x10)
        # Mask 1: left half white (255), right half black (0)
        # Mask 2: top half white (255), bottom half black (0)
        # Intersection: top-left quarter should be white

        mask1 = [[255] * 5 + [0] * 5 for _ in range(10)]
        mask2 = [[255] * 10 for _ in range(5)] + [[0] * 10 for _ in range(5)]

        mask1_bytes = create_mask_bytes(mask1)
        mask2_bytes = create_mask_bytes(mask2)

        df = pl.DataFrame(
            {
                "mask1": [mask1_bytes],
                "mask2": [mask2_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        m1 = pl.col("mask1").cv.pipe(pipe)
        m2 = pl.col("mask2").cv.pipe(pipe)

        # Use reduce_sum to verify the intersection
        result = df.select(
            intersection_sum=m1.bitwise_and(m2)
            .pipe(Pipeline().reduce_sum())
            .sink("native")
        )

        # The intersection should have non-zero values only in top-left quarter
        # Top-left quarter: 5*5 = 25 pixels with value 255
        # Expected sum: 25 * 255 = 6375
        total = result.row(0)[0]
        assert total == 25 * 255, f"Expected {25 * 255}, got {total}"

    def test_bitwise_or_union(self) -> None:
        """Test that bitwise_or computes union correctly."""
        # Same masks as above
        # Union: everything except bottom-right quarter should be white

        mask1 = [[255] * 5 + [0] * 5 for _ in range(10)]
        mask2 = [[255] * 10 for _ in range(5)] + [[0] * 10 for _ in range(5)]

        mask1_bytes = create_mask_bytes(mask1)
        mask2_bytes = create_mask_bytes(mask2)

        df = pl.DataFrame(
            {
                "mask1": [mask1_bytes],
                "mask2": [mask2_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        m1 = pl.col("mask1").cv.pipe(pipe)
        m2 = pl.col("mask2").cv.pipe(pipe)

        # Use reduce_sum to verify the union
        result = df.select(
            union_sum=m1.bitwise_or(m2).pipe(Pipeline().reduce_sum()).sink("native")
        )

        # The union should have non-zero values in:
        # - Top-left (5*5), top-right (5*5), bottom-left (5*5) = 75 pixels
        total = result.row(0)[0]
        assert total == 75 * 255, f"Expected {75 * 255}, got {total}"

    def test_bitwise_xor_symmetric_difference(self) -> None:
        """Test that bitwise_xor computes symmetric difference correctly."""
        # Same masks as above
        # XOR: top-right and bottom-left should be white (areas in one but not both)

        mask1 = [[255] * 5 + [0] * 5 for _ in range(10)]
        mask2 = [[255] * 10 for _ in range(5)] + [[0] * 10 for _ in range(5)]

        mask1_bytes = create_mask_bytes(mask1)
        mask2_bytes = create_mask_bytes(mask2)

        df = pl.DataFrame(
            {
                "mask1": [mask1_bytes],
                "mask2": [mask2_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        m1 = pl.col("mask1").cv.pipe(pipe)
        m2 = pl.col("mask2").cv.pipe(pipe)

        # Use reduce_sum to verify the XOR
        result = df.select(
            xor_sum=m1.bitwise_xor(m2).pipe(Pipeline().reduce_sum()).sink("native")
        )

        # The XOR should have non-zero values in:
        # - Top-right (5*5), bottom-left (5*5) = 50 pixels
        total = result.row(0)[0]
        assert total == 50 * 255, f"Expected {50 * 255}, got {total}"


class TestMaskIoU:
    """Test mask IoU computation."""

    def test_iou_perfect_overlap(self) -> None:
        """Test IoU = 1.0 for identical masks."""
        # Create identical masks
        mask = [[255] * 10 for _ in range(10)]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame(
            {
                "pred": [mask_bytes],
                "target": [mask_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        pred = pl.col("pred").cv.pipe(pipe).alias("pred")
        target = pl.col("target").cv.pipe(pipe).alias("target")

        result = df.select(iou=mask_iou(pred, target))

        iou_value = result.row(0)[0]
        assert abs(iou_value - 1.0) < 0.001, f"Expected IoU ~1.0, got {iou_value}"

    def test_iou_no_overlap(self) -> None:
        """Test IoU ≈ 0 for non-overlapping masks."""
        # Create non-overlapping masks
        mask1 = [[255] * 5 + [0] * 5 for _ in range(10)]  # Left half
        mask2 = [[0] * 5 + [255] * 5 for _ in range(10)]  # Right half

        mask1_bytes = create_mask_bytes(mask1)
        mask2_bytes = create_mask_bytes(mask2)

        df = pl.DataFrame(
            {
                "pred": [mask1_bytes],
                "target": [mask2_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        pred = pl.col("pred").cv.pipe(pipe).alias("pred")
        target = pl.col("target").cv.pipe(pipe).alias("target")

        result = df.select(iou=mask_iou(pred, target))

        iou_value = result.row(0)[0]
        assert iou_value < 0.001, f"Expected IoU ~0, got {iou_value}"

    def test_iou_partial_overlap(self) -> None:
        """Test IoU for partially overlapping masks."""
        # Mask 1: left half white (50 pixels)
        # Mask 2: top half white (50 pixels)
        # Intersection: top-left quarter (25 pixels)
        # Union: 75 pixels
        # Expected IoU: 25/75 = 0.333...

        mask1 = [[255] * 5 + [0] * 5 for _ in range(10)]
        mask2 = [[255] * 10 for _ in range(5)] + [[0] * 10 for _ in range(5)]

        mask1_bytes = create_mask_bytes(mask1)
        mask2_bytes = create_mask_bytes(mask2)

        df = pl.DataFrame(
            {
                "pred": [mask1_bytes],
                "target": [mask2_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        pred = pl.col("pred").cv.pipe(pipe).alias("pred")
        target = pl.col("target").cv.pipe(pipe).alias("target")

        result = df.select(iou=mask_iou(pred, target))

        iou_value = result.row(0)[0]
        expected = 25 / 75
        assert abs(iou_value - expected) < 0.01, (
            f"Expected IoU ~{expected:.3f}, got {iou_value:.3f}"
        )


class TestMaskDice:
    """Test mask Dice coefficient computation."""

    def test_dice_perfect_overlap(self) -> None:
        """Test Dice = 1.0 for identical masks."""
        # Create identical masks
        mask = [[255] * 10 for _ in range(10)]
        mask_bytes = create_mask_bytes(mask)

        df = pl.DataFrame(
            {
                "pred": [mask_bytes],
                "target": [mask_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        pred = pl.col("pred").cv.pipe(pipe).alias("pred")
        target = pl.col("target").cv.pipe(pipe).alias("target")

        result = df.select(dice=mask_dice(pred, target))

        dice_value = result.row(0)[0]
        assert abs(dice_value - 1.0) < 0.001, f"Expected Dice ~1.0, got {dice_value}"

    def test_dice_no_overlap(self) -> None:
        """Test Dice ≈ 0 for non-overlapping masks."""
        # Create non-overlapping masks
        mask1 = [[255] * 5 + [0] * 5 for _ in range(10)]  # Left half
        mask2 = [[0] * 5 + [255] * 5 for _ in range(10)]  # Right half

        mask1_bytes = create_mask_bytes(mask1)
        mask2_bytes = create_mask_bytes(mask2)

        df = pl.DataFrame(
            {
                "pred": [mask1_bytes],
                "target": [mask2_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        pred = pl.col("pred").cv.pipe(pipe).alias("pred")
        target = pl.col("target").cv.pipe(pipe).alias("target")

        result = df.select(dice=mask_dice(pred, target))

        dice_value = result.row(0)[0]
        assert dice_value < 0.001, f"Expected Dice ~0, got {dice_value}"

    def test_dice_partial_overlap(self) -> None:
        """Test Dice for partially overlapping masks."""
        # Mask 1: left half white (50 pixels)
        # Mask 2: top half white (50 pixels)
        # Intersection: 25 pixels
        # Sum of areas: 50 + 50 = 100 pixels
        # Expected Dice: 2 * 25 / 100 = 0.5

        mask1 = [[255] * 5 + [0] * 5 for _ in range(10)]
        mask2 = [[255] * 10 for _ in range(5)] + [[0] * 10 for _ in range(5)]

        mask1_bytes = create_mask_bytes(mask1)
        mask2_bytes = create_mask_bytes(mask2)

        df = pl.DataFrame(
            {
                "pred": [mask1_bytes],
                "target": [mask2_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        pred = pl.col("pred").cv.pipe(pipe).alias("pred")
        target = pl.col("target").cv.pipe(pipe).alias("target")

        result = df.select(dice=mask_dice(pred, target))

        dice_value = result.row(0)[0]
        expected = 2 * 25 / 100
        assert abs(dice_value - expected) < 0.01, (
            f"Expected Dice ~{expected:.3f}, got {dice_value:.3f}"
        )


class TestMaskMetricsBatch:
    """Test mask metrics on multiple rows."""

    def test_iou_multiple_rows(self) -> None:
        """Test IoU computation on multiple rows."""
        # Create three pairs with different overlaps
        # Row 1: Perfect overlap (IoU = 1)
        # Row 2: No overlap (IoU = 0)
        # Row 3: Partial overlap (IoU = 0.333)

        full_mask = [[255] * 10 for _ in range(10)]
        left_mask = [[255] * 5 + [0] * 5 for _ in range(10)]
        right_mask = [[0] * 5 + [255] * 5 for _ in range(10)]
        top_mask = [[255] * 10 for _ in range(5)] + [[0] * 10 for _ in range(5)]

        df = pl.DataFrame(
            {
                "pred": [
                    create_mask_bytes(full_mask),
                    create_mask_bytes(left_mask),
                    create_mask_bytes(left_mask),
                ],
                "target": [
                    create_mask_bytes(full_mask),
                    create_mask_bytes(right_mask),
                    create_mask_bytes(top_mask),
                ],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        pred = pl.col("pred").cv.pipe(pipe).alias("pred")
        target = pl.col("target").cv.pipe(pipe).alias("target")

        result = df.select(iou=mask_iou(pred, target))

        iou_values = result["iou"].to_list()

        # Row 1: Perfect overlap
        assert abs(iou_values[0] - 1.0) < 0.01, (
            f"Row 1: Expected IoU ~1.0, got {iou_values[0]}"
        )

        # Row 2: No overlap
        assert iou_values[1] < 0.01, f"Row 2: Expected IoU ~0, got {iou_values[1]}"

        # Row 3: Partial overlap (25/75 = 0.333)
        expected = 25 / 75
        assert abs(iou_values[2] - expected) < 0.02, (
            f"Row 3: Expected IoU ~{expected:.3f}, got {iou_values[2]}"
        )


class TestIoUDiceRelationship:
    """Test the mathematical relationship between IoU and Dice."""

    def test_dice_iou_relationship(self) -> None:
        """Verify that Dice = 2*IoU / (1 + IoU) holds."""
        # Create masks with known overlap
        mask1 = [[255] * 5 + [0] * 5 for _ in range(10)]
        mask2 = [[255] * 10 for _ in range(5)] + [[0] * 10 for _ in range(5)]

        mask1_bytes = create_mask_bytes(mask1)
        mask2_bytes = create_mask_bytes(mask2)

        df = pl.DataFrame(
            {
                "pred": [mask1_bytes],
                "target": [mask2_bytes],
            }
        )

        pipe = Pipeline().source("image_bytes").grayscale()
        pred = pl.col("pred").cv.pipe(pipe).alias("pred")
        target = pl.col("target").cv.pipe(pipe).alias("target")

        # Compute both metrics
        result_iou = df.select(iou=mask_iou(pred, target))
        result_dice = df.select(dice=mask_dice(pred, target))

        iou_value = result_iou.row(0)[0]
        dice_value = result_dice.row(0)[0]

        # Mathematical relationship: Dice = 2*IoU / (1 + IoU)
        expected_dice = 2 * iou_value / (1 + iou_value)
        assert abs(dice_value - expected_dice) < 0.02, (
            f"Dice should equal 2*IoU/(1+IoU). "
            f"Got Dice={dice_value:.4f}, expected {expected_dice:.4f} from IoU={iou_value:.4f}"
        )
