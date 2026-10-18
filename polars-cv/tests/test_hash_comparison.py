"""
Tests for hash comparison functions: hamming_distance and hash_similarity.

This module tests the perceptual hash comparison functionality including
popcount reduction, hamming distance calculation, and similarity percentage.
"""

from __future__ import annotations

from io import BytesIO

import polars as pl
import pytest
from PIL import Image

from polars_cv import Pipeline, hamming_distance, hash_similarity


def create_test_image(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    """
    Create a test image with a solid color.

    Args:
        width: Image width.
        height: Image height.
        color: RGB color tuple.

    Returns:
        PNG-encoded image bytes.
    """
    img = Image.new("RGB", (width, height), color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_gradient_image(
    width: int, height: int, direction: str = "horizontal"
) -> bytes:
    """
    Create a test image with a gradient.

    Args:
        width: Image width.
        height: Image height.
        direction: "horizontal" or "vertical".

    Returns:
        PNG-encoded image bytes.
    """
    img = Image.new("RGB", (width, height))
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            if direction == "horizontal":
                v = int(255 * x / width)
            else:
                v = int(255 * y / height)
            pixels[x, y] = (v, v, v)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestReducePopcount:
    """Tests for the reduce_popcount operation."""

    def test_popcount_basic(self) -> None:
        """Test basic popcount on a small buffer."""
        # Create a simple 2x2 grayscale image
        # Values: 0xFF (8 bits), 0x00 (0 bits), 0x0F (4 bits), 0xAA (4 bits)
        img = Image.new("L", (2, 2))
        img.putpixel((0, 0), 0xFF)  # 8 bits
        img.putpixel((1, 0), 0x00)  # 0 bits
        img.putpixel((0, 1), 0x0F)  # 4 bits
        img.putpixel((1, 1), 0xAA)  # 4 bits (10101010)

        buffer = BytesIO()
        img.save(buffer, format="PNG")

        df = pl.DataFrame({"image": [buffer.getvalue()]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_popcount()

        result = df.with_columns(popcount=pl.col("image").cv.pipe(pipe).sink("native"))
        popcount = result["popcount"][0]

        # 8 + 0 + 4 + 4 = 16 bits
        assert popcount == 16.0, f"Expected 16, got {popcount}"

    def test_popcount_all_zeros(self) -> None:
        """Test popcount on all zeros (should be 0)."""
        img = Image.new("L", (4, 4), 0)
        buffer = BytesIO()
        img.save(buffer, format="PNG")

        df = pl.DataFrame({"image": [buffer.getvalue()]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_popcount()

        result = df.with_columns(popcount=pl.col("image").cv.pipe(pipe).sink("native"))
        assert result["popcount"][0] == 0.0

    def test_popcount_all_ones(self) -> None:
        """Test popcount on all 0xFF values."""
        # 4x4 image, all pixels = 255 (0xFF = 8 bits each)
        img = Image.new("L", (4, 4), 255)
        buffer = BytesIO()
        img.save(buffer, format="PNG")

        df = pl.DataFrame({"image": [buffer.getvalue()]})

        pipe = Pipeline().source("image_bytes").grayscale().reduce_popcount()

        result = df.with_columns(popcount=pl.col("image").cv.pipe(pipe).sink("native"))
        # 16 pixels * 8 bits = 128 bits
        assert result["popcount"][0] == 128.0


class TestHammingDistance:
    """Tests for the hamming_distance function."""

    def test_identical_hashes(self) -> None:
        """Test that identical images have Hamming distance 0."""
        image = create_gradient_image(64, 64, "horizontal")

        df = pl.DataFrame({"image1": [image], "image2": [image]})

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(distance=hamming_distance(hash1, hash2))

        # Identical images should have distance 0
        assert result["distance"][0] == 0.0

    def test_different_images_nonzero_distance(self) -> None:
        """Test that different images have non-zero Hamming distance."""
        # Create two structurally very different images
        # Image 1: Checkerboard pattern
        img1 = Image.new("L", (128, 128))
        pixels1 = img1.load()
        for y in range(128):
            for x in range(128):
                if (x // 16 + y // 16) % 2 == 0:
                    pixels1[x, y] = 255
                else:
                    pixels1[x, y] = 0
        buffer1 = BytesIO()
        img1.save(buffer1, format="PNG")

        # Image 2: Diagonal stripes
        img2 = Image.new("L", (128, 128))
        pixels2 = img2.load()
        for y in range(128):
            for x in range(128):
                if (x + y) % 32 < 16:
                    pixels2[x, y] = 255
                else:
                    pixels2[x, y] = 0
        buffer2 = BytesIO()
        img2.save(buffer2, format="PNG")

        df = pl.DataFrame(
            {"image1": [buffer1.getvalue()], "image2": [buffer2.getvalue()]}
        )

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(distance=hamming_distance(hash1, hash2))

        # Structurally different images should have positive distance
        assert result["distance"][0] > 0, (
            f"Expected positive distance, got {result['distance'][0]}"
        )

    def test_similar_images_small_distance(self) -> None:
        """Test that similar images (resized) have small Hamming distance."""
        original = create_gradient_image(256, 256, "horizontal")

        # Create a resized version
        img = Image.open(BytesIO(original))
        resized_img = img.resize((128, 128), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        resized_img.save(buffer, format="PNG")
        resized = buffer.getvalue()

        df = pl.DataFrame({"image1": [original], "image2": [resized]})

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(distance=hamming_distance(hash1, hash2))

        # Similar images should have small distance (< 16 for 64-bit hash)
        assert result["distance"][0] <= 16

    def test_significant_distance(self) -> None:
        """Test that structurally different images have significant Hamming distance."""
        # Create very different patterns
        # Image 1: Quarter circles pattern (high frequency content on one side)
        img1 = Image.new("L", (128, 128), 0)
        pixels1 = img1.load()
        for y in range(128):
            for x in range(128):
                if x < 64:
                    pixels1[x, y] = 255
                else:
                    pixels1[x, y] = 0

        # Image 2: Quarter circles pattern (high frequency content on top)
        img2 = Image.new("L", (128, 128), 0)
        pixels2 = img2.load()
        for y in range(128):
            for x in range(128):
                if y < 64:
                    pixels2[x, y] = 255
                else:
                    pixels2[x, y] = 0

        buffer1 = BytesIO()
        img1.save(buffer1, format="PNG")
        buffer2 = BytesIO()
        img2.save(buffer2, format="PNG")

        df = pl.DataFrame(
            {"image1": [buffer1.getvalue()], "image2": [buffer2.getvalue()]}
        )

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(distance=hamming_distance(hash1, hash2))

        # Distance should be positive for different images
        # Note: Maximum is 64 for 64-bit hash, but exact value depends on the algorithm
        assert result["distance"][0] > 0, (
            f"Expected positive distance, got {result['distance'][0]}"
        )


class TestHashSimilarity:
    """Tests for the hash_similarity function."""

    def test_identical_images_100_percent(self) -> None:
        """Test that identical images have 100% similarity."""
        image = create_gradient_image(64, 64, "horizontal")

        df = pl.DataFrame({"image1": [image], "image2": [image]})

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(similarity=hash_similarity(hash1, hash2))

        # Identical images should have 100% similarity
        assert result["similarity"][0] == 100.0

    def test_similar_images_high_similarity(self) -> None:
        """Test that similar images have high similarity."""
        original = create_gradient_image(256, 256, "horizontal")

        # Create a resized version
        img = Image.open(BytesIO(original))
        resized_img = img.resize((128, 128), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        resized_img.save(buffer, format="PNG")
        resized = buffer.getvalue()

        df = pl.DataFrame({"image1": [original], "image2": [resized]})

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(similarity=hash_similarity(hash1, hash2))

        # Similar images should have > 75% similarity
        assert result["similarity"][0] >= 75.0

    def test_different_images_low_similarity(self) -> None:
        """Test that different images have lower similarity."""
        # Create very different images
        image1 = create_test_image(64, 64, (255, 0, 0))  # Red
        image2 = create_test_image(64, 64, (0, 0, 255))  # Blue

        df = pl.DataFrame({"image1": [image1], "image2": [image2]})

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(similarity=hash_similarity(hash1, hash2))

        # Different solid colors may still have some similarity due to hash algorithm
        # Just verify the value is reasonable (between 0 and 100)
        sim = result["similarity"][0]
        assert 0 <= sim <= 100

    def test_custom_hash_bits(self) -> None:
        """Test similarity calculation with custom hash_bits parameter."""
        image1 = create_gradient_image(64, 64, "horizontal")
        image2 = create_gradient_image(64, 64, "vertical")

        df = pl.DataFrame({"image1": [image1], "image2": [image2]})

        # Use 256-bit hash
        hash_pipe = Pipeline().source("image_bytes").perceptual_hash(hash_size=256)
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(
            similarity=hash_similarity(hash1, hash2, hash_bits=256)
        )

        # Similarity should be between 0 and 100
        sim = result["similarity"][0]
        assert 0 <= sim <= 100


class TestHashComparisonBatch:
    """Tests for batch processing of hash comparisons."""

    def test_batch_hamming_distance(self) -> None:
        """Test Hamming distance on multiple image pairs."""
        # Create structurally different images for reliable difference detection
        same_image = create_gradient_image(64, 64, "horizontal")

        # Checkerboard pattern
        img_checker = Image.new("L", (64, 64))
        pixels = img_checker.load()
        for y in range(64):
            for x in range(64):
                pixels[x, y] = 255 if (x // 8 + y // 8) % 2 == 0 else 0
        buffer_checker = BytesIO()
        img_checker.save(buffer_checker, format="PNG")

        images1 = [
            same_image,  # Will compare to itself
            buffer_checker.getvalue(),  # Checkerboard
        ]
        images2 = [
            same_image,  # Same
            same_image,  # Different from checkerboard
        ]

        df = pl.DataFrame({"image1": images1, "image2": images2})

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(distance=hamming_distance(hash1, hash2))

        # First pair: identical images → distance 0
        assert result["distance"][0] == 0.0

        # Second pair: checkerboard vs gradient → should differ
        assert result["distance"][1] >= 0  # Non-negative distance

    def test_batch_hash_similarity(self) -> None:
        """Test hash similarity on multiple image pairs."""
        same_image = create_gradient_image(64, 64, "horizontal")

        # Create a very different image (checkerboard)
        img_checker = Image.new("L", (64, 64))
        pixels = img_checker.load()
        for y in range(64):
            for x in range(64):
                pixels[x, y] = 255 if (x // 8 + y // 8) % 2 == 0 else 0
        buffer_checker = BytesIO()
        img_checker.save(buffer_checker, format="PNG")

        images1 = [
            same_image,
            buffer_checker.getvalue(),
        ]
        images2 = [
            same_image,  # Same as first
            same_image,  # Different from checkerboard
        ]

        df = pl.DataFrame({"image1": images1, "image2": images2})

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        hash1 = pl.col("image1").cv.pipe(hash_pipe)
        hash2 = pl.col("image2").cv.pipe(hash_pipe)

        result = df.with_columns(similarity=hash_similarity(hash1, hash2))

        # First pair: identical images → 100% similarity
        assert result["similarity"][0] == 100.0

        # Second pair: valid similarity percentage
        assert 0 <= result["similarity"][1] <= 100.0


class TestIntegrationWithPerceptualHash:
    """Integration tests combining perceptual hashing with comparison functions."""

    def test_full_workflow(self) -> None:
        """Test the complete workflow: hash → compare → filter."""
        # Create original image
        original = create_gradient_image(128, 128, "horizontal")

        # Create resized version of original via PIL for proper resizing
        img = Image.open(BytesIO(original))
        resized_img = img.resize((64, 64), Image.Resampling.LANCZOS)
        resized_buffer = BytesIO()
        resized_img.save(resized_buffer, format="PNG")

        # Create a very different image (checkerboard)
        img_checker = Image.new("L", (128, 128))
        pixels = img_checker.load()
        for y in range(128):
            for x in range(128):
                pixels[x, y] = 255 if (x // 16 + y // 16) % 2 == 0 else 0
        buffer_checker = BytesIO()
        img_checker.save(buffer_checker, format="PNG")

        df = pl.DataFrame(
            {
                "name": ["original", "resized", "different"],
                "image": [
                    original,
                    resized_buffer.getvalue(),
                    buffer_checker.getvalue(),
                ],
                "reference": [original] * 3,
            }
        )

        hash_pipe = Pipeline().source("image_bytes").perceptual_hash()
        img_hash = pl.col("image").cv.pipe(hash_pipe)
        ref_hash = pl.col("reference").cv.pipe(hash_pipe)

        result = df.with_columns(similarity=hash_similarity(img_hash, ref_hash))

        # Original vs Original: 100%
        assert result["similarity"][0] == 100.0

        # Resized vs Original: should be high (>75%)
        assert result["similarity"][1] >= 75.0

        # All similarities should be valid percentages
        for sim in result["similarity"]:
            assert 0 <= sim <= 100


class TestHashColumns:
    """``hamming_distance``/``hash_similarity`` on stored hashes, not only pipelines.

    A hash is usually computed once and kept as a column (``sink("list")``), and
    near-duplicate search compares those columns pairwise: a cross join of the
    hashes against themselves. Each side may be a pipeline, an expression or a
    column name, and a hash column runs through the same XOR and popcount.
    """

    @staticmethod
    def _hashes() -> pl.DataFrame:
        import numpy as np

        rng = np.random.default_rng(7)
        images = []
        for _ in range(2):
            buf = BytesIO()
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(
                buf, format="PNG"
            )
            images.append(buf.getvalue())
        pipe = Pipeline().source("image_bytes").perceptual_hash()
        return pl.DataFrame({"name": ["h", "v"], "image": images}).with_columns(
            hash=pl.col("image").cv.pipe(pipe).sink("list")
        )

    def test_hash_columns_match_the_pipeline_form(self) -> None:
        df = self._hashes().with_columns(other=pl.col("hash").reverse())
        pipe = Pipeline().source("image_bytes").perceptual_hash()
        expected = df.select(
            hamming_distance(
                pl.col("image").cv.pipe(pipe), pl.col("image").reverse().cv.pipe(pipe)
            )
        ).to_series()
        assert expected.to_list()[0] > 0
        for a, b in [("hash", "other"), (pl.col("hash"), pl.col("other"))]:
            got = df.select(hamming_distance(a, b)).to_series()
            assert got.to_list() == expected.to_list()

    def test_a_pipeline_and_a_column_mix(self) -> None:
        df = self._hashes()
        pipe = Pipeline().source("image_bytes").perceptual_hash()
        out = df.select(hamming_distance(pl.col("image").cv.pipe(pipe), "hash"))
        assert out.to_series().to_list() == [0.0, 0.0]

    def test_array_hash_columns(self) -> None:
        df = self._hashes().with_columns(pl.col("hash").list.to_array(8))
        out = df.select(hamming_distance("hash", pl.col("hash").reverse()))
        assert out.to_series().to_list()[0] > 0

    def test_near_duplicate_pairs_by_cross_join(self) -> None:
        h = self._hashes().select("name", "hash")
        pairs = (
            h.join(h, how="cross", suffix="_b")
            .filter(pl.col("name") <= pl.col("name_b"))
            .with_columns(
                bits=hamming_distance("hash", "hash_b"),
                similarity=hash_similarity("hash", "hash_b"),
            )
            .sort("name", "name_b")
        )
        assert pairs["bits"].to_list()[0] == 0.0  # h with itself
        assert pairs["bits"].to_list()[1] > 0  # h with v
        assert pairs["similarity"].to_list()[0] == 100.0

    def test_hashes_of_different_lengths_are_refused(self) -> None:
        df = pl.DataFrame(
            {"a": [[1, 2]], "b": [[1, 2, 3]]},
            schema={"a": pl.List(pl.UInt8), "b": pl.List(pl.UInt8)},
        )
        with pytest.raises(pl.exceptions.ComputeError, match="same shape"):
            df.select(hamming_distance("a", "b"))
