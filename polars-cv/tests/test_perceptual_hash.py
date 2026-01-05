"""
Tests for perceptual hashing functionality.

This module tests the perceptual hash operation through the full pipeline,
including hash computation, similarity detection, and all algorithm variants.
"""

from __future__ import annotations

from io import BytesIO

import polars as pl
from PIL import Image

from polars_cv import HashAlgorithm, Pipeline


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


class TestPerceptualHashPipeline:
    """Tests for perceptual hash pipeline integration."""

    def test_basic_phash_pipeline(self) -> None:
        """Test basic perceptual hash pipeline execution."""
        # Create test image
        image_bytes = create_test_image(64, 64, (255, 0, 0))

        # Create DataFrame with image
        df = pl.DataFrame({"image": [image_bytes]})

        # Build pipeline with perceptual hash
        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")

        # Execute pipeline
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        # Verify output
        hash_value = result["hash"][0]
        assert hash_value is not None
        assert len(hash_value) == 8  # 64-bit hash = 8 bytes
        # List sink returns f64 values, which should be integer-like u8 values
        assert all(isinstance(b, (int, float)) for b in hash_value)
        # Verify values are valid byte range
        assert all(0 <= int(b) <= 255 for b in hash_value)

    def test_all_hash_algorithms(self) -> None:
        """Test that all hash algorithms work."""
        image_bytes = create_test_image(64, 64, (0, 255, 0))
        df = pl.DataFrame({"image": [image_bytes]})

        algorithms = [
            HashAlgorithm.AVERAGE,
            HashAlgorithm.DIFFERENCE,
            HashAlgorithm.PERCEPTUAL,
            HashAlgorithm.BLOCKHASH,
        ]

        for algo in algorithms:
            pipe = (
                Pipeline()
                .source("image_bytes")
                .perceptual_hash(algorithm=algo)
                .sink("list")
            )
            result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))
            hash_value = result["hash"][0]

            assert hash_value is not None, f"Algorithm {algo} returned None"
            assert len(hash_value) == 8, f"Algorithm {algo} produced wrong hash size"

    def test_same_image_same_hash(self) -> None:
        """Test that identical images produce identical hashes."""
        image_bytes = create_gradient_image(128, 128, "horizontal")

        df = pl.DataFrame({"image": [image_bytes, image_bytes]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        # List columns return Series when indexed, so convert to list
        hash1 = result["hash"][0].to_list()
        hash2 = result["hash"][1].to_list()

        assert hash1 == hash2, "Same image should produce same hash"

    def test_different_images_different_hash(self) -> None:
        """Test that structurally different images produce different hashes."""
        # Create two images with very different structure
        # Image 1: Checkerboard pattern
        img1 = Image.new("RGB", (128, 128))
        pixels1 = img1.load()
        for y in range(128):
            for x in range(128):
                if (x // 16 + y // 16) % 2 == 0:
                    pixels1[x, y] = (255, 255, 255)
                else:
                    pixels1[x, y] = (0, 0, 0)
        buffer1 = BytesIO()
        img1.save(buffer1, format="PNG")

        # Image 2: Diagonal stripes
        img2 = Image.new("RGB", (128, 128))
        pixels2 = img2.load()
        for y in range(128):
            for x in range(128):
                if (x + y) % 32 < 16:
                    pixels2[x, y] = (255, 255, 255)
                else:
                    pixels2[x, y] = (0, 0, 0)
        buffer2 = BytesIO()
        img2.save(buffer2, format="PNG")

        df = pl.DataFrame({"image": [buffer1.getvalue(), buffer2.getvalue()]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        # List columns return Series when indexed, so convert to list
        hash1 = result["hash"][0].to_list()
        hash2 = result["hash"][1].to_list()

        # Hashes should be different due to structural difference
        assert hash1 != hash2, (
            "Structurally different images should produce different hashes"
        )

    def test_hash_with_preprocessing(self) -> None:
        """Test perceptual hash after preprocessing operations."""
        image_bytes = create_gradient_image(256, 256)

        df = pl.DataFrame({"image": [image_bytes]})

        # Apply preprocessing before hashing
        pipe = (
            Pipeline()
            .source("image_bytes")
            .resize(height=128, width=128)
            .grayscale()
            .perceptual_hash()
            .sink("list")
        )

        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))
        hash_value = result["hash"][0]

        assert hash_value is not None
        assert len(hash_value) == 8

    def test_larger_hash_size(self) -> None:
        """Test with larger hash sizes."""
        image_bytes = create_test_image(128, 128, (128, 128, 128))

        df = pl.DataFrame({"image": [image_bytes]})

        # Use 256-bit hash (32 bytes)
        pipe = (
            Pipeline().source("image_bytes").perceptual_hash(hash_size=256).sink("list")
        )

        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))
        hash_value = result["hash"][0]

        assert hash_value is not None
        assert len(hash_value) == 32  # 256 bits = 32 bytes

    def test_string_algorithm_parameter(self) -> None:
        """Test that string algorithm names work."""
        image_bytes = create_test_image(64, 64, (100, 100, 100))

        df = pl.DataFrame({"image": [image_bytes]})

        # Use string instead of enum
        pipe = (
            Pipeline()
            .source("image_bytes")
            .perceptual_hash(algorithm="perceptual")
            .sink("list")
        )

        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))
        hash_value = result["hash"][0]

        assert hash_value is not None
        assert len(hash_value) == 8

    def test_multiple_images_batch(self) -> None:
        """Test processing multiple images in a batch."""
        images = [
            create_test_image(64, 64, (255, 0, 0)),
            create_test_image(64, 64, (0, 255, 0)),
            create_test_image(64, 64, (0, 0, 255)),
            create_gradient_image(64, 64, "horizontal"),
            create_gradient_image(64, 64, "vertical"),
        ]

        df = pl.DataFrame({"image": images})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        # All rows should have hashes
        assert result["hash"].null_count() == 0

        # All hashes should be 8 bytes
        for hash_value in result["hash"]:
            assert len(hash_value) == 8


class TestHashSimilarity:
    """Tests for hash similarity and comparison."""

    @staticmethod
    def hamming_distance(hash1: list[int], hash2: list[int]) -> int:
        """
        Compute Hamming distance between two hashes.

        Args:
            hash1: First hash as list of bytes.
            hash2: Second hash as list of bytes.

        Returns:
            Number of differing bits.
        """
        if len(hash1) != len(hash2):
            raise ValueError("Hashes must be same length")

        distance = 0
        for b1, b2 in zip(hash1, hash2):
            # Cast to int in case list sink returns f64 values
            xor = int(b1) ^ int(b2)
            distance += bin(xor).count("1")
        return distance

    def test_resize_robustness(self) -> None:
        """Test that hashes are similar after resize."""
        # Create original image
        original = create_gradient_image(256, 256, "horizontal")

        # Create resized version
        img = Image.open(BytesIO(original))
        resized_img = img.resize((128, 128), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        resized_img.save(buffer, format="PNG")
        resized = buffer.getvalue()

        df = pl.DataFrame({"image": [original, resized]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        # List columns return Series when indexed, so convert to list
        hash1 = result["hash"][0].to_list()
        hash2 = result["hash"][1].to_list()

        # Hashes should be similar (low Hamming distance)
        distance = self.hamming_distance(hash1, hash2)
        # Allow up to 16 bits difference (25% of 64 bits) for resize
        assert distance <= 16, f"Resize changed hash too much: {distance} bits"

    def test_format_conversion_robustness(self) -> None:
        """Test that hashes are similar after format conversion (PNG -> JPEG)."""
        # Create original PNG
        img = Image.new("RGB", (128, 128), (100, 150, 200))

        # Add some structure so the hash has something to work with
        pixels = img.load()
        for y in range(64):
            for x in range(128):
                pixels[x, y] = (200, 100, 50)

        png_buffer = BytesIO()
        img.save(png_buffer, format="PNG")
        png_bytes = png_buffer.getvalue()

        jpeg_buffer = BytesIO()
        img.save(jpeg_buffer, format="JPEG", quality=90)
        jpeg_bytes = jpeg_buffer.getvalue()

        df = pl.DataFrame({"image": [png_bytes, jpeg_bytes]})

        pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")
        result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))

        # List columns return Series when indexed, so convert to list
        hash1 = result["hash"][0].to_list()
        hash2 = result["hash"][1].to_list()

        # High quality JPEG should produce nearly identical hash
        distance = self.hamming_distance(hash1, hash2)
        # Allow up to 8 bits difference (12.5% of 64 bits) for high-quality JPEG
        assert distance <= 8, f"JPEG conversion changed hash too much: {distance} bits"


class TestHashAlgorithmComparison:
    """Tests comparing different hash algorithms."""

    def test_algorithms_produce_different_hashes(self) -> None:
        """Test that different algorithms produce different hash values."""
        image_bytes = create_gradient_image(128, 128, "horizontal")

        df = pl.DataFrame({"image": [image_bytes]})

        algorithms = [
            HashAlgorithm.AVERAGE,
            HashAlgorithm.DIFFERENCE,
            HashAlgorithm.PERCEPTUAL,
            HashAlgorithm.BLOCKHASH,
        ]

        hashes = {}
        for algo in algorithms:
            pipe = (
                Pipeline()
                .source("image_bytes")
                .perceptual_hash(algorithm=algo)
                .sink("list")
            )
            result = df.with_columns(hash=pl.col("image").cv.pipeline(pipe))
            hashes[algo] = result["hash"][0]

        # At least some algorithms should produce different hashes
        unique_hashes = set(tuple(h) for h in hashes.values())
        assert len(unique_hashes) >= 2, (
            "Different algorithms should produce different hashes"
        )
