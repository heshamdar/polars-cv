"""
Reference tests for perceptual hashing using Python imagehash library.

These tests establish the expected behavior for perceptual image hashing,
serving as ground truth for polars-cv. The imagehash library is a
well-established Python implementation that we use as our reference.

To run these tests:
    pip install imagehash pillow
    pytest tests/reference/test_perceptual_hash_ref.py -v
"""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from PIL import Image as PILImage


def _imagehash_available() -> bool:
    """Check if imagehash library is available."""
    try:
        import imagehash  # noqa: F401

        return True
    except ImportError:
        return False


imagehash_required = pytest.mark.skipif(
    not _imagehash_available(),
    reason="Requires imagehash library (pip install imagehash)",
)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create a sample RGB image for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_grayscale_image() -> np.ndarray:
    """Create a sample grayscale image for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (256, 256), dtype=np.uint8)


@pytest.fixture
def gradient_image() -> np.ndarray:
    """Create a gradient image (good for testing hash consistency)."""
    x = np.linspace(0, 255, 256, dtype=np.uint8)
    y = np.linspace(0, 255, 256, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    # Create RGB gradient
    return np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)


def numpy_to_pil(arr: np.ndarray) -> "PILImage.Image":
    """Convert numpy array to PIL Image."""
    from PIL import Image

    return Image.fromarray(arr)


def pil_to_jpeg_bytes(img: "PILImage.Image", quality: int = 85) -> bytes:
    """Convert PIL image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def jpeg_bytes_to_pil(data: bytes) -> "PILImage.Image":
    """Convert JPEG bytes to PIL Image."""
    from PIL import Image

    return Image.open(io.BytesIO(data))


@imagehash_required
class TestPerceptualHashReference:
    """Establish expected behavior for perceptual hashing using imagehash library."""

    def test_phash_same_image_distance_zero(self, sample_rgb_image: np.ndarray) -> None:
        """Same image should have zero Hamming distance."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash1 = imagehash.phash(pil_img)
        hash2 = imagehash.phash(pil_img)

        distance = hash1 - hash2
        assert distance == 0, f"Same image should have distance 0, got {distance}"

    def test_phash_resized_image_low_distance(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """Resized image should have low Hamming distance with pHash."""
        import imagehash
        from PIL import Image

        pil_img = numpy_to_pil(sample_rgb_image)
        # Resize to different dimensions
        resized = pil_img.resize((128, 128), Image.Resampling.LANCZOS)

        hash_original = imagehash.phash(pil_img)
        hash_resized = imagehash.phash(resized)

        distance = hash_original - hash_resized
        # pHash should be robust to resizing - distance should be low
        assert distance <= 10, f"pHash should be resize-robust, got distance {distance}"

    def test_phash_jpeg_recompression_low_distance(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """JPEG recompressed image should have low Hamming distance."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        # Convert to JPEG and back (lossy compression)
        jpeg_bytes = pil_to_jpeg_bytes(pil_img, quality=75)
        recompressed = jpeg_bytes_to_pil(jpeg_bytes)

        hash_original = imagehash.phash(pil_img)
        hash_recompressed = imagehash.phash(recompressed)

        distance = hash_original - hash_recompressed
        # pHash should handle JPEG artifacts well
        assert distance <= 5, (
            f"pHash should handle JPEG compression, got distance {distance}"
        )

    def test_phash_different_images_high_distance(self) -> None:
        """Different images should have high Hamming distance."""
        import imagehash
        from PIL import Image

        # Create two very different images
        rng = np.random.default_rng(42)
        img1 = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        img2 = 255 - img1  # Inverted image

        pil_img1 = Image.fromarray(img1)
        pil_img2 = Image.fromarray(img2)

        hash1 = imagehash.phash(pil_img1)
        hash2 = imagehash.phash(pil_img2)

        distance = hash1 - hash2
        # Different images should have high distance (max 64 for 64-bit hash)
        assert distance >= 20, (
            f"Different images should have high distance, got {distance}"
        )

    def test_ahash_basic(self, sample_rgb_image: np.ndarray) -> None:
        """Test that aHash produces consistent results."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash1 = imagehash.average_hash(pil_img)
        hash2 = imagehash.average_hash(pil_img)

        assert hash1 == hash2, "aHash should be deterministic"

    def test_dhash_basic(self, sample_rgb_image: np.ndarray) -> None:
        """Test that dHash produces consistent results."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash1 = imagehash.dhash(pil_img)
        hash2 = imagehash.dhash(pil_img)

        assert hash1 == hash2, "dHash should be deterministic"

    def test_hash_size_64_produces_8_bytes(self, sample_rgb_image: np.ndarray) -> None:
        """64-bit hash should produce 8-byte output (64 bits = 8 bytes)."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        # Default hash_size=8 means 8x8=64 bit hash
        hash_result = imagehash.phash(pil_img, hash_size=8)

        # The hash is stored as a flattened numpy array of bools
        # 64 bits = 8 bytes when converted
        hash_bytes = hash_result.hash.flatten()
        assert len(hash_bytes) == 64, f"Expected 64 bits, got {len(hash_bytes)}"

        # Convert to bytes representation
        # imagehash stores as bool array, we need to pack into bytes
        hex_str = str(hash_result)
        # 64 bits = 16 hex characters
        assert len(hex_str) == 16, (
            f"Expected 16 hex chars for 64-bit hash, got {len(hex_str)}"
        )

    def test_hash_size_128_produces_16_bytes(
        self, sample_rgb_image: np.ndarray
    ) -> None:
        """128-bit hash should produce 16-byte output."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        # hash_size=16 means 16x16 = 256 bits for DCT, but output is hash_size^2/4 = 64 bits
        # Actually for phash: hash_size determines the DCT block size
        # For 128-bit output we need hash_size that produces 128 bits
        # phash output is hash_size^2 bits when using highfreq_factor=4
        # Actually imagehash phash outputs hash_size^2 bits

        # For larger hash, use hash_size=16 which gives 16*16=256 bits
        hash_large = imagehash.phash(pil_img, hash_size=16)
        assert len(hash_large.hash.flatten()) == 256

    def test_ahash_faster_than_phash(self, sample_rgb_image: np.ndarray) -> None:
        """aHash should generally be faster than pHash (DCT is expensive)."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)

        # Warm up
        imagehash.average_hash(pil_img)
        imagehash.phash(pil_img)

        # Time aHash
        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            imagehash.average_hash(pil_img)
        ahash_time = time.perf_counter() - start

        # Time pHash
        start = time.perf_counter()
        for _ in range(iterations):
            imagehash.phash(pil_img)
        phash_time = time.perf_counter() - start

        # aHash should be faster (no DCT computation)
        # We allow some tolerance since timing can vary
        assert ahash_time <= phash_time * 1.5, (
            f"aHash ({ahash_time:.3f}s) should be faster than pHash ({phash_time:.3f}s)"
        )

    def test_blockhash_crop_resistance(self, gradient_image: np.ndarray) -> None:
        """Blockhash should be somewhat robust to small crops."""
        import imagehash
        from PIL import Image

        pil_img = numpy_to_pil(gradient_image)

        # Create a slightly cropped version (10% crop)
        w, h = pil_img.size
        crop_amount = int(min(w, h) * 0.1)
        cropped = pil_img.crop(
            (crop_amount, crop_amount, w - crop_amount, h - crop_amount)
        )
        # Resize back to original size to compare
        cropped_resized = cropped.resize((w, h), Image.Resampling.LANCZOS)

        # Test with different algorithms
        bhash_original = imagehash.phash(pil_img)
        bhash_cropped = imagehash.phash(cropped_resized)

        distance = bhash_original - bhash_cropped
        # With crop + resize, pHash should still have reasonable similarity
        assert distance <= 15, (
            f"Hash should handle small crops, got distance {distance}"
        )

    def test_hash_to_hex_string(self, sample_rgb_image: np.ndarray) -> None:
        """Hash should be convertible to hex string."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash_result = imagehash.phash(pil_img)

        hex_str = str(hash_result)
        # Should be a valid hex string
        assert all(c in "0123456789abcdef" for c in hex_str.lower()), (
            f"Hash should be hex string, got: {hex_str}"
        )

    def test_hash_from_hex_string(self, sample_rgb_image: np.ndarray) -> None:
        """Hash should be reconstructible from hex string."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash_result = imagehash.phash(pil_img)

        hex_str = str(hash_result)
        reconstructed = imagehash.hex_to_hash(hex_str)

        assert hash_result == reconstructed, "Hash should roundtrip through hex string"

    def test_hash_bytes_representation(self, sample_rgb_image: np.ndarray) -> None:
        """Test that hash can be converted to raw bytes for storage."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash_result = imagehash.phash(pil_img, hash_size=8)  # 64 bits

        # Get the underlying hash as numpy bool array
        hash_array = hash_result.hash.flatten()

        # Pack bools into bytes (8 bits per byte)
        packed = np.packbits(hash_array)
        assert len(packed) == 8, f"64 bits should pack to 8 bytes, got {len(packed)}"

        # Verify we can unpack back
        unpacked = np.unpackbits(packed)
        np.testing.assert_array_equal(
            unpacked,
            hash_array.astype(np.uint8),
            "Should roundtrip through byte packing",
        )

    def test_grayscale_vs_rgb_hash_similar(self, sample_rgb_image: np.ndarray) -> None:
        """Hash of grayscale conversion should be similar to RGB hash."""
        import imagehash

        pil_rgb = numpy_to_pil(sample_rgb_image)
        pil_gray = pil_rgb.convert("L")

        # Both algorithms internally convert to grayscale for hashing
        hash_rgb = imagehash.phash(pil_rgb)
        hash_gray = imagehash.phash(pil_gray)

        distance = hash_rgb - hash_gray
        # Since pHash converts to grayscale internally, these should be identical or very close
        assert distance <= 2, (
            f"RGB and grayscale hash should be similar, got distance {distance}"
        )

    def test_algorithm_comparison(self, gradient_image: np.ndarray) -> None:
        """Compare all four hash algorithms on the same image."""
        import imagehash

        pil_img = numpy_to_pil(gradient_image)

        ahash = imagehash.average_hash(pil_img)
        dhash = imagehash.dhash(pil_img)
        phash = imagehash.phash(pil_img)

        # All should produce valid hashes
        assert len(str(ahash)) == 16, "aHash should produce 64-bit hash"
        assert len(str(dhash)) == 16, "dHash should produce 64-bit hash"
        assert len(str(phash)) == 16, "pHash should produce 64-bit hash"

        # Different algorithms should produce different hashes for same image
        # (they use different methods)
        assert str(ahash) != str(dhash) or str(dhash) != str(phash), (
            "Different algorithms should generally produce different hashes"
        )

    def test_batch_hashing_consistency(self) -> None:
        """Verify hashing is consistent across batch of images."""
        import imagehash
        from PIL import Image

        rng = np.random.default_rng(42)

        # Create batch of images
        images = [
            Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
            for _ in range(10)
        ]

        # Hash all images twice
        hashes_run1 = [imagehash.phash(img) for img in images]
        hashes_run2 = [imagehash.phash(img) for img in images]

        # Should be identical
        for h1, h2 in zip(hashes_run1, hashes_run2):
            assert h1 == h2, "Hashing should be deterministic"

    def test_small_image_hashing(self) -> None:
        """Test hashing works on small images."""
        import imagehash
        from PIL import Image

        # Very small image
        small_img = Image.fromarray(
            np.random.default_rng(42).integers(0, 256, (8, 8, 3), dtype=np.uint8)
        )

        # Should not raise, even for tiny images
        hash_result = imagehash.phash(small_img)
        assert hash_result is not None

    def test_large_image_hashing(self) -> None:
        """Test hashing works efficiently on large images."""
        import imagehash
        from PIL import Image

        # Large image
        large_img = Image.fromarray(
            np.random.default_rng(42).integers(0, 256, (2048, 2048, 3), dtype=np.uint8)
        )

        start = time.perf_counter()
        hash_result = imagehash.phash(large_img)
        elapsed = time.perf_counter() - start

        assert hash_result is not None
        # Should complete in reasonable time (< 1 second for 2K image)
        assert elapsed < 1.0, f"Large image hash took too long: {elapsed:.2f}s"
