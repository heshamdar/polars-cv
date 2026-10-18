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
import polars as pl
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

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
@plugin_required
@pytest.mark.parity(
    ops=["perceptual_hash"],
    against="imagehash `phash` (64-bit, hash_size=16 for 256 bits), `average_hash` and `dhash` on the same images",
    tolerance=(
        "Hamming bits: phash <= 4 (<= 12 on a colour gradient), 256-bit <= 40, "
        "average/difference <= 10 on smooth images"
    ),
)
class TestPerceptualHashReference:
    """``perceptual_hash()`` against ``imagehash`` on the same images.

    Each test keeps the property it documents for imagehash and holds
    polars-cv to it too: both the property (distance between a pair of images)
    and the hash itself, within the Hamming tolerances ``TestPhashMatchesImagehash``
    establishes (the grayscale and the resize are the engine's own ops, so
    bits differ from imagehash's by a few, not zero).
    """

    @staticmethod
    def _pcv(
        images: "list[PILImage.Image]",
        algorithm: str = "perceptual",
        hash_size: int = 64,
    ) -> list[bytes]:
        """polars-cv hash bytes of each image."""
        from polars_cv import Pipeline

        df = pl.DataFrame({"img": [pil_to_png_bytes(i) for i in images]})
        pipe = (
            Pipeline()
            .source("image_bytes")
            .perceptual_hash(algorithm=algorithm, hash_size=hash_size)
        )
        rows = df.select(h=pl.col("img").cv.pipe(pipe).sink("list"))["h"].to_list()
        return [bytes(r) for r in rows]

    @staticmethod
    def _int(h: bytes) -> int:
        """Bits row-major, most significant first, as ``int(str(imagehash), 16)``."""
        return int.from_bytes(h, "big")

    @staticmethod
    def _ih(h: object) -> int:
        return int(str(h), 16)

    def test_phash_same_image_distance_zero(self, sample_rgb_image: np.ndarray) -> None:
        """Same image should have zero Hamming distance."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash1 = imagehash.phash(pil_img)
        hash2 = imagehash.phash(pil_img)

        distance = hash1 - hash2
        assert distance == 0, f"Same image should have distance 0, got {distance}"

        ours1, ours2 = self._pcv([pil_img, pil_img])
        assert ours1 == ours2, "polars-cv: same image should have distance 0"
        assert (self._int(ours1) ^ self._ih(hash1)).bit_count() <= 4

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

        ours_original, ours_resized = (
            self._int(h) for h in self._pcv([pil_img, resized])
        )
        ours_distance = (ours_original ^ ours_resized).bit_count()
        assert ours_distance <= 10, f"polars-cv pHash resize distance {ours_distance}"
        assert (ours_original ^ self._ih(hash_original)).bit_count() <= 4
        assert (ours_resized ^ self._ih(hash_resized)).bit_count() <= 4

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

        ours_original, ours_recompressed = (
            self._int(h) for h in self._pcv([pil_img, recompressed])
        )
        ours_distance = (ours_original ^ ours_recompressed).bit_count()
        assert ours_distance <= 5, f"polars-cv pHash JPEG distance {ours_distance}"
        assert (ours_original ^ self._ih(hash_original)).bit_count() <= 4
        assert (ours_recompressed ^ self._ih(hash_recompressed)).bit_count() <= 4

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

        ours1, ours2 = (self._int(h) for h in self._pcv([pil_img1, pil_img2]))
        ours_distance = (ours1 ^ ours2).bit_count()
        assert ours_distance >= 20, f"polars-cv distance {ours_distance}"
        assert (ours1 ^ self._ih(hash1)).bit_count() <= 4
        assert (ours2 ^ self._ih(hash2)).bit_count() <= 4

    def test_ahash_basic(self, sample_rgb_image: np.ndarray) -> None:
        """Test that aHash produces consistent results."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash1 = imagehash.average_hash(pil_img)
        hash2 = imagehash.average_hash(pil_img)

        assert hash1 == hash2, "aHash should be deterministic"

        # polars-cv's average hash: deterministic, and close to imagehash's on
        # natural-looking (smooth) images. It is image_hasher's, which resizes
        # with its own filter; measured 2-8 differing bits of 64 on smooth
        # images. (On white noise
        # the 8x8 means all sit near 128 and the bits are rounding-sensitive:
        # 18 bits, so noise is not compared.)
        assert (
            self._pcv([pil_img] * 2, "average")[0] == self._pcv([pil_img], "average")[0]
        )
        for a in _smooth_images(6):
            ours = np.unpackbits(
                np.frombuffer(self._pcv([numpy_to_pil(a)], "average")[0], np.uint8),
            )
            theirs = imagehash.average_hash(numpy_to_pil(a)).hash.ravel()
            assert int((ours != theirs).sum()) <= 10

    def test_dhash_basic(self, sample_rgb_image: np.ndarray) -> None:
        """Test that dHash produces consistent results."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash1 = imagehash.dhash(pil_img)
        hash2 = imagehash.dhash(pil_img)

        assert hash1 == hash2, "dHash should be deterministic"

        # As aHash: image_hasher's own resize; measured 1-8 differing bits of
        # 64 on smooth images.
        assert (
            self._pcv([pil_img] * 2, "difference")[0]
            == self._pcv([pil_img], "difference")[0]
        )
        for a in _smooth_images(6):
            ours = np.unpackbits(
                np.frombuffer(self._pcv([numpy_to_pil(a)], "difference")[0], np.uint8),
            )
            theirs = imagehash.dhash(numpy_to_pil(a)).hash.ravel()
            assert int((ours != theirs).sum()) <= 10

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

        (ours,) = self._pcv([pil_img], hash_size=64)
        assert len(ours) == 8, f"polars-cv: 64 bits should be 8 bytes, got {len(ours)}"
        assert (self._int(ours) ^ int(hex_str, 16)).bit_count() <= 4

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

        # polars-cv's `hash_size` is the number of bits: 256 is imagehash's 16.
        (ours,) = self._pcv([pil_img], hash_size=256)
        assert len(ours) == 32
        assert (self._int(ours) ^ self._ih(hash_large)).bit_count() <= 40

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

        # The timing above documents imagehash only (polars-cv's speed is a
        # benchmark question, not a parity one). The hashes timed are compared:
        (ours,) = self._pcv([pil_img])
        assert (self._int(ours) ^ self._ih(imagehash.phash(pil_img))).bit_count() <= 4

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

        ours_original, ours_cropped = (
            self._int(x) for x in self._pcv([pil_img, cropped_resized])
        )
        assert (ours_original ^ ours_cropped).bit_count() <= 15
        # A colour gradient has many near-zero DCT coefficients whose bits sit
        # on gray-rounding noise: measured 10 bits from imagehash's on the
        # original (0-2 on natural images), hence 12 here rather than 4.
        assert (ours_original ^ self._ih(bhash_original)).bit_count() <= 12
        assert (ours_cropped ^ self._ih(bhash_cropped)).bit_count() <= 12

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

        # polars-cv's bytes are that hex string's bytes: same length, and the
        # hex of the bytes is within a few bits of imagehash's.
        (ours,) = self._pcv([pil_img])
        ours_hex = ours.hex()
        assert len(ours_hex) == len(hex_str) == 16
        assert all(c in "0123456789abcdef" for c in ours_hex)
        assert (int(ours_hex, 16) ^ int(hex_str, 16)).bit_count() <= 4

    def test_hash_from_hex_string(self, sample_rgb_image: np.ndarray) -> None:
        """Hash should be reconstructible from hex string."""
        import imagehash

        pil_img = numpy_to_pil(sample_rgb_image)
        hash_result = imagehash.phash(pil_img)

        hex_str = str(hash_result)
        reconstructed = imagehash.hex_to_hash(hex_str)

        assert hash_result == reconstructed, "Hash should roundtrip through hex string"

        # polars-cv's bytes load as an imagehash through their hex.
        (ours,) = self._pcv([pil_img])
        from_ours = imagehash.hex_to_hash(ours.hex())
        assert from_ours - hash_result <= 4

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

        # polars-cv's bytes are that packing (row-major, MSB first).
        (ours,) = self._pcv([pil_img], hash_size=64)
        ours_bits = np.unpackbits(np.frombuffer(ours, dtype=np.uint8))
        assert ours_bits.shape == unpacked.shape
        assert int((ours_bits != unpacked).sum()) <= 4

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

        ours_rgb, ours_gray = (self._int(h) for h in self._pcv([pil_rgb, pil_gray]))
        assert (ours_rgb ^ ours_gray).bit_count() <= 2
        assert (ours_rgb ^ self._ih(hash_rgb)).bit_count() <= 4
        assert (ours_gray ^ self._ih(hash_gray)).bit_count() <= 4

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

        # polars-cv's three algorithms: 64-bit hashes that differ from one
        # another as imagehash's do, and each is within a few bits of its
        # imagehash counterpart (a/d: measured 1-2 bits here; p: 10 bits on
        # this colour gradient, see the crop test).
        ours = {
            a: self._pcv([pil_img], a)[0]
            for a in ("average", "difference", "perceptual")
        }
        assert all(len(h) == 8 for h in ours.values())
        assert len(set(ours.values())) > 1

        def bits(h: bytes) -> np.ndarray:
            return np.unpackbits(np.frombuffer(h, np.uint8))

        assert int((bits(ours["average"]) != ahash.hash.ravel()).sum()) <= 4
        assert int((bits(ours["difference"]) != dhash.hash.ravel()).sum()) <= 4
        assert (self._int(ours["perceptual"]) ^ self._ih(phash)).bit_count() <= 12

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

        # polars-cv: the same batch, hashed twice, is identical, and row i is
        # within a few bits of imagehash's hash of image i (not of a neighbour).
        ours1, ours2 = self._pcv(images), self._pcv(images)
        assert ours1 == ours2
        for o, h in zip(ours1, hashes_run1, strict=True):
            assert (self._int(o) ^ self._ih(h)).bit_count() <= 4

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

        (ours,) = self._pcv([small_img])
        assert len(ours) == 8
        assert (self._int(ours) ^ self._ih(hash_result)).bit_count() <= 4

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

        # The time bound documents imagehash; polars-cv's hash of the same
        # image is compared (measured 2 bits).
        (ours,) = self._pcv([large_img])
        assert (self._int(ours) ^ self._ih(hash_result)).bit_count() <= 4


def _smooth_images(n: int, size: tuple[int, int] = (240, 320)) -> list[np.ndarray]:
    """Distinct natural-looking images: random low-frequency blobs in colour."""
    from PIL import Image, ImageFilter

    rng = np.random.default_rng(7)
    out = []
    for _ in range(n):
        small = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
        img = Image.fromarray(small).resize(size[::-1], Image.Resampling.BICUBIC)
        out.append(np.asarray(img.filter(ImageFilter.GaussianBlur(3))))
    return out


def _pcv_phash(arrays: list[np.ndarray], hash_size: int = 64) -> list[int]:
    from polars_cv import Pipeline

    df = pl.DataFrame({"img": [pil_to_png_bytes(numpy_to_pil(a)) for a in arrays]})
    pipe = Pipeline().source("image_bytes").perceptual_hash(hash_size=hash_size)
    rows = df.select(h=pl.col("img").cv.pipe(pipe).sink("list"))["h"].to_list()
    return [int.from_bytes(bytes(r), "big") for r in rows]


def pil_to_png_bytes(img: "PILImage.Image") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@imagehash_required
@plugin_required
@pytest.mark.parity(
    ops=["perceptual_hash"],
    against="imagehash `phash` (default 64-bit, and hash_size=16 for 256 bits)",
    tolerance="Hamming bits: <= 2 at 32x32, exact on flat/gradient, <= 4 resized, <= 40 of 256",
)
class TestPhashMatchesImagehash:
    """``perceptual_hash()`` (the default algorithm) is standard pHash.

    The same steps as ``imagehash.phash``: grayscale, a 32x32 Lanczos resize,
    a 2-D DCT-II, the 8x8 lowest frequencies thresholded at their median,
    bits row-major and most significant first. The grayscale and the resize
    are the engine's own ops, so the bits are within a few of imagehash's,
    not identical. It used to threshold a
    16x16-input DCT at its mean, which the DC term dominates: distinct images
    averaged ~18 differing bits instead of ~32, and on COCO val2017 236 of
    5,000 distinct images shared a hash with another.
    """

    def test_within_two_bits_at_32x32(self) -> None:
        """At 32x32 both resizes are the identity; only the gray rounding
        differs (the engine's BT.601 in 8-bit fixed point, Pillow's in 16).
        """
        import imagehash

        for a in _smooth_images(12, size=(32, 32)):
            expected = int(str(imagehash.phash(numpy_to_pil(a))), 16)
            got = _pcv_phash([a])[0]
            assert (got ^ expected).bit_count() <= 2

    def test_exact_on_degenerate_gray_images(self) -> None:
        """Gradients and flat images: most DCT coefficients are exactly 0.

        Their bits must not fall on rounding noise (they compare as 0, not
        above a median of 0), or two copies of one image differ widely. Gray
        input (R = G = B) rounds the same in both, so the bits are identical.
        """
        import imagehash

        x = np.linspace(0, 255, 32)
        gray = [
            np.tile(x, (32, 1)),  # horizontal gradient
            np.tile(x[:, None], (1, 32)),  # vertical gradient
            np.full((32, 32), 77.0),  # flat
        ]
        for g in gray:
            a = np.repeat(g.astype(np.uint8)[..., None], 3, axis=2)
            expected = int(str(imagehash.phash(numpy_to_pil(a))), 16)
            assert _pcv_phash([a]) == [expected]

    def test_close_to_imagehash_after_resizing(self) -> None:
        """Larger images: the two Lanczos resizes round differently."""
        import imagehash

        images = _smooth_images(12)
        ours = _pcv_phash(images)
        theirs = [int(str(imagehash.phash(numpy_to_pil(a))), 16) for a in images]
        dist = [(a ^ b).bit_count() for a, b in zip(ours, theirs, strict=True)]
        assert max(dist) <= 4, dist

    @pytest.mark.parity(
        ops=["perceptual_hash"],
        against="Definition: distinct images spread near 32 of 64 bits apart",
        tolerance="Hamming bits: mean > 26, min > 6",
    )
    def test_distinct_images_are_far_apart(self) -> None:
        """A useful hash spreads distinct images near 32 of 64 bits apart."""
        hashes = _pcv_phash(_smooth_images(40))
        dist = [
            (a ^ b).bit_count() for i, a in enumerate(hashes) for b in hashes[i + 1 :]
        ]
        assert np.mean(dist) > 26, np.mean(dist)
        assert min(dist) > 6, min(dist)

    def test_hash_size_256_is_imagehash_16(self) -> None:
        """256 bits is the 16x16 block. Its higher frequencies are small on a
        smooth image, so gray rounding flips more of them: about 10%.
        """
        import imagehash

        for a in _smooth_images(6, size=(64, 64)):
            expected = int(str(imagehash.phash(numpy_to_pil(a), hash_size=16)), 16)
            got = _pcv_phash([a], hash_size=256)[0]
            assert (got ^ expected).bit_count() <= 40


@imagehash_required
@plugin_required
@pytest.mark.parity(
    ops=["perceptual_hash"],
    against="imagehash `phash`, `average_hash` and `dhash`, bits read most significant first",
    tolerance="<= 2 Hamming bits for every algorithm",
)
def test_every_algorithm_packs_bits_msb_first() -> None:
    """The algorithms share one output column type, so one bit order: row-major,
    most significant bit first, as imagehash's hex. aHash and dHash used to
    come out least significant first (CR-115), 0 bits from imagehash read one
    way and tens of bits off read the documented way."""
    import imagehash
    from PIL import Image

    yy, xx = np.mgrid[0:128, 0:128]
    img = ((xx * 2 + yy) % 256).astype(np.uint8)
    img[40:90, 30:70] = 30
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    df = pl.DataFrame({"i": [buf.getvalue()]})
    refs = {
        "perceptual": imagehash.phash,
        "average": imagehash.average_hash,
        "difference": imagehash.dhash,
    }
    for algorithm, ref in refs.items():
        pipe = Pipeline().source("image_bytes").perceptual_hash(algorithm=algorithm)
        h = df.select(pl.col("i").cv.pipe(pipe).sink("list")).item().to_list()
        bits = np.unpackbits(np.array(h, dtype=np.uint8))
        expected = ref(Image.open(io.BytesIO(buf.getvalue()))).hash.flatten()
        assert int((bits != expected).sum()) <= 2, algorithm
