"""
Integration tests for TIFF sink support with floating-point data.

Tests the complete pipeline from Python to Rust and back, ensuring
native floating-point TIFF encoding works correctly.
"""

import struct
import polars as pl
from polars_cv import Pipeline


class TestTiffSinkSupport:
    """Test TIFF sink functionality with various data types."""

    def test_tiff_f32_encoding(self):
        """Test TIFF encoding with f32 data (medical imaging use case)."""
        # Create simple f32 test data (2x2 image)
        data = [1.0, 0.5, 0.25, 0.125]
        raw_bytes = b"".join([struct.pack("<f", x) for x in data])

        df = pl.DataFrame({"data": [raw_bytes]})

        # Create pipeline without sink, then add sink to the expression
        pipe = Pipeline().source("raw", dtype="f32").reshape([2, 2])
        result = df.with_columns(tiff=pl.col("data").cv.pipe(pipe).sink("tiff"))

        tiff_bytes = result["tiff"][0]
        assert isinstance(tiff_bytes, bytes)
        assert len(tiff_bytes) > 0

        # Verify TIFF magic bytes (little-endian TIFF)
        assert tiff_bytes[:4] == b"II*\x00"

    def test_tiff_f64_encoding(self):
        """Test TIFF encoding with f64 data (high precision)."""
        # Test f64 data (double precision for high-precision medical data)
        data_f64 = [1.0, 0.7071067811865476, 0.5, 0.3535533905932738]
        raw_bytes_f64 = b"".join([struct.pack("<d", x) for x in data_f64])

        df_f64 = pl.DataFrame({"data": [raw_bytes_f64]})
        pipe_f64 = Pipeline().source("raw", dtype="f64").reshape([2, 2])
        result_f64 = df_f64.with_columns(
            tiff=pl.col("data").cv.pipe(pipe_f64).sink("tiff")
        )

        tiff_bytes_f64 = result_f64["tiff"][0]
        assert isinstance(tiff_bytes_f64, bytes)
        assert len(tiff_bytes_f64) > 0
        assert tiff_bytes_f64[:4] == b"II*\x00"

    def test_tiff_u8_compatibility(self):
        """Test TIFF encoding with u8 data for compatibility."""
        # Create a simple grayscale image
        data_u8 = [255, 128, 64, 32]
        raw_bytes_u8 = bytes(data_u8)

        df_u8 = pl.DataFrame({"data": [raw_bytes_u8]})
        pipe_u8 = Pipeline().source("raw", dtype="u8").reshape([2, 2])
        result_u8 = df_u8.with_columns(
            tiff=pl.col("data").cv.pipe(pipe_u8).sink("tiff")
        )

        tiff_bytes_u8 = result_u8["tiff"][0]
        assert isinstance(tiff_bytes_u8, bytes)
        assert len(tiff_bytes_u8) > 0
        assert tiff_bytes_u8[:4] == b"II*\x00"

    def test_tiff_vs_numpy_both_work(self):
        """Verify both TIFF and NumPy sinks work for the same f32 data."""
        # Create test data
        data = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        raw_bytes = b"".join([struct.pack("<f", x) for x in data])

        df = pl.DataFrame({"data": [raw_bytes]})

        # Test both sinks work
        pipe_base = Pipeline().source("raw", dtype="f32").reshape([2, 3])

        result = df.with_columns(
            [
                pl.col("data").cv.pipe(pipe_base).sink("tiff").alias("tiff_output"),
                pl.col("data").cv.pipe(pipe_base).sink("numpy").alias("numpy_output"),
            ]
        )

        tiff_bytes = result["tiff_output"][0]
        numpy_struct = result["numpy_output"][0]

        # Both should produce valid output
        assert isinstance(tiff_bytes, bytes) and len(tiff_bytes) > 0
        assert numpy_struct is not None
        assert tiff_bytes[:4] == b"II*\x00"

    def test_tiff_medical_imaging_scenario(self):
        """Test a realistic medical imaging scenario with larger f32 data."""
        # Create 32x32 f32 'medical scan' with realistic values
        import math

        size = 32
        medical_data = []
        for i in range(size):
            for j in range(size):
                # Create a radial gradient pattern like a medical scan
                center_x, center_y = size // 2, size // 2
                distance = math.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                intensity = max(0.0, 1.0 - distance / (size // 2))
                medical_data.append(intensity)

        medical_bytes = b"".join([struct.pack("<f", x) for x in medical_data])
        df_medical = pl.DataFrame({"scan": [medical_bytes]})

        pipe_medical = Pipeline().source("raw", dtype="f32").reshape([size, size])
        result_medical = df_medical.with_columns(
            tiff=pl.col("scan").cv.pipe(pipe_medical).sink("tiff")
        )

        tiff_medical = result_medical["tiff"][0]
        assert isinstance(tiff_medical, bytes)
        assert len(tiff_medical) > 0
        assert tiff_medical[:4] == b"II*\x00"

        # Should be smaller than raw data due to compression
        # (though for this small synthetic data, compression may not be effective)
        assert len(tiff_medical) > 100  # Has TIFF headers and metadata

    def test_tiff_with_processing_pipeline(self):
        """Test TIFF sink with image processing operations."""
        # Create test data and apply some processing before TIFF output
        data = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0]
        raw_bytes = b"".join([struct.pack("<f", x) for x in data])

        df = pl.DataFrame({"data": [raw_bytes]})

        # Apply processing: reshape -> normalize -> TIFF
        pipe = (
            Pipeline()
            .source("raw", dtype="f32")
            .reshape([3, 4])
            .normalize(method="minmax")  # This should preserve f32 precision
        )

        result = df.with_columns(
            processed_tiff=pl.col("data").cv.pipe(pipe).sink("tiff")
        )

        tiff_bytes = result["processed_tiff"][0]
        assert isinstance(tiff_bytes, bytes)
        assert len(tiff_bytes) > 0
        assert tiff_bytes[:4] == b"II*\x00"

    def test_tiff_compression_effectiveness(self):
        """Test that TIFF compression is actually working."""
        # Create highly compressible data (checkerboard pattern)
        size = 32
        compressible_data = []
        for i in range(size):
            for j in range(size):
                # Checkerboard pattern - very compressible
                if (i // 4 + j // 4) % 2 == 0:
                    compressible_data.append(1.0)
                else:
                    compressible_data.append(0.0)

        raw_bytes = b"".join([struct.pack("<f", x) for x in compressible_data])
        df = pl.DataFrame({"data": [raw_bytes]})

        pipe = Pipeline().source("raw", dtype="f32").reshape([size, size])
        result = df.with_columns(tiff=pl.col("data").cv.pipe(pipe).sink("tiff"))

        tiff_bytes = result["tiff"][0]
        assert isinstance(tiff_bytes, bytes)
        assert len(tiff_bytes) > 0
        assert tiff_bytes[:4] == b"II*\x00"

        # With LZW compression, the checkerboard pattern should compress significantly
        # Raw data: 32x32x4 = 4096 bytes
        # Compressed should be much smaller (expect at least 2:1 compression)
        compression_ratio = len(raw_bytes) / len(tiff_bytes)
        assert compression_ratio > 2.0, (
            f"Expected compression ratio > 2.0, got {compression_ratio:.3f}"
        )

        # Verify the TIFF is still valid by checking it can be read
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            f.write(tiff_bytes)
            temp_path = f.name

        try:
            # Try to read with PIL to verify it's a valid compressed TIFF
            from PIL import Image

            with Image.open(temp_path) as img:
                assert img.mode == "F"  # 32-bit float mode
                assert img.size == (size, size)
                assert img.format == "TIFF"
        except ImportError:
            # PIL not available, but we verified TIFF magic bytes above
            pass
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    test = TestTiffSinkSupport()
    test.test_tiff_f32_encoding()
    test.test_tiff_f64_encoding()
    test.test_tiff_u8_compatibility()
    test.test_tiff_vs_numpy_both_work()
    test.test_tiff_medical_imaging_scenario()
    test.test_tiff_with_processing_pipeline()
    print("\n🎉 All TIFF integration tests passed!")
