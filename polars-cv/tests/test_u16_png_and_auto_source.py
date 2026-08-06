"""Integration tests for bit-depth-preserving PNG encoding and source("auto").

Covers two related features:

* PNG sinks preserve each image's native precision — u8 -> 8-bit PNG,
  u16 -> 16-bit PNG — while the 8-bit-only formats (JPEG/WebP) raise a clear,
  actionable error for non-u8 input.
* ``source("auto")`` (now the default) routes to a concrete decode path from
  the column's Polars dtype at runtime.
"""

from __future__ import annotations

import io

import numpy as np
import polars as pl
import pytest

from polars_cv import Pipeline
from polars_cv._types import SourceFormat
from tests.conftest import make_test_png as create_test_png
from tests.conftest import plugin_required


def _make_u16_png(width: int = 4, height: int = 4) -> tuple[bytes, np.ndarray]:
    """Return (png_bytes, array) for a 16-bit grayscale PNG.

    The values span the full u16 range so any 8-bit truncation is detectable.
    """
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL/Pillow required for this test")

    rng = np.random.default_rng(1234)
    arr = rng.integers(0, 65536, size=(height, width), dtype=np.uint16)
    # Pin a couple of high values that would collapse under an 8-bit downscale.
    arr[0, 0] = 65535
    arr[0, 1] = 30000
    img = Image.fromarray(arr, mode="I;16")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), arr


def _decode_png_to_array(png_bytes: bytes) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(png_bytes)))


@plugin_required
class TestU16PngEncoding:
    """PNG sinks preserve native bit depth."""

    def test_u16_png_roundtrip_preserves_values(self) -> None:
        png_bytes, original = _make_u16_png()
        df = pl.DataFrame({"img": [png_bytes]})

        pipe = Pipeline().source("image_bytes")
        result = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("png"))

        out_bytes = result["out"][0]
        decoded = _decode_png_to_array(out_bytes)
        assert decoded.dtype == np.uint16
        np.testing.assert_array_equal(decoded, original)

    def test_mixed_u8_and_u16_batch(self) -> None:
        """A Binary column mixing an 8-bit and a 16-bit PNG sinks both rows."""
        u16_png, u16_arr = _make_u16_png()
        u8_png = create_test_png(4, 4, (200, 100, 50))
        df = pl.DataFrame({"img": [u8_png, u16_png]})

        pipe = Pipeline().source("image_bytes")
        result = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("png"))

        u8_out = _decode_png_to_array(result["out"][0])
        assert u8_out.dtype == np.uint8

        u16_out = _decode_png_to_array(result["out"][1])
        assert u16_out.dtype == np.uint16
        np.testing.assert_array_equal(u16_out, u16_arr)

    def test_jpeg_rejects_u16_with_actionable_error(self) -> None:
        u16_png, _ = _make_u16_png()
        df = pl.DataFrame({"img": [u16_png]})

        pipe = Pipeline().source("image_bytes")
        with pytest.raises(Exception, match="8-bit|cast"):
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("jpeg"))

    def test_webp_rejects_u16_with_actionable_error(self) -> None:
        u16_png, _ = _make_u16_png()
        df = pl.DataFrame({"img": [u16_png]})

        pipe = Pipeline().source("image_bytes")
        with pytest.raises(Exception, match="8-bit|cast"):
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("webp"))

    def test_jpeg_accepts_u16_after_cast(self) -> None:
        """Casting to u8 first lets the 8-bit encoder accept the image."""
        u16_png, _ = _make_u16_png()
        df = pl.DataFrame({"img": [u16_png]})

        pipe = Pipeline().source("image_bytes").cast("u8")
        result = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("jpeg"))
        assert result["out"].dtype == pl.Binary


@plugin_required
class TestAutoSource:
    """source("auto") routes by column dtype (and is the default)."""

    def test_auto_is_the_default_source_format(self) -> None:
        # source() with no format argument defaults to "auto".
        assert Pipeline().source()._source.format == SourceFormat.AUTO

    def test_auto_default_routes_binary_image(self) -> None:
        png = create_test_png(8, 8)
        df = pl.DataFrame({"img": [png]})

        # A bare source() (defaulting to "auto") and an explicit source("auto")
        # both route a Binary image column through the image decoder.
        default_pipe = Pipeline().source().resize(height=4, width=4)
        auto_pipe = Pipeline().source("auto").resize(height=4, width=4)

        for pipe in (default_pipe, auto_pipe):
            result = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("png"))
            decoded = _decode_png_to_array(result["out"][0])
            assert decoded.shape[:2] == (4, 4)

    def test_auto_binary_image_to_numpy(self) -> None:
        png = create_test_png(8, 8)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("auto").resize(height=4, width=4)
        result = df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert "out" in result.columns

    def test_auto_string_column_is_file_path(self, tmp_path) -> None:
        png = create_test_png(8, 8)
        path = tmp_path / "img.png"
        path.write_bytes(png)
        df = pl.DataFrame({"path": [str(path)]})

        pipe = Pipeline().source("auto").resize(height=4, width=4)
        result = df.with_columns(out=pl.col("path").cv.pipe(pipe).sink("png"))
        decoded = _decode_png_to_array(result["out"][0])
        assert decoded.shape[:2] == (4, 4)

    def test_auto_list_column(self) -> None:
        # A List column of pixel values behaves like source("list").
        data = [[1.0, 2.0, 3.0, 4.0]]
        df = pl.DataFrame({"vals": data}, schema={"vals": pl.List(pl.Float32)})

        auto = Pipeline().source("auto").reshape([2, 2])
        explicit = Pipeline().source("list").reshape([2, 2])

        auto_res = df.with_columns(out=pl.col("vals").cv.pipe(auto).sink("blob"))
        exp_res = df.with_columns(out=pl.col("vals").cv.pipe(explicit).sink("blob"))
        assert auto_res["out"][0] == exp_res["out"][0]

    def test_auto_array_column(self) -> None:
        df = pl.DataFrame(
            {"vals": [[1.0, 2.0, 3.0, 4.0]]},
            schema={"vals": pl.Array(pl.Float32, 4)},
        )

        auto = Pipeline().source("auto").reshape([2, 2])
        explicit = Pipeline().source("array").reshape([2, 2])

        auto_res = df.with_columns(out=pl.col("vals").cv.pipe(auto).sink("blob"))
        exp_res = df.with_columns(out=pl.col("vals").cv.pipe(explicit).sink("blob"))
        assert auto_res["out"][0] == exp_res["out"][0]

    def test_auto_list_sink_over_list_column(self) -> None:
        # A typed `list` sink over an auto source must plan (dtype/ndim resolve
        # from the List column at runtime), matching an explicit `list` source.
        df = pl.DataFrame(
            {"vals": [[1.0, 2.0, 3.0, 4.0]]},
            schema={"vals": pl.List(pl.Float32)},
        )
        auto = Pipeline().source("auto")
        explicit = Pipeline().source("list")

        auto_res = df.with_columns(out=pl.col("vals").cv.pipe(auto).sink("list"))
        exp_res = df.with_columns(out=pl.col("vals").cv.pipe(explicit).sink("list"))
        assert auto_res["out"].dtype == exp_res["out"].dtype
        assert auto_res["out"][0].to_list() == exp_res["out"][0].to_list()

    def test_auto_array_sink_over_array_column(self) -> None:
        df = pl.DataFrame(
            {"vals": [[1.0, 2.0, 3.0, 4.0]]},
            schema={"vals": pl.Array(pl.Float32, 4)},
        )
        auto = Pipeline().source("auto")
        explicit = Pipeline().source("array")

        # array sink needs a deterministic shape (independent of source format).
        auto_res = df.with_columns(
            out=pl.col("vals").cv.pipe(auto).sink("array", shape=[4])
        )
        exp_res = df.with_columns(
            out=pl.col("vals").cv.pipe(explicit).sink("array", shape=[4])
        )
        assert auto_res["out"].dtype == exp_res["out"].dtype

    def test_auto_image_list_sink_still_requires_dtype(self) -> None:
        # An auto source over a Binary image column cannot resolve the element
        # dtype at plan time, so a typed `list` sink must still error clearly —
        # deferred to Rust's schema resolution.
        png = create_test_png(4, 4)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("auto")
        with pytest.raises(Exception, match="dtype|auto"):
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("list"))

    def test_auto_image_list_sink_is_refused_even_with_an_explicit_dtype(self) -> None:
        # This test used to assert the opposite: that an explicit dtype "lets
        # the typed `list` sink plan over an auto image source (previously the
        # ndim guard rejected this)". The ndim guard was right and the widening
        # was the bug — an explicit dtype settles the *element type*, not the
        # *rank*, and a list sink's Polars dtype encodes both.
        #
        # It passed because it asserted only that the column came back, and it
        # ran eagerly, where nothing compares the plan against the result. In
        # lazy use the same pipeline published `List(UInt8)` and collected to
        # `List(List(List(UInt8)))`. `dtype_for_output` now refuses a list sink
        # whose rank it cannot name; the divergence is guarded from the other
        # side by tests/test_schema_parity_sources.py.
        png = create_test_png(4, 4)
        df = pl.DataFrame({"img": [png]})

        pipe = Pipeline().source("auto", dtype="u8")
        with pytest.raises(Exception, match="rank at planning time"):
            df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("list"))

        # The two documented ways out both work: name the source, so the rank
        # is known, or use a sink that does not encode the rank in its dtype.
        named = Pipeline().source("image_bytes", dtype="u8")
        lf = df.lazy().with_columns(out=pl.col("img").cv.pipe(named).sink("list"))
        assert lf.collect_schema()["out"] == lf.collect()["out"].dtype

        lf = df.lazy().with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
        assert lf.collect_schema()["out"] == lf.collect()["out"].dtype

    def test_auto_blob_roundtrip(self) -> None:
        """A VIEW-blob Binary column routes through the blob decoder."""
        png = create_test_png(8, 8)
        df = pl.DataFrame({"img": [png]})

        # Produce a blob column first, then re-ingest via source("auto").
        blob_pipe = Pipeline().source("image_bytes").resize(height=4, width=4)
        blobs = df.with_columns(blob=pl.col("img").cv.pipe(blob_pipe).sink("blob"))

        auto_pipe = Pipeline().source("auto")
        result = blobs.with_columns(out=pl.col("blob").cv.pipe(auto_pipe).sink("numpy"))
        assert "out" in result.columns

    def test_auto_unroutable_dtype_errors(self) -> None:
        """A plain numeric column cannot be routed and raises an actionable error."""
        df = pl.DataFrame({"nums": [1, 2, 3]})

        pipe = Pipeline().source("auto")
        with pytest.raises(Exception, match="auto source cannot infer"):
            df.with_columns(out=pl.col("nums").cv.pipe(pipe).sink("numpy"))
