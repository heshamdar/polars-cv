"""
Tests for `.cv.read_bytes()` — reading a path column's bytes without decoding.

`read_bytes` is the fetch half of the `file_path` source, exposed on its own
(both go through `crate::fetch`). The tests below pin the two properties that
motivate it — byte-identical passthrough and shared-mechanism parity with the
source — plus the usual null/error behaviour.

Remote coverage uses a local HTTP server, so nothing here needs the network.
"""

from __future__ import annotations

import http.server
import io
import threading
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from polars_cv import CloudOptions, Pipeline
from tests.conftest import plugin_required


def _png(width: int, height: int, value: int) -> bytes:
    from PIL import Image

    arr = np.full((height, width, 3), value, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg(width: int, height: int, value: int, quality: int = 92) -> bytes:
    from PIL import Image

    # A gradient, not a flat fill: flat images survive a JPEG round trip almost
    # exactly, which would weaken the passthrough assertion below.
    row = np.linspace(0, 255, width, dtype=np.uint8)
    arr = np.repeat(row[None, :], height, axis=0)
    arr = np.stack([arr, np.full_like(arr, value), arr[:, ::-1]], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@pytest.fixture(scope="module")
def http_image_server():
    """A local HTTP server serving deterministic PNGs at /img/<n>.png."""
    images = {f"/img/{i}.png": _png(4 + i, 4 + i, 50 + i) for i in range(8)}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            body = images.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence per-request logging
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, images
    server.shutdown()


# ============================================================
# Byte identity — the property that motivates the feature
# ============================================================


@plugin_required
class TestByteIdentity:
    def test_jpeg_round_trips_byte_for_byte(self, tmp_path: Path) -> None:
        """The exact JPEG file comes back; decoding + re-encoding does not."""
        original = _jpeg(64, 48, value=120)
        path = tmp_path / "gradient.jpg"
        path.write_bytes(original)

        df = pl.DataFrame({"path": [str(path)]})
        passthrough = df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"][0]
        assert passthrough == original

        # The contrast: decode -> re-encode cannot reproduce the file.
        pipe = Pipeline().source("file_path")
        reencoded = df.with_columns(out=pl.col("path").cv.pipe(pipe).sink("jpeg"))[
            "out"
        ][0]
        assert reencoded != original

    def test_png_round_trips_byte_for_byte(self, tmp_path: Path) -> None:
        original = _png(12, 9, value=77)
        path = tmp_path / "flat.png"
        path.write_bytes(original)

        df = pl.DataFrame({"path": [str(path)]})
        got = df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"][0]
        assert got == original

    def test_non_image_bytes_are_returned_untouched(self, tmp_path: Path) -> None:
        """Fetching does not care what the bytes are — no decode is attempted."""
        payload = bytes(range(256)) * 4
        path = tmp_path / "arbitrary.bin"
        path.write_bytes(payload)

        df = pl.DataFrame({"path": [str(path)]})
        assert df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"][0] == payload

    def test_file_uri_reads_the_same_file(self, tmp_path: Path) -> None:
        original = _png(8, 8, value=200)
        path = tmp_path / "uri.png"
        path.write_bytes(original)

        df = pl.DataFrame({"path": [f"file://{path}"]})
        assert df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"][0] == original


# ============================================================
# Remote reads (hermetic local HTTP server)
# ============================================================


@plugin_required
class TestRemoteReads:
    def test_http_bytes_are_identical_to_served_body(self, http_image_server) -> None:
        base, images = http_image_server
        paths = [f"{base}/img/{i}.png" for i in range(4)]
        df = pl.DataFrame({"path": paths})

        got = df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"].to_list()
        assert got == [images[f"/img/{i}.png"] for i in range(4)]

    def test_repeated_urls_resolve_consistently(self, http_image_server) -> None:
        """Distinct paths are deduped for fetching; every row still gets bytes."""
        base, images = http_image_server
        paths = [f"{base}/img/{i % 2}.png" for i in range(8)]
        df = pl.DataFrame({"path": paths})

        got = df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"].to_list()
        assert got == [images[f"/img/{i % 2}.png"] for i in range(8)]

    def test_mixed_local_and_remote_paths(
        self, http_image_server, tmp_path: Path
    ) -> None:
        base, images = http_image_server
        local = tmp_path / "local.png"
        local_bytes = _png(5, 5, value=11)
        local.write_bytes(local_bytes)

        df = pl.DataFrame({"path": [f"{base}/img/3.png", str(local)]})
        got = df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"].to_list()
        assert got == [images["/img/3.png"], local_bytes]

    def test_cloud_options_are_accepted_as_dataclass_and_dict(
        self, http_image_server
    ) -> None:
        """Credentials are inert for http:// but must cross the boundary cleanly."""
        base, images = http_image_server
        df = pl.DataFrame({"path": [f"{base}/img/1.png"]})
        expected = images["/img/1.png"]

        as_obj = df.with_columns(
            raw=pl.col("path").cv.read_bytes(
                cloud_options=CloudOptions(aws_region="eu-west-1")
            )
        )["raw"][0]
        as_dict = df.with_columns(
            raw=pl.col("path").cv.read_bytes(
                cloud_options={"aws_region": "eu-west-1", "aws_endpoint": "http://x"}
            )
        )["raw"][0]
        assert as_obj == expected
        assert as_dict == expected


# ============================================================
# Shared mechanism: read_bytes and file_path must not drift
# ============================================================


@plugin_required
class TestSharedMechanismWithFilePathSource:
    def test_missing_local_file_error_matches_the_source(self, tmp_path: Path) -> None:
        """Both paths report the failure through crate::fetch, so text matches."""
        missing = str(tmp_path / "does_not_exist.png")
        df = pl.DataFrame({"path": [missing]})

        with pytest.raises(Exception) as via_read_bytes:  # noqa: PT011
            df.with_columns(raw=pl.col("path").cv.read_bytes()).height

        pipe = Pipeline().source("file_path")
        with pytest.raises(Exception) as via_source:  # noqa: PT011
            df.with_columns(out=pl.col("path").cv.pipe(pipe).sink("png")).height

        marker = "Failed to read local file"
        assert marker in str(via_read_bytes.value)
        assert marker in str(via_source.value)
        assert missing in str(via_read_bytes.value)

    def test_remote_error_text_matches_the_source(self, http_image_server) -> None:
        base, _ = http_image_server
        url = f"{base}/img/missing.png"
        df = pl.DataFrame({"path": [url]})

        with pytest.raises(Exception) as via_read_bytes:  # noqa: PT011
            df.with_columns(raw=pl.col("path").cv.read_bytes()).height

        pipe = Pipeline().source("file_path")
        with pytest.raises(Exception) as via_source:  # noqa: PT011
            df.with_columns(out=pl.col("path").cv.pipe(pipe).sink("png")).height

        marker = "Failed to read remote file"
        assert marker in str(via_read_bytes.value)
        assert marker in str(via_source.value)

    def test_bytes_feed_the_pipeline_identically_to_file_path(
        self, tmp_path: Path
    ) -> None:
        """read_bytes -> image_bytes source == file_path source, same pixels."""
        path = tmp_path / "img.png"
        path.write_bytes(_png(16, 10, value=90))
        df = pl.DataFrame({"path": [str(path)]})

        via_path = df.with_columns(
            out=pl.col("path").cv.pipe(Pipeline().source("file_path")).sink("png")
        )["out"][0]
        via_bytes = df.with_columns(raw=pl.col("path").cv.read_bytes()).with_columns(
            out=pl.col("raw").cv.pipe(Pipeline().source("image_bytes")).sink("png")
        )["out"][0]
        assert via_path == via_bytes


# ============================================================
# Composition with the header-only metadata family
# ============================================================


@plugin_required
class TestCompositionWithMetadata:
    def test_metadata_over_a_path_column(self, tmp_path: Path) -> None:
        """The gap this closes: header metadata for files named by path."""
        path = tmp_path / "wide.png"
        path.write_bytes(_png(37, 11, value=5))
        df = pl.DataFrame({"path": [str(path)]})

        out = df.with_columns(raw=pl.col("path").cv.read_bytes()).with_columns(
            w=pl.col("raw").cv.width(),
            h=pl.col("raw").cv.height(),
        )
        assert out["w"][0] == 37
        assert out["h"][0] == 11

    def test_filter_then_decode_only_survivors(self, tmp_path: Path) -> None:
        sizes = [(40, 40), (8, 8), (60, 60), (4, 4)]
        paths = []
        for i, (w, h) in enumerate(sizes):
            p = tmp_path / f"i{i}.png"
            p.write_bytes(_png(w, h, value=100 + i))
            paths.append(str(p))

        df = pl.DataFrame({"path": paths})
        pipe = Pipeline().source("image_bytes").resize(height=4, width=4)
        result = (
            df.with_columns(raw=pl.col("path").cv.read_bytes())
            .filter(pl.col("raw").cv.width() > 20)
            .with_columns(thumb=pl.col("raw").cv.pipe(pipe).sink("png"))
        )
        # Only the two large images survive, and the originals are still intact.
        assert result.height == 2
        assert result["raw"].to_list() == [
            Path(paths[0]).read_bytes(),
            Path(paths[2]).read_bytes(),
        ]
        assert all(t is not None for t in result["thumb"].to_list())


# ============================================================
# Streaming / morsel splitting
# ============================================================


@plugin_required
class TestStreaming:
    def test_streaming_and_in_memory_agree(self, tmp_path: Path) -> None:
        """The fetch is per call, so splitting the column must not change it."""
        paths = []
        for i in range(24):
            p = tmp_path / f"m{i}.png"
            p.write_bytes(_png(6, 6, value=i))
            paths.append(str(p))

        lf = pl.LazyFrame({"path": paths}).with_columns(
            raw=pl.col("path").cv.read_bytes()
        )
        eager = lf.collect()["raw"].to_list()
        streamed = lf.collect(engine="streaming")["raw"].to_list()

        assert eager == streamed
        assert eager == [Path(p).read_bytes() for p in paths]


# ============================================================
# Nulls and error policy
# ============================================================


@plugin_required
class TestErrorHandling:
    def test_null_paths_yield_null_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "present.png"
        path.write_bytes(_png(6, 6, value=42))
        df = pl.DataFrame({"path": [str(path), None]})

        got = df.with_columns(raw=pl.col("path").cv.read_bytes())["raw"].to_list()
        assert got == [path.read_bytes(), None]

    def test_on_error_null_isolates_the_failing_row(self, tmp_path: Path) -> None:
        good = tmp_path / "good.png"
        good.write_bytes(_png(6, 6, value=42))
        missing = str(tmp_path / "gone.png")
        df = pl.DataFrame({"path": [str(good), missing]})

        got = df.with_columns(raw=pl.col("path").cv.read_bytes(on_error="null"))[
            "raw"
        ].to_list()
        assert got == [good.read_bytes(), None]

    def test_on_error_raise_is_the_default(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"path": [str(tmp_path / "gone.png")]})
        with pytest.raises(Exception):  # noqa: B017, PT011
            df.with_columns(raw=pl.col("path").cv.read_bytes()).height

    def test_all_null_column_yields_all_null_binary(self) -> None:
        df = pl.DataFrame({"path": [None, None]}, schema={"path": pl.Null})
        out = df.with_columns(raw=pl.col("path").cv.read_bytes())
        assert out["raw"].to_list() == [None, None]
        assert out.schema["raw"] == pl.Binary

    def test_non_string_column_is_rejected(self) -> None:
        df = pl.DataFrame({"path": [1, 2, 3]})
        with pytest.raises(Exception) as exc:  # noqa: PT011
            df.with_columns(raw=pl.col("path").cv.read_bytes()).height
        assert "String column" in str(exc.value)


class TestPlanTimeValidation:
    """No plugin needed: these fail while the expression is being built."""

    def test_invalid_on_error_rejected_before_execution(self) -> None:
        with pytest.raises(ValueError, match="Unknown on_error value"):
            pl.col("path").cv.read_bytes(on_error="skip")

    def test_invalid_cloud_options_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="cloud_options must be"):
            pl.col("path").cv.read_bytes(cloud_options=["not", "options"])

    def test_output_dtype_is_binary_at_plan_time(self) -> None:
        lf = pl.LazyFrame({"path": ["a.png"]}).with_columns(
            raw=pl.col("path").cv.read_bytes()
        )
        assert lf.collect_schema()["raw"] == pl.Binary
