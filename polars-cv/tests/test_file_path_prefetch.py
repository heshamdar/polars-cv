"""
Tests for remote file_path prefetching and cloud_options round-trip.

Remote `file_path` sources are fetched concurrently before the row loop
(per batch); these tests exercise that path hermetically with a local HTTP
server — no external network. The cloud_options round-trip test checks the
fix for the known gap where `SourceSpec.to_dict()` dropped credentials.
"""

from __future__ import annotations

import http.server
import io
import json
import threading

import numpy as np
import polars as pl
import pytest

from polars_cv import CloudOptions, Pipeline, numpy_from_struct
from tests.conftest import plugin_required


def _png(width: int, height: int, value: int) -> bytes:
    from PIL import Image

    arr = np.full((height, width, 3), value, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
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
    yield base
    server.shutdown()


class TestCloudOptionsRoundTrip:
    def test_cloud_options_serialized_into_graph_json(self) -> None:
        opts = CloudOptions(aws_region="eu-west-1", aws_access_key_id="AKID")
        pipe = Pipeline().source("file_path", cloud_options=opts).grayscale()
        spec = json.loads(pipe._to_json())
        assert spec["source"]["cloud_options"] == {
            "aws_region": "eu-west-1",
            "aws_access_key_id": "AKID",
        }

    def test_cloud_options_survive_graph_pipeline(self) -> None:
        # Through the full PipelineGraph serialization (the path the plugin
        # actually receives), not just the single-pipeline spec.
        from polars_cv._graph import PipelineGraph

        opts = CloudOptions(aws_region="eu-west-1", anonymous=True)
        pipe = Pipeline().source("file_path", cloud_options=opts).grayscale()
        graph = PipelineGraph()
        graph.add_node("n0", pipe, column=pl.col("paths"))
        graph.set_output("n0", "numpy")
        spec = json.loads(graph._to_json())
        assert spec["nodes"]["n0"]["source"]["cloud_options"] == {
            "aws_region": "eu-west-1",
            "anonymous": "true",
        }


@plugin_required
class TestRemotePrefetch:
    def test_http_sources_through_prefetch(self, http_image_server: str) -> None:
        urls = [f"{http_image_server}/img/{i}.png" for i in range(8)]
        df = pl.DataFrame({"paths": urls})
        pipe = Pipeline().source("file_path").grayscale()
        out = df.with_columns(out=pl.col("paths").cv.pipe(pipe).sink("numpy"))
        for i in range(8):
            arr = numpy_from_struct(out["out"][i])
            assert arr.shape == (4 + i, 4 + i, 1)

    def test_duplicate_urls_fetched_once_results_consistent(
        self, http_image_server: str
    ) -> None:
        url = f"{http_image_server}/img/3.png"
        df = pl.DataFrame({"paths": [url] * 20})
        pipe = Pipeline().source("file_path")
        out = df.with_columns(out=pl.col("paths").cv.pipe(pipe).sink("numpy"))
        ref = numpy_from_struct(out["out"][0])
        for i in range(20):
            np.testing.assert_array_equal(numpy_from_struct(out["out"][i]), ref)

    def test_failing_url_respects_source_on_error_null(
        self, http_image_server: str
    ) -> None:
        good = f"{http_image_server}/img/1.png"
        missing = f"{http_image_server}/img/does-not-exist.png"
        df = pl.DataFrame({"paths": [good, missing, good]})
        pipe = Pipeline().source("file_path", on_error="null").grayscale()
        out = df.with_columns(out=pl.col("paths").cv.pipe(pipe).sink("numpy"))
        assert out["out"][0]["data"] is not None
        assert out["out"][1]["data"] is None
        assert out["out"][2]["data"] is not None

    def test_failing_url_raises_by_default(self, http_image_server: str) -> None:
        df = pl.DataFrame({"paths": [f"{http_image_server}/img/nope.png"]})
        pipe = Pipeline().source("file_path")
        with pytest.raises(pl.exceptions.ComputeError, match="remote file"):
            df.with_columns(out=pl.col("paths").cv.pipe(pipe).sink("numpy"))

    def test_mixed_local_and_remote_paths(
        self, http_image_server: str, tmp_path
    ) -> None:
        local = tmp_path / "local.png"
        local.write_bytes(_png(6, 6, 200))
        df = pl.DataFrame({"paths": [str(local), f"{http_image_server}/img/2.png"]})
        pipe = Pipeline().source("file_path").grayscale()
        out = df.with_columns(out=pl.col("paths").cv.pipe(pipe).sink("numpy"))
        assert numpy_from_struct(out["out"][0]).shape == (6, 6, 1)
        assert numpy_from_struct(out["out"][1]).shape == (6, 6, 1)

    def test_streaming_engine_prefetches_per_morsel(
        self, http_image_server: str
    ) -> None:
        urls = [f"{http_image_server}/img/{i % 8}.png" for i in range(64)]
        df = pl.DataFrame({"paths": urls})
        pipe = Pipeline().source("file_path").grayscale()
        out = (
            df.lazy()
            .with_columns(out=pl.col("paths").cv.pipe(pipe).sink("numpy"))
            .collect(engine="streaming")
        )
        for i in (0, 31, 63):
            expected = 4 + (i % 8)
            assert numpy_from_struct(out["out"][i]).shape == (expected, expected, 1)
