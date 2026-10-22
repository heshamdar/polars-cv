"""Header-only metadata over a remote path reads a prefix, not the object.

``.cv.image_info()`` (and ``width``/``height``/...) on a remote path used to
fetch the whole object to read a header of a few hundred bytes. It now asks
for the first 64 KiB, as a local file is read, and only fetches more when the
header is not in it (a JPEG behind a large comment). A server that ignores
``Range`` and sends the whole body still works.
"""

from __future__ import annotations

import http.server
import io
import threading

import numpy as np
import polars as pl
import pytest
from PIL import Image

import polars_cv  # noqa: F401

from .conftest import plugin_required

pytestmark = plugin_required

PREFIX = 64 * 1024


def _noise_png(width: int, height: int) -> bytes:
    arr = np.random.default_rng(0).integers(0, 256, (height, width, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", compress_level=0)
    return buf.getvalue()


def _jpeg_behind_a_comment(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height)).save(buf, "JPEG", comment=b"x" * 60_000)
    body = buf.getvalue()
    # A second comment segment, so the frame header sits past the prefix.
    com = b"\xff\xfe" + (60_002).to_bytes(2, "big") + b"y" * 60_000
    return body[:2] + com + body[2:]


@pytest.fixture(scope="module")
def server():
    files = {
        "/big.png": _noise_png(600, 500),
        "/late.jpg": _jpeg_behind_a_comment(33, 21),
    }
    log: list[tuple[str, str | None, int]] = []
    honour_range = {"on": True}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = files[self.path]
            rng = self.headers.get("Range")
            if rng and honour_range["on"]:
                start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
                part = body[start : end + 1]
                self.send_response(206)
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{start + len(part) - 1}/{len(body)}",
                )
            else:
                part = body
                self.send_response(200)
            self.send_header("Content-Length", str(len(part)))
            self.end_headers()
            self.wfile.write(part)
            log.append((self.path, rng, len(part)))

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", files, log, honour_range
    srv.shutdown()


def _info(url: str) -> dict:
    return pl.DataFrame({"p": [url]}).select(pl.col("p").cv.image_info()).item()


def test_a_remote_header_reads_a_prefix(server) -> None:
    base, files, log, _ = server
    log.clear()
    assert _info(f"{base}/big.png")["width"] == 600
    assert len(files["/big.png"]) > 4 * PREFIX
    assert log == [("/big.png", f"bytes=0-{PREFIX - 1}", PREFIX)]


def test_a_header_past_the_prefix_reads_the_rest(server) -> None:
    base, _, log, _ = server
    log.clear()
    info = _info(f"{base}/late.jpg")
    assert (info["width"], info["height"]) == (33, 21)
    assert len(log) == 2, log
    assert log[0][1] == f"bytes=0-{PREFIX - 1}"


def test_a_server_that_ignores_range_still_answers(server) -> None:
    base, files, log, honour = server
    honour["on"] = False
    try:
        log.clear()
        assert _info(f"{base}/big.png")["height"] == 500
        assert log[0][2] == len(files["/big.png"])
    finally:
        honour["on"] = True


def test_a_pipeline_still_reads_the_whole_object(server) -> None:
    base, files, log, _ = server
    log.clear()
    pipe = polars_cv.Pipeline().source("file_path")
    out = pl.DataFrame({"p": [f"{base}/big.png"]}).select(
        pl.col("p").cv.pipe(pipe).sink("numpy")
    )
    assert polars_cv.numpy_from_struct(out.item()).shape == (500, 600, 3)
    assert log == [("/big.png", None, len(files["/big.png"]))]
