"""``check_cloud_access`` against a store, offline: a minimal S3 server.

``check_cloud_access`` exercises the credential once (a ``HEAD`` of an
object, a one-level listing of a prefix) through the same store a read
builds. These run it against a local S3-compatible endpoint
(``aws_endpoint`` + ``aws_allow_http``; the server ignores signatures), so
``ok``, ``not_found`` and a prefix listing are checked without a network.
"""

from __future__ import annotations

import email.utils
import http.server
import threading
from urllib.parse import parse_qs, urlparse

import polars as pl
import pytest

import polars_cv

from .conftest import make_test_png, plugin_required

pytestmark = plugin_required

BUCKET = "bkt"


@pytest.fixture(scope="module")
def store():
    objects = {
        "imgs/a.png": make_test_png(7, 5),
        "imgs/b.png": make_test_png(3, 2),
        "empty/e.png": b"",
    }
    log: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def _key(self) -> str | None:
            path = urlparse(self.path).path.lstrip("/")
            bucket, _, key = path.partition("/")
            return key if bucket == BUCKET else None

        def _head(self, body: bytes) -> None:
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"0"')
            self.send_header("Last-Modified", email.utils.formatdate(usegmt=True))

        def do_HEAD(self):  # noqa: N802
            log.append(f"HEAD {self.path}")
            key = self._key()
            if key not in objects:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self._head(objects[key])
            self.end_headers()

        def do_GET(self):  # noqa: N802
            rng = self.headers.get("Range")
            log.append(f"GET {self.path}" + (f" {rng}" if rng else ""))
            query = parse_qs(urlparse(self.path).query)
            if "list-type" in query:
                prefix = query.get("prefix", [""])[0]
                keys = [k for k in sorted(objects) if k.startswith(prefix)]
                contents = "".join(
                    f"<Contents><Key>{k}</Key><Size>{len(objects[k])}</Size>"
                    "<LastModified>2026-01-01T00:00:00.000Z</LastModified>"
                    '<ETag>"0"</ETag></Contents>'
                    for k in keys
                )
                body = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    "<ListBucketResult><Name>bkt</Name>"
                    f"<Prefix>{prefix}</Prefix><KeyCount>{len(keys)}</KeyCount>"
                    f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>"
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            key = self._key()
            if key not in objects:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = objects[key]
            rng = self.headers.get("Range")
            if rng and not body:
                # S3: a range no byte of an empty object satisfies.
                error = b"<Error><Code>InvalidRange</Code></Error>"
                self.send_response(416)
                self.send_header("Content-Length", str(len(error)))
                self.end_headers()
                self.wfile.write(error)
                return
            if rng:
                start, end = (int(x) for x in rng.removeprefix("bytes=").split("-"))
                part = body[start : end + 1]
                self.send_response(206)
                self.send_header(
                    "Content-Range",
                    f"bytes {start}-{start + len(part) - 1}/{len(body)}",
                )
                self._head(part)
                self.end_headers()
                self.wfile.write(part)
                return
            self.send_response(200)
            self._head(body)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    options = {
        "aws_endpoint": f"http://127.0.0.1:{srv.server_address[1]}",
        "aws_allow_http": "true",
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
        "aws_region": "us-east-1",
    }
    yield options, log
    srv.shutdown()


def _check(path: str, options: dict) -> dict:
    return polars_cv.check_cloud_access(path, cloud_options=options)


def test_an_object_that_exists_is_ok(store) -> None:
    options, log = store
    log.clear()
    report = _check(f"s3://{BUCKET}/imgs/a.png", options)
    assert report["status"] == "ok", report
    assert report["backend"] == "aws"
    assert log == [f"HEAD /{BUCKET}/imgs/a.png"]


def test_a_missing_object_is_not_found(store) -> None:
    options, _ = store
    assert _check(f"s3://{BUCKET}/imgs/zz.png", options)["status"] == "not_found"


def test_a_prefix_is_listed_one_level(store) -> None:
    options, log = store
    log.clear()
    assert _check(f"s3://{BUCKET}/imgs/", options)["status"] == "ok"
    ((request,),) = [log]
    query = parse_qs(urlparse(request.split(" ", 1)[1]).query)
    assert query["delimiter"] == ["/"] and query["prefix"] == ["imgs/"], request


def test_a_prefix_with_nothing_under_it_is_not_found(store) -> None:
    """An empty listing is no such prefix: an object store has no
    directories, so a prefix exists only by the objects under it. It used to
    report ``ok``."""
    options, _ = store
    assert _check(f"s3://{BUCKET}/nothing/", options)["status"] == "not_found"


def test_a_read_through_the_same_options_agrees(store) -> None:
    """What the check reached is what a pipeline reads."""
    options, _ = store
    path = f"s3://{BUCKET}/imgs/a.png"
    assert _check(path, options)["status"] == "ok"
    info = pl.DataFrame({"p": [path]}).select(
        pl.col("p").cv.image_info(cloud_options=options, allowed_roots=["*"])
    )
    assert info.item()["width"] == 7


def test_a_header_read_takes_a_prefix(store) -> None:
    options, log = store
    log.clear()
    path = f"s3://{BUCKET}/imgs/a.png"
    info = pl.DataFrame({"p": [path]}).select(
        pl.col("p").cv.image_info(cloud_options=options, allowed_roots=["*"])
    )
    assert info.item()["height"] == 5
    assert log == [f"GET /{BUCKET}/imgs/a.png bytes=0-65535"], log


def test_an_empty_object_reads_as_no_header(store) -> None:
    """S3 refuses a range of an empty object (``InvalidRange``, which
    ``object_store`` reports as a generic error): read whole, it has no
    header, a null as an empty local file gives."""
    options, log = store
    log.clear()
    path = f"s3://{BUCKET}/empty/e.png"
    info = pl.DataFrame({"p": [path]}).select(
        pl.col("p").cv.image_info(cloud_options=options, allowed_roots=["*"])
    )
    assert info.item() is None
    assert log == [
        f"GET /{BUCKET}/empty/e.png bytes=0-65535",
        f"GET /{BUCKET}/empty/e.png",
    ], log
