"""A crop (or a level) of a TIFF named by path reads only what it decodes.

A ``file_path`` source used to read the whole file for every row, so one
256-pixel patch of a 2 GB slide read 2 GB. For a TIFF, a crop right after the
source (or a pyramid level above 0) now opens the file and reads its IFDs and
the chunks the window overlaps; every other file is read whole, as before.

Measured where it matters, at the user-facing call: ``_lib._fetch_bytes_read``
counts every byte the plugin's path reads take from a file or a store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

import polars_cv._lib as lib
from polars_cv import OptFlags, Pipeline, numpy_from_struct
from polars_cv._optimize import PASS_NAMES
from tests.conftest import (
    TiffFixture,
    make_image_png,
    plugin_required,
    write_tiled_tiff,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

tifffile = pytest.importorskip("tifffile")

_ON = OptFlags.all()
_OFF = OptFlags(**{**{n: True for n in PASS_NAMES}, "roi_decode": False})


def _measured(run: Callable[[], pl.DataFrame]) -> tuple[pl.DataFrame, int]:
    """``run()`` and the bytes the plugin read from files and stores for it."""
    before = lib._fetch_bytes_read()
    out = run()
    return out, lib._fetch_bytes_read() - before


def _slide(tmp_path: Path, **kwargs: object) -> TiffFixture:
    """A 1024x1024 uncompressed slide in 64-pixel tiles: 3 MiB of pixels."""
    options: dict[str, object] = {"height": 1024, "width": 1024, "tile": (64, 64)}
    options.update(kwargs)
    return write_tiled_tiff(tmp_path / "slide.tif", **options)  # type: ignore[arg-type]


def _crop(df: pl.DataFrame, flags: OptFlags = _ON, **source: object) -> pl.DataFrame:
    pipe = (
        Pipeline()
        .source("file_path", **source)  # type: ignore[arg-type]
        .crop(top=pl.col("t"), left=pl.col("l"), height=64, width=64)
    )
    return df.select(o=pl.col("p").cv.pipe(pipe).sink("numpy", opt_flags=flags))


@plugin_required
class TestLocalFiles:
    def test_a_window_reads_a_small_part_of_the_file(self, tmp_path: Path) -> None:
        """Four 64x64 windows of a 3 MiB slide read under 5% of it — and
        decode exactly the written pixels."""
        fx = _slide(tmp_path)
        size = fx.path.stat().st_size
        origins = [(0, 0), (100, 200), (512, 960), (960, 0)]
        df = pl.DataFrame(
            {
                "p": [str(fx.path)] * 4,
                "t": [o[0] for o in origins],
                "l": [o[1] for o in origins],
            }
        )
        out, read = _measured(lambda: _crop(df))
        assert read < 0.05 * size, (read, size)
        for (t, left), row in zip(origins, out["o"], strict=True):
            np.testing.assert_array_equal(
                numpy_from_struct(row), fx.levels[0][t : t + 64, left : left + 64]
            )

    def test_without_the_pass_the_whole_file_is_read(self, tmp_path: Path) -> None:
        """The measurement is real: a whole-image decode reads the file."""
        fx = _slide(tmp_path)
        df = pl.DataFrame({"p": [str(fx.path)], "t": [0], "l": [0]})
        _, read = _measured(lambda: _crop(df, _OFF))
        assert read >= fx.path.stat().st_size

    def test_a_level_reads_its_own_tiles(self, tmp_path: Path) -> None:
        """Level 2 of a 3-level pyramid (1/16 of the pixels) reads well under
        level 0's share of the file."""
        fx = _slide(tmp_path, levels=3)
        size = fx.path.stat().st_size
        df = pl.DataFrame({"p": [str(fx.path)]})
        pipe = Pipeline().source("file_path", level=2)
        out, read = _measured(
            lambda: df.select(o=pl.col("p").cv.pipe(pipe).sink("numpy"))
        )
        np.testing.assert_array_equal(numpy_from_struct(out["o"][0]), fx.levels[2])
        assert read < 0.15 * size, (read, size)

    def test_any_other_file_is_read_whole(self, tmp_path: Path) -> None:
        """A PNG has no chunks to pick: it is read whole and cropped."""
        path = tmp_path / "a.png"
        path.write_bytes(make_image_png(80, 80, seed=2))
        df = pl.DataFrame({"p": [str(path)], "t": [8], "l": [8]})
        out, read = _measured(lambda: _crop(df))
        assert read >= path.stat().st_size
        assert numpy_from_struct(out["o"][0]).shape == (64, 64, 3)

    def test_the_sandbox_still_applies(self, tmp_path: Path) -> None:
        """``allowed_roots`` refuses a path outside it on the ranged read too."""
        fx = _slide(tmp_path)
        df = pl.DataFrame({"p": [str(fx.path)], "t": [0], "l": [0]})
        elsewhere = str(tmp_path / "elsewhere")
        with pytest.raises(pl.exceptions.ComputeError, match="allowed"):
            _crop(df, allowed_roots=[elsewhere])
        assert _crop(df, allowed_roots=[elsewhere], on_error="null")["o"].to_list() == [
            None
        ]

    def test_a_missing_file_fails_as_before(self, tmp_path: Path) -> None:
        df = pl.DataFrame({"p": [str(tmp_path / "absent.tif")], "t": [0], "l": [0]})
        for flags in (_ON, _OFF):
            with pytest.raises(pl.exceptions.ComputeError, match="absent.tif"):
                _crop(df, flags)
            assert _crop(df, flags, on_error="null")["o"].to_list() == [None]

    def test_a_layout_the_chunk_reader_leaves_falls_back_whole(
        self, tmp_path: Path
    ) -> None:
        """A TIFF layout only the ``tiff`` crate decodes (floats under the
        floating-point predictor) is read whole and cropped, with the same
        pixels as without the pass."""
        path = tmp_path / "fp.tif"
        rng = np.random.default_rng(0)
        data = rng.random((96, 96), dtype=np.float32)
        tifffile.imwrite(path, data, tile=(16, 16), compression="zlib", predictor=3)
        df = pl.DataFrame({"p": [str(path)], "t": [8], "l": [8]})
        on, read = _measured(lambda: _crop(df))
        off = _crop(df, _OFF)
        assert on.equals(off)
        np.testing.assert_array_equal(
            numpy_from_struct(on["o"][0])[..., 0], data[8:72, 8:72]
        )
        assert read >= path.stat().st_size


class _Server:
    """A loopback HTTP server over a directory, logging every request.

    It answers ``HEAD`` and ranged ``GET`` (206) like a real object store's
    HTTP front, unless ``honour_ranges`` is off, when it sends the whole body
    (200) whatever was asked, as some servers do.
    """

    def __init__(
        self,
        root: Path,
        *,
        honour_ranges: bool = True,
        latency: float = 0.0,
        etag: bool = False,
    ) -> None:
        import hashlib
        import http.server
        import threading
        import time

        self.requests: list[tuple[str, str, str | None]] = []
        self.served = 0
        #: The most requests in flight at once.
        self.peak = 0
        in_flight = 0
        lock = threading.Lock()
        outer = self

        def enter() -> None:
            nonlocal in_flight
            with lock:
                in_flight += 1
                outer.peak = max(outer.peak, in_flight)
            time.sleep(latency)

        def leave() -> None:
            nonlocal in_flight
            with lock:
                in_flight -= 1

        def validator(handler: http.server.BaseHTTPRequestHandler, body: bytes) -> None:
            if etag:
                handler.send_header("ETag", f'"{hashlib.sha1(body).hexdigest()}"')

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _body(self) -> bytes | None:
                path = root / self.path.lstrip("/")
                return path.read_bytes() if path.is_file() else None

            def do_HEAD(self) -> None:  # noqa: N802
                enter()
                try:
                    self._head()
                finally:
                    leave()

            def _head(self) -> None:
                body = self._body()
                outer.requests.append(("HEAD", self.path, None))
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                validator(self, body)
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                enter()
                try:
                    self._get()
                finally:
                    leave()

            def _get(self) -> None:
                body = self._body()
                asked = self.headers.get("Range")
                outer.requests.append(("GET", self.path, asked))
                if body is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if asked and honour_ranges:
                    first, last = asked.removeprefix("bytes=").split("-")
                    start, end = int(first), min(int(last) + 1, len(body))
                    part = body[start:end]
                    self.send_response(206)
                    self.send_header(
                        "Content-Range", f"bytes {start}-{end - 1}/{len(body)}"
                    )
                else:
                    part = body
                    self.send_response(200)
                self.send_header("Content-Length", str(len(part)))
                validator(self, body)
                self.end_headers()
                self.wfile.write(part)
                outer.served += len(part)

            def log_message(self, *args: object) -> None:
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def server(tmp_path: Path):  # type: ignore[no-untyped-def]
    srv = _Server(tmp_path)
    yield srv
    srv.close()


@plugin_required
class TestRemoteFiles:
    def test_a_window_reads_ranges_of_the_object(
        self, tmp_path: Path, server: _Server
    ) -> None:
        """Windows of a remote slide decode exactly, from ranged requests
        totalling a small part of it; no request fetches it whole."""
        fx = _slide(tmp_path)
        size = fx.path.stat().st_size
        url = f"{server.base}/{fx.path.name}"
        origins = [(0, 0), (100, 200), (512, 960)]
        df = pl.DataFrame(
            {"p": [url] * 3, "t": [o[0] for o in origins], "l": [o[1] for o in origins]}
        )
        out, read = _measured(lambda: _crop(df))
        for (t, left), row in zip(origins, out["o"], strict=True):
            np.testing.assert_array_equal(
                numpy_from_struct(row), fx.levels[0][t : t + 64, left : left + 64]
            )
        gets = [r for r in server.requests if r[0] == "GET"]
        assert gets and all(asked is not None for _, _, asked in gets), server.requests
        assert server.served < 0.10 * size, (server.served, size)
        assert read == server.served

    def test_without_the_pass_the_object_is_fetched_whole(
        self, tmp_path: Path, server: _Server
    ) -> None:
        fx = _slide(tmp_path)
        df = pl.DataFrame({"p": [f"{server.base}/{fx.path.name}"], "t": [0], "l": [0]})
        _crop(df, _OFF)
        assert server.requests == [("GET", f"/{fx.path.name}", None)]

    def test_a_level_reads_ranges_too(self, tmp_path: Path, server: _Server) -> None:
        fx = _slide(tmp_path, levels=3)
        df = pl.DataFrame({"p": [f"{server.base}/{fx.path.name}"]})
        pipe = Pipeline().source("file_path", level=2)
        out = df.select(o=pl.col("p").cv.pipe(pipe).sink("numpy"))
        np.testing.assert_array_equal(numpy_from_struct(out["o"][0]), fx.levels[2])
        assert server.served < 0.15 * fx.path.stat().st_size

    def test_a_server_ignoring_ranges_still_gives_the_window(
        self, tmp_path: Path
    ) -> None:
        """A 200 to a ranged request is the whole body: the ranges are cut
        from it, and the pixels are right."""
        fx = _slide(tmp_path)
        srv = _Server(tmp_path, honour_ranges=False)
        try:
            df = pl.DataFrame(
                {"p": [f"{srv.base}/{fx.path.name}"], "t": [64], "l": [128]}
            )
            out = _crop(df)
        finally:
            srv.close()
        np.testing.assert_array_equal(
            numpy_from_struct(out["o"][0]), fx.levels[0][64:128, 128:192]
        )

    def test_any_other_remote_file_is_fetched_whole(
        self, tmp_path: Path, server: _Server
    ) -> None:
        path = tmp_path / "a.png"
        path.write_bytes(make_image_png(80, 80, seed=2))
        df = pl.DataFrame({"p": [f"{server.base}/a.png"], "t": [8], "l": [8]})
        out = _crop(df)
        assert numpy_from_struct(out["o"][0]).shape == (64, 64, 3)
        assert ("GET", "/a.png", None) in server.requests

    def test_the_sandbox_refuses_before_any_request(
        self, tmp_path: Path, server: _Server
    ) -> None:
        fx = _slide(tmp_path)
        df = pl.DataFrame({"p": [f"{server.base}/{fx.path.name}"], "t": [0], "l": [0]})
        assert _crop(df, allowed_roots=["https://elsewhere/"], on_error="null")[
            "o"
        ].to_list() == [None]
        assert server.requests == []

    def test_a_missing_object_is_the_rows_error(
        self, tmp_path: Path, server: _Server
    ) -> None:
        df = pl.DataFrame({"p": [f"{server.base}/absent.tif"], "t": [0], "l": [0]})
        with pytest.raises(pl.exceptions.ComputeError, match="404"):
            _crop(df)
        assert _crop(df, on_error="null")["o"].to_list() == [None]

    def test_slide_info_reads_the_header_not_the_object(
        self, tmp_path: Path, server: _Server
    ) -> None:
        """A remote slide's pyramid comes from ranged reads of its IFDs, and
        matches the local file's."""
        fx = _slide(tmp_path, levels=3, svs_extras=True)
        remote = pl.DataFrame({"p": [f"{server.base}/{fx.path.name}"]})
        local = pl.DataFrame({"p": [str(fx.path)]})
        info = remote.select(pl.col("p").cv.slide_info())["p"].to_list()
        assert info == local.select(pl.col("p").cv.slide_info())["p"].to_list()
        assert len(info[0]["levels"]) == 3
        assert server.served < 0.10 * fx.path.stat().st_size, server.served


def _window(
    path: str, top: int | None, left: int, size: int, **source: object
) -> pl.DataFrame:
    """One ``size``-pixel window of ``path`` (``top`` may be null), under
    ``source(**source)``; ``on_null_param`` is taken from ``source`` too."""
    on_null = source.pop("on_null_param", "raise")
    df = pl.DataFrame(
        {"p": [path], "t": [top], "l": [left]},
        schema={"p": pl.String, "t": pl.UInt32, "l": pl.UInt32},
    )
    pipe = (
        Pipeline()
        .source("file_path", **source)  # type: ignore[arg-type]
        .on_null_param(on_null)  # type: ignore[arg-type]
        .crop(top=pl.col("t"), left=pl.col("l"), height=size, width=size)
    )
    return df.select(o=pl.col("p").cv.pipe(pipe).sink("numpy"))


def _huge_slide(path: Path, side: int = 10240) -> Path:
    """A slide whose level 0 decodes to more than the decode limit (300 MiB
    at the default side) but whose file is small: constant 256-pixel RGB
    tiles, Deflate-compressed, written one tile at a time."""
    tile = np.full((256, 256, 3), 200, np.uint8)
    n = (side // 256) ** 2
    tifffile.imwrite(
        path,
        (tile for _ in range(n)),
        shape=(side, side, 3),
        dtype=np.uint8,
        tile=(256, 256),
        photometric="rgb",
        compression="zlib",
    )
    return path


@plugin_required
class TestWindowCost:
    """A window costs its header, its tiles' entries in the chunk tables and
    its tiles: never the whole tile index, so a patch of a large slide costs
    what a patch of a small one does."""

    @pytest.mark.parametrize("level", [0, 2])
    def test_a_window_costs_the_same_whatever_the_slide_size(
        self, tmp_path: Path, level: int
    ) -> None:
        """The large slide's level-0 tile index alone (65,536 tiles: 512 KiB)
        is many times what a window of the small one reads in all."""
        reads = {}
        for name, side in (("small", 512), ("large", 4096)):
            fx = write_tiled_tiff(
                tmp_path / f"{name}.tif",
                height=side,
                width=side,
                channels=1,
                tile=(16, 16),
                levels=3,
            )
            out, reads[name] = _measured(
                lambda fx=fx: _window(str(fx.path), 40, 24, 16, level=level)
            )
            np.testing.assert_array_equal(
                numpy_from_struct(out["o"][0]), fx.levels[level][40:56, 24:40]
            )
        assert reads["large"] <= reads["small"] + 16 * 1024, reads
        assert reads["large"] < 64 * 1024, reads

    def test_a_remote_window_costs_the_same_whatever_the_slide_size(
        self, tmp_path: Path, server: _Server
    ) -> None:
        """Remote, the index is not fetched either: the large slide serves
        a few structure blocks and the window's tiles."""
        fx = write_tiled_tiff(
            tmp_path / "large.tif", height=4096, width=4096, channels=1, tile=(16, 16)
        )
        out = _window(f"{server.base}/{fx.path.name}", 2000, 3000, 16)
        np.testing.assert_array_equal(
            numpy_from_struct(out["o"][0]), fx.levels[0][2000:2016, 3000:3016]
        )
        assert server.served < 320 * 1024, server.served


@plugin_required
class TestRefusedWindows:
    """A window that does not resolve, or that the crop refuses, decodes no
    pixels: the row fails, or nulls, exactly as the crop would after a whole
    decode — at the cost of the header."""

    def test_a_null_window_parameter_nulls_the_row_unread(self, tmp_path: Path) -> None:
        fx = _slide(tmp_path)
        size = fx.path.stat().st_size
        out, read = _measured(
            lambda: _window(str(fx.path), None, 0, 64, on_null_param="null")
        )
        assert out["o"].to_list() == [None]
        assert read < 0.05 * size, (read, size)

    def test_a_null_window_parameter_raises_unread(self, tmp_path: Path) -> None:
        fx = _slide(tmp_path)
        size = fx.path.stat().st_size
        before = lib._fetch_bytes_read()
        with pytest.raises(pl.exceptions.ComputeError, match="null value"):
            _window(str(fx.path), None, 0, 64)
        assert lib._fetch_bytes_read() - before < 0.05 * size

    def test_a_null_window_parameter_of_a_null_path_is_null(self) -> None:
        """A null input is null before any parameter is read, as without
        the pass."""
        df = pl.DataFrame(
            {"p": [None], "t": [None], "l": [0]},
            schema={"p": pl.String, "t": pl.UInt32, "l": pl.UInt32},
        )
        pipe = (
            Pipeline()
            .source("file_path")
            .crop(top=pl.col("t"), left=pl.col("l"), height=4, width=4)
        )
        for flags in (_ON, _OFF):
            out = df.select(o=pl.col("p").cv.pipe(pipe).sink("numpy", opt_flags=flags))
            assert out["o"].to_list() == [None]

    def test_a_window_outside_a_slide_is_the_crops_error(self, tmp_path: Path) -> None:
        """Outside a slide too large to decode whole, the row fails with the
        crop's own message (not the decode limit's), having read the header."""
        path = _huge_slide(tmp_path / "huge.tif")
        before = lib._fetch_bytes_read()
        with pytest.raises(
            pl.exceptions.ComputeError, match=r"Crop: .*lies outside the input"
        ):
            _window(str(path), 0, 10200, 64)
        assert lib._fetch_bytes_read() - before < 64 * 1024

    def test_a_window_outside_the_image_reads_only_the_header(
        self, tmp_path: Path
    ) -> None:
        """The same message as without the pass, without decoding the file."""
        fx = _slide(tmp_path)
        size = fx.path.stat().st_size
        df = pl.DataFrame({"p": [str(fx.path)], "t": [1000], "l": [0]})
        messages, reads = [], []
        for flags in (_ON, _OFF):
            before = lib._fetch_bytes_read()
            with pytest.raises(pl.exceptions.ComputeError) as excinfo:
                _crop(df, flags)
            reads.append(lib._fetch_bytes_read() - before)
            messages.append(str(excinfo.value))
        assert messages[0] == messages[1]
        assert "lies outside" in messages[0]
        assert reads[0] < 0.05 * size, reads


@plugin_required
class TestNoSilentWholeReads:
    """Where a ranged read cannot serve a TIFF, the file is read whole only
    when that is a read a whole decode could use; otherwise the row fails
    naming why, without reading it."""

    def test_an_unsupported_layout_too_large_to_read_whole_fails_unread(
        self, tmp_path: Path
    ) -> None:
        """Zstandard tiles in a file over the whole-read limit: the error
        names the compression, and the file is not read."""
        path = tmp_path / "z.tif"
        data = np.zeros((64, 64), np.uint8)
        tifffile.imwrite(path, data, tile=(16, 16), compression="zstd")
        with path.open("r+b") as f:
            f.truncate(300 * 1024 * 1024)  # sparse padding past the limit
        before = lib._fetch_bytes_read()
        with pytest.raises(pl.exceptions.ComputeError, match="(?i)compression 50000"):
            _window(str(path), 0, 0, 8)
        assert lib._fetch_bytes_read() - before < 1024 * 1024

    def test_slide_info_of_an_unreadable_tiff_reads_only_its_header(
        self, tmp_path: Path
    ) -> None:
        """A TIFF whose first IFD lies past its end is no image (null, as its
        bytes give) — found from the header, not by reading the file."""
        path = tmp_path / "broken.tif"
        path.write_bytes(b"II*\x00" + (1 << 30).to_bytes(4, "little"))
        with path.open("r+b") as f:
            f.truncate(64 * 1024 * 1024)
        before = lib._fetch_bytes_read()
        df = pl.DataFrame({"p": [str(path)]})
        assert df.select(pl.col("p").cv.slide_info())["p"].to_list() == [None]
        assert lib._fetch_bytes_read() - before < 1024 * 1024

    def test_a_server_ignoring_ranges_is_read_once(self, tmp_path: Path) -> None:
        """A server that answers every ranged request with the whole body is
        downloaded once per call, not once per block or window."""
        fx = _slide(tmp_path, levels=2)
        size = fx.path.stat().st_size
        srv = _Server(tmp_path, honour_ranges=False)
        try:
            df = pl.DataFrame(
                {
                    "p": [f"{srv.base}/{fx.path.name}"] * 3,
                    "t": [0, 64, 512],
                    "l": [0, 128, 960],
                }
            )
            out = _crop(df)
        finally:
            srv.close()
        for row, (t, left) in zip(
            out["o"], [(0, 0), (64, 128), (512, 960)], strict=True
        ):
            np.testing.assert_array_equal(
                numpy_from_struct(row), fx.levels[0][t : t + 64, left : left + 64]
            )
        assert srv.served <= size, (srv.served, size)


def _windows_frame(url: str, origins: list[tuple[int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "p": [url] * len(origins),
            "t": [o[0] for o in origins],
            "l": [o[1] for o in origins],
        }
    )


def _is_block(request: tuple[str, str, str | None]) -> bool:
    """Whether a request fetched a structure block: 64 KiB at a multiple of
    64 KiB (the last block of a file may be shorter)."""
    method, _, asked = request
    if method != "GET" or not asked:
        return False
    first, last = (int(v) for v in asked.removeprefix("bytes=").split("-"))
    return first % 65536 == 0 and last == first + 65535


#: 48 windows of a 1024x1024 slide in 64-pixel tiles, each over its own
#: 2x2 tiles: no two windows share a chunk.
_ORIGINS = [(128 * (i // 8) + 3, 128 * (i % 8) + 5) for i in range(48)]


@plugin_required
class TestRemoteConcurrency:
    """Remote windows are latency-bound: a call keeps many requests in
    flight, not one per thread, and a later call does not re-read a
    slide's structure it has already read."""

    def test_windows_are_read_ahead_of_their_rows(self, tmp_path: Path) -> None:
        """With 100 ms per request, the server sees more requests at once
        than the plugin has threads: each row's chunks are already in flight
        when it runs."""
        fx = _slide(tmp_path)
        srv = _Server(tmp_path, latency=0.1)
        try:
            out = _crop(_windows_frame(f"{srv.base}/{fx.path.name}", _ORIGINS))
        finally:
            srv.close()
        for (t, left), row in zip(_ORIGINS, out["o"], strict=True):
            np.testing.assert_array_equal(
                numpy_from_struct(row), fx.levels[0][t : t + 64, left : left + 64]
            )
        assert srv.peak > pl.thread_pool_size(), (srv.peak, pl.thread_pool_size())
        # Read ahead, not read twice: no chunk range is requested twice.
        # (Structure blocks may be: the streaming engine can split the rows
        # over several calls, and without a validator each reads them.)
        chunks = [
            r[2] for r in srv.requests if r[0] == "GET" and r[2] and not _is_block(r)
        ]
        repeated = sorted({c for c in chunks if chunks.count(c) > 1})
        assert repeated == [], repeated

    def test_a_later_call_reuses_the_structure_it_read(self, tmp_path: Path) -> None:
        """A second call on an unchanged object (same ETag) asks its head
        and fetches each row's chunks: no structure block again."""
        fx = _slide(tmp_path)
        srv = _Server(tmp_path, etag=True)
        try:
            df = _windows_frame(f"{srv.base}/{fx.path.name}", _ORIGINS[:6])
            _crop(df)
            first = len(srv.requests)
            out = _crop(df)
            second = srv.requests[first:]
        finally:
            srv.close()
        for (t, left), row in zip(_ORIGINS[:6], out["o"], strict=True):
            np.testing.assert_array_equal(
                numpy_from_struct(row), fx.levels[0][t : t + 64, left : left + 64]
            )
        assert [r for r in second if _is_block(r)] == [], second
        assert any(_is_block(r) for r in srv.requests[:first])

    def test_a_changed_object_is_read_afresh(self, tmp_path: Path) -> None:
        """A rewritten object (a new ETag) is not served the old structure:
        its windows are the new pixels, even where its tiles moved."""
        fx = _slide(tmp_path)
        srv = _Server(tmp_path, etag=True)
        try:
            df = _windows_frame(f"{srv.base}/{fx.path.name}", _ORIGINS[:6])
            _crop(df)
            fx = _slide(tmp_path, tile=(32, 32), seed=7)
            out = _crop(df)
        finally:
            srv.close()
        for (t, left), row in zip(_ORIGINS[:6], out["o"], strict=True):
            np.testing.assert_array_equal(
                numpy_from_struct(row), fx.levels[0][t : t + 64, left : left + 64]
            )

    def test_without_a_validator_nothing_is_reused(self, tmp_path: Path) -> None:
        """An object with no ETag or Last-Modified cannot be told unchanged,
        so each call reads its structure again."""
        fx = _slide(tmp_path)
        srv = _Server(tmp_path)
        try:
            df = _windows_frame(f"{srv.base}/{fx.path.name}", _ORIGINS[:6])
            _crop(df)
            first = len(srv.requests)
            _crop(df)
            second = srv.requests[first:]
        finally:
            srv.close()
        assert any(_is_block(r) for r in second), second

    def test_slide_info_reads_rows_concurrently(self, tmp_path: Path) -> None:
        """Remote ``slide_info`` rows run side by side, not one after the
        other."""
        if pl.thread_pool_size() < 2:
            pytest.skip("one thread: nothing to run side by side")
        fx = _slide(tmp_path, levels=3)
        names = []
        for i in range(12):
            copy = tmp_path / f"s{i}.tif"
            copy.write_bytes(fx.path.read_bytes())
            names.append(copy.name)
        srv = _Server(tmp_path, latency=0.1)
        try:
            df = pl.DataFrame({"p": [f"{srv.base}/{n}" for n in names]})
            info = df.select(pl.col("p").cv.slide_info())["p"].to_list()
        finally:
            srv.close()
        assert all(len(i["levels"]) == 3 for i in info)
        assert srv.peak > 1, srv.peak

    def test_a_probe_does_not_take_a_rows_read_ahead(self, tmp_path: Path) -> None:
        """Row 0 (another file) reads ahead the slide's rows before any of
        them runs, so the slide's first row finds its window claimed. That
        row's format probe must leave the claim to the row's own read, or
        the window is fetched twice."""
        fx = _slide(tmp_path)
        png = tmp_path / "first.png"
        png.write_bytes(make_image_png(80, 80, seed=4))
        srv = _Server(tmp_path)
        try:
            origins = _ORIGINS[:8]
            df = pl.DataFrame(
                {
                    "p": [f"{srv.base}/first.png"] + [f"{srv.base}/{fx.path.name}"] * 8,
                    "t": [0] + [o[0] for o in origins],
                    "l": [0] + [o[1] for o in origins],
                }
            )
            out = _crop(df)
        finally:
            srv.close()
        for (t, left), row in zip(origins, out["o"][1:], strict=True):
            np.testing.assert_array_equal(
                numpy_from_struct(row), fx.levels[0][t : t + 64, left : left + 64]
            )
        chunks = [
            r[2]
            for r in srv.requests
            if r[0] == "GET" and r[2] and r[1].endswith(".tif") and not _is_block(r)
        ]
        repeated = sorted({c for c in chunks if chunks.count(c) > 1})
        assert repeated == [], repeated
