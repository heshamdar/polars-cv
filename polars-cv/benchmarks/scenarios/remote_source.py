"""Remote (cloud) source benchmarks — the `file_path` fetch path over HTTP.

The rest of the suite feeds polars-cv bytes that are already in memory, so the
whole `fetch.rs` / `cloud.rs` stage — the one every `s3://`, `gs://`, `az://`
and `http://` path goes through — was never measured. This scenario measures it.

## Why HTTP against a local server, and not S3

The transports differ in *signing*, not in structure: every remote scheme lands
in `cloud::read_file`, which builds a backend client and issues one GET per
file, and every one of them is driven by the same `fetch::prefetch` →
`cloud::read_files_concurrent` batching. A loopback HTTP server exercises that
structure with the wide-area latency taken out, which is the point — a WAN
measurement is dominated by the network and hides what the plugin costs. Real
S3/GCS numbers need credentials and a bucket, so they cannot be a committed
benchmark; this can.

The server can *inject* a fixed per-request delay (`--latency-ms`) to model a
wide-area link, which is what makes the concurrency behaviour visible: with
per-request latency L and N files, a fully pipelined fetch takes about
`N/concurrency * L`, and anything slower is the harness rather than the link.

## What it reports

Three measurements over one corpus, so the differences isolate one stage each:

| operation                | measures                                       |
|--------------------------|------------------------------------------------|
| `remote_local_paths`     | control: same pipeline, local filesystem paths |
| `remote_http_paths`      | the full remote path: fetch + decode + ops     |
| `remote_http_read_bytes` | `.cv.read_bytes()` — fetch only, no decode     |

`remote_http_paths - remote_local_paths` is the cost of fetching rather than
reading, and `remote_http_read_bytes` says how much of that is the fetch alone.

It also reports **connections per file** — not a timing, a count, taken from
the server. It answers a question timings only hint at: whether the client
reuses connections across files, or pays a fresh TCP (and, off loopback, TLS)
handshake for every one. A ratio near 1.0 means no reuse.
"""

from __future__ import annotations

import http.server
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from benchmarks.frameworks import BenchmarkResult
from benchmarks.utils.data_gen import temporary_image_set
from benchmarks.utils.memory import run_timed_with_memory
from polars_cv import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Server-side delay applied to every request, in milliseconds. 0 measures the
#: plugin's own overhead; a non-zero value models a wide-area link and is what
#: makes the batching behaviour observable.
DEFAULT_LATENCY_MS = 0.0

#: Matches `fetch::DEFAULT_CONCURRENCY`. Not imported — it is a Rust constant
#: with no FFI accessor — so it is only used to *report* the expected number of
#: waves, never to drive the client.
PLUGIN_CONCURRENCY = 16


@dataclass
class ServeStats:
    """What the server observed, as opposed to what the client timed."""

    #: Distinct TCP connections accepted.
    connections: int
    #: Requests served.
    requests: int

    @property
    def requests_per_connection(self) -> float:
        return self.requests / self.connections if self.connections else 0.0


class _CountingServer(http.server.ThreadingHTTPServer):
    """An HTTP server that counts the connections it accepts.

    `ThreadingHTTPServer` so concurrent fetches are actually served
    concurrently — a single-threaded server would serialize them and make every
    concurrency measurement below report the serial number.
    """

    daemon_threads = True
    # Keep the accept queue deeper than the plugin's concurrency, so a wave of
    # parallel connects is not throttled by the backlog rather than by the
    # client.
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], handler: type, root: Path) -> None:
        super().__init__(address, handler)
        self.root = root
        self.latency_seconds = 0.0
        self.stats = ServeStats(connections=0, requests=0)
        self._lock = threading.Lock()

    def get_request(self) -> tuple[Any, Any]:
        request = super().get_request()
        with self._lock:
            self.stats.connections += 1
        return request

    def note_request(self) -> None:
        with self._lock:
            self.stats.requests += 1


class _Handler(http.server.BaseHTTPRequestHandler):
    """Serve one file per GET, with an optional injected delay.

    Deliberately not `SimpleHTTPRequestHandler`: that one resolves paths
    against the process working directory and logs every request to stderr,
    both of which get in the way here.
    """

    protocol_version = "HTTP/1.1"  # keep-alive, so reuse is *possible* to observe

    # Without this, a *reused* connection stalls ~40 ms per request: the headers
    # and the body go out as separate writes, and Nagle on the second one waits
    # for the delayed ACK of the first. A fresh-connection-per-file client never
    # sees it, because the close flushes — so leaving it on would have made the
    # no-reuse case look artificially good and hidden the very thing this
    # scenario measures.
    disable_nagle_algorithm = True

    def do_GET(self) -> None:  # noqa: N802 — name fixed by the stdlib base class
        server: _CountingServer = self.server  # type: ignore[assignment]
        server.note_request()

        if server.latency_seconds:
            # Models a wide-area round trip. Applied before the body is sent so
            # it lands on the request, not on the transfer.
            threading.Event().wait(server.latency_seconds)

        name = self.path.lstrip("/").split("?", 1)[0]
        target = server.root / name
        # The corpus is generated by this module, so a miss is a bug here
        # rather than untrusted input — but resolve anyway so a `..` in a path
        # cannot escape the served directory.
        if not target.resolve().is_relative_to(server.root.resolve()):
            self.send_error(403)
            return
        try:
            body = target.read_bytes()
        except OSError:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence per-request logging; it would dominate the measurement."""


@contextmanager
def serve_directory(root: Path, latency_ms: float = 0.0) -> "Iterator[_CountingServer]":
    """Serve *root* on an ephemeral loopback port for the duration of the block."""
    server = _CountingServer(("127.0.0.1", 0), _Handler, root)
    server.latency_seconds = latency_ms / 1000.0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _pipeline() -> Pipeline:
    """The pipeline every measurement below runs, so only the source differs."""
    return (
        Pipeline()
        .source("file_path")
        .resize(height=224, width=224)
        .normalize(method="minmax")
    )


def _timed(
    label: str,
    call: Any,
    image_count: int,
    image_size: tuple[int, int],
    warmup_iterations: int,
    benchmark_iterations: int,
) -> BenchmarkResult:
    """Run *call* warmup+timed times and shape the result like every scenario."""
    for _ in range(warmup_iterations):
        call()

    total_time = 0.0
    peak_memory = 0.0
    for _ in range(benchmark_iterations):
        _, elapsed, mem_stats = run_timed_with_memory(call)
        total_time += elapsed
        peak_memory = max(peak_memory, mem_stats.peak_memory_mb)

    avg_time = total_time / benchmark_iterations
    return BenchmarkResult(
        framework="polars-cv-eager",
        operation=label,
        image_count=image_count,
        image_size=image_size,
        total_time_seconds=avg_time,
        throughput_images_per_second=image_count / avg_time,
        latency_ms_per_image=(avg_time / image_count) * 1000,
        peak_memory_mb=peak_memory,
    )


def run_remote_source(
    image_count: int = 300,
    image_size: tuple[int, int] = (256, 256),
    warmup_iterations: int = 1,
    benchmark_iterations: int = 3,
    latency_ms: float = DEFAULT_LATENCY_MS,
    verbose: bool = True,
) -> tuple[list[BenchmarkResult], ServeStats]:
    """Measure the remote fetch path against a local-path control.

    Returns the results and the server's own view of the last measured run, so
    the connection-reuse ratio can be reported alongside the timings.
    """
    height, width = image_size
    results: list[BenchmarkResult] = []

    with temporary_image_set(image_count, height, width) as image_set:
        assert image_set.file_paths is not None
        root = image_set.file_paths[0].parent
        local_paths = [str(p) for p in image_set.file_paths]

        pipe = _pipeline()

        local_df = pl.DataFrame({"path": local_paths})

        def run_local() -> Any:
            return (
                local_df.lazy()
                .select(out=pl.col("path").cv.pipe(pipe).sink("numpy"))
                .collect()
            )

        results.append(
            _timed(
                "remote_local_paths",
                run_local,
                image_count,
                (width, height),
                warmup_iterations,
                benchmark_iterations,
            )
        )
        if verbose:
            print(f"  {results[-1].operation}: {results[-1]}")

        with serve_directory(root, latency_ms) as server:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            http_df = pl.DataFrame(
                {"path": [f"{base}/{p.name}" for p in image_set.file_paths]}
            )

            def run_http() -> Any:
                return (
                    http_df.lazy()
                    .select(out=pl.col("path").cv.pipe(pipe).sink("numpy"))
                    .collect()
                )

            def run_http_bytes() -> Any:
                return (
                    http_df.lazy().select(out=pl.col("path").cv.read_bytes()).collect()
                )

            results.append(
                _timed(
                    "remote_http_paths",
                    run_http,
                    image_count,
                    (width, height),
                    warmup_iterations,
                    benchmark_iterations,
                )
            )
            if verbose:
                print(f"  {results[-1].operation}: {results[-1]}")

            # Reset before the fetch-only measurement so the reported
            # connection ratio describes exactly that run.
            server.stats = ServeStats(connections=0, requests=0)

            results.append(
                _timed(
                    "remote_http_read_bytes",
                    run_http_bytes,
                    image_count,
                    (width, height),
                    warmup_iterations,
                    benchmark_iterations,
                )
            )
            if verbose:
                print(f"  {results[-1].operation}: {results[-1]}")

            stats = server.stats

    return results, stats


def run_benchmarks(
    image_counts: list[int] | None = None,
    image_sizes: list[tuple[int, int]] | None = None,
    warmup_iterations: int = 1,
    benchmark_iterations: int = 3,
    latency_ms: float = DEFAULT_LATENCY_MS,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Suite entry point: the matrix, flattened to a result list."""
    counts = image_counts or [300]
    sizes = image_sizes or [(256, 256)]
    results: list[BenchmarkResult] = []
    for size in sizes:
        for count in counts:
            if verbose:
                print(f"\nRemote source: {count} images at {size}")
            batch, stats = run_remote_source(
                image_count=count,
                image_size=size,
                warmup_iterations=warmup_iterations,
                benchmark_iterations=benchmark_iterations,
                latency_ms=latency_ms,
                verbose=verbose,
            )
            results += batch
            if verbose:
                print(
                    f"  server saw {stats.connections} connections for "
                    f"{stats.requests} requests "
                    f"({stats.requests_per_connection:.2f} requests/connection)"
                )
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=DEFAULT_LATENCY_MS,
        help="server-side delay per request, modelling a wide-area link",
    )
    args = parser.parse_args(argv)

    results, stats = run_remote_source(
        image_count=args.count,
        image_size=(args.size, args.size),
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        latency_ms=args.latency_ms,
    )

    print("\n" + "=" * 72)
    print(f"{'operation':<26}{'img/s':>12}{'ms/img':>12}{'peak MB':>12}")
    print("-" * 72)
    for r in results:
        print(
            f"{r.operation:<26}{r.throughput_images_per_second:>12.1f}"
            f"{r.latency_ms_per_image:>12.3f}{r.peak_memory_mb:>12.1f}"
        )
    print("-" * 72)
    print(
        f"connections={stats.connections} requests={stats.requests} "
        f"({stats.requests_per_connection:.2f} requests/connection; "
        f"1.00 means no connection reuse)"
    )
    print(f"plugin fetch concurrency is {PLUGIN_CONCURRENCY} files per wave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
