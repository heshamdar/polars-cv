"""The remote fetch path reuses connections and respects a global budget.

Both properties are read off the **server's** counters — integers, not timings —
so nothing here is a performance assertion that a slow machine can flake.

They are checked in a subprocess because both are process-wide facts:
``POLARS_CONCURRENCY_BUDGET`` is read once into a `OnceLock` at the first permit
acquisition, and the HTTP connection pool and object-store cache live for the
life of the process. Setting the environment variable in-process would make the
result depend on test ordering, which is exactly the kind of guard this
repository treats as broken.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import plugin_required

#: The child script. It serves a generated corpus over loopback HTTP, runs one
#: `.cv.read_bytes()` over every URL, and prints what the server observed.
#:
#: The handler holds each request open briefly so that overlap is *observable*.
#: That is not a timing assertion — the delay only widens the window in which
#: concurrent requests can be counted; the assertion is on the integer
#: high-water mark, which cannot exceed the budget however slow the machine is.
_CHILD = """
import json, sys, threading, http.server

import polars as pl
import polars_cv  # noqa: F401  (registers the .cv namespace)

PORT_HOLDER = {}
N = int(sys.argv[1])
BODY = b"x" * 256


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lock = threading.Lock()
        self.connections = 0
        self.requests = 0
        self.inflight = 0
        self.peak_inflight = 0

    def get_request(self):
        request = super().get_request()
        with self.lock:
            self.connections += 1
        return request


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def do_GET(self):
        srv = self.server
        with srv.lock:
            srv.requests += 1
            srv.inflight += 1
            srv.peak_inflight = max(srv.peak_inflight, srv.inflight)
        try:
            # Widen the overlap window so concurrency is observable at all.
            threading.Event().wait(0.05)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)
        finally:
            with srv.lock:
                srv.inflight -= 1

    def log_message(self, *a):
        pass


server = Server(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_address[1]}"

df = pl.DataFrame({"path": [f"{base}/img{i}.bin" for i in range(N)]})
out = df.lazy().select(pl.col("path").cv.read_bytes()).collect()
assert out.height == N, out.height

server.shutdown()
print(json.dumps({
    "requests": server.requests,
    "connections": server.connections,
    "peak_inflight": server.peak_inflight,
}))
"""


def _run(tmp_path, *, count: int, budget: str | None) -> dict[str, int]:
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(_CHILD))

    env = dict(os.environ)
    if budget is not None:
        env["POLARS_CONCURRENCY_BUDGET"] = budget
    # Keep the plugin single-threaded so the only concurrency measured is the
    # fetch fan-out, not the engine's morsel parallelism.
    env["POLARS_MAX_THREADS"] = "1"

    result = subprocess.run(
        [sys.executable, str(script), str(count)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, f"child failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


@plugin_required
def test_connections_are_reused_across_files(tmp_path):
    """Many files must not mean many connections.

    Before the shared client, this was 1.00 requests per connection exactly —
    a fresh TCP (and, off loopback, TLS) handshake for every image.

    The bound is deliberately loose: what is being pinned is "connections scale
    with concurrency, not with file count", and the exact number of pooled
    connections is the budget's business, not this test's.
    """
    stats = _run(tmp_path, count=64, budget="4")

    # Non-vacuity: if the fetch silently did nothing, the counts below would
    # pass trivially.
    assert stats["requests"] == 64, stats

    assert stats["connections"] <= 16, (
        f"64 requests opened {stats['connections']} connections "
        f"({stats['requests'] / max(stats['connections'], 1):.2f} per connection) — "
        f"connections are scaling with file count, so the pool is not being reused"
    )


@plugin_required
def test_in_flight_requests_stay_within_the_global_budget(tmp_path):
    """`POLARS_CONCURRENCY_BUDGET` bounds concurrent requests, process-wide.

    The fan-out used to be a constant 16 *per plugin call*, which the streaming
    engine multiplies by the number of morsels in flight — so nothing bounded
    the total, and nothing let a user set it. One semaphore now does both.
    """
    budget = 4
    stats = _run(tmp_path, count=64, budget=str(budget))

    assert stats["requests"] == 64, stats
    assert stats["peak_inflight"] <= budget, (
        f"peak in-flight requests was {stats['peak_inflight']} with "
        f"POLARS_CONCURRENCY_BUDGET={budget}; the budget is not bounding the fan-out"
    )
    # And it must actually be *using* the budget, or a serial implementation
    # would satisfy the bound above while being far slower.
    assert stats["peak_inflight"] > 1, (
        "requests never overlapped; the fetch is serial, which the bound above "
        "cannot distinguish from a correctly bounded fan-out"
    )


@plugin_required
def test_a_larger_budget_is_honoured(tmp_path):
    """The bound tracks the budget rather than being a hardcoded ceiling.

    Without this, a constant that happens to sit below the tested budget would
    satisfy the previous test forever.
    """
    stats = _run(tmp_path, count=64, budget="12")

    assert stats["requests"] == 64, stats
    assert stats["peak_inflight"] > 4, (
        f"peak in-flight was {stats['peak_inflight']} at a budget of 12 — the "
        f"fan-out is pinned to something narrower than the budget"
    )
    assert stats["peak_inflight"] <= 12, stats


if __name__ == "__main__":  # pragma: no cover - convenience for manual runs
    raise SystemExit(pytest.main([__file__, "-v"]))
