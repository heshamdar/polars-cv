"""A call's remote fetches run a bounded window ahead of its rows.

Under Polars 2.0's streaming engine one plugin call is one morsel, and a
Parquet row group is one morsel: thousands of paths. The fetch used to read
every remote path of the call before the first row decoded, so the whole
call's encoded images were resident at once and no fetch overlapped a decode.
Now each row fetches ahead by polars' concurrency budget
(``POLARS_CONCURRENCY_BUDGET``), and a body is freed after its last row reads
it. A bound below the call's size also proves the overlap: had every fetch
finished before the first row, every body would have been resident.

Checked in a subprocess, at every consumer of the fetch (``read_bytes``, the
``file_path`` source, the header-only ``width()``): the budget is read once per
process, and the residency figure is the plugin's
(``_lib._last_fetch_peak_resident``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

from tests.conftest import plugin_required

_CHILD = """
import io, json, sys, threading, http.server

import polars as pl
import polars_cv
import polars_cv._lib as lib
from PIL import Image

N, DISTINCT = int(sys.argv[1]), int(sys.argv[2])
buf = io.BytesIO()
Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="PNG")
PNG = buf.getvalue()


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lock = threading.Lock()
        self.hits = {}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        with self.server.lock:
            self.server.hits[self.path] = self.server.hits.get(self.path, 0) + 1
        self.send_response(200)
        self.send_header("Content-Length", str(len(PNG)))
        self.end_headers()
        self.wfile.write(PNG)

    def log_message(self, *a):
        pass


server = Server(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_address[1]}"
# Row i reads path i % DISTINCT: with DISTINCT < N a path's uses are far apart.
df = pl.DataFrame({"p": [f"{base}/{i % DISTINCT}.png" for i in range(N)]})

out = {}
def run(name, query, check):
    server.hits.clear()
    result = query()
    out[name] = {
        "ok": check(result),
        "requests": sum(server.hits.values()),
        "distinct_requested": len(server.hits),
        "peak": lib._last_fetch_peak_resident(),
    }

run("read_bytes", lambda: df.select(pl.col("p").cv.read_bytes())["p"],
    lambda s: s.to_list() == [PNG] * N)
pipe = polars_cv.Pipeline().source("file_path")
run("file_path", lambda: df.select(pl.col("p").cv.pipe(pipe).sink("numpy"))["p"],
    lambda s: s.null_count() == 0 and s.len() == N)
run("width", lambda: df.select(pl.col("p").cv.width())["p"],
    lambda s: s.to_list() == [8] * N)
server.shutdown()
print(json.dumps(out))
"""


def _run(tmp_path, *, rows: int, distinct: int, threads: int, budget: int) -> dict:
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(_CHILD))
    env = dict(
        os.environ,
        POLARS_CONCURRENCY_BUDGET=str(budget),
        POLARS_MAX_THREADS=str(threads),
    )
    result = subprocess.run(
        [sys.executable, str(script), str(rows), str(distinct)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, f"child failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


@plugin_required
def test_a_call_holds_a_window_of_its_fetches(tmp_path):
    """One row thread holds at most the budget ahead plus its current row."""
    budget = 4
    stats = _run(tmp_path, rows=64, distinct=64, threads=1, budget=budget)
    for consumer, s in stats.items():
        assert s["ok"], (consumer, s)
        assert s["requests"] == 64, (consumer, s)
        assert s["peak"] <= budget + 1, (
            f"{consumer}: {s['peak']} fetched bodies were resident at once out of "
            f"64; the window is the budget ({budget}) plus the row being read"
        )


@plugin_required
def test_the_window_holds_with_rows_on_several_threads(tmp_path):
    """Each row thread fetches ahead of its own rows: several threads hold
    several windows, still far below the call."""
    budget, threads = 4, 4
    stats = _run(tmp_path, rows=128, distinct=128, threads=threads, budget=budget)
    for consumer, s in stats.items():
        assert s["ok"], (consumer, s)
        assert s["requests"] == 128, (consumer, s)
        # A thread's window can reach (budget) rows into the next range.
        bound = threads * 2 * (budget + 1)
        assert s["peak"] <= bound, (consumer, s, bound)


@plugin_required
def test_each_distinct_path_is_fetched_once_per_call(tmp_path):
    """A path repeated far apart in a call is requested once: its body stays
    until its last row reads it."""
    stats = _run(tmp_path, rows=64, distinct=16, threads=1, budget=4)
    for consumer, s in stats.items():
        assert s["ok"], (consumer, s)
        assert s["requests"] == 16, (consumer, s)
        assert s["distinct_requested"] == 16, (consumer, s)
