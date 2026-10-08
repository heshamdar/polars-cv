"""Remote reads run on polars' async runtime, the plugin's only one.

The plugin links its own copy of polars, and with it polars' tokio runtime
(``polars_core::runtime::ASYNC``, sized by ``POLARS_ASYNC_THREAD_COUNT``),
which polars-io's object stores already spawn onto. A second runtime of the
plugin's own doubled the threads of every remote read and ignored that
setting. ``clippy.toml`` refuses a new runtime at lint time; this checks the
result at the user-facing entry point.

Checked in a subprocess: a runtime lives for the life of the process, so an
in-process check would depend on which test read a URL first.

Threads are counted by tokio's default name (``tokio-runtime-worker``, which
Linux shortens to 15 characters), which blocking-pool threads share. The child
therefore reads IP-literal URLs only: no DNS lookup, so no blocking thread.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from tests.conftest import plugin_required

_CHILD = """
import json, os, threading, http.server

import polars as pl
import polars_cv  # noqa: F401  (registers the .cv namespace)
from polars_cv import CloudOptions

BODY = b"remote bytes"


def tokio_threads():
    names = []
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                names.append(f.read().strip())
        except FileNotFoundError:
            pass
    return sum(name.startswith("tokio-runtime-w") for name in names)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *a):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
server.daemon_threads = True
threading.Thread(target=server.serve_forever, daemon=True).start()
port = server.server_address[1]

before = tokio_threads()
http_read = pl.DataFrame({"p": [f"http://127.0.0.1:{port}/a.bin"]}).select(
    pl.col("p").cv.read_bytes()
)["p"][0]
s3 = CloudOptions(
    anonymous=True,
    storage_options={
        "aws_endpoint": f"http://127.0.0.1:{port}",
        "aws_allow_http": "true",
        "aws_region": "us-east-1",
    },
)
s3_read = pl.DataFrame({"p": ["s3://bucket/b.bin"]}).select(
    pl.col("p").cv.read_bytes(cloud_options=s3)
)["p"][0]
after = tokio_threads()
server.shutdown()
print(json.dumps({
    "http": http_read == BODY,
    "s3": s3_read == BODY,
    "started": after - before,
}))
"""


@plugin_required
@pytest.mark.skipif(
    not os.path.isdir("/proc/self/task"), reason="counts threads through /proc"
)
def test_remote_reads_run_on_polars_async_runtime(tmp_path):
    """An http:// and an s3:// read start exactly polars' async runtime's
    workers, as many as POLARS_ASYNC_THREAD_COUNT says, and nothing else."""
    cpus = os.cpu_count() or 1
    # Distinct from the CPU count, which a runtime of the plugin's own
    # (one worker per CPU) would start instead.
    workers = 3 if cpus != 3 else 2
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(_CHILD))
    env = dict(os.environ, POLARS_ASYNC_THREAD_COUNT=str(workers))
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, f"child failed:\n{result.stdout}\n{result.stderr}"
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    assert stats["http"] and stats["s3"], stats
    assert stats["started"] == workers, (
        f"the remote reads started {stats['started']} tokio threads; polars' "
        f"async runtime has {workers} (POLARS_ASYNC_THREAD_COUNT) and should be "
        f"the only one (the CPU count is {cpus})"
    )
