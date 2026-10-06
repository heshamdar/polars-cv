"""
Out-of-core (OOC) / spill-to-disk correctness checks (polars >= 2.0).

Polars' streaming engine (the default engine for lazy queries since 2.0)
runs `group_by`, `sort`, joins and windows under a `polars-ooc` memory
manager: past the memory budget it spills cold data to disk as Arrow IPC and
reads it back. These tests verify that polars-cv's Binary/Struct/List/Array
(and extension-typed) pipeline outputs round-trip correctly through those
operators on both sides:

  - plugin *output* gets passed through a spill-capable operator
    (group_by/sort/join downstream of `.cv.pipe()`),
  - plugin *input* arrives via a spill-capable operator's result
    (group_by upstream of `.cv.pipe()`),

and that a query forced to spill (a 1 MB budget) really does spill — read
from polars' own `POLARS_OOC_LOG_METRICS` report — and still matches the
in-memory engine. (Up to polars 1.x / polars-ooc 0.54 the spill backend was
an in-memory stub, which a canary here pinned; it is real now.) That the
budget sees the plugin's own memory at all is `test_allocator.py`'s.

Marked `slow` because the env-var tests spawn a subprocess (the OOC config
is read once into a process-wide `LazyLock`, so it can only be exercised
by varying environment *before* the interpreter starts).
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from polars_cv import Pipeline
from tests.conftest import plugin_required

pytestmark = [pytest.mark.slow, plugin_required]


def _png(seed: int, size: int = 8) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()


def _make_df(n: int):
    import polars as pl

    imgs = [_png(i) for i in range(n)]
    keys = list(range(n % 7, n % 7 + n))
    return pl.DataFrame({"img": imgs, "key": keys})


class TestSpillCapableOperatorsDownstream:
    """Plugin output flows into a spill-capable operator."""

    def test_groupby_blob_output_matches_eager(self) -> None:
        import polars as pl

        df = _make_df(300)
        blob_pipe = (
            Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()
        )
        df = df.with_columns(key=pl.col("key") % 11)
        lf = df.with_columns(b=pl.col("img").cv.pipe(blob_pipe).sink("blob")).lazy()

        streamed = (
            lf.group_by("key")
            .agg(pl.col("b").first())
            .sort("key")
            .collect(engine="streaming")
        )
        eager = (
            lf.group_by("key")
            .agg(pl.col("b").first())
            .sort("key")
            .collect(engine="in-memory")
        )
        assert streamed.equals(eager)

    def test_groupby_blob_list_output_matches_eager(self) -> None:
        import polars as pl

        df = _make_df(120)
        df = df.with_columns(key=pl.col("key") % 5)
        blob_pipe = (
            Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()
        )
        lf = df.with_columns(b=pl.col("img").cv.pipe(blob_pipe).sink("blob")).lazy()

        streamed = (
            lf.group_by("key").agg(pl.col("b")).sort("key").collect(engine="streaming")
        )
        eager = (
            lf.group_by("key").agg(pl.col("b")).sort("key").collect(engine="in-memory")
        )
        assert streamed.equals(eager)

    def test_sort_on_plugin_scalar_with_blob_carried_matches_eager(self) -> None:
        import polars as pl

        df = _make_df(300)
        sum_pipe = Pipeline().source("image_bytes").grayscale().reduce_sum()
        blob_pipe = (
            Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()
        )
        lf = df.with_columns(
            s=pl.col("img").cv.pipe(sum_pipe).sink("native"),
            b=pl.col("img").cv.pipe(blob_pipe).sink("blob"),
        ).lazy()

        streamed = lf.sort("s").collect(engine="streaming")
        eager = lf.sort("s").collect(engine="in-memory")
        assert streamed.equals(eager)

    def test_join_on_two_plugin_blob_columns_matches_eager(self) -> None:
        import polars as pl

        n = 150
        blob_pipe = (
            Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()
        )

        left = _make_df(n).rename({"img": "img1"})
        right_imgs = [_png(i + 1000) for i in range(n)]
        right = pl.DataFrame({"img2": right_imgs, "key": left["key"]})

        left_lf = left.with_columns(
            b1=pl.col("img1").cv.pipe(blob_pipe).sink("blob")
        ).lazy()
        right_lf = right.with_columns(
            b2=pl.col("img2").cv.pipe(blob_pipe).sink("blob")
        ).lazy()

        joined = left_lf.join(right_lf, on="key")
        streamed = joined.collect(engine="streaming").sort("key")
        eager = joined.collect(engine="in-memory").sort("key")
        assert streamed.equals(eager)


class TestSpillCapableOperatorsUpstream:
    """Plugin input arrives via the output of a spill-capable operator."""

    def test_groupby_then_pipe_matches_direct_pipe(self) -> None:
        import polars as pl

        df = _make_df(200)
        df = df.with_columns(key=pl.col("key") % 13)
        blob_pipe = (
            Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()
        )

        grouped = df.lazy().group_by("key").agg(pl.col("img").first())
        post_groupby = (
            grouped.with_columns(out=pl.col("img").cv.pipe(blob_pipe).sink("blob"))
            .sort("key")
            .collect(engine="streaming")
        )

        direct = (
            df.with_columns(out=pl.col("img").cv.pipe(blob_pipe).sink("blob"))
            .group_by("key")
            .agg(pl.col("out").first())
            .sort("key")
        )
        assert post_groupby.select("key", "out").equals(direct.select("key", "out"))

    def test_sort_then_pipe_matches_direct_pipe(self) -> None:
        import polars as pl

        df = _make_df(200)
        blob_pipe = (
            Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()
        )

        sorted_first = (
            df.lazy()
            .sort("key")
            .with_columns(out=pl.col("img").cv.pipe(blob_pipe).sink("blob"))
            .collect(engine="streaming")
        )
        direct = df.sort("key").with_columns(
            out=pl.col("img").cv.pipe(blob_pipe).sink("blob")
        )
        assert sorted_first.equals(direct)


class TestCompiledGraphCacheAcrossSpillCapableOps:
    """The compiled-graph cache key is pure graph_json + column names with
    zero data-derived state, so a spill-capable neighbour shouldn't perturb
    correctness across many morsels. No spill/recompile counter is exposed
    by the plugin, so this is verified indirectly via per-row correctness
    across enough rows to span many streaming morsels (same approach as
    `test_graph_cache.py::TestCacheStreaming`)."""

    def test_many_rows_through_groupby_stay_correct(self) -> None:
        import polars as pl

        n = 2000
        df = _make_df(n)
        df = df.with_columns(key=pl.col("key") % 23)
        blob_pipe = (
            Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()
        )
        lf = df.with_columns(b=pl.col("img").cv.pipe(blob_pipe).sink("blob")).lazy()

        streamed = (
            lf.group_by("key")
            .agg(pl.col("b").first())
            .sort("key")
            .collect(engine="streaming")
        )
        eager = (
            lf.group_by("key")
            .agg(pl.col("b").first())
            .sort("key")
            .collect(engine="in-memory")
        )
        assert streamed.equals(eager)


def _run_subprocess_driver(
    env_overrides: dict[str, str], driver_code: str
) -> subprocess.CompletedProcess[str]:
    """Run `driver_code` in a fresh interpreter with `env_overrides` set
    before startup (the OOC config is a process-wide LazyLock read once)."""
    env = {**os.environ, **env_overrides}
    result = subprocess.run(
        [sys.executable, "-c", driver_code],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return result


_DRIVER = """
import polars as pl
from polars_cv import Pipeline

def _png(seed, size=48):
    import io
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()

n = 300
df = pl.DataFrame({"img": [_png(i) for i in range(n)], "key": [i % 17 for i in range(n)]})
pipe = Pipeline().source("image_bytes", dtype="u8").resize(height=64, width=64)
img = pl.col("img").cv.pipe(pipe)
lf = df.lazy().with_columns(
    blob=img.sink("blob"),
    numpy=img.sink("numpy"),
    ndarray=img.sink("ndarray"),
    array=img.sink("array", shape=[64, 64, 3]),
)
for name, q in {
    "sort": lf.sort("key", maintain_order=True),
    "group_by": lf.group_by("key").agg(pl.all().exclude("img").first()).sort("key"),
}.items():
    streamed = q.collect(engine="streaming")
    eager = q.collect(engine="in-memory")
    assert streamed.equals(eager), f"{name}: the streaming run diverged from eager"
print("OK")
"""

#: One spill context's report (`POLARS_OOC_LOG_METRICS=1`), with its count of
#: spills. Polars' debug output, so this format can change: the forced-spill
#: test then finds no report and fails rather than passing on nothing.
_SPILL_REPORT = re.compile(
    r"^spill_stats\(([^)]*)\):.* spill\(succ=[^,]*, n=(\d+)\)", re.M
)


def _spills(stderr: str) -> dict[str, int]:
    """Spills per spill context (`sort`, `group_by`, ...), most seen per name."""
    out: dict[str, int] = {}
    for name, n in _SPILL_REPORT.findall(stderr):
        out[name] = max(out.get(name, 0), int(n))
    return out


class TestOutOfCore:
    """Plugin outputs stay correct whether or not the engine spills them."""

    def test_correct_without_spilling(self) -> None:
        r = _run_subprocess_driver(
            {"POLARS_OOC_MEMORY_BUDGET_MB": "100000", "POLARS_OOC_LOG_METRICS": "1"},
            _DRIVER,
        )
        assert "OK" in r.stdout
        assert not any(_spills(r.stderr).values()), r.stderr[-2000:]

    def test_correct_when_forced_to_spill(self, tmp_path: Path) -> None:
        r = _run_subprocess_driver(
            {
                "POLARS_OOC_MEMORY_BUDGET_MB": "1",
                "POLARS_OOC_SPILL_MIN_BYTES": "0",
                "POLARS_OOC_LOG_METRICS": "1",
                "POLARS_OOC_SPILL_DIR": str(tmp_path),
            },
            _DRIVER,
        )
        assert "OK" in r.stdout
        spills = _spills(r.stderr)
        assert sum(spills.values()) > 0, f"nothing spilled: {r.stderr[-2000:]}"
        # Spill files are removed as their frames are read back or dropped.
        assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
