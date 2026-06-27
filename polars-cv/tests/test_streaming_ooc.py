"""
Out-of-core (OOC) / spill-to-disk correctness checks (polars >= 0.54).

Polars' new streaming engine wires a `polars-ooc` memory-manager into
`group_by`, `sort`, and equi-`join` so larger-than-RAM lazy queries can, in
principle, spill cold partitions to disk. These tests verify that
polars-cv's Binary/Struct/List/Array-typed pipeline outputs round-trip
correctly through those operators on both sides:

  - plugin *output* gets passed through a spill-capable operator
    (group_by/sort/join downstream of `.cv.pipe()`),
  - plugin *input* arrives via a spill-capable operator's result
    (group_by upstream of `.cv.pipe()`).

Source inspection of `polars-ooc-0.54.4` (pinned in this repo's Cargo.lock,
matching the installed polars 1.42.0 wheel) found that the actual
disk-backed spill implementation for `DataFrame` is a literal
`// TODO: just a dummy spill for now` stub (`impl Spillable for DataFrame`
in `spill_frame.rs` clones the frame in memory instead of writing to disk),
and `MemoryManager::spill`/`spill_blocking` in `memory_manager.rs` are
empty no-op bodies. So no real disk I/O happens regardless of the
`POLARS_OOC_*` env knobs in this polars version -- the scaffolding
(SpillToken/SpillFrame/SpillContext, config knobs, group_by/join node
wiring) has landed, but the backend hasn't. `test_spill_directory_stays_empty`
documents this explicitly as a regression canary: it should start failing
(in a good way) the day upstream actually implements disk-backed spilling.

Marked `slow` because the env-var tests spawn a subprocess (the OOC config
is read once into a process-wide `LazyLock`, so it can only be exercised
by varying environment *before* the interpreter starts).
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
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


def _run_subprocess_driver(env_overrides: dict[str, str], driver_code: str) -> str:
    """Run `driver_code` in a fresh interpreter with `env_overrides` set
    before startup (the OOC config is a process-wide LazyLock read once)."""
    env = {**os.environ, **env_overrides}
    result = subprocess.run(
        [sys.executable, "-c", driver_code],
        cwd=str(Path(__file__).parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return result.stdout


_DRIVER = """
import sys
import polars as pl
from polars_cv import Pipeline

def _png(seed, size=8):
    import io
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="PNG")
    return buf.getvalue()

n = 500
imgs = [_png(i) for i in range(n)]
keys = [i % 17 for i in range(n)]
df = pl.DataFrame({"img": imgs, "key": keys})
blob_pipe = Pipeline().source("image_bytes").resize(height=4, width=4).grayscale()
lf = df.with_columns(b=pl.col("img").cv.pipe(blob_pipe).sink("blob")).lazy()

streamed = lf.group_by("key").agg(pl.col("b").first()).sort("key").collect(engine="streaming")
eager = lf.group_by("key").agg(pl.col("b").first()).sort("key").collect(engine="in-memory")
assert streamed.equals(eager), "OOC-config-influenced run diverged from eager"
print("OK")
"""


class TestOocEnvVarsDoNotChangeCorrectness:
    """`POLARS_OOC_SPILL_POLICY=spill` plus aggressive thresholds forces the
    streaming engine's group_by node down the SpillFrame-registration path
    for cold partitions. Plugin output must stay correct either way."""

    def test_spill_policy_no_spill(self) -> None:
        out = _run_subprocess_driver({"POLARS_OOC_SPILL_POLICY": "no_spill"}, _DRIVER)
        assert "OK" in out

    def test_spill_policy_spill_aggressive_thresholds(self) -> None:
        out = _run_subprocess_driver(
            {
                "POLARS_OOC_SPILL_POLICY": "spill",
                "POLARS_OOC_MEMORY_BUDGET_FRACTION": "0.0",
                "POLARS_OOC_SPILL_MIN_BYTES": "0",
            },
            _DRIVER,
        )
        assert "OK" in out


class TestSpillDirectoryStaysEmpty:
    """Regression canary: documents that polars 0.54.4's OOC backend never
    actually writes to `POLARS_OOC_SPILL_DIR`, even when spilling is forced
    on with the most aggressive thresholds available, because `Spillable
    for DataFrame` is a dummy in-memory-clone stub upstream. If this test
    starts failing, upstream has wired up real disk-backed spilling and the
    other tests in this file should be revisited to check actual spill
    files instead of just correctness."""

    def test_spill_directory_stays_empty(self) -> None:
        with tempfile.TemporaryDirectory() as spill_dir:
            out = _run_subprocess_driver(
                {
                    "POLARS_OOC_SPILL_POLICY": "spill",
                    "POLARS_OOC_MEMORY_BUDGET_FRACTION": "0.0",
                    "POLARS_OOC_SPILL_MIN_BYTES": "0",
                    "POLARS_OOC_SPILL_DIR": spill_dir,
                },
                _DRIVER,
            )
            assert "OK" in out
            files = list(Path(spill_dir).rglob("*"))
            assert files == [], (
                f"expected no spill files (polars-ooc 0.54.4's DataFrame spill "
                f"backend is a dummy in-memory stub), but found: {files}"
            )
