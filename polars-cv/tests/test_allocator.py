"""
The plugin allocates through polars' own allocator (CR-60).

pyo3-polars' `PolarsAllocator` relays every allocation to the allocator polars
exports as the `polars.polars._allocator` capsule, and falls back to the
system `malloc` -- silently -- when that capsule cannot be imported. A polars
release that moves the capsule would put the plugin back on glibc, whose heap
trimming made a call that holds many large rows re-fault its memory on every
call after the first (~40% slower). This makes that fallback a failure.
"""

import pytest

from tests.conftest import plugin_required

pytestmark = [pytest.mark.structural, plugin_required]


def test_the_plugin_allocates_with_polars_allocator() -> None:
    import polars_cv._lib as lib

    assert lib.__allocator__ == "polars", (
        "the plugin's PolarsAllocator could not import polars' allocator "
        "capsule and fell back to the system allocator"
    )


def test_polars_counts_the_plugins_memory() -> None:
    """Polars >= 2.0's out-of-core engine spills when its count of allocated
    bytes passes the memory budget, and that count includes what the plugin
    allocates through the capsule. It must see an output while it is held and
    see it go when dropped: memory allocated where polars cannot count it but
    freed where it can drives the count below the truth, so it never spills.

    Run in a subprocess so the count starts clean. ``_estimate_memory_usage``
    is polars-private, so this reads it at its one place.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(
        """
        import io
        import numpy as np
        import polars as pl
        from PIL import Image
        from polars_cv import Pipeline

        count = pl._plr._estimate_memory_usage
        # Hold a counted 160 MB first, so an undercount shows below it rather
        # than clamping at zero.
        floor = pl.Series([1.0] * 20_000_000)
        buf = io.BytesIO()
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(buf, format="PNG")
        # Many rows, each below polars' 4 MB per-thread batching threshold.
        src = pl.DataFrame({"x": [buf.getvalue()] * 40})
        pipe = Pipeline().source("image_bytes").resize(height=512, width=512)
        before = count()
        out = src.select(pl.col("x").cv.pipe(pipe.cast("f32")).sink("blob"))
        size = out["x"].bin.size().sum()
        held = count() - before
        del out
        print(size, held, count() - before)
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    size, held, after = map(int, r.stdout.split())
    slack = 64 * 2**20  # polars' per-thread batching, over a few threads
    assert abs(held - size) < slack, (size, held)
    assert abs(after) < slack, (size, after)
