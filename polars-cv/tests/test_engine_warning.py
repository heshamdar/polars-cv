"""The single-threaded-call warning is based on time, not row count (CR-32).

It used to fire when one plugin call carried 50 000 rows. Image rows cost
milliseconds each, so a single-threaded run could take tens of seconds without
ever reaching that. It also went quiet for the rest of the process the first
time any two calls overlapped. It now fires when one call runs longer than
``POLARS_CV_ENGINE_WARN_SECONDS`` with no other call overlapping it.

The warning is once per process, so every case runs in a fresh interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from .conftest import plugin_required

MESSAGE = "ran on one thread"

SCRIPT = textwrap.dedent(
    """
    import polars as pl
    from polars_cv import Pipeline
    from tests.conftest import make_test_png

    df = pl.DataFrame({"img": [make_test_png(32, 32)] * 16})
    pipe = Pipeline().source("image_bytes", dtype="u8").blur(sigma=1.0)
    df.lazy().select(o=pl.col("img").cv.pipe(pipe).sink("numpy")).collect(
        engine="in-memory"
    )
    """
)


def _run(env: dict[str, str]) -> str:
    full = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("POLARS_CV_ENGINE_WARN")
        and k != "POLARS_CV_SILENCE_ENGINE_WARNING"
    }
    full.update(env)
    full.setdefault("POLARS_MAX_THREADS", "4")
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=full,
        capture_output=True,
        text=True,
        check=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return proc.stderr


@plugin_required
class TestEngineWarning:
    def test_a_slow_lone_call_warns_whatever_its_row_count(self) -> None:
        assert MESSAGE in _run({"POLARS_CV_ENGINE_WARN_SECONDS": "0.000001"})

    def test_a_fast_call_does_not_warn(self) -> None:
        assert MESSAGE not in _run({})

    def test_silencing_still_works(self) -> None:
        stderr = _run(
            {
                "POLARS_CV_ENGINE_WARN_SECONDS": "0.000001",
                "POLARS_CV_SILENCE_ENGINE_WARNING": "1",
            }
        )
        assert MESSAGE not in stderr

    def test_the_row_threshold_variable_is_reported_not_ignored(self) -> None:
        # A removed setting must not silently do nothing.
        stderr = _run({"POLARS_CV_ENGINE_WARN_ROWS": "10"})
        assert "POLARS_CV_ENGINE_WARN_ROWS is no longer read" in stderr
        assert "POLARS_CV_ENGINE_WARN_SECONDS" in stderr

    @pytest.mark.parametrize("bad", ["abc", "-1", "0"])
    def test_an_unusable_threshold_is_reported(self, bad: str) -> None:
        stderr = _run({"POLARS_CV_ENGINE_WARN_SECONDS": bad})
        assert "POLARS_CV_ENGINE_WARN_SECONDS" in stderr
