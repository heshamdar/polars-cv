"""The measurement machinery measures what it says."""

from __future__ import annotations

import time

import numpy as np
import polars as pl
import pytest
from PIL import Image

from bench import harness, workloads
from bench.workloads import thumb_size


def test_summarize_median_and_iqr() -> None:
    s = harness.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert s["median_s"] == 3.0
    assert (s["q1_s"], s["q3_s"], s["iqr_s"]) == (2.0, 4.0, 2.0)


def test_peak_sampler_sees_an_allocation() -> None:
    """A 300 MiB array held for a few samples shows up in the peak."""
    before = harness.tree_pss_bytes()
    with harness.PeakSampler() as s:
        a = np.ones(300 * 2**20, np.uint8)
        time.sleep(harness.SAMPLE_S * 4)
        del a
    assert s.peak - before > 250 * 2**20


def test_measure_counts_runs_and_warmup() -> None:
    calls = []
    t = harness.measure(lambda: calls.append(1), runs=3, warmup=2)
    assert len(calls) == 5 and len(t.times_s) == 3


def test_pinned_env_sets_every_thread_variable() -> None:
    env = harness.pinned_env(3)
    assert all(env[k] == "3" for k in harness.THREAD_ENV)


@pytest.mark.parametrize(
    "size",
    [
        (640, 480),
        (480, 640),
        (500, 333),
        (333, 500),
        (200, 145),
        (612, 612),
        (640, 427),
        (427, 640),
        (1000, 3),
        (3, 1000),
        (256, 100),
        # Where a plain round() of the scaled side disagrees with Pillow.
        (142, 1024),
        (198, 1024),
        (1024, 254),
    ],
)
def test_thumb_size_matches_pillow(size: tuple[int, int]) -> None:
    """OpenCV and torchvision size their thumbnails exactly as PIL does."""
    im = Image.new("RGB", size)
    im.thumbnail((256, 256))
    assert thumb_size(*size) == im.size


# --- polars-cv with its own row parallelism off ------------------------------


def test_polars_cv_modes_cover_every_polars_cv_library() -> None:
    """Each polars-cv library names its engine and pool setting explicitly.

    A name the table does not know must not fall back to another mode.
    """
    pcv = {lib for lib in workloads.LIBRARIES if lib.startswith("polars-cv")}
    assert pcv == set(workloads.POLARS_CV_MODES)
    with pytest.raises(KeyError):
        workloads.prepare("polars-cv-bogus", "w1", "path", pl.DataFrame(), 2)


@pytest.mark.skipif(
    not workloads.PCV_HAS_ROW_PARALLELISM_SWITCH,
    reason="set_row_parallelism arrived in polars-cv 0.38.0",
)
def test_streaming_nopool_keeps_every_call_on_one_thread(manifest) -> None:
    """The streaming engine's morsels are the only parallelism.

    ``prepare`` turns the plugin's row splitting off (the cell is its own
    process), and every plugin call of the job then runs its rows on the one
    thread polars called it on: the plugin's pool is never woken.

    The switch is checked directly. The worker count is checked over several
    runs, since with the pool on a streaming call wakes a helper only when
    the pool has an idle thread at that moment (measured: 2 workers in 3 of 5
    runs of this job), so one run could read 1 by chance.
    """
    import polars_cv
    import polars_cv._lib as lib

    sample = manifest.head(64)
    job = workloads.prepare("polars-cv-streaming-nopool", "w1", "path", sample, 2)
    try:
        assert polars_cv.row_parallelism() is False
        workers = []
        for _ in range(5):
            out = job()
            workers.append(lib._last_split_workers())
        assert workers == [1] * 5
        assert out.shape == (64, workloads.SIZE, workloads.SIZE, 1)
    finally:
        polars_cv.set_row_parallelism(True)


@pytest.mark.skipif(
    workloads.PCV_HAS_ROW_PARALLELISM_SWITCH,
    reason="the installed polars-cv has set_row_parallelism",
)
def test_streaming_nopool_is_not_applicable_before_the_switch() -> None:
    """On a wheel without the switch the variant refuses; it never runs with
    the pool on under a name that says it is off."""
    for w in workloads.WORKLOADS:
        assert ("polars-cv-streaming-nopool", w) in workloads.NOT_APPLICABLE
