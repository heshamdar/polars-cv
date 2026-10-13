"""The measurement machinery measures what it says."""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image

from bench import harness
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
