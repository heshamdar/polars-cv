"""Every implementation computes what the PIL reference computes.

A benchmark is only a comparison if the libraries do the same work, so each
(library, workload) is run on a sample of COCO images, from both in-memory
bytes and file paths, and held to the PIL reference within the tolerance
``bench.parity.TOLERANCES`` documents for it. Resize kernels, JPEG decoders and
hash implementations differ between libraries, so the tolerances are not
zero. Each records why it is what it is, and the measured differences are
written to ``results/parity.json`` for the benchmark README.

A library that cannot do a workload raises ``NotApplicable`` with its reason.
It must never quietly run a different operation, so the N/A table is tested
too.

The sample is 40 images: the 10 single-channel images in val2017, 26 others,
and 4 of those repeated under new ids, so workload 3 has duplicate groups to
find.
"""

from __future__ import annotations

import json
import re

import polars as pl
import pytest

from bench import parity, workloads
from bench.workloads import LIBRARIES, NOT_APPLICABLE, WORKLOADS, NotApplicable

REPORT: dict[str, dict] = {}


@pytest.fixture(scope="session")
def sample(manifest: pl.DataFrame) -> pl.DataFrame:
    gray = manifest.filter(pl.col("channels") == 1)
    color = manifest.filter(pl.col("channels") == 3).head(26)
    base = pl.concat([gray, color])
    dupes = base.slice(10, 4)
    return pl.concat([base, dupes]).drop("id").with_row_index("id")


@pytest.fixture(scope="session")
def reference(sample: pl.DataFrame) -> dict:
    return {
        w: workloads.to_common(w, workloads.prepare("pil", w, "path", sample, 1)())
        for w in WORKLOADS
    }


@pytest.fixture(scope="session", autouse=True)
def write_report():
    yield
    if REPORT:
        parity.write_report(REPORT)


CASES = [
    (lib, w, src)
    for lib in LIBRARIES
    for w in WORKLOADS
    for src in ("bytes", "path")
    if (lib, w) not in NOT_APPLICABLE
]


@pytest.mark.parametrize(("library", "workload", "source"), CASES)
def test_matches_reference(library, workload, source, sample, reference) -> None:
    threads = 1 if library in workloads.SINGLE_THREAD_ONLY else 2
    out = workloads.to_common(
        workload, workloads.prepare(library, workload, source, sample, threads)()
    )
    diff = parity.compare(workload, reference[workload], out)
    REPORT[f"{library}/{workload}/{source}"] = diff
    tol = parity.TOLERANCES[(library, workload)]
    failures = parity.check(diff, tol)
    assert not failures, f"{library} {workload} {source}: {failures}; diff={diff}"


@pytest.mark.parametrize(("library", "workload"), sorted(NOT_APPLICABLE))
def test_not_applicable_refuses(library, workload, sample) -> None:
    """An N/A pair raises with its reason instead of running something else."""
    with pytest.raises(
        NotApplicable, match=re.escape(NOT_APPLICABLE[(library, workload)])
    ):
        workloads.prepare(library, workload, "path", sample, 1)


def test_every_pair_is_covered() -> None:
    """Each (library, workload) has a tolerance or an N/A reason, never neither."""
    for lib in LIBRARIES:
        for w in WORKLOADS:
            assert ((lib, w) in parity.TOLERANCES) != ((lib, w) in NOT_APPLICABLE), (
                lib,
                w,
            )


def test_reference_is_exact_against_itself(sample, reference) -> None:
    """The comparison reports zero difference for identical outputs.

    Guards ``compare`` against a metric that is always small.
    """
    for w in WORKLOADS:
        diff = parity.compare(w, reference[w], reference[w])
        assert not parity.check(diff, parity.EXACT), (w, diff)


def test_comparison_detects_a_wrong_output(sample, reference) -> None:
    """A different operation (no grayscale, wrong size) is caught."""
    wrong = workloads.prepare(
        "pil", "w4", "path", sample.with_columns(target_h=pl.col("target_h") + 1), 1
    )()
    diff = parity.compare("w4", reference["w4"], workloads.to_common("w4", wrong))
    assert parity.check(diff, parity.TOLERANCES[("polars-cv-eager", "w4")])
    flipped = reference["w1"][:, ::-1, :]
    diff = parity.compare("w1", reference["w1"], flipped)
    assert parity.check(diff, parity.TOLERANCES[("polars-cv-eager", "w1")])
    json.dumps(diff)  # the report must serialise


# --- Workload 3 on the whole set: no false duplicates ---------------------------


@pytest.fixture(scope="session")
def full_groups(manifest: pl.DataFrame) -> dict:
    """Duplicate groups each library's pHash finds over all 5,000 images."""
    return {}


def _groups(library: str, manifest: pl.DataFrame, cache: dict) -> set:
    if library not in cache:
        threads = workloads.cpu_count()
        if library in workloads.SINGLE_THREAD_ONLY:
            threads = 1
        job = workloads.prepare(library, "w3", "path", manifest, threads)
        cache[library] = workloads.to_common("w3", job())[1]
    return cache[library]


@pytest.mark.slow
@pytest.mark.parametrize(
    "library",
    [
        pytest.param(
            "polars-cv-eager",
            marks=pytest.mark.xfail(
                not parity.PCV_STANDARD_PHASH,
                strict=True,
                reason=(
                    "polars-cv < 0.38 perceptual_hash() gives 236 distinct COCO "
                    "images identical hashes (64 groups); distinct images average "
                    "18 differing bits, against ~31 for imagehash, OpenCV and Daft"
                ),
            ),
        ),
        "opencv-mp",
        "daft",
    ],
)
def test_w3_full_set_finds_no_false_duplicates(library, manifest, full_groups) -> None:
    """COCO val2017 has no duplicate images; the reference finds no groups."""
    reference = _groups("pil-mp", manifest, full_groups)
    assert reference == set()
    found = _groups(library, manifest, full_groups)
    REPORT[f"{library}/w3/full-set-false-duplicate-groups"] = {"groups": len(found)}
    assert found == reference, f"{len(found)} false duplicate groups"
