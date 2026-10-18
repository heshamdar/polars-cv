"""Executable specifications for defects that are verified but not yet fixed.

Every test in this module is ``xfail(strict=True)``. Each one:

* describes a defect that has been **confirmed against running code or source**,
  not a suspicion;
* asserts the behaviour the codebase *should* have, so it fails today; and
* names, in its docstring, what "fixed" looks like.

``strict=True`` is the point. When someone lands the fix, the test XPASSes and
the suite goes **red** — which is the signal to delete the marker and let the
test join the suite proper. A backlog that lives in prose is a backlog that
rots; this one cannot silently become stale in either direction. It is the same
reasoning as `AGENTS.md`'s "The Single-Authority Refactor: What Was Done, What
Is Left" section, made executable.

These are the items recorded as deferred in that review: the ones where a fix is
a design change rather than a correction, and so wants its own commit. Adding a
test here is not a way to avoid fixing something — it is a way to stop the
knowledge evaporating between sessions.

Do **not** put a flaky or environment-dependent test here. `xfail` marks "known
broken", never "sometimes fails".
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tests.conftest import plugin_required

# Each gap carries its own lane: a source scan is `structural` (pre-commit runs
# it with no compiled extension), a runtime one `plugin_required`. Each is
# listed, with its fix, under its id in the root `ISSUES.md`.


def _gap(reason: str) -> pytest.MarkDecorator:
    """Mark a verified, unfixed defect. Strict, so a fix fails the suite.

    Only an ``AssertionError`` is the expected failure: every gap states the
    defect as an ``assert``, so a gap that breaks for any other reason (a
    renamed helper, a fixture that no longer builds) fails the suite rather
    than reading as the defect.
    """
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)


# Two detections on one image, each overlapping its ground-truth box.
_PREDS = pl.DataFrame(
    {
        "image_id": ["a", "a"],
        "class_id": ["c", "c"],
        "score": [0.9, 0.8],
        "bbox": [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 10.0, 10.0]],
    }
)
_GTS = pl.DataFrame(
    {
        "image_id": ["a", "a"],
        "class_id": ["c", "c"],
        "bbox": [[0.0, 0.0, 10.0, 11.0], [20.0, 20.0, 10.0, 14.0]],
    }
)


@plugin_required
@_gap("CR-77: sweep thresholds are compared by exact float equality")
def test_a_swept_threshold_reports_its_metrics() -> None:
    """``np.arange(0.5, 0.96, 0.05)`` stores 0.75 as ``0.7500000000000002``, so
    the report drops ``map_75``. Fixed when a sweep normalises its thresholds
    once where it is built, and this report names ``map_75``."""
    from polars_cv.metrics import evaluate_detections

    report = evaluate_detections(
        _PREDS,
        _GTS,
        box_format="xywh",
        iou_thresholds=list(np.arange(0.5, 0.96, 0.05)),
    )
    assert "map_75" in report.metrics


@plugin_required
@_gap("CR-76: negative-size boxes are accepted and score as false positives")
def test_a_negative_size_box_is_refused() -> None:
    """``xywh`` data passed as ``"xyxy"`` gives boxes of negative size, which
    the bbox reader accepts and the matcher scores as 0-IoU false positives.
    Fixed when the reader refuses ``width < 0`` / ``height < 0`` as it refuses
    a non-finite field."""
    from polars_cv.metrics import match_detections

    swapped = _PREDS.with_columns(
        pl.Series("bbox", [[10.0, 10.0, 0.0, 0.0], [30.0, 30.0, 20.0, 20.0]])
    )
    raised = False
    try:
        match_detections(swapped, _GTS, box_format="xyxy").detections.collect()
    except (pl.exceptions.ComputeError, ValueError):
        raised = True
    assert raised, "a box of negative size was accepted"


@plugin_required
@_gap("CR-115: average/difference hashes pack bits LSB-first, perceptual MSB-first")
def test_every_hash_algorithm_packs_bits_msb_first() -> None:
    """The three algorithms share one output column type, so one bit order.
    pHash is MSB-first (as documented), aHash and dHash LSB-first, so each is
    0 bits from imagehash read one way and tens of bits off read the other.
    Fixed when every algorithm is MSB-first: then each matches imagehash's
    hash read MSB-first."""
    import io

    imagehash = pytest.importorskip("imagehash")
    from PIL import Image

    from polars_cv import Pipeline

    yy, xx = np.mgrid[0:128, 0:128]
    img = ((xx * 2 + yy) % 256).astype(np.uint8)
    img[40:90, 30:70] = 30
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    df = pl.DataFrame({"i": [buf.getvalue()]})
    refs = {
        "perceptual": imagehash.phash,
        "average": imagehash.average_hash,
        "difference": imagehash.dhash,
    }
    for algorithm, ref in refs.items():
        pipe = Pipeline().source("image_bytes").perceptual_hash(algorithm=algorithm)
        h = df.select(pl.col("i").cv.pipe(pipe).sink("list")).item().to_list()
        bits = np.unpackbits(np.array(h, dtype=np.uint8))
        expected = ref(Image.open(io.BytesIO(buf.getvalue()))).hash.flatten()
        assert int((bits != expected).sum()) <= 2, algorithm
