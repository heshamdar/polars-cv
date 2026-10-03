"""``evaluate_detections`` against pycocotools' ``COCOeval`` on random data.

Skipped unless pycocotools is installed (it is not a dependency). The data
avoid what the two deliberately do differently: scores are distinct (COCO
orders tied scores by input order, polars-cv makes a tie one PR point), there
are no crowd regions, and every box is in COCO's "all" area range.
"""

from __future__ import annotations

import contextlib
import io
import random

import polars as pl
import pytest

from polars_cv.metrics import evaluate_detections

from ..conftest import plugin_required


def _dataset(seed: int, jitter: float) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    scores = iter(rng.sample(range(1, 10**6), 2000))
    images, gts, dts = [], [], []
    for i in range(25):
        images.append({"id": i, "width": 200, "height": 200})
        for _ in range(rng.randint(0, 5)):
            c = rng.randint(1, 3)
            x, y = rng.uniform(0, 150), rng.uniform(0, 150)
            w, h = rng.uniform(5, 50), rng.uniform(5, 50)
            gts.append(
                {
                    "id": len(gts) + 1,
                    "image_id": i,
                    "category_id": c,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            for _ in range(rng.randint(0, 2)):
                cls = c if rng.random() < 0.9 else rng.randint(1, 3)
                box = [
                    x + rng.gauss(0, jitter),
                    y + rng.gauss(0, jitter),
                    max(1.0, w + rng.gauss(0, jitter)),
                    max(1.0, h + rng.gauss(0, jitter)),
                ]
                dts.append(
                    {
                        "image_id": i,
                        "category_id": cls,
                        "bbox": box,
                        "score": next(scores) / 1e6,
                    }
                )
        for _ in range(rng.randint(0, 3)):
            box = [
                rng.uniform(0, 150),
                rng.uniform(0, 150),
                rng.uniform(5, 40),
                rng.uniform(5, 40),
            ]
            dts.append(
                {
                    "image_id": i,
                    "category_id": rng.randint(1, 3),
                    "bbox": box,
                    "score": next(scores) / 1e6,
                }
            )
    gt = {
        "images": images,
        "annotations": gts,
        "categories": [{"id": c} for c in (1, 2, 3)],
    }
    return gt, dts


@plugin_required
@pytest.mark.parametrize(("seed", "jitter"), [(1, 4.0), (2, 4.0), (11, 1.0), (12, 1.0)])
def test_coco_summary_matches_cocoeval(seed: int, jitter: float) -> None:
    coco_mod = pytest.importorskip("pycocotools.coco")
    cocoeval_mod = pytest.importorskip("pycocotools.cocoeval")

    gt, dts = _dataset(seed, jitter)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = coco_mod.COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dts)
        ev = cocoeval_mod.COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    want = {
        "map": ev.stats[0],
        "map_50": ev.stats[1],
        "map_75": ev.stats[2],
        "mar": ev.stats[8],
    }

    anns = gt["annotations"]
    report = evaluate_detections(
        pl.DataFrame(
            {
                "image_id": [str(d["image_id"]) for d in dts],
                "class_id": [str(d["category_id"]) for d in dts],
                "bbox": [d["bbox"] for d in dts],
                "score": [d["score"] for d in dts],
            }
        ),
        pl.DataFrame(
            {
                "image_id": [str(a["image_id"]) for a in anns],
                "class_id": [str(a["category_id"]) for a in anns],
                "bbox": [a["bbox"] for a in anns],
            }
        ),
        box_format="xywh",
        images=[str(i["id"]) for i in gt["images"]],
    )
    for name, value in want.items():
        assert report.value(name) == pytest.approx(value, abs=1e-12), name
