"""Detection metrics with ContourMatcher, BBoxMatcher, and PreMatchedAdapter.

Run:
    uv run python polars-cv/examples/06_detection_metrics.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars_cv import (
    BBoxMatcher,
    ContourMatcher,
    PreMatchedAdapter,
    average_precision,
    confusion_at_threshold,
    f1_at_threshold,
    froc_curve,
    lroc_curve,
    mean_average_precision,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
)
from polars_cv.metrics import bootstrap_pr_auc

OUTPUT_DIR = Path(__file__).parent / "outputs"


def gaussian_heatmap(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Create a smooth heatmap with one Gaussian peak."""
    y, x = np.indices((size, size))
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)).astype(np.float32)


def make_gt_mask(size: int, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Create a binary mask rectangle."""
    arr = np.zeros((size, size), dtype=np.uint8)
    arr[y0 : y0 + h, x0 : x0 + w] = 255
    return arr


def plot_curve(
    df: pl.DataFrame, x_col: str, y_col: str, title: str, out_name: str
) -> None:
    """Save a simple line plot for metric curves."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df[x_col].to_list(), df[y_col].to_list(), marker="o", linewidth=1.6)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / out_name
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Saved:", out)


def contour_matcher_section() -> None:
    """Evaluate contour-based heatmap detections."""
    rows = []
    for idx, (cx, cy, gt_rect) in enumerate(
        [
            (24, 30, (15, 18, 18, 18)),
            (52, 44, (44, 38, 18, 18)),
            (66, 26, (58, 20, 16, 16)),
            (20, 66, (12, 58, 18, 18)),
        ]
    ):
        heat = gaussian_heatmap(96, cx, cy, sigma=8.5)
        gt_mask = make_gt_mask(96, *gt_rect)
        rows.append(
            {
                "image_id": f"img-{idx}",
                "class_id": "lesion",
                "pred_heatmap": heat.tolist(),
                "gt_mask": gt_mask.tolist(),
            }
        )
    df = pl.DataFrame(rows)

    contour_table = ContourMatcher(iou_threshold=0.4, extraction_threshold=0.2).match(
        df,
        pred_col="pred_heatmap",
        gt_col="gt_mask",
        class_col="class_id",
        image_id_col="image_id",
    )

    pr = precision_recall_curve(contour_table, class_id="lesion")
    froc = froc_curve(contour_table)
    print("\nContourMatcher metrics:")
    print("AP:", round(pr.ap(), 4))
    print(
        "FROC AUC:",
        round(froc.auc(), 4),
        "Sens@1FP:",
        round(froc.sensitivity_at_fp(1.0), 4),
    )
    print(
        "Threshold metrics @0.4:",
        "P=",
        round(precision_at_threshold(contour_table, 0.4), 4),
        "R=",
        round(recall_at_threshold(contour_table, 0.4), 4),
        "F1=",
        round(f1_at_threshold(contour_table, 0.4), 4),
    )
    print("Confusion @0.4:", confusion_at_threshold(contour_table, 0.4))
    print(
        "mAP (COCO-style thresholds):", round(mean_average_precision(contour_table), 4)
    )

    # IoU re-thresholding without re-matching.
    strict_table = contour_table.at_iou_threshold(0.6)
    print(
        "AP at stricter IoU 0.6:",
        round(average_precision(strict_table, class_id="lesion"), 4),
    )

    boot = bootstrap_pr_auc(contour_table, n_bootstrap=128, seed=7, class_id="lesion")
    print(
        "Bootstrap AP 95% CI:",
        f"[{boot.ci_lower:.4f}, {boot.ci_upper:.4f}]",
        f"(point={boot.point_estimate:.4f})",
    )

    plot_curve(
        pr.curve,
        "recall",
        "precision",
        "PR Curve (ContourMatcher)",
        "06_pr_contour.png",
    )
    plot_curve(
        froc.curve, "fp_per_image", "sensitivity", "FROC Curve", "06_froc_contour.png"
    )


def bbox_matcher_section() -> None:
    """Evaluate bbox matching and metrics."""
    df = pl.DataFrame(
        {
            "image_id": ["b0", "b1", "b2"],
            "class_id": ["lesion", "lesion", "lesion"],
            "pred_bboxes": [
                [{"x": 12.0, "y": 12.0, "width": 20.0, "height": 20.0}],
                [{"x": 44.0, "y": 40.0, "width": 18.0, "height": 20.0}],
                [{"x": 60.0, "y": 16.0, "width": 18.0, "height": 16.0}],
            ],
            "pred_scores": [[0.92], [0.71], [0.52]],
            "gt_bboxes": [
                [{"x": 14.0, "y": 14.0, "width": 20.0, "height": 20.0}],
                [{"x": 45.0, "y": 42.0, "width": 18.0, "height": 18.0}],
                [{"x": 10.0, "y": 12.0, "width": 18.0, "height": 16.0}],
            ],
        }
    )

    table = BBoxMatcher(iou_threshold=0.5).match(
        df,
        pred_col="pred_bboxes",
        gt_col="gt_bboxes",
        score_col="pred_scores",
        class_col="class_id",
        image_id_col="image_id",
    )
    pr = precision_recall_curve(table, class_id="lesion")
    print("\nBBoxMatcher AP:", round(pr.ap(), 4))


def prematched_section() -> None:
    """Evaluate pre-matched detections using the adapter."""
    pre = pl.DataFrame(
        {
            "image_id": ["p0", "p0", "p1", "p2", "p2"],
            "class_id": ["lesion"] * 5,
            "score": [0.95, 0.42, 0.84, 0.67, 0.33],
            "is_tp": [True, False, True, False, False],
            "n_gts": [1, 1, 1, 0, 0],
            "gt_label": [True, True, True, False, False],
        }
    )
    table = PreMatchedAdapter().match(
        pre,
        pred_col="score",
        gt_col="is_tp",
        image_id_col="image_id",
        class_col="class_id",
        n_gts_col="n_gts",
        gt_label_col="gt_label",
    )
    pr = precision_recall_curve(table, class_id="lesion")
    lroc = lroc_curve(table)
    print("\nPreMatchedAdapter AP:", round(pr.ap(interpolation="11_point"), 4))
    print("PreMatchedAdapter LROC AUC:", round(lroc.auc(), 4))
    plot_curve(
        lroc.curve,
        "fpf",
        "sensitivity",
        "LROC Curve (PreMatched)",
        "06_lroc_prematched.png",
    )


def main() -> None:
    """Run all detection metric sections."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contour_matcher_section()
    bbox_matcher_section()
    prematched_section()


if __name__ == "__main__":
    main()
