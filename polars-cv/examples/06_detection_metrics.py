"""Detection metrics demo on one parameterized synthetic dataset.

Run:
    uv run python polars-cv/examples/06_detection_metrics.py --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from detection_data import SyntheticDetectionConfig, generate_detection_dataset
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
from polars_cv.metrics._types import COL_CLASS_ID, COL_IMAGE_ID

OUTPUT_DIR = Path(__file__).parent / "outputs"


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


def build_prematched_input_from_table(table: object) -> pl.DataFrame:
    """Build adapter input from a matched DetectionTable.

    This demonstrates the intended use of ``PreMatchedAdapter``: when TP/FP
    assignments are already available from an upstream system, you can still
    reuse the metric API.
    """
    detections_df, meta_df = table.collect()  # type: ignore[call-arg]
    return detections_df.join(
        meta_df.select([COL_IMAGE_ID, COL_CLASS_ID, "n_gts", "gt_label"]),
        on=[COL_IMAGE_ID, COL_CLASS_ID],
        how="left",
    ).select(
        image_id=pl.col(COL_IMAGE_ID),
        class_id=pl.col(COL_CLASS_ID),
        score=pl.col("score"),
        is_tp=pl.col("is_tp"),
        n_gts=pl.col("n_gts"),
        gt_label=pl.col("gt_label"),
    )


def contour_matcher_section(df: pl.DataFrame, args: argparse.Namespace) -> object:
    """Evaluate contour-based metrics on shared synthetic data."""
    contour_table = ContourMatcher(
        iou_threshold=args.contour_iou_threshold,
        extraction_threshold=args.extraction_threshold,
        min_contour_area=args.min_contour_area,
        gt_min_contour_area=args.gt_min_contour_area,
    ).match(
        df,
        pred_col="pred_heatmap",
        gt_col="gt_mask",
        class_col="class_id",
        image_id_col="image_id",
    )

    pr = precision_recall_curve(contour_table, class_id="lesion")
    froc = froc_curve(contour_table)
    lroc = lroc_curve(contour_table)
    print("\nContourMatcher metrics:")
    print("AP:", round(pr.ap(), 4))
    print(
        "FROC AUC:",
        round(froc.auc(), 4),
        "\nFROC AUC normalized:",
        round(froc.auc(normalize=True), 4),
        "\nFROC AUC (0, 0.5):",
        round(froc.auc(fp_range=(0, 0.5)), 4),
        "\nFROC AUC (0, 0.5) normalized:",
        round(froc.auc(fp_range=(0, 0.5), normalize=True), 4),
        "\nSens@1FP:",
        round(froc.sensitivity_at_fp(1.0), 4),
    )
    print(
        "LROC AUC:",
        round(lroc.auc(), 4),
        "\nLROC AUC normalized:",
        round(lroc.auc(normalize=True), 4),
        "\nLROC AUC (0, 0.5):",
        round(lroc.auc(fpf_range=(0, 0.5)), 4),
        "\nLROC AUC (0, 0.5) normalized:",
        round(lroc.auc(fpf_range=(0, 0.5), normalize=True), 4),
        "\nSens@0.5FPF:",
        round(lroc.sensitivity_at_fpf(0.5), 4),
    )
    print(
        f"Threshold metrics @{args.score_threshold}:",
        "P=",
        round(precision_at_threshold(contour_table, args.score_threshold), 4),
        "R=",
        round(recall_at_threshold(contour_table, args.score_threshold), 4),
        "F1=",
        round(f1_at_threshold(contour_table, args.score_threshold), 4),
    )
    print(
        f"Confusion @{args.score_threshold}:",
        confusion_at_threshold(contour_table, args.score_threshold),
    )
    print(
        "mAP (COCO-style thresholds):",
        round(mean_average_precision(contour_table), 4),
    )

    strict_table = contour_table.at_iou_threshold(args.strict_iou_threshold)
    print(
        f"AP at stricter IoU {args.strict_iou_threshold}:",
        round(average_precision(strict_table, class_id="lesion"), 4),
    )

    boot = bootstrap_pr_auc(
        contour_table,
        n_bootstrap=args.bootstrap_samples,
        seed=args.seed + 101,
        class_id="lesion",
    )
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
    plot_curve(lroc.curve, "fpf", "sensitivity", "LROC Curve", "06_lroc_contour.png")
    return contour_table


def bbox_matcher_section(df: pl.DataFrame, args: argparse.Namespace) -> object:
    """Evaluate bbox matching and metrics on the same dataset."""
    table = BBoxMatcher(iou_threshold=args.bbox_iou_threshold).match(
        df,
        pred_col="pred_bboxes",
        gt_col="gt_bboxes",
        score_col="pred_scores",
        class_col="class_id",
        image_id_col="image_id",
    )
    pr = precision_recall_curve(table, class_id="lesion")
    print("\nBBoxMatcher AP:", round(pr.ap(), 4))
    return table


def prematched_section(bbox_table: object) -> None:
    """Run metrics through PreMatchedAdapter from already matched detections."""
    pre = build_prematched_input_from_table(bbox_table)
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
    print("\nPreMatchedAdapter metrics (rebuilt from BBoxMatcher TP/FP assignments):")
    print("PreMatchedAdapter AP:", round(pr.ap(interpolation="11_point"), 4))
    print("PreMatchedAdapter LROC AUC:", round(lroc.auc(), 4))
    plot_curve(
        lroc.curve,
        "fpf",
        "sensitivity",
        "LROC Curve (PreMatched)",
        "06_lroc_prematched.png",
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI options for data generation and matcher behavior."""
    parser = argparse.ArgumentParser(
        description="Run detection metric demos on one parameterized synthetic dataset."
    )
    parser.add_argument("--n-images", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--positive-rate", type=float, default=0.7)
    parser.add_argument("--max-gts-per-image", type=int, default=3)
    parser.add_argument(
        "--miss-rate",
        type=float,
        default=0.12,
        help="Probability a GT has no corresponding prediction.",
    )
    parser.add_argument(
        "--fp-box-rate",
        type=float,
        default=0.55,
        help="Poisson lambda for false-positive predicted boxes per image.",
    )
    parser.add_argument("--localization-jitter", type=float, default=2.0)
    parser.add_argument("--heatmap-sigma", type=float, default=6.0)
    parser.add_argument("--heatmap-noise-std", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--contour-iou-threshold", type=float, default=0.4)
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.5)
    parser.add_argument("--extraction-threshold", type=float, default=0.25)
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=5.0,
        help="Prediction contour area filter; >0 suppresses tiny noise contours.",
    )
    parser.add_argument(
        "--gt-min-contour-area",
        type=float,
        default=5.0,
        help="GT contour area filter to avoid fragmented GT over-counting.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--strict-iou-threshold", type=float, default=0.6)
    parser.add_argument("--bootstrap-samples", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    """Run all detection metric sections."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    dataset_config = SyntheticDetectionConfig(
        n_images=args.n_images,
        image_size=args.image_size,
        positive_rate=args.positive_rate,
        max_gt_per_image=args.max_gts_per_image,
        miss_rate=args.miss_rate,
        fp_box_rate=args.fp_box_rate,
        localization_jitter=args.localization_jitter,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_noise_std=args.heatmap_noise_std,
        seed=args.seed,
    )
    df = generate_detection_dataset(dataset_config)
    print("Synthetic dataset configuration:")
    print(" ", dataset_config)
    print(
        "Matcher configuration:",
        {
            "contour_iou_threshold": args.contour_iou_threshold,
            "bbox_iou_threshold": args.bbox_iou_threshold,
            "extraction_threshold": args.extraction_threshold,
            "min_contour_area": args.min_contour_area,
            "gt_min_contour_area": args.gt_min_contour_area,
        },
    )

    contour_matcher_section(df, args)
    bbox_table = bbox_matcher_section(df, args)
    prematched_section(bbox_table)


if __name__ == "__main__":
    main()
