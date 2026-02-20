"""Detection metrics demo — comprehensive reference for polars-cv metrics.

Demonstrates all detection metrics using synthetic data:
  - ContourMatcher (heatmap + binary mask input)
  - PreMatchedAdapter (pre-computed TP/FP input)
  - FROC curve + AUC + bootstrap CI
  - LROC curve + AUC
  - Precision-Recall curve + AP
  - Mean Average Precision (mAP) with COCO IoU thresholds
  - Precision / Recall / F1 at a fixed threshold
  - Confusion matrix
  - Vectorized bootstrap for PR AUC

Run with:
    uv run python notebooks/froc_detection_demo.py
    uv run python notebooks/froc_detection_demo.py --help
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from polars_cv.metrics import (
    ContourMatcher,
    PreMatchedAdapter,
    average_precision,
    bootstrap_pr_auc,
    confusion_at_threshold,
    f1_at_threshold,
    froc_curve,
    lroc_curve,
    mean_average_precision,
    precision_at_threshold,
    precision_recall_curve,
    recall_at_threshold,
)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def add_blob(
    heatmap: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    peak: float,
) -> None:
    """Add a radial confidence blob in-place."""
    yy, xx = np.indices(heatmap.shape, dtype=np.float64)
    distance_sq = (xx - center_x) ** 2 + (yy - center_y) ** 2
    mask = distance_sq <= float(radius**2)
    score = peak * (1.0 - (distance_sq / float(max(radius**2, 1))))
    heatmap[mask] = np.maximum(heatmap[mask], score[mask])


def make_rect_mask(
    width: int,
    height: int,
    x: int,
    y: int,
    box_w: int,
    box_h: int,
) -> np.ndarray:
    """Build one binary mask with a rectangular target."""
    mask = np.zeros((height, width), dtype=np.float64)
    x0 = max(0, min(x, width - 1))
    y0 = max(0, min(y, height - 1))
    x1 = max(x0 + 1, min(x + box_w, width))
    y1 = max(y0 + 1, min(y + box_h, height))
    mask[y0:y1, x0:x1] = 1.0
    return mask


def build_synthetic_dataset(
    n_samples: int = 40,
    seed: int = 7,
    width: int = 64,
    height: int = 64,
) -> pl.DataFrame:
    """Create synthetic heatmap + GT-mask rows for detection metrics."""
    if n_samples <= 0:
        raise ValueError("`n_samples` must be > 0.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for idx in range(n_samples):
        has_target = bool(rng.random() < 0.65)
        weight = float(rng.uniform(0.8, 1.2))

        heatmap = np.zeros((height, width), dtype=np.float64)
        gt_mask = np.zeros((height, width), dtype=np.float64)

        if has_target:
            box_w = int(rng.integers(8, 18))
            box_h = int(rng.integers(8, 18))
            x = int(rng.integers(2, max(width - box_w - 1, 3)))
            y = int(rng.integers(2, max(height - box_h - 1, 3)))
            gt_mask = make_rect_mask(width, height, x, y, box_w, box_h)

            center_x = x + box_w // 2 + int(rng.integers(-3, 4))
            center_y = y + box_h // 2 + int(rng.integers(-3, 4))
            add_blob(
                heatmap,
                center_x=max(0, min(center_x, width - 1)),
                center_y=max(0, min(center_y, height - 1)),
                radius=int(rng.integers(5, 11)),
                peak=float(rng.uniform(0.65, 0.99)),
            )

            # Add occasional off-target false-positive blob.
            if rng.random() < 0.35:
                add_blob(
                    heatmap,
                    center_x=int(rng.integers(4, width - 4)),
                    center_y=int(rng.integers(4, height - 4)),
                    radius=int(rng.integers(4, 9)),
                    peak=float(rng.uniform(0.25, 0.7)),
                )
        else:
            if rng.random() < 0.55:
                add_blob(
                    heatmap,
                    center_x=int(rng.integers(4, width - 4)),
                    center_y=int(rng.integers(4, height - 4)),
                    radius=int(rng.integers(4, 9)),
                    peak=float(rng.uniform(0.2, 0.85)),
                )

        rows.append(
            {
                "image_id": f"case_{idx:04d}",
                "pred_heatmap": heatmap.tolist(),
                "gt_mask": gt_mask.tolist(),
                "gt_label": has_target,
                "sample_weight": weight,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "image_id": pl.String,
            "pred_heatmap": pl.List(pl.List(pl.Float64)),
            "gt_mask": pl.List(pl.List(pl.Float64)),
            "gt_label": pl.Boolean,
            "sample_weight": pl.Float64,
        },
    )


def build_prematched_dataset(seed: int = 42) -> pl.DataFrame:
    """Create a pre-matched dataset for demonstrating PreMatchedAdapter.

    Returns a flat table with per-detection rows and pre-computed TP/FP labels.
    This is the kind of data you'd have if matching was already done externally.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    for img_idx in range(30):
        n_dets = int(rng.integers(0, 6))
        n_gts = int(rng.integers(0, 4))
        has_target = n_gts > 0

        for det_idx in range(n_dets):
            is_tp = bool(det_idx < n_gts and rng.random() < 0.7)
            rows.append(
                {
                    "image_id": f"img_{img_idx:03d}",
                    "confidence": float(rng.uniform(0.1, 0.99)),
                    "is_tp": is_tp,
                    "n_gts": n_gts,
                    "gt_label": has_target,
                }
            )

    return pl.DataFrame(
        rows,
        schema={
            "image_id": pl.String,
            "confidence": pl.Float64,
            "is_tp": pl.Boolean,
            "n_gts": pl.Int64,
            "gt_label": pl.Boolean,
        },
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def maybe_enable_agg_backend() -> None:
    """Enable Agg backend when no display is available."""
    if os.environ.get("MPLBACKEND") is None and os.environ.get("DISPLAY") in {"", None}:
        import matplotlib

        matplotlib.use("Agg")


def plot_samples(
    dataset: pl.DataFrame,
    n_show: int,
    save_path: Path | None,
    show: bool,
) -> None:
    """Plot a few synthetic heatmap/mask samples."""
    if n_show <= 0:
        return
    maybe_enable_agg_backend()
    import matplotlib.pyplot as plt

    sampled = dataset.head(min(n_show, dataset.height))
    n = sampled.height
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes_array = np.array(axes).reshape(-1)

    for ax, row in zip(axes_array, sampled.iter_rows(named=True)):
        heatmap = np.array(row["pred_heatmap"], dtype=np.float64)
        mask = np.array(row["gt_mask"], dtype=np.float64)
        ax.imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0)
        ax.contour(mask, levels=[0.5], colors=["lime"], linewidths=1.5)
        ax.set_title(
            f"{row['image_id']} | label={int(bool(row['gt_label']))} | w={row['sample_weight']:.2f}",
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_array[n:]:
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_curve(
    x: list[float],
    y: list[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path | None,
    show: bool,
) -> None:
    """Plot a generic 2D curve."""
    maybe_enable_agg_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, marker="o", markersize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_multi_curves(
    curves: dict[str, tuple[list[float], list[float]]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path | None,
    show: bool,
) -> None:
    """Plot multiple curves on the same axes."""
    maybe_enable_agg_backend()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (x, y) in curves.items():
        ax.plot(x, y, marker="o", markersize=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Demo sections
# ---------------------------------------------------------------------------


def section_contour_matcher(
    dataset: pl.DataFrame,
    *,
    iou_threshold: float,
    extraction_threshold: float,
    min_contour_area: float,
    auto_resize: bool,
    n_bootstrap: int,
    seed: int,
    save_dir: Path | None,
    show: bool,
) -> None:
    """Demonstrate the ContourMatcher path with all metric types."""
    print("=" * 70)
    print("  CONTOUR MATCHER: heatmap + binary mask -> all metrics")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 1: Build DetectionTable via ContourMatcher
    # ------------------------------------------------------------------
    matcher = ContourMatcher(
        iou_threshold=iou_threshold,
        extraction_threshold=extraction_threshold,
        min_contour_area=min_contour_area,
        auto_resize=auto_resize,
        gt_min_contour_area=max(min_contour_area, 1.0),
    )
    t0 = time.perf_counter()
    table = matcher.match(
        dataset.lazy(),
        pred_col="pred_heatmap",
        gt_col="gt_mask",
        image_id_col="image_id",
        weight_col="sample_weight",
    )
    match_time = time.perf_counter() - t0
    print(f"  Matching completed in {match_time:.3f}s")
    print()

    # ------------------------------------------------------------------
    # Step 2: FROC
    # ------------------------------------------------------------------
    print("--- FROC Curve ---")
    froc = froc_curve(table)
    print(froc.curve)
    print()
    print(f"  FROC AUC (full):               {froc.auc():.4f}")
    print(f"  FROC pAUC [0, 2 FP/image]:     {froc.auc(fp_range=(0.0, 2.0)):.4f}")
    print(f"  Sensitivity @ 1.0 FP/image:    {froc.sensitivity_at_fp(1.0):.4f}")

    if n_bootstrap > 0:
        froc_ci = froc.bootstrap_ci(n_bootstrap=n_bootstrap, seed=seed, metric="auc")
        print(
            f"  FROC AUC {froc_ci.confidence:.0%} CI:  "
            f"[{froc_ci.ci_lower:.4f}, {froc_ci.ci_upper:.4f}]"
        )
    print()

    print("  FROC Summary Table:")
    print(froc.summary_table())
    print()

    # ------------------------------------------------------------------
    # Step 3: LROC
    # ------------------------------------------------------------------
    print("--- LROC Curve ---")
    lroc = lroc_curve(table)
    print(lroc.curve)
    print()
    print(f"  LROC AUC (full):           {lroc.auc():.4f}")
    print(f"  Sensitivity @ FPF=0.25:    {lroc.sensitivity_at_fpf(0.25):.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Precision-Recall
    # ------------------------------------------------------------------
    print("--- Precision-Recall Curve ---")
    pr = precision_recall_curve(table)
    ap_all = pr.ap(interpolation="all_points")
    ap_11pt = pr.ap(interpolation="11_point")
    print(f"  Detections in PR curve:    {pr.curve.height}")
    print(f"  Total GTs:                 {pr.total_gts}")
    print(f"  AP (all-points):           {ap_all:.4f}")
    print(f"  AP (11-point):             {ap_11pt:.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 5: Average Precision convenience
    # ------------------------------------------------------------------
    print("--- Average Precision (convenience function) ---")
    ap_val = average_precision(table)
    print(f"  AP:  {ap_val:.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 6: Mean Average Precision with COCO IoU thresholds
    # ------------------------------------------------------------------
    print("--- Mean Average Precision (mAP) ---")
    coco_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]
    map_coco = mean_average_precision(table, iou_thresholds=coco_thresholds)
    map_50 = mean_average_precision(table, iou_thresholds=[0.5])
    map_75 = mean_average_precision(table, iou_thresholds=[0.75])
    print(f"  mAP@[.5:.95]:  {map_coco:.4f}")
    print(f"  mAP@0.50:      {map_50:.4f}")
    print(f"  mAP@0.75:      {map_75:.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 7: Precision, Recall, F1 at threshold
    # ------------------------------------------------------------------
    for thr in [0.3, 0.5, 0.7]:
        p = precision_at_threshold(table, thr)
        r = recall_at_threshold(table, thr)
        f1 = f1_at_threshold(table, thr)
        print(f"  @ threshold={thr:.1f}:  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print()

    # ------------------------------------------------------------------
    # Step 8: Confusion matrix
    # ------------------------------------------------------------------
    print("--- Confusion Matrix @ threshold=0.5 ---")
    cm = confusion_at_threshold(table, threshold=0.5)
    print(f"  TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}")
    print()

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    froc_sorted = froc.curve.sort("fp_per_image")
    plot_curve(
        x=froc_sorted["fp_per_image"].cast(pl.Float64).to_list(),
        y=froc_sorted["sensitivity"].fill_null(0.0).cast(pl.Float64).to_list(),
        xlabel="False Positives per Image",
        ylabel="Sensitivity",
        title="FROC Curve (ContourMatcher)",
        save_path=(save_dir / "froc_curve.png") if save_dir else None,
        show=show,
    )
    lroc_sorted = lroc.curve.sort("fpf")
    plot_curve(
        x=lroc_sorted["fpf"].cast(pl.Float64).to_list(),
        y=lroc_sorted["sensitivity"].fill_null(0.0).cast(pl.Float64).to_list(),
        xlabel="False Positive Fraction",
        ylabel="Sensitivity",
        title="LROC Curve (ContourMatcher)",
        save_path=(save_dir / "lroc_curve.png") if save_dir else None,
        show=show,
    )
    pr_sorted = pr.curve.sort("recall")
    plot_curve(
        x=pr_sorted["recall"].cast(pl.Float64).to_list(),
        y=pr_sorted["precision"].cast(pl.Float64).to_list(),
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-Recall Curve (AP={ap_all:.3f})",
        save_path=(save_dir / "pr_curve.png") if save_dir else None,
        show=show,
    )


def section_prematched(
    *,
    seed: int,
    n_bootstrap: int,
    save_dir: Path | None,
    show: bool,
) -> None:
    """Demonstrate the PreMatchedAdapter path."""
    print("=" * 70)
    print("  PRE-MATCHED ADAPTER: pre-computed TP/FP -> metrics")
    print("=" * 70)
    print()

    data = build_prematched_dataset(seed=seed)
    print("  Input data (first 10 rows):")
    print(data.head(10))
    print()

    adapter = PreMatchedAdapter()
    table = adapter.match(
        data,
        pred_col="confidence",
        gt_col="is_tp",
        image_id_col="image_id",
        n_gts_col="n_gts",
        gt_label_col="gt_label",
    )

    # PR curve
    pr = precision_recall_curve(table)
    ap_val = average_precision(table)
    print(f"  Detections:  {pr.curve.height}")
    print(f"  AP:          {ap_val:.4f}")
    print()

    # Threshold metrics
    for thr in [0.3, 0.5, 0.7]:
        p = precision_at_threshold(table, thr)
        r = recall_at_threshold(table, thr)
        f1 = f1_at_threshold(table, thr)
        print(f"  @ threshold={thr:.1f}:  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print()

    # Confusion
    cm = confusion_at_threshold(table, threshold=0.5)
    print(f"  Confusion @ 0.5:  TP={cm['tp']}  FP={cm['fp']}  FN={cm['fn']}")
    print()

    # Vectorized bootstrap
    if n_bootstrap > 0:
        print("--- Vectorized Bootstrap (PR AUC) ---")
        t0 = time.perf_counter()
        bs = bootstrap_pr_auc(table, n_bootstrap=n_bootstrap, seed=seed)
        bs_time = time.perf_counter() - t0
        print(f"  Point estimate:  {bs.point_estimate:.4f}")
        print(f"  {bs.confidence:.0%} CI:  [{bs.ci_lower:.4f}, {bs.ci_upper:.4f}]")
        print(f"  Time ({n_bootstrap} iterations):  {bs_time:.3f}s")
        print()

    # PR plot
    if pr.curve.height > 0:
        pr_sorted = pr.curve.sort("recall")
        plot_curve(
            x=pr_sorted["recall"].cast(pl.Float64).to_list(),
            y=pr_sorted["precision"].cast(pl.Float64).to_list(),
            xlabel="Recall",
            ylabel="Precision",
            title=f"PR Curve — PreMatchedAdapter (AP={ap_val:.3f})",
            save_path=(save_dir / "pr_curve_prematched.png") if save_dir else None,
            show=show,
        )


def section_iou_rethresholding(
    dataset: pl.DataFrame,
    *,
    iou_threshold: float,
    extraction_threshold: float,
    min_contour_area: float,
    auto_resize: bool,
    save_dir: Path | None,
    show: bool,
) -> None:
    """Show how IoU re-thresholding avoids re-matching."""
    print("=" * 70)
    print("  IOU RE-THRESHOLDING: match once, evaluate at multiple IoUs")
    print("=" * 70)
    print()

    matcher = ContourMatcher(
        iou_threshold=iou_threshold,
        extraction_threshold=extraction_threshold,
        min_contour_area=min_contour_area,
        auto_resize=auto_resize,
        gt_min_contour_area=max(min_contour_area, 1.0),
    )
    table = matcher.match(
        dataset.lazy(),
        pred_col="pred_heatmap",
        gt_col="gt_mask",
        image_id_col="image_id",
    )

    curves_data: dict[str, tuple[list[float], list[float]]] = {}
    for iou_thr in [0.1, 0.25, 0.5, 0.75]:
        rethresholded = table.at_iou_threshold(iou_thr)
        pr = precision_recall_curve(rethresholded)
        ap = pr.ap()
        if pr.curve.height > 0:
            pr_sorted = pr.curve.sort("recall")
            curves_data[f"IoU={iou_thr} (AP={ap:.3f})"] = (
                pr_sorted["recall"].cast(pl.Float64).to_list(),
                pr_sorted["precision"].cast(pl.Float64).to_list(),
            )
        print(f"  IoU={iou_thr:.2f}:  AP={ap:.4f}")
    print()

    if curves_data:
        plot_multi_curves(
            curves_data,
            xlabel="Recall",
            ylabel="Precision",
            title="PR Curves at Different IoU Thresholds (single matching pass)",
            save_path=(save_dir / "pr_iou_comparison.png") if save_dir else None,
            show=show,
        )


def section_scaling_benchmark(
    *,
    sizes: list[int],
    seed: int,
    width: int,
    height: int,
    iou_threshold: float,
    extraction_threshold: float,
    min_contour_area: float,
    auto_resize: bool,
    profile_csv: str | None,
) -> None:
    """Benchmark scaling trends across dataset sizes."""
    print("=" * 70)
    print("  SCALING BENCHMARK")
    print("=" * 70)
    print()

    bench_rows: list[dict[str, float | int]] = []
    for idx, n_samples in enumerate(sizes):
        dataset = build_synthetic_dataset(
            n_samples=n_samples,
            seed=seed + idx,
            width=width,
            height=height,
        )
        matcher = ContourMatcher(
            iou_threshold=iou_threshold,
            extraction_threshold=extraction_threshold,
            min_contour_area=min_contour_area,
            auto_resize=auto_resize,
            gt_min_contour_area=max(min_contour_area, 1.0),
        )

        t0 = time.perf_counter()
        table = matcher.match(
            dataset.lazy(),
            pred_col="pred_heatmap",
            gt_col="gt_mask",
            image_id_col="image_id",
            weight_col="sample_weight",
        )
        froc_result = froc_curve(table)
        froc_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        # Reuse table for LROC
        lroc_result = lroc_curve(table)
        lroc_time = time.perf_counter() - t1

        t2 = time.perf_counter()
        pr_result = precision_recall_curve(table)
        ap_val = pr_result.ap()
        pr_time = time.perf_counter() - t2

        bench_rows.append(
            {
                "n_samples": n_samples,
                "match_and_froc_s": froc_time,
                "lroc_s": lroc_time,
                "pr_and_ap_s": pr_time,
                "froc_auc": float(froc_result.auc()),
                "lroc_auc": float(lroc_result.auc()),
                "ap": float(ap_val),
            }
        )

    bench_df = pl.DataFrame(bench_rows)
    print(bench_df)
    print()

    if profile_csv is not None:
        bench_df.write_csv(profile_csv)
        print(f"  Saved CSV to: {profile_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for demo behavior."""
    parser = argparse.ArgumentParser(
        description="Detection metrics demo — comprehensive reference for polars-cv metrics."
    )
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--extraction-threshold", type=float, default=0.1)
    parser.add_argument("--min-contour-area", type=float, default=0.0)
    parser.add_argument("--auto-resize", action="store_true", default=True)
    parser.add_argument("--no-auto-resize", action="store_true")
    parser.add_argument("--show-samples", type=int, default=6)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--profile-scaling", action="store_true")
    parser.add_argument("--profile-sizes", type=str, default="500,1000,2000")
    parser.add_argument("--profile-csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the comprehensive detection metrics demo."""
    args = parse_args()
    auto_resize = False if args.no_auto_resize else args.auto_resize

    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(140)
    pl.Config.set_float_precision(4)

    dataset = build_synthetic_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )

    save_dir = Path(args.save_dir) if args.save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    # Dataset summary
    print("=" * 70)
    print("  DATASET SUMMARY")
    print("=" * 70)
    print()
    print(
        dataset.select(
            "image_id",
            "gt_label",
            "sample_weight",
            gt_pixels=pl.col("gt_mask")
            .list.eval(pl.element().list.sum().cast(pl.Float64))
            .list.sum(),
        )
    )
    print()

    # Sample visualizations
    plot_samples(
        dataset=dataset,
        n_show=args.show_samples,
        save_path=(save_dir / "heatmap_mask_samples.png") if save_dir else None,
        show=show,
    )

    # Section 1: ContourMatcher with all metrics
    section_contour_matcher(
        dataset,
        iou_threshold=args.iou_threshold,
        extraction_threshold=args.extraction_threshold,
        min_contour_area=args.min_contour_area,
        auto_resize=auto_resize,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        save_dir=save_dir,
        show=show,
    )

    # Section 2: PreMatchedAdapter
    section_prematched(
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        save_dir=save_dir,
        show=show,
    )

    # Section 3: IoU re-thresholding
    section_iou_rethresholding(
        dataset,
        iou_threshold=args.iou_threshold,
        extraction_threshold=args.extraction_threshold,
        min_contour_area=args.min_contour_area,
        auto_resize=auto_resize,
        save_dir=save_dir,
        show=show,
    )

    if save_dir is not None:
        print(f"  Saved all visualizations to: {save_dir}")

    # Optional: scaling benchmark
    if args.profile_scaling:
        sizes = [int(token.strip()) for token in args.profile_sizes.split(",") if token]
        section_scaling_benchmark(
            sizes=sizes,
            seed=args.seed,
            width=args.width,
            height=args.height,
            iou_threshold=args.iou_threshold,
            extraction_threshold=args.extraction_threshold,
            min_contour_area=args.min_contour_area,
            auto_resize=auto_resize,
            profile_csv=args.profile_csv,
        )


if __name__ == "__main__":
    main()
