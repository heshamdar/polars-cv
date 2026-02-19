from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import polars as pl
from polars_cv.metrics import FROCAnalyzer, LROCAnalyzer


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
    ax.plot(x, y, marker="o")
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for demo behavior."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    return parser.parse_args()


def main() -> None:
    """Run FROC/LROC demo with heatmap + mask inputs."""
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

    lazy_dataset = dataset.lazy()

    froc = FROCAnalyzer(
        iou_threshold=args.iou_threshold,
        extraction_threshold=args.extraction_threshold,
        min_contour_area=args.min_contour_area,
        auto_resize=auto_resize,
    ).compute(
        lazy_dataset,
        pred_col="pred_heatmap",
        gt_mask_col="gt_mask",
        gt_label_col="gt_label",
        image_id_col="image_id",
        weight_col="sample_weight",
        stratify_col="gt_label",
    )

    lroc = LROCAnalyzer(
        iou_threshold=args.iou_threshold,
        extraction_threshold=args.extraction_threshold,
        min_contour_area=args.min_contour_area,
        auto_resize=auto_resize,
    ).compute(
        lazy_dataset,
        pred_col="pred_heatmap",
        gt_mask_col="gt_mask",
        gt_label_col="gt_label",
        image_id_col="image_id",
        weight_col="sample_weight",
        stratify_col="gt_label",
    )

    print("=== Input Dataset ===")
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

    print("=== FROC Curve Points ===")
    print(froc.curve)
    print()
    print("FROC AUC (full):", round(froc.auc(), 4))
    print("FROC pAUC [0, 2 FP/image]:", round(froc.auc(fp_range=(0.0, 2.0)), 4))
    print(
        "Sensitivity @ FP/image=1.0:",
        round(froc.sensitivity_at_fp(1.0), 4),
    )
    froc_ci = froc.bootstrap_ci(n_bootstrap=200, seed=args.seed, metric="auc")
    print(
        f"FROC AUC bootstrap {froc_ci.confidence:.0%} CI: "
        f"[{froc_ci.ci_lower:.4f}, {froc_ci.ci_upper:.4f}]"
    )
    print()

    print("=== LROC Curve Points ===")
    print(lroc.curve)
    print()
    print("LROC AUC (full):", round(lroc.auc(), 4))
    print("LROC pAUC [0, 1 FPF]:", round(lroc.auc(fpf_range=(0.0, 1.0)), 4))
    print("Sensitivity @ FPF=0.25:", round(lroc.sensitivity_at_fpf(0.25), 4))
    lroc_ci = lroc.bootstrap_ci(n_bootstrap=200, seed=args.seed, metric="auc")
    print(
        f"LROC AUC bootstrap {lroc_ci.confidence:.0%} CI: "
        f"[{lroc_ci.ci_lower:.4f}, {lroc_ci.ci_upper:.4f}]"
    )
    print()

    save_dir = Path(args.save_dir) if args.save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    plot_samples(
        dataset=dataset,
        n_show=args.show_samples,
        save_path=(save_dir / "heatmap_mask_samples.png") if save_dir else None,
        show=not args.no_show,
    )
    froc_curve = froc.curve.sort("fp_per_image")
    plot_curve(
        x=froc_curve["fp_per_image"].cast(pl.Float64).to_list(),
        y=froc_curve["sensitivity"].fill_null(0.0).cast(pl.Float64).to_list(),
        xlabel="False Positives per Image",
        ylabel="Sensitivity",
        title="FROC Curve",
        save_path=(save_dir / "froc_curve.png") if save_dir else None,
        show=not args.no_show,
    )
    lroc_curve = lroc.curve.sort("fpf")
    plot_curve(
        x=lroc_curve["fpf"].cast(pl.Float64).to_list(),
        y=lroc_curve["sensitivity"].fill_null(0.0).cast(pl.Float64).to_list(),
        xlabel="False Positive Fraction",
        ylabel="Sensitivity",
        title="LROC Curve",
        save_path=(save_dir / "lroc_curve.png") if save_dir else None,
        show=not args.no_show,
    )
    if save_dir is not None:
        print(f"Saved visualizations to: {save_dir}")


if __name__ == "__main__":
    main()
