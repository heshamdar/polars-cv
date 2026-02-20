"""Reductions and feature extraction with polars-cv.

Run:
    uv run python polars-cv/examples/05_reductions_and_features.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from polars_cv import CONTOUR_SCHEMA, Pipeline, mask_dice, mask_iou, numpy_from_struct

OUTPUT_DIR = Path(__file__).parent / "outputs"


def make_heatmap(size: int = 96) -> np.ndarray:
    """Create a synthetic heatmap with two hotspots."""
    y, x = np.indices((size, size))
    center_1 = np.exp(-((x - 26) ** 2 + (y - 30) ** 2) / (2 * 12.0**2))
    center_2 = np.exp(-((x - 68) ** 2 + (y - 62) ** 2) / (2 * 10.0**2))
    return (center_1 + 0.9 * center_2).astype(np.float32)


def contour_rect(x: float, y: float, w: float, h: float) -> dict[str, object]:
    """Create a contour rectangle dictionary."""
    return {
        "exterior": [
            {"x": x, "y": y},
            {"x": x + w, "y": y},
            {"x": x + w, "y": y + h},
            {"x": x, "y": y + h},
        ],
        "holes": [],
        "is_closed": True,
    }


def save_heatmap_image(arr: np.ndarray, output: Path) -> None:
    """Save a heatmap visualization."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("Synthetic heatmap")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    print("Saved:", output)


def main() -> None:
    """Run reduction and feature extraction examples."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap = make_heatmap()
    save_heatmap_image(heatmap, OUTPUT_DIR / "05_heatmap_input.png")

    gt_contour = contour_rect(16.0, 14.0, 30.0, 34.0)
    pred_contour = contour_rect(20.0, 20.0, 34.0, 32.0)

    df = pl.DataFrame(
        {
            "heatmap": [heatmap.tolist()],
            "gt_contour": [gt_contour],
            "pred_contour": [pred_contour],
        },
        schema_overrides={"gt_contour": CONTOUR_SCHEMA, "pred_contour": CONTOUR_SCHEMA},
    )

    heat_expr = (
        pl.col("heatmap").cv.pipe(Pipeline().source("list", dtype="f32")).alias("heat")
    )
    gt_mask_expr = (
        pl.col("gt_contour")
        .cv.pipe(
            Pipeline().source(
                "contour", width=96, height=96, fill_value=255, background=0
            ),
        )
        .alias("gt_mask")
    )
    pred_mask_expr = (
        pl.col("pred_contour")
        .cv.pipe(
            Pipeline().source(
                "contour", width=96, height=96, fill_value=255, background=0
            ),
        )
        .alias("pred_mask")
    )

    stats_lazy = heat_expr.statistics_lazy(include=["mean", "std", "min", "max", "sum"])
    reduced = df.with_columns(
        summary=stats_lazy.sink(
            {
                "stat_mean": "native",
                "stat_std": "native",
                "stat_min": "native",
                "stat_max": "native",
                "stat_sum": "native",
            }
        )
    )
    print("\nReduction summary struct:")
    print(reduced.select("summary"))

    # TODO: consolidate these into one with_columns when sink contract mismatches are resolved.
    for name, expr in [
        ("shape_vec", heat_expr.pipe(Pipeline().extract_shape()).sink("native")),
        (
            "histogram",
            heat_expr.pipe(Pipeline().histogram(bins=8, output="normalized")).sink(
                "native"
            ),
        ),
        ("row_max", heat_expr.pipe(Pipeline().reduce_max(axis=1)).sink("numpy")),
        ("row_min", heat_expr.pipe(Pipeline().reduce_min(axis=1)).sink("numpy")),
        ("argmax_col", heat_expr.pipe(Pipeline().reduce_argmax(axis=1)).sink("numpy")),
        ("argmin_col", heat_expr.pipe(Pipeline().reduce_argmin(axis=1)).sink("numpy")),
    ]:
        try:
            op_df = df.with_columns(**{name: expr})
            print(f"{name} computed; dtype={op_df.schema[name]}")
        except Exception as exc:  # noqa: BLE001
            print(f"{name} unavailable in this build:", exc)

    try:
        percentile_df = df.with_columns(
            percentile_95=heat_expr.pipe(Pipeline().reduce_percentile(q=95.0)).sink(
                "native"
            )
        )
        print("Percentile(95):", percentile_df["percentile_95"][0])
    except Exception as exc:  # noqa: BLE001
        # TODO: remove this fallback once reduce_percentile dtype contract mismatch is fixed.
        print("Percentile(95) unavailable in this build:", exc)

    # Contour-region scoring from the heatmap.
    label_scored = df.with_columns(
        region_scores=heat_expr.label_reduce(
            contours=pl.concat_list([pl.col("gt_contour"), pl.col("pred_contour")]),
            reduction="mean",
            region_mode="interior",
        ).sink("native"),
    )
    print("\nLabel-reduced contour scores:", label_scored["region_scores"][0])

    # Inside/outside masked features.
    inside_stats = heat_expr.apply_mask(gt_mask_expr).statistics(
        include=["mean", "max"]
    )
    outside_stats = heat_expr.apply_mask(gt_mask_expr, invert=True).statistics(
        include=["mean", "max"]
    )
    mask_metrics = df.with_columns(
        inside=inside_stats,
        outside=outside_stats,
        mask_iou=mask_iou(pred_mask_expr, gt_mask_expr),
        mask_dice=mask_dice(pred_mask_expr, gt_mask_expr),
    )
    print("\nMasked features and mask similarity:")
    print(mask_metrics.select("inside", "outside", "mask_iou", "mask_dice"))

    # Save GT mask output image.
    gt_mask_df = df.with_columns(mask=gt_mask_expr.sink("numpy"))
    gt_mask_arr = numpy_from_struct(gt_mask_df["mask"][0], copy=False)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(gt_mask_arr.squeeze(-1), cmap="gray")
    ax.set_title("GT rasterized mask")
    ax.axis("off")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "05_gt_mask.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
