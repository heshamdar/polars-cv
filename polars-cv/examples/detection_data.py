"""Single synthetic dataset generator for detection metric examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SyntheticDetectionConfig:
    """Controls generation of one shared synthetic detection dataset."""

    n_images: int = 60
    image_size: int = 96
    positive_rate: float = 0.7
    max_gt_per_image: int = 3
    miss_rate: float = 0.12
    fp_box_rate: float = 0.55
    localization_jitter: float = 2.0
    heatmap_sigma: float = 6.0
    heatmap_noise_std: float = 0.015
    seed: int = 7


def _gaussian_heatmap(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Create one Gaussian peak as a float32 heatmap."""
    y, x = np.indices((size, size))
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)).astype(np.float32)


def _safe_rect(
    x: int,
    y: int,
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int, int, int]:
    """Clamp rectangle to image bounds."""
    w = max(6, min(width, image_size - 2))
    h = max(6, min(height, image_size - 2))
    x0 = int(np.clip(x, 0, image_size - w))
    y0 = int(np.clip(y, 0, image_size - h))
    return x0, y0, w, h


def _draw_rect(mask: np.ndarray, rect: tuple[int, int, int, int], value: int) -> None:
    """Fill a rectangle into mask in place."""
    x0, y0, w, h = rect
    mask[y0 : y0 + h, x0 : x0 + w] = value


def generate_detection_dataset(config: SyntheticDetectionConfig) -> pl.DataFrame:
    """Generate one dataset usable by contour and bbox matchers."""
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    for idx in range(config.n_images):
        is_positive = bool(rng.random() < config.positive_rate)
        gt_boxes: list[dict[str, float]] = []
        pred_boxes: list[dict[str, float]] = []
        pred_scores: list[float] = []
        gt_mask = np.zeros((config.image_size, config.image_size), dtype=np.uint8)
        heatmap = rng.normal(
            0.0, config.heatmap_noise_std, size=(config.image_size, config.image_size)
        ).astype(np.float32)

        n_targets = (
            int(rng.integers(1, config.max_gt_per_image + 1)) if is_positive else 0
        )
        for _ in range(n_targets):
            w = int(rng.integers(14, 24))
            h = int(rng.integers(14, 24))
            x0 = int(rng.integers(0, config.image_size - w))
            y0 = int(rng.integers(0, config.image_size - h))
            x0, y0, w, h = _safe_rect(x0, y0, w, h, config.image_size)
            gt_boxes.append(
                {"x": float(x0), "y": float(y0), "width": float(w), "height": float(h)}
            )
            _draw_rect(gt_mask, (x0, y0, w, h), value=255)

            if rng.random() >= config.miss_rate:
                px = float(x0 + rng.normal(0.0, config.localization_jitter))
                py = float(y0 + rng.normal(0.0, config.localization_jitter))
                pw = float(w + rng.normal(0.0, config.localization_jitter * 0.35))
                ph = float(h + rng.normal(0.0, config.localization_jitter * 0.35))
                sx, sy, sw, sh = _safe_rect(
                    int(px),
                    int(py),
                    int(max(8.0, pw)),
                    int(max(8.0, ph)),
                    config.image_size,
                )
                score = float(np.clip(rng.normal(0.84, 0.08), 0.05, 0.99))
                pred_boxes.append(
                    {
                        "x": float(sx),
                        "y": float(sy),
                        "width": float(sw),
                        "height": float(sh),
                    }
                )
                pred_scores.append(score)
                heatmap += score * _gaussian_heatmap(
                    config.image_size,
                    sx + sw / 2.0,
                    sy + sh / 2.0,
                    sigma=config.heatmap_sigma,
                )

        n_fp = int(rng.poisson(config.fp_box_rate))
        for _ in range(n_fp):
            w = int(rng.integers(10, 24))
            h = int(rng.integers(10, 24))
            x0 = int(rng.integers(0, config.image_size - w))
            y0 = int(rng.integers(0, config.image_size - h))
            sx, sy, sw, sh = _safe_rect(x0, y0, w, h, config.image_size)
            fp_score = float(np.clip(rng.normal(0.35, 0.12), 0.01, 0.9))
            pred_boxes.append(
                {
                    "x": float(sx),
                    "y": float(sy),
                    "width": float(sw),
                    "height": float(sh),
                }
            )
            pred_scores.append(fp_score)
            heatmap += fp_score * _gaussian_heatmap(
                config.image_size,
                sx + sw / 2.0,
                sy + sh / 2.0,
                sigma=config.heatmap_sigma * 0.9,
            )

        rows.append(
            {
                "image_id": f"img-{idx}",
                "class_id": "lesion",
                "pred_heatmap": np.clip(heatmap, 0.0, 1.0).astype(np.float32).tolist(),
                "gt_mask": gt_mask.tolist(),
                "pred_bboxes": pred_boxes,
                "pred_scores": pred_scores,
                "gt_bboxes": gt_boxes,
            }
        )

    return pl.DataFrame(rows)
