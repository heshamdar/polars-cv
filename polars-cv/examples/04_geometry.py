"""Geometry operations: point, contour, bbox, and domain transitions.

Run:
    uv run python polars-cv/examples/04_geometry.py
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from PIL import Image
from polars_cv import (
    BBOX_SCHEMA,
    CONTOUR_SCHEMA,
    CONTOUR_SET_SCHEMA,
    POINT_SCHEMA,
    POINT_SET_SCHEMA,
    Pipeline,
    numpy_from_struct,
)

OUTPUT_DIR = Path(__file__).parent / "outputs"


def box_contour(x: float, y: float, w: float, h: float) -> dict[str, object]:
    """Create a rectangular contour dictionary."""
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


def make_mask_like_image(size: int = 96) -> list[list[list[int]]]:
    """Create a 3D list mask image for contour extraction demo."""
    arr = np.zeros((size, size, 1), dtype=np.uint8)
    arr[12:44, 10:50, 0] = 255
    arr[52:84, 46:88, 0] = 255
    return arr.tolist()


def to_png_bytes(size: int = 96) -> bytes:
    """Create a simple PNG image for visualization context."""
    y, x = np.indices((size, size))
    rgb = np.stack([x % 255, y % 255, (x + y) % 255], axis=2).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def save_mask(mask: np.ndarray, name: str) -> None:
    """Save a mask image for example outputs."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask.squeeze(-1) if mask.ndim == 3 else mask, cmap="gray")
    ax.axis("off")
    fig.tight_layout()
    out = OUTPUT_DIR / name
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("Saved:", out)


def point_demo(df: pl.DataFrame) -> None:
    """Run point namespace operations."""
    out = df.select(
        normalized=pl.col("point_a").point.normalize(96, 96),
        absolute=pl.col("point_a").point.normalize(96, 96).point.to_absolute(96, 96),
        translated=pl.col("point_a").point.translate(dx=8.0, dy=-3.0),
        scaled=pl.col("point_a").point.scale(sx=1.3, sy=0.7),
        dist=pl.col("point_a").point.distance(pl.col("point_b")),
        manhattan=pl.col("point_a").point.manhattan_distance(pl.col("point_b")),
        angle=pl.col("point_a").point.angle_to(pl.col("point_b")),
        midpoint=pl.col("point_a").point.midpoint(pl.col("point_b")),
        inside=pl.col("point_a").point.within_bbox(pl.col("bbox_a")),
        x=pl.col("point_a").point.x(),
        y=pl.col("point_a").point.y(),
    )
    print("\nPoint ops:")
    print(out)


def contour_demo(df: pl.DataFrame) -> None:
    """Run contour namespace operations."""
    out = df.select(
        area=pl.col("contour_a").contour.area(),
        perimeter=pl.col("contour_a").contour.perimeter(),
        centroid=pl.col("contour_a").contour.centroid(),
        bbox=pl.col("contour_a").contour.bounding_box(),
        hull=pl.col("contour_a").contour.convex_hull(),
        is_convex=pl.col("contour_a").contour.is_convex(),
        iou=pl.col("contour_a").contour.iou(pl.col("contour_b")),
        dice=pl.col("contour_a").contour.dice(pl.col("contour_b")),
        hausdorff=pl.col("contour_a").contour.hausdorff_distance(pl.col("contour_b")),
        contains=pl.col("contour_a").contour.contains_point(pl.col("point_a")),
        pairwise=pl.col("contour_set_pred").contour.pairwise_iou(
            pl.col("contour_set_gt")
        ),
        matches=pl.col("contour_set_pred").contour.match_detections(
            pl.col("contour_set_gt"),
            threshold=0.4,
            scores=pl.col("pred_scores"),
        ),
    )
    print("\nContour ops:")
    print(out.select("area", "perimeter", "iou", "dice", "hausdorff", "contains"))


def bbox_demo(df: pl.DataFrame) -> None:
    """Run bbox namespace operations."""
    out = df.select(
        bbox_iou=pl.col("bbox_set_pred").bbox.pairwise_iou(pl.col("bbox_set_gt")),
        bbox_match=pl.col("bbox_set_pred").bbox.match_detections(
            pl.col("bbox_set_gt"),
            threshold=0.4,
            scores=pl.col("pred_scores"),
        ),
    )
    print("\nBBox ops:")
    print(out)


def transition_demo(df: pl.DataFrame) -> None:
    """Demonstrate buffer<->contour transitions."""
    extracted = df.with_columns(
        extracted=pl.col("mask_list")
        .cv.pipe(
            Pipeline()
            .source("list", dtype="u8")
            .extract_contours(mode="external", method="simple")
        )
        .sink("native"),
    )
    extracted_item = extracted["extracted"].to_list()[0]
    extracted_count = len(extracted_item) if extracted_item is not None else 0
    print("\nExtracted contour count:", extracted_count)

    rasterized = df.with_columns(
        mask=pl.col("contour_a")
        .cv.pipe(
            Pipeline().source(
                "contour", width=96, height=96, fill_value=255, background=0
            )
        )
        .sink("numpy"),
    )
    mask_arr = numpy_from_struct(rasterized["mask"][0], copy=False)
    save_mask(mask_arr, "04_geometry_rasterized_mask.png")


def main() -> None:
    """Run all geometry demos."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    contour_a = box_contour(10.0, 12.0, 36.0, 30.0)
    contour_b = box_contour(24.0, 20.0, 30.0, 34.0)
    contour_set_pred = [contour_a, box_contour(52.0, 54.0, 30.0, 24.0)]
    contour_set_gt = [contour_b, box_contour(54.0, 56.0, 28.0, 20.0)]

    df = pl.DataFrame(
        {
            "image": [to_png_bytes()],
            "point_a": [{"x": 18.0, "y": 20.0}],
            "point_b": [{"x": 70.0, "y": 74.0}],
            "point_set": [[{"x": 10.0, "y": 8.0}, {"x": 22.0, "y": 20.0}]],
            "contour_a": [contour_a],
            "contour_b": [contour_b],
            "contour_set_pred": [contour_set_pred],
            "contour_set_gt": [contour_set_gt],
            "bbox_a": [{"x": 12.0, "y": 12.0, "width": 38.0, "height": 30.0}],
            "bbox_set_pred": [
                [
                    {"x": 10.0, "y": 12.0, "width": 36.0, "height": 30.0},
                    {"x": 52.0, "y": 54.0, "width": 30.0, "height": 24.0},
                ]
            ],
            "bbox_set_gt": [
                [
                    {"x": 24.0, "y": 20.0, "width": 30.0, "height": 34.0},
                    {"x": 54.0, "y": 56.0, "width": 28.0, "height": 20.0},
                ]
            ],
            "pred_scores": [[0.92, 0.63]],
            "mask_list": [make_mask_like_image()],
        },
        schema_overrides={
            "point_a": POINT_SCHEMA,
            "point_b": POINT_SCHEMA,
            "point_set": POINT_SET_SCHEMA,
            "contour_a": CONTOUR_SCHEMA,
            "contour_b": CONTOUR_SCHEMA,
            "contour_set_pred": CONTOUR_SET_SCHEMA,
            "contour_set_gt": CONTOUR_SET_SCHEMA,
            "bbox_a": BBOX_SCHEMA,
            "bbox_set_pred": pl.List(BBOX_SCHEMA),
            "bbox_set_gt": pl.List(BBOX_SCHEMA),
        },
    )

    point_demo(df)
    contour_demo(df)
    bbox_demo(df)
    transition_demo(df)


if __name__ == "__main__":
    main()
