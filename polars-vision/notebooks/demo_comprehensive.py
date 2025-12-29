# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🖼️ polars-vision: Comprehensive Demo
#
# This notebook provides a complete demonstration of the **polars-vision** plugin - a high-performance vision/array processing plugin for Polars DataFrames.
#
# ## What is polars-vision?
#
# polars-vision enables:
# - **Lazy, zero-copy image processing** on DataFrame columns
# - **Composable pipelines** that fuse multiple operations into single plugin calls
# - **Dynamic parameters** using Polars expressions for per-row customization
# - **Geometry operations** for contours, points, and bounding boxes
# - **Seamless ML integration** with NumPy, PyTorch, and other frameworks
#
# The plugin leverages **view-buffer**, a Rust crate providing stride-aware tensor operations with automatic kernel fusion.
#
# ---
#
# ## Table of Contents
#
# 1. [Setup & Imports](#1-setup--imports)
# 2. [Basic Pipeline Operations](#2-basic-pipeline-operations)
# 3. [DType Promotion & Normalization](#3-dtype-promotion--normalization)
# 4. [Dynamic Parameters with Expressions](#4-dynamic-parameters-with-expressions)
# 5. [Geometry Operations](#5-geometry-operations)
# 6. [Lazy Pipeline Composition](#6-lazy-pipeline-composition)
# 7. [Multi-Output Pipelines](#7-multi-output-pipelines)
# 8. [ML Workflow: IoU Calculation](#8-ml-workflow-iou-calculation)
# 9. [Lazy Scalability Demo](#9-lazy-scalability-demo)
# 10. [PyTorch Integration](#10-pytorch-integration)
# 11. [Conclusion](#11-conclusion)

# %% [markdown]
# ## 1. Setup & Imports
#
# First, let's import the necessary packages and set up helper functions for displaying images.

# %%
# Core imports
import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from PIL import Image
import matplotlib.pyplot as plt

# polars-vision imports
from polars_vision import (
    Pipeline,
    CONTOUR_SCHEMA,
    POINT_SCHEMA,
    BBOX_SCHEMA,
)
from polars_vision.geometry.schemas import contour_from_points

# Display settings
plt.rcParams["figure.figsize"] = [12, 4]
plt.rcParams["figure.dpi"] = 100

print(f"✅ Polars version: {pl.__version__}")
print("✅ polars-vision loaded successfully")

# %%
# Helper functions for displaying images


def bytes_to_image(data: bytes) -> Image.Image:
    """Convert image bytes (PNG/JPEG) to PIL Image."""
    return Image.open(io.BytesIO(data))


def numpy_bytes_to_array(
    data: bytes, shape: tuple[int, ...], dtype: Any = np.uint8
) -> np.ndarray:
    """Convert numpy-format bytes back to ndarray.

    Note: The numpy sink includes a 26-byte header with shape/dtype info.
    This function skips the header and extracts just the array data.
    """
    # Skip the 26-byte header that polars-vision adds
    header_size = 26
    array_data = data[header_size:]
    return np.frombuffer(array_data, dtype=dtype).reshape(shape)


def display_images(
    images: list[Any], titles: list[str] | None = None, cmap: str | None = None
) -> None:
    """Display multiple images side by side."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, images)):
        if isinstance(img, bytes):
            img = bytes_to_image(img)
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()


def display_arrays(
    arrays: list[np.ndarray], titles: list[str] | None = None, cmap: str = "viridis"
) -> None:
    """Display multiple numpy arrays as heatmaps."""
    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (ax, arr) in enumerate(zip(axes, arrays)):
        im = ax.imshow(arr, cmap=cmap)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i])
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()


print("✅ Helper functions defined")

# %%
# Create sample test images for the demo


def create_test_image(
    width: int = 256, height: int = 256, pattern: str = "gradient"
) -> bytes:
    """Create a test image with various patterns."""
    if pattern == "gradient":
        # RGB gradient pattern
        r = np.linspace(0, 255, width, dtype=np.uint8)
        g = np.linspace(0, 255, height, dtype=np.uint8)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = r[np.newaxis, :]  # Red gradient horizontal
        img[:, :, 1] = g[:, np.newaxis]  # Green gradient vertical
        img[:, :, 2] = 128  # Blue constant
    elif pattern == "checkerboard":
        block_size = 32
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    img[i : i + block_size, j : j + block_size] = [255, 255, 255]
                else:
                    img[i : i + block_size, j : j + block_size] = [50, 50, 50]
    elif pattern == "circles":
        # Concentric circles
        y, x = np.ogrid[:height, :width]
        cx, cy = width // 2, height // 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = ((np.sin(r / 10) + 1) * 127.5).astype(np.uint8)
        img[:, :, 1] = ((np.cos(r / 15) + 1) * 127.5).astype(np.uint8)
        img[:, :, 2] = 100
    elif pattern == "heatmap":
        # Gaussian heatmap for ML demo
        y, x = np.ogrid[:height, :width]
        cx, cy = width // 2 + 30, height // 2 - 20
        sigma = 50
        gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        img = (gaussian * 255).astype(np.uint8)
        img = np.stack([img, img, img], axis=-1)  # Grayscale as RGB
    else:
        # Random noise
        rng = np.random.default_rng(42)
        img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

    # Convert to PNG bytes
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


# Create test images
test_images = {
    "gradient": create_test_image(256, 256, "gradient"),
    "checkerboard": create_test_image(256, 256, "checkerboard"),
    "circles": create_test_image(256, 256, "circles"),
    "heatmap": create_test_image(256, 256, "heatmap"),
    "noise": create_test_image(256, 256, "noise"),
}

# Display them
display_images(
    [
        test_images["gradient"],
        test_images["checkerboard"],
        test_images["circles"],
        test_images["heatmap"],
    ],
    ["Gradient", "Checkerboard", "Circles", "Heatmap"],
)
print(f"Created {len(test_images)} test images")

# %% [markdown]
# ## 2. Basic Pipeline Operations
#
# polars-vision uses a fluent **Pipeline** API to define image processing operations. A complete pipeline has three parts:
#
# 1. **Source**: How to interpret input data (`image_bytes`, `blob`, `raw`, `file_path`)
# 2. **Operations**: The transformations to apply (resize, grayscale, normalize, etc.)
# 3. **Sink**: The output format (`numpy`, `torch`, `png`, `jpeg`, `blob`)
#
# ### 2.1 Your First Pipeline

# %%
# Define a simple resize pipeline
resize_pipe = (
    Pipeline()
    .source("image_bytes")  # Input is PNG/JPEG bytes
    .resize(height=128, width=128)  # Resize to 128x128
    .sink("png")  # Output as PNG bytes
)

# Print the pipeline structure
print("Pipeline specification:")
print(resize_pipe)
print()

# Create a DataFrame with images
df = pl.DataFrame(
    {
        "name": ["gradient", "checkerboard", "circles"],
        "image": [
            test_images["gradient"],
            test_images["checkerboard"],
            test_images["circles"],
        ],
    }
)

# Apply the pipeline using .cv.pipeline()
result = df.with_columns(resized=pl.col("image").cv.pipeline(resize_pipe))

print(f"Original DataFrame schema: {df.schema}")
print(f"Result DataFrame schema: {result.schema}")

# Display original vs resized
row = result.row(0, named=True)
display_images([row["image"], row["resized"]], ["Original (256x256)", "Resized (128x128)"])

# %% [markdown]
# ### 2.2 Resize Filter Types
#
# polars-vision supports three resize filter types:
# - **nearest**: Fastest, best for pixel art or binary masks
# - **bilinear**: Good balance of speed and quality
# - **lanczos3**: Best quality, slower (default)

# %%
# Compare resize filters
filters = ["nearest", "bilinear", "lanczos3"]
resized_images = []

for filter_type in filters:
    pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=64, width=64, filter=filter_type)
        .sink("png")
    )
    result = pl.DataFrame({"img": [test_images["checkerboard"]]}).with_columns(
        out=pl.col("img").cv.pipeline(pipe)
    )
    resized_images.append(result["out"][0])

display_images(
    [test_images["checkerboard"]] + resized_images,
    ["Original"] + [f"{f} (64x64)" for f in filters],
)

# %% [markdown]
# ### 2.3 Common Image Operations
#
# Let's explore common image operations with intermediate outputs:

# %%
# Grayscale conversion
gray_pipe = Pipeline().source("image_bytes").grayscale().sink("png")

# Threshold (binary)
threshold_pipe = Pipeline().source("image_bytes").grayscale().threshold(128).sink("png")

# Blur
blur_pipe = Pipeline().source("image_bytes").blur(sigma=3.0).sink("png")

# Apply all to gradient image
test_df = pl.DataFrame({"img": [test_images["gradient"]]})
ops_result = test_df.with_columns(
    gray=pl.col("img").cv.pipeline(gray_pipe),
    threshold=pl.col("img").cv.pipeline(threshold_pipe),
    blur=pl.col("img").cv.pipeline(blur_pipe),
)

row = ops_result.row(0, named=True)
display_images(
    [row["img"], row["gray"], row["threshold"], row["blur"]],
    ["Original", "Grayscale", "Threshold (128)", "Blur (σ=3)"],
)

# %%
# Flip operations and cropping
flip_h_pipe = Pipeline().source("image_bytes").flip_h().sink("png")
flip_v_pipe = Pipeline().source("image_bytes").flip_v().sink("png")
crop_pipe = (
    Pipeline()
    .source("image_bytes")
    .crop(top=50, left=50, height=100, width=150)
    .sink("png")
)

test_df = pl.DataFrame({"img": [test_images["gradient"]]})
flip_result = test_df.with_columns(
    flip_h=pl.col("img").cv.pipeline(flip_h_pipe),
    flip_v=pl.col("img").cv.pipeline(flip_v_pipe),
    crop=pl.col("img").cv.pipeline(crop_pipe),
)

row = flip_result.row(0, named=True)
display_images(
    [row["img"], row["flip_h"], row["flip_v"], row["crop"]],
    ["Original", "Flip Horizontal", "Flip Vertical", "Cropped (100x150)"],
)

# %% [markdown]
# ### 2.4 Chained Operations
#
# Pipeline operations can be chained together. The operations are executed in a single pass through the Rust backend:

# %%
# Complex chained pipeline - common preprocessing for ML
ml_preprocess_pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=256, width=256)  # Resize to standard size
    .crop(top=16, left=16, height=224, width=224)  # Center crop
    .flip_h()  # Data augmentation
    .sink("png")
)

print("ML Preprocessing Pipeline:")
print(ml_preprocess_pipe)

# Apply to test image
result = pl.DataFrame({"img": [test_images["circles"]]}).with_columns(
    processed=pl.col("img").cv.pipeline(ml_preprocess_pipe)
)

row = result.row(0, named=True)
display_images(
    [row["img"], row["processed"]],
    ["Original (256x256)", "After ML Preprocessing (224x224)"],
)

# %% [markdown]
# ## 3. DType Promotion & Normalization
#
# polars-vision implements an automatic **DType Promotion System** that handles type conversions seamlessly. Operations like `normalize` accept any numeric input and automatically promote integers to floats.
#
# ### Key Concepts:
# - **MinMax normalization**: Scales values to [0, 1] range
# - **ZScore normalization**: Centers data around 0 with unit standard deviation
# - Outputs are float32 by default (configurable with `out_dtype`)

# %%
# Normalization pipelines
# Note: When using 'numpy' sink, we get raw bytes that can be converted back to arrays

# MinMax normalization - outputs float32 in [0, 1]
minmax_pipe = (
    Pipeline()
    .source("image_bytes")
    .grayscale()  # Convert to single channel for easier visualization
    .normalize(method="minmax")
    .sink("numpy")
)

# ZScore normalization - outputs float32 with mean=0, std=1
zscore_pipe = (
    Pipeline()
    .source("image_bytes")
    .grayscale()
    .normalize(method="zscore")
    .sink("numpy")
)

# Apply both
result = pl.DataFrame({"img": [test_images["gradient"]]}).with_columns(
    minmax=pl.col("img").cv.pipeline(minmax_pipe),
    zscore=pl.col("img").cv.pipeline(zscore_pipe),
)

# Convert back to arrays for visualization
# Note: After grayscale, shape is (256, 256, 1), after normalize dtype is f32
minmax_arr = numpy_bytes_to_array(result["minmax"][0], (256, 256, 1), dtype=np.float32)
zscore_arr = numpy_bytes_to_array(result["zscore"][0], (256, 256, 1), dtype=np.float32)

print(f"MinMax range: [{minmax_arr.min():.3f}, {minmax_arr.max():.3f}]")
print(f"ZScore mean: {zscore_arr.mean():.3f}, std: {zscore_arr.std():.3f}")

display_arrays(
    [minmax_arr.squeeze(), zscore_arr.squeeze()],
    ["MinMax Normalized [0,1]", "ZScore Normalized (μ=0, σ=1)"],
)

# %%
# Scale and clamp operations
# These also support automatic dtype promotion

# Scale by factor
scale_pipe = (
    Pipeline()
    .source("image_bytes")
    .grayscale()
    .scale(factor=0.5)  # Halve all values
    .sink("numpy")
)

# Clamp to range
clamp_pipe = (
    Pipeline()
    .source("image_bytes")
    .grayscale()
    .normalize(method="minmax")  # [0, 1]
    .clamp(min_val=0.2, max_val=0.8)  # Clip to [0.2, 0.8]
    .sink("numpy")
)

result = pl.DataFrame({"img": [test_images["gradient"]]}).with_columns(
    scaled=pl.col("img").cv.pipeline(scale_pipe),
    clamped=pl.col("img").cv.pipeline(clamp_pipe),
)

# scale output is f32 (promoted from u8), clamp is also f32
scaled_arr = numpy_bytes_to_array(result["scaled"][0], (256, 256, 1), dtype=np.float32)
clamped_arr = numpy_bytes_to_array(result["clamped"][0], (256, 256, 1), dtype=np.float32)

print(f"Scaled range: [{scaled_arr.min():.1f}, {scaled_arr.max():.1f}]")
print(f"Clamped range: [{clamped_arr.min():.2f}, {clamped_arr.max():.2f}]")

display_arrays(
    [scaled_arr.squeeze(), clamped_arr.squeeze()], ["Scaled (×0.5)", "Clamped [0.2, 0.8]"]
)

# %% [markdown]
# ## 4. Dynamic Parameters with Expressions
#
# One of polars-vision's most powerful features is **dynamic parameters**. Any pipeline parameter can be a Polars expression (`pl.col(...)`) that gets resolved per-row at execution time.
#
# This enables:
# - Per-image resize dimensions based on metadata
# - Adaptive thresholding based on image statistics
# - Dynamic cropping based on detected regions

# %%
# Dynamic resize - each row gets different dimensions!
dynamic_resize_pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(
        height=pl.col("target_h"), width=pl.col("target_w")
    )  # Expression parameters!
    .sink("png")
)

# Create DataFrame with per-row dimensions
df = pl.DataFrame(
    {
        "name": ["small", "medium", "large"],
        "image": [test_images["circles"]] * 3,
        "target_h": [64, 128, 200],
        "target_w": [64, 128, 200],
    }
)

result = df.with_columns(resized=pl.col("image").cv.pipeline(dynamic_resize_pipe))

print("Each image resized to different dimensions:")
print(
    result.select(
        "name",
        "target_h",
        "target_w",
        pl.col("resized").bin.size().alias("output_bytes"),
    )
)

# Display all resized images
display_images(
    [result["resized"][i] for i in range(3)],
    [
        f"{row['name']} ({row['target_h']}x{row['target_w']})"
        for row in result.iter_rows(named=True)
    ],
)

# %%
# Dynamic crop based on bounding box columns
dynamic_crop_pipe = (
    Pipeline()
    .source("image_bytes")
    .crop(
        top=pl.col("bbox_y"),
        left=pl.col("bbox_x"),
        height=pl.col("bbox_h"),
        width=pl.col("bbox_w"),
    )
    .sink("png")
)

# Simulate detected bounding boxes
df = pl.DataFrame(
    {
        "image": [test_images["gradient"]] * 3,
        "region": ["top-left", "center", "bottom-right"],
        "bbox_x": [10, 80, 150],
        "bbox_y": [10, 80, 150],
        "bbox_w": [80, 100, 90],
        "bbox_h": [80, 100, 90],
    }
)

result = df.with_columns(cropped=pl.col("image").cv.pipeline(dynamic_crop_pipe))

display_images(
    [result["cropped"][i] for i in range(3)],
    [f"Crop: {row['region']}" for row in result.iter_rows(named=True)],
)

# %% [markdown]
# ## 5. Geometry Operations
#
# polars-vision provides a comprehensive geometry module for working with **contours**, **points**, and **bounding boxes**. This is essential for computer vision tasks like:
# - Object detection and segmentation
# - Annotation processing
# - IoU/Dice metrics calculation
#
# ### 5.1 Contour Schema
#
# Contours are stored as Polars Struct columns with the following schema:

# %%
# Show the contour schema
print("CONTOUR_SCHEMA:")
print(CONTOUR_SCHEMA)
print()
print("POINT_SCHEMA:")
print(POINT_SCHEMA)
print()
print("BBOX_SCHEMA:")
print(BBOX_SCHEMA)

# %%
# Create contours using the helper function
contours = [
    # Square contour
    contour_from_points([(50, 50), (50, 150), (150, 150), (150, 50)]),
    # Triangle contour
    contour_from_points([(100, 30), (30, 170), (170, 170)]),
    # Irregular polygon (L-shape)
    contour_from_points(
        [(20, 20), (20, 180), (100, 180), (100, 100), (180, 100), (180, 20)]
    ),
]

# Create DataFrame with contours
contour_df = pl.DataFrame(
    {
        "name": ["square", "triangle", "L-shape"],
        "contour": contours,
    }
).cast({"contour": CONTOUR_SCHEMA})

print("Contour DataFrame:")
print(contour_df)

# %% [markdown]
# ### 5.2 Geometric Measures
#
# The `.contour` namespace provides operations for computing geometric properties.
#
# > **Note**: The contour operations require Rust implementation. Below we show the API
# > and compute measures manually using the Shoelace formula as a reference.

# %%
# Manual computation of geometric measures (reference implementation)
# The .contour namespace would provide these directly once implemented


def compute_contour_area(contour_dict: dict) -> float:
    """Compute contour area using the Shoelace formula."""
    points = contour_dict["exterior"]
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i]["x"] * points[j]["y"]
        area -= points[j]["x"] * points[i]["y"]
    return abs(area / 2)


def compute_contour_perimeter(contour_dict: dict) -> float:
    """Compute contour perimeter."""
    points = contour_dict["exterior"]
    n = len(points)
    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = points[j]["x"] - points[i]["x"]
        dy = points[j]["y"] - points[i]["y"]
        perimeter += np.sqrt(dx * dx + dy * dy)
    return perimeter


def compute_contour_centroid(contour_dict: dict) -> tuple[float, float]:
    """Compute contour centroid."""
    points = contour_dict["exterior"]
    cx = sum(p["x"] for p in points) / len(points)
    cy = sum(p["y"] for p in points) / len(points)
    return cx, cy


# Compute measures for each contour
measures = []
for row in contour_df.iter_rows(named=True):
    contour = row["contour"]
    area = compute_contour_area(contour)
    perimeter = compute_contour_perimeter(contour)
    cx, cy = compute_contour_centroid(contour)
    measures.append(
        {
            "name": row["name"],
            "area": area,
            "perimeter": perimeter,
            "centroid_x": cx,
            "centroid_y": cy,
        }
    )

measures_df = pl.DataFrame(measures)
print("Geometric Measures (computed manually):")
print(measures_df)

# %%
# The polars-vision API would look like this once implemented:
# measures = contour_df.with_columns(
#     area=pl.col("contour").contour.area(),
#     perimeter=pl.col("contour").contour.perimeter(),
#     centroid=pl.col("contour").contour.centroid(),
#     bbox=pl.col("contour").contour.bounding_box(),
# )
print("\nExpected API (once Rust backend is implemented):")
print('  pl.col("contour").contour.area()')
print('  pl.col("contour").contour.perimeter()')
print('  pl.col("contour").contour.centroid()')

# %% [markdown]
# ### 5.3 Rasterizing Contours to Masks
#
# The `rasterize` operation converts contours to binary masks.
#
# > **Note**: This requires Rust backend implementation. Below we show manual rasterization
# > using PIL as a reference.

# %%
# Manual rasterization using PIL (reference implementation)
from PIL import ImageDraw  # noqa: E402


def rasterize_contour(contour_dict: dict, width: int, height: int) -> bytes:
    """Rasterize a contour to a binary mask PNG."""
    points = contour_dict["exterior"]
    polygon = [(p["x"], p["y"]) for p in points]

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(polygon, fill=255)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# Rasterize all contours
masks = []
for row in contour_df.iter_rows(named=True):
    mask = rasterize_contour(row["contour"], 200, 200)
    masks.append(mask)

result = contour_df.with_columns(mask=pl.Series("mask", masks))

display_images(
    [result["mask"][i] for i in range(3)],
    [f"{row['name']} mask" for row in result.iter_rows(named=True)],
    cmap="gray",
)

# %%
# The polars-vision API would look like this:
# rasterize_pipe = (
#     Pipeline()
#     .source("contour")
#     .rasterize(width=200, height=200, fill_value=255, background=0)
#     .sink("png")
# )
# result = contour_df.with_columns(mask=pl.col("contour").cv.pipeline(rasterize_pipe))
print("Contour rasterization pipeline API:")
print('  Pipeline().source("contour").rasterize(width=200, height=200).sink("png")')

# %% [markdown]
# ## 6. Lazy Pipeline Composition
#
# polars-vision supports **lazy pipeline composition** using `LazyPipelineExpr`. This enables:
#
# 1. **Fused execution**: Multiple pipelines combined into a single plugin call
# 2. **Binary operations**: Add, subtract, multiply, divide arrays
# 3. **Mask application**: Apply masks from contours or other images
#
# ### Two Modes:
# - **Eager mode**: `pl.col("x").cv.pipeline(pipe)` - Returns `pl.Expr` directly (requires sink)
# - **Lazy mode**: `pl.col("x").cv.pipe(pipe)` - Returns `LazyPipelineExpr` for composition

# %%
# Lazy mode example - compose pipelines before execution

# Define pipelines WITHOUT sinks (lazy mode)
img_pipe = Pipeline().source("image_bytes").resize(height=200, width=200)
mask_pipe = Pipeline().source("contour").rasterize(width=200, height=200)

# Create lazy expressions using .cv.pipe()
img_expr = pl.col("image").cv.pipe(img_pipe)  # Returns LazyPipelineExpr
mask_expr = pl.col("contour").cv.pipe(mask_pipe)  # Returns LazyPipelineExpr

print(f"img_expr type: {type(img_expr)}")
print(f"mask_expr type: {type(mask_expr)}")
print()
print("These are NOT Polars expressions yet - they need .sink() to materialize!")

# %%
# Compose operations and finalize with .sink()

# The full apply_mask composition requires contour rasterization in Rust.
# Here we demonstrate the image pipeline composition that works:

# Simple lazy composition (single input)
img_pipe = Pipeline().source("image_bytes").resize(height=200, width=200)
img_expr = pl.col("image").cv.pipe(img_pipe)

# Add more operations through composition
# (Note: apply_mask with contours requires Rust backend for contour source)
final_expr = img_expr.sink("png")

# Now it's a real Polars expression
print(f"final_expr type: {type(final_expr)}")

# Create test data and execute
compose_df = pl.DataFrame({"image": [test_images["circles"]]})
result = compose_df.with_columns(resized=final_expr)

# For the full mask application demo, we use manual rasterization
contour = contour_from_points([(50, 50), (50, 150), (150, 150), (150, 50)])
mask_bytes = rasterize_contour(contour, 200, 200)

# Apply mask manually in numpy
resized_img = Image.open(io.BytesIO(result["resized"][0])).convert("RGB")
mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
masked_arr = np.array(resized_img) * (np.array(mask_img)[:, :, np.newaxis] / 255)
masked_pil = Image.fromarray(masked_arr.astype(np.uint8))
buffer = io.BytesIO()
masked_pil.save(buffer, format="PNG")
masked_bytes = buffer.getvalue()

display_images(
    [result["resized"][0], mask_bytes, masked_bytes],
    ["Resized Image", "Rasterized Mask", "Image × Mask (manual)"],
)

# %%
# The full API once Rust backend supports contour sources:
print("Full mask application API (once contour source is implemented):")
print("""
img_pipe = Pipeline().source("image_bytes").resize(height=200, width=200)
mask_pipe = Pipeline().source("contour").rasterize(width=200, height=200)

img_expr = pl.col("image").cv.pipe(img_pipe)
mask_expr = pl.col("contour").cv.pipe(mask_pipe)

masked_expr = img_expr.apply_mask(mask_expr).sink("png")
result = df.with_columns(masked=masked_expr)
""")

# %% [markdown]
# ## 7. Multi-Output Pipelines
#
# polars-vision supports **multi-output pipelines** using aliases. This allows you to:
#
# 1. Mark intermediate points in a pipeline with `.alias(name)`
# 2. Return multiple outputs as a Struct column with `.sink({alias: format, ...})`
#
# This is more efficient than running separate pipelines because:
# - Shared subexpressions are computed once
# - Single plugin call for all outputs

# %%
# Multi-output pipeline with aliases
# This API is defined but requires Rust backend implementation

# Show the Python API structure
multi_pipe = (
    Pipeline()
    .source("image_bytes")
    .alias("original")  # Checkpoint: decoded image
    .resize(height=128, width=128)
    .alias("resized")  # Checkpoint: after resize
    .grayscale()
    .alias("gray")  # Checkpoint: after grayscale
    .threshold(128)
    .alias("binary")  # Checkpoint: final binary
)

print("Pipeline with aliases:")
print(f"Aliases defined: {multi_pipe.get_aliases()}")

# %%
# Multi-output isn't implemented in Rust backend yet.
# Workaround: Run separate pipelines for each checkpoint

df = pl.DataFrame({"image": [test_images["circles"]]})

# Define separate pipelines for each checkpoint
original_pipe = Pipeline().source("image_bytes").sink("png")
resized_pipe = Pipeline().source("image_bytes").resize(height=128, width=128).sink("png")
gray_pipe = (
    Pipeline().source("image_bytes").resize(height=128, width=128).grayscale().sink("png")
)
binary_pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=128, width=128)
    .grayscale()
    .threshold(128)
    .sink("png")
)

# Run all pipelines
result = df.with_columns(
    original=pl.col("image").cv.pipeline(original_pipe),
    resized=pl.col("image").cv.pipeline(resized_pipe),
    gray=pl.col("image").cv.pipeline(gray_pipe),
    binary=pl.col("image").cv.pipeline(binary_pipe),
)

# Display all intermediate outputs
row = result.row(0, named=True)
display_images(
    [row["original"], row["resized"], row["gray"], row["binary"]],
    ["Original (decoded)", "Resized (128x128)", "Grayscale", "Binary (thresh=128)"],
)

# %%
# Once implemented, the multi-output API would be:
print("Multi-output API (once Rust backend is implemented):")
print("""
multi_pipe = (
    Pipeline()
    .source("image_bytes")
    .alias("original")
    .resize(height=128, width=128)
    .alias("resized")
    .grayscale()
    .alias("gray")
).sink({
    "original": "png",
    "resized": "png",
    "gray": "png"
})

result = df.with_columns(outputs=pl.col("image").cv.pipeline(multi_pipe))
# Extract with: pl.col("outputs").struct.field("original")
""")

# %% [markdown]
# ## 8. ML Workflow: IoU Calculation
#
# Now let's build a complete **ML-style workflow** that demonstrates:
#
# 1. Generating fake heatmap predictions (simulating model output)
# 2. Processing ground truth contour annotations
# 3. Rasterizing both to masks
# 4. Computing **IoU** (Intersection over Union) and **Dice** coefficients
# 5. Visualizing predictions vs ground truth

# %%
# Generate synthetic ML data


def create_heatmap_prediction(cx: int, cy: int, sigma: float, size: int = 200) -> bytes:
    """Create a fake heatmap prediction (simulating model output)."""
    y, x = np.ogrid[:size, :size]
    gaussian = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
    # Convert to 8-bit grayscale PNG
    img = (gaussian * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode="L")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def create_ground_truth_contour(
    cx: int, cy: int, radius: int, n_points: int = 32
) -> dict[str, Any]:
    """Create a circular ground truth contour."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]
    return contour_from_points(points)


# Create dataset with predictions and ground truth
np.random.seed(42)
n_samples = 5

data: dict[str, list[Any]] = {
    "sample_id": list(range(n_samples)),
    "prediction": [],
    "ground_truth": [],
}

for i in range(n_samples):
    # Ground truth center and radius
    gt_cx, gt_cy = 100 + np.random.randint(-20, 20), 100 + np.random.randint(-20, 20)
    gt_radius = 40 + np.random.randint(-10, 10)

    # Prediction center (with some error)
    pred_cx = gt_cx + np.random.randint(-15, 15)
    pred_cy = gt_cy + np.random.randint(-15, 15)
    pred_sigma = gt_radius * 0.6  # Spread of heatmap

    data["prediction"].append(create_heatmap_prediction(pred_cx, pred_cy, pred_sigma))
    data["ground_truth"].append(create_ground_truth_contour(gt_cx, gt_cy, gt_radius))

ml_df = pl.DataFrame(data).cast({"ground_truth": CONTOUR_SCHEMA})
print(f"ML DataFrame schema: {ml_df.schema}")
print(ml_df.head())

# %%
# Process predictions and ground truth

# Prediction pipeline: threshold heatmap to binary mask
pred_pipe = (
    Pipeline()
    .source("image_bytes")
    .threshold(128)  # Threshold at 50% intensity
    .sink("png")
)

# Process predictions with polars-vision
processed = ml_df.with_columns(
    pred_mask=pl.col("prediction").cv.pipeline(pred_pipe),
)

# Ground truth: rasterize contours manually (contour source not implemented in Rust)
gt_masks = []
for row in ml_df.iter_rows(named=True):
    gt_mask = rasterize_contour(row["ground_truth"], 200, 200)
    gt_masks.append(gt_mask)

processed = processed.with_columns(gt_mask=pl.Series("gt_mask", gt_masks))

# Visualize first sample
row = processed.row(0, named=True)
display_images(
    [row["prediction"], row["pred_mask"], row["gt_mask"]],
    ["Raw Heatmap", "Thresholded Prediction", "Ground Truth Mask"],
    cmap="gray",
)

# %%
# Calculate IoU and Dice using contour operations
# First, we need to extract contours from the prediction masks

# For contour-based IoU, we can use the .contour.iou() operation directly
# This requires having both as contour format

# Alternative approach: compute IoU manually from rasterized masks
# (This would be done in Python since we have the masks as images)


def compute_iou_from_masks(mask1_bytes: bytes, mask2_bytes: bytes) -> float:
    """Compute IoU from two PNG mask bytes."""
    m1 = np.array(Image.open(io.BytesIO(mask1_bytes)).convert("L")) > 128
    m2 = np.array(Image.open(io.BytesIO(mask2_bytes)).convert("L")) > 128
    intersection = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return float(intersection / union) if union > 0 else 0.0


def compute_dice_from_masks(mask1_bytes: bytes, mask2_bytes: bytes) -> float:
    """Compute Dice coefficient from two PNG mask bytes."""
    m1 = np.array(Image.open(io.BytesIO(mask1_bytes)).convert("L")) > 128
    m2 = np.array(Image.open(io.BytesIO(mask2_bytes)).convert("L")) > 128
    intersection = np.sum(m1 & m2)
    denom = np.sum(m1) + np.sum(m2)
    return float(2 * intersection / denom) if denom > 0 else 0.0


# Compute metrics for all samples
metrics = []
for row in processed.iter_rows(named=True):
    iou = compute_iou_from_masks(row["pred_mask"], row["gt_mask"])
    dice = compute_dice_from_masks(row["pred_mask"], row["gt_mask"])
    metrics.append({"sample_id": row["sample_id"], "iou": iou, "dice": dice})

metrics_df = pl.DataFrame(metrics)
print("Segmentation Metrics:")
print(metrics_df)
print(f"\nMean IoU: {metrics_df['iou'].mean():.3f}")
print(f"Mean Dice: {metrics_df['dice'].mean():.3f}")

# %%
# Visualize overlay of predictions vs ground truth


def create_overlay(pred_bytes: bytes, gt_bytes: bytes) -> np.ndarray:
    """Create RGB overlay: green=GT, red=pred, yellow=overlap."""
    pred = np.array(Image.open(io.BytesIO(pred_bytes)).convert("L")) > 128
    gt = np.array(Image.open(io.BytesIO(gt_bytes)).convert("L")) > 128

    h, w = pred.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Red channel: prediction
    overlay[:, :, 0] = pred.astype(np.uint8) * 255
    # Green channel: ground truth
    overlay[:, :, 1] = gt.astype(np.uint8) * 255
    # Yellow where both overlap (R+G)

    return overlay


# Create overlays for first 3 samples
overlays = []
titles = []
for i, row in enumerate(processed.head(3).iter_rows(named=True)):
    overlay = create_overlay(row["pred_mask"], row["gt_mask"])
    overlays.append(overlay)
    titles.append(f"Sample {i} (IoU={metrics[i]['iou']:.2f})")

print("Overlay: Green=GT, Red=Pred, Yellow=Overlap")
display_images(overlays, titles)

# %% [markdown]
# ## 9. Lazy Scalability Demo
#
# polars-vision integrates seamlessly with Polars' **lazy execution engine**. This enables:
#
# 1. **Memory efficiency**: Process data that doesn't fit in memory
# 2. **Query optimization**: Operations are fused and optimized
# 3. **Streaming**: Process data in chunks
#
# Let's demonstrate by:
# 1. Generating a configurable synthetic dataset on disk
# 2. Lazily scanning and processing the data
# 3. Sinking results to parquet

# %%
# Create a temporary directory for our demo dataset
import shutil  # noqa: E402

DEMO_DIR = Path(tempfile.mkdtemp(prefix="polars_vision_demo_"))
print(f"Demo directory: {DEMO_DIR}")

# Configuration
N_IMAGES = 50  # Number of images to generate
IMAGE_SIZE = 128  # Size of each image

print(f"Generating {N_IMAGES} synthetic images...")


# %%
# Generate images and save to parquet
def generate_dataset(n_images: int, size: int, output_path: Path) -> pl.DataFrame:
    """Generate a synthetic image dataset and save to parquet."""
    records = []
    patterns = ["gradient", "checkerboard", "circles", "noise"]

    for i in range(n_images):
        pattern = patterns[i % len(patterns)]
        image_bytes = create_test_image(size, size, pattern)

        # Add some metadata
        records.append(
            {
                "id": i,
                "pattern": pattern,
                "image": image_bytes,
                "target_size": 64 + (i % 3) * 32,  # 64, 96, or 128
                "threshold": 100 + (i % 5) * 20,  # 100, 120, 140, 160, 180
            }
        )

    df = pl.DataFrame(records)
    df.write_parquet(output_path)
    return df


# Generate and save
input_path = DEMO_DIR / "raw_images.parquet"
source_df = generate_dataset(N_IMAGES, IMAGE_SIZE, input_path)

print(f"Saved {N_IMAGES} images to {input_path}")
print(f"File size: {input_path.stat().st_size / 1024:.1f} KB")
print(f"\nSchema: {source_df.schema}")
print(source_df.head(3))

# %%
# Lazy processing pipeline
# This demonstrates the power of lazy evaluation

# Define reusable pipelines
preprocess_pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=64, width=64)  # Standard size
    .grayscale()
    .normalize(method="minmax")
    .sink("numpy")
)

# Create lazy query
lazy_query = (
    pl.scan_parquet(input_path)
    .filter(pl.col("id") < 20)  # Only process subset for demo
    .with_columns(processed=pl.col("image").cv.pipeline(preprocess_pipe))
    .select("id", "pattern", "processed")
)

# Show the lazy query plan
print("Lazy Query Plan:")
print(lazy_query.explain())

# %%
# Execute and sink to parquet
output_path = DEMO_DIR / "processed_images.parquet"

# Collect (execute) the lazy query
result = lazy_query.collect()

# Save results
result.write_parquet(output_path)

print(f"Processed {len(result)} images")
print(f"Output saved to {output_path}")
print(f"Output file size: {output_path.stat().st_size / 1024:.1f} KB")
print()
print(result.head())

# %%
# Verify the processed data
processed_data = pl.read_parquet(output_path)

# Convert one sample back to array for visualization
sample = processed_data.row(0, named=True)
arr = numpy_bytes_to_array(sample["processed"], (64, 64, 1), dtype=np.float32)

print(f"Processed array shape: {arr.shape}")
print(f"Value range: [{arr.min():.3f}, {arr.max():.3f}]")

# Display a few samples
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes):
    sample = processed_data.row(i, named=True)
    arr = numpy_bytes_to_array(sample["processed"], (64, 64, 1), dtype=np.float32)
    ax.imshow(arr.squeeze(), cmap="viridis")
    ax.set_title(f"ID={sample['id']}, {sample['pattern']}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# Cleanup demo directory
shutil.rmtree(DEMO_DIR)
print(f"Cleaned up {DEMO_DIR}")

# %% [markdown]
# ## 10. PyTorch Integration
#
# polars-vision can output directly to **torch format** for seamless ML integration. The `torch` sink produces bytes that can be converted to PyTorch tensors.
#
# ### Workflow:
# 1. Process images with polars-vision pipeline
# 2. Sink to `torch` format
# 3. Convert to PyTorch tensors
# 4. Feed to DataLoader for training

# %%
# Check if PyTorch is available
try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
    print(f"✅ PyTorch version: {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed - skipping torch integration demo")

# %%
if TORCH_AVAILABLE:
    # Pipeline that outputs torch-compatible format
    # ImageNet-style preprocessing
    torch_pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=224, width=224)  # ImageNet size
        .normalize(method="minmax")  # Scale to [0, 1]
        .sink("torch")  # Output as torch-compatible bytes
    )

    # Process batch of images
    batch_df = pl.DataFrame(
        {
            "image": [
                test_images["gradient"],
                test_images["circles"],
                test_images["checkerboard"],
            ],
            "label": [0, 1, 2],
        }
    )

    processed = batch_df.with_columns(
        tensor_bytes=pl.col("image").cv.pipeline(torch_pipe)
    )

    print(f"Processed {len(processed)} images")
    print(f"Tensor bytes column dtype: {processed['tensor_bytes'].dtype}")

# %%
if TORCH_AVAILABLE:
    # Convert bytes to PyTorch tensors
    def bytes_to_torch(
        data: bytes, shape: tuple[int, ...], dtype: Any = torch.float32
    ) -> torch.Tensor:
        """Convert torch-format bytes to PyTorch tensor."""
        arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
        return torch.from_numpy(arr.copy())

    # Create tensor batch
    tensors = []
    labels = []

    for row in processed.iter_rows(named=True):
        # Shape after processing: (224, 224, 3) for RGB, float32
        tensor = bytes_to_torch(row["tensor_bytes"], (224, 224, 3))
        # Transpose to PyTorch format: (C, H, W)
        tensor = tensor.permute(2, 0, 1)
        tensors.append(tensor)
        labels.append(row["label"])

    # Stack into batch
    batch_tensor = torch.stack(tensors)
    batch_labels = torch.tensor(labels)

    print(f"Batch tensor shape: {batch_tensor.shape}")
    print(f"Batch tensor dtype: {batch_tensor.dtype}")
    print(f"Batch labels: {batch_labels}")
    print(f"Value range: [{batch_tensor.min():.3f}, {batch_tensor.max():.3f}]")

# %%
if TORCH_AVAILABLE:
    # Create a simple Dataset class for DataLoader integration

    class PolarsVisionDataset(Dataset):
        """PyTorch Dataset backed by a Polars DataFrame with polars-vision preprocessing."""

        def __init__(
            self, df: pl.DataFrame, image_col: str, label_col: str, pipeline: Pipeline
        ) -> None:
            # Pre-process all images
            self.df = df.with_columns(_tensor=pl.col(image_col).cv.pipeline(pipeline))
            self.label_col = label_col

        def __len__(self) -> int:
            return len(self.df)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
            row = self.df.row(idx, named=True)
            tensor = bytes_to_torch(row["_tensor"], (224, 224, 3))
            tensor = tensor.permute(2, 0, 1)  # (C, H, W)
            label = row[self.label_col]
            return tensor, label

    # Create dataset and dataloader
    dataset = PolarsVisionDataset(batch_df, "image", "label", torch_pipe)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate through batches
    print("DataLoader iteration:")
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"  Batch {batch_idx}: images shape={images.shape}, labels={labels.tolist()}")

# %% [markdown]
# ## 11. Conclusion
#
# This notebook demonstrated the key capabilities of **polars-vision**:
#
# ### ✅ What We Covered
#
# 1. **Basic Pipeline Operations**
#    - Source/sink architecture with multiple format support
#    - Resize, grayscale, threshold, blur, crop, flip operations
#    - Chained pipelines with single-pass execution
#
# 2. **DType Promotion & Normalization**
#    - Automatic type promotion (u8 → f32)
#    - MinMax and ZScore normalization
#    - Scale and clamp operations
#
# 3. **Dynamic Parameters**
#    - Using Polars expressions for per-row parameters
#    - Dynamic resize, crop, and threshold based on metadata
#
# 4. **Geometry Operations**
#    - Contour schemas and creation
#    - Geometric measures (area, perimeter, centroid, bbox)
#    - Rasterization of contours to masks
#
# 5. **Lazy Pipeline Composition**
#    - `cv.pipe()` vs `cv.pipeline()` modes
#    - Binary operations and mask application
#    - Fused execution of composed pipelines
#
# 6. **Multi-Output Pipelines**
#    - Alias-based checkpoints
#    - Struct output with multiple formats
#
# 7. **ML Workflow: IoU Calculation**
#    - Heatmap prediction processing
#    - Ground truth contour handling
#    - IoU/Dice metric computation
#
# 8. **Lazy Scalability**
#    - Integration with Polars lazy execution
#    - Parquet read/write workflows
#
# 9. **PyTorch Integration**
#    - Torch format output
#    - DataLoader-compatible datasets
#
# ### 🔗 Resources
#
# - **Repository**: [polars-vision](https://github.com/your-org/polars-vision)
# - **view-buffer**: The underlying Rust tensor orchestration library
# - **Polars Documentation**: [pola.rs](https://pola.rs)

# %%
print("🎉 Demo complete! polars-vision provides:")
print("   • High-performance image processing in Polars")
print("   • Zero-copy operations where possible")
print("   • Composable, reusable pipelines")
print("   • Seamless ML framework integration")

