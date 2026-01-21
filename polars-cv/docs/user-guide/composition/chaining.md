# Pipeline Chaining

Learn how to chain and compose pipelines for reusable, modular image processing.

## Basic Chaining

Chain operations within a single pipeline:

```python
from polars_cv import Pipeline

pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .grayscale()
    .normalize(method="minmax")
    .sink("numpy")
)
```

## Modular Chaining

The recommended way to compose complex pipelines is to chain modular `Pipeline` objects onto an expression using `.pipe()`.

```python
import polars as pl
from polars_cv import Pipeline

# 1. Define operation fragments
preprocess = Pipeline().resize(height=128, width=128)
gray_scale = Pipeline().grayscale()
thresholding = Pipeline().threshold(128)

# 2. Chain them onto a column
base = pl.col("image").cv.pipe(Pipeline().source("image_bytes"))
processed = (
    base
    .pipe(preprocess)
    .pipe(gray_scale)
    .pipe(thresholding)
)

# 3. Execute with a sink
result = df.with_columns(output=processed.sink("png"))
```

## Reusable Fragments

Fragments can be reused across different pipelines:

```python
# Define once
standard_size = Pipeline().resize(height=224, width=224)
normalization = Pipeline().normalize(method="minmax")

# Use in different expressions
train_expr = (
    pl.col("image").cv.pipe(Pipeline().source("image_bytes"))
    .pipe(standard_size)
    .pipe(normalization)
)
```

## Pipeline Factories

Create functions that generate configured pipelines:

```python
def create_resize_pipeline(size: int) -> Pipeline:
    """Create a resize pipeline with specified size."""
    return Pipeline().source("image_bytes").resize(height=size, width=size)


def create_augmentation(flip: bool = True, blur_sigma: float = 0.0) -> Pipeline:
    """Create an augmentation pipeline with configurable options."""
    ops = Pipeline()
    if flip:
        ops = ops.flip_h()
    if blur_sigma > 0:
        ops = ops.blur(sigma=blur_sigma)
    return ops


# Use factories
base = pl.col("image").cv.pipe(create_resize_pipeline(224))
augmented = base.pipe(create_augmentation(flip=True, blur_sigma=1.5))
```

## Configuration-Driven Pipelines

Build pipelines from configuration dictionaries:

```python
from typing import Any


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Build a pipeline from configuration."""
    pipe = Pipeline().source("image_bytes")
    
    if "target_size" in config:
        size = config["target_size"]
        pipe = pipe.resize(height=size, width=size)
    
    if config.get("grayscale", False):
        pipe = pipe.grayscale()
    
    if config.get("normalize", False):
        method = config.get("normalize_method", "minmax")
        pipe = pipe.normalize(method=method)
    
    return pipe


# Training configuration
train_config = {
    "target_size": 224,
    "flip_horizontal": True,
    "normalize": True,
    "normalize_method": "minmax",
}

# Inference configuration
inference_config = {
    "target_size": 224,
    "grayscale": True,
    "normalize": True,
}

# Build and apply
train_pipe = build_pipeline(train_config)
inference_pipe = build_pipeline(inference_config)
```

## Branching Pipelines

Create branches from a common base:

```python
# Common preprocessing
base = (
    pl.col("image")
    .cv.pipe(Pipeline().source("image_bytes").resize(128, 128))
    .alias("resized")
)

# Branch 1: RGB output
rgb_branch = base.sink("png")

# Branch 2: Grayscale
gray_branch = base.pipe(Pipeline().grayscale()).alias("gray").sink("png")

# Branch 3: Binary mask
binary_branch = (
    base
    .pipe(Pipeline().grayscale())
    .pipe(Pipeline().threshold(128))
    .alias("binary")
    .sink("png")
)

# Execute all branches
result = df.with_columns(
    rgb=rgb_branch,
    gray=gray_branch,
    binary=binary_branch,
)
```

## Best Practices

1. **Define Fragments**: Create reusable operation groups
2. **Use Factories**: Parameterize common pipeline patterns
3. **Configuration**: Support config-driven pipeline creation
4. **Naming**: Use `.alias()` for multi-output branches
5. **Testing**: Test fragments independently

## Next Steps

- [Multi-Output](multi-output.md) - Extract multiple outputs
- [Binary Operations](binary-ops.md) - Combine pipelines

