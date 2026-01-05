# ML Integration

polars-vision integrates seamlessly with machine learning frameworks.

## NumPy Integration

### Converting Output to NumPy

Use `numpy_from_bytes()` to convert pipeline output:

```python
from polars_vision import Pipeline, numpy_from_bytes
import numpy as np

# Pipeline with numpy sink
pipe = Pipeline().source("image_bytes").resize(224, 224).normalize().sink("numpy")

result = df.with_columns(tensor=pl.col("image").cv.pipeline(pipe))

# Convert to NumPy array
arr = numpy_from_bytes(result["tensor"][0])
print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
# Shape: (224, 224, 3), dtype: float32
```

### Batch Processing

```python
# Process all images
arrays = [numpy_from_bytes(b) for b in result["tensor"]]
batch = np.stack(arrays)
print(f"Batch shape: {batch.shape}")
# Batch shape: (N, 224, 224, 3)
```

## PyTorch Integration

### Direct Tensor Conversion

```python
import torch
from polars_vision import Pipeline, numpy_from_bytes

# Pipeline with torch-compatible output
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=224, width=224)
    .normalize(method="minmax")
    .sink("torch")
)

result = df.with_columns(tensor=pl.col("image").cv.pipeline(pipe))

# Convert to PyTorch tensor
def bytes_to_tensor(data: bytes) -> torch.Tensor:
    arr = numpy_from_bytes(data)
    tensor = torch.from_numpy(arr.copy())
    return tensor.permute(2, 0, 1)  # HWC → CHW

tensors = [bytes_to_tensor(b) for b in result["tensor"]]
batch = torch.stack(tensors)
print(f"Batch shape: {batch.shape}")
# Batch shape: torch.Size([N, 3, 224, 224])
```

### PyTorch Dataset

```python
from torch.utils.data import Dataset, DataLoader
import polars as pl
from polars_vision import Pipeline, numpy_from_bytes
import torch


class PolarsVisionDataset(Dataset):
    """PyTorch Dataset backed by polars-vision preprocessing."""
    
    def __init__(
        self,
        df: pl.DataFrame,
        image_col: str,
        label_col: str,
        pipeline: Pipeline,
    ) -> None:
        # Pre-process all images
        self.df = df.with_columns(
            _tensor=pl.col(image_col).cv.pipeline(pipeline)
        )
        self.label_col = label_col
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.row(idx, named=True)
        arr = numpy_from_bytes(row["_tensor"])
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        label = row[self.label_col]
        return tensor, label


# Create dataset
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(224, 224)
    .normalize()
    .sink("torch")
)

dataset = PolarsVisionDataset(df, "image", "label", pipe)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    # images: (32, 3, 224, 224)
    # labels: (32,)
    pass
```

## Segmentation Metrics

Use native `mask_iou()` and `mask_dice()` for evaluation:

```python
from polars_vision import Pipeline, mask_iou, mask_dice
import polars as pl

# Define pipelines for predictions and ground truth
pred_pipe = Pipeline().source("image_bytes").grayscale().threshold(128)
gt_pipe = Pipeline().source("contour", width=256, height=256)

# Compute metrics in one pass
result = df.with_columns(
    iou=mask_iou(
        pl.col("prediction").cv.pipe(pred_pipe),
        pl.col("ground_truth").cv.pipe(gt_pipe),
    ),
    dice=mask_dice(
        pl.col("prediction").cv.pipe(pred_pipe),
        pl.col("ground_truth").cv.pipe(gt_pipe),
    ),
)

# Aggregate metrics
mean_iou = result["iou"].mean()
mean_dice = result["dice"].mean()
print(f"mIoU: {mean_iou:.4f}, mDice: {mean_dice:.4f}")
```

## Data Augmentation

Create augmentation pipelines:

```python
import random


def create_augmented_pipeline(
    size: int = 224,
    flip_prob: float = 0.5,
    blur_prob: float = 0.3,
) -> Pipeline:
    """Create an augmentation pipeline."""
    pipe = (
        Pipeline()
        .source("image_bytes")
        .resize(height=size, width=size)
    )
    
    if random.random() < flip_prob:
        pipe = pipe.flip_h()
    
    if random.random() < blur_prob:
        pipe = pipe.blur(sigma=random.uniform(0.5, 2.0))
    
    return pipe.normalize().sink("torch")


# Per-epoch augmentation
for epoch in range(num_epochs):
    augment_pipe = create_augmented_pipeline()
    augmented = df.with_columns(
        tensor=pl.col("image").cv.pipeline(augment_pipe)
    )
```

## Multi-Task Pipelines

Output multiple representations for multi-task learning:

```python
# Base image processing
base = (
    pl.col("image")
    .cv.pipe(Pipeline().source("image_bytes").resize(256, 256))
    .alias("resized")
)

# Classification branch
cls_branch = (
    base
    .pipe(Pipeline().crop(top=16, left=16, height=224, width=224))
    .pipe(Pipeline().normalize())
    .alias("classification")
)

# Segmentation branch (full resolution)
seg_branch = (
    base
    .pipe(Pipeline().normalize())
    .alias("segmentation")
)

# Merge and sink
merged = cls_branch.merge_pipe(seg_branch)
result = df.with_columns(
    outputs=merged.sink({
        "classification": "torch",
        "segmentation": "torch",
    })
)
```

## Cloud Data for Training

Train on cloud-stored data:

```python
# Define pipeline for cloud images
pipe = (
    Pipeline()
    .source("file_path")
    .resize(224, 224)
    .normalize()
    .sink("torch")
)

# DataFrame with S3 paths
df = pl.DataFrame({
    "path": ["s3://bucket/train/img001.jpg", ...],
    "label": [0, 1, 2, ...],
})

# Process and convert to PyTorch
result = df.with_columns(tensor=pl.col("path").cv.pipeline(pipe))
dataset = PolarsVisionDataset(result, "_tensor", "label", pipe=None)
```

## Best Practices

1. **Preprocessing Consistency**: Use the same pipeline for train/eval
2. **Lazy Processing**: Process during DataLoader iteration if memory-constrained
3. **Native Metrics**: Use `mask_iou()`, `mask_dice()` for segmentation
4. **Multi-Output**: Extract multiple scales/features in one pass
5. **Cloud Data**: Use `file_path` source for cloud-stored datasets

## Next Steps

- [Perceptual Hashing](operations/hashing.md) - Image similarity for deduplication
- [Multi-Output](composition/multi-output.md) - Extract multiple features

