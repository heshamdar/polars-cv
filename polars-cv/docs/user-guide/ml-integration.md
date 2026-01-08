# ML Integration

polars-cv integrates with machine learning frameworks for batch preprocessing,
but it's important to understand when to use polars-cv versus traditional tools.

## When to Use polars-cv

polars-cv is designed for **batch-columnar preprocessing**, not per-sample augmentation.
Choose the right tool for each part of your pipeline:

| Use Case | polars-cv | torchvision |
|----------|---------------|-------------|
| **Batch inference** | ✅ Recommended | Works |
| **ETL / feature extraction** | ✅ Recommended | Not designed for |
| **Cloud data loading** | ✅ Recommended | Requires extra libs |
| **Heavy preprocessing** (decode, resize, normalize) | ✅ Recommended | Works |
| **Training with augmentation** | Preprocessing only | ✅ Recommended for augmentation |
| **Random per-sample transforms** | ❌ Not supported | ✅ Recommended |

**Key insight**: polars-cv excels at the *deterministic* parts of your pipeline
(decoding, resizing, normalization), while PyTorch handles *random* augmentation
(flips, rotations, color jitter). The recommended pattern is to use both together.

### What polars-cv Does NOT Do

polars-cv intentionally does not implement random augmentation:

- ❌ Random horizontal/vertical flip
- ❌ Random crop
- ❌ Random rotation
- ❌ Color jitter (brightness, contrast, saturation, hue)
- ❌ Random affine transforms

For these operations, use `torchvision.transforms` in your PyTorch Dataset's
`__getitem__` method, as shown in the patterns below.

## Architecture: Polars vs PyTorch DataLoaders

**polars-cv** is designed for **batch-columnar processing**:

- Processes entire columns/Series at once
- Leverages Rust parallelism and SIMD optimizations
- Lazy evaluation allows query optimization before execution
- Best performance when processing thousands of rows together

**PyTorch DataLoader** is designed for **sample-wise processing**:

- Calls `__getitem__(idx)` for individual samples
- Batches samples *after* individual retrieval
- Uses multiprocessing workers for parallel loading
- Applies transforms per-sample

These paradigms don't directly align, but several patterns enable effective interoperability.

## ImageNet-Style Normalization

For standard ImageNet preprocessing, use the preset normalization with built-in constants:

```python
from polars_cv import Pipeline, IMAGENET_MEAN, IMAGENET_STD

# Standard ImageNet preprocessing pipeline
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(height=256, width=256)
    .crop(top=16, left=16, height=224, width=224)  # Center crop
    .scale(1 / 255.0)  # [0, 255] -> [0, 1] like ToTensor()
    .normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD)
    .sink("torch")
)

# Or load from file paths
file_pipe = (
    Pipeline()
    .source("file_path")  # Load from paths in DataFrame
    .resize(height=256, width=256)
    .crop(top=16, left=16, height=224, width=224)
    .scale(1 / 255.0)
    .normalize(method="preset", mean=IMAGENET_MEAN, std=IMAGENET_STD)
    .sink("torch")
)

# Apply to DataFrame with paths
df = pl.read_parquet("metadata.parquet")  # columns: path, label
result = df.with_columns(tensor=pl.col("path").cv.pipeline(file_pipe))
```

The preset normalization applies channel-wise normalization matching torchvision's
`transforms.Normalize()`. The `scale(1/255.0)` step matches `ToTensor()`.

**ImageNet Constants:**
- `IMAGENET_MEAN = [0.485, 0.456, 0.406]` (RGB channels)
- `IMAGENET_STD = [0.229, 0.224, 0.225]` (RGB channels)

## NumPy Integration

### Converting Output to NumPy

Use `numpy_from_struct()` to convert pipeline output:

```python
from polars_cv import Pipeline, numpy_from_struct
import numpy as np

# Pipeline with numpy sink
pipe = Pipeline().source("image_bytes").resize(224, 224).normalize().sink("numpy")

result = df.with_columns(tensor=pl.col("image").cv.pipeline(pipe))

# Convert to NumPy array (each row is a struct with data, dtype, shape)
arr = numpy_from_struct(result["tensor"][0])
print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
# Shape: (224, 224, 3), dtype: float32
```

### Batch Processing

```python
# Process all images in one pass
arrays = [numpy_from_struct(s) for s in result["tensor"]]
batch = np.stack(arrays)
print(f"Batch shape: {batch.shape}")
# Batch shape: (N, 224, 224, 3)
```

## PyTorch Integration

### Direct Tensor Conversion

```python
import torch
from polars_cv import Pipeline, numpy_from_struct

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
def struct_to_tensor(data: dict) -> torch.Tensor:
    arr = numpy_from_struct(data)
    tensor = torch.from_numpy(arr.copy())
    return tensor.permute(2, 0, 1)  # HWC → CHW

tensors = [struct_to_tensor(s) for s in result["tensor"]]
batch = torch.stack(tensors)
print(f"Batch shape: {batch.shape}")
# Batch shape: torch.Size([N, 3, 224, 224])
```

## PyTorch Dataset Patterns

### Pattern 1: Pre-Epoch Batch Processing (Recommended)

This pattern processes the entire dataset with Polars before the DataLoader iterates.
It's the most efficient approach when you can fit preprocessed data in memory.

```python
from torch.utils.data import Dataset, DataLoader
import polars as pl
from polars_cv import Pipeline, numpy_from_struct
import torch


class PreprocessedPolarsDataset(Dataset):
    """
    PyTorch Dataset with batch preprocessing.
    
    polars-cv preprocesses ALL images in __init__ using batch processing.
    The DataLoader then retrieves already-processed samples efficiently.
    Per-sample augmentations are applied in __getitem__.
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        image_col: str,
        label_col: str,
        pipeline: Pipeline,
        transform: callable | None = None,  # PyTorch augmentations
    ) -> None:
        # Batch preprocess ALL images with polars-cv
        # This leverages Polars' parallel execution and SIMD optimizations
        self.df = df.with_columns(
            _tensor=pl.col(image_col).cv.pipeline(pipeline)
        )
        self.label_col = label_col
        self.transform = transform  # Per-sample augmentation (PyTorch)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.row(idx, named=True)
        arr = numpy_from_struct(row["_tensor"])
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        
        # Apply PyTorch augmentations (varies per-epoch if random)
        if self.transform:
            tensor = self.transform(tensor)
        
        label = row[self.label_col]
        return tensor, label


# Usage
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(224, 224)
    .normalize()
    .sink("torch")
)

# PyTorch transforms for per-epoch augmentation
from torchvision import transforms
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
])

dataset = PreprocessedPolarsDataset(df, "image", "label", pipe, transform=augment)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

for images, labels in dataloader:
    # images: (32, 3, 224, 224)
    # labels: (32,)
    pass
```

**Pros:**

- Leverages full Polars batch optimization
- Preprocessing happens once, not per-epoch
- Works with multi-worker DataLoaders
- Supports per-sample augmentation via PyTorch transforms

**Cons:**

- All preprocessed data must fit in memory
- Cannot change preprocessing per-epoch

**When to use:** Most training scenarios where dataset fits in memory.

### Pattern 2: Per-Epoch Lazy Processing

For scenarios where you want different preprocessing each epoch (e.g., varying blur
or crop parameters), reprocess at each epoch boundary:

```python
class EpochProcessedDataset(Dataset):
    """
    Dataset that reprocesses data each epoch.
    
    Call set_epoch() before each epoch to trigger reprocessing.
    Useful when preprocessing parameters should vary per-epoch.
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        image_col: str,
        label_col: str,
        pipeline_factory: callable,  # Function that returns a Pipeline
        transform: callable | None = None,
    ) -> None:
        self.source_df = df
        self.image_col = image_col
        self.label_col = label_col
        self.pipeline_factory = pipeline_factory
        self.transform = transform
        self.processed_df = None
        self._process()
    
    def _process(self) -> None:
        """Process all images with current pipeline."""
        pipeline = self.pipeline_factory()
        self.processed_df = self.source_df.with_columns(
            _tensor=pl.col(self.image_col).cv.pipeline(pipeline)
        )
    
    def set_epoch(self, epoch: int) -> None:
        """Call at epoch start to reprocess with new augmentations."""
        self._process()
    
    def __len__(self) -> int:
        return len(self.processed_df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.processed_df.row(idx, named=True)
        arr = numpy_from_struct(row["_tensor"])
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, row[self.label_col]


# Usage with varying blur per epoch
import random

def create_augmented_pipeline() -> Pipeline:
    """Create pipeline with random augmentation parameters."""
    pipe = Pipeline().source("image_bytes").resize(224, 224)
    
    # Random blur with varying sigma
    if random.random() < 0.3:
        pipe = pipe.blur(sigma=random.uniform(0.5, 2.0))
    
    # Random horizontal flip
    if random.random() < 0.5:
        pipe = pipe.flip_h()
    
    return pipe.normalize().sink("torch")

dataset = EpochProcessedDataset(
    df, "image", "label",
    pipeline_factory=create_augmented_pipeline
)

for epoch in range(num_epochs):
    dataset.set_epoch(epoch)  # Reprocess with new random augmentations
    for images, labels in DataLoader(dataset, batch_size=32, shuffle=True):
        # Train...
        pass
```

**Pros:**

- Polars augmentations can vary per-epoch
- Still uses batch processing

**Cons:**

- Full reprocessing each epoch (can be slow for large datasets)
- Memory usage during reprocessing

**When to use:** When augmentations that Polars handles (blur, flip, resize) should vary per-epoch.

### Pattern 3: IterableDataset with Streaming Batches

For very large datasets that don't fit in memory, use `IterableDataset` with
Polars streaming engine and batch collection for true lazy iteration:

```python
from torch.utils.data import IterableDataset, DataLoader
import polars as pl
from polars_cv import Pipeline, numpy_from_struct
import torch


class StreamingPolarsDataset(IterableDataset):
    """
    IterableDataset that processes data using Polars streaming engine with batch collection.
    
    Data is processed and yielded in batches as they're produced, enabling true
    lazy iteration without materializing the entire result in memory.
    
    This uses `collect_batches(engine="streaming")` to process chunks incrementally,
    yielding samples as batches are ready rather than waiting for the full result.
    """
    
    def __init__(
        self,
        lazy_df: pl.LazyFrame,
        image_col: str,
        label_col: str,
        pipeline: Pipeline,
        transform: callable | None = None,
        chunk_size: int = 1000,
    ) -> None:
        self.lazy_df = lazy_df
        self.image_col = image_col
        self.label_col = label_col
        self.pipeline = pipeline
        self.transform = transform
        self.chunk_size = chunk_size
    
    def __iter__(self):
        # Build the processing query
        query = (
            self.lazy_df
            .with_columns(
                _tensor=pl.col(self.image_col).cv.pipeline(self.pipeline)
            )
        )
        
        # Process in batches using streaming engine
        # This yields batches as they're processed, not after full materialization
        for batch_df in query.collect_batches(engine="streaming", chunk_size=self.chunk_size):
            for row in batch_df.iter_rows(named=True):
                arr = numpy_from_struct(row["_tensor"])
                tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
                
                if self.transform:
                    tensor = self.transform(tensor)
                
                yield tensor, row[self.label_col]


# Usage
pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(224, 224)
    .normalize()
    .sink("torch")
)

# Create lazy DataFrame
lazy_df = pl.scan_parquet("s3://bucket/images.parquet")

# Note: Shuffling with IterableDataset is limited. For per-epoch shuffling,
# consider using Pattern 1 if your dataset fits in memory, or implement
# custom shuffling logic at the LazyFrame level before processing.

dataset = StreamingPolarsDataset(lazy_df, "image", "label", pipe, chunk_size=1000)
dataloader = DataLoader(dataset, batch_size=32)

for images, labels in dataloader:
    # Train...
    pass
```

**Important notes on streaming batches:**

- `collect_batches(engine="streaming")` processes and yields data in chunks incrementally
- Each batch is materialized only when needed, enabling true out-of-core processing
- Peak memory usage is limited to the chunk size, not the full dataset
- Works seamlessly with cloud data sources (S3, GCS, Azure)
- Adjust `chunk_size` based on available memory and processing speed trade-offs
- **Note**: `collect_batches()` is marked as unstable in Polars, but it's the correct
  method for true lazy iteration. The API may change in future Polars versions.

**Pros:**

- True lazy iteration - batches materialize only when needed
- Lower peak memory during processing (limited to chunk size)
- Works with cloud data sources (S3, GCS, Azure)
- No manual cache management needed

**Cons:**

- Cannot use `num_workers > 0` with IterableDataset reliably
- Shuffling must be handled separately (e.g., add random column to LazyFrame)
- Requires LazyFrame (not regular DataFrame)

**When to use:** Very large datasets, cloud data sources, memory-constrained processing.

## Pattern Comparison

| Pattern | Memory | Per-Epoch Augment | Workers | Shuffling | Best For |
|---------|--------|-------------------|---------|-----------|----------|
| **Pre-Epoch Batch** | High | PyTorch only | ✅ | ✅ | Most training (dataset fits in memory) |
| **Per-Epoch Lazy** | High | Polars + PyTorch | ✅ | ✅ | Varying Polars preprocessing per-epoch |
| **Streaming Batches** | Low | PyTorch only | ❌ | ⚠️ | Very large datasets, cloud sources |

**Memory**: High = full dataset in memory, Low = chunked/streaming  
**Workers**: ✅ = supports `num_workers > 0`, ❌ = IterableDataset limitation  
**Shuffling**: ✅ = native support, ⚠️ = must handle at LazyFrame level

## Multi-Worker Considerations

When using `DataLoader(num_workers > 0)`:

1. **Polars DataFrames are picklable** - they can be sent to worker processes
2. **Each worker gets a copy** - memory usage scales with workers
3. **Avoid lazy processing in workers** - process before DataLoader iteration
4. **IterableDataset + workers** - requires careful worker sharding

```python
# Safe: Preprocess before DataLoader
dataset = PreprocessedPolarsDataset(df, "image", "label", pipe)
loader = DataLoader(dataset, batch_size=32, num_workers=4)  # Works!

# Risky: Processing in __getitem__
class BadDataset(Dataset):
    def __getitem__(self, idx):
        # Don't do this - creates new DataFrame per sample!
        row_df = pl.DataFrame({"image": [self.df["image"][idx]]})
        processed = row_df.with_columns(
            _tensor=pl.col("image").cv.pipeline(self.pipeline)
        )
        ...
```

## Segmentation Metrics

Use native `mask_iou()` and `mask_dice()` for evaluation:

```python
from polars_cv import Pipeline, mask_iou, mask_dice
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

## Division of Responsibilities

For optimal results, divide preprocessing between Polars and PyTorch:

**Use polars-cv for:**

- Heavy preprocessing (decode, resize, normalize)
- Operations that benefit from batch processing
- Cloud data loading (S3, GCS, Azure)
- Deterministic operations (same every epoch)

**Use PyTorch transforms for:**

- Random augmentations (flips, rotations, crops)
- Per-sample variations
- Operations that should vary per-epoch
- GPU-accelerated augmentations (if using torchvision.transforms.v2)

```python
# Optimal hybrid approach
polars_pipe = (
    Pipeline()
    .source("image_bytes")
    .resize(224, 224)
    .normalize()
    .sink("torch")
)

pytorch_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

dataset = PreprocessedPolarsDataset(
    df, "image", "label", 
    pipeline=polars_pipe,
    transform=pytorch_augment
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
    "path": ["s3://bucket/train/img001.jpg", "s3://bucket/train/img002.jpg"],
    "label": [0, 1],
})

# Process all cloud images in batch
result = df.with_columns(tensor=pl.col("path").cv.pipeline(pipe))

# Then use PreprocessedPolarsDataset pattern
dataset = PreprocessedPolarsDataset(result, "_tensor_placeholder", "label", pipe=None)
```

## Best Practices

1. **Batch First**: Always preprocess with Polars in batches, not per-sample
2. **Augment with PyTorch**: Use `torchvision.transforms` for random augmentations
3. **Hybrid Approach**: Polars for heavy ops, PyTorch for per-epoch variation
4. **Memory Awareness**: Choose pattern based on dataset size vs. available RAM
5. **Worker Safety**: Process before DataLoader when using `num_workers > 0`
6. **Native Metrics**: Use `mask_iou()`, `mask_dice()` for segmentation eval

## Benchmark: Batch Preprocessing Comparison

To compare batch preprocessing performance between HuggingFace/torchvision and
polars-cv for inference workloads, run the comparison benchmark:

```bash
cd polars-cv
python -m benchmarks.inference_pipeline_comparison --num-images 1000 --batch-size 32
```

This benchmark:

- Generates synthetic ImageFolder data with Parquet metadata
- Compares **batch preprocessing time** (both upfront, fair comparison)
- Measures DataLoader throughput and memory usage
- Runs inference serving comparison with ResNet18
- Verifies that both pipelines produce equivalent outputs

**Note**: This benchmark focuses on deterministic preprocessing for inference.
For training workloads, use the hybrid pattern shown above where polars-cv
handles heavy preprocessing and PyTorch handles random augmentation.

See `benchmarks/inference_pipeline_comparison.py` for the full implementation.

## Next Steps

- [Perceptual Hashing](operations/hashing.md) - Image similarity for deduplication
- [Multi-Output](composition/multi-output.md) - Extract multiple features
