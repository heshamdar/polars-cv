# ML Integration

polars-vision integrates with machine learning frameworks, but understanding the architectural differences between Polars and PyTorch is crucial for efficient pipelines.

## Architecture: Polars vs PyTorch DataLoaders

**Polars-vision** is designed for **batch-columnar processing**:

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
# Process all images in one pass
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

## PyTorch Dataset Patterns

### Pattern 1: Pre-Epoch Batch Processing (Recommended)

This pattern processes the entire dataset with Polars before the DataLoader iterates.
It's the most efficient approach when you can fit preprocessed data in memory.

```python
from torch.utils.data import Dataset, DataLoader
import polars as pl
from polars_vision import Pipeline, numpy_from_bytes
import torch


class PreprocessedPolarsDataset(Dataset):
    """
    PyTorch Dataset with batch preprocessing.
    
    polars-vision preprocesses ALL images in __init__ using batch processing.
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
        # Batch preprocess ALL images with polars-vision
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
        arr = numpy_from_bytes(row["_tensor"])
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
        arr = numpy_from_bytes(row["_tensor"])
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

### Pattern 3: IterableDataset with Streaming

For very large datasets that don't fit in memory, use `IterableDataset` with
Polars streaming execution:

```python
from torch.utils.data import IterableDataset, DataLoader
import polars as pl
from polars_vision import Pipeline, numpy_from_bytes
import torch


class StreamingPolarsDataset(IterableDataset):
    """
    IterableDataset that processes data using Polars streaming engine.
    
    Data is processed in chunks by Polars' streaming engine, reducing
    peak memory usage for very large datasets.
    
    Note: Streaming processes the full query, but in chunks. The entire
    result is still materialized before yielding - this is for memory
    efficiency during processing, not true lazy iteration.
    """
    
    def __init__(
        self,
        lazy_df: pl.LazyFrame,
        image_col: str,
        label_col: str,
        pipeline: Pipeline,
        transform: callable | None = None,
    ) -> None:
        self.lazy_df = lazy_df
        self.image_col = image_col
        self.label_col = label_col
        self.pipeline = pipeline
        self.transform = transform
    
    def __iter__(self):
        # Process with streaming engine (processes in chunks internally)
        df = (
            self.lazy_df
            .with_columns(
                _tensor=pl.col(self.image_col).cv.pipeline(self.pipeline)
            )
            .collect(engine="streaming")
        )
        
        for row in df.iter_rows(named=True):
            arr = numpy_from_bytes(row["_tensor"])
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

dataset = StreamingPolarsDataset(lazy_df, "image", "label", pipe)
dataloader = DataLoader(dataset, batch_size=32)

for images, labels in dataloader:
    # Train...
    pass
```

**Important notes on streaming:**

- `collect(engine="streaming")` processes data in chunks internally
- This reduces peak memory during *processing*, not during iteration
- The full result is still materialized before the iterator yields
- For true out-of-core processing, use chunked reading (Pattern 4)

**Pros:**

- Lower peak memory during processing
- Works with cloud data sources

**Cons:**

- Cannot use `num_workers > 0` with IterableDataset reliably
- Shuffling must be handled separately
- Data still materializes before iteration

**When to use:** Cloud data sources, memory-constrained processing.

### Pattern 4: Chunked Processing (Large Datasets)

For datasets too large to fit in memory, process in chunks:

```python
class ChunkedPolarsDataset(Dataset):
    """
    Dataset that processes data in chunks on-demand.
    
    Only keeps one chunk of preprocessed data in memory at a time.
    Chunks are processed when first accessed and cached.
    """
    
    def __init__(
        self,
        df: pl.DataFrame,
        image_col: str,
        label_col: str,
        pipeline: Pipeline,
        chunk_size: int = 1000,
        transform: callable | None = None,
    ) -> None:
        self.df = df
        self.image_col = image_col
        self.label_col = label_col
        self.pipeline = pipeline
        self.chunk_size = chunk_size
        self.transform = transform
        self._chunk_cache = {}  # chunk_idx -> processed_df
        self._max_cached_chunks = 3
    
    def _get_chunk(self, chunk_idx: int) -> pl.DataFrame:
        """Get or compute a chunk."""
        if chunk_idx not in self._chunk_cache:
            # Evict oldest chunks if cache is full
            while len(self._chunk_cache) >= self._max_cached_chunks:
                oldest = min(self._chunk_cache.keys())
                del self._chunk_cache[oldest]
            
            # Process this chunk
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, len(self.df))
            chunk_df = self.df.slice(start, end - start)
            
            processed = chunk_df.with_columns(
                _tensor=pl.col(self.image_col).cv.pipeline(self.pipeline)
            )
            self._chunk_cache[chunk_idx] = processed
        
        return self._chunk_cache[chunk_idx]
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size
        
        chunk = self._get_chunk(chunk_idx)
        row = chunk.row(local_idx, named=True)
        
        arr = numpy_from_bytes(row["_tensor"])
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)
        
        if self.transform:
            tensor = self.transform(tensor)
        
        return tensor, row[self.label_col]


# Usage
pipe = Pipeline().source("image_bytes").resize(224, 224).normalize().sink("torch")

dataset = ChunkedPolarsDataset(df, "image", "label", pipe, chunk_size=1000)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Pros:**

- Works with datasets larger than memory
- Still leverages batch processing within chunks
- Compatible with shuffling and workers

**Cons:**

- Cache management overhead
- Shuffling may cause cache thrashing
- Less efficient than full batch processing

**When to use:** Datasets too large to preprocess entirely.

## Pattern Comparison

| Pattern | Memory | Per-Epoch Augment | Workers | Best For |
|---------|--------|-------------------|---------|----------|
| **Pre-Epoch Batch** | High | PyTorch only | ✅ | Most training |
| **Per-Epoch Lazy** | High | Polars + PyTorch | ✅ | Varying augmentation |
| **Streaming** | Medium | PyTorch only | ❌ | Cloud data |
| **Chunked** | Low | PyTorch only | ⚠️ | Large datasets |

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

## Division of Responsibilities

For optimal results, divide preprocessing between Polars and PyTorch:

**Use polars-vision for:**

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

## Next Steps

- [Perceptual Hashing](operations/hashing.md) - Image similarity for deduplication
- [Multi-Output](composition/multi-output.md) - Extract multiple features
