# Functions

Utility functions for working with polars-cv outputs.

## NumPy Conversion

### numpy_from_bytes

Convert pipeline output bytes to NumPy array.

```python
from polars_cv import numpy_from_bytes

arr = numpy_from_bytes(result["tensor"][0])
print(f"Shape: {arr.shape}, dtype: {arr.dtype}")
```

### numpy_shape

Get shape from numpy sink bytes without full conversion.

```python
from polars_cv import numpy_shape

shape = numpy_shape(bytes_data)
print(f"Shape: {shape}")
```

### numpy_dtype

Get dtype from numpy sink bytes.

```python
from polars_cv import numpy_dtype

dtype = numpy_dtype(bytes_data)
print(f"Dtype: {dtype}")
```

## Mask Metrics

### mask_iou

Compute IoU between two mask pipelines.

```python
from polars_cv import mask_iou, Pipeline
import polars as pl

pred_pipe = Pipeline().source("image_bytes").grayscale().threshold(128)
gt_pipe = Pipeline().source("contour", width=256, height=256)

result = df.with_columns(
    iou=mask_iou(
        pl.col("prediction").cv.pipe(pred_pipe),
        pl.col("ground_truth").cv.pipe(gt_pipe),
    )
)
```

### mask_dice

Compute Dice coefficient between two mask pipelines.

```python
from polars_cv import mask_dice

result = df.with_columns(
    dice=mask_dice(
        pl.col("prediction").cv.pipe(pred_pipe),
        pl.col("ground_truth").cv.pipe(gt_pipe),
    )
)
```

## Hash Comparison

### hamming_distance

Compute Hamming distance between two perceptual hashes.

```python
from polars_cv import hamming_distance, Pipeline
import polars as pl

hash_pipe = Pipeline().source("image_bytes").perceptual_hash()

result = df.with_columns(
    distance=hamming_distance(
        pl.col("image_a").cv.pipe(hash_pipe),
        pl.col("image_b").cv.pipe(hash_pipe),
    )
)
```

### hash_similarity

Compute similarity percentage between two perceptual hashes.

```python
from polars_cv import hash_similarity

result = df.with_columns(
    similarity=hash_similarity(
        pl.col("image_a").cv.pipe(hash_pipe),
        pl.col("image_b").cv.pipe(hash_pipe),
        hash_bits=64,  # Match your hash size
    )
)
```

## Types

### CloudOptions

Configuration for cloud storage access.

```python
from polars_cv import CloudOptions

options = CloudOptions(
    aws_region="us-east-1",
    aws_access_key_id="...",
    aws_secret_access_key="...",
)

pipe = Pipeline().source("file_path", cloud_options=options).sink("numpy")
```

### HashAlgorithm

Perceptual hash algorithm selection.

```python
from polars_cv import HashAlgorithm

Pipeline().perceptual_hash(algorithm=HashAlgorithm.PERCEPTUAL)
Pipeline().perceptual_hash(algorithm=HashAlgorithm.AVERAGE)
Pipeline().perceptual_hash(algorithm=HashAlgorithm.DIFFERENCE)
Pipeline().perceptual_hash(algorithm=HashAlgorithm.BLOCKHASH)
```

