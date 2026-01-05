# Perceptual Hashing

polars-vision provides perceptual image hashing for finding visually similar images.

## Overview

Unlike cryptographic hashes (MD5, SHA), perceptual hashes produce similar fingerprints for visually similar images, even after transformations.

**Use Cases:**

- Duplicate detection in large datasets
- Image similarity search
- Content deduplication
- Copyright detection

## Basic Usage

```python
from polars_vision import Pipeline
import polars as pl

# Create hash pipeline
hash_pipe = Pipeline().source("image_bytes").perceptual_hash().sink("list")

df = pl.DataFrame({"image": [image_bytes]})
result = df.with_columns(hash=pl.col("image").cv.pipeline(hash_pipe))
```

## Hash Algorithms

```python
from polars_vision import HashAlgorithm

Pipeline().perceptual_hash(algorithm=HashAlgorithm.AVERAGE)
Pipeline().perceptual_hash(algorithm=HashAlgorithm.DIFFERENCE)
Pipeline().perceptual_hash(algorithm=HashAlgorithm.PERCEPTUAL)  # Default
Pipeline().perceptual_hash(algorithm=HashAlgorithm.BLOCKHASH)
```

| Algorithm | Speed | Robustness | Best For |
|-----------|-------|------------|----------|
| `AVERAGE` | Fastest | Lower | Quick approximate matching |
| `DIFFERENCE` | Fast | Medium | General purpose |
| `PERCEPTUAL` | Medium | High | Most use cases (default) |
| `BLOCKHASH` | Medium | High | Crop-resistant matching |

## Hash Size

Configure the hash size (default 64 bits):

```python
Pipeline().perceptual_hash(hash_size=64)   # 8x8 = 64 bits
Pipeline().perceptual_hash(hash_size=256)  # 16x16 = 256 bits
```

Larger hashes provide more precision but require more storage.

## Comparing Hashes

### Native Functions

Use `hamming_distance()` and `hash_similarity()` for efficient batch comparison:

```python
from polars_vision import hamming_distance, hash_similarity, Pipeline
import polars as pl

# Define hash pipelines
pipe_a = Pipeline().source("image_bytes").perceptual_hash()
pipe_b = Pipeline().source("image_bytes").perceptual_hash()

# Compare all pairs using cross-join
left = df.select(pl.col("id").alias("id_a"), pl.col("image").alias("image_a"))
right = df.select(pl.col("id").alias("id_b"), pl.col("image").alias("image_b"))
cross = left.join(right, how="cross")

# Compute similarity using native functions
result = cross.with_columns(
    distance=hamming_distance(
        pl.col("image_a").cv.pipe(pipe_a),
        pl.col("image_b").cv.pipe(pipe_b),
    ),
    similarity=hash_similarity(
        pl.col("image_a").cv.pipe(pipe_a),
        pl.col("image_b").cv.pipe(pipe_b),
        hash_bits=64,
    ),
)
```

### Understanding Results

- **Hamming Distance**: Number of differing bits (0 = identical)
- **Similarity**: Percentage of matching bits (100% = identical)

| Similarity | Interpretation |
|------------|----------------|
| 95-100% | Likely duplicate or near-duplicate |
| 85-95% | Very similar (minor edits) |
| 70-85% | Somewhat similar |
| < 70% | Different images |

## Robustness Testing

Perceptual hashes are robust to common transformations:

```python
# Original and transformed versions
original = test_image
resized = resize_and_back(test_image)  # 64→256 pixels
blurred = apply_blur(test_image)
jpeg = convert_to_jpeg(test_image)

# All should have high similarity to original
# Typically 90%+ for minor transformations
```

**Robust To:**

- Resizing (scale up/down)
- Light blur or smoothing
- Format conversion (PNG → JPEG)
- Minor color adjustments
- Light compression artifacts

**NOT Robust To:**

- Heavy cropping
- Rotation (unless using BLOCKHASH)
- Major color changes
- Significant content edits

## Duplicate Detection

Complete example of finding duplicates:

```python
from polars_vision import Pipeline, hash_similarity
import polars as pl

# Hash all images
hash_pipe = Pipeline().source("image_bytes").perceptual_hash()

# Create pairwise comparison
left = df.select(pl.col("id").alias("id_a"), pl.col("image").alias("image_a"))
right = df.select(pl.col("id").alias("id_b"), pl.col("image").alias("image_b"))
cross = left.join(right, how="cross").filter(pl.col("id_a") < pl.col("id_b"))

# Compute similarity
result = cross.with_columns(
    similarity=hash_similarity(
        pl.col("image_a").cv.pipe(hash_pipe),
        pl.col("image_b").cv.pipe(hash_pipe),
        hash_bits=64,
    )
)

# Find potential duplicates
duplicates = result.filter(pl.col("similarity") >= 85.0)
print(f"Found {len(duplicates)} potential duplicate pairs")
```

## Performance Tips

1. **Use Native Functions**: `hamming_distance()` and `hash_similarity()` are optimized for batch processing
2. **Pre-filter**: Filter by other metadata before expensive hash comparison
3. **Index by Hash**: For very large datasets, consider hash-based indexing
4. **Choose Algorithm**: AVERAGE is fastest, PERCEPTUAL is most robust

## Next Steps

- [Image Operations](image-ops.md) - Preprocessing before hashing
- [Multi-Output](../composition/multi-output.md) - Hash multiple versions

