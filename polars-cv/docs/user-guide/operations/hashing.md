# Perceptual Hashing

polars-cv provides perceptual hashing for finding visually similar images. Unlike cryptographic hashes, perceptual hashes produce similar fingerprints for similar images.

## Basic Usage

```python
from polars_cv import Pipeline
import polars as pl

# Create hash pipeline
hash_pipe = Pipeline().source("image_bytes").perceptual_hash()

df = pl.DataFrame({"image": [image_bytes]})
result = df.with_columns(
    hash=pl.col("image").cv.pipe(hash_pipe).sink("list")
)
```

## Algorithms

| Algorithm | Speed | Robustness | Best For |
|-----------|-------|------------|----------|
| `perceptual` | Medium | High | Most use cases (default) |
| `average` | Fastest | Lower | Quick approximate matching |
| `difference` | Fast | Medium | General purpose |
| `blockhash` | Medium | High | Crop-resistant matching |

```python
Pipeline().source("image_bytes").perceptual_hash(algorithm="average")
```

## Comparing Hashes

Use `hamming_distance()` and `hash_similarity()` for batch comparison:

```python
from polars_cv import hamming_distance, hash_similarity

# Compare hashes from two pipelines
result = df.with_columns(
    similarity=hash_similarity(
        pl.col("image_a").cv.pipe(hash_pipe),
        pl.col("image_b").cv.pipe(hash_pipe)
    )
)
```

## Next Steps

- [Image Operations](image-ops.md) - Preprocessing before hashing
- [Reductions](reductions.md) - Custom analysis

