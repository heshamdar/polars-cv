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

`perceptual` is the standard pHash that `imagehash.phash` computes:
grayscale, a Lanczos resize to 32×32, a 2-D DCT, and the 8×8 lowest
frequencies compared with their median. The bits come row-major, most
significant first, so `int.from_bytes(bytes(h), "big")` reads a 64-bit hash
like the integer `int(str(imagehash.phash(img)), 16)`. The two are close but
not identical: the grayscale and the resize are polars-cv's own ops, which
round differently from Pillow's, so 64-bit hashes typically differ from
imagehash's in 0 to 2 bits. `hash_size` is the number of bits:
`hash_size=256` is the 16×16 block, as in `imagehash.phash(img, hash_size=16)`.

Every algorithm packs its bits the same way: row-major, most significant bit
first, as imagehash's hex does. So `average` and `difference` read like
`imagehash.average_hash` and `imagehash.dhash` too.

## Comparing Hashes

Use `hamming_distance()` and `hash_similarity()` for batch comparison. Each
side is either a hash pipeline or a stored hash: a column from `sink("list")`
or `sink("array", ...)`, as an expression or a column name.

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

Hashes are usually computed once and kept. Near-duplicate search over them is
a cross join of the hash column with itself:

```python
hashes = df.select("path", hash=pl.col("image").cv.pipe(hash_pipe).sink("list"))
pairs = (
    hashes.join(hashes, how="cross", suffix="_b")
    .filter(pl.col("path") < pl.col("path_b"))
    .with_columns(bits=hamming_distance("hash", "hash_b"))
    .filter(pl.col("bits") <= 10)
)
```

Two hashes of different lengths are refused rather than compared.

## Next Steps

- [Image Operations](image-ops.md) - Preprocessing before hashing
- [Reductions](reductions.md) - Custom analysis
