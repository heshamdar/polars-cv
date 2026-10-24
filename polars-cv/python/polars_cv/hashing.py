"""
Comparison of perceptual hashes: Hamming distance and percentage similarity.

Re-exported from :mod:`polars_cv`.
"""

from __future__ import annotations

import polars as pl

from .expressions import CvNamespace
from .lazy import LazyPipelineExpr
from .pipeline import Pipeline

#: A perceptual hash, as either function below takes it: a pipeline that ends
#: in ``perceptual_hash()``, or a stored hash (``sink("list")`` or
#: ``sink("array", ...)``) as an expression or a column name.
HashLike = LazyPipelineExpr | pl.Expr | str


def _hash_xor(hash1: HashLike, hash2: HashLike) -> LazyPipelineExpr:
    """The XOR of two hashes, in the plugin: a set bit is a differing one.

    A stored hash runs through ``source()``, which reads a ``list[u8]`` or
    ``Array[u8]`` column as the 1-D buffer ``perceptual_hash()`` produced, so
    pipelines and columns meet in the same op; hashes of different lengths are
    refused by the XOR's shape check.
    """

    def lift(h: HashLike) -> LazyPipelineExpr:
        if isinstance(h, LazyPipelineExpr):
            return h
        if isinstance(h, str):
            h = pl.col(h)
        if isinstance(h, pl.Expr):
            return CvNamespace(h).pipe(Pipeline().source())
        raise TypeError(
            f"a hash is a LazyPipelineExpr, a pl.Expr or a column name, got {type(h).__name__}"
        )

    return lift(hash1).bitwise_xor(lift(hash2))


def hamming_distance(hash1: HashLike, hash2: HashLike) -> pl.Expr:
    """
    Compute Hamming distance between two perceptual hashes.

    Either side may be a hash pipeline (``pl.col("img").cv.pipe(hash_pipe)``)
    or a stored hash: a ``list[u8]``/``Array[u8]`` column from
    ``sink("list")``/``sink("array")``, given as an expression or a name. So
    near-duplicate search over stored hashes is a cross join:

        >>> pairs = h.join(h, how="cross", suffix="_b").with_columns(
        ...     bits=hamming_distance("hash", "hash_b")
        ... )

    Args:
        hash1: First hash.
        hash2: Second hash, the same length as the first.

    Returns:
        The number of differing bits (UInt32).
    """
    # A scalar sinks as Float64, which holds any bit count exactly.
    differing = _hash_xor(hash1, hash2).pipe(Pipeline().reduce_popcount())
    return differing.sink("native").cast(pl.UInt32)


def hash_similarity(hash1: HashLike, hash2: HashLike) -> pl.Expr:
    """
    Compute similarity percentage [0, 100] between two hashes.

    Takes the same hashes as :func:`hamming_distance`. The share of bits that
    agree, out of the hashes' own length: a 256-bit hash is scored out of 256
    with nothing to say so.

    Args:
        hash1: First hash.
        hash2: Second hash, the same length as the first.
    """
    xor = _hash_xor(hash1, hash2)
    # The XOR is as long as either hash, so its byte count is the hash length:
    # one graph sinks it beside its popcount, and the length is never assumed.
    differing = xor.pipe(Pipeline().reduce_popcount()).alias("differing")
    counts = xor.alias("xor").merge_pipe(differing)
    out = counts.sink({"xor": "list", "differing": "native"})
    bits = out.struct.field("xor").list.len() * 8
    return 100.0 * (1.0 - out.struct.field("differing") / bits)
