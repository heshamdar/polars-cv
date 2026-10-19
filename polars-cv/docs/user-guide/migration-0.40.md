# Migrating from 0.39 to 0.40

0.40 gives every perceptual-hash algorithm one bit order, scores hash
comparisons out of the hashes' own length, and changes the graph wire format.
This page lists what to check coming from 0.39; the
[changelog](../changelog.md) has the full list.

## Results that change

**`average`, `difference` and `blockhash` hashes pack their bits most
significant first.** `perceptual` (the default) always did, as imagehash's hex
does; the other three came out least significant first, so one hash column
held two bit orders depending on `algorithm=`. A hash stored by 0.39 with one
of those three algorithms differs from 0.40's: recompute it, or reverse the
bits of each byte.

```python
REVERSED = [int(f"{b:08b}"[::-1], 2) for b in range(256)]
stored.with_columns(
    pl.col("hash").list.eval(
        pl.element().replace_strict(range(256), REVERSED, return_dtype=pl.UInt8)
    )
)
```

Hamming distances between two hashes of the same algorithm are unchanged, and
`perceptual` hashes are unchanged.

**`hamming_distance()` returns `UInt32`,** a count of bits, not `Float64`.
Code that compares it with a float still works; a schema pinned to `Float64`
does not.

## Signatures that change

**`hash_similarity()` has no `hash_bits=`.** It defaulted to 64, so a hash
of any other length was scored out of 64 bits unless you restated the size.
The length is now read from the hashes. Drop the argument; passing it raises
`TypeError`.

```python
hash_similarity(a, b)  # was hash_similarity(a, b, hash_bits=256)
```

Both functions now also take stored hashes (a `sink("list")`/`sink("array")`
column, as an expression or a name), see
[Perceptual Hashing](operations/hashing.md).

## Persisted graphs

**The graph wire format is version 2.** A graph serialized by 0.39 (a stored
`graph_json`, or one written by hand) is refused by its version, with a
message to rebuild it: build the pipeline again and serialize it with 0.40.
Pipelines built through `Pipeline` and `.cv.pipe()` need no change.

If you build graphs with `PipelineGraph.add_node` directly, `upstream=` is
replaced by `input=` (keyword-only, like `alias=`): a node reads either a
column or one other node, and a node that reads both, or neither, is refused
where it is added. The nodes its ops read (a binary operand, a mask,
`rasterize(shape=)`) are dependencies already and are no longer listed.
