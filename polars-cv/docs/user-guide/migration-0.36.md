# Migrating from 0.35 to 0.36

0.36 is a performance release with one new parameter. Nothing is removed, and
every output dtype is unchanged. One result can change for an input you are
unlikely to have. This page lists it; the [changelog](../changelog.md) has the
full list.

## Results that change

**`equalize_histogram` and `perceptual_hash` of a `u64` or `i64` image.** Both
work in `u8`, and the 64-bit integers reached it through a wrapping cast: an
`i64` 300 became 44 and a -5 became 251. They now saturate, as every narrower
integer already did (300 → 255, -5 → 0). Only values outside `[0, 255]`
change.

## New

**`compact=True` on the `numpy`, `torch` and `ndarray` sinks.** A tensor sink
hands a row over without copying where it can, so a cropped or flipped row
carries the whole storage it was cut from, and Parquet and IPC write all of
it. `compact=True` makes each row's `data` exactly its elements, row-major, at
offset 0, copying only rows that are not already. Use it for a column you will
persist:

```python
df.with_columns(
    crop=pl.col("image").cv.pipe(pipe).sink("ndarray", compact=True)
).write_parquet("crops.parquet")
```

The other sinks reject `compact`.

## Behaviour you get for free

- **Fewer copies of a view.** Blur, erode, dilate, morphology gradient, pad,
  `pad_to_size`, `equalize_histogram` and `channel_swap` read a crop or a
  vertical flip where it lies; `convolve2d` and `convert_color` read any view
  without packing it first; binary ops (`add`, `blend`, `apply_mask`, …) read
  each operand block by block instead of packing both. A tensor sink that
  must copy a view copies it once, not twice.
- **fast_image_resize 6.1**, with a faster AVX2 vertical pass for `u8`. Resize
  output is unchanged.

## For contributors

`scripts/verify.sh` now runs only what CI runs on every push (about 5 minutes
on 4 cores). Add `--slow` for the slow lane, before a release and after engine
changes. `--fast` is gone: it was the new default, and passing it fails with a
message.
