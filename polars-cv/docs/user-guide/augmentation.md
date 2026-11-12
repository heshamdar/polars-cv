# Training-Time Augmentation

polars-cv has no random ops. A random augmentation is a deterministic op whose
parameter is a column, and the randomness is in how you fill the column. That
keeps every augmentation reproducible, inspectable (the parameters are in the
frame, next to the image they were applied to) and free to depend on anything
else in the row.

Every block on this page is a section of
[`examples/14_augmentation.py`](https://github.com/heshamdar/polars-cv/blob/main/polars-cv/examples/14_augmentation.py),
which the test suite runs.

## Parameters as columns

A random-resized-crop needs a window per row; a flip needs a boolean; a
rotation an angle. Each is a column, computed from a uniform value per row:

```python
--8<-- "examples/14_augmentation.py:params"
```

Three ways to produce the uniform values, in order of how far the
reproducibility reaches:

- **A hash of (row id, epoch, parameter name)**, as above. It needs no state
  and does not depend on the batch a row lands in or on the order rows
  arrive in, so it works under the streaming engine and across workers.
  Polars guarantees `hash` only within one Polars version: the same seed
  gives other values after an upgrade.
- **A NumPy generator materialised into columns**:
  `df.with_columns(u=pl.Series(rng.random(df.height)))`. Reproducible across
  versions, but the values follow row position, so the frame must be in
  memory and in a fixed order.
- **`pl.int_range(pl.len())` as the id** when the data has none. It numbers
  rows by position, so it has the same limit.

## The pipeline

Any parameter that does not change the output's rank, dtype or domain takes
an expression ([Parameters](concepts/parameters.md)), which covers what
augmentation uses: `crop`'s window, `flip_h(when=)`/`flip_v(when=)`,
`rotate`'s angle, `adjust_contrast`, `adjust_brightness`, `adjust_gamma`,
`blur`'s sigma, and every coefficient of a `warp_affine` matrix.

```python
--8<-- "examples/14_augmentation.py:pipeline"
```

The crop comes first, on the full image, and the resize brings every row to
one size, so the batch leaves through the `array` sink as a single array
(see [Choosing a sink](concepts/pipelines.md#choosing-a-sink)). The order
matters for cost as well as for the result: `rotate` and `adjust_contrast`
then work on 64x64 pixels, not on the full image.

Geometric augmentations can be one resample. `rotate`, a scale and a
translation compose into one 2x3 matrix; write its six coefficients as
expressions and pass them to `warp_affine`, and the image is interpolated
once where three chained ops would interpolate three times
([Image Operations](operations/image-ops.md)).

## An epoch is a query

An epoch's parameters are a function of the epoch number, so each epoch is
the same query with another number in it:

```python
--8<-- "examples/14_augmentation.py:epochs"
```

The same seed and epoch reproduce the batch exactly, and a row's result does
not depend on the rows around it.

## What this is not

- **Not a shuffling loader.** A query returns rows in the frame's order.
  Shuffle with Polars (`df.sample(fraction=1.0, shuffle=True, seed=epoch)`)
  before the pipeline, or in the loader that consumes the result.
- **Not batch-level augmentation.** MixUp and CutMix combine two rows. Join
  the frame to a shuffled copy of itself so each row carries both images, and
  blend them with the [binary ops](composition/binary-ops.md); nothing here
  does it for you.
- **Not a per-sample Python callback.** There is no hook that runs your
  function on each image. An augmentation that has no op is done in NumPy
  after the sink.
