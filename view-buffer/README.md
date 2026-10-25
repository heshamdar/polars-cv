# view-buffer

The zero-copy, stride-aware tensor engine behind
[polars-cv](https://github.com/heshamdar/polars-cv).

view-buffer is developed and versioned with polars-cv (same version number) and
is not published to crates.io; depend on it by path or git. Most users want
polars-cv itself, which exposes everything here as Polars expressions.

## What it provides

- **`ViewBuffer`**: an N-dimensional, strided array over shared storage (a Rust
  `Vec` or an Arrow buffer). Transpose, reshape, flip, crop, squeeze and
  expand-dims change only the layout and copy nothing.
- **`ViewExpr`**: a lazy expression graph. `.plan()` optimises it (fusing runs
  of element-wise scalar ops into one pass, reading views in place) and
  `.execute()` runs it.
- **Operations with declared contracts**: every op states its output shape,
  dtype, memory effect and spatial dependency (`ops::Op`), and the planner
  reads those rules rather than restating them. Image ops cover resize,
  colour conversion, filters, morphology, distance transforms, lookup tables,
  affine warps, histograms and perceptual hashes.
- **Geometry**: contour extraction, rasterisation (with supersampling),
  measures, shaping (`offset`, `stroke`, `smooth`, set operations) and
  pairwise matching (`geometry/`).
- **Interop**: zero-copy views to and from `ndarray`, Arrow and polars-arrow,
  plus `image`-crate decoding and encoding.
- **The VIEW binary protocol** (`protocol.rs`): a 64-byte header plus shape,
  strides and contiguous data, for moving buffers between processes or
  pipelines.

## Features

Image decoding/encoding (`ImageAdapter`), the resize kernels and perceptual
hashing are always built in; the features below gate the interop modules.

| Feature | Enables |
|---------|---------|
| `ndarray_interop` (default) | Zero-copy `ndarray::ArrayViewD` views (`AsNdarray`, `FromNdarray`) |
| `arrow_interop` | Zero-copy Arrow buffers (`FromArrow`, `ToArrow`) |
| `polars_interop` | polars-arrow buffers, as the polars-cv plugin uses |
| `serde` | `Serialize`/`Deserialize` for the op types |
| `all_interops` | Every interop above |

## Examples

```toml
[dependencies]
view-buffer = { git = "https://github.com/heshamdar/polars-cv", features = ["ndarray_interop"] }
```

A lazy graph over a buffer:

```rust
use view_buffer::{DType, ViewBuffer, ViewExpr};

let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
let buffer = ViewBuffer::from_vec(data);

let result = ViewExpr::new_source(buffer)
    .reshape(vec![10, 10])  // zero-copy
    .transpose(vec![1, 0])  // zero-copy
    .scale(2.0)             // fused element-wise pass
    .relu()
    .cast(DType::U8)
    .plan()
    .execute();

assert_eq!(result.shape(), &[10, 10]);
```

An image pipeline:

```rust
use view_buffer::{FilterType, ImageAdapter, ViewExpr};

let buffer = ImageAdapter::open("input.png")?;
let result = ViewExpr::new_source(buffer)
    .resize(400, 300, FilterType::Lanczos3) // width, height
    .crop(vec![50, 50, 0], vec![250, 250, 3]) // start, end
    .flip(vec![1])
    .grayscale()
    .threshold(128.0)
    .plan()
    .execute();
ImageAdapter::save(&result, "output.png")?;
```

To and from `ndarray`, and the binary protocol:

```rust
use view_buffer::{AsNdarray, FromNdarray, ViewBuffer};

let buffer = ViewBuffer::from_array(ndarray::Array2::<f32>::ones((100, 100)).into_dyn());
let view = buffer.as_array_view::<f32>()?; // no copy
assert_eq!(view.sum(), 10_000.0);

let blob = buffer.to_blob();
let back = ViewBuffer::from_blob(&blob)?;
assert_eq!(back.shape(), buffer.shape());
```

## Development

Run the tests through the PyO3 environment wrapper so they do not invalidate
the polars-cv build (see the repository's `CLAUDE.md`):

```bash
scripts/with-pyo3-env.sh cargo test -p view-buffer --all-features
```

Integration tests are one binary: add a module to `tests/it/main.rs` rather
than a new `tests/*.rs`. The module tree, the op-contract rules and the
guards that hold them are in [`AGENTS.md`](AGENTS.md).

## License

Licensed under either of [MIT](../LICENSE-MIT) or [Apache-2.0](../LICENSE-APACHE),
at your option.
