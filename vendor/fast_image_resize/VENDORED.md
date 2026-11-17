# Vendored `fast_image_resize` 6.1.0

Source: crates.io `fast_image_resize` 6.1.0 (MIT OR Apache-2.0, licence files
kept), wired in by the workspace's `[patch.crates-io]`.

Only `src/`, the licences, README and CHANGELOG are vendored. The manifest
drops tests, benches, dev-dependencies, profiles and the `for_testing`
feature's test dependencies (the feature is kept, empty, because `src` names
it in `cfg`s).

## Local changes

- `Resizer` caches convolution coefficients (`src/resizer.rs`:
  `CoefficientsKey`, `CoefficientsCache`), keyed by one axis's source size,
  crop (exact bits), destination size, filter and kernel adaptivity; eight
  most-recently-used entries. They were recomputed on every call: about 10%
  of a decode → resize(224, lanczos3) → normalize pipeline's wall time. A hit
  returns a copy of what was computed, so results are bit-identical.
  `FilterType::Custom` bypasses the cache: its equality ignores the function.
  Guarded by `view-buffer/tests/it/resize_coefficient_cache.rs`.

Drop this directory and the patch once upstream ships an equivalent.
