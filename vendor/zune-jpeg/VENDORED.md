# Vendored `zune-jpeg` 0.5.15

Source: crates.io `zune-jpeg` 0.5.15 (MIT OR Apache-2.0 OR Zlib, all three
licence files kept), wired in by the workspace's `[patch.crates-io]`. It
replaces the 0.5.x zune-jpeg `image` decodes JPEGs with as well as the one
polars-cv calls directly; the 0.4.x zune-jpeg another dependency pulls in is
untouched.

Only `src/`, the licences, README and Changelog are vendored. The manifest
drops the (empty) dev-dependencies; tests, benches and fuzz targets were
never published. Every feature `src` names in `cfg`s is kept.

## Local changes

- Scaled output (`JpegDecoder::set_idct_scale`, `IdctScale`; new module
  `src/scaled.rs`): decode at 1/2, 1/4 or 1/8 of each dimension, sizes
  rounded up as libjpeg does (`ceil(len * n / 8)`), the libjpeg way. Each
  8x8 block goes through a reduced IDCT straight to its 4x4, 2x2 or 1x1
  output (libjpeg's `jidctred.c`; the 1x1 is zune's own DC arithmetic);
  a subsampled component gets a larger IDCT where that lands it at the
  output size (libjpeg's `jdmaster.c` rule), and what is left is
  upsampled with libjpeg's `jdsample.c` filters; colour conversion is
  zune's. Covers baseline and progressive, every subsampling with integral
  expansion, restart intervals, and composes with Luma output. A baseline
  file decodes through the full decode's MCU loop: one whose components
  all arrive in the first scan one MCU row at a time, colour converted at
  once where every component lands at the output size and copied into
  whole-image planes otherwise; one with components in separate scans
  straight into those planes. Scalar code, no `unsafe`.
  `dimensions()` and `output_buffer_size()` report the scaled size.
  Touches the MCU loop only to place a block by its component's
  `block_size` (8 unless scaled) and to call its `scaled_idct`. Within a
  step of libjpeg-turbo's scaled decode; the default, `IdctScale::Full`,
  is byte-identical to the release. Guarded by
  `view-buffer/tests/it/jpeg_scaled.rs`.
- Progressive at 1/8 (`JpegDecoder::ac_coefficients_unread`,
  `skip_entropy_coded_data` in `src/mcu_prog.rs`): an AC scan of a
  component whose 1/8 IDCT is the 1x1 (its DC alone), or which the output
  does not read, is skipped to the next marker that is not a restart
  marker or stuffed byte, not Huffman-decoded; DC scans, refinements
  included, and a subsampled chroma's AC scans (its 1/8 IDCT is the 2x2)
  are decoded as before. The output is byte-identical to decoding every
  scan; only corrupt data in a skipped scan no longer fails or truncates
  the decode. Guarded by the same test file.

Drop this directory and the patch once upstream ships an equivalent.
