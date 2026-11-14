# Changelog

All notable changes to **polars-cv** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

polars-cv is 0.x: a minor release may break the API, and each one that does
has a [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration/).
Entries up to 0.41.0 were condensed in November 2026; their full original text
is in the git history of this file.

## [Unreleased]

Graph inspection is unified under `expr.graph()`/`explain()`, the tensor sink
is now one `sink("tensor")` that can be read back, and `sink()` gains typed
keywords. Path reads now require `allowed_roots`.

### Breaking changes

- `sink(..., return_expr=False)` is removed; write
  `expr.graph(format, opt_flags=...)`, or `expr.graph(Sink.jpeg(quality=70))`
  for a sink with settings.
- `sink()`'s `opt_flags` is keyword-only: `sink("png", False)` now raises
  `TypeError`.
- `Pipeline.explain()` and `Pipeline.to_graph(column)` are removed; use
  `pl.col(c).cv.pipe(pipe).graph(format)` or `.explain(format)`.
- `explain(optimized=False, opt_flags=...)` is refused, as the two disagree.
- `PipelineGraph.show_graph()` returns a `graphviz.Digraph`, and the `viz` extra
  is `graphviz` alone (`networkx` and `pydot` are no longer used).
- `sink("numpy")`, `sink("torch")` and `sink("ndarray")` are now
  `sink("tensor")` (likewise `Sink.tensor(...)`, `SinkFormat.TENSOR`); rows,
  bytes and strides are unchanged, and the column is always typed
  `polars_cv.ndarray`.
- Polars struct operations do not see through the `polars_cv.ndarray` tag; go
  through `pl.col("x").ext.storage().struct.field("shape")` or
  `df.with_columns(pl.col("x").ext.storage()).unnest("x")`.
- A tensor dtype check is `isinstance(dtype, polars_cv.NdArrayType)`. To
  concatenate with a plain-struct column from an earlier release, tag the old
  one with `pl.col("x").ext.to(polars_cv.NdArrayType())`.
- `show_images(format="numpy")` is now `format="tensor"`.
- The `(format, {params})` tuple in a multi-output `sink()` is removed:
  `{"thumb": ("jpeg", {"quality": 85})}` is `{"thumb": Sink.jpeg(quality=85)}`
  (the tuple raises `TypeError`).
- A misspelled `sink()` keyword is now a `TypeError` rather than a `ValueError`.
- `PipelineGraph.set_output(node, "jpeg", quality=85)` is
  `set_output(node, Sink.jpeg(quality=85))`; `set_multi_output` takes a dict of
  format names or `Sink`s and no longer reads shared keywords.
- `image_info()` returns a fifth field, `orientation: Int64`; code that unnests
  it into a fixed set of columns gains a column.
- The metrics are no longer re-exported from `polars_cv`:
  `from polars_cv import froc_auc` becomes `from polars_cv.metrics import froc_auc`.
- `Pipeline.thumbnail(max_size)` is removed; use
  `.source("image_bytes", decode_max_size=64)` instead of
  `.source("image_bytes").thumbnail(64)`.
- `extract_contours` accepts only a single-channel `u8` mask and raises
  otherwise (it previously traced garbage from other dtypes or channel counts);
  apply `.threshold()` or `.cast("u8")`, `.grayscale()` or `.channel_select()`
  first.
- `CloudOptions.token_command` is now a non-empty `list[str]` run without a
  shell; a string raises `TypeError`. Write
  `token_command=["my-broker", "get-token"]`, or `["sh", "-c", "a | b"]` if you
  need shell syntax.
- Path reads are default-deny: `source("file_path")`, an `auto` source over a
  `String` column, `.cv.read_bytes()` and the path-column metadata methods fail
  without `allowed_roots`, remote URLs included. Name the permitted directories
  and URI prefixes, or pass `allowed_roots=["*"]` for paths you trust.
- `polars_cv.lazy.region_stats` is now `polars_cv.masks.region_stats` (still
  importable from `polars_cv`).

### Security

- `zstd-sys` 2.0.16 to 2.1.2, for RUSTSEC-2026-0410 (wasm32 only; not a target
  of polars-cv).
- RUSTSEC-2026-0391, -0398, -0403 (`zstd`) and RUSTSEC-2026-0441 (`pin-utils`,
  unmaintained) are ignored in `deny.toml`: none is reachable from the plugin,
  and the fixes await a polars bump.
- RUSTSEC-2026-0105 (`core2`) and RUSTSEC-2024-0436 (`paste`) no longer apply;
  they came in only through `image`'s AVIF encoder, which is no longer built.

### Added

- `.bbox` accessors `area()`, `to_coords(format)`, `to_contour()`,
  `translate(dx, dy)`, `scale(sx, sy)`, `normalize`/`to_absolute(width, height)`,
  `clip(width, height)`, `pad(amount)` and `expand(factor)`, each taking a
  literal or per-row expression.
- `.bbox.nms(scores, iou_threshold=0.5)` returns the kept positions of each
  row's set, highest score first; it matches OpenCV's `NMSBoxes`.
- `Pipeline.crop_box(box)` crops each row to the smallest whole-pixel window
  containing its box, clipped to the image.
- `polars_cv.geometry.BoxFormat` is now an enum; `"xyxy"`, `"xywh"` and
  `"cxcywh"` are accepted as before.
- `expr.explain(format)` prints the graph `sink(format)` runs, with each op's
  planned domain, dtype and shape; `optimized=False` prints it as written.
  `expr.show_graph(format)` draws the same description.
- `expr.graph(format)` returns the graph, and `output_dtype()` and
  `current_domain()` give a node's planned state.
- `source("tensor")` reads back the tensor sink's output (and `source()` auto
  routes to it), at each row's own dtype, shape and strides; a row whose view
  lies outside its data is a row error.
- `source(orientation="exif")` turns JPEG, PNG, WebP and TIFF images by their
  EXIF tag as `PIL.ImageOps.exif_transpose` does; the default `"stored"` is
  unchanged.
- `image_info().orientation` reports the EXIF tag (1 to 8, null when absent).
- `sink()` has typed keywords `shape`, `quality`, `dtype`, `compact`, `names`
  and `where`, and `polars_cv.Sink` builds a format with its parameters
  (`Sink.jpeg(quality=85)`, `Sink.array(shape=[224, 224, 3])`), accepted
  wherever a format name is.
- New guides: "Training-Time Augmentation", and "Choosing a sink" in the
  pipelines guide; the quickstart gains a troubleshooting section.

### Changed

- Error messages are plainer: a failing row prints the codec's message, a
  refused sink name no longer ends in a JSON position, and a failing pipeline
  is named (`Pipeline execution failed (column 'img' to the png sink): ...`).
  Match on the part that stayed.
- `.cv.pipe()` refuses a pipeline with no source, naming `.source()`.
- A null in a parameter column names both remedies: `fill_null`, or
  `Pipeline.on_null_param("null")`.
- `IntOrExpr`, `FloatOrExpr` and `BoolOrExpr` are exported from `polars_cv`.
- The extension and wheel are about 6% smaller: the AVIF encoder and TGA
  decoder are no longer built, and nothing that decoded before stops decoding.
- Open work is tracked in GitHub issues; `ISSUES.md` is removed.

### Fixed

- A box of negative size (for example `xywh` data passed as
  `box_format="xyxy"`) is refused, naming the format to check, instead of
  silently scoring as a 0-IoU false positive; zero-size boxes are still read.
- Constants of the scalar ops (`scale`, `clamp*`, `add_constant`,
  `adjust_contrast`, `adjust_gamma`, `convolve2d` coefficients, pad/crop fill
  values) were rounded to f32 before f64 arithmetic: `scale(1/3)` on `Float64`
  gave `0.3333333432674408`. Output changes for 32/64-bit data only.
- A JPEG that `decode_max_size` would not shrink went through the slower
  scaled decoder; it now takes the full decoder, so output changes (to match
  no option) for a JPEG smaller than twice its `decode_max_size`.
- Three error messages, and the `tensor` source's refusal, named the removed
  `numpy` sink; they now name `tensor`.

## [0.41.0] — 2026-10-23

Reductions, filters and windows for image analytics: one `reduce` takes any
statistic or percentile, and new mask, region and block reducers, rank filters,
`crop_clip`/`crop_pad` and `polars_cv.this` size expressions are added.
Upgrading from 0.40: see the [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.41/).

### Breaking changes

- `normalize(method="preset", mean=, std=)` is now `normalize_preset(mean=, std=)`;
  results are unchanged, and `normalize`'s methods are `"minmax"` and `"zscore"`.
- `LabelReduction` is now `Statistic`; the spellings `"max"`, `"mean"` and
  `"sum"` are unchanged. A percentile is `"pNN"` (`"p90"`, `"p62.5"`), refused
  outside [0, 100].
- `.cv.width()`, `.cv.height()`, `.cv.channels()` and `.cv.image_info()`'s size
  fields are `Int64`, not `UInt32`, so arithmetic like `width() - 32` no longer
  wraps; code that cast the result or compared schemas needs `Int64`.

### Added

- `polars_cv.this`: parameters computed per row from the buffer an op reads
  (`this.height`, `.width`, `.channels`, `.ndim`, combined with `+ - * / //`),
  e.g. `crop(top=0.2 * this.height, height=0.6 * this.height)`.
- `reduce(reduction, axis=None)` reduces every value or an axis by any
  statistic (`"p90"`, `"median"`, `"count"`, ...).
- `img.reduce_masked(mask, reduction=, axis=)` reduces over the pixels a mask
  selects.
- `masked_select(mask)` (lazy-only) returns an image's values under a mask as a
  1-D buffer, like NumPy's `image[mask > 0].ravel()`.
- `block_reduce(height=, width=, reduction=)` pools fixed blocks with any
  reduction.
- `resize(filter="area")` averages each output pixel's footprint (OpenCV's
  `INTER_AREA`).
- `img.resize_like(other, filter=)` resizes an image onto another node's grid.
- `rank_filter(height, width, q=50, border=)`, `median_filter(size)` and
  `img.background_divide(sigma)`.
- `normalize_percentile(low=, high=)`, `normalize_range(low=, high=)` and
  `normalize_percentile_masked(mask, low=, high=)` (lazy-only) window an image
  to [0, 1] by percentile, explicit range or masked percentile.
- `statistics(include=[..., "median", "p90", "count"], mask=)`, and
  `polars_cv.region_stats(regions, {name: (map, statistic)})` reduces several
  maps over each contour region.
- `label_reduce(reduction=)` takes every reduction (`"min"`, `"std"`,
  `"median"`, `"pNN"`, `"count"`), on the pipeline op and `.contour` accessor.
- `flip(axes, when=)`, `flip_h(when=)` and `flip_v(when=)` flip only rows a
  per-row condition selects.
- `sink("long", names=[...], where=)` writes a buffer as one struct per pixel.
- `grayscale(keepdims=False)` gives `[H, W]` from any input.
- `crop_clip(...)` and `crop_pad(..., value=0)` handle windows reaching past
  the image's edge; `crop` still refuses them.
- `polars_cv.check_cloud_access(path, cloud_options=None)` exercises the
  credential a read would use before any row is read, raising
  `polars_cv.CloudCredentialError` on failure.

### Changed

- Header-only metadata on a remote path (`.cv.image_info()`, `width()`, ...)
  reads the first 64 KiB instead of the whole object.
- `show_images` accepts an expression, so one field of a struct column can be
  shown.
- A failing row is named within its batch (`(row 11 of 12 in this batch)`),
  for the pipeline and every geometry accessor; a bad per-row parameter names
  its op and parameter.

### Fixed

- `reduce_percentile(q)` with `q` outside [0, 100] returned the minimum or
  maximum; it is now refused.
- `label_reduce` and `region_stats` over a region with no pixels gave 0.0; they
  now give NaN (0 for `sum` and `count`).
- The contour matcher dropped a zero-scored detection but kept one scored NaN
  (a region with no pixels), matching it as the most confident; both are now
  dropped.
- A federated gcloud ADC failed every `gs://` read when no `cloud_options` was
  passed.
- A failing `token_command` or delegated ADC now names its credential and the
  command's stderr.
- `stroke(join="miter")` painted pixels in a pocket its ring pinches off.
- The `LazyPipelineExpr` methods for `apply_mask` and `channel_merge` sent
  their operand as `other`, whatever the op named it.

### Performance

- `rank_filter` on u8 is about 6× faster for a 15×15 median.
- `region_stats` is about 1.9× faster for eight maps.
- `statistics(mask=)` gathers the masked pixels once rather than per statistic
  (3.2× faster for the default four, measured on a debug build).

## [0.40.0] — 2026-10-19

One bit order for every hash, hash comparisons over stored hashes, and a
version-2 graph wire format.
Upgrading from 0.39: see the
[migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.40/).

### Breaking changes

- `perceptual_hash(algorithm="average" | "difference" | "blockhash")` now packs
  bits most significant first, like `"perceptual"` and imagehash; recompute
  stored hashes or reverse the bits of each byte. Hamming distances between
  two hashes are unchanged.
- `hash_similarity()` no longer takes `hash_bits=`; the score is out of the
  hashes' own length, and passing the argument raises `TypeError`.
- `hamming_distance()` returns a `UInt32` bit count, not `Float64`.
- The graph wire format is version 2, and graphs serialized by 0.39 are
  refused with a message to rebuild them.
- `PipelineGraph.add_node(..., input=)` replaces `upstream=`.

### Added

- `hamming_distance()` and `hash_similarity()` accept stored hash columns
  (`sink("list")`/`sink("array")`, as an expression or name), so near-duplicate
  search is a cross join: `hamming_distance("hash", "hash_b")`. Hashes of
  different lengths are refused.
- A [parity report](https://heshamdar.github.io/polars-cv/user-guide/parity/)
  lists every reference comparison, its tolerance and its outcome.
- PyPI metadata: project URLs, keywords and classifiers.

### Changed

- `LazyPipelineExpr.pipe()` with a sourced pipeline on a continuation is now
  refused where it is written, not later at sink.
- A step that reads a node the graph does not have now raises one error naming
  both nodes, when the graph is parsed.

### Fixed

- `extract_shape` and `label_reduce` skipped validation of their operands
  against the row's shapes and dtypes.
- 115 reference tests never ran polars-cv; each now compares its output with
  the reference.

## [0.39.0] — 2026-10-18

Binary ops no longer broadcast, `channel_select` keeps the channel axis, and
`apply_mask` reads a mask on its own dtype's scale. Contours gain set
operations, shaping, stroking and open lines in pipelines, and CSE now shares
all equal work. Upgrading from 0.38: see the
[migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.39/).

### Breaking changes

- `channel_select(i)` returns `[H, W, 1]`, not `[H, W]`; pass
  `keepdims=False` for the 2-D result.
- Binary ops (`add`, `subtract`, `multiply`, `divide`, `blend`, `maximum`,
  `minimum`, bitwise ops) no longer broadcast: operands of different shapes
  are refused, at build where sizes are known. Use `squeeze`/`expand_dims` to
  change a rank.
- `apply_mask` reads the mask on its own scale (a u8 mask weighs `m / 255`, a
  float mask `m` in `[0, 1]`), whatever the image dtype; other mask dtypes are
  refused. A u8 image under a u8 mask is unchanged.

### Added

- `squeeze(axis)` and `expand_dims(axis)` remove or insert a size-1 axis as
  zero-copy views; negative axes count from the end.
- Contour set operations: `union`, `intersection`, `difference` and
  `symmetric_difference`, as pipeline ops and `.contour` accessors.
- Contour shaping and drawing: `offset(distance, join="round")`,
  `union_all()`, `smooth(iterations=1)`, `densify(max_segment_length)` and
  `stroke(width=1.0, cap="round", join="round")`, with new `LineJoin` and
  `LineCap` enums. `rasterize` still fills only; draw lines with
  `stroke(...).rasterize(...)`.
- Open contours (lines) work in pipelines via `source("contour")`; new
  `boundary()` gives a region's rings as lines and `close()` joins a line's
  ends into a region.
- `rasterize(..., supersample=n)` anti-aliases masks with `n` from 1 to 16;
  the default of 1 is unchanged.
- `apply_lut(table, out_dtype="u8")` maps u8/u16 values through a lookup table,
  including palettes that turn one channel into `K`.
- `distance_transform(signed=False)` gives the exact Euclidean distance
  transform of a single-channel mask as f32.
- `coordinate_grid(normalized=False)` gives an f32 `[H, W, 2]` grid of pixel
  centres.
- `footprint="disk"` on `erode`, `dilate`, `morphology_gradient`,
  `morphology_open` and `morphology_close` for round morphology.
- A multi-output `sink()` dict value may be `(format, {params})`, so each
  output takes its own parameters.
- New guide: [Drawing with polars-cv](https://heshamdar.github.io/polars-cv/user-guide/drawing/).

### Changed

- `channel_merge` joins channel axes: operands may be `[H, W]` or `[H, W, C]`,
  so `photo.channel_merge(alpha)` gives an RGBA image.
- CSE now computes each distinct value once, fuses linear runs of ops and
  repeats until the graph settles; outputs are byte-identical, and
  `opt_flags=False` shows the graph as written.
- The README is shorter and leads with one decode per image, and the docs gain
  a Parameters & Errors page and a migration section.

### Fixed

- `apply_mask` refused a `[H, W, 1]` mask over a `[H, W]` image.
- `channel_merge`, `apply_mask` and `merge_pipe` ran the left operand's ops
  twice.
- An alias given to two different nodes is now refused, and a node can carry
  several aliases.
- `show_graph()` hid a root node's ops and named nodes `None`.
- `stroke()` of a ring narrower than the stroke left a hole in the band.
- `stroke()` of a ring with no points panicked.
- A line returning to its first point was stroked with caps instead of as a
  loop.
- `smooth` and `densify` could abort the process on huge outputs; a row past
  2^22 points now fails with an error.
- Contour results with an inferred layout (`holes: List(Null)`) panicked in
  set-level functions, and `boundary()` lines read back as regions; every
  contour result is now `{exterior, holes, is_closed}`.
- A contour column of the wrong layout was turned into nulls by
  `on_error("null")`; it now fails the query.
- `rasterize(shape=)` onto another node's canvas skipped the op's check, so a
  per-row `supersample` of 0 panicked.
- `.contour.offset(inf)`, `densify(0)`, `stroke(width=0)` and `largest(k=0)`
  failed only at collect; they are now refused at build.
- Op-backed `.contour` accessors now give the same results, nulls and error
  messages as the pipeline op; an empty contour's `bounding_box` is NaN on
  both.
- Error messages named the op twice (`blur: blur: sigma ...`); a build-time
  refusal now begins with the op's Python name and a row error with its engine
  step's name.
- A literal `histogram` `range=` that is non-finite or inverted is refused at
  build.
- A non-finite float literal such as `simplify(float("nan"))` failed on the
  wire; `"nan"`, `"inf"` and `"-inf"` now cross it.
- `clamp` panicked on a NaN or crossed bound; it is now refused.
- The thread count a call uses is measured from its first rows, so calls of
  expensive rows reach the whole pool sooner. A call of only a few expensive
  rows is slower (8 rows of ~18 ms on 4 threads took ~58 ms against ~39 ms).
- The sdist is no longer rejected by PyPI over missing license files; the
  build now requires `maturin>=1.9.3`. Install 0.39.0 rather than 0.38.0,
  whose PyPI wheels were built from 0.39 code.

### Performance

- `footprint="disk"` morphology is linear in the radius; a 31-px disk is
  2.2–2.4x faster.
- `apply_lut` copies less per row.

## [0.38.0] — 2026-10-15

Standard pHash and a switch for the plugin's row parallelism. Stored hashes
must be recomputed. Upgrading from 0.37: see the
[migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.38/).

### Breaking changes

- `perceptual_hash()` with the default `algorithm="perceptual"` now computes
  standard pHash like `imagehash.phash`, within 0-2 bits of it; recompute
  stored hashes. `average`, `difference` and `blockhash` are unchanged.

### Security

- `rustls` 0.23.45 to 0.23.46, for RUSTSEC-2026-0346 to -0349; it is the TLS
  stack of remote `file_path` reads.

### Added

- `polars_cv.set_row_parallelism(enabled)` and `row_parallelism()` turn the
  plugin's own row splitting off or on for the process.
- License texts (`LICENSE-MIT`, `LICENSE-APACHE`) now ship with the package.
- A rewritten README with cross-library benchmark results.

### Fixed

- `blur(sigma=0)` returned an all-zero image; it is now no blur. A negative,
  NaN or infinite sigma is refused (a negative one used to be treated as no
  blur).
- On many-core hosts, calls of cheap rows woke the whole plugin pool and could
  run far slower than at 16 threads (19.5 s against 0.4 s on 72 threads).

## [0.37.0] — 2026-10-11

Detection metrics now stream, treat a null group key as a group of its own, and
take `carry=`. A null group's FROC/LROC curve is no longer empty, and
`match_detections(images=...)` no longer checks that `images` has one row per
image. Upgrading from 0.36: see the
[migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.37/).

### Added

- `carry=` on every matcher and on `match_detections` copies columns (or
  expressions) of the `image_metadata` frame into the result, one value per
  (image, class) row, usable as grouping keys.

### Changed

- A null group key is a group of its own throughout `polars_cv.metrics`, so a
  null group's FROC/LROC curve is no longer empty. In `match_detections`, a
  null `image_id` is one image.
- `grouped_scan` refuses an ungrouped reverse `CumMax`; group by a constant
  column instead.
- `match_detections(images=..., weight=/group=)` no longer checks one row per
  image in `images`; a repeated `image_id` repeats that image's objects.

### Fixed

- Several metrics steps (FROC/LROC, partial AUC, sensitivity-at, interpolation,
  the weighted bootstrap and 11/101-point AP) fell back to Polars' in-memory
  engine under Polars 2 and now stream. `group_objects`' per-(image, class)
  lists remain in-memory.

## [0.36.0] — 2026-10-11

Performance, plus `compact=` on tensor sinks. Crops and vertical flips are read
where they lie by more ops, and a tensor sink copies a view at most once. Upgrading
from 0.35: see the
[migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.36/).

### Added

- `sink("numpy"|"ndarray"|"torch", compact=True)` writes only each row's
  elements, so Parquet and IPC do not persist a view's whole backing storage.
  Other sinks refuse `compact`.

### Changed

- `scripts/verify.sh` runs the slow lane only with `--slow`; `--fast` is
  removed.

### Fixed

- `equalize_histogram` and `perceptual_hash` of a u64 or i64 image wrapped
  into u8 instead of saturating (an i64 300 became 44).

### Performance

- `blur`, `erode`, `dilate`, `morphology_gradient`, `pad`, `pad_to_size`,
  `equalize_histogram` and `channel_swap` read a crop or vertical flip without
  a copy.
- `convolve2d`, `convert_color`, convolution resizes of f64 and 32/64-bit
  integers, and `equalize_histogram` of non-u8 dtypes read any view without
  packing it first; a vertical-flip f32 `convolve2d` goes from 27 to 7 ms.
- Binary ops and `apply_mask` read operands in any layout without packing
  them; a 1024×1024×3 u8 `add` on a crop goes from 0.90 to 0.24 ms.
- A tensor sink that has to copy a view copies it once instead of twice.

## [0.35.0] — 2026-10-09

Fixes and performance for threading, remote reads and bootstrap plans. Nearest
resize at exact pixel-centre ties and `DetectionTable.collect()` can change
results. Upgrading from 0.34: see the
[migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.35/).

### Changed

- `DetectionTable.collect(engine=)` defaults to `"auto"` (was `"streaming"`),
  and the metrics use Polars' default engine, so
  `pl.Config.set_engine_affinity` is respected. Results are unchanged.
- New guide material: sizing and bounding a streaming call, and feeding a
  training loop with `LazyFrame.collect_batches` and a fixed-shape `array`
  sink.

### Fixed

- Nearest-neighbour resize took the wrong source pixel at exact pixel-centre
  ties (2 → 21 rows, 2 → 7); output changes only at those rows and columns.
- A streaming run could make the next run of the same pipeline
  single-threaded (3.9x slower on 4 cores); a call now uses idle pool threads
  within a budget of `POLARS_MAX_THREADS`.
- Remote reads started a second async runtime beside Polars'; they now use
  Polars' runtime, sized by `POLARS_ASYNC_THREAD_COUNT`.
- A call held every remote file it read before decoding any; each row now
  fetches its path and the next rows' up to `POLARS_CONCURRENCY_BUDGET`
  ahead. `POLARS_CONCURRENCY_BUDGET` bounds polars-cv's requests separately
  from Polars' scans.
- Bootstrap intervals ran two steps on Polars' in-memory engine; every
  bootstrap plan now streams, with bounds unchanged.

### Performance

- Nearest-neighbour resize: 2× upscale is 1.6–1.9× faster.
- Remote `file_path` reads are 51–61% faster at 1 thread.

## [0.34.0] — 2026-10-07

Polars 2.0 support, and equal pipelines in one query now run once.
Upgrading from 0.33: see the [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.34/).

### Breaking changes

- Requires `polars>=2.0.0,<3.0` (was `>=1.44.2,<2.0`).
- Seeded bootstrap intervals differ from 0.33 for the same seed, because
  polars 2.0 changed `Expr.hash` values; they stay reproducible for a given
  seed and polars version.
- A Parquet file with a tagged column, read without importing polars-cv, now
  loads as `pl.Extension("polars_cv.point", ...)` instead of a plain struct;
  use `.ext.storage()` or set `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage`.

### Added

- Two equal pipeline expressions in one query (built separately) now run as one
  plugin call, and the compiled-graph cache is reused across rebuilt
  expressions.
- polars-cv's outputs and memory take part in Polars 2.0's out-of-core
  spilling.

### Changed

- Metrics `.over()` windows now stream natively on Polars 2.0 instead of
  falling back to the in-memory engine.

### Fixed

- The extension segfaulted on import on macOS; no released wheel was affected.
- `froc_curve_lazy(thresholds=[1, ...])` raised under polars 2.0; thresholds are
  now passed as Float64.

### Performance

- `.point` accessors are 12–43% faster with about 25–30% less peak memory under
  polars 2.0.

## [0.33.0] — 2026-10-06

Weighted bootstrap intervals are corrected: they were too narrow when weights
were estimated from the sample, and the new `weight_scheme=` says what the
weights are. Requires Python >= 3.11.
Upgrading from 0.32: see the [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.33/).

### Breaking changes

- Python 3.11 or newer is required (was 3.10); wheels are tagged `cp311-abi3`.
- Weighted bootstrap intervals now default to `weight_scheme="reestimate"`; pass
  `weight_scheme="stratified"` for 0.32's draw. Unweighted intervals and point
  estimates are unchanged.

### Added

- `weight_scheme=` on `bootstrap_ci`, `froc_auc_ci_lazy`, `lroc_auc_ci_lazy`,
  `average_precision_ci_lazy` and `Report.ci`: `"reestimate"` (default),
  `"stratified"` or `"fixed"` (known or continuous weights, which now get an
  interval instead of null bounds); an unknown scheme raises.
- `Report.ci` takes `weight_rtol=`; `"fixed"` raises if given `strata` or
  `weight_rtol`.

### Fixed

- Weighted bootstrap intervals were too narrow: standard errors were 7–17% low
  and 95% intervals covered 0.88–0.93.
- A bootstrap replicate that drew no positive (or, for Mann-Whitney, no
  negative) image was scored with a meaningless value; it now nulls its group's
  bounds.
- A weight-0 image no longer nulls its group's bounds.

## [0.32.0] — 2026-10-03

Detection evaluation in one call (`evaluate_detections`, `evaluate_heatmaps`,
`evaluate_segmentation`), weighted precision-recall metrics, and bounded
bootstrap memory. Several metric values were wrong and are now computed as
documented: multi-class matching, AP over tied scores, and multi-class or
grouped bootstrap draws. Requires polars >= 1.44.2.
Upgrading from 0.31: see the [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.32/).

### Breaking changes

- Requires `polars>=1.44.2` (was `>=1.43.2`).
- Precision-recall metrics (`precision_recall_curve`, `average_precision`,
  `mean_average_precision`, `precision_at_threshold`, `recall_at_threshold`,
  `f1_at_threshold`, `confusion_at_threshold`) now use `image_metadata.weight`;
  unit weights give the same values as before.
- `ConfusionResult.precision`/`recall`/`f1` now read the weighted values.
- All-points AP no longer over-counts score ties: a tied block mixing TPs and
  FPs now scores lower (TP@0.9 then {TP, FP, FP}@0.5 against 2 GTs was 0.875,
  now 0.75); untied results are unchanged.
- The 11-point recall grid is `i * 0.1` (was `i / 10`), so a recall landing
  exactly on 0.3, 0.6 or 0.7 compares as in the VOC devkit and COCO.
- Bootstrap intervals now stratify on weight cells; a group with a weight cell
  holding a single unit reports null `ci_lower`/`ci_upper`, and
  `average_precision_ci_lazy` is now weighted.
- `DetectionTable.to_per_image()` returns `max_score`, `top_is_tp` and
  `best_tp_score`, and no `detections` list.
- A failing mask-instance region check in `match_detections` is now a
  `polars.exceptions.ComputeError` (was `ValueError`).

### Added

- One-call evaluation: `evaluate_detections(preds, gts, box_format=)` (COCO-style
  by default, matching pycocotools to 1e-12 on mAP, AP50, AP75 and AR@100),
  `evaluate_heatmaps(df, heatmap=, gt=)` (FROC and CPM), and
  `evaluate_segmentation(df, pred=, target=)`.
- `DetectionReport` (from the two detection evaluators) has `.summary`,
  `.per_class`, `.per_threshold`, `.ci(metric, by=)`, `.matches(t)`,
  `.pr_curve`, `.froc`, `.confusion` and `.table`.
- `segmentation_measures(...)` gives Dice, IoU, ASSD, HD, HD95 and directed mean
  distances per row, with `frame=`, `spacing=` and an explicit empty-mask
  policy; `SegmentationReport.ci` bootstraps it.
- `match_detections(predictions, ground_truth, ...)` builds a `DetectionTable`
  from long object tables of boxes, polygons or instance masks; `group_objects`
  reshapes them and applies `max_detections`.
- `bbox_from_coords(coords, format=)` reads `"xyxy"`, `"xywh"` or `"cxcywh"`
  boxes into `BBOX_SCHEMA`.
- `.contour.single(label=)` returns each row's one contour; a set of any other
  size fails the query naming the row's `label`.
- `.contour.set_boundary_distances(other, sample_step=, frame=)` measures the
  boundary distance between two contour sets, as MONAI does.
- IoU sweeps: `BBoxMatcher(iou_threshold=[...])` and
  `ContourMatcher(iou_threshold=[...])` match once per threshold, as COCO does,
  and `DetectionTable.iou_thresholds` lists them. Evaluations that would pool
  thresholds are refused.
- `Statistic` objects (`AP`, `Recall`, `PrecisionAt`, `RecallAt`, `F1At`,
  `FROCSensitivity`, `CPM`, `FROCAUC`, `LROCAUC`, `LROCSensitivity`,
  `MeanOver`, `mean_ap()`) expose every detection metric as a grouped lazy
  reduction via `.by_group(table, keys)` and `.value(table)`.
- `bootstrap_ci(table, statistic, ...)` is one interval engine for any
  `Statistic`; an mAP interval resamples images once across all classes.
- `interpolation="101_point"` on `average_precision`, `mean_average_precision`
  and `PrecisionRecallResult.auc`.
- `group_by=` on `froc_sensitivity_at_fp`, `froc_summary_table` and
  `lroc_sensitivity_at_fpf`.
- Weighted PR metrics take `weight_agg=`; the PR curve gains
  `cum_weighted_tp`/`cum_weighted_fp`, `PrecisionRecallResult` gains
  `weighted_gts`, and `ConfusionResult` gains `weighted_tp`/`weighted_fp`/
  `weighted_fn`.
- Bootstrap intervals take `strata=` and `weight_rtol=` to control the weight
  cells.

### Changed

- Bootstrap replicates are evaluated in batches of 50, so peak memory follows
  the batch rather than `n_bootstrap`; bounds are bit for bit unchanged and
  there is no new parameter.
- Metrics run on Polars' streaming engine wherever it can; AP, mAP and other
  reductions that must be reproducible are now exact and no longer vary between
  runs or thread counts.

### Fixed

- Multi-class `BBoxMatcher` and `ContourMatcher` input duplicated every detection
  once per class, corrupting per-class AP, mAP and FROC; images with no
  predictions no longer leave a null-score row in a `BBoxMatcher` table.
- `DetectionTable.to_per_image()`'s `top_is_tp` depended on row order when two
  detections tied for the top score; the tie now goes to the lowest `det_idx`.
- `.contour.boundary_distances(frame=)` spiked next to the frame (about 30 px
  instead of 1 px), inflating GT-to-prediction mean, `hd` and `hd95`; distances
  are now continuous at the edge.
- Multi-class bootstrap replicates over-drew images that were positive for one
  class and negative for another; single-class tables are unchanged.
- `match_detections(images=...)` repeated an image listed twice; it now fails
  the query on a repeated `image_id`.
- Grouped bootstrap draws (for example `group_by="class_id"`) brought other
  groups' rows into each replicate.

### Performance

- Bootstrap AP with 2000 images, 50k detections and 1000 replicates: peak
  memory 4257 MB to 614 MB, and 20.4 s to 14.8 s.

## [0.31.0] — 2026-10-03

Open polylines become a first-class contour (`is_closed=False` is read), with
boundary distances, coverage matching for line-shaped ground truth, closing a
line along the image frame, and coordinate-list constructors. FROC/LROC curves
are read past their end by one policy, null by default. Contour labelling and
`label_reduce` region scans are orders of magnitude faster.
Upgrading from 0.30: see the [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.31/).

### Breaking changes

- `froc_auc(fp_range=...)` now gives a null AUC when `fp_range` reaches past the
  curve; pass `extrapolate="flat"` for the previous value. LROC values are
  unchanged.
- `extract_contours(mode="tree")` is removed; it ran as `mode="all"`, so use
  `mode="all"`.
- Open contours (`is_closed=False`) are now measured as polylines instead of
  rings, and region functions (`area`, `centroid`, `iou`, `contains_point`,
  `correspond`, `label_reduce`, ...) refuse them, naming the row.
- `PreMatchedAdapter` now raises on keys and columns it used to drop silently: a
  `class_id` column on detections or `image_meta` requires `class_col`, and a
  named but missing `iou_col`, `det_idx_col`, `n_gts_col`, `weight_col`,
  `gt_label_col` or `group_col` raises.
- Without `iou_col`, a `PreMatchedAdapter` table's `iou` is null (was 0.0), and
  `DetectionTable.at_iou_threshold` (so `mean_average_precision`) raises on it.
- A bootstrap replicate whose FROC/LROC curve stops short of the window now
  nulls the bounds instead of scoring 0.0.
- `ContourMatcher` with `score_col` and a heatmap prediction is now refused
  instead of ignoring `score_col`.

### Security

- `allowed_roots` could be escaped with a `file://` URL naming a host, making
  any local file readable through `.cv.read_bytes()`, the `file_path` source
  and header-metadata reads; such URLs are now refused, and `file://` URLs are
  percent-decoded.

### Added

- `extrapolate="none"|"flat"` on `froc_auc`, `froc_sensitivity_at_fp`,
  `froc_summary_table`, `lroc_auc`, `lroc_sensitivity_at_fpf`,
  `froc_auc_ci_lazy` and `lroc_auc_ci_lazy`.
- `.contour.on_error("raise" | "null")` (also on `.point` and `.bbox`) nulls
  rows a function refuses instead of failing the query; errors about the
  operand column still raise.
- `.contour.boundary_distances(other, sample_step=None, frame=...)` returns
  `Struct{mean_a_to_b, mean_b_to_a, assd, hd, hd95}`; `frame` drops boundary
  traced by the image edge rather than an annotation.
- `.contour.correspond_by_coverage(other, tolerance, threshold=0.5, order=None,
  sample_step=None)` pairs contours by boundary coverage, for line-shaped
  targets.
- `CORRESPONDENCE_SCHEMA` gains `duplicate: List(Boolean)`, flagging repeated
  hits on an already-claimed target.
- `.contour.close_along_border(width, height, arc="shortest", max_snap=2.0)`
  closes an open line whose ends lie on the image frame.
- `.contour.largest(k)` and the pipeline op `largest(k=1)` keep the `k` largest
  contours by area.
- `extract_contours(min_area_fraction=)` filters contours by area as a
  fraction of the image size.
- `point_from_coords`, `contour_from_coords` and `contour_set_from_coords` in
  `polars_cv.geometry` build geometry from `[x, y]` or `[y, x]` pairs;
  `.point.to_coords(order)` and `.contour.to_coords(order)` are the inverse.
- `.point` accessors take point sets (`POINT_SET_SCHEMA`), returning one value
  per point as a `List`.
- `ContourMatcher(score_reduction=, score_region_mode=)` pass through to
  `label_reduce` for scoring detections.
- `ContourMatcher` parameters `iou_threshold`, `extraction_threshold`,
  `min_contour_area`, `coverage_tolerance` and the new
  `min_contour_area_fraction` accept Polars expressions per row.
- `ContourMatcher` accepts contour or contour-set predictions in `pred_col`
  scored by `score_col`, and line-shaped ground truth via `match_by="coverage"`
  with `coverage_tolerance`; `duplicates="ignore"` drops repeated hits.
- `DetectionTable.filter_images(ids | predicate)` keeps the named images and
  their detections.
- `.cv.width()`, `.height()`, `.channels()` and `.image_dtype()` take a `String`
  column of paths, reading only each file's header; `.cv.image_info()` returns
  `Struct{width, height, channels, dtype}`.
- `froc_operating_range(table, group_by=...)` reports each curve's
  `max_fp_per_image` and `max_sensitivity`.

### Fixed

- `ContourMatcher` now fails the query on a null score for contour predictions,
  which previously made GT count as missed and the detection as a false
  positive.
- `close_along_border` refuses a NaN or negative `max_snap` and a non-finite
  frame size.
- `boundary_distances` and `correspond_by_coverage` with `sample_step` looped
  forever on edges between vertices about 1e308 apart.
- `point_from_coords` and `contour_from_coords` refuse a non-finite coordinate,
  naming the row.
- Every `.point`/`.contour`/`.bbox` accessor failed with "geometry row 1 out of
  bounds" when an operand had one row; one-row operands now broadcast.
- `contours.pipe(Pipeline().simplify(2.0))` raised "expects contour input but
  pipeline is currently in buffer domain"; a sourceless `Pipeline()` now takes
  the domain of its first op.
- The streaming guide wrongly claimed the plugin stays within
  `POLARS_MAX_THREADS`; up to about twice the setting can be runnable.

### Performance

- `extract_contours` labels runs instead of pixels: finding the borders of a
  3328 x 2560 blob went from about 140 ms to under 1 ms.
- `label_reduce` with `region_mode="interior"`/`"boundary"` scans by scanline: a
  4,096-vertex disc of radius 1000 went from about 11 s to about 23 ms.

## [0.30.0] — 2026-10-02

Every op now computes and stores in the dtype it declares, following NumPy and
OpenCV where they define an answer: colour ranges per dtype, NumPy promotion,
IEEE division, NaN propagation, NumPy histogram binning and pixel-edge
contours. Kernels are substantially faster across the board.
Upgrading from 0.29: see the [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.30/).

### Breaking changes

- Colour conversions use each dtype's value range: floats are in [0, 1] (Lab to
  RGB now gives `f32` in [0, 1], so scale 0-255 float images by 1/255 first),
  and u16/u32/u64 span 0..MAX for YCbCr, HSV and Lab.
- `to_hsv`, `to_lab`, `to_ycbcr` and `convert_color` to or from those spaces
  refuse signed integer dtypes; u8 values move by at most one unit.
- Aspect-preserving resizes (`resize_to_height`, `resize_to_width`,
  `resize_max`, `resize_min`, `resize_scale`, `letterbox`) derive sizes by exact
  integer arithmetic, rounded half up and never below 1; sizes change only for
  inputs that previously landed just below .5 or at 0.
- `relu`, `clamp_min` and `clamp_max` propagate NaN, as NumPy and PyTorch do;
  use `fill_nan` first where a bound was wanted.
- `reduce_max`/`reduce_min` return NaN if any input is NaN, `reduce_argmax`/
  `reduce_argmin` return the first NaN's index, and `reduce_percentile` of NaN
  input is NaN.
- Binary `maximum`/`minimum`, `dilate`/`erode`, `morphology_gradient`, MinMax
  `normalize` and HSV conversion now propagate a NaN from either side.
- Non-finite geometry is refused instead of warped: NaN or infinite
  `rotate_and_scale`, `rotate` and `warp_affine` parameters, and a NaN,
  infinite, zero or negative `resize_scale` factor.
- `histogram` follows NumPy for NaN: a NaN pixel is in no bin and its
  `quantized` index is one past the last bin, and a non-finite auto or supplied
  `range` raises.
- `histogram` drops values outside its edges or `range` instead of clamping them
  into the end bins; use infinite outer edges (`bins=[-inf, 50, 200, inf]`) to
  keep everything.
- `histogram` raises on a `range` whose min exceeds its max, and checks that
  explicit edges increase at plan time.
- `label_reduce(reduction="max")` propagates NaN, and any geometry with a NaN or
  infinite coordinate is an error naming the row (it panicked or gave wrong
  results before).
- Mixed dtypes promote as NumPy's `result_type` does: `u8`+`i8` is `i16`,
  `u32`+`f32` is `f64`, and `u64` with a signed integer is `f64` (bitwise ops
  refuse it).
- Float-promoting ops (`scale`, `sqrt`, `divide`, `convolve2d`, gamma, the
  scalar math family) now give `f64` for `u32`/`i32`/`u64`/`i64` input,
  computed exactly; 8/16-bit integers still give `f32`.
- Requires `polars>=1.43.2` (was `>=1.41.1`); older polars made the plugin panic
  on a sliced `Array` column with nulls.
- `divide` follows IEEE 754 for a zero divisor (`x / 0` is `inf`, `0 / 0` is
  `nan`); it gave 0 before. Replace non-finite values afterwards if 0 was
  wanted.
- `ratio` is removed; use `divide` (and `.scale(255)` for the documented
  scaling).
- `extract_contours` traces along pixel edges: a `w x h` blob now bounds `w x h`
  (was `(w-1) x (h-1)`), a single pixel is a unit square of area 1, and
  `min_area` filters on pixel count. Vertices, `area`, `perimeter`,
  `bounding_box` and IoU change accordingly.
- Detection metrics move for any `ContourMatcher` evaluation: IoU against ground
  truth rises (most for small lesions), some former misses become true
  positives, and one-pixel-thick detections now count as false positives.
- `ContourMatcher(min_contour_area=)` defaults to `0.0` (was `1.0`), and
  `gt_min_contour_area=` defaults to `1.0` and no longer follows
  `min_contour_area` (`None` is no longer accepted).
- `normalize(method="zscore")` computes mean and standard deviation exactly, so
  values can differ from 0.29.0 in the last bits.

### Added

- `histogram(bins=[-math.inf, 0, math.inf])`: explicit edges may start at `-inf`
  and end at `inf`.
- `source("list", require_contiguous=True)` accepts rectangular `List` rows; it
  refused every `List` row before.

### Fixed

- Empty images are refused cleanly instead of panicking: `reduce_argmax`/
  `reduce_argmin` over an empty axis, `pad` with `mode="edge"`, `"reflect"` or
  `"symmetric"`, `reduce_max`/`min`/`percentile` over nothing, aspect-preserving
  resizes, and an `Array` column with a zero-size dimension.
- `convert_color(rgb -> gray)` disagreed with `grayscale()` in the last bit for
  some dtypes; both now use one luma.
- Histogram edges now meet integer pixels exactly (u64/i64 values above 2**53
  landed in the wrong bin), and a value on a bin edge is counted in the bin the
  edges give.
- `ViewBuffer::reshape` accepted a strided view in release builds.
- 8-bit HSV hue could round to 180; it now wraps to 0 as in OpenCV.
- A gray + alpha image now round-trips through the `tiff` sink as two channels.
- The `scalar_fusion` pass changed an integer chain's result (u16
  `invert().cast("u8")` of 1000 was 255 fused and 23 unfused); every pass is
  now byte-identical on and off.
- A blob or raw column sunk to `numpy`/`ndarray`/`torch` returned row 0's
  pixels for every row; every row is now returned.
- `crop(top=1).channel_select(0)` and other `reshape`/`channel_select` after a
  crop with an offset read the uncropped rows.
- `reshape` after `flip`, `transpose`, quarter `rotate` or `crop` failed with
  "cannot reshape a non-contiguous view"; it now copies first, as NumPy does.
- `channel_swap`, `channel_merge`, RGB/BGR, gray to RGB and
  `morphology_gradient` panicked or rounded values on dtypes other than u8; they
  are now exact on every dtype.
- Resizing, blurring and `rgb->gray` lost precision on 32/64-bit integer and
  f64 images (u32 16777217 became 16777216); they are now exact, and nearest
  resampling is an exact gather.
- Two-buffer ops (`add`, `subtract`, `multiply`, `blend`, `maximum`, `minimum`,
  bitwise) were computed in f32 for most dtypes (u32 `16777219 ^ 16777221` gave
  0, not 6), and `blend` was a plain saturating multiply on integers; they are
  now exact, with `blend` normalized as `round(a*b / MAX)`.
- Colour conversions silently misread a channel count they cannot handle (such
  as `to_bgr` of a 5-channel image); they now accept exactly the space's
  channels or those plus alpha.
- `convolve2d` (and `sobel`, `laplacian`, `sharpen`) on f64 planned f64 but
  executed f32 and failed; it now returns f64.
- A literal float parameter could land an ulp away from the same value given as
  an expression (visible in `warp_affine`'s matrix and `threshold` near a u64
  pixel); literals now parse exactly.
- `rotate_and_scale` with a per-row angle, centre or scale differed in the last
  bits from the same literals on f64 images; they now match.
- `threshold` on non-u8 integer pixels above 2**53 compared through f64 and
  answered for a neighbour; it now compares exactly.
- `rotate`/`warp_affine` filled off-image pixels with a truncated border value
  (u8 `border_value=7.5` gave 7, 300 gave 0); it is now rounded and saturated
  (8 and 255).
- `ContourMatcher` silently dropped one-pixel-thick regions, and `label_reduce`
  returned 0.0 for a zero-area contour; thin regions are now scored, measured
  and matched on their own pixels.
- `rotate(0)` spread NaN or infinity to neighbouring pixels; it now returns the
  image unchanged.
- `rotate` and `warp_affine` stored the top of the u64/i64 range as 0; they now
  saturate.
- A `list`/`array` value the declared dtype cannot hold (`300` or `-1` as `u8`,
  or `NaN`) became 0; it now fails naming the value, or nulls the row under
  `on_error="null"`.
- `sink("blob")` of a whole-row crop wrote the rest of the image after the crop,
  reading past the image's memory for a crop below the top row.
- `grayscale` of a non-u8 gray + alpha image mixed alpha into the intensity (u16
  (1000, 65535) gave 46239); it now returns the gray channel.
- `invert` on i8/i16/u32/i32/u64/i64 failed with a planned-dtype error; every
  integer dtype now inverts in its own dtype (`-1 - x` for signed, as NumPy's
  `~x`).

### Performance

- Broadcasting two-buffer ops are about 29x faster (1024x1024 RGB under a mask:
  about 104 ms to about 3.6 ms).
- Kernels use AVX2 on CPUs that have it, whatever the wheel was built for, with
  byte-identical output on every CPU.
- Float to integer `cast` is about 2x faster, and f32 to u8 a further 1.6–2.3x.
- u8 `grayscale` is 5–10x faster and u8 `threshold` 1.4–3x; non-u8 grayscale is
  1.4–2.9x faster.
- Per-value ops are much faster: u8 `adjust_gamma` 22–27x, u8 ImageNet-style
  `normalize` 5–12x, u8 `adjust_contrast` 2–5.5x, z-score 2.2–4x, and fused
  u8 chains 1.8–2.5x.
- Materialising a crop, flip or transpose is faster (at 1024² RGB u8: vertical
  flip about 22x, horizontal flip about 5x, transpose about 5.5x), and `cast`,
  per-value ops, `grayscale` and `resize` read crops and flips in place.
- Cheap rows cost about half as much: `invert` on 8×8 rows is 1.3x faster eager
  and 1.4x streaming.
- The plugin allocates through polars' allocator: 64×64 pipelines are
  1.15–1.66x faster and a repeated large `blob` sink 2x, with about 10% higher
  peak memory.
- `erode`/`dilate` with `iterations` and `blur` are 1.1–1.7x faster.
- `rotate` and `warp_affine` are 1.4–2.2x faster; a rotation by 0° or 360°
  returns its input.
- `list` sources are read in place: 3.6x faster for 64×64 u8 rows, 2x for flat
  f32 rows.
- The half-precision tensor sink (`sink("numpy"|"torch", dtype="f16")`) is 3–6x
  faster and runs in parallel.

### Removed

- `ratio` (use `divide`) and the `polars_cv._lib.rotation_matrix_2d` FFI.

## [0.29.0] — 2026-09-27

Plan-time optimizations are now one explicit, toggleable phase, and many silent
wrong results are now errors.
Upgrading from 0.28: see the [migration guide](https://heshamdar.github.io/polars-cv/user-guide/migration-0.29/).

### Breaking changes

- Builder methods take their one required parameter positional-or-keyword and
  every other parameter keyword-only, e.g. `.normalize("minmax")` is now
  `.normalize(method="minmax")`.
- Keyword-only now: `convert_color(from_space=, to_space=)`,
  `histogram(bins=, range=, closed=, output=)`,
  `normalize(method=, mean=, std=, out_dtype=)`, `reduce_max/mean/min(axis=)`,
  `reduce_std(axis=, ddof=)`, `warp_affine(matrix=, output_size=)`,
  `perceptual_hash(hash_size=, algorithm=)`, `.point.interpolate(t=)`.
- Positional now: `adjust_contrast(factor)`, `adjust_gamma(gamma)`,
  `channel_select(index)`, `channel_swap(order)`, `simplify(tolerance)`,
  `label_reduce(contours)`, `convolve2d(kernel)`.
- `source()` keywords default to `None`; `on_error=` or `require_contiguous=` on
  a format that does not read them is now refused.
- `source()` no longer takes `width`, `height`, `shape`, `fill_value` or
  `background`; write `source("contour").rasterize(...)`.
- `canny` now matches `cv2.Canny` pixel for pixel (more edges, no pre-blur, a
  colour image uses the strongest channel); add `.blur(sigma=...)` before it for
  the old smoothing.
- `.contour.scale()` now defaults to `origin="centroid"`, as
  `Pipeline.scale_contour()` does; it scaled about `(0, 0)`.
- `.contour.label_reduce(image)`'s `image` is required.
- `extract_contours()` yields `[]`, not null, for an image with no contours.
- A null row of `sink("numpy")`/`"torch"`/`"ndarray"` is now a null value; test
  `row is None`, not `row["data"] is None`. `numpy_from_struct(None)` raises
  `ValueError`.
- `Pipeline.output_dtype()` can return `"auto_float"`.
- `PipelineGraph.to_expr()` needs an optimized graph (`.sink()` produces one);
  `PipelineGraph.set_root_column()` is removed, and `to_graph(column)` requires
  its column.
- Hand-built graph JSON follows the typed definitions: unknown, misspelled,
  missing or inapplicable fields are rejected by name, as are the node fields
  `alias`, `domain`, `output_dtype`, the `expr_column_names` kwarg, and a sink
  `dtype` other than `"f16"`/`"float16"`. Renamed fields: `warp_affine(output_size=)`,
  `histogram(range=)`, the binary ops' `other`, `apply_mask`'s `mask`,
  `channel_merge`'s `others`, `rasterize`'s `size`.
- Requires `polars>=1.41.1` and `numpy>=2.0.2`; `networkx`, `graphviz` and
  `pydot` move to the `viz` extra (`pip install 'polars-cv[viz]'`), needed only
  by `show_graph()`. Wheels are tagged `cp310-abi3`.
- The single-thread warning and `POLARS_CV_ENGINE_WARN_SECONDS`,
  `POLARS_CV_ENGINE_WARN_ROWS` and `POLARS_CV_SILENCE_ENGINE_WARNING` are gone.
- `_graph.LIB_PATH` and `_namespace._LIB_PATH` are removed.

### Added

- Optimizations run as one phase at `.sink()`; `polars_cv.OptFlags`
  (`OptFlags.all()` / `OptFlags.none()`) toggles each pass, per call with
  `.sink(opt_flags=...)` or per process with `POLARS_CV_OPTIMIZATIONS`. All are
  on by default.
- `Pipeline.explain(optimized=True)` shows the chain after the passes `.sink()`
  would apply.
- New passes: `identity_elimination` (removes no-op pads, same-dtype casts and
  full-frame crops) and `spatial_window_pushdown` (moves an H/W crop ahead of
  preceding pointwise ops).
- `numpy_from_column(column)` reads a numpy/ndarray sink column as zero-copy
  read-only views; null rows are `None`, and `copy=True` returns owned arrays.
- The planner tracks every dimension at any rank, so rank-1, rank-2 and rank-4
  outputs publish their shape and `.sink("array")` needs no `shape=` for them.
  `assert_shape(dims=[...])` accepts any rank, and `source("array")` over a
  fixed-size `Array` column publishes the whole shape.
- `source("contour")` without a canvas decodes to the contour domain, so a
  pipeline can start from a contour column and measure or transform it.
- Arrow extension types `polars_cv.ndarray`, `.point`, `.contour` and `.bbox`
  (`NdArrayType`, `PointType`, `ContourType`, `BBoxType`) keep a column's
  identity through Parquet/IPC; every `.cv`, `.point`, `.contour` and `.bbox`
  accessor and every source accepts tagged columns.
- `sink("ndarray")` is the numpy sink tagged `polars_cv.ndarray`;
  `numpy_from_struct` and `show_images` read it. `sink("numpy")` is unchanged.
- Hand-built graph JSON may omit a field that has a declared default.
- Benchmarks: `maturin develop --profile benchmark` builds with thin LTO,
  `--select` and `--changed REF` run only the cases a change can move, and
  `run_suite` refuses debug builds unless given `--allow-debug-build`.

### Changed

- Every `.cv.pipe(...)` call runs its rows in parallel on the plugin's thread
  pool (sized by `POLARS_MAX_THREADS`): 3–4× faster on 4 cores in the measured
  cases.
- Every `.contour`, `.point` and `.bbox` function uses every core on the
  in-memory engine.
- An op no row could run on its planned input now raises when it is appended,
  naming the op: an image op on a rank-1 buffer, a crop past a known edge, a
  channel index past a known channel count, a repeated `transpose` axis, or
  operands whose known sizes cannot broadcast.
- Binary ops plan each axis they can, and `apply_mask` and `channel_merge` are
  planned against the nodes they read.
- `assert_shape` is a checked operation: a mismatch fails the row naming the
  assertion, `dims=` entries may be per-row expressions, and an assertion is
  never optimized away.
- `.sink()` validates the graph, so sink problems raise `ValueError` there.
- A float-promoting op over an unknown dtype (`scale`, `sqrt`, `sobel`, ...
  after `source("image_bytes")`) plans as `auto_float`.
- Errors name the op, field and valid values; a source or sink keyword the
  format does not read is refused naming where it applies.
- A `.point` distance function given a contour set now says so instead of
  failing to find an `x` field.

### Fixed

- Inputs an op cannot handle are now row errors that `on_error` handles, not
  panics, and some silent wrong results now raise: `reshape` to a different
  element count, image ops on anything but `[H, W]`/`[H, W, C]`, `transpose`
  with too few axes, and a reshape after a transpose/flip/crop.
- `crop` with a negative `top`/`left`/`height`/`width`, or a window past the
  image, now raises; it used to return a shifted or shrunk window. Giving only
  `height` or `width` no longer discards it.
- `source("raw")` rejects a byte length that is not a whole number of elements
  instead of dropping the remainder.
- A jagged or null `List` row is refused; it used to read past the row's values
  or come back re-rowed. `.contour.label_reduce`'s image operand is decoded the
  same way.
- A null contour coordinate is an error, not the origin.
- `.contour.contains_point` and other geometry functions read points and bboxes
  by field name and refuse null or non-`Float64` fields; they read a missing
  value as 0.0 and tested a `{y, x}` point swapped.
- `.point.rotate(origin=...)` with a null origin gives a null row, not a
  rotation about `(0, 0)`; a null input point gives a null result point.
- `numpy_from_struct` refuses a struct describing memory outside its data.
- `img.apply_mask(contours)`, `img.add(contours)` and `channel_merge` over a
  contour-domain node are refused when the pipeline is built, not at every row.
- `apply_contour_mask()` ran only the first op of the contour pipeline,
  dropping a `translate()` or `scale_contour()` before the rasterize.
- A per-row parameter was validated as a stand-in value:
  `channel_select(index=pl.col("i"))` on a known `[H, W]` buffer was refused.
- An image of unknown size was planned as square: `resize_max(7,
  filter=pl.col("f"))` published `[7, 7]` for a 100x50 image that executes as
  7x4.
- Distinct expression parameters could collapse into one (e.g. two
  `pl.lit(pl.Series(...))`), so `.scale(pl.lit(s1)).scale(pl.lit(s2))` applied
  `s1` twice.
- `cast(A)` then `cast(B)` dropped a narrowing intermediate cast: a
  fractional-`f32` `.cast("u8").cast("f32")` returned `0.5` instead of `1.0`.
- A malformed VIEW blob could cause undefined behaviour; it is now a row error.
- `on_error("null")` did not cover failures raised as panics, such as
  operands that cannot broadcast; one bad row failed the whole query.
- A row the sink cannot represent is an error, not a null; `list` and `array`
  sink rows that differ from the plan (dtype, element count) are errors.
- The OpenCV and Pillow benchmark adapters now time the same operation as
  polars-cv.

### Performance

- Less per-row executor overhead, and image decode and encode no longer copy
  every pixel an extra time.
- `canny` on u8 input runs in integer arithmetic.
- `blob` and `raw` sources read aligned rows in place.
- f32 scalar ops write in place in pipelines.
- `list` and `array` sinks cost about what the numpy sink costs (they were
  ~34–40× slower).
- Contour columns are read and written straight from Arrow: `.contour.translate()`
  about 6× faster on a 90k-point batch.
- `source("array")` is no longer quadratic in batch size.
- `blur()` uses AVX2 when available, about 1.2–1.3× faster, in the published
  wheels too.

### Removed

- `Pipeline.output_encoding()` and `PlanState.DIM_NAMES`.
- `convolve2d(ksize=)`, `sobel(ksize=)` and `laplacian(ksize=)`; the kernel is
  `3x3` for `sobel`/`laplacian`, and `convolve2d`'s side comes from `kernel`'s
  length (a square of an odd number). `sobel(axis=)` accepts only `"x"`/`"y"`.
- Loose contour-struct matching: the ring must be the `exterior` field (a
  `points` field or the first list field is no longer read).
- The per-row `"triangle"` alias for resize `filter=` (use `"bilinear"`) and
  `convert_color`'s `"grey"`/`"grayscale"` (use `"gray"`).
- The `affine_fusion` optimization pass; `rotate`, `warp_affine` and
  `rotate_and_scale` are unchanged and run unfused.

## [0.28.0] — 2026-09-12

### Breaking changes

- `froc_auc(method="trapezoidal")` raises without an `fp_range`, e.g.
  `fp_range=(0.0, 8.0)`.
- `froc_auc` and `lroc_auc` default to `correction="normalize"` (mean
  sensitivity over the window); pass `correction=None` for the raw partial area.
  `lroc_auc` defaults `fpf_range` to `(0.0, 1.0)`, and `froc_auc_ci_lazy` and
  `lroc_auc_ci_lazy` mirror these defaults.
- `partial_auc`'s below-curve `UserWarning` is gone.

### Added

- Elementwise `Pipeline` methods `neg`, `abs`, `sqrt`, `square`, `reciprocal`,
  `sign`, `floor`, `ceil`, `round`, `trunc`, `clamp_min`, `clamp_max`,
  `add_constant` and `subtract_constant`. They promote to float, fuse with
  adjacent scalar ops, and `round` breaks ties to even; the bound of
  `clamp_min`, `clamp_max`, `add_constant` and `subtract_constant` may be a
  per-row expression.

### Fixed

- Weighted FROC partial AUC changed with the scale of the weights; sub-unit
  weights roughly halved it. A zero weighted denominator now gives null, and so
  does LROC's zero-mass case.
- `correction="normalize"` returned run-to-run varying, often negative values
  on the in-memory engine.

### Removed

- The eager `trapz_auc`, `partial_auc` and `_interp` helpers; every AUC now
  reduces lazily.

## [0.27.0] — 2026-09-10

### Breaking changes

- `correction="mcclish"` is removed from `froc_auc`, `lroc_auc`, `partial_auc`,
  `trapz_auc` and `MetricResult.auc`; use `correction="normalize"`. An
  unrecognised `correction` now raises `ValueError` instead of degrading to the
  raw area.

## [0.26.0] — 2026-09-08

### Breaking changes

- The eager bootstrap API is removed: `bootstrap_froc_auc`, `bootstrap_lroc_auc`,
  `bootstrap_pr_auc`, `bootstrap_metric_sequential`, `BootstrapResult` and
  `MetricResult.bootstrap_ci`. Use `froc_auc_ci_lazy`, `lroc_auc_ci_lazy` or
  `average_precision_ci_lazy`, which return a `pl.LazyFrame`
  `[*group_by, <metric>, ci_lower, ci_upper]`; e.g.
  `bootstrap_froc_auc(t, seed=0).ci_lower` becomes
  `froc_auc_ci_lazy(t, seed=0).collect()["ci_lower"].item()`.
- `shear` and `rotate_and_scale` require `output_size` (and `rotate_and_scale`
  `center`) as keyword-only arguments; omitting one raises `TypeError`.
- All-points AP now collapses tied scores into one precision-recall point
  (scikit-learn/COCO convention), so results on inputs with tied scores change;
  distinct scores are unaffected.

### Changed

- `group_by=` on the lazy CI functions gives one bound row per group; a group
  with no positive target (or, for `method="mann_whitney"`, no negative) keeps
  its estimate with null `ci_lower`/`ci_upper`.
- `mean_average_precision(interpolation="all_points")` is a single lazy plan.

### Fixed

- `grayscale` on non-`u8` data was tracked as `u8`, so a fused op after it (e.g.
  `grayscale -> invert -> scale`) inverted normalized `f32` as `255 - x`
  instead of `1 - x`.

### Removed

- `PrecisionRecallResult.detection_table`, `MetricResult.interpolate` and
  `MetricResult.summary_table` (use `interpolate_curve_lazy`), and the
  `heatmap=` alias on `.contour.label_reduce()` (use `image=`).

## [0.25.0] — 2026-09-05

### Changed

- Bootstrap resampling is fully lazy and streaming, and a given `seed` gives a
  bit-identical interval regardless of thread count. `seed=None` is now
  deterministic. The draw stream differs from earlier releases, so seeded
  intervals change.
- `MetricResult.bootstrap_ci` is vectorized; `bootstrap_ci(metric="auc")`
  matches `bootstrap_pr_auc`. Supported metrics are `auc`, `precision_at` and
  `recall_at`; any other raises.

### Removed

- `MetricResult._reconstruct`, `MetricResult._resolve_sampling_entities` and
  `DetectionTable.image_ids_and_strata`.

## [0.24.0] — 2026-09-03

### Breaking changes

- `froc_sensitivity_at_fp`, `froc_summary_table` and `lroc_sensitivity_at_fpf`
  return a `LazyFrame`; a scalar is `froc_sensitivity_at_fp(t, fp).collect().item()`.
- `froc_curve_lazy` and `froc_auc` no longer raise on conflicting duplicate
  weights; they resolve them per the new `weight_agg` keyword.

### Added

- `weight_agg` (`"first"` default, `"min"`, `"max"`, `"mean"`, `"sum"`) resolves
  each `(image[, class])` weight.
- `froc_auc` and `lroc_auc` with `method="mann_whitney"` are weighted by
  `image_metadata.weight` at `level="detection"` and `level="image"`.

### Changed

- `froc_curve_lazy` and `froc_auc` do no eager collects when built.

## [0.23.0] — 2026-09-03

### Breaking changes

- `froc_auc` and `lroc_auc` return a `LazyFrame` with one row per group; use
  `froc_auc(table).collect().item()` for a float.
- Removed `FROCResult`, `LROCResult`, `froc_curve` and `lroc_curve`, and the
  multi-metric-dict bootstrap for FROC/LROC (call `bootstrap_froc_auc` or
  `bootstrap_lroc_auc` once per metric).

### Added

- `froc_curve_lazy` and `lroc_curve_lazy` return the curve as a `LazyFrame`, and
  `froc_sensitivity_at_fp`, `lroc_sensitivity_at_fpf` and `froc_summary_table`
  are standalone lazy functions.
- `bootstrap_froc_auc` and `bootstrap_lroc_auc` run every replicate in one lazy
  plan, are reproducible for a given `seed`, and take `sample_col` (entity-level
  resampling) and `level` (image- or detection-level Mann-Whitney).
- `ContourMatcher.match()` and `BBoxMatcher.match()` accept a pre-decoded
  `LazyPipelineExpr` for `pred_col` / `gt_col`, sharing one decode and one
  collect.

### Changed

- The contour and bbox matchers are lazy and `DetectionTable.collect` runs both
  derived frames in one `pl.collect_all`; empty and populated inputs now give
  tables with the same dtypes.

## [0.22.0] — 2026-09-01

### Breaking changes

- `.contour.match_detections()` and `.bbox.match_detections()` are now
  `.contour.correspond()` and `.bbox.correspond()`, which take a visit order (a
  permutation) instead of scores; matching results are unchanged.
- `correspond` rejects a wrong-length, out-of-range or repeated visit order,
  naming the row.
- `n_tp`, `n_fp`, `n_fn` and `pred_idx` are gone from the match result, and
  `MATCH_RESULT_SCHEMA` is replaced by `CORRESPONDENCE_SCHEMA` (`right_idx`,
  `overlap`), exported at the package top level.

### Added

- `polars_cv.dtype_name_for` maps a Polars dtype to the engine's name.

### Changed

- Overlap ties go to the smallest index, and an overlap larger by one ULP is not
  a tie.

### Fixed

- Predictions on a ground-truth-free image no longer carry a null payload in
  the metrics matchers; they remain false positives.

## [0.21.0] — 2026-08-24

### Added

- `Pipeline.scale_contour(origin=)`, matching `.contour.scale`; the defaults
  still differ (`"centroid"` and `"origin"`).
- Ambient AWS configuration is honoured: `~/.aws/config` and
  `~/.aws/credentials` supply a region and keys when the options do not, so a
  machine with an unrelated AWS profile may sign requests that went unsigned
  before; pass `anonymous=True` for a public bucket.

### Changed

- `rotate(expand=True)` output dimensions can differ by one pixel from 0.20.0
  for about 1 in 5000 inputs.
- Remote reads reuse connections across files and calls, and fetch concurrency
  is one global budget, `POLARS_CONCURRENCY_BUDGET` (default
  `max(thread count, 10)`).
- Store-build and read errors carry polars' context.

### Fixed

- `ContourMatcher` failed with "Invalid blob magic bytes" on an image-bytes
  (PNG) mask column; a `String` column of mask paths now works too.
- `Pipeline.to_graph()` discarded `on_error` and `on_null_param`.
- `extract_shape().reduce_sum()` planned cleanly and then failed every row.
- An unmappable dtype string was silently typed `UInt8`; it now errors.
- A misspelled key in the graph wire format (e.g. `allowed_roots`, which
  disabled the path sandbox) and a misspelled kwarg such as
  `sink("jpeg", qualtiy=50)` were silently ignored; they are now rejected.
- `gcs_bearer_token` on an `s3://` path is now an error instead of being
  ignored.
- S3 object keys were double percent-encoded: `s3://bucket/a b.png` requested
  `a%2520b.png`.
- A misspelled storage option is still rejected.

### Performance

- The remote source no longer opens a connection per file.

### Removed

- `match_detections(strategy=)` on `.contour` and `.bbox`; it was ignored, and
  `strategy="hungarian"` matched greedily.

## [0.20.0] — 2026-08-13

Every `.contour` accessor now reads contour sets, and `assert_shape` can pin
rank.

### Added

- Every `.contour` accessor accepts either a single contour or a contour set
  (what `extract_contours()` produces) and wraps its result to match, e.g.
  `area()` returns `Float64` or `List(Float64)`.
- `iou`, `dice` and `hausdorff_distance` broadcast a set against a single
  contour; a set on both sides raises.
- `Pipeline.assert_shape(dims=[...])` pins the output rank as well as the
  per-dimension sizes, which lets a `list`/`array` source reach a fixed-shape
  `array` sink.
- `.contour.ensure_winding(direction=)` and `.contour.scale(origin=)` accept
  Polars expressions, and now follow `on_null(...)`.

### Changed

- The `array` sink's error now names the missing dimensions and what each
  remedy supplies.

### Fixed

- `assert_shape` raises at the call when it contradicts a known dimension,
  instead of failing at `collect()`; `height=`/`width=`/`channels=` are
  rejected once the rank is known not to be 3 (use `dims=`).
- `.contour.ensure_winding()` and `.contour.scale(origin=)` raise on an
  unrecognised value instead of guessing: `ensure_winding("CW")` used to return
  counter-clockwise, and `scale(origin="top_left")` scaled about the centroid.
- `numpy_from_struct()` read `"u8"` and `"i8"` as uint64/int64; it now uses this
  project's dtype names and rejects numpy character codes and `bool`/`f2`.
- `statistics(include=<generator>)` raised `IndexError`.

## [0.19.0] — 2026-08-09

Strict rejection of ignored parameters, plan-time schema fixes, and corrected
detection metrics.

### Breaking changes

- `average_precision(interpolation="all_points")` (the default) omitted the
  first recall segment; reported AP is now higher and correct. `11_point` is
  unchanged.
- FROC/LROC curves are now sorted by descending `threshold`; reported AUC
  changes.
- FROC `fp_per_image` now divides by the number of distinct `image_id`s, not
  metadata rows.
- Duplicated `image_id`s with conflicting weights in FROC metadata now raise
  `ValueError`.
- `.contour.label_reduce()` scores a contour that covers no pixel centre at its
  centroid value instead of 0.0, matching `Pipeline.label_reduce()`.
- `MetricResult.interpolate`, `sensitivity_at_fp`, `sensitivity_at_fpf` and
  `summary_table` return null past the curve's observed range instead of
  repeating the last value.
- `PreMatchedAdapter.match()` warns unless you pass `image_meta`, the frame
  defining the full image population; combining `image_meta` with `n_gts_col`,
  `weight_col`, `gt_label_col` or `group_col` raises.
- `source("auto")` over a `Binary` image column into a `list` sink is now a
  planning error; name the source or use another sink.
- `sink("webp", quality=)` now raises, as do any unknown sink keywords and any
  `sink()`/`source()` parameter that does not apply to the chosen format (e.g.
  `source("image_bytes", width=224)`, `shape=` on a png sink).
- `source("contour", dtype=...)` raises (it always yields `u8`), and
  `cloud_options` on a non-path source raises instead of warning.
- `rasterize(anti_alias=)` is removed; it was ignored.
- `out_dtype="preserve"` is removed; use `preserve_dtype=True`. `out_dtype`
  accepts any dtype name.
- A newer Python paired with an older compiled extension now fails on unknown
  graph node fields; re-run `maturin develop`.
- `polars_cv.geometry.validation` and its exception classes are removed; they
  were never raised.

### Added

- `source("file_path", allowed_roots=[...])` and
  `.cv.read_bytes(allowed_roots=[...])` sandbox which local paths and remote
  URI prefixes a path column may read; off by default, and a refused path
  follows `on_error`.
- Operation parameters that no op reads are now rejected.

### Fixed

- `scale(out_dtype=)` and `clamp(out_dtype=)` were accepted and ignored; they
  now cast the result.
- `warp_affine` returned the input unchanged for a singular matrix; it now
  raises, naming the determinant.
- `source("contour")` could not read the contour sets that
  `extract_contours().sink("native")` emits.
- Rasterizing several contours with `fill_value < background` returned an
  all-background canvas.
- `source("contour")` now publishes its `[H, W, 1]` `u8` schema, so it plans
  into `list` and `array` sinks without a no-op `.cast("u8")`; `rasterize(width=,
  height=)` publishes its canvas shape.
- Planned and produced dtypes disagreed on `list`/`array` sources followed by
  promoting ops, on multi-root graphs with differently typed columns, and when
  null rows came first. They now agree.
- `png`/`jpeg`/`webp`/`tiff` sinks now reject unencodable dtypes or shapes at
  planning time rather than part-way through `collect()`.
- `perceptual_hash().sink("native")`, `extract_shape().sink("array", shape=)`
  and `perceptual_hash().sink("array")` planned a schema and then failed.
- `transpose` and `flip` raise `ValueError` for out-of-range axes instead of a
  panic.
- `assert_shape(...)` keeps later shape-changing ops accurate, and
  `transpose`, `channel_select`, `channel_merge` and axis reductions no longer
  publish stale shapes.
- The lazy `.pipe(p).resize_to_height(50)` and similar reported wrong
  shapes (`resize_max`, `pad`, `pad_to_size`, `rotate`).
- `FROC` double-counted detections when `image_id` repeated in metadata, and
  its bootstrap intervals could exclude the point estimate; bootstrap draws are
  now distinct units.
- `MetricResult.auc` and `.interpolate` were ambiguous on tied x values;
  `sensitivity_at_fp(0.0)` now reports the best sensitivity at zero false
  positives.
- `partial_auc` rejected integer bounds such as `fp_range=(0, 8)`.

### Removed

- Unused public items in `view-buffer` (cost reporting, `layout_report`,
  `from_slice_simd_aligned`, `is_simd_aligned`, dead geometry helpers and the
  `numpy_interop`/`torch_interop` features).

## [0.18.0] — 2026-07-31

Contour geometry is now computed exactly; reported IoU, Dice and contour
measures change.

### Breaking changes

- Contour IoU and Dice were wrong for non-convex or clockwise contours (a
  clockwise square scored 0.0 against itself) and are now exact, symmetric,
  winding-independent and hole-aware. IoU and Dice generally go up; recheck
  thresholds. This affects `.contour.iou()`, `.dice()`, `.pairwise_iou()`,
  `.match_detections()`, `ContourMatcher`, `DetectionTable` and the
  FROC/LROC/PR curves.
- `extract_contours` now traces region outlines; before, a filled region came
  back as many degenerate 2x2 contours. Hole rims trace correctly and `mode="all"`
  reports one border per enclosed background region. The outline runs through
  boundary pixel centres, so a `w x h` region has bounds `(w-1) x (h-1)`.
- Rasterized masks (`source("contour")`, `Pipeline.rasterize()`) were one pixel
  too wide at right-hand edges; a mask is now exactly the pixels whose centre
  lies inside the shape.
- `area` and `centroid` now measure the exterior minus the union of the holes,
  so overlapping or nested holes no longer double-count.
- `hausdorff_distance` also walks hole vertices, so a holed contour against the
  solid shape reports a positive distance; on an empty contour it returns
  infinity again.
- `centroid` of a zero-area ring is length-weighted. `simplify(tolerance=0.0)`
  is a no-op and never collapses a ring below 4 points.
  `nearest_point_on_contour` returns null for an all-identical ring.
- Point-in-polygon and convexity tests use exact predicates, so points within
  `1e-10` of an edge are no longer "on the boundary".

### Added

- `polars_cv.build_info()` reports the Python, compiled-extension and installed
  versions, to detect a stale extension.

### Changed

- Bounding-box IoU is computed analytically from the rectangle overlap.
- Hole-ness comes only from `CONTOUR_SCHEMA`'s `holes` field; ring winding
  never marks a hole.

## [0.17.0] — 2026-07-31

Null handling for per-row expression parameters.

### Added

- `Pipeline.on_null_param("raise" | "null")` chooses between failing the query
  (default) and nulling the rows whose parameter is null. The geometry
  namespaces use `on_null("null")` on the accessor.

### Fixed

- A null operand in `merge_pipe`, `apply_mask` or `channel_merge` raised
  "references unknown node" instead of nulling the row.
- `source("contour", shape=...)` failed on rows with a null shape reference, and
  failed outright when the expression was referenced only as the shape.
- A null in a non-primitive parameter column reported a cast failure instead of
  the null-value error.

## [0.16.0] — 2026-07-30

Per-row expression parameters across the geometry namespaces and many more
image ops.

### Added

- `.contour`, `.point` and `.bbox` numeric parameters accept expressions:
  `normalize`, `to_absolute`, `translate`, `scale`, `simplify`, `area(signed=)`
  and `match_detections(threshold=)` on `.contour`; `normalize`, `to_absolute`,
  `translate`, `scale`, `rotate(angle=)` and `interpolate(t=)` on `.point`;
  `match_detections(threshold=)` on `.bbox`.
- List-valued parameters accept per-element expressions (length stays literal):
  `convolve2d(kernel=)`, `normalize(mean=, std=)`, `channel_swap(order=)`; this
  makes `sharpen(strength=)` per-row.
- Per-row enums and flags: `filter` on resize variants and `letterbox`,
  `interpolation` on `rotate`/`warp_affine`, `pad(mode=)`,
  `pad_to_size(position=)`, `convolve2d(border=, normalize=)`,
  `extract_contours(mode=, method=)`, `label_reduce(reduction=, region_mode=)`,
  `apply_mask(invert=)`, `area(signed=)`.
- `rotate_and_scale(angle=, scale=, center=, output_size=)`, and the contour
  source's `fill_value`/`background`, accept expressions.
- `letterbox(filter=)` is exposed, and `FilterType` now includes `catmullrom`
  and `gaussian`.

### Changed

- A Boolean column no longer satisfies a numeric per-row parameter.

### Fixed

- `warp_affine` on `list`/`array` sources planned the wrong output rank.
- Single-channel affine warps (`warp_affine`, arbitrary-angle `rotate`, `shear`,
  `rotate_and_scale`) dropped the channel axis of `[H, W, 1]` input.
- `convolve2d` skipped kernel validation when `ksize` was an expression.
- Structural parameters given an expression (`cast(dtype=)`,
  `normalize(method=)`, `histogram(closed=, output=)`,
  `perceptual_hash(algorithm=)`, `transpose`/`flip` axes) now raise a clear
  "is structural" `TypeError`.
- The geometry guide used `ref_width=`/`ref_height=` for `point.normalize` and
  `point.to_absolute`; the parameters are `width`/`height`.

## [0.15.0] — 2026-07-26

### Added

- `.cv.read_bytes(cloud_options=, on_error=)` reads a path column's bytes
  verbatim as `Binary`, without decoding, so files round-trip unchanged and the
  header-only `.cv.width()` and friends can reach local and remote files.

## [0.14.0] — 2026-07-25

### Breaking changes

- `Pipeline.source()` defaults to `"auto"` instead of `"image_bytes"`: `String`
  columns read as paths, `List` as `list`, `Array` as `array`, and `Binary` as
  `blob` when it carries the VIEW magic, else `image_bytes`. An explicit format
  overrides it.

### Added

- `source("auto")` infers the decode path from the column dtype and raises on
  an unroutable dtype.
- A `u16` buffer encodes to a 16-bit PNG.

### Fixed

- JPEG/WebP sinks given a non-`u8` buffer, and float buffers into any image
  sink, now fail with an error that suggests `.cast("u8")` or the PNG/TIFF sinks.

## [0.13.0] — 2026-07-24

### Added

- Federated GCS credentials (`external_account`,
  `external_account_authorized_user`) work via the `gcloud` CLI; set
  `POLARS_CV_DISABLE_GCS_FEDERATION=1` to disable.
- `CloudOptions.token_command` supplies an OAuth token from any shell command
  for GCS and Azure; it raises with `s3://` sources.

### Fixed

- `gcs_bearer_token` was defeated by unparseable ambient credentials;
  `object_store` is bumped to 0.13.

## [0.12.0] — 2026-07-23

### Breaking changes

- The `warp_affine` `matrix` graph-JSON is now per-element; regenerate any
  persisted or hand-written graph. Pipelines built through Python are
  unaffected.

### Added

- `.sink("numpy"|"torch", dtype="f16")` outputs half precision.
- `warp_affine(matrix=)` accepts per-element expressions and `shear(sx=, sy=)`
  accepts expressions.
- `Pipeline.thumbnail(max_size)` chainable form of the IDCT-scaled decode.
- A one-time warning for large single-threaded in-memory batches, pointing to
  `engine="streaming"`; silence with `POLARS_CV_SILENCE_ENGINE_WARNING=1`.
- `CloudOptions.storage_options` forwards native `object_store` config keys and
  wins over named fields; `CloudOptions.gcs_bearer_token` supplies a GCS token.

### Fixed

- `cloud_options` on a non-`file_path` source warns instead of silently
  dropping credentials.
- `normalize(out_dtype=)` other than f32 failed the dtype check; it now casts
  to the requested dtype. `out_dtype="preserve"` is rejected.

## [0.11.0] — 2026-07-10

### Added

- `source("file_path")` reads `file://` URLs.

### Changed

- Cloud access is signed by default; GCS now honors
  `CloudOptions(anonymous=True)` like S3 and Azure.

### Fixed

- Malformed `blob` headers, Arrow C-Data offsets and null-bearing arrays are
  validated instead of risking overflow or wrong reads.
- Panics in axis reductions over 1-D buffers and in `u8` RGBA to Lab; the Lab
  path now preserves dtype.
- Metrics: VOC 11-point AP denominator and bootstrap PR-AUC estimator.
- Colon-bearing local paths are read literally.
- `gamma`'s integer range, rotation fusion shapes and an owned-buffer
  deallocation mismatch.
- Builds again on current stable Rust.

## [0.10.0] — 2026-06-13

### Added

- `scale`, `clamp` and `adjust_brightness` accept `preserve_dtype=True`.
- `Pipeline.on_error("raise" | "null" | "null_with_message")` nulls failing
  rows instead of failing the batch.
- Every chainable `Pipeline` op can be called on the lazy expression from
  `.cv.pipe(...)`.
- `rasterize(shape=<expr>)` and `source("contour", shape=<expr>)` size the
  canvas from another pipeline's output.
- `rotate()` accepts `interpolation` and `border_value`.

### Changed

- Planned dtype and shape now match execution for promoting binary ops and
  16-bit image sources.

### Fixed

- `cloud_options` round-trips for remote `file_path` sources.

### Performance

- More kernel fusion (casts, `rotate` into `warp_affine`) and faster `erode`,
  `dilate`, `convolve2d`, Gaussian blur and Canny.
- Remote `file_path` sources prefetch concurrently, and compiled graphs are
  cached across calls.

### Removed

- `OPERATION_CONTRACTS`, `AlphaMode` and `NdimEffect`.

---

_Releases earlier than 0.10.0 predate this changelog; see the git history for
details._

[Unreleased]: https://github.com/heshamdar/polars-cv/compare/v0.41.0...HEAD
[0.41.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.41.0
[0.40.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.40.0
[0.39.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.39.0
[0.38.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.38.0
[0.37.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.37.0
[0.36.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.36.0
[0.35.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.35.0
[0.34.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.34.0
[0.33.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.33.0
[0.32.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.32.0
[0.31.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.31.0
[0.30.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.30.0
[0.29.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.29.0
[0.28.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.28.0
[0.27.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.27.0
[0.26.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.26.0
[0.25.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.25.0
[0.24.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.24.0
[0.23.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.23.0
[0.22.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.22.0
[0.21.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.21.0
[0.20.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.20.0
[0.19.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.19.0
[0.18.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.18.0
[0.17.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.17.0
[0.16.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.16.0
[0.15.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.15.0
[0.14.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.14.0
[0.13.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.13.0
[0.12.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.12.0
[0.11.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.11.0
[0.10.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.10.0
