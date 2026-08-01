# Changelog

All notable changes to **polars-cv** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.18.0] — 2026-07-31

### Fixed

- **Contour IoU and Dice were wrong for most real contours.** Overlap went through
  a hand-rolled Sutherland–Hodgman clipper, which is only correct when the *clip*
  polygon is convex, and whose half-plane test hard-coded a counter-clockwise
  winding. Neither condition is required by `CONTOUR_SCHEMA`, and neither was
  enforced. Concretely, against an identical copy of itself: a clockwise-wound
  square scored **0.0**, an L-shape **0.2**, a U-shape **0.0**. `iou(a, b)` was not
  even symmetric — it returned 1.0 or 0.0 for the same pair depending on argument
  order. Holes compounded it: they were subtracted from each contour's area but
  ignored when intersecting, so a holed contour against itself produced an
  intersection larger than its union, saved only by a final clamp; with a large
  enough hole the union went negative and the result collapsed to 0.0.

  Segmentation contours are essentially never convex, so this affected
  `.contour.iou()`, `.contour.dice()`, `.contour.pairwise_iou()`,
  `.contour.match_detections()`, and everything built on them —
  `ContourMatcher`, `DetectionTable`, and the FROC/LROC/precision-recall curves.

  Overlap is now computed exactly, by `geo`'s boolean operations. Results are
  winding-independent, hole-aware and symmetric. **This changes reported numbers**:
  IoU and Dice will generally go up, so thresholds calibrated against the old
  behaviour should be re-checked.

- **Bounding-box IoU** no longer round-trips through polygon clipping; it is
  computed analytically from the rectangle overlap.

- **Overlapping and nested hole rings.** `Contour::area` subtracted each hole's
  area in turn, double-counting wherever two hole rings overlapped, and reporting
  a different region than `contains_point` and `rasterize` (which have always
  treated the region as the exterior minus the *union* of the hole rings). Area
  now uses that same region. A square with a hole containing a further ring reports
  3600 rather than 3200, and `iou` against the solid square reports 0.36 rather
  than 0.4348.

- **`hausdorff_distance` on an empty contour** returned `-1.797e308` — a negative
  distance — because `geo` folds with `Bounded::min_value()` where the previous
  implementation used `f64::INFINITY`. It returns `INFINITY` again.

- **Rasterized masks were one pixel too wide at every right-hand edge.** The
  scanline filler behind `source("contour", ...)` and `Pipeline.rasterize()`
  rounded each span outward — `ceil(left)` to `floor(right)` — instead of asking
  which pixel *centres* fall inside it. Since the scanline itself samples at
  `y + 0.5`, the vertical extent was already right, so masks came out asymmetric:
  a 400×400 box rasterized 401×400 pixels, and every interior hole was likewise
  a column too wide. A mask's pixel count therefore did not agree with the
  contour's `area()`, and `apply_mask` with a contour mask included a column of
  pixels outside the contour. Masks are now exactly the set of pixels whose
  centre lies in the shape — the rule `contains_point` and the area measures
  already followed — so an axis-aligned box on integer coordinates rasterizes to
  precisely its area. Zero-width or zero-height output no longer panics.

- **`centroid` measured a different region than `area`.** It was the one measure
  left on the holes-as-interior-rings representation, so it subtracted each hole's
  moment in turn — double-counting wherever two hole rings overlap, and subtracting
  a nested ring that lies in an already-removed part. It now measures the same
  region as `area`, `contains_point`, `iou` and rasterization. For a 100×100 square
  with holes `[10,50]²` and `[30,70]²`, the centroid moves from `(54.71, 54.71)` —
  a shape that does not exist — to `(53.89, 53.89)`. Contours with no holes, and
  contours whose holes are disjoint, are unaffected.

### Changed

- **`view-buffer` now depends on [`geo`](https://crates.io/crates/geo)** (0.33,
  `default-features = false`) for its polygon maths. Alongside the clipper, the
  hand-rolled Douglas–Peucker simplification, Graham-scan convex hull, ray-casting
  point-in-polygon, convexity test, shoelace centroid and point-to-segment
  projection were deleted in favour of `geo`'s implementations. Public signatures
  in `view-buffer::geometry` are unchanged, but several **semantics** are not:

  - The boolean ops, convexity test and point-in-polygon now use exact orientation
    predicates instead of epsilon comparisons. A point within `1e-10` of an edge
    previously read as *on the boundary* and now reads as inside or outside.
    (Area, centroid, simplification and distance remain ordinary floating point.)
  - `hausdorff_distance` walks hole vertices as well as the exterior. A holed
    contour against the same solid shape used to report 0.0 and now reports a
    positive distance.
  - `centroid` of a **zero-area** ring returns `geo`'s length-weighted centroid
    rather than the mean of the vertices. Non-degenerate contours are unaffected.
  - `simplify(tolerance=0.0)` is now a no-op rather than dropping collinear
    points, and simplification runs on the closed ring, so a ring is never
    collapsed below 4 points — `simplify` with a huge tolerance returns the shape
    instead of a degenerate 2-point line.
  - `nearest_point_on_contour` returns null for a ring whose points are all
    identical, where it used to return that point.

- **Hole-ness is documented as structural, and only structural.** The `holes`
  field of `CONTOUR_SCHEMA` is the sole carrier; ring winding is never interpreted
  as a hole signal. The schema docstrings previously also stated a "CCW = exterior,
  CW = hole" convention that no code read or enforced — following it (via
  `.contour.ensure_winding("cw")`) is precisely what produced an IoU of 0.
  `.contour.winding()` reports point order and `.contour.ensure_winding()` sets it;
  nothing else consults it. `is_closed` is documented as reserved: it is written
  unconditionally as `true` and never read back.

### Added

- **`polars_cv.build_info()`** reports the three versions that must agree:
  `__version__` (the imported Python source), the compiled extension's version
  (now exposed as `polars_cv._lib.__version__`, baked in from `Cargo.toml`), and
  the installed distribution's. The install is editable, so Python edits are live
  while the compiled extension stays at its last `maturin develop` — after a `git
  pull` that touches Rust, an unrebuilt environment silently runs new Python
  against old Rust. This makes that visible.

- **`tests/test_version_consistency.py`** fails when the version manifests drift
  apart or when the installed package is stale. The release checklist in
  `CONTRIBUTING.md` previously noted that nothing checked them.

## [0.17.0] — 2026-07-31

### Added

- **Null handling for per-row expression parameters.** A parameter given as a
  `pl.Expr` is read from an ordinary column, which may contain nulls; until now
  that was unconditionally fatal. `Pipeline.on_null_param("raise" | "null")`
  chooses between failing the query (the default, unchanged) and yielding null
  for the affected rows — the same null-in/null-out behaviour a null *input
  image* has always had. The `.contour` / `.point` / `.bbox` namespaces bypass
  the graph engine, so they carry the policy on the accessor instead:
  `pl.col("c").contour.on_null("null").normalize(pl.col("w"), 100)`.

  This is one shared mechanism rather than per-operation handling: the policy
  rides on `ParamCtx`, and every null — numeric, enum, flag or list element, for
  any of the ~70 operations — passes through `ParamCol::on_null`. No `resolve_op`
  arm changed.

  Two properties distinguish it from the existing `on_error("null")`:

  - **Node-scoped.** A null parameter leaves that node without an output for the
    row, which propagates through the path a null input already takes, so only
    the outputs that actually depend on it go null — not every output of the row.
  - **Independent of error reporting.** A null parameter is not treated as an
    error, so it records no `_error` message under `null_with_message`, and
    decode, encode and genuine operation failures still raise.

  There is deliberately no "fallback default" mode: `pl.col("h").fill_null(224)`
  already expresses it in Polars, and adding a per-parameter policy would have
  had to enter the `ParamValue` wire format and its equality/hash (which CSE
  depends on).

### Fixed

- A null operand no longer raises "references unknown node". A node that
  produced nothing for a row — null input bytes, `source(on_error="null")`, or
  now a null parameter — was indistinguishable from a node missing from the
  graph, so a null image in a `merge_pipe` / `apply_mask` / `channel_merge`
  operand column failed the query instead of nulling that row. All five
  cross-node operand reads now go through `CompiledGraph::operand`, which
  separates the two cases.
- `source("contour", shape=...)` no longer fails a row whose shape reference is
  null. A null image in the shape branch raised `Shape reference '<id>' not
  found. Ensure the shape source is defined before this contour pipeline.` —
  a message pointing at a graph-wiring problem that did not exist.
- `source("contour", shape=expr)` now works when `expr` is referenced *only* as
  the shape, with no nulls involved. The source recorded the reference by node
  id but never registered it as a dependency, so the node was left out of the
  graph entirely and execution failed on a dangling reference. It happened to
  work whenever the shape node was also consumed some other way — masking with
  it, which is what every example does — which is why it went unnoticed.
- A null in a non-primitive parameter column reported a cast failure from
  `try_extract` rather than the null-value error, bypassing the null path.

### Documentation

- The API reference gained a `BBoxNamespace` section — `.bbox` was the one
  namespace with no reference page — plus the `on_null` policy the three
  geometry accessors share, which is inherited from a mixin and so was not
  rendered from any of their own class bodies.
- `LazyPipelineExpr`'s reference page now says that the `Pipeline` methods
  mirrored onto it are generated at import time, which is why they are absent
  from the members list below it.
- The composition rule for `on_null_param` is stated correctly: unlike
  `on_error`, an explicit `"raise"` composed with a `"null"` gives the graph
  `"null"` rather than being rejected as a conflict, because only non-default
  policies are collected and there are only two values.
- `shape=` on the contour source is documented as registering an upstream
  dependency, so referencing a pipeline only as a shape is a supported shape of
  graph, and a null in that branch nulls the mask.

## [0.16.0] — 2026-07-30

### Added

- **Per-row expression parameters in the geometry namespaces.** `.contour`,
  `.point` and `.bbox` previously accepted only literals — and four `.contour`
  methods annotated `int | pl.Expr` while unconditionally raising `TypeError` on
  exactly that type, an impossible signature that the API reference published.
  Their numeric parameters now resolve per row:

  ```python
  # Normalize each contour against its own image's dimensions
  df.with_columns(
      norm=pl.col("contour").contour.normalize(pl.col("img_w"), pl.col("img_h"))
  )
  ```

  Covers `normalize`, `to_absolute`, `translate`, `scale`, `simplify`,
  `area(signed=)` and `match_detections(threshold=)` on `.contour`; `normalize`,
  `to_absolute`, `translate`, `scale`, `rotate(angle=)` and `interpolate(t=)` on
  `.point`; and `match_detections(threshold=)` on `.bbox`. Aggregations
  broadcast and null parameters are errors, matching the image operations.

- **Per-row elements in list-valued parameters.** The list *length* stays
  structural — it fixes a kernel size or channel count at planning time — while
  each element may now be an expression, the encoding `warp_affine`'s matrix
  already used. This covers `convolve2d(kernel=)`, `normalize(mean=, std=)` and
  `channel_swap(order=)`, and in turn makes `sharpen(strength=)` per-row, whose
  docstring previously described the limitation as permanent.

- **Per-row non-structural enums and flags.** `filter` (every resize variant and
  `letterbox`), `interpolation` (`rotate`, `warp_affine`), `pad(mode=)`,
  `pad_to_size(position=)`, `convolve2d(border=)`, `extract_contours(mode=,
  method=)`, `label_reduce(reduction=, region_mode=)` — on both `Pipeline` and
  the `.contour` namespace — plus the flags `apply_mask(invert=)`,
  `convolve2d(normalize=)` and `area(signed=)`. A parameter is eligible only
  when it has no effect on output shape, rank or dtype; that invariant is what
  lets plan-time shape probing substitute the default.
  (`rasterize(anti_alias=)` is plumbed identically, but view-buffer's
  rasterizer still ignores the flag, so it has no observable effect yet.)

- **`rotate_and_scale(angle=, scale=, center=, output_size=)`** accept
  expressions, building the affine matrix from expression arithmetic.
- **`letterbox(filter=)`** is now exposed; Rust already read it. The historical
  `lanczos3` default is unchanged.
- **Contour-source `fill_value` / `background`** accept expressions, matching the
  identical parameters on the `rasterize` op.
- **`FilterType` exposes all five view-buffer filters.** `catmullrom` and
  `gaussian` were deliberately withheld from Python while Rust accepted them.
  Making `filter` per-row broke that restriction asymmetrically — a literal was
  checked against the smaller Python enum, a column value went straight to
  Rust's larger table — so rather than validate the same subset twice, the
  subset was dropped and `FilterType` is now full parity.

### Changed

- **A Boolean column no longer satisfies a numeric per-row parameter.** It used
  to coerce to `0`/`1`, which reads as a mis-routed expression far more often
  than as a value someone chose; passing one now raises the same cast error a
  `String` column does.

### Fixed

- **Plan/execution rank desync for ops with list-valued parameters on `list` and
  `array` sources.** `fold_output_rank` re-serialized ops that graph compilation
  had already bound, and the compiled-only `Slot`/`List` forms are
  `#[serde(skip)]` — so the op silently failed to resolve, the output rank was
  recorded as unknown, and a planned `List(List(List(f32)))` collapsed to
  `List(f32)`. Affected `warp_affine` on list/array sources.
- **Single-channel affine warps dropped the channel axis.** `warp_affine` (and
  the arbitrary-angle `rotate`/`shear`/`rotate_and_scale` paths that share its
  kernel) collapsed a `[H, W, 1]` input to `[H, W]`, contradicting the op's own
  `infer_shape`, which preserves the input rank. A `[H, W]` input still produces
  `[H, W]`; only the explicit single-channel axis is now retained. This was
  masked by the rank-folding bug above.
- **`convolve2d` skipped kernel/`ksize` validation entirely when `ksize` was an
  expression**, letting a mismatched kernel reach Rust unchecked. The kernel
  length is structural, so it is now always validated as an odd perfect square.
- **Structural parameters given an expression now report why.** `cast(dtype=)`,
  `normalize(method=)`, `histogram(closed=, output=)`,
  `perceptual_hash(algorithm=)` and the `transpose`/`flip` axis lists failed
  opaquely inside `bool()` ("the truth value of an Expr is ambiguous"), on
  `.value`, or at JSON encoding; they now raise the same clear "is structural"
  `TypeError` as the other literal-only parameters.
- **`docs/user-guide/operations/geometry.md`** documented `point.normalize` and
  `point.to_absolute` with `ref_width=` / `ref_height=` — the Rust kwarg names.
  Both examples raised `TypeError`; the Python parameters are `width` / `height`.
- The `Pipeline` docstring claimed *all* operations accept expressions, and the
  image-ops guide's "all resize variants" claim sat directly above `thumbnail`,
  whose `max_size` is literal-only. Both now state the actual rule.
- The quickstart repeated the same "any parameter" claim; it now states the
  eligibility rule and links the full table. The pipelines concept page states
  the rule once where chaining is introduced, and points at the geometry
  namespaces, which now follow it too. The geometry guide called `.contour` and
  `.point` the geometry namespaces while documenting `.bbox` alongside them.

## [0.15.0] — 2026-07-26

### Added

- **`.cv.read_bytes()` — read a path column's bytes without decoding.** The
  `file_path` source has always been two stages (fetch the bytes a path names,
  then decode them as an image); the fetch stage is now available on its own and
  returns `Binary`. Bytes come back verbatim, so an encoded file survives the
  round trip byte-for-byte and can be written back unchanged — the only lossless
  path through the plugin, since re-encoding a decoded JPEG never reproduces the
  original and no image sink carries EXIF/ICC metadata. It also lets the
  header-only metadata methods (`.cv.width()` and friends, which take binary
  columns) reach local and remote files for the first time:

  ```python
  df = df.with_columns(raw=pl.col("path").cv.read_bytes(cloud_options=options))
  df = df.filter(pl.col("raw").cv.width() > 512)   # header-only, no decode
  df = df.with_columns(thumb=pl.col("raw").cv.pipe(pipe).sink("png"))
  ```

  Takes the same `cloud_options` as `source("file_path")` and the same
  `on_error` values (`"raise"` / `"null"`). Fetching is per plugin call — one
  morsel under the streaming engine — so distinct remote paths are deduped and
  fetched concurrently exactly as the source already does, and a bytes column
  stays morsel-bounded under `engine="streaming"`.

### Changed

- **The `file_path` source and `read_bytes` share one fetch implementation**
  (`src/fetch.rs`). The path-to-bytes stage — remote dedup and concurrent
  prefetch, the local-read path that avoids misparsing colon-bearing filenames
  as cloud URLs, and the `on_error` vocabulary — now lives in one module used by
  both, so their credentials, concurrency, and error messages cannot drift.
  Behaviour of existing pipelines is unchanged.

## [0.14.0] — 2026-07-25

### Added

- **`source("auto")` — decode path inferred from the column dtype.** A source no
  longer has to be told how to read its column when the Polars dtype already
  says: `String` → `file_path`, `List` → `list`, `Array` → `array`, and `Binary`
  → `blob` when the bytes carry the VIEW protocol magic, `image_bytes`
  otherwise. The resolution happens once per batch (the column dtype is constant
  across rows), so the now-default path costs no per-row work. Everything that
  applied to the concrete formats still applies: `cloud_options` and concurrent
  remote prefetch (an auto source over a URL column still prefetches),
  `decode_max_size`, and `require_contiguous`. A dtype that cannot be routed
  (e.g. a plain numeric column) raises and names the explicit formats to choose
  from.
- **16-bit PNG output.** A `u16` buffer now encodes to a 16-bit PNG
  (`Luma16`/`LumaA16`/`Rgb16`/`Rgba16`) instead of failing with "Image export
  requires U8 dtype", so a 16-bit PNG or TIFF read through a pipeline and back
  out to `sink("png")` keeps its bit depth end to end.

### Changed

- **`Pipeline.source()` defaults to `"auto"`** (was `"image_bytes"`). A `Binary`
  image column behaves exactly as before; a `String` column, which previously
  had to be spelled `source("file_path")`, now reads as a path; and a VIEW-tagged
  `Binary` column resolves to `blob` rather than failing an image decode. Passing
  an explicit format still overrides the inference in every case.

### Fixed

- **Actionable errors for bit-depth mismatches at the image sinks.** JPEG and
  WebP are 8-bit formats: sinking a non-`u8` buffer to them now fails before
  encoding with an error naming the dtype and pointing at `.cast("u8")` or the
  PNG/TIFF sinks, instead of a generic encode failure. Float buffers into any
  image sink likewise suggest casting to an integer dtype or sinking to TIFF,
  which supports floating point.

## [0.13.0] — 2026-07-24

### Added

- **Federated GCS authentication (Workload/Workforce Identity Federation)** —
  Application Default Credentials of type `external_account` and
  `external_account_authorized_user` (e.g. an OIDC identity exchanged into Google
  through an identity pool) now work without minting a bearer token by hand.
  When polars-cv detects a federated ADC (from `GOOGLE_APPLICATION_CREDENTIALS`,
  an explicit `google_application_credentials` option, or the well-known gcloud
  path), it delegates to `gcloud auth application-default print-access-token` —
  which understands the entire federation matrix — and uses the resulting token,
  cached until just before it expires. Requires the `gcloud` CLI on `PATH`; set
  `POLARS_CV_DISABLE_GCS_FEDERATION=1` to disable.
- **`CloudOptions.token_command`** — a provider-agnostic hook: any shell command
  whose stdout is an OAuth access token. Use it to source tokens from a custom
  broker, wrapper script, or CLI. It applies to both OAuth-bearer backends —
  **GCS and Azure** (e.g. `az account get-access-token`) — takes precedence over
  the automatic `gcloud` delegation, and its output is cached like any other
  token. It does not apply to S3 (SigV4, not bearer-based); passing it with an
  `s3://` source raises rather than silently ignoring the credential.

### Fixed

- **GCS `gcs_bearer_token` no longer defeated by unparseable ambient
  credentials** — bumped `object_store` to 0.13. In 0.12 the GCS builder read
  and parsed Application Default Credentials (`GOOGLE_APPLICATION_CREDENTIALS`
  or the well-known `~/.config/gcloud/application_default_credentials.json`)
  inside `build()` *unconditionally*, before honoring an explicitly supplied
  credential provider. So when the ambient ADC was a federated
  `external_account_authorized_user` credential — exactly the case
  `gcs_bearer_token` exists to work around — `build()` failed on the ADC parse
  and the bearer token was never consulted. object_store 0.13 skips the ADC
  read whenever an explicit credential (or service account) is configured,
  restoring the intended escape-hatch behavior. The bump also dedupes
  `object_store` to a single version, since polars 0.54 already pulls 0.13.

## [0.12.0] — 2026-07-23

### Added

- **f16 tensor output** — `.sink("numpy"|"torch", dtype="f16")` downcasts the
  output to half precision at the encode boundary, halving the output-tensor
  bytes and host→device transfer. (The engine has no native f16 dtype; only the
  sink container is f16.)
- **Per-row affine/shear parameters** — `warp_affine(matrix=...)` accepts a
  per-element mix of literals and Polars expressions, and `shear(sx=..., sy=...)`
  accepts expressions, so a batch can apply a different (e.g. random) affine or
  shear per row in one call.
- **`Pipeline.thumbnail(max_size)`** — explicit, chainable form of the existing
  JPEG IDCT-scaled decode (`source(..., decode_max_size=...)`), for cheap
  decode-aware curation passes.
- **Single-threaded execution warning** — a one-time notice when a large batch
  runs single-threaded under the in-memory engine, pointing to
  `engine="streaming"`. Silence with `POLARS_CV_SILENCE_ENGINE_WARNING=1`;
  tune with `POLARS_CV_ENGINE_WARN_ROWS`.
- **`CloudOptions.storage_options` pass-through** — arbitrary cloud options are
  now forwarded verbatim to the underlying `object_store` backend using its
  native config keys (e.g. `google_service_account_key`,
  `google_application_credentials`, `aws_endpoint`), so any option the backend
  understands is available without new plumbing. Keys in `storage_options` win
  over the named `CloudOptions` fields on collision. This replaces the previous
  fixed allow-list, which silently dropped unrecognized keys.
- **`CloudOptions.gcs_bearer_token`** — supply a pre-obtained GCS OAuth access
  token directly. This unblocks federated/brokered Google credentials (ADC of
  type `external_account_authorized_user`) that `object_store` cannot parse
  natively: mint a token out of band (e.g.
  `gcloud auth application-default print-access-token`) and pass it in.

### Changed

- **`warp_affine` `matrix` wire format** — the serialized `matrix` parameter is
  now an array of per-element `ParamValue` entries (each may be a literal or an
  expression) rather than a raw float array. This is an internal graph-JSON
  change; pipelines built through the Python API are unaffected. A persisted or
  hand-written graph using the old raw-float `matrix` array must be regenerated.

### Fixed

- **`cloud_options` on non-`file_path` sources** — passing `cloud_options` to a
  source format that ignores it (e.g. `image_bytes`) now emits a `UserWarning`
  instead of silently dropping the credentials.
- **`normalize(out_dtype=...)` planned vs. produced dtype** — the planner
  honored `out_dtype` but execution always produced f32, so any `out_dtype`
  other than f32 tripped the dtype-contract guard. Normalization now casts its
  f32 result to the requested dtype, so `normalize(out_dtype="u8"|"f64")` works
  and the planned dtype always matches the produced dtype. (`out_dtype="preserve"`
  is rejected with a clear message — it is not meaningful for normalization.)

## [0.11.0] — 2026-07-10

### Added

- **`file://` URLs in the local read path** — `source("file_path")` now decodes
  `file://` URIs, not just bare local paths, routing local reads through the
  same `cloud::read_file` entry point as remote schemes.

### Changed

- **Cloud access is signed by default, anonymous is opt-in** — GCS now honors
  `CloudOptions(anonymous=True)` (via `with_skip_signature`) like S3 and Azure.
  The previously documented "anonymous-first" credential chain — which no
  provider actually implemented — is gone; requests are signed unless you
  explicitly opt into anonymous access.

### Fixed

- **Blob header validation** — untrusted `blob` VIEW-protocol headers are now
  validated with checked arithmetic and stride-span checks, rejecting malformed
  inputs instead of risking overflow.
- **Arrow C-Data import** — the array offset is respected and null-bearing
  arrays are rejected on import, fixing silently wrong reads of sliced/offset
  Arrow buffers.
- **Buffer deallocation** — owned buffers are freed with their original
  alignment, fixing a mismatched-layout deallocation.
- **Materializing kernels** now declare contiguous output, and `gamma`'s integer
  range is corrected.
- **Metrics** — the VOC 11-point AP denominator and the bootstrap PR-AUC
  estimator mismatch are fixed.
- **Panics eliminated** — axis reductions over 1-D buffers, and the
  `u8` RGBA→Lab alpha color path (which also now preserves dtype) no longer
  panic.
- **Rotation fusion** uses position-correct shapes, and batch re-folds seed from
  the post-source state.
- Colon-bearing local paths are read literally, and a dead `Cast` stride arm was
  removed.
- **Builds on current stable Rust** — the transitive `ethnum` dependency is
  pinned to 1.5.3, which drops the `TryFromIntError` transmute that failed to
  compile (`E0512`) on newer `rustc`. The stack builds cleanly against polars
  0.54.4 / pyo3-polars 0.27 / pyo3 0.28.

---

## [0.10.0] — 2026-06-13

### Added

- **`preserve_dtype` on intensity ops** — `scale`, `clamp`, and
  `adjust_brightness` accept `preserve_dtype=True` to cast the `f32` result back
  to the input dtype (e.g. `u8` in → `u8` out) instead of promoting to `f32`.
  Requires a known input dtype and is mutually exclusive with `out_dtype`.
- **Graph-level error policy** — `Pipeline.on_error("raise" | "null" |
  "null_with_message")` (mirrored on `LazyPipelineExpr`). Failing rows null all
  outputs, optionally attaching a reserved `_error` message field, instead of
  failing the whole batch.
- **Full `LazyPipelineExpr` method parity** — every chainable `Pipeline`
  operation can now be called directly on the lazy expression returned by
  `.cv.pipe(...)`, without wrapping each step in its own `Pipeline`. Guarded by
  `test_lazy_pipeline_method_parity`.
- **Shape references for rasterization** — `rasterize(shape=<expr>)` and
  `source("contour", shape=<expr>)` size the raster canvas from a referenced
  pipeline's output `[H, W]`, implemented end-to-end.
- **Rotation parameters** — `rotate()` accepts `interpolation` and
  `border_value`.

### Changed

- **Single schema authority in view-buffer** — the Python planner no longer
  keeps a parallel contract table. Per-op output domain, dtype, rank, and
  channel count are read from view-buffer's declarative rules
  (`OutputChannelRule` / `OutputRankRule`). The `OPERATION_CONTRACTS`,
  `AlphaMode`, and `NdimEffect` structures were removed.
- **Plan == exec** — planned dtype/shape now matches execution for promoting
  binary ops (e.g. true division → `f32`) and for explicitly-typed 16-bit image
  sources.
- **Rotation unified through the affine path** — arbitrary-angle `rotate()`
  routes through `ComputeOp::RotateAffine` → `apply_affine_warp()`; zero-copy
  90/180/270 fast paths are preserved.
- **Portable CI / wheel builds** — abi3 wheels on Linux and a fixed macOS build.

### Fixed

- `cloud_options` now round-trips for remote `file_path` sources.
- Assorted `ruff` / `clippy` lint findings; CI excludes flaky network-marked
  tests.

### Performance

- Kernel fusion extended: casts fold into the `FusedKernel`
  (`u8 → cast(f32) → scale → clamp → relu` runs as one pass); `rotate` →
  `warp_affine` chains fuse.
- Vectorized `erode`/`dilate`, `convolve2d` (interior/border split), and
  separable Gaussian blur; Canny rewritten without `atan2` (bit-exact).
- Remote `file_path` sources prefetch concurrently per batch.
- Graphs compile once into a process-wide cache (`graph/compiled.rs`), so repeat
  invocations (e.g. per streaming morsel) pay only a hash lookup.

---

_Releases earlier than 0.10.0 predate this changelog; see the git history for
details._

[0.18.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.18.0
[0.17.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.17.0
[0.16.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.16.0
[0.15.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.15.0
[0.14.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.14.0
[0.13.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.13.0
[0.12.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.12.0
[0.11.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.11.0
[0.10.0]: https://github.com/heshamdar/polars-cv/releases/tag/v0.10.0
