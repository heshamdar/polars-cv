# Changelog

All notable changes to **polars-cv** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **`ContourMatcher` could not read an image-bytes mask column.**
  `metrics/_matching/_contour.py` decided the source format itself and mapped
  every `Binary` column to `"blob"`, so a PNG mask failed with "Invalid blob
  magic bytes" — while the same column read fine through `source("auto")`,
  which checks the VIEW magic bytes and falls back to `image_bytes`. Metrics
  now uses `source("auto")` and contributes only the one fact the Polars schema
  settles that Rust cannot infer at plan time: a nested column's leaf dtype.
  The `_gt_h`/`_gt_w` derivation likewise branches on the Polars dtype (which
  decides whether a cheaper native path exists) rather than on a format string.

  A `String` mask column now routes to `file_path` through `auto` instead of
  raising in Python, so a column of paths to mask files works where it
  previously did not.

- **`Pipeline.to_graph()` silently discarded `on_error` and `on_null_param`.**
  Three functions built a `Pipeline` field by field — `_clone`,
  `_create_sub_pipeline`, and CSE's `_create_shared_node` — and the second
  carried 11 of the 14 fields. `PipelineGraph._to_dict` reads the per-row
  policies off the *node* pipeline, and `to_graph()` makes its sub-pipeline the
  graph's only node, so a public
  `Pipeline().source(...).on_error("null").to_graph(col)` executed under
  `"raise"`. CSE escaped it by accident: the hoist unions over all nodes and the
  originals kept their policy.

  All three now inherit the whole state through one `_copy_state_from`, driven
  by a `_STATE_COPIERS` table, and override only what they mean to change — so a
  new field is carried by default rather than by remembering three call sites.
  `test_pipeline_state_copy_is_complete` fails if the table and `__init__`
  disagree in either direction.

- **A vector-domain reduction was accepted at plan time and rejected at
  execution.** *(see also the two notes below on what this deliberately does
  not extend to, and on fused rotate sizes.)* `GraphStep::input_domains` declares that binary ops and
  reductions take `[Buffer, Vector]`, and the Python planner validates against
  it — but execution re-derived the precondition by hand at ten sites, each with
  a hardcoded `"<Step> requires Buffer"` string, and `NodeOutput::as_buffer()`
  returns `None` for a vector. So `extract_shape().reduce_sum()` planned clean
  and failed the row. The ten sites are now one `step_buffer_operand` that reads
  the declared contract, materializing a 1-D buffer for a vector the step admits
  — the equivalence the planner's own docstring already claimed. A domain the
  contract does *not* admit is still refused.

- **An unmappable dtype string is no longer typed `UInt8`.**
  `dtype_str_to_polars` ended in `_ => DataType::UInt8`, guarded only by a
  source-scanning ratchet, with a comment conceding that any unmatched string
  was silently typed. It now parses through `DType::from_short_name` and returns
  `PolarsResult`, and the Polars spelling is an exhaustive match on `DType`, so
  an eleventh dtype fails to compile rather than defaulting.

- **`deny_unknown_fields` now covers the nested wire-format structs.** serde's
  attribute does not descend, so closing `GraphNode` left `SourceSpec`,
  `OutputSpec`, `UnifiedGraph` and `GraphKwargs` accepting anything. That
  mattered most on `SourceSpec`, which carries `allowed_roots`: a misspelled key
  deserialized to `None`, i.e. no path sandbox, silently.


- **`gcs_bearer_token` on an `s3://` path now errors instead of being ignored.**
  S3 signs with SigV4 and cannot use an OAuth bearer token; the value was
  accepted and silently discarded, which is the bug `token_command` was already
  rejected for one field over.

- **S3 object keys are no longer double percent-encoded.** `read_s3` passed
  `url.path()` — already encoded — to `Path::from`, which encodes again, so
  `s3://bucket/a b.png` requested the key `a%2520b.png`. Reads now use the raw
  prefix, which also makes polars-cv agree with `pl.scan_parquet` on the same
  URL.

- **A misspelled storage option is still rejected.** polars'
  `parse_untyped_config` silently drops keys it does not recognise ("Silently
  ignores custom upstream storage_options"), so delegating naively would have
  turned `unknown GCS storage option 'x'` into a request that quietly did the
  wrong thing. Keys are validated against each backend's own vocabulary before
  the options are handed over, lower-cased exactly as polars does so the
  accepted set is neither wider nor narrower than what polars would act on.

### Changed

- **`ViewExpr::apply_op` reads each image op's declared dtype rule.** The
  `Canny` and `HistogramEqualize` arms hardcoded `DType::U8` and
  `strides: None` — a second copy of a rule `image.rs` already states as
  `OutputDTypeRule::Fixed(DType::U8)`, in the module the planner reads the
  first copy from over the FFI. The dtype-preserving group beside them
  restated `PreserveInput` the same way. All three collapse into one arm that
  resolves shape, strides and dtype the way the `Filter` and `Compute`
  neighbours already did. Still exhaustive, so a new `ImageOpKind` has to say
  where it belongs. A new `expr.rs` unit test asserts the property for every
  image op rather than for the two names.

- **`display.py` reads the generated dtype table.** It carried a third Python
  copy of the VIEW wire codes, guarded by a regex over its own source — the
  weakest guard kind, and one a reformat defeats. `gen_dtype_names.py` now
  emits `NUMPY_BY_WIRE_CODE` from `dtype_table!`, and the regex guard went with
  the duplicate it existed for.

- **The matcher conformance sweep is derived** from
  `metrics._matching.__all__` with a floor assertion, rather than naming its
  three subjects literally.

- **`Pipeline.scale_contour` takes `origin=`.** It hardcoded
  `ScaleOrigin::Centroid` and exposed no way to choose, while the same-named
  `.contour.scale` accessor exposed `origin=` defaulting to `"origin"` — so a
  square scaled by 2 landed at (1,1)-(5,5) through the pipeline and (4,4)-(8,8)
  through the namespace. Both surfaces now take the parameter and agree for
  every explicit value.

  The defaults still differ: `scale_contour` keeps `"centroid"` and
  `.contour.scale` keeps `"origin"`, because each is what that surface has
  always done and changing either would silently move existing output.
  Aligning them is a separate, deliberate decision; it is pinned in
  `tests/test_known_gaps.py` until it is taken.

- **The generated type stub derives its imports.** `gen_lazy_stub.py` carried a
  hardcoded `polars_cv._types` import list, so a `Pipeline` parameter annotated
  with any type it did not happen to name produced a stub referencing an
  undefined symbol — and `test_lazy_stub_is_current`, being a
  regenerate-and-diff, had the same defect on both sides and could not see it.
  `scale_contour(origin=)` is the annotation that proved it. The block is now
  derived from what the rendered stub references, read from `_types`' own
  definitions.

- **The rotation matrix has one implementation again.** Affine fusion
  transliterated `AffineParams::from_rotation` into Python line for line, so a
  `rotate()` got its matrix from Rust when it stood alone and from Python when a
  neighbouring op made it fusible. The two had already drifted — Python
  normalised `angle % 360` where Rust did not, and rounded the expand bounding
  box half-to-even against Rust's half-away-from-zero — and nothing compared
  them: the test that appeared to cross-check compared Python against a third
  copy of the same arithmetic in `test_affine_builder.py`. The planner now reads
  the matrix through a new `rotate_affine_params` FFI, the test helper reads the
  same authority, and `test_removed_surfaces.py` rejects trigonometry
  reappearing in the fusion helper.

- **`GraphStep::input_domains` is exhaustive.** It was the one contract method
  with a `_ =>` catch-all, so a new multi-domain variant would silently have
  been given a single domain — in the method `CLAUDE.md` names as the authority
  for accepted input domains.

- **Staleness of the compiled extension is detectable within a release cycle.**
  `test_compiled_plugin_is_not_stale` compared release *versions*, and both
  sides read the same `Cargo.toml` literal, so they agreed throughout the entire
  window in which Rust gets edited without a rebuild — it could only fire across
  a version bump. A new `build.rs` bakes a content hash of both crates' sources
  into the extension, `build_info()` recomputes it from the working tree, and
  `test_compiled_plugin_matches_the_rust_sources` compares them. This matters
  because 53% of the suite is gated on a `.so` merely existing, of any age: a
  stale one does not skip those tests, it runs them against old Rust.

- **`.pre-commit-config.yaml` moved to the repo root**, where `pre-commit`
  actually resolves it. It sat in `polars-cv/`, so the hook was never loaded —
  while `tests/AGENTS.md`, the `pyproject.toml` marker description and ten
  module docstrings all justified themselves with "the pre-commit hook enforces
  this". The config's own contents gave it away: `cargo fmt --all` and
  `uv run --directory polars-cv` are both written for the root.

- **Guards that read as coverage and enforced less.** `test_no_local_png_factories`
  documented two fixture names and grepped for one; the unchecked half had
  drifted into eleven copies, each having dropped conftest's Pillow-missing
  skip. Three sweeps in `test_schema_parity_chains.py` discarded their
  `plan_or_reject` results, and a fourth asserted only things its own types
  guarantee. `test_verify_script_covers_every_ci_check` searched the raw
  `verify.sh` including comments, so commenting out the structural lane passed
  both it and its sibling. The `Matcher` protocol was pinned by
  `isinstance`, which checks attribute presence and never signatures. Two
  guard-fixture modules were in the structural lane but pinned by neither lane
  mechanism.


- **The remote source stopped opening one connection per file.** `read_http`
  built a `reqwest::Client` — which *is* the connection pool — inside the
  per-file read, and `read_s3` / `read_gcs` / `read_azure` each built a fresh
  object store per file. The 0.20.0 benchmark counted the result exactly: 1.00
  requests per connection, so every image paid a TCP handshake and, off
  loopback, a TLS handshake. Under the streaming engine it paid them again on
  every morsel, because nothing scoped to a plugin call can outlive one.

  The three cloud schemes now go through `polars-io`'s `build_object_store`,
  which keeps a process-wide store cache; `http(s)://` keeps its own pooled
  client, because polars deliberately does not cache HTTP stores (that suits a
  few large Parquet files, not many small images). Measured over three
  successive plugin calls on 64 URLs: 16 connections for the first call — one
  per concurrent worker rather than one per file — and none at all for the
  calls after it.

  Delegating rather than reimplementing is also what makes caching a
  *credentialed* store safe, which is the question 0.20.0 measured but declined
  to answer. polars keys the cache on the bucket **and** the credentials, and
  `exec_with_rebuild_retry_on_err` rebuilds the store with cached credentials
  cleared and retries once on any object-store error, swapping the new store in
  through the `Arc` the cache holds.

- **Fetch concurrency is a global budget, not a per-call constant.** It was 16
  per plugin invocation, and the streaming engine invokes the plugin once per
  morsel across threads — so the real number of in-flight requests was (morsels
  in flight × 16), bounded by nothing and settable by no one. Every request now
  takes one permit from polars' own semaphore, so our fetches and polars' own
  scans share one bound: `POLARS_CONCURRENCY_BUDGET`, defaulting to
  `max(thread count, 10)`.

  The chunked fan-out went with it. It spawned 16 scoped OS threads and joined
  *all* of them before starting the next chunk, so the slowest file in a wave
  stalled the wave; `buffer_unordered` starts the next file as soon as any one
  finishes. `fetch::DEFAULT_CONCURRENCY` and the `max_concurrency` parameter are
  deleted — a per-call knob beside a global budget would be a second authority
  for one number.

### Removed

- **`DomainOp` (view-buffer).** A public trait with zero implementors, a doc
  example implementing it for a type that does not exist, and an entry in
  `view-buffer/AGENTS.md` describing it as a live part of the op contract.
  Domain dispatch actually lives on `GraphStep` in the plugin. It was also the
  only reason `ops/traits.rs` imported `NodeOutput` — a graph concept reaching
  into the engine's trait module — so deleting it closed that too. Pinned by
  `test_removed_surfaces.py`.

- **`match_detections(strategy=)` on `.contour` and `.bbox`.** Typed
  `Literal["greedy"]`, rode the wire as `ContourKwargs.strategy`, and selected
  nothing — the greedy policy is unconditional in the shared `match_from_matrix`.
  The contour entry point checked the value and discarded it; the bbox one did
  not check at all, so `strategy="hungarian"` was accepted, ignored, and matched
  greedily. Guarded by `test_removed_surfaces.py`.

### Notes

- **Binary ops still refuse a runtime vector, and that is deliberate.** The
  reduction fix above coerces a runtime `NodeOutput::Vector` to a 1-D buffer,
  but only for steps whose published output shape derives from that operand
  alone. `Binary` declares it accepts `vector` — it must, because
  `perceptual_hash` plans as one — yet a phash *executes* as a buffer, so the
  only operands that ever arrive as a runtime vector are `extract_shape` and
  `label_reduce`. Coercing those would broadcast against the other operand
  while the planner had already published the *left* operand's rank:
  `extract_shape().add(image)` would plan rank 1 and execute rank 3. It raises
  instead, naming the reason.

- **`rotate(expand=True)` output dimensions can differ by one pixel from
  0.20.0** for a small fraction of inputs (~1 in 5000 over a 200k-case sweep),
  now that the fused path reads the engine's matrix rather than recomputing it.
  The cause is precision, not convention: `AffineParams::from_rotation` takes
  `angle_deg: f32` where the Python transliteration used f64. The new value is
  the one an unfused rotate has always produced, so this makes the two paths
  agree rather than changing what a rotate means.

- **`to_graph()` now emits `shape_asserted: true`** where it emitted `false`,
  because the sub-pipeline it builds inherits `_asserted_dims` along with the
  rest of the state. It only selects which of two error strings
  `validate_output_schema` prints, and the new value is the correct one — it is
  what `.cv.pipe()` already emitted — but `graph_json` is the compiled-graph
  cache key, so affected pipelines mint a new cache entry once.

- Ambient AWS configuration is now honoured: `build_aws` reads `~/.aws/config`
  and `~/.aws/credentials` for a region and keys when the options do not supply
  them, and issues one `HEAD` per bucket to discover an unset region. Neither
  happened before, so a machine with an unrelated AWS profile may sign a request
  that previously went unsigned — pass `anonymous=True` for a public bucket.
- Store-build and read errors now carry polars' context.
- `polars` gains the `aws`/`gcp`/`azure` features, which pull in polars-io's
  `cloud` stack (and transitively `file_cache`, `serde_json`, `rand`).

## [0.20.0] — 2026-08-13

### Added

- **Every `.contour` accessor now takes a set of contours, not just one.** A
  column may be a `CONTOUR_SCHEMA` struct per row or the `CONTOUR_SET_SCHEMA`
  list of them that `extract_contours()` produces; the result is wrapped to
  match, so `area()` returns `Float64` for the first and `List(Float64)` for the
  second, one entry per contour. Fifteen of the eighteen accessors called
  `parse_contour` directly and rejected a set outright, which meant the
  namespace could not read the column its own pipeline produced.

  Two-operand accessors (`iou`, `dice`, `hausdorff_distance`) **broadcast**: a
  set on one side and a single contour on the other yields one result per
  contour, whichever side the set is on. A set on *both* sides raises rather
  than guessing — it could mean the N×M matrix (`pairwise_iou`) or an
  index-wise pairing (`.explode()` one side), and those are different answers.
  The set-level accessors (`pairwise_iou`, `match_detections`, `label_reduce`)
  run the rule backwards through the same `parse_contour_set` repack, so a lone
  contour is read as a set of one.

  The arity is **one value, read from the column dtype**, and it drives both
  halves of every accessor: `src/geom_arity.rs` wraps the declared element type
  for `output_type_func` and the produced results with the same `Arity::wrap`,
  and `contour_accessor!` emits both from a single `-> <elem>` declaration. That
  matters because the two halves are otherwise unconnected — declaring
  `Float64` while building `List(Float64)` is exactly the plan/exec divergence
  `test_schema_parity_namespaces` exists to catch, and per-accessor list
  handling would have been fifteen chances to introduce it. The case table there
  is now swept in *both* arities, so an accessor that skips the macro fails
  rather than shipping a schema its data contradicts.

- **`Pipeline.assert_shape(dims=[...])`** — the positional spelling, and the one
  that pins the output **rank** alongside the per-dimension sizes. This is what
  makes an assertion reach the published schema for a `list`/`array` source,
  whose rank is not knowable until execution: the `height=`/`width=`/`channels=`
  form set the hints, but `expected_shape` only publishes at rank 3, so the
  assertion changed nothing and the `array` sink went on refusing the query.

- **`.contour.ensure_winding(direction=)` and `.contour.scale(origin=)` accept a
  Polars expression**, resolved per row like every other non-structural
  parameter. Per-row eligibility is decided by whether a value affects the
  output's shape, rank or dtype — not by its type — and neither of these does:
  the column stays `List(Struct(CONTOUR_SCHEMA))` whichever winding or origin a
  row picks. They were nonetheless literal-only while `label_reduce`'s
  `reduction` and `region_mode`, two enums of the same kind in the same
  namespace, were already per-row.

  Worse, the rejection asserted the opposite of the truth — *"'direction' is
  structural (it fixes the output shape/rank/dtype at planning time)"* — for a
  parameter that fixes none of them. Both now ride `_ArgBinder`, like `scale`'s
  own `sx`/`sy`: the expression is appended as a plugin input, named in
  `input_slots`, and read per row against the same `NAMED` table a literal is
  validated against, so moving the check to execution does not lose it. Routing
  them through `contour_row` also brings both under the shared null-parameter
  policy, so `on_null("null")` now applies to them.

  `test_non_structural_geometry_enums_accept_an_expression` reads the geometry
  namespaces' live signatures, so a new non-structural enum has to be plumbed
  for per-row use or exempted with the structural reason.

- **The remote source has a benchmark.** Every scenario in the suite was handed
  bytes that were already in memory, so the `fetch.rs` / `cloud.rs` stage — the
  one every `s3://`, `gs://`, `az://` and `http://` path goes through — had no
  coverage at all. `benchmarks/scenarios/remote_source.py` measures it against a
  loopback HTTP server, selectable from the regression suite as
  `--scenarios remote`.

  Loopback rather than a real bucket because the transports differ in *signing*,
  not in structure: each lands in `cloud::read_file`, which builds a client and
  issues one GET per file, and each is batched by the same `fetch::prefetch`. A
  wide-area measurement is dominated by the link and hides what the plugin
  costs; a `--latency-ms` knob injects a synthetic delay for when modelling the
  link *is* the point. Real S3/GCS numbers need credentials and a bucket, so
  they cannot be a committed benchmark.

  It reports three timings over one corpus whose differences isolate a stage,
  and one thing that is not a timing: **requests per connection**, counted by
  the server. On this code that ratio is 1.00 — a fresh connection for every
  file, because `read_http` constructs a `reqwest::Client` inside the per-file
  read and the three cloud backends each build a store per file. On loopback
  that costs about 0.5 ms/file; against a real endpoint it is a TCP *and* TLS
  handshake per file. Measured and reported rather than fixed: caching a
  credentialed store is a correctness question before it is a performance one.

### Fixed

- **`assert_shape` rejects a contradiction where it is written, instead of
  blaming the plugin at `collect()`.** `resize(224, 224).assert_shape(height=999)`
  was accepted, published as the output schema, and reported much later by
  `validate_output_schema` as *"the planner's shape contract disagrees with the
  Rust implementation"* — the plugin taking the blame for a value the caller had
  typed three lines earlier. An assertion that contradicts a dimension the
  planner already knows now raises at the call, naming both values and the op
  that established the known one. When a divergence *does* survive to execution,
  the message distinguishes the two cases: an output whose shape was asserted
  says so and points at the assertion, while an inferred one keeps the
  contract-bug wording, which is correct there. The two are told apart by a
  `shape_asserted` flag on the output spec, tracked per dimension and cleared
  whenever the schema fold recomputes it — the `rasterize(shape=<node>)` canvas
  is deliberately *not* counted, since it comes from another node's inferred
  hints.

  The `height`/`width`/`channels` keywords are also rejected once the rank is
  known to be anything but 3. The hints are positional — they are dimensions 0,
  1 and 2 — so after a `transpose([2, 0, 1])` dimension 0 is the channel axis
  and calling it `height` is a lie. `dims=` is the honest spelling there.

- **The `array` sink's error names remedies that exist.** It advised
  `.resize()` / `.assert_shape()`, and the source that reaches it most often is
  a `list`/`array` column — whose shape genuinely is not knowable until
  execution, and which therefore got the same advice back after following it.
  `.resize()` supplies height and width but never the channel count. The message
  now names the missing dimensions and what each of the three remedies actually
  supplies, and is written once per side rather than at four sites. The
  unreachable copy in `encode.rs` became an internal-invariant error rather than
  a fourth restatement of user advice, since `dtype_for_output` reads the same
  two fields and refuses first.

- **`.contour.ensure_winding()` and `.contour.scale(origin=)` reject a value
  they do not recognise instead of guessing one.** These were the only two
  user-facing string parameters in the plugin that did not read a `NAMED`
  table. Both parsed by hand and ended in `_ => <default>`, so a spelling the
  parser did not know was answered with a plausible one:
  `ensure_winding("CW")` returned *counter*-clockwise — the opposite of the
  request — and `scale(origin="top_left")` scaled about the centroid. Silently,
  in both cases; a capitalisation that does not match is not an exotic input.

  `ScaleOrigin` and `Winding` now go through `named_variants!` + `registry!`
  like every other enum, which is also what makes them checkable: they are
  surfaced over `enum_variants`, covered by `every_named_enum_is_registered`,
  and demanded by `test_every_rust_enum_is_parity_checked` — hence
  `_types.ScaleOrigin` and `_types.Winding`, which replace the `Literal[...]`
  annotations that used to shadow them. The long `"clockwise"` /
  `"counterclockwise"` spellings the parser has always accepted are kept, as
  aliases inside `NAMED` rather than a second table.

  One behaviour change beyond the rejection: a `contour_scale` graph that omits
  `origin` entirely now defaults to `Origin` rather than `Centroid`, matching
  what the Python signature has always declared. Every pipeline built through
  `.contour.scale()` already sent an explicit value, so this is reachable only
  by hand-writing the plugin kwargs.

- **`numpy_from_struct()` no longer reads `"u8"` as `uint64`.** Its dtype
  allowlist was hand-typed and admitted numpy's *character codes* alongside its
  spelled names. numpy reads `"u8"` as uint64 and `"i8"` as int64 — the
  opposite of what those strings mean everywhere else in this project, where
  they are `dtype_table!`'s short names for uint8 and int8. So a caller
  hand-building a struct with `dtype="u8"` — the spelling `.cast("u8")` uses —
  got a uint64 reinterpretation of the bytes: wrong values, and either a wrong
  shape or a `reshape` error far from the cause. The function is exported from
  the package root and takes a plain dict, so this needed no compiled plugin to
  reach. The list also admitted `bool`/`b1`/`f2`, which the engine cannot
  produce.

  The accepted names are now generated into `python/polars_cv/_dtype_names.py`
  from `dtype_table!` by `scripts/gen_dtype_names.py`, guarded by
  regenerate-and-diff. Character codes are rejected; the numpy/torch sink has
  never emitted them.

- **The contour struct layout is declared once, and checked against what the
  engine emits.** Three places spelled it out: `contour_struct_dtype()` in
  `src/graph/encode.rs` (the only one execution reads), the public
  `CONTOUR_SCHEMA`, and a private copy in `metrics/_matching/_contour.py` that
  did not import the public one. A fourth site, `tests/test_contour_source.py`,
  shadowed `CONTOUR_SET_SCHEMA` with a local alias of the constant it had just
  imported. The private copy and the shadow are gone; `CONTOUR_SCHEMA` remains,
  as the public mirror.

  Nothing had been checking any of them. The two Python copies agreed with each
  other, which is not the property that matters — a field renamed or reordered
  in Rust would have left both still agreeing and both wrong, and the metrics
  matcher casts a real sink output to its copy, so the divergence would have
  surfaced as a cast error inside a detection pipeline rather than at the schema.

  `test_contour_schema_is_what_rust_emits` runs an extraction through the plugin
  and compares `CONTOUR_SET_SCHEMA` to the dtype Polars actually returned, so it
  is behaviour-observed rather than two Python constants compared to each other.
  It was watched failing against a deliberately reordered `contour_struct_dtype()`.

- **A `DetectionTable`'s dtypes are declared, not invented at the empty path.**
  `DETECTION_COLUMNS` and `IMAGE_META_REQUIRED` listed column *names* with no
  dtypes, so the only place the dtypes existed was a hand-written literal inside
  `_empty_detection_table()` — twelve `pl.` types that nothing connected to what
  a populated match returns. The two frames meet: an all-background image yields
  the empty table and a normal one does not, and a caller concatenating them
  would have discovered a disagreement at the concat rather than at the schema.

  They are now `DETECTION_SCHEMA` / `IMAGE_META_SCHEMA`, name→dtype maps that
  `_empty_detection_table()` builds from directly.
  `test_matcher_schemas_match_the_declaration` runs `ContourMatcher`,
  `BBoxMatcher` and `PreMatchedAdapter` on real inputs and compares each
  result's schema to the declaration, so the declaration records observed
  behaviour rather than an intention; a new matcher is added to `_PRODUCERS`,
  which is what gets its output checked at all. Watched failing by widening one
  declared dtype — the three matchers rejected it while the empty path, which
  builds from the declaration, could not.

  `from_matched` still validates **names only**. It accepts caller-assembled
  frames, and rejecting a `UInt64` `det_idx` that Polars compares happily
  against a `UInt32` would break working pipelines for no benefit; what the
  matchers produce and what a caller may supply are different claims.

- **`statistics()` and `statistics_lazy()` read one table instead of four.**
  Each carried a `valid_stats` set *and* a five-arm `if/elif` dispatch — the
  same list of statistics written out four times — and both dispatches ended in
  `else: continue`. So the two halves could disagree in the direction that says
  nothing: a name the validator accepted but the chain did not know was dropped
  from the output silently, producing a struct with a missing field rather than
  an error.

  Both now build their nodes through `_stat_nodes()`, which reads
  `_STAT_REDUCERS` for the accepted names *and* the dispatch, so the two cannot
  come apart. The methods differ only in the alias prefix and in whether they
  sink, which is what they always differed in.
  `test_stat_reducers_are_all_pipeline_methods` rejects an entry naming a
  method that does not exist or one that is not a reduction — watched failing
  for both.

  `_require_concrete_sink_dtype` also stops spelling its own `("list",
  "array")`. That vocabulary is now `SINKS_WITH_TYPED_ELEMENTS` in `_types`,
  beside the `SOURCES_RESOLVED_FROM_COLUMN` the same function already read from
  there.

- **"Channels from rank" is derived in one place.** The mapping
  `3 => Some(shape[2]), 2 => Some(1), _ => None` appeared character-for-character
  at three sites, each feeding the same `ImageCodec` check: `execute.rs` (the
  encoder), `graph/decode.rs` (the planner), and `to_dynamic_image` in
  view-buffer's `interop/image.rs` — whose own comment said "a second copy here
  is how the two would come to disagree" while being exactly that.

  Both now call `ImageCodec::check_shape`, which takes the *shape* and derives
  the rank and channel count itself, so neither caller can derive them
  differently. `image_metadata.rs` keeps its own, different mapping: it answers
  `.cv.channels()` for buffers of any rank and its callers need a number rather
  than a maybe, so it interprets ranks 0 and 1 that the codec check must reject.
  A comment there now says so, rather than leaving it looking like a third copy.

- **Three error/null policies stop being spelled by hand in Python, and
  `enum_variants` loses its last hand-written arm.** `("raise", "null")` and
  friends appeared at five Python sites covering *three* different policies —
  `RowErrorPolicy` (3 values), `NullParamPolicy` (2) and the `file_path` /
  `read_bytes` fetch policy (2). Over-listing a value errors loudly at the
  boundary; under-listing one rejects a value the plugin would have accepted,
  silently, forever.

  The fetch policy was not even an enum: `parse_on_error` matched two string
  literals and hand-wrote `(expected 'raise' or 'null')`. It is now
  `FetchErrorPolicy`, and all three carry `named_variants!` tables mirrored by
  `_types.RowErrorPolicy` / `NullParamPolicy` / `FetchErrorPolicy`, which the
  five sites read.

  Reaching them required lending the plugin crate the mechanism rather than
  cloning its exception: `named_variants!` and `registry!` are exported, and
  the plugin declares `PLUGIN_REGISTRY`, which `enum_variants`/`enum_names`
  chain onto the engine's. Registering an enum is therefore the same act as
  getting it checked — `test_every_rust_enum_is_parity_checked` reads
  `enum_names()` and demands a Python mirror for each.

  `BinaryOp` came along: its name table lived in the plugin crate purely by
  habit, which is what made it the one enum needing a bespoke FFI arm and an
  exemption from that test. `BINARY_OPS` is now `BinaryOp::NAMED` beside the
  enum in view-buffer, registered like everything else — so a new `BinaryOp`
  variant no longer compiles until it is named. The `BINARY_OP_ENUM` constant
  and the `enum_variants` special case are deleted.

  Two new Rust tests cover the one thing the parity test cannot see: these
  policies have *two* mechanisms describing one vocabulary, serde's
  `rename_all` reading the wire and `NAMED` publishing it.
  `row_error_policy_names_match_serde` and its `NullParamPolicy` twin compare
  them, so a rename on one side alone can no longer leave Python confidently
  sending a value the graph cannot parse.

- **Four guards that only asserted an absence now also assert a presence.** A
  test whose every assertion is "this string is not there" passes just as
  happily when the thing it guards has been renamed, moved or deleted — it
  stops distinguishing "the pattern is gone" from "the code is gone". Each was
  watched failing after the positive half was made to fail:

  - `test_enum_validation_uniform` now builds each parameter with a *real*
    variant as well as a bogus one, and asserts that variant is one
    `enum_variants` actually publishes. Writing it caught a hard-coded
    `"dhash"` that `HashAlgorithm` does not have.
  - `TestParamPolicyRatchet` asserts `execute.rs` still mentions `resolve_usize`
    and `params::get` before ratcheting against two idioms that use them.
    Renaming the resolver used to leave the scan green.
  - `test_domain_vocabulary_declared_once` now checks that a wrong-domain op
    still raises and that the reported domain is a `_types.Domain` member.
    Deleting the domain check entirely used to pass every assertion.
  - `test_anti_alias_is_gone_from_the_type_stub` requires the stub to declare
    `rasterize` first — an empty `lazy.pyi` used to satisfy it.

- **The documentation is built by CI.** `docs.yml` was `workflow_dispatch:`-only
  and ran `mkdocs build --clean`, so nothing checked the docs on a push or a PR
  and a warning did not fail the build. A broken link or a nav entry pointing at
  a deleted page was found by whoever next deployed by hand, if at all.

  A `docs` job in `ci.yml` now runs `mkdocs build --strict` on every push and
  PR, the deploy workflow uses `--strict` too (so it cannot ship a site CI would
  reject), and `scripts/verify.sh` runs the same command — which
  `test_verify_script_covers_every_ci_check` now requires, so the three cannot
  come apart. The job is in `ci.yml` deliberately: that test reads only that
  file, so a check placed anywhere else is one it cannot see. Watched failing
  against a nav entry naming a missing page.

  `docs/panic-audit.md` joins the nav under a Development section. It was the
  one page in `docs/` absent from `nav`, i.e. built but unreachable from the
  site. (Not a `--strict` blocker, as it happens — mkdocs reports an
  out-of-nav page at INFO, not WARNING — but a page nothing links to is not
  published in any useful sense.)

- **The examples are executed, and two of them were broken.**
  `polars-cv/examples/` held fourteen files that nothing ran: no test, no
  workflow, and `ruff`'s `extend-exclude` skips the directory. They read as
  documentation while being unverified for the life of the directory.
  `test_examples_run` executes each as a subprocess and requires exit 0, marked
  `slow` and discovered through `tests/_discovery.py` so an empty sweep fails
  rather than passes.

  Its first run found two independent bugs in `08_ml_integration.py`:

  - `.assert_shape(channels=3, height=96, width=96)` was applied *after*
    `.transpose([2, 0, 1])`. Those keywords describe an `[H, W, C]` buffer, so
    asserting them on an already-transposed one made the planner publish
    `[96, 96, 3]` for something that executes as `[3, 96, 96]` — the query died
    at `collect()` with a plan/exec mismatch. The assertion moves before the
    transpose, where it means what it says.
  - The `array` sink was given no `shape=`, on the assumption that
    `.assert_shape()` supplies one. It does not, and neither does `.resize()` —
    only `.sink("array", shape=[...])` does, which the example now passes.

  `matplotlib` joins the dev group: ten of the thirteen runnable examples plot,
  and without it those ten would not skip but fail, since the runner asserts
  exit 0.

- **The documentation's vocabularies are checked against the code's.** Three
  places in `docs/` restate something the code owns, all hand-maintained and
  none compared to anything: `domains.md`'s table (now required to *equal*
  `_types.Domain`), `sources.md`'s `auto`-resolution table (its formats must
  exist in `SourceFormat`), and the methods called in the operations pages'
  Python examples (each must resolve to a real `Pipeline` /
  `LazyPipelineExpr` / namespace method).

  The last one is not the table-diff it looks like it should be: those pages
  name operations in prose headings, not tables, so a "rows ⊆ `OP_NAMES`" guard
  would have matched nothing and read as green forever. It reads the code
  blocks instead — what a reader actually copies. Its `_FOREIGN_METHODS`
  exemption list is itself guarded, and immediately caught three of our own
  methods wrongly exempted while it was being written.

  The Markdown readers live in `tests/_doc_tables.py` with committed fixtures
  in `tests/test_doc_table_fixtures.py`, which caught a lookbehind that
  excluded `df.method(` — the ordinary shape of a call, i.e. most of every page.

- **The structural guards run before a commit, not just in CI.** They check the
  shape of the codebase — registry parity, single-authority ratchets,
  removed-surface pins, the doc vocabularies — and read the source tree rather
  than exercising a pipeline, so the whole lane is ~650 tests in about five
  seconds. That is affordable in a pre-commit hook; the full suite is not.

  `-m structural` is declared in `pyproject.toml` and applied per module. A
  `structural-checks` hook joins `.pre-commit-config.yaml` (which previously
  stopped after `cargo clippy`), the `lint` job runs the lane without waiting
  on the build matrix, and `scripts/verify.sh` runs it as its own check so a
  failure says "the hook would have caught this" rather than arriving buried in
  the full suite's output. The hook passes `--no-sync`: without it `uv run`
  re-syncs the project and rebuilds the Rust extension on every commit.

  Two guards keep the lane from becoming decorative.
  `test_every_source_scanning_module_declares_its_lane` derives its set —
  anything importing `_discovery`, the mandated way to find files to scan, must
  declare `structural` or `slow` — so a new source-scanning guard cannot join
  without choosing. `test_the_structural_lane_is_not_empty` catches a renamed
  marker or a dropped declaration, which would otherwise make the hook select
  nothing and pass in milliseconds.

- **A check added to CI can no longer be missing from `verify.sh` unnoticed.**
  `test_verify_script_covers_every_ci_check` pinned a hand-written list in both
  directions, but a check added to CI and never added to that list was simply
  invisible: the local run kept reporting PASS while CI failed.

  `test_no_ci_check_is_missing_from_the_verify_script` reads every command in
  `ci.yml`'s `run:` blocks and requires each to be **classified** — either a
  check `verify.sh` must also run, or setup. An unclassified command fails, so
  a new checker cannot pass unnoticed. It proved itself immediately: adding the
  structural lane to CI made it fail until the command was classified.

- **`scripts/verify.sh` no longer rebuilds the extension in release mode.** The
  `mkdocs` check added above was written without `--no-sync`, so `uv run` synced
  the project on every verification — and syncing builds the plugin through the
  PEP 517 backend, which maturin runs at `--profile release`. That is a
  multi-minute re-optimisation of the whole polars stack, on every run, for a
  documentation build that never loads the extension. With `--no-sync` (what
  every other lane in the script already passes) the check takes about six
  seconds.

- **Review pass: five guards that could pass while checking nothing, and six
  false statements.** An independent review of the entries above found that the
  weakest part of the work was the part meant to be strongest.

  Guards that could be evaded, each now watched failing against the exact
  evasion:

  - `test_no_ci_check_is_missing_from_the_verify_script` read only the
    `run: |` block form, so `cargo test -p view-buffer` — a check `ci.yml` has
    run all along — was invisible to it. It also classified a whole line by its
    first token, so `cd polars-cv && deno lint` filed under `cd`, and exempted
    every `uv run python scripts/check_*.py` as setup. It now reads both `run:`
    forms, splits on `&&`/`;`/`||`, and has no blanket script exemption.
  - The same guard compared `verify.sh` as one string, so a *comment* naming a
    deleted check satisfied it. It now reads only `run_check` lines.
  - `_module_marks` walked the whole AST, so a `pytestmark` nested in a class
    counted as a module declaration: moving one cut the lane from 651 tests to
    547 while both lane guards reported it healthy.
  - The lane's "derived" set was a substring match on `"_discovery import"`,
    which `from tests import _discovery as _d` defeated. It is parsed now.
  - `test_the_structural_lane_is_not_empty` compared a count against a
    threshold with a margin of one. It names the load-bearing modules instead.
  - Nothing read `.pre-commit-config.yaml` at all, though the hook is the
    premise for the marker on ten modules. `test_the_precommit_hook_runs_the_structural_lane` does.

  And the exemption lists, which are how a guard gets quietly narrowed:
  `_FOREIGN_METHODS` is now **empty** — once the resolution learned about the
  metrics API and module-level `pl.col`, all 34 entries turned out to be
  resolvable or absent from the docs — and `_NON_SCRIPT_EXAMPLES` is guarded
  against exempting anything with a `__main__` block.

- **`test_benchmark_list_is_current` now exists.** `benchmarks/AGENTS.md` cited
  it as pinning the benchmark list. It had never been written — a guard's name
  in prose, which is the failure this whole effort is against. It exists, it
  reads `get_single_op_benchmarks()`, and it was watched failing on all three
  ways that sentence has been wrong: a wrong count, a drifted name, and a
  benchmark added to the code.

- **A plugin-crate `named_variants!` table left out of `PLUGIN_REGISTRY` now
  fails.** Exporting the macro gave the plugin the ability to declare
  vocabularies without giving it `every_named_enum_is_registered`, which scans
  view-buffer only. An unregistered enum was invisible to `enum_variants`, to
  `enum_names()` and so to the parity test — 128 tests passed with one sitting
  there. `every_plugin_named_enum_is_registered` is the missing half.

- **`_STAT_REDUCERS` is pinned by value, not just by shape.** The existing
  guard checked that each entry resolves to a reduction; it could not tell one
  reduction from another, so `"mean": "reduce_max"` — the plausible copy/paste
  error in the refactor that *removed* the explicit dispatch — passed the whole
  suite, because the fixtures paint constant images where mean equals max.
  `test_each_statistic_computes_its_own_reduction` uses a gradient on which all
  five differ.

- **The documented-method sweep covers every page.** It read five hand-picked
  pages of twenty-one, because `pl.col` and `pl.DataFrame` are module-level and
  the resolution only knew `pl.Expr`/`pl.DataFrame`, making `col` unresolvable
  almost everywhere. The fence reader also missed indented fences (admonitions,
  tabs, list items) and annotated ones (`title=`, `hl_lines=`) — both forms
  `mkdocs.yml` enables — so examples inside them were never checked. Fixtured.

- **`statistics(include=<generator>)` raised `IndexError` instead of a
  message.** `_stat_nodes` iterated `include` twice, so a one-shot iterable was
  exhausted by the validation pass and produced no nodes. Off the declared
  `list[str] | None`, but a raw `IndexError` from a private helper is not an
  answer. It is materialised once now.

- **The `zero_copy` benchmark scenario could never run.** It is offered in
  `ALL_SCENARIOS` and named in `benchmark.yml`'s input description, but
  selecting it died in `_aggregate_best` with `'BenchmarkResult' object has no
  attribute 'framework'`. The scenario returned its own `BenchmarkResult` — a
  different record that happened to shadow
  `benchmarks.frameworks.BenchmarkResult` — and `_run_once` appended those to a
  list of these without converting. Nothing ever exercised it.

  The local record is `IngestionResult` now, so two records meaning different
  things read differently, and `to_suite_results` is the one conversion between
  them. `_run_once` rejects any result that is not the suite's own record, so a
  scenario that skips the conversion fails at the boundary naming the scenario
  rather than several frames later naming a missing field.

- **The benchmark workflow measured a binary nobody installs.** `ci.yml` and
  `publish.yml` both clear `RUSTFLAGS` so `.cargo/config.toml`'s
  `target-cpu=x86-64-v3` is not baked in; `benchmark.yml` did not. It was
  therefore timing an AVX2/BMI2/FMA build while published wheels are built
  without those, so its numbers described a configuration no user runs. Cleared,
  with the same comment the other two carry.

- **A release tag is checked against the version being published.** maturin
  reads the version from `polars-cv/pyproject.toml`; a tag is only a label on a
  commit, and nothing compared the two. A release tagged `v0.20.0` cut before
  the bump landed would have uploaded `polars_cv-0.19.0` under it, or built
  every wheel and then failed at the upload with PyPI's "file already exists". A
  `check-version` job runs first, and both build jobs need it.

  It reads one manifest rather than re-comparing all four: the other three are
  pinned to `pyproject.toml` by `test_version_consistency.py`, which `ci.yml`
  runs on every push to main, so restating that comparison in the workflow would
  be a second authority for one fact. A `workflow_dispatch` run has no tag to
  check against, so it warns rather than skipping silently.

- **Documentation that described code as it is not.** `geometry.md` still called
  `scale`'s `origin` and `ensure_winding`'s `direction` literal-only "structural
  parameters" after both became per-row expressions, and omitted
  `label_reduce`'s two enums from the list of expression-capable parameters. It
  said nothing about every `.contour` accessor now reading a contour *set*.
  `image-ops.md` documented only the `height`/`width`/`channels` spelling of
  `assert_shape`, not `dims=` — the one that pins the rank, and so the only one
  that gets a `list`/`array` source to a fixed-shape `array` sink.
  `CONTRIBUTING.md`'s release steps claimed GitHub Actions publishes to TestPyPI
  and tests the install from there; `publish.yml` has never done either.

  Four remaining recommendations of `maturin develop --release` for the develop
  loop — in `build_info()`'s docstring and three `test_version_consistency.py`
  messages — now say `maturin develop`, finishing what the earlier docs pass
  started. The benchmark docs keep `--release`, which they need.

## [0.19.0] — 2026-08-09

### Added

- **Operation parameters that no code path reads are now rejected.** `OpSpec`
  is the documented exception to this crate's `deny_unknown_fields` rule — its
  params ride on `#[serde(flatten)]`, which serde cannot combine with
  unknown-field rejection — so the wire format could not refuse a parameter no
  operation understood, and nothing else did either. That is how `scale` and
  `clamp` carried an `out_dtype` for releases.

  Every parameter read now goes through `OpParams`, a tracking wrapper, and
  `resolve_op` rejects any name the arm did not touch. `resolve_op_inner` takes
  the op *name* and the wrapper, never the `OpSpec`, so an arm cannot reach the
  underlying map and read without recording — the compiler found two of `crop`'s
  reads doing exactly that when the signature was narrowed. Reads are recorded
  in a bitmask over the map's key order, so tracking allocates nothing on the
  per-row path. It fires at both ends of the boundary with no extra wiring:
  `resolve_op_from_json` backs `op_schema`, so a stray parameter fails in Python
  while the `Pipeline` is being built, and `CompiledGraph::compile` resolves
  every spec before any row executes. `OpParams::acknowledge` is the explicit
  escape for a parameter a layer above the arm consumes (`rasterize`'s
  `shape_ref`) or that a branch legitimately cannot use (`rotate`'s
  `interpolation` and `border_value` on the lattice rotations, which resample
  nothing).

- **A schema-parity matrix: plan-time dtype == execution-time dtype, swept.**
  Polars publishes an expression's dtype at planning time and again when the
  data arrives, and in this plugin the two are computed by different code —
  `dtype_for_output` from the hint bundle Python folds into the graph JSON, and
  `build_series_from_spec` from the buffers execution actually produced. The
  runtime half *reads the data* to decide: the typed list builder takes both
  the leaf dtype and the nesting depth from the first non-null row, and
  `resolved_output_specs` resolves every output's `"auto"` dtype from
  `inputs.first()`. Whether that agrees with the plan depended on which row
  happened to be first.

  Six new files under `tests/test_schema_parity_*.py`, on a shared harness
  (`tests/_schema_parity.py`) that asserts the planned dtype against the
  produced one under **both** engines, streaming first, and checks the engines
  agree. Rejection at build or plan time is an acceptable outcome for any cell;
  planning succeeding and execution then failing or producing something else is
  not. Every sweep carries an `assert_not_vacuous` check, because a sweep whose
  cells all reject is green and worthless.

  The axes:
  - **Row layout.** Nine patterns — all-null, null-first, null-last, a null run
    longer than a morsel, 64 alternating rows across a streaming batch
    boundary, and rows whose shapes differ from one another, chunked so the
    boundaries do not line up with the nulls. This is the axis that
    distinguishes "the plan" from "row 0".
  - **Operations.** All 71 chainable ops, driven from `tests/_op_cases.py` —
    the table `test_op_case_table_is_complete` already pins to
    `_chainable_pipeline_ops()`, moved out of `test_append_contract.py` so both
    read one copy. An op cannot join the library without getting a
    plan-vs-exec cell. Roughly forty had none before, including every
    reduction, all four morphology ops, every colour conversion, and all five
    histogram output modes.
  - **Sources and dtypes.** Every `SourceFormat` (including `auto`, the
    default, which had no plan-vs-exec test at all) and all ten `DType`s; only
    four were previously compared against data. Channel counts 1/2/3/4,
    including the 2-channel `GrayA` that `StripProcessRestore` produces and
    that nothing had ever sunk.
  - **Chains.** Composed effects asserted at every prefix, the steps
    `op_infer_shape` refuses, all eleven binary ops (read from Rust's
    `BINARY_OPS` registry), CSE-shared prefixes, two-root graphs, and affine
    fusion.
  - **`sink("array")`**, asserted in both directions: a knowable shape must be
    accepted and exact, an unknowable one refused while planning. Its
    dimensions are cross-checked against the shape the `numpy` sink reports for
    the same pipeline — two independent producers of one fact.
  - **The namespace accessors.** `.cv`, `.contour`, `.point` and `.bbox` are
    separate `#[polars_expr]` functions declaring their own output types, a
    second plan/runtime pair with no parity test — a gap
    `geometry/AGENTS.md` flagged. The case tables are completeness-asserted
    against the namespaces' real public methods.

  Three files carried their own copy of the plan-vs-exec assertion
  (`_assert_plan_matches_data`, `_planned_and_realized`,
  `_assert_plan_equals_exec`); they now call the harness, and pick up the
  streaming engine in the process.

  **Four divergences found, and fixed** (see *Fixed* below).

  Also recorded, without a bug marker: `expected_shape` is gated on rank 3, so
  a rank-1 or rank-2 pipeline can never auto-shape an array sink even when the
  planner has the dimensions. That is a deliberate conservative choice —
  publishing `[H, W, C]` for a rank-2 output is how `channel_select` once
  declared a schema execution could not produce — and the test asserts both the
  refusal and that the withheld dimensions were correct.

- **`allowed_roots`: a sandbox for path columns.** `source("file_path",
  allowed_roots=[...])` and `.cv.read_bytes(allowed_roots=[...])` restrict
  which locations a path column may read from. Until now neither surface
  sanitized paths — they read whatever the column named, any local file and
  any reachable URL, which is right for your own paths and wrong for paths
  that arrived as data.

  **Opt-in.** The default is unrestricted, so every existing pipeline is
  unaffected and an unrestricted source serializes byte-identically to before
  (`graph_json` is the compiled-graph cache key, so an always-emitted field
  would split the cache for everyone and change nothing else). Once you ask
  for a sandbox it denies by default: a path matching no entry is refused.

  One list covers local and remote — an entry that parses as a remote URI is
  matched as a URI prefix, anything else as a local directory. Splitting them
  is how a sandbox comes to cover the filesystem while leaving `s3://` open.
  Local paths are canonicalized before comparison, so `..` segments and
  symlinks out of the tree are resolved rather than compared as text, and
  matching is component-wise, so `/srv/images` does not admit
  `/srv/images-private`. A refusal is an ordinary read failure as far as
  `on_error` is concerned, so `on_error="null"` nulls those rows.

  The check lives in `fetch.rs`, the one stage the two surfaces share, and
  both `prefetch` and `row_bytes` take the policy as a **required** argument —
  a caller that forgets it fails to compile rather than silently reading
  unrestricted. Denied remote paths are dropped before the fetch, so a refusal
  never becomes a network request.

### Fixed

- **`scale(out_dtype=...)` and `clamp(out_dtype=...)` were accepted and then
  discarded.** The parameter was validated, serialized into the op's params —
  so it entered the op's identity, and therefore the CSE and compiled-graph
  cache keys — and read by nobody: no `resolve_op` arm looked at it, and
  `output_dtype_for` honours an `out_dtype` override only for the
  `Configurable` dtype rule, which is `normalize`'s, not theirs
  (`PromoteToFloat`). A plan-vs-exec test could not see it, because plan and
  execution *agreed*; they agreed on the promoted float rather than on what was
  asked for.

  Both now honour it, by lowering the request to a trailing `cast` — the
  mechanism `preserve_dtype=True` already used, and the one kernel fusion
  already handles exactly once (`try_fuse` pins `FusedKernel::out_dtype` to what
  the unfused chain would have produced). Giving the ops their own output dtype
  instead would have meant teaching `extract_ops` and that pinning about it, and
  would have turned `PromoteToFloat` — which preserves f64 input — into a
  `Configurable(F32)` that silently downgrades it. `out_dtype` and
  `preserve_dtype` are now one mechanism with one target, in
  `_out_dtype_target` / `_apply_out_dtype`.

- **`warp_affine` silently substituted the identity for a singular matrix.**
  Warping is inverse mapping — for each output pixel, ask where it came from —
  so a matrix collapsing the plane onto a line or a point has no answer. The
  runner returned the input unchanged and reported success, so a caller who
  passed a degenerate transform got their image back with no signal. It is now
  rejected where the user supplies it, naming the determinant.
  `AffineParams::is_invertible` is the single authority, read by both the
  rejection and the runner's `debug_assert`. The check is skipped under a
  plan-time probe, where every expression parameter is bound to the same
  placeholder and a per-row matrix is singular by construction; per-row values
  are checked per row, where they exist.

- **The channel rule had two readings.** Python's `_update_channels_from_rule`
  re-derived the output channel count by parsing the stringified rule, and
  disagreed with view-buffer's `OutputChannelRule::apply` on `NotApplicable`:
  `apply` returns "no channel count", Python left the hint unchanged. It stayed
  invisible because every `NotApplicable` op also drops below rank 3, where
  `_drop_hints_below_rank` clears channels anyway — except
  `histogram(output="quantized")`, which preserves rank and was mislabelled
  `NotApplicable` while actually preserving channels. Two errors cancelled.

  Quantized now declares `PreserveChannels` (its own comment already said so),
  and Python reads `apply` through a new `op_output_channels` FFI instead of
  holding a copy of the arithmetic. `_drop_hints_below_rank` stays: it is a
  separate, correct invariant, not the mitigation it was doubling as.

- **The contour round trip did not close: `source("contour")` could not read a
  contour set.** `extract_contours().sink("native")` emits `List(CONTOUR_SCHEMA)`
  — one *set* per row — but the source parsed a single `Struct` per row and read
  a list as one contour's ring of points. Feeding the sink's own output back in
  therefore failed with `Point struct missing 'x' field`, the contour struct's
  `exterior`/`holes`/`is_closed` being read as a point's `x`/`y`. The asymmetry
  was known and worked around in `test_schema_parity_sources.py`, which fed back
  a single element of the set rather than the set.

  The source now reads both shapes, through one parser (`parse_contour_set`).
  The two list forms are told apart by element dtype — a `List` of point structs
  is a ring, anything else in a `List` is a set — rather than by trying one and
  falling back, because the fallback is what turned a legitimate contour set
  into a complaint about points. Both shapes are now swept by
  `test_schema_parity_sources.py::test_contour_source`.

- **Rasterizing more than one contour with `fill_value < background` returned an
  all-background canvas.** The graph executor rendered each contour to its own
  mask and folded them with `max`, which is only union-like when the fill value
  is the larger of the two: `rasterize(fill_value=0, background=255)` silently
  erased every region as soon as the contour set held two of them (one contour
  was fine, which is why it survived). Sequential painting into one canvas would
  have swapped that for a different fault — a later contour's hole eating an
  earlier contour's fill, making the mask depend on the set's order.

  `rasterize` now takes the whole set, resolves coverage as a union first and
  colours it once, so neither fault is reachable and the single-contour case is
  the one-element case rather than a second implementation. The
  contour source and the `rasterize` op share it, so a mask no longer depends on
  which of the two routes produced it — asserted by
  `TestMaskContourRoundTrip::test_the_two_routes_to_a_mask_agree`.

- **`.sink()` accepted any keyword at all, and dropped the ones its format
  does not read.** The sink takes `**kwargs` and spreads them into the graph's
  `SinkSpec`, so `sink("jpeg", qualtiy=50)` built a graph carrying `qualtiy`,
  serde dropped it as an unknown field, and the query encoded at the default
  quality with nothing said. Of the three real keywords only `dtype` policed
  where it applied: `quality` on a png sink and `shape` on a png sink were
  accepted and ignored.

  `quality` turned out to be jpeg-only. `SinkSpec` documented it as "JPEG and
  WebP" and the sink docstring said "jpeg/webp", but the WebP arm of
  `encode_image` calls an encoder that takes no quality argument — so a webp
  quality has always been silently discarded. It is now rejected rather than
  accepted-and-dropped; supporting it is an encoder change, not a parameter
  change.

  Both ends of the pipeline now answer this question from one place:
  `SOURCE_PARAM_APPLIES` and `SINK_PARAM_APPLIES` in `_types.py`, next to the
  specs they describe, read by one `reject_inapplicable_params`. A name that is
  not in the table is rejected too, which is what closes the open `**kwargs`.
  Rust's `SinkSpec` gained `#[serde(deny_unknown_fields)]`, the same mechanism
  `GraphNode` already uses, so a hand-built graph cannot carry a sink field
  nothing reads either.

  The `quality` declaration is checked against the encoders rather than merely
  asserted: for each image sink, whether the output changes between quality 10
  and 95 must equal what the table claims.

- **Three Python spec classes nothing referenced.** `_types.SinkSpec`,
  `OutputSpec` and `MultiSinkSpec` were unreachable — the sink is built from
  raw kwargs in `_graph.py`, and nothing in the package or tests names them.
  They also carried a fourth, stale copy of the applicability fact
  (`if format == JPEG or WEBP: result["quality"]`), wrong about WebP in exactly
  the way the docstrings were. Deleted, with a guard in
  `test_removed_surfaces.py`.

- **Source parameters that do not apply to the chosen format were silently
  ignored, and each one policed itself differently.** `Pipeline.source()` takes
  eleven keywords, and most apply to a subset of the eight formats. Of the
  seven with a limited scope, one raised (`decode_max_size`), one warned
  (`cloud_options`), and five were dropped without a word: `width`, `height`,
  `shape`, `fill_value` and `background` outside `"contour"`,
  `require_contiguous` outside the nested-column decode, `allowed_roots`
  outside a path read. So `source("image_bytes", width=224)` built a pipeline
  that ignored the size, and `source("contour", allowed_roots=[...])` accepted
  a sandbox it never applied.

  Applicability now has one authority — `_SOURCE_PARAM_APPLIES`, each parameter
  against exactly the formats whose decode reads it — and one check, which
  raises. `_is_supplied` compares against the signature's own defaults, and
  `source()` hands the check `locals()` rather than a hand-written list, so
  the parameters it validates are the parameters it has.

  Guarded three ways: the table's keys must equal `source()`'s keywords, so a
  new parameter cannot inherit "applies everywhere" by omission; the check must
  read `locals()`; and the full parameter x format grid is swept, every
  applicable pair accepted and every inapplicable pair rejected. `thumbnail()`
  reads the table too — it writes `decode_max_size`, and had drifted into
  refusing an `"auto"` source that `source(decode_max_size=...)` accepted.

  Two behaviour changes: `cloud_options` on a non-path source now raises
  instead of warning (and its warn-and-drop branch, unreachable once the check
  ran first, is gone rather than left behind) (a warning is filtered out of a query's output and the
  credentials do nothing either way), and `dtype` on a contour source is
  rejected as described below. `test_cloud_options_on_non_file_path_source_warns`
  was rewritten to pin the rejection, and the `decode_max_size` and
  `thumbnail()` message assertions now match the shared wording.

- **`source("contour")` published a rank and nothing else, then dropped an
  asserted dtype.** The source decodes by rasterizing, so what it hands the
  first op is what the `rasterize` op hands its successor — an `[H, W, 1]` u8
  mask. It stated a hand-written `_expected_ndim = 3` and left the dtype
  `"auto"`, the channel count and the canvas unstated, while
  `GeometryOp::Rasterize` had declared all four all along. Both typed sinks need
  a concrete element dtype, so `source("contour").sink("list")` was unplannable
  and a no-op `.cast("u8")` was the only way through; `sink("array")`
  additionally demanded an explicit `shape=` for a canvas fixed by the source's
  own `width`/`height`.

  The source now folds `GeometryOp::Rasterize`'s contract through the same
  `op_contract` / `op_schema` / `op_infer_shape` FFI an appended op goes
  through, so the two routes to a mask publish one contract
  (`TestContourSourcePlanTimeContract`). `source("contour", dtype=...)` is
  rejected rather than dropped: rasterizing fixes u8, and the parameter reached
  `SourceSpec.dtype` and stopped there — an asserted `"f32"` bought a u8 column.

- **`rasterize(width=, height=)` published no shape, though its docstring said
  `infer_shape` supplied one.** The planner declines to call `op_infer_shape`
  when the input rank is unknown, and the contour domain has no rank — so the
  one op whose canvas is fixed entirely by its own parameters was the one op
  never asked. `_input_dims_for` now recognises a step that builds a buffer out
  of a non-buffer domain (from the contract: `input_domains` excludes buffer,
  `output_domain` is buffer) and asks with no input dims, which is what a
  contour input actually is. Ops that consume a buffer are unaffected, and the
  contour measures — whose `infer_shape` describes one value *per contour*, not
  the vector's length — stay out of it by producing a vector rather than a
  buffer.

  `op_infer_shape` on a `rasterize(shape=<node>)` also reported a 1x1 canvas as
  fact: introspection has to supply width/height for the op to resolve at all,
  and the placeholders were literals, so they read as fixed across probes. They
  are expression placeholders now, and the dimension reads as unknown
  (`TestOpInferShapeAuthority::test_rasterize_by_shape_reference_is_unknown`).

- **The runtime series builders read the data to decide the schema.**
  `build_typed_list_series_from_rows_with_dtype` took both the element dtype
  and the nesting depth from the *first non-null row*, using the planner's
  `expected_dtype`/`expected_shape`/`expected_ndim` only as a fallback when
  every row was null; the array builder likewise fell back to the first row's
  shape. That inverts the contract — the spec is what `collect_schema()`
  already promised — and it made the outcome depend on where the nulls fell: an
  all-null column honoured the plan while the same pipeline with one value in
  it did not.

  The spec is now the authority in both builders. A row whose data contradicts
  it is an error naming both dtypes rather than a silent reinterpretation, and
  the array builder has no data fallback at all: a fixed-shape column whose
  dtype depends on which row arrived first is the thing that sink exists to
  rule out.

- **An `"auto"` output dtype was taken from the input column, not folded
  through the ops.** `resolved_output_specs` assigned the input column's leaf
  type straight to any output still marked `"auto"`, which is correct only for
  a preserve-input lineage: `source("list")` over a `List(UInt8)` column
  followed by `scale()` planned `List(UInt8)` for data that arrives f32. The
  per-row `validate_output_schema` guard caught it, so it surfaced as a
  mid-`collect()` failure rather than a wrong column — planned wrong either way.

  The dtype is now folded through the ops' `OutputDTypeRule`s from the resolved
  source type by `fold_output_dtype`, the exact twin of the `fold_output_rank`
  that sits beside it and had been doing this for the rank all along.

- **A partially-resolved dtype is no longer discarded.** Folding a rule over an
  unknown input used to collapse back to `"auto"`, throwing away real
  information: whatever a PNG or TIFF turns out to decode to, `PromoteToFloat`
  of it is *a float*. `PlannedDType` (view-buffer, `core/dtype.rs`) is the
  three-state lattice — `Known(DType)`, `SomeFloat`, `Unknown` — and
  `OutputDTypeRule::resolve_planned` is the symbolic twin of `resolve`.

  `SomeFloat` is deliberately not a dtype: `PromoteToFloat` yields f32 for
  integers and preserves f64, so there is no single answer, and inventing one
  is a lie the runtime guard would catch. It is enough to decide a codec
  though — `ImageCodec::check_planned` refuses only when *every* candidate
  fails — which closes the last plan-then-fail case:
  `source("image_bytes").scale(...).sink("jpeg")` is now a planning error
  instead of an encoder error, while the same source *without* a promoting op
  still plans, because it could still decode to u8.

  Every site that compared a dtype string to `"auto"` now asks
  `PlannedDType::parse(...).is_concrete()` instead — a hand-written comparison
  would have let the second sentinel through to `dtype_str_to_polars`'s `u8`
  arm.

- **A multi-root graph resolved every output's dtype from the first column.**
  `merge_pipe` and the binary ops join two `pl.col()` lineages into one
  `vb_graph` call, and `resolved_output_specs` filled in each output's still-
  `"auto"` element dtype from `inputs.first()` regardless of which branch the
  output belonged to. Two list columns of different leaf types planned
  `Struct({x: List(UInt8), y: List(UInt8)})` and executed with `y` as Float32 —
  caught by the per-row `validate_output_schema` guard, so it surfaced as a
  mid-`collect()` failure rather than a wrong answer, but planned wrong either
  way. Two *separate* expressions never showed it: each is its own plugin call
  with its own single input.

  Each output is now resolved against the column its own lineage reads, found
  by walking `upstream` to a root and reading `column_bindings`.

- **`source("auto")` over a Binary column planned a rank it could not know.**
  `auto` leaves `_expected_ndim` unset, and the `list` sink waived the
  unknown-rank error for it on the theory that Rust resolves the rank from the
  column type. That holds for a List/Array column, where `resolved_output_specs`
  folds the nesting depth through the op rank rules, and not for image bytes,
  whose rank is only settled by the decode. `dtype_for_output` fell through to a
  depth-1 `List(UInt8)` while execution produced `List(List(List(UInt8)))`.

  It now refuses instead, naming the two ways out (name the source, or use a
  sink that does not encode the rank in its dtype). `auto` over List/Array
  columns is unaffected.

  **This changes an accepted query into a planning error.**
  `source("auto", dtype=...)` into a `list` sink over a binary image column was
  accepted before; it was never correct, and eagerly it happened to return a
  column, which is why `test_auto_image_list_sink_with_explicit_dtype` passed
  while asserting only that the column came back. An explicit dtype settles the
  element type, not the rank.

- **The image-encoder sinks checked their preconditions in the encoder.**
  `png`, `jpeg`, `webp` and `tiff` each restrict the buffer's dtype, and all
  four need an image-shaped buffer — facts that follow from the buffer's
  description, which is exactly what `OutputSpec` carries. Checking them at
  encode time meant a pipeline that promotes to f32, or flattens to rank 1,
  planned as `Binary` and then died part-way through `collect()`.

  `ImageCodec::check_support` (view-buffer, `interop/image.rs`) is now the one
  table: read by `dtype_for_output` when the query is planned, by `encode_sink`,
  and by the encoders themselves — `encode_tiff`'s fallback arm asks it for the
  message rather than writing a second one. An unknown is never a rejection, so
  a source that could still decode to u8 is not refused — but "unknown" is now
  a lattice rather than a single state, so a *partially* resolved dtype is not
  wasted (see below).

  Guarded in Rust by `the_table_never_promises_what_an_encoder_cannot_deliver`,
  which runs every (codec, dtype, channels) cell through both the table and the
  real encoder and requires that the table never admits what the encoder
  refuses. Two cells where it is deliberately *stricter* — 16-bit JPEG and WebP,
  which `image` accepts by silently halving the bit depth — are listed with
  their reason and separately pinned.

- **Three vector-domain sinks planned one thing and executed another.** The two
  halves of the sink contract keyed on different facts: `dtype_for_output`
  decides the Polars dtype from `(expected_domain, format)`, while
  `encode_node_output` decided the *value* from the runtime `NodeOutput`
  variant. Those are not the same thing — a domain can arrive in more than one
  representation. A perceptual hash is a `vector`-domain output that rides as a
  `Buffer` (`apply_perceptual_hash` returns a 1-D `u8` buffer); `extract_shape`
  produces a real `Vector`. Wherever the two diverged, so did the contract:

  - `perceptual_hash().sink("native")` planned `List(UInt8)` and failed at
    `collect()` with "Buffer outputs require explicit format" — `native` on a
    vector output is the documented spelling.
  - `extract_shape().sink("array", shape=[3])` planned `Array(Float64, 3)` and
    failed with "Unsupported sink format: array". The schema arm for
    `("vector", "array")` had been added to fix an earlier divergence *without*
    the encode arm that makes it real.
  - `perceptual_hash().sink("array")` reported a shape mismatch against the
    buffer rather than encoding the hash.

  The pairs that did work did so because the two dispatches happened to agree,
  not by construction. `encode_node_output` now keys on the planned domain —
  the same key its counterpart uses — and its arms mirror that function's one
  for one; the `NodeOutput` variant is used only to reach the data, which is
  what it actually tells you.

  Guarded by `tests/test_sink_contract.py`, which sweeps every
  (domain representation × `SinkFormat`) pair and asserts the *relationship*
  rather than a blessed list: a pair rejected at plan time is fine, but one
  that survives planning must execute to exactly the dtype planning promised.
  Both axes are completeness-asserted, and the two vector representations are
  pinned to encode identically.

### Changed

- **Declaring a `named_variants!` table and registering it are one act.**
  `every_named_enum_is_registered` fails if an enum declares a `NAMED` table
  without appearing in `naming::REGISTRY`. Registration is what surfaces an
  enum over `enum_variants`, what gets its names checked for duplicates, and
  what puts it in `enum_names()` — which the Python suite iterates to decide
  what to parity-check. An unregistered enum was therefore not "private", it
  was unchecked, with a Python mirror free to disagree with it. The check runs
  in both directions, so a scan that stops matching fails rather than passing
  vacuously.

- **`SourceFormat` is pinned to Rust's `KNOWN_SOURCE_FORMATS`.** They are two
  hand-written spellings of one vocabulary, and a note in the test suite
  claimed there was nothing to pin them to — true of sink formats (which are
  matched on and rejected by fall-through, never enumerated), not of sources,
  which the graph validator checks against an explicit list. A Python-only
  format built a graph the validator rejected at execution; a Rust-only one was
  a decode path nothing could reach.

- **`op_contract` publishes each fact once.** It carried both `input_domain`
  (a single `Domain`) and `input_domains` (the accepted set) across the FFI;
  only the set was ever read, and the two were free to disagree the moment a
  step accepted more than one domain — which binary ops and reductions do. The
  singular key is gone and `test_contract_publishes_no_second_spelling` pins
  the key set in both directions.

- **The graph wire format is closed at the node.** `GraphNode` now carries
  `#[serde(deny_unknown_fields)]`, so a stale or misspelled key fails loudly
  instead of being silently dropped — which is how `shape_hints` went on being
  emitted long after its last reader. Fields only Python consumes
  (`domain`, `output_dtype`) are declared on the Rust struct to keep it closed.
  `OpSpec` is deliberately *not* closed: its parameters ride on
  `#[serde(flatten)]`, which serde documents as incompatible with
  `deny_unknown_fields`.

  **This is a behaviour change under version skew.** A newer Python emitting a
  node field an older compiled `.so` does not declare now fails the query
  rather than dropping the field silently. The install is editable, so a stale
  extension is the normal state after pulling Rust changes — re-run
  `maturin develop`, or check `polars_cv.build_info()`, which reports the three
  versions that must agree.

- **Two `unreachable!()` panics removed from `build_plan`.** They were reachable
  by an op author declaring `MemoryEffect::View` on a compute or image op — a
  runtime abort for a contract mistake. The materialisation decision is now a
  direct comparison that treats a `View` declaration as "no materialisation
  needed", which is what it means.


- **Binary ops and reductions declare that they accept a vector.** Their
  `GraphStep` contract said `buffer`, which read as "images only" and was
  wrong: a perceptual hash is a 1-D `u8` buffer encoded as a `vector`, and the
  library's own `hamming_distance` is `hash_a ^ hash_b → reduce_popcount` with
  both operands in `vector`. Nothing enforced input domains from that contract
  until the planner started to, so the mistake was invisible. `input_domains()`
  is now a set (`["buffer", "vector"]` for those two), which keeps
  `extract_contours().reduce_sum()` rejected at build time — widening them to
  `Domain::Any` would not have. The rejection message names the accepted set,
  so it now reads "expects buffer or vector input".

- **Shape hints are invalidated, not carried forward, when a step has no
  inferable shape.** Axis reductions, histograms, channel merge and the binary
  ops are graph-level steps `op_infer_shape` rejects; keeping the pre-op H/W
  across them let a pipeline publish `[100, 200, 2]` for data that executes as
  `[200, 3, 2]`. Unknown is always safe — a typed sink asks for an explicit
  shape instead of planning a wrong one.

- **`transpose` and `flip` validate their axes against the tracked rank.**
  `transpose([1,0,2])` after a `channel_select` (rank 2) reached view-buffer's
  `infer_shape`, which indexes the input shape unchecked, and surfaced as a
  `PanicException` with a Rust backtrace from an ordinary builder call. The
  builders now raise `ValueError`, and `op_infer_shape` catches unwinds so no
  future op can leak a panic into the planner.

- **Each operation's input domain now comes from its Rust contract.** Builders
  passed their own expected domain to `_validate_domain(self.DOMAIN_BUFFER, ...)`
  at 53 call sites, restating a fact `op_contract(...)["input_domain"]` already
  published and nothing read. The check moved into the append path, and
  `Pipeline`'s `DOMAIN_*` string constants — a third copy of the domain
  vocabulary behind Rust's `Domain::NAMED` and the Python `Domain` enum — are
  gone.

- **`op_infer_shape` now covers geometry steps.** `rasterize`'s output canvas is
  fixed by its own width/height, but the FFI accepted only buffer ops, so the
  Python builder assigned those hints itself as a side effect of building the
  op's parameters — which the continuation replay skipped. `GeometryOp`'s
  `infer_shape` is now read like any other, and a step whose shape is genuinely
  data-dependent (`extract_contours`) reports "not knowable" rather than rank 0.

- **`assert_shape()` records the position it was written at.** A user assertion
  outranks inference, but only from that point on, so it can survive a
  continuation replay without overriding later ops that legitimately change the
  shape (`assert_shape(channels=3).grayscale()` still reports 1 channel).

- **`average_precision(interpolation="all_points")` omitted the first recall
  segment.** `_all_points_ap` built the monotone precision envelope correctly
  then integrated only over the recall values present in the curve, never
  anchoring at recall = 0. The leftmost block — `recall[0] × envelope[0]` —
  was dropped every time, so a three-detection curve that should score 2/3
  reported 1/3, and a perfect single-TP result reported 0.0. Both
  `_all_points_ap` and the matching per-replicate trapezoid in
  `bootstrap_pr_auc` now prepend a recall-0 anchor (COCO / scikit-learn
  `Σ (Rₙ − Rₙ₋₁) · Pₙ` with `R₀ = 0`). **This changes reported AP numbers**
  for `interpolation="all_points"` (the default); `11_point` is unaffected.

- **`PreMatchedAdapter` derived the image population from detections only.**
  Images with no detections got no metadata row, silently deleting the
  negative population and inflating recall / FP-per-image. `match()` now
  accepts an optional `image_meta` frame that defines the full evaluation
  population; omitting it keeps the previous behaviour but emits a
  `UserWarning`. `image_meta` is the *sole* source of `image_metadata`, so
  combining it with `n_gts_col` / `weight_col` / `gt_label_col` /
  `group_col` — which only ever described how to derive metadata from the
  *detection* frame — raises rather than accepting the arguments and
  discarding them.

- **FROC double-counted detections when `image_id` repeated in metadata.**
  `_curve_from_detections` joined detections to `image_metadata` weights
  without deduping by `image_id`, so a shared rendered image owned by two
  cases (or a bootstrap-with-replacement draw) fan-out-multiplied every
  detection before TP/FP aggregation. The weight lookup is now unique by
  `(image_id, class_id)`. The same join bug made `FROCResult.bootstrap_ci`
  produce intervals that excluded the point estimate and replica
  sensitivities above 1.0; both are fixed.

- **A FROC metadata row is one (image, class), but the FP-per-image
  denominator counted rows.** `froc_curve` took `n_images` from
  `image_metadata.height`, so a two-class table divided the false-positive
  rate by two. The image count is now the number of distinct `image_id`s,
  read once through `_count_images` by both `froc_curve` and
  `FROCResult._reconstruct` — the two had drifted apart, the replicate
  counting draws while the point estimate counted rows, and the bootstrap
  distribution was therefore not on the same scale as the estimate it
  bracketed. `_curve_from_detections` no longer takes `n_images` at all; it
  reads the metadata it was already given.

- **Bootstrap draws are distinct evaluation units, not repeated images.**
  `FROCResult._reconstruct` resampled with replacement and carried the
  duplicates as repeated `image_id`s, leaving every downstream count to
  guess whether a repeat was a redraw or one image legitimately owned by two
  cases. Each draw now gets a synthetic `image_id` (`<id>#draw<n>`), so a
  redraw contributes its own detections, its own `n_gts` and its own slot in
  the FP-per-image denominator, and a repeated `image_id` again means only
  one thing.

- **FROC / LROC curves were returned in ascending-threshold order**, so
  `fp_per_image` / `fpf` ran downwards through the frame and
  `plt.step(..., where="post")` drew backwards. Curves are now sorted by
  **descending `threshold`**, which is ascending `fp_per_image` / `fpf`.
  Sorting on the x-column directly looks equivalent and is not: thresholds
  are unique so the order is total, while `fp_per_image` ties constantly —
  every threshold bucket adding only true positives leaves it unchanged —
  and Polars' `sort` defaults to `maintain_order=False`, so the y at each tie
  boundary, and therefore the trapezoid leaving it, was unspecified and could
  differ between runs and platforms. (Commit `7829291` fixed the same class
  of nondeterminism in `bootstrap_pr_auc` after it surfaced as a macOS CI
  failure.) **This changes reported FROC/LROC AUC**, deterministically.

- **`MetricResult.auc` and `.interpolate` each re-sorted the curve by x
  alone**, inheriting that same tie ambiguity no matter how the curve was
  stored, and then read the *first* row of a tie group — so
  `sensitivity_at_fp(0.0)` reported the origin's `0.0` rather than the
  sensitivity a detector actually reaches while making no false positives.
  Both now go through one `MetricResult._curve_xy`, which collapses tied x to
  the maximum y (the standard ROC upper envelope) and yields strictly
  increasing x. `summary_table`'s y column is Float64 even when every
  operating point is out of range, instead of collapsing to `Null` dtype.

- **The unweighted branch of `_curve_from_detections` was unreachable.**
  `weight` is in `IMAGE_META_REQUIRED` and `DetectionTable.from_matched`
  validates it, so `has_weights` was always true and the second curve
  implementation behind it could never run. Deleted; an all-1.0 weight column
  reduces the weighted formulas to the plain counts exactly.

- **FROC weighted sensitivity was order-dependent when a duplicated
  `image_id` carried conflicting weights.** The weight lookup deduped by
  `image_id` (first row wins) while denominators summed every metadata row,
  so `[1, 5]` vs `[5, 1]` flipped sensitivity (`0.1667` vs `0.8333`). Equal
  weights still dedupe cleanly; conflicting weights now raise `ValueError`
  pointing at a composite `image_id` or a single weight per unit. The lookup
  key is `(image_id, class_id)` when class is present, and the *conflict
  check* is on `image_id` alone — which subsumes it. A `weight` is a property
  of an image, and the FP-per-image denominator dedupes on `image_id`, so two
  classes of one image disagreeing about its weight left that denominator
  depending on which row `unique` happened to keep.

- **`MetricResult.interpolate` (and `sensitivity_at_fp` /
  `sensitivity_at_fpf` / `summary_table`) clamped past the observed x-range
  instead of returning null.** Queries beyond the curve's max now return
  `None` / null so an unreachable operating point is visible rather than
  silently repeating the last y value. `examples/06_detection_metrics.py`
  formats through a `fmt()` helper accordingly — `round(None, 4)` is a
  `TypeError`.

- **`partial_auc` rejected integer bounds.** The boundary points it appends at
  `lo` / `hi` were built with an inferred dtype, so the natural spelling of a
  range — `froc.auc(fp_range=(0, 8))`, as the docs and
  `examples/06_detection_metrics.py` both write it — produced an `Int64`
  Series that would not concatenate onto the `Float64` curve and raised
  `SchemaError: type Int64 is incompatible with expected type Float64`
  whenever the curve did not already span the requested range. Bounds are
  coerced to float and the boundary Series are constructed as `Float64`
  explicitly.

- **`.contour.label_reduce()` was a second, divergent implementation.** The
  accessor scored contours with a plugin-local routine while
  `Pipeline.label_reduce()` used the engine's `score_contours_on_buffer` — the
  same math, moved into `view-buffer` earlier, whose plugin-side copy was never
  deleted. The two had drifted:

  - `region_mode="boundary"` was rejected by the accessor and accepted by the
    pipeline.
  - A contour catching no pixel centre (a sub-pixel detection) scored **0.0**
    through the accessor and its **centroid value** through the pipeline. The
    accessor's zero then read downstream as a detection with no evidence.
  - The accessor copied the whole image into a `Vec<Vec<f64>>` before scoring.

  Both entry points now call `score_contours_on_buffer` and parse `reduction` /
  `region_mode` against the engine's `NAMED` tables, so accepted names cannot
  drift either. **This changes reported numbers** for `.contour.label_reduce()`
  on sub-pixel contours, which now score at their centroid instead of 0.0.
  `Pipeline.label_reduce()` and `ContourMatcher` are unaffected — they were
  already on the engine path.

  The existing parity test covered only `region_mode="bbox"`, the one mode the
  two agreed on; it now runs across every reduction and region mode.


- `measures::signed_area` is now `geo`'s signed area rather than a hand-written
  shoelace loop, and `transforms::ensure_winding` reads `measures::winding`
  instead of re-deriving it — leaving `geo` the single authority for polygon
  area. (`geo` treats a `LineString` as one-dimensional with zero area, so the
  ring is measured as the hole-free polygon it bounds.)

### Removed

- **`OutputDType`, a partial second dtype table whose one distinct value meant
  "the default".** It listed `f32`/`f64`/`u8` alongside the spellings
  `dtype_table!` already owns, plus `"preserve"`, documented in the enum and in
  `clamp`'s docstring as "keep input dtype (floats preserved, integers -> f32)"
  — character for character what `OutputDTypeRule::PromoteToFloat` does, which
  is what passing nothing already did. It was a synonym for the default, not an
  unimplemented feature: `normalize` rejected it by hand for that reason, and
  `scale`/`clamp` accepted it into the op's identity and dropped it.

  The behaviour the word suggests (u8 in, u8 out) is `preserve_dtype=True`,
  which is wired and tested. `out_dtype` now validates against `DType`, so every
  dtype is requestable and there is one table of dtype names.

- **Ten unreferenced `view-buffer` public items.** Zero-argument wrappers over a
  parameterised primitive (`from_slice_simd_aligned`, `is_simd_aligned` — the
  `from_slice_aligned` / `is_aligned` pair answers strictly more, and
  `SIMD_ALIGNMENT` is now the named argument in their documented example), a
  `size_bytes` whose docstring claimed a tiling caller that did not exist, the
  `layout_report` / `LayoutReport` inspection pair with no consumer, `ScalarOp::
  op_type` superseded by the `description` beside it, and four `NodeOutput`
  accessors and constructors nothing called. `ImageView::get_pixel` was kept
  and documented instead: it is that public view type's only accessor, so
  deleting it would leave it unable to read what it borrows.

- **Three surviving second copies.** `polars_dtype_to_str` (graph/compiled.rs)
  was a second Polars-type→dtype map beside `decode::dtype_from_polars_datatype`
  — the two were free to disagree about which Polars types are buffer elements
  at all, and did. `output::dtype_to_string` was a second name for
  `DType::numpy_name` with one caller. `cloud::is_cloud_path` was a second
  path classifier beside the one `fetch.rs` actually uses, reachable only from
  its own test; an unused classifier sitting next to a used one is an
  invitation to pick the wrong one.

- **`scripts/verify.sh`'s pinned Rust toolchain.** It set
  `RUSTUP_TOOLCHAIN=1.96` — a third declaration behind `rust-toolchain.toml`
  (`stable`) and CI (`dtolnay/rust-toolchain@stable`), disagreeing with both.
  A local `PASS` was checking a different compiler than CI, and downloading a
  toolchain to do it. `rust-toolchain.toml` is now the only declaration.

- **`rasterize(anti_alias=)`.** It was threaded from the builder through the op
  spec, the JSON graph, `resolve_rasterize_style` and `GeometryOp::Rasterize`
  into `geometry::rasterize`, whose signature named it `_anti_alias` and ignored
  it. Documented as "not yet implemented", but not free: it entered the op's
  identity, so two pipelines that behave identically hashed differently for CSE
  and compiled to separate graph-cache entries. Passing it now raises
  `TypeError` instead of being silently discarded.

- **view-buffer's unreachable pipeline-composition layer.** `ops/io.rs`
  (`SourceFormat`, `SinkFormat`, `PlaceholderMeta`) and the `ExprNode`
  variants only it fed — `LazySource`, `Placeholder`, `Sink` — plus their
  constructors. Nothing in the workspace called them; the plugin builds its own
  source/sink vocabulary. Every `match` over `ExprNode` carried arms for them,
  two of which were `panic!("must be resolved before building plan")`. Also
  removes the never-enabled `numpy_interop` / `torch_interop` features, and
  retires the "three-way format representation split" that two comments
  described as a divergence to live with.

- **The cost-reporting subsystem.** `ops/cost.rs`, `OpCost`, `OpCostReport`,
  `PipelineCostReport`, `ViewExpr::cost_report()`, `explain_costs()` and
  `Op::intrinsic_cost()`. Exercised only by view-buffer's own tests — no Python
  surface reached it, so every op author maintained a declaration for nobody.
  `MemoryEffect` is retained and its documentation corrected: it called itself
  legacy and pointed at `intrinsic_cost()`, when it is in fact the
  materialisation authority (`build_plan` inserts `MaterializeContiguous` from
  it) and cost could never have replaced it, because the `MemoryEffect ->
  OpCost` conversion collapsed `StridePreserving` and `RequiresContiguous` into
  a single value.

- **Node-level `shape_hints` from the graph JSON.** No Rust code ever read the
  key. Because `graph_json` is the compiled-graph cache key, two pipelines that
  execute identically but carry different hints occupied separate cache
  entries. Plan-time shape still crosses the boundary as `expected_shape` on
  the output spec, which Rust does read.


- **Unreachable geometry vocabulary.** `GeometryOp` carried ten variants
  (`Winding`, `Flip`, `EnsureWinding`, `Normalize`, `ToAbsolute`, `IsConvex`,
  `ContainsPoint`, `IoU`, `Dice`, `HausdorffDistance`) that no `resolve_op` arm
  could construct, together with an arm in the plugin's graph executor whose only
  job was to reject them. Those operations are served by the `.contour` namespace,
  which calls the `view-buffer` geometry functions directly. `GeometryOp` now
  lists only what the pipeline graph routes, and the executor's match is
  exhaustive by construction rather than by a runtime error string.

- **Dead `view-buffer::geometry` API**: `pairwise::bbox_to_contour`,
  `predicates::point_in_polygon`, `measures::distance_to_segment`,
  `Point::distance_squared_to`, `Contour::{from_int_tuples, add_hole, iter,
  points}` and `BoundingBox::union` had no callers. `rasterize_simple` is now
  `#[cfg(test)]`: it is the per-pixel oracle the scanline filler is checked
  against, not a second rasterization path.

- **`polars_cv.geometry.validation`**, along with the re-exported
  `GeometryValidationError`, `OpenContourError`, `CoordinateRangeError` and
  `InvalidContourError`. The three validator functions were never called and the
  exception classes were never raised, so nothing could catch them. The
  `Raises: OpenContourError` line on `.contour.area()` documented an error that
  could not occur.

### Fixed

- **Plan-time shape tracking was opt-in, and most builders opted out.** Appending
  an operation required a sequence of calls each builder made by hand. All 60
  ran the schema fold (`_update_output_dtype`); only 19 also updated the shape
  hints. Ops that change shape but skipped it published a planned schema
  execution could not produce, which surfaced at `collect()` as
  *"planned shape [...] but execution produced [...]. The planner's shape
  contract disagrees with the Rust implementation."*

  - `transpose(axes)` kept the pre-transpose H/W (a 100×200 image still reported
    100×200 after `[1,0,2]`) and kept a channel count its `unknown` channel rule
    says is not knowable.
  - `channel_select(index)` dropped rank 3 → 2 while `expected_shape` kept
    publishing a three-dimensional shape.
  - `channel_merge` kept the first operand's channel count.

  `Pipeline._push_op()` is now the only code that may mutate `_ops`, and it
  applies the domain check, the schema fold and the hint update together.
  Guarded structurally by `test_op_append_is_structurally_exclusive`, which
  fails if anything else touches `_ops` — replacing a ratchet that enumerated
  one of the two required calls.

- **The lazy continuation's shape replay never ran.** `LazyPipelineExpr.pipe()`
  re-applied each op's shape effect over the upstream hints, but assigned
  `_expected_ndim` *after* that loop, so every replayed op saw `ndim = None` and
  the H/W half of the replay returned at its opening guard. A post-loop overlay
  masked it for absolute targets like `resize(h, w)` while producing wrong
  values elsewhere — `.pipe(p).resize_to_height(50)` reported a width of 50,
  `resize_max(120)` reported a square, and `pad`/`pad_to_size`/`rotate` silently
  kept the upstream shape.

  The continuation now folds state and hints together one op at a time through
  the same append path the eager builders use, so the two spellings of an
  operation — `.pipe(p.op())` and `.pipe(p).op()` — agree by construction.
  Pinned by `test_eager_and_lazy_agree_on_shape_state` across every chainable
  op, with the op table completeness-asserted so a new operation cannot join
  without a case.

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

- **`extract_contours` did not trace boundaries at all.** The Moore-neighbour
  walk resumed its neighbourhood sweep five positions past the direction it had
  just moved, rather than at the background cell it had arrived from. From the
  top-left of a filled square its first step went *inward* along the diagonal; it
  then bounced between four cells and returned to the start. A filled 400×400
  region came back as **399 degenerate 2×2 contours, one per row** — never its
  outline. Everything downstream inherited it: `.extract_contours()`, contour
  metrics, and `ContourMatcher`, where `metrics` carried a filter for detections
  whose "rasterized interior is empty and are provably artifacts of the boundary
  tracer". A region now traces to exactly one border, following its rim.

  Two related faults went with it. The sweep's starting side is now chosen the
  way Suzuki–Abe does — west where a foreground run begins, east where one ends —
  so a **hole's rim traces correctly** instead of wandering the interior until a
  length guard stopped it; `mode="all"` reports the exterior plus one border per
  enclosed background region. And a trace no longer *starts* at a cell touching
  background only diagonally (the inside of a reflex corner, or a cell
  catty-corner to a hole), which was producing spurious extra contours on L, U
  and plus shapes.

  The traced outline runs through the **centres** of the boundary pixels, so it
  is inset by half a pixel: a region filling `w × h` pixels returns bounding
  `(w-1) × (h-1)`. That is inherent to describing a pixel set by a polygon, and
  is now documented and asserted rather than incidental.

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
