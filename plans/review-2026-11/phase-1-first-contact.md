# Phase 1 — Correctness and first contact

Implementation plan for the first group of findings from the external review
of 0.41.0 (commit `c0d77d5`, 2026-11-11). The ledger entries are CR-136 to
CR-140 in the root [`ISSUES.md`](../../ISSUES.md); this file holds the detail.
Phases [2](phase-2-stabilise.md) and [3](phase-3-depth.md) follow.

**Goal:** remove the five things that make a first evaluation fail. Nothing
here adds an op.

**Order:** CR-136, CR-139, CR-138, CR-137, CR-140. The first two are defects
with no open decision; CR-137 waits on one.

**Rules that apply to every item** (from `CLAUDE.md` and the ledger):

- Test first. Each item starts with a failing test at the user-facing entry
  point, watched failing for the reason it claims.
- "Bit-identical output is the default" (`ISSUES.md`, Decisions). An item that
  changes pixels or values says so, and ships as a CHANGELOG "Fixed" entry
  with a finding and a guard, or as an opt-in.
- A wire or signature change means: re-bless the catalogue, run
  `scripts/gen_ops.py`, `maturin develop`, and a migration-guide entry.
- Finish with `scripts/verify.sh`, reading its own PASS/FAIL line.

Every measurement below was taken on 2026-11-11 on a 4-core container with a
`--profile benchmark` build of the checkout.

---

## CR-136 — Scalar constants are `f32`, whatever the data

**Re-verified: confirmed, and wider than first reported.**

### Evidence

Constants are `f32` on the wire (`M::V<f32>`, 24 fields: 14 in
`view-buffer/src/ops/compute.rs`, 8 in `ops/image.rs`, one each in
`ops/filter.rs` and `ops/view.rs`) and in `ScalarOp` (`ops/scalar.rs:11`).
The f64 path widens them (`ScalarOp::apply_f64`, whose comment says so). So on
data whose accumulator is f64 — `f64` and the 32/64-bit integers — the
arithmetic is f64 but the constant has already lost precision.

| Call | polars-cv | NumPy |
|---|---|---|
| `Float64 [1, 3]`, `scale(1/3)` | `0.3333333432674408, 1.0000000298023224` | `0.3333333333333333, 1.0` |
| `Float64 [1.0]`, `add_constant(0.1)` | `1.1000000014901161` | `1.1` |
| `Float64 [0.1, 0.30000000000000004]`, `clamp_min(0.3)` | `0.30000001192092896` twice | `0.3, 0.30000000000000004` |
| `Int64 [16777217, …]`, `clamp(0, 16777217)` | `16777216` | `16777217` |
| `Int64 [16777217]`, `add_constant(16777217)` | `33554433` | `33554434` |
| `Float64` column parameter, `scale(pl.col("k"))`, `k = 1/3` | `0.3333333432674408` | `0.3333333333333333` |

The third row raises a value that already met the bound. The integer rows are
wrong in the last digit past 2^24. `Float32` and 8/16-bit data are unaffected:
`Float32 [1, 3] * (1/3)` matches NumPy's f32 result exactly.

The code already treats this as a defect in one place: `F64Step`
(`kernels/elementwise/mod.rs:98`) exists because "contrast's mean is an f64
statistic and must not be rounded to f32 first".

### Approach

1. Make the wire fields of the scalar family `f64`: `Scale.factor`,
   `AddConstant`/`SubtractConstant.value`, `Clamp.min`/`max`,
   `ClampMin`/`ClampMax.value`, `NormalizeRange.low`/`high`,
   `AdjustContrast.factor`, `AdjustGamma.gamma`, and the `mean`/`std` lists of
   `normalize_preset`. Leave geometric and filter parameters (`Rotate.angle`,
   `Blur.sigma`, thresholds) as they are unless the audit in step 4 says
   otherwise.
2. Carry `f64` constants in `ScalarOp`. The f32 path narrows at use
   (`c as f32`), which is the same f32 the wire produced before, so 8/16-bit
   and `Float32` output stays bit-identical. The f64 path uses the constant
   as given.
3. Delete the `as f64` widenings in `apply_f64` and the special-case
   `F64Step::{Sub,Mul,Add,Div}` variants that exist only to smuggle an f64
   constant past `ScalarOp` — one representation, not two.
4. Audit the remaining `M::V<f32>` fields: for each, either the op computes
   in f32 by contract (leave it, and say so in its doc comment) or it reaches
   an f64 path (convert it).
5. Integer bounds: decide whether `clamp` on an integer buffer keeps
   promoting to float. Out of scope here unless the audit shows the bound is
   still inexact after step 1 (an `f64` holds integers to 2^53).

### Tests first

- `tests/test_known_gaps.py`: one `xfail(strict=True)` per table row above,
  through `Pipeline` on an `Array` column. Watch each fail today.
- A reference test against NumPy for the scalar family on `Float64`, `Int32`,
  `Int64`, `UInt32`, exact equality.
- Bit-identity: the golden corpus and the existing u8/f32 reference tests must
  pass unchanged. This is the guard for step 2.
- `test_expr_param_identity.py`: a literal and the same value from a
  `Float64` column still give the same result.

### Risks

- **Op identity.** A constant enters CSE and the compiled-graph cache key.
  `1/255` as f64 and as f32 are different keys; check that no golden wire
  file pins the f32 spelling without being re-blessed.
- **`repr`.** `scale(factor=0.003921568859368563)` becomes
  `0.00392156862745098`. Snapshot tests of `repr`/`explain` move.
- **Output change on f64 data.** Deliberate: CHANGELOG "Fixed", ledger
  finding, the tests above as the guard.

### Done when

The six rows match NumPy, the u8/f32 corpus is byte-identical, and the pins
are deleted from `test_known_gaps.py`.

---

## CR-139 — `decode_max_size` can be slower than a full decode

**Re-verified: confirmed, and the scale-1 case is worse than reported.**

### Evidence

`decode_jpeg_scaled` (`polars-cv/src/execute.rs:23`) sends every JPEG with
`decode_max_size` set through the `jpeg-decoder` crate, at whatever scale the
decoder picks, including 1. The full path uses the `image` crate's decoder
(`zune-jpeg`). `zune-jpeg` 0.5.8 has no scaled IDCT, so the two paths cannot
share a decoder.

Decode only, 300 JPEGs of 620x496, median of 9, two rounds. A `cargo check`
was running, so absolute times are inflated; the ordering held in both rounds
and in two earlier uncontended runs (full 206–214 ms against 225–236 ms at
`decode_max_size=224`, 400 images).

| Source | Scale chosen | Round 1 | Round 2 |
|---|---|---:|---:|
| full decode | — | 362 ms | 336 ms |
| `decode_max_size=4096` | 1 | 466 ms | 447 ms |
| `decode_max_size=320` | 1/2 | 405 ms | 479 ms |
| `decode_max_size=224` | 1/2 | 357 ms | 350 ms |
| `decode_max_size=128` | 1/4 | 292 ms | 312 ms |
| `decode_max_size=64` | 1/8 | 251 ms | 260 ms |

At scale 1 the option costs about 30% **and changes pixels**: the two
decoders differ by up to 10 levels on the same file. On a 12-megapixel JPEG
with a 224 target (scale 1/8) it was 2.2 times faster, so the option is worth
keeping.

### Approach

1. In `decode_jpeg_scaled`, read the header, compute the factor the decoder
   would choose, and return `None` (full path) when it is 1. That alone
   removes the 30% case and makes `decode_max_size >= image` bit-identical to
   leaving it out.
2. Measure 1/2 properly (uncontended, the regression suite's `--profile
   benchmark`, several image sizes). If the scaled path does not win at 1/2,
   fall through there as well and document the option as "1/4 and smaller".
3. Add a regression-suite case for each factor
   (`benchmarks/regression/`), so the crossover is measured, not remembered.
4. Correct the docstring in `formats/source.rs`: it says the option "skips
   work"; say when.

### Tests first

- A test that `decode_max_size` larger than the image gives bytes identical to
  no `decode_max_size`. Fails today (max difference 10).
- The existing `test_decode_hint.py` invariant (long side never below
  `min(decode_max_size, original)`) must keep passing.

### Risks

- Output changes for callers who set `decode_max_size` at or above the image
  size. The option was already documented as not bit-identical, and the new
  behaviour is the full decoder's, so this is a "Fixed" entry.
- A longer-term option is to drop `jpeg-decoder` for a decoder with a fast
  scaled IDCT. Not planned: it is a new dependency for one option.

### Done when

No factor is slower than the full decode in the regression suite, and scale 1
is byte-identical.

---

## CR-138 — The tensor-struct sinks cannot be read back

**Re-verified: confirmed. No decision recorded against it.**

### Evidence

`sink("blob")`, `"array"`, `"list"` and `"png"` outputs all feed
`Pipeline().source()`. `sink("numpy")`, `"ndarray"` and `"torch"` fail with
"auto source cannot infer a decode path for column dtype Struct(...)".
`Source` (`polars-cv/src/formats/source.rs`) has eight variants and none
takes the struct; `Source::route` has no `Struct` arm. A `polars_cv.ndarray`
column survives a Parquet round trip and still cannot be read, because
`_plugin.call` hands the plugin `.ext.storage()`, the same struct.

### Approach

1. Add a `Source::Tensor` variant (wire name `"tensor"`; one name for the
   three sinks, which share one struct), with `dtype` and `on_error` as
   `Blob` has them: the row carries its own dtype, so a declared one is
   checked at decode and a mismatch is a row error.
2. Decode in `graph/decode.rs` beside `decode_binary_row`: build the
   `ViewBuffer` over the row's `data` with its `shape`, `strides` and
   `offset`, zero-copy where the bytes are aligned, exactly as the blob path
   does. Validate that the view lies inside `data` before constructing it;
   this is untrusted input and the check is the only thing between a
   malformed row and an out-of-bounds read.
3. Route it: a `Struct` whose fields are exactly `NUMPY_OUTPUT_SCHEMA`'s goes
   to `Source::Tensor` in `Source::route`. Compare against the one schema
   authority (`ext_types.rs`), not a second list of field names.
4. Plan-time facts: dtype and rank are per-row, so the planned state is
   `auto` dtype and unknown rank unless `dtype=` and an `assert_shape` say
   otherwise. This is CR-82's question from the other side; do not add
   metadata to the extension type here.
5. `gen_ops.py` regenerates `source()` and its docstring.

### Tests first

- Round trip for each of `numpy`, `ndarray`, `torch`: sink, then
  `source()` (auto) and `source("tensor")`, compare with the direct result.
  Include a non-contiguous row (a flip, negative stride), `compact=True`, the
  `f16` downcast, and a Parquet write/read between the two.
- Malformed rows: `offset` past the end, a stride that walks out of `data`,
  `shape` product larger than `data`, an unknown `dtype` string. Each is a
  row error under `on_error`, never a panic.
- Schema parity: planned schema equals executed schema for a tensor source
  followed by each sink.

### Risks

- `strides` and `offset` from a file are attacker-controlled. Step 2's bounds
  check needs its own unit tests and a fuzz-style sweep.
- The `auto` router now accepts a struct; any other five-field struct with
  the same names would route too. Matching the full schema, dtypes included,
  keeps that narrow.

### Done when

Every sink's output is readable by some source, and the docs' sink table says
which.

---

## CR-137 — EXIF orientation is ignored

**Re-verified: confirmed, but the recommended default changed.**

### Evidence

An 80x40 JPEG tagged Orientation=6 decodes as 40x80x3 (stored layout) and
`image_info()` reports `width: 80, height: 40`. No doc mentions orientation.

What other libraries do with the same file:

| Library | Default | Other mode |
|---|---|---|
| `cv2.imdecode` | applies it (80x40) | `IMREAD_IGNORE_ORIENTATION` |
| PIL `Image.open` | ignores it (40x80) | `ImageOps.exif_transpose` |
| polars-cv 0.41 | ignores it | none |

The review recommended applying it by default. That is **withdrawn**: the
project's parity reference is PIL, which ignores it, and applying it by
default changes pixels for every existing user, against the bit-identical
decision. The gap is that there is no way to ask for it and no way to see it.

### Decision needed (owner)

Whether the default ever becomes "apply". This plan ships the opt-in and
leaves the default alone.

### Approach

1. Source parameter `orientation: "stored" | "exif"` on `image_bytes`,
   `file_path` and `auto`, default `"stored"`. A `named_variants!` enum in
   the registry, literal-only. It changes height and width per row, which the
   planner already treats as unknown for a decoded image, so it adds no new
   plan-time unknown.
2. Decode: `image` 0.25.9 already exposes it. Replace
   `image::load_from_memory` in `ImageAdapter::decode`
   (`view-buffer/src/interop/image.rs:352`) with a reader that takes the
   decoder, reads `ImageDecoder::orientation()`, decodes, and calls
   `DynamicImage::apply_orientation` when asked. The custom TIFF path and the
   scaled JPEG path (`execute.rs`) need the same step; put it in one function
   both call.
3. Prefer view ops over `apply_orientation` where the result is a
   `ViewBuffer`: the eight orientations are flips and a transpose, which are
   zero-copy here. Measure before choosing.
4. `image_info()`: add an `orientation` field (1–8, null when absent) read
   from the header. Do not change `width`/`height`; they stay the stored
   size, and the docs say how to get the displayed size. Changing the struct
   is a schema change: migration-guide entry.
5. Document it in `concepts/sources.md` and `operations/metadata.md`,
   including that annotations drawn in a viewer usually refer to the
   displayed orientation.

### Tests first

- All eight orientations against `PIL.ImageOps.exif_transpose`, exact, for
  JPEG; plus PNG and WebP with an EXIF chunk, and TIFF.
- `orientation="stored"` is byte-identical to today for the whole golden
  corpus.
- `image_info().orientation` for each of the eight, and null for a file
  without the tag, on bytes and on paths (the header-only read must reach the
  tag without a full fetch).
- With `decode_max_size`: orientation applied after scaling gives the same
  shape rule as before.

### Risks

- The header-only path stops at the frame header today, deliberately skipping
  "large EXIF or comment segments" (`operations/metadata.md`). Reading the
  tag means parsing APP1 on the way; keep the bounded read.
- A malformed EXIF block must not fail a decode that succeeds today: on a
  parse error, treat the tag as absent.

### Done when

A user can get display-oriented pixels with one keyword and can see the tag
without decoding.

---

## CR-140 — No Windows wheel; Python 3.14 untested

**Re-verified: no recorded decision and no platform-specific code found.
Not verified on the platforms themselves.**

### Evidence

`publish.yml` builds Linux x86_64, Linux aarch64 and macOS arm64. CI tests
3.11 to 3.13 on Ubuntu and macOS. `grep` finds no `cfg(windows)`,
`cfg(unix)` or `std::os::unix` in `polars-cv/src` or `view-buffer/src`, and
the CHANGELOG's allocator fix notes "Linux and Windows were not affected", so
Windows was at least considered. The wheel is abi3 (`cp311-abi3`), so 3.14
needs no new build, only a test lane and a classifier.

### Approach

1. Add `windows-latest` to the CI test matrix on pushes to `main` only, as
   macOS is. Expect failures and fix them before touching `publish.yml`.
   Likely areas, none confirmed:
   - `fetch.rs` `PathPolicy`: canonicalisation returns `\\?\` verbatim paths
     on Windows; component-wise matching against `allowed_roots` and the
     `file://` handling need tests with drive letters.
   - `.cargo/config.toml` scopes `lld` and `target-cpu` to x86_64 Linux;
     confirm nothing else assumes it.
   - `allocator.rs` reports a re-entered lookup with a raw write to stderr.
   - Test helpers that assume `/tmp` or POSIX paths.
2. Add the `x86_64-pc-windows-msvc` wheel to `publish.yml`.
3. Add 3.14 to the CI matrix; add the classifier once green. Check the
   `dependency-floors` job still resolves on 3.14.
4. Intel macOS: decide separately; it is a shrinking platform and a wheel per
   release.
5. Update README "Install" and `getting-started/installation.md`.

### Tests first

The CI lane is the test. `tests/test_ci_tool_pins.py` and
`test_build_efficiency.py` pin parts of the workflows; extend them so the
platform list in the README and the matrix in `publish.yml` cannot disagree.

### Done when

`pip install polars-cv` works on Windows x86_64 and the suite is green there
and on 3.14.
