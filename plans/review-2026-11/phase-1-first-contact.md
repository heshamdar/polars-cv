# Phase 1 — Correctness and first contact

Implementation plan for the first group of findings from the external review
of 0.41.0 (commit `c0d77d5`, 2026-11-11). The ledger entries are CR-136 to
CR-140 in the root [`ISSUES.md`](../../ISSUES.md); this file holds the detail.
Phases [2](phase-2-stabilise.md) and [3](phase-3-depth.md) follow.

**Goal:** remove the five things that make a first evaluation fail. Nothing
here adds an op.

**Order:** CR-137, CR-140. CR-136 and CR-138 are closed (`ISSUES.md`, Closed
findings). CR-139 and the default of CR-137 each wait on a decision.

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

## CR-139 — `decode_max_size` is no faster at scale 1/2 (**decision**)

**Scale 1 is fixed** (`decode_jpeg_scaled` returns `None` when the factor is
1, so the row takes the full decoder, byte for byte) and the regression suite
has a case per factor (`targeted:codec_jpeg_decode_half`/`_quarter`/
`_eighth`, beside `codec_jpeg_decode`). What is left is one decision.

### Evidence

Decode only, into the numpy sink, median of 9, two rounds, uncontended, a
`--profile benchmark` build on a 4-core container (2026-11-11). Ratios are
against the full decode of the same round:

| Image | Rows | Full | 1/2 | 1/4 | 1/8 |
|---|---:|---:|---:|---:|---:|
| 620x496 | 300 | 98.5 ms | 1.08x | 0.82x | 0.73x |
| | | 105.5 ms | 1.04x | 0.84x | 0.74x |
| 1280x960 | 100 | 136.1 ms | 1.02x | 0.84x | 0.72x |
| | | 148.6 ms | 0.96x | 0.72x | 0.66x |
| 4000x3000 | 12 | 218.7 ms | 0.85x | 0.68x | 0.62x |
| | | 215.2 ms | 0.96x | 0.72x | 0.64x |

1/4 and 1/8 always win. 1/2 is within a few percent of the full decode either
way: the scaled path (`jpeg-decoder`) is a slower decoder than the full one
(`zune-jpeg`, which has no scaled IDCT), and at 1/2 that cancels what the
smaller IDCT saves. It still returns a quarter of the pixels, so the memory
and every later op are cheaper; the docs now say so.

### Decision needed (owner)

Whether 1/2 falls through to the full decoder as scale 1 does.

- **Keep 1/2 (as now).** No decode-time gain, a real memory and downstream
  gain. The plan's "no factor slower than the full decode" is then not met
  on small images, by 2 to 8%.
- **Fall through at 1/2.** The option becomes "1/4 and smaller". A caller
  who gets half-size rows today gets full-size ones: a shape change, not
  only a pixel change, and more memory.

Recommendation: keep it. The option is an assertion about how many pixels
the pipeline needs, and honouring it at 1/2 costs nothing measurable.

A longer-term option is a decoder with a fast scaled IDCT in place of
`jpeg-decoder`. Not planned: a new dependency for one option.

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
