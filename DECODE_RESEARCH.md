# Faster image decoding for polars-cv: what is worth doing

_Research report from the 2026-11-17 optimisation campaign (see `OPTIMIZATION_LOG.md`). Research only: no code was changed._

**Bottom line.** The current decode engines are already at or close to the best available for one core. fdeflate is in the libdeflate/Wuffs class, and zune-jpeg is at least as fast as libjpeg-turbo here. Swapping engines or tuning Huffman decoding is worth 0–15%. Real gains come from **work the planner can avoid**: DCT-domain downscaling fused with resize, crop pushdown into the decoder, and luma-only JPEG decoding.

## Local measurements (this machine: 4-core Xeon 2.1 GHz, 1 thread, no SMT)

The measurement scripts ran from a session scratch directory, not this repository. The standalone Rust bench (`zj/`) was built `--release --offline` outside the workspace. Re-run them before relying on a number for a decision.

| Measurement | Result |
|---|---|
| zune-jpeg 0.5.8, 768² q90 4:2:0, RGB vs Luma output | 2.47 ms vs 1.90 ms → **1.30x** |
| same, 4:4:4 | 3.79 ms vs 2.53 ms → **1.50x** |
| zune Luma output vs the project's grayscale `(77R+150G+29B+128)>>8` of the RGB decode | max \|diff\| **1**, on **2.3%** of pixels (4:2:0 and 4:4:4) |
| jpeg-decoder 0.3.2 (the current `decode_max_size` path), 768² 4:2:0: full / 1/2 / 1/4 / 1/8 | 3.79 / 2.53 / 1.83 / 1.75 ms |
| → jpeg-decoder's scaled path compared with zune's full decode (2.47 ms) | 1/2: **none**; 1/4: 1.35x; 1/8: 1.41x. Huffman decoding is the floor. |
| libjpeg-turbo via Pillow, 768² 4:2:0: luma / 1/2 / 1/4 / 1/8 / luma+1/4, vs full RGB | 1.81x / 1.38x / 1.90x / 2.08x / 2.41x |
| Pillow decode + Lanczos to 224, DCT pre-scale vs full decode, 4000×3000 q90 | 1/2: **3.2x** (57.8 dB); 1/4: **6.7x** (55.6 dB, max diff 2); 1/8: **7.6x** (52.3 dB, max diff 5) |
| same, 768² → 224 at 1/2 | 2.1x (51.5 dB, max diff 7) |

PSNR caveat: these are synthetic smooth+noise images. Real photos have more high-frequency content and will score lower, so re-measure on a real corpus before setting a tolerance.

Inferred PNG inflate rate: 0.54 ms for about 197 KB of output is about 5.8 cycles per output byte. fdeflate already uses a 12-bit table with up to two literals per entry, four chained lookups per 56-bit refill and 8-byte reads (see `fdeflate-0.3.7/src/decompress.rs:459-510`). That is the libdeflate design, and its speed fits a serial table-lookup chain plus branch mispredictions.

## Ranked table

| # | Idea | Applies to | Expected speedup / work avoided | Exact? | Cost / risk | Dependency / licence | Refs |
|---|---|---|---|---|---|---|---|
| 1 | **Planner-derived DCT-domain downscale**: decode at 1/2, 1/4 or 1/8 when a resize to the target follows. Extend the existing explicit `decode_max_size`, e.g. `decode_max_size="auto"`, read from the first resize. Optionally move the scaled path from jpeg-decoder to a zune-based scaled IDCT. | JPEG; `decode → [elementwise] → resize(h,w)` with source ≥2× target; large photos | decode+resize: **3–7.6x for 12 MP → 224** (measured, Pillow); decode alone 1.35–1.4x at 1/4–1/8 (jpeg-decoder vs zune, measured). No decode gain at 1/2 with jpeg-decoder. A zune scaled IDCT would add about 1.2–1.4x at 1/2 (estimate: it removes most of IDCT, colour conversion and upsampling, and keeps Huffman). | Tolerance: 52–58 dB synthetic, max diff ≤5. Must stay opt-in (as `decode_max_size` already is). | "auto": **S**. zune scaled IDCT: **M–L** (fork or upstream; must cover upsampling at reduced scale). | None for "auto". Fork of zune-jpeg is MIT/Apache/zlib. | jpeg-decoder `scale()`; libjpeg `scale_num/denom`; Mozilla bug 2033250 does the same: IDCT scale then Lanczos3, behind an off-by-default pref |
| 2 | **Crop pushdown into the decoder** (when `crop` is the node's first step) | PNG and JPEG; `decode → crop(...)` (random-crop augmentation, ROI extraction) | PNG: inflate stops after row `top+h`. Saves `(H-top-h)/H`: 15% for a centred crop of 50% area, 71% for the top 224 rows of 768, 0% for a bottom crop. JPEG: rows below the crop skipped entirely; rows above still need Huffman but skip IDCT, upsampling and colour; columns outside skip IDCT and colour. About **1.35x** for a centred 50%-area crop, **3–4.5x** for a small top-left crop (estimate, with Huffman about 65% of zune decode). | **Bit-exact** for PNG. Bit-exact for JPEG if decoding is whole iMCU rows plus one upsampling context row. Note libjpeg-turbo's `jpeg_crop_scanline` is *not* exact at the edges with fancy upsampling. | PNG: **S** (`png::Reader::read_row`, stop early). JPEG: **M–L** (zune has no region API; needs a fork or upstream change). Risk: per-row crop parameters must resolve before decode; EXIF orientation maps the rectangle to stored coordinates. | PNG: direct `png` dependency (already transitive). JPEG: zune fork, or TurboJPEG 3 (`tj3SetCroppingRegion`; IJG/BSD-3/zlib licences; C build needs cmake and nasm). | png 0.18 `Reader::read_row`; libjpeg.txt `jpeg_skip_scanlines` / `jpeg_crop_scanline` |
| 3 | **Luma-only JPEG decode** when the plan starts with `grayscale` | JPEG YCbCr; `decode → grayscale → ...` | **1.30x** (4:2:0) and **1.50x** (4:4:4) on decode, measured with zune. The grayscale op itself is also removed. Chroma is still Huffman-parsed (`discard_mcu_block`) but gets no IDCT, upsampling or colour conversion. | **±1 LSB on about 2.3% of pixels**, so opt-in only | **S**: zune-jpeg supports it (`jpeg_set_out_colorspace(Luma)`), but image-rs does not expose it, so call zune directly for JPEG. Must reproduce image-rs's output exactly for the RGB path (fixture test). | zune-jpeg already a transitive dependency (MIT/Apache/zlib) | zune-jpeg `mcu.rs:480-505, 671-712` |
| 4 | Progressive JPEG decoded only through its DC scan(s), stopping before AC refinement, at a 1/8 target | progressive JPEG plus #1's large downscale | Likely 70–90% of decode avoided (most bits are in AC scans; estimate, not measured) | Equal to a 1/8 scaled decode of the same file | M (fork); niche: progressive files only | zune fork | JPEG spec, spectral selection |
| 5 | Decode PNG straight into the ViewBuffer with `png` (no DynamicImage, no zeroed buffer); optionally skip CRC/Adler checks (`ignore_checksums`) | PNG, every plan | **≤5%** (allocation 2% + copy; unfilter is 3%) | Exact; skipping checksums changes error behaviour | S | `png` direct | image-png discussion #416 |
| 6 | zune-jpeg Huffman tuning: pre-unstuff the scan (SIMD scan for 0xFF), 8-byte branchless refill, 10–11-bit lookahead with two coefficients per entry | JPEG, every plan | **5–15%** of decode (estimate). libjpeg-turbo maintainer measured 0–11% (average 3–7%) for a similar 2-table change, against a claimed 40%. A patent reports two coefficients per lookup 44% of the time at 11 bits. | Exact | M; carrying a fork of security-sensitive code | zune fork | libjpeg-turbo PR #657 (closed, "won't integrate", Jan 2026); USPTO 6956511 |
| 7 | Cross-image interleaved entropy decoding: two rows' streams decoded in lockstep on one core to hide the serial lookup-latency chain | PNG inflate (96% of PNG time), JPEG Huffman | Research bet: 0–1.6x. Only pays if the loop is latency-bound, not branch-miss-bound. No published software evidence; hardware patents report 1.9x from 2 symbols per cycle. | Exact | **L**, high risk: hand-written inflate; doubles the working set | Own code | USPTO 9513919 (speculative decompression) |

## Top 3: implementation sketches

### 1. Planner-derived DCT downscale (opt-in)

**Where it hooks in.** The executor already has the exact hook: `decode_image_bytes(bytes, source)` → `source.decode_max_size()` → `decode_jpeg_scaled` (`polars-cv/src/execute.rs:27-94`).

- Add an `"auto"` spelling of `decode_max_size`. Keep the literal form.
- At compile time, the node's plan (`PlannedDecode` in `graph/compiled/run.rs:399`) resolves "auto" to the long side of the node's first `resize` step, provided only steps that commute with scaling come before it: elementwise ops, `cast`, `normalize`, `grayscale`.
- Put this in a node-scope pass in `passes.rs` / `pass_catalog`, so the pass registry owns it. Do not use a side channel.
- The resize step stays. It now reads a smaller input. A per-row resize size (`pl.col`) means the target is unknown at plan time, so "auto" must refuse rather than guess.

**Faster engine (phase 2).** Replace jpeg-decoder's scaled path with a scaled IDCT inside zune-jpeg:

- 1/8: DC only.
- 1/4 and 1/2: reduced IDCT kernels.
- Chroma upsampling applied at the reduced scale.

This gives the 1/2 case a real decode gain; today it gives none (2.53 ms vs 2.47 ms).

**Exactness.** It stays an explicit assertion, as today. The docstring states the tolerance.

**Benchmark that would prove it.** A regression case:

- Pipeline: `decode → resize(224, lanczos3) → normalize` on 12 MP and 768² q90 photos (a real corpus, e.g. Kodak/DIV2K, not synthetic), with and without "auto".
- Accept if: at least 3x per row on 12 MP, at least 1.3x on 768², and PSNR ≥ 40 dB against the full path.
- Falsified if: either speed target is missed, or PSNR falls below the documented tolerance.

### 2. Crop pushdown (bit-exact, automatic)

**The decode request.** Add a `DecodeRequest { region: Option<Rect> }` to `PlannedDecode`, filled by a pass when the node's first step is `crop`.

- Literal crop: computed once.
- Per-row crop: the crop's `Param`s resolve through the same `ParamCtx` *before* `decode_source_row`. Today the steps re-resolve after decode, at `run.rs:~437`.
- Pass the request to `decode_image_bytes`, which returns exactly the cropped buffer. The crop step is then dropped.
- Shape and dtype rules are unchanged, so the planned schema is unchanged.

**PNG.** Decode with `png::Decoder` directly:

- Read rows with `Reader::read_row` and stop after row `top+h`.
- Copy only columns `left..left+w` into the output.
- Interlaced (Adam7) images fall back to a full decode.
- Apply the same transformations image-rs uses (EXPAND, 16-bit to native endianness).

**JPEG.** Patch zune-jpeg:

- Stop the MCU loop after the iMCU row containing `top+h`, plus one context row for vertical fancy upsampling.
- For MCU rows above the crop, and blocks outside its columns, use `discard_mcu_block` (no dequantisation, no IDCT).
- Run colour conversion only on the output rows.
- Restart markers, when present, would also let Huffman skip rows above the crop by scanning for RSTn. That is optional.

**Orientation.** With `orientation="exif"`, map the rectangle back to stored coordinates first.

**Tests.** A fixture over random rectangles, both chroma subsamplings and all 8 EXIF orientations must give `decode(crop pushed) == crop(decode)` byte for byte.

**Benchmark.** Top 224 rows of a 1024² PNG: expect at least 3x. Centred 50% crop: expect about 1.15x (PNG) and about 1.3x (JPEG). The change is falsified if a bottom-aligned crop shows a regression.

### 3. Luma-only JPEG (opt-in)

**What changes.** When the first step is `grayscale` and the source allows inexact luma (one explicit source flag, the same pattern as `decode_max_size`):

- Decode JPEG with zune-jpeg directly, using `DecoderOptions::jpeg_set_out_colorspace(ColorSpace::Luma)`.
- Produce `[H,W,1]`, or `[H,W]` if `keepdims=false`, and drop the grayscale step.
- Non-YCbCr JPEGs (CMYK, Adobe RGB transform) and non-JPEG formats keep the normal path.

**Guards.**
- The RGB path through direct zune must match `image::load_from_memory` byte for byte.
- The luma path must be within 1 LSB of `grayscale(decode)`.

**Benchmark.** `decode → grayscale → resize(224)` on 4:2:0 photos. Expect 1.25–1.35x on decode and slightly more end to end, since the grayscale op disappears. It is falsified if the gain is under 1.15x.

**Combining with #1.** Luma at 1/4 scale measured 2.4x in libjpeg-turbo.

## Rejected / not worth it here

- **Inflate swap (libdeflate, ISA-L igzip, zlib-ng, zlib-rs, Wuffs):**
  - fdeflate measured 2.1x stock zlib here.
  - Wuffs' own table puts libdeflate and Wuffs at 1.3–1.7x zlib (≈500 vs 323 MB/s on a 100k zlib stream).
  - fdeflate claims parity with zlib-ng and zlib-rs on PNG data.
  - igzip is roughly at libdeflate's level, ±15% depending on platform.
  - Expected 0.9–1.15x over fdeflate, plus a C/asm build (nasm, aarch64) for little or nothing.
- **zune-png:** measured 0.34x.
- **Wuffs or spng as a whole PNG decoder:** 2024–2026 corpus benchmarks show image-png level with or ahead of Wuffs and spng (≈375–410 MP/s vs 317–381).
- **Fusing inflate+unfilter, SIMD Paeth:** unfilter is 3% of PNG time and png 0.18 already has SIMD Paeth. The ceiling is ≤3%.
- **Decoding straight into the target dtype (f32):** saves one cast pass, a few percent. It is better done as elementwise fusion downstream.
- **libjpeg-turbo / TurboJPEG as the main JPEG engine:** no single-core gain over zune-jpeg 0.5.8. Here zune decoded in 2.47 ms against 4.0 ms through Pillow, though Pillow adds overhead. It would add a C build.
- **jpegli:** an encoder project. Google makes no decode-speed claim and recommends libjpeg-turbo for decoding.
- **Intra-image parallelism (rapidgzip-style speculative inflate, restart-marker parallel JPEG, multi-threaded post-processing):** adds no throughput when every core already decodes its own row. It helps only batches smaller than the core count or single huge images.
- **AVX-512 VBMI / SIMD Huffman decoding:** I found no published 2023–2026 single-stream decoder result. The SIMD work is on encoding (IEEE TCE 2023), or relies on multiple streams laid out by the encoder (Huff0, rANS/Recoil). Neither applies to existing PNG/JPEG files.
- **GPU (nvJPEG):** out of scope.

## Sources

- Wuffs benchmarks: https://github.com/google/wuffs/blob/main/doc/benchmarks.md
- image-png performance discussion: https://github.com/image-rs/image-png/discussions/416
- libjpeg-turbo PR #657: https://github.com/libjpeg-turbo/libjpeg-turbo/pull/657
- libjpeg.txt: https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/doc/libjpeg.txt
- zlib-ng issues #986 and #1486 (via search snippets)
- zlib-rs blog: https://trifectatech.org/blog/zlib-rs-is-faster-than-c/
- r/rust "memory-safe PNG decoders now vastly outperform C" (Dec 2024, via snippet)
- jpegli announcement: https://opensource.googleblog.com/2024/04/introducing-jpegli-new-jpeg-coding-library.html
- Mozilla bug 2033250 (via snippet)
- Patents: USPTO 6956511, 9513919
- Local crate sources: fdeflate-0.3.7, zune-jpeg-0.5.8, png-0.18.0
