# Academic literature on faster decode for polars-cv: what beats DECODE_RESEARCH.md

_Academic-literature follow-up to `DECODE_RESEARCH.md` (2026-11-17 campaign). Research only: no code was changed. arxiv.org was outside this environment's network allowlist, so paper figures are from abstracts and search summaries._

_Research only, 2026-11-17. No repo changes. Builds on `/home/user/polars-cv/DECODE_RESEARCH.md` ("prior doc"); does not repeat it._

**Bottom line.** The literature does not contain a single-stream, unchanged-encoding entropy decoder that beats libdeflate/zune-class engines on one core; every big parallel/SIMD entropy result (Recoil, PivCo-Huffman, Lepton, Huff0, GPU decoders, rapidgzip) needs either many cores, a GPU, or a re-encoded file. So the prior doc's central claim stands. What the literature *adds* is three exact, planner-driven ways to skip entropy work that the prior doc either missed or under-specified:

1. **Progressive-JPEG scan selection**: skip whole AC scans at memchr speed when the plan needs only DC (1/8 scale, 8×8 block means, global means). Measured: the DC scans are **9–11% of the entropy-coded bytes**. Exact.
2. **Skip-only multi-code tables** for coefficients the plan discards (chroma in luma mode, all AC at 1/8, AC beyond the kept band at 1/4–1/2, rows above a crop). Prior art is the Scalado patent family (~2004–05, now expired by age). Exact.
3. **N/8 scaled IDCT** (IJG jpeg-7+/libjpeg-turbo `jidctint.c`, N = 1…16) instead of power-of-two only, so "auto" downscale lands closer to the resize target. Approximate, in the same class as prior #1.

The ML-from-DCT literature is real and reproducible, but on this system it is capped at about 1.5x decode, because Huffman is about 65% of zune's time. It would also be a new user-facing sink.

## Local measurements (this machine, 1 thread, Pillow 12 / libjpeg-turbo; scripts ran from a session scratch directory, not this repository)

| Measurement | Result |
|---|---|
| Progressive JPEG (libjpeg `jpeg_simple_progression`, q90), byte share of DC scans (first Al=1 + refine) | 512×600 photo: **9.0%** (7.7 + 1.3); 2048×2400: **11.1%** (9.3 + 1.8) |
| Script structure | DC first scan has **Al=1** (LSB missing); the DC refinement scan is **7th of 10**. So "decode only the first scan" is *not* exact, but "decode scans with Ss=0, skip the rest" is. |
| Progressive vs baseline full decode (libjpeg-turbo) | 6.74 vs 4.39 ms (1.5x slower); 91.5 vs 34.8 ms (2.6x slower) |
| libjpeg-turbo 1/8 "draft" on progressive: full file vs DC-scans-only file | 3.81 → 1.05 ms (3.6x); 40.2 → 19.0 ms (2.1x). Outputs **differ** (max 69) because libjpeg applies inter-block smoothing when AC is absent. zune-jpeg has no smoothing (`grep smooth` empty), so a zune-based path would be exact. |
| Cost of finding scan boundaries (numpy 0xFF scan, whole 808 KB file) | 0.27 ms, about 0.7% of the full decode |
| Grayscale JPEG: global mean from DC (1/8 decode) vs pixel mean | 77.125 vs 77.014; 77.081 vs 77.018 (error 0.06–0.11 levels, from clamp and rounding) |
| `block_reduce(8×8, mean)` vs DC image | max \|d\| 0.92, mean \|d\| 0.25 → approximate, not bit-exact |
| Thread pool | polars `THREAD_POOL` = logical CPUs, so SMT hosts already interleave two decodes per core in hardware. This box has no SMT. |

Caveat: there were two test images, one a real photo and one a 4× upscale with noise. Re-measure the progressive scan shares on a real progressive corpus (mozjpeg output uses different scripts).

## Annotated bibliography

### 1. Compressed-domain (DCT-domain) processing

- **Smith & Rowe, "Algebraic operations on compressed images", 1993; Chang & Messerschmitt, IEEE JSAC 1995 (compositing in the DCT domain).** These are the foundations: linear pixel operations (scale, add, blend) map to coefficient operations. *Applicable?* Only as theory. Our sinks are pixels or tensors, and the Huffman floor bounds any saving to the IDCT, upsampling and colour share (about 35%).
- **Shen & Sethi, "Inner-block operations on compressed images", ACM MM 1995; "Direct feature extraction from compressed images", SPIE 1996.** These define the DC image (1/8 thumbnail) and block-level features. *Applicable:* yes, and it is the basis of idea A below. Mehrabi et al. (2016) extend it to 2×2 and 4×4 sub-block means from partial coefficients.
- **Merhav & Bhaskaran, "Fast algorithms for DCT-domain image down-sampling and inverse motion compensation", IEEE TCSVT 7(3) 1997 (HP; US 5708732).** 37–50% computation saving against spatial downsampling, with the output staying in 8×8 DCT syntax. *Not applicable:* this is for transcoding, where the output is DCT. For pixel output it reduces to a scaled IDCT, which the prior doc's #1 already covers.
- **Dugad & Ahuja, "A fast scheme for image size change in the compressed domain", IEEE TCSVT 11(4) 2001.** Halving takes the low 4×4 coefficients and applies a 4×4 IDCT (about 1.25 mult/pixel), roughly +4 dB over bilinear. *Already covered:* this is exactly libjpeg's 1/2 scaled IDCT (prior #1).
- **Mukherjee & Mitra, "Image resizing in the compressed domain using subband DCT", IEEE TCSVT 2002; "Arbitrary resizing of images in DCT space", IEE Proc. VISP 152(2) 2005; Park & Park, IEEE TIP 2006 (arbitrary ratio via composite-length DCT).** These resize by rational L/M directly in the DCT domain. *Approximate.* The kernel is ideal-lowpass per block, not Lanczos3, so output semantics would change. The decode gain over N/8 IDCT plus a pixel resize is negligible because Huffman dominates. **Not recommended.**
- **IJG jpeg-7/8 "DCT scaling" (G. Vollbeding), carried in libjpeg-turbo `jidctint.c` (`jpeg_idct_NxN`, N = 1…16 from one 8×8 block).** This is engineering, not peer-reviewed, but it is the production form of the Dugad–Ahuja and Mukherjee–Mitra line. It produces scales of N/8, not arbitrary ratios. **New relative to the prior doc**, which only considered 1/2, 1/4 and 1/8 (idea C).
- **Feng & Jiang, "JPEG compressed image retrieval via statistical features", Pattern Recognition 2003.** Mean and variance computed directly in the DCT domain (DC plus Parseval over AC). *Applicable* to `reduce_mean`, `reduce_sum` and `reduce_std` on a JPEG source, but approximate (measured mean error 0.06–0.11 levels) → opt-in (idea D). Histograms, min/max and percentiles are **not** computable this way.
- **Martucci, "Symmetric convolution and the DCT/DST", IEEE TSP 1994; Kresch & Malah, IEEE TIP 1999; Viswanath, Mukherjee & Biswas, JVCIR 2011 (block-DCT filtering via symmetric convolution).** Exact linear convolution needs zero padding, which block-DCT data lacks. Block-wise versions show blocking or boundary artefacts for large kernels. **Not applicable:** blur still has to emit pixels, so it saves nothing over IDCT then blur.
- **Lossless DCT-domain flip, transpose and rotate-90 (jpegtran; Park et al. 2023 show all RandAugment ops in DCT).** Exact, but these save only one memory pass in our pipeline. **No gain.**
- **Kang, Mathur, Veeramacheneni, Bailis & Zaharia, "Jointly optimizing preprocessing and inference for DNN-based visual analytics" (Smol), PVLDB 14(2) 2021.** This is the closest *planner* paper. It chooses decode resolution and ROI jointly with the model, decodes only the needed macroblocks or rows, and reports up to **5.9x** end-to-end throughput at fixed accuracy. Its main point is that CPU preprocessing was 9x slower than ResNet-50 on a T4. *Applicable* as validation of prior #1 and #2 (planner-derived scale and region). It adds no new decoder mechanism.

### 2. Learning from compressed representations

- **Gueguen, Sergeev, Kadlec, Liu & Yosinski, "Faster neural networks straight from JPEG", NeurIPS 2018.** ResNet-50 on DCT inputs (Y 28×28×64, CbCr 14×14×64) is **1.77x** faster at the same accuracy. *Relevance:* a `sink("dct")` would emit exactly what jpeg2dct/libjpeg give after Huffman. That is a new API.
- **Xu et al., "Learning in the frequency domain", CVPR 2020.** Static frequency-channel selection gives +1.60% top-1 on ResNet-50 (+0.63% on MobileNetV2), and up to 87.5% of channels can be pruned with little loss. *Relevance:* a DCT sink could keep only low-frequency channels. Huffman still has to parse every coefficient, so the decode saving is the same.
- **Ehrlich & Davis, "Deep residual learning in the JPEG transform domain", ICCV 2019.** Exact convolution and BN in the JPEG domain, with an approximate ReLU; speed is reported only qualitatively. Model-side work, out of scope.
- **Park & Johnson, "RGB no more: minimally-decoded JPEG vision transformers", CVPR 2023.** Decode latency **−61.5%**, ViT-Ti training **+39.2%**, inference **+17.9%**, and a theoretical data-loading cost reduction of up to 85%. All RandAugment ops are implemented on DCT data. *Relevance:* shows that a DCT sink needs DCT-native augmentation, or the gain is lost converting back.
- **dos Santos, Sebe & Almeida, "CNNs for JPEGs: a study in computational cost", arXiv 2012.14426 (rev. 2023).** Shows the DCT-input model often costs more, offsetting the decode saving. This is a skeptical counterweight.
- **Kuchnik, Amvrosiadis & Smith, "Progressive Compressed Records", PVLDB 14(11) 2021.** Read only the first k progressive scans: ≥50% fewer bytes tolerated for many tasks, **1.6–2.6x** training speedup. *Needs re-encoding the dataset* into their record format, so it does not apply to user files. Its mechanism (scan-level skipping) is what idea A uses on unmodified progressive files.
- **Srinivasan et al., JPEG-domain super-resolution, arXiv 2512.04284 (Dec 2025).** 2.6x data-loading speedup (abstract only; not verified).

### 3. Fast entropy decoding

- **Klein & Wiseman, "Parallel Huffman decoding with applications to JPEG files", The Computer Journal 46(5) 2003 (DCC 2000).** Huffman codes self-synchronise, on average in under 72 bits on practical data, and they adapt this to JPEG, including the DC-predictor fix-up. *Applicable* only as single-core 2-way ILP within one stream (idea H). Their result is multi-processor.
- **Weißenberger & Schmidt, "Massively parallel Huffman decoding on GPUs", ICPP 2018; "Accelerating JPEG decompression on GPUs", HiPC 2021 (arXiv 2111.09219).** Up to 51x libjpeg-turbo on an A100. **GPU-only.**
- **Sodsong et al., PMAM 2014 / CCPE 2015 (CPU+GPU split, 5.2x libjpeg-turbo SIMD); JParEnt, CCPE 2017 (4.3x; its CPU boundary-finding pre-scan alone costs 45% of libjpeg-turbo's entropy time).** Heterogeneous or multi-core only. The 45% pre-scan figure is a useful bound: even *locating* codewords costs about half of decoding them.
- **Lin et al., "Recoil: parallel rANS decoding with decoder-adaptive scalability", ICPP 2023.** 90+ GB/s on a GPU, 11+ GB/s with AVX-512. It is rANS and needs encoder-side metadata. **Not applicable.**
- **Żukowski, "PivCo-Huffman", arXiv 2606.05765 (Jun 2026).** A SIMD wavelet-tree Huffman variant that beats Huff0 and Oodle. It is a **new format**. **Not applicable.**
- **Giesen, "Interleaved entropy coders", arXiv 1402.3392 (2014).** Interleaving independent coder states on one core gives **>1.6x** for scalar code and **>2x** with SIMD. Collet's Huff0 (blog, 2015) reports 4 interleaved streams at about 600 MB/s against 350–450 MB/s for the best single-stream multi-symbol table. *Relevance:* this is the only quantitative evidence for prior #7 (cross-image interleaving). It shows the latency-hiding mechanism is real on one core, but both measurements use encoder-designed streams. On SMT hosts, polars' logical-CPU pool already does this in hardware, and it also hides branch mispredictions, which software interleaving does not. **No paper measures cross-file interleaving for JPEG or DEFLATE.**
- **Multi-symbol and skip tables: Lucent patents US 6771824 / 7043088; US 5650905; Scalado patent family US 7711205, 7711195, 7720298, 7965897, 8396306 ("stepping through the JPEG bitstream with a table lookup returning bit length and coefficient count… a 16-bit lookup can return two or more codes… AC need not be decoded").** No peer-reviewed CPU evaluation was found. *Applicable:* skip-only tables for discarded coefficients (idea B). Collet's measurement suggests about 2 symbols per lookup is the practical limit for *decoding*. Skipping can fold the value bits into the length, so it does better.
- **Knespel & Brunst, "Rapidgzip", HPDC 2023; pugz.** 8.7 GB/s on 128 cores. Multi-core only; no single-core gain.
- **Horn et al., "Lepton", NSDI 2017.** Handover words enable 8-thread decode. **Needs a re-encoded file.**

### 4. JPEG decoding

- **Merhav & Vasudev, multiplication-free approximate IDCT, ICIP 1999.** 32% faster IDCT, under 1 dB loss. *Low value:* zune's DC-only shortcut already cuts IDCT to about 18% of decode (zune's own comment), so the ceiling is about 6%, and the result is inexact. Cintra/Bayer approximate DCTs (e.g. arXiv 2410.15244) are evaluated at the transform level only.
- **ROI decoding: Smol (above); patents US 9326000, 9357226 (restart-marker offset index), 8538180.** Restart markers allow exact skipping to any restart interval with a memchr search for RSTn. This refines prior #2 and is idea F.

### 5. PNG / DEFLATE

- **No peer-reviewed single-stream inflate result beats libdeflate-class decoders.** Engineering sources agree that inflate dominates PNG time (W3C public-png, 2025).
- **Apple `iDOT` chunk (de facto; PNG WG 2025 is moving it into the Extensions document); libspng `mARK` restart-marker proposal.** iDOT marks a zlib reset at half the image height. A crop entirely in the bottom half can then start inflating at that offset and skip the top half exactly. This is niche: mainly Apple screenshots and exports (idea F).

### 6. Other

- **AVX-512 VBMI / gather Huffman:** the only results are for encoding (IEEE TCE 2023, +12% over Huff0) or multi-stream formats. Giesen notes that about 32 interleaved streams are needed to fill AVX-512 gathers. **Nothing for single-stream decoding.**
- **Learned or neural decoders:** no credible CPU-throughput result.

## Ranked new ideas (not in the prior doc, or materially changing it)

Gains are single-core throughput per image, in the regime where every core already decodes its own image.

| # | Idea | Plans / ops | Expected gain | Exactness | Cost | Public API change? |
|---|---|---|---|---|---|---|
| **A** | **Progressive JPEG scan selection.** For a DC-only consumer, Huffman-decode only scans with Ss=0 (DC first plus refinement), skip AC scans by searching for the next marker, then do a 1/8 "IDCT" (DC/8+128), upsampling and colour at 1/64 the pixel count. **Corrects prior #4:** "DC scan(s)" must include the late DC-refinement scan (Al=1 in the first scan). | Progressive JPEG + "auto"/explicit downscale to ≤1/8 (12 MP → 224 is 1/8), `block_reduce(8×8)` on block-aligned grids, idea D's statistics | Entropy work 9–11% of bytes (measured). Estimate **4–8x** against a full progressive decode at 1/8. jpeg-decoder's 1/8 path parses every scan, and progressive decodes 1.5–2.6x slower than baseline to begin with. Zero gain on baseline files. | **Bit-exact** against a 1/8 scaled decode, provided no block smoothing (zune has none) | **M** (DC-only progressive path: about 400 lines in a zune fork or own code; security-sensitive parser) | No, if it is behind `decode_max_size` / "auto" (prior #1) |
| **B** | **Skip-only multi-code entropy tables.** For blocks or coefficients the plan discards, a 12–14-bit table returns (total bits, coefficient positions advanced, EOB/ZRL flag) for 1–3 complete code+value units per lookup. Today zune's `discard_mcu_block` (`bitstream.rs:436`) does one lookup plus `get_bits` per symbol. | Baseline JPEG: 1/8 decode (≈95% of symbols discarded), 1/4 and 1/2 (AC past the kept zigzag band), luma-only (prior #3: all chroma blocks), crop rows above the region (prior #2) | Discard loop 1.3–1.6x faster (estimate). Whole decode: **1.2–1.4x on the 1/8 path** (the prior doc measured the 1/8 path at Huffman floor, 1.75 ms against 2.47 ms full), **1.05–1.1x on luma-only**, more on top-crops | **Bit-exact** (it only advances the bit position and DC predictor; DC is still fully decoded) | **M** (zune fork; table build per DHT, about 16–64 KB; fuzz against `discard_mcu_block`) | No |
| C | **N/8 scaled IDCT (N = 1…16)** in place of {1/8, 1/4, 1/2}. "auto" picks the smallest N/8 whose output is at least the resize target. | `decode → … → resize` with ratios of 1.2–8x (e.g. 768² → 224 uses 3/8 → 288², not 1/2 → 384²) | IDCT, upsampling, colour and resize input shrink by (N/4)² against the next power of two: 44% fewer pixels for 3/8 against 1/2. Estimate **+10–25%** end-to-end over prior #1 phase 2 for these ratios; nothing at ≥8x | Approximate, in the same tolerance class as prior #1 (opt-in) | **M** (part of the zune scaled-IDCT work; port NxN kernels from `jidctint.c`, IJG licence notice) | No beyond prior #1 |
| D | **DCT-domain statistics:** `reduce_mean` / `reduce_sum` from DC; `reduce_std` via Parseval on dequantised coefficients; `block_reduce(mean, 8k×8k aligned)` from DC | `decode → [grayscale] → reduce_*` (dataset QC, normalisation statistics) | Baseline: about 1.5x decode (Huffman floor). Progressive + A: about 10x | **Approximate:** mean error 0.06–0.11 levels and block mean error ≤0.92 (measured; from clamp and rounding). RGB stats via the colour matrix add clamp error. | M | **Yes:** needs an explicit opt-in (inexact). **Needs user approval.** |
| E | **SMT note (no code).** Polars' pool already uses logical CPUs, so on SMT hosts two decodes share each core and hide Huffman latency and mispredictions. This captures most of what prior #7's software interleave could. | All | Lowers prior #7's expected value on SMT hosts; benchmark on an SMT host before attempting #7 | Exact | S (benchmark only) | No |
| F | **Marker-indexed skipping for crop pushdown:** JPEG restart intervals (memchr for RSTn, exact DC reset) and PNG `iDOT` (start inflate at the half-height reset) | `decode → crop` where the crop lies below a restart row or below the iDOT midpoint | Rows above the crop cost ≈0 instead of a Huffman parse. Large but rare: DRI is common in camera JPEGs, iDOT in Apple PNGs | Bit-exact | S–M on top of prior #2 | No |
| G | **DCT-tensor sink** (Gueguen 2018; Xu 2020; Park 2023) with DCT-native exact ops (block-aligned crop, flip) | New `sink("dct")` | Decode ≤1.5x (IDCT, upsampling and colour skipped; Huffman stays). Model-side 1.77x per Gueguen, but dos Santos 2023 shows it is often offset | Exact coefficients | L | **Yes: new user-facing feature. Needs user approval.** |
| H | **Klein–Wiseman self-synchronising 2-way split** of one JPEG scan, interleaved on one core | Batch smaller than core count, or single huge images | Research bet, 1.0–1.4x on the Huffman part; branch mispredictions are not hidden | Exact (with DC fix-up and verified sync) | L | No |

Rejected after review: arbitrary-ratio DCT resize (Mukherjee–Mitra, Park & Park); DCT-domain filtering and blur; approximate IDCT; DCT-domain flip/rotate; every GPU, multi-core or re-encoding scheme (Recoil, PivCo, Lepton, PCR, rapidgzip, Weißenberger, Sodsong). Reasons are in the bibliography.

## Top 2: implementation sketches

### A. Progressive scan selection (exact)

**Hook.** This is the same place as prior #1. When the resolved decode scale is 1/8 (from `decode_max_size`, "auto", or a block-aligned `block_reduce(8)` consumer recognised by a node-scope pass in `passes.rs`), `decode_image_bytes` (`polars-cv/src/execute.rs`) calls a `jpeg_dc_only(bytes) -> ViewBuffer` path for SOF2 (progressive) files. Baseline files go to B.

**Decoder.**
1. Parse headers (SOF2, DQT, DHT, DRI, SOS) with zune's header code, or a forked minimal parser.
2. For each SOS:
   - If `Ss == 0` (a DC scan, first or refinement), Huffman-decode it with zune's existing `decode_prog_dc_first` / `decode_prog_dc_refine` into a per-component DC plane (`i16`, one per block).
   - Otherwise skip to the next marker with `memchr(0xFF)`, ignoring `FF00` and `FFD0–D7`. The spec guarantees a scan's entropy data contains no other marker.
3. Output: `clamp(((dc·q0 + 4) >> 3) + 128)` per block. This matches zune's own DC-only IDCT shortcut (`idct/avx2.rs:131`) and keeps the rounding identical.
4. Upsample chroma at the 1/8 scale with the same rule as the scaled path, then colour-convert.
5. EXIF orientation is applied after decoding, as today.

**Guard.**
- Fixture test: for progressive files from libjpeg, mozjpeg and Pillow (4:2:0, 4:4:4, grayscale, with and without DRI), `dc_only(file)` must equal the existing 1/8 path, or zune's full decode block-averaged through the DC formula, byte for byte.
- Watch the guard fail by mutation: drop the refinement scan, which gives an LSB-off DC.

**Benchmark / falsifier.** A regression case: `decode(decode_max_size=auto) → resize(224) → normalize` on a real progressive corpus (re-encode a DIV2K/Kodak subset with mozjpeg defaults *and* libjpeg `-progressive`, at 768² and 12 MP).
- **Accept:** at least 4x per row on 12 MP progressive against the current path.
- **Falsified:** less than 2x, or if mozjpeg scripts put DC and AC in the same scan. Interleaved DC+AC is forbidden by the spec (Ss=0 implies Se=0), so that last case should never occur; the test proves it.
- Also report what share of the user's real data is progressive. If it is under 5%, park the idea.

### B. Skip-only entropy tables (exact)

**Table.** Built once per AC Huffman table (in `HuffmanTable::new`, forked zune). For each W-bit prefix (W = 12 first; try 14), greedily walk complete units where code length + value length ≤ remaining bits. Store a `u32`:
- `bits_consumed` (5 bits)
- `positions_advanced` = Σ(run+1), with ZRL = 16 (7 bits)
- `eob` flag
- `units` count; 0 means fall back to the existing slow path.

**Loop.**
```
while pos < 64 { refill; e = skip[peek(W)]; if e.units==0 { slow_one_symbol(); continue }
  drop(e.bits); if e.eob { break } pos += e.adv }
```
Plus a guard: if `pos + adv > 64`, use the slow path. This catches corrupt streams and matches today's error behaviour.

**Where it is used.**
- `discard_mcu_block` (luma-only chroma; rows above a crop).
- A new `decode_mcu_block_prefix(k)` for scaled decodes: decode normally up to zigzag index k (1/8: k=0; 1/4: the 2×2 band, k=4; 1/2: the 4×4 band, k=24), then switch to the skip loop for the rest of the block.

**Guard.**
- A differential fuzz (cargo-fuzz or a proptest corpus) of `skip` against `discard_mcu_block`: identical bit position and DC predictor after every block, on random valid streams and corrupt ones.
- Watch it fail by flipping the ZRL advance to 15.

**Benchmark / falsifier.** Decode-only micro-bench (prior doc's `zj/` style) on 768² and 12 MP baseline q75/q90 photos:
- Accept: 1/8 decode improves at least 1.2x against zune's discard path; luma-only improves at least 1.05x.
- **Falsified:** the 1/8 gain is under 1.1x. That would mean the loop is bound by refill or branch mispredictions rather than lookups; check `perf stat` for branch misses on an SMT-less host.

## Budget used

- About 35 web searches.
- 2 page fetches, both refused (arxiv.org and ar5iv are not in the session's allowed network domains), so paper claims come from search-result summaries and abstracts. Numbers marked "reported" are the authors' own and were not re-verified.
- About 1 CPU-minute of local Python measurements.
- No repo changes.
