# Phase 9b: JPEG encoder evaluation

`image::codecs::jpeg::JpegEncoder` (production, through
`ImageAdapter::encode_jpeg`) against `jpeg-encoder` 0.7.1 (`simd`), on a
synthesized corpus: a gradient with mild noise (smooth, like large areas of
a photo) and uniform noise, at 256²/512²/1024², L8 and Rgb8, quality 75 and
90. Wheels' target (`RUSTFLAGS="-C target-cpu=x86-64"`), release, median of
15–41 encodes per cell after a warm-up, nothing else running.

**The eval is a standalone crate, not a view-buffer test.** jpeg-encoder is
IJG-licensed, and the workspace's `cargo deny` checks dev-dependencies too, so
a `view-buffer/tests/jpeg_encode_eval.rs` (as the plan had it) would itself
have required allowing IJG, which the owner reserved for adoption. Its source
is here: `eval-Cargo.toml` and `eval-main.rs`; to rerun, copy them into an
empty directory as `Cargo.toml` and `src/main.rs`, then `cargo run --release`.

jpeg-encoder is run at 4:4:4 (`SamplingFactor::F_1_1`), the production
encoder's sampling (image 0.25 writes every component 1×1), so the outputs
are like for like. Its default, 4:2:0, is timed for information.

## Gate (the owner's decision): geomean speedup ≥ 1.5x, PSNR within 0.5 dB

| run | geomean speedup (4:4:4) | worst \|ΔPSNR\| | verdict |
|---|---:|---:|---|
| 1 | 1.65x | 0.04 dB | passes |
| 2 | 1.62x | 0.04 dB | passes |
| 3 | 1.65x | 0.04 dB | passes |

Per cell (run 1; `raw/` has all three):

| pattern | color | size | q | image ms | jpeg-encoder ms | speedup | 4:2:0 ms | PSNR image | PSNR jpeg-encoder | bytes image | bytes jpeg-encoder |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gradient | L8 | 1024 | 75 | 7.76 | 4.89 | 1.58x | 4.90 | 41.10 | 41.11 | 38,275 | 42,147 (+10%) |
| gradient | L8 | 1024 | 90 | 10.07 | 8.61 | 1.17x | 6.86 | 42.23 | 42.27 | 111,668 | 121,176 (+9%) |
| gradient | Rgb8 | 1024 | 75 | 24.34 | 20.35 | 1.20x | 8.37 | 40.40 | 40.42 | 42,208 | 44,177 (+5%) |
| gradient | Rgb8 | 1024 | 90 | 27.90 | 15.78 | 1.77x | 9.68 | 40.88 | 40.84 | 114,057 | 128,897 (+13%) |
| noise | L8 | 1024 | 75 | 20.18 | 11.69 | 1.73x | 11.15 | 28.53 | 28.54 | 582,071 | 582,881 (+0.1%) |
| noise | L8 | 1024 | 90 | 22.67 | 10.72 | 2.12x | 16.41 | 36.39 | 36.43 | 823,199 | 824,399 (+0.1%) |
| noise | Rgb8 | 1024 | 75 | 58.11 | 34.32 | 1.69x | 17.49 | 22.27 | 22.27 | 1,313,993 | 1,315,976 (+0.2%) |
| noise | Rgb8 | 1024 | 90 | 65.33 | 35.00 | 1.87x | 17.66 | 29.94 | 29.96 | 1,957,891 | 1,960,856 (+0.2%) |

## Outcome: the gate passes; not adopted (the owner's decision)

The gate the owner set passes in all three runs. One cost the gate does not
measure: **at the same quality setting, jpeg-encoder's files are 5–13%
larger on smooth content** (0.1–0.3% on noise), for the same PSNR (±0.04
dB). Put to the owner as new information for an output-changing decision,
they kept image's encoder (2026-09-28). IJG stays out of `deny.toml`, and
JPEG bytes are unchanged. Should it be revisited, the handover's Phase 9
section lists what adoption involves.
