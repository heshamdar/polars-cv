# Post-merge validation of PR #101 (typed-op migration)

Base: `ac2e95b` (main before PR #101). Candidate: this branch after the two
performance fixes below. Both built `maturin build --release` (fat LTO) and
installed into one venv, so the harness, polars (1.42.0) and every other
dependency are identical. Machine: 4-core cloud container, 15 GB.

## Correctness: outputs are unchanged

`outputs_base_vs_head.txt`: every benchmark single op and pipeline, through
both adapters (streaming and eager), on four images (three 256×256 patterns
and a 97×131 one). All 292 comparable outputs are **byte-identical** between
base and candidate, so each one's error against OpenCV is unchanged too (the
non-zero ones are pre-existing algorithm differences, e.g. border handling).
The only difference is `crop_center` on the 97×131 image: its 128×128 window
runs past the image, which base silently shrank and the candidate refuses
(CR-42, in the CHANGELOG).

## Two regressions found and fixed

Measured against base, isolated with same-profile variant builds:

| cause | cost before the fix | fix |
|---|---|---|
| `aligned_copy` (CR-41) assembled a binary row 8 bytes at a time | 72 µs vs 10 µs per 196 KB row | one `copy_from_slice` into a `Vec<u64>` |
| every call split its rows over the plugin pool (CR-32), including the streaming engine's concurrent morsel calls | no-op `blob → blob` streaming 2.6x slower per row at 1 thread, ~2x at 4 | split only a call that runs alone after a call that ran alone (`CallTracker`) |

## Throughput

Regression harness (`single_ops,pipelines`, 300 × 256², base and candidate
run back to back). Throughput change, candidate vs base:

| configuration | eager | streaming |
|---|---|---|
| 4 threads (`t4-*.json`) | +195% to +387% on all 25 | noisy here (see below) |
| 1 thread, the harness's calibrated gate (`t1-*.json`) | 22 neutral, +51% sharpen, +21% sobel_x; −8 to −10% crop, normalize, rotate_90 | +14% to +73% on scalar ops and pipelines; −7% to −17% on flips, crop, resize, threshold, histogram_equalize, heavy_pipeline |

Noise: the same base binary re-run two hours apart moved by up to ±19% at 4
threads on this container, so only back-to-back runs are compared. A focused
best-of-80 streaming run at 4 threads (the harness's decoded-blob path, 300
images) put the candidate within ±5% of base on resize, grayscale, flip,
blur and invert.

Open: the 1-thread single-op regressions are small and inconsistent across
engines (`flip_vertical` −17% streaming, −5% eager, with identical per-row
code at one thread); they want a quieter machine before being chased.

## Plan build

`python -m benchmarks.plan_build` (µs per builder append, best of 7):

| scenario | base | candidate |
|---|---:|---:|
| chain | 110.2 | 15.8 |
| mixed | 213.2 | 17.9 |
| lazy continuation | 104.9 | 17.4 |
