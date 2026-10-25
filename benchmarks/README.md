# polars-cv vs PIL, OpenCV, torchvision, Daft and Hugging Face datasets

A reproducible comparison on COCO val2017: four workloads, ten
implementations, two input sources, one thread and all cores. Every
implementation is held to the PIL reference by parity tests before anything is
timed, and every number below comes from `results/*.json`, which also records
the hardware and every library version.

This directory is its own uv project. It benchmarks the **published**
`polars-cv` wheel (pinned in `pyproject.toml`), which is what users install,
not the local checkout. The regression suite that measures the checkout lives
in [`polars-cv/benchmarks/`](../polars-cv/benchmarks/).

The results below were measured before 0.38.0 reached PyPI, with a wheel built
from the release commit by `maturin build --release`, the profile the
published wheels use (its source hash, `51bf4cc9965f3148`, is the one
`polars_cv.build_info()` reports for 0.38.0). Each results file records
`polars-cv 0.38.0` as installed and `0.37.0` from `uv.lock`, since the lock can
only name a version PyPI serves. The pin moves to 0.38.0 when it is published.

## Results

4 threads (all cores), COCO val2017, files read from paths, polars-cv 0.38.0.
Images per second is the median of 5 runs, with the interquartile range in the
full tables. Peak memory is the PSS of the process and its workers.

![Throughput per workload at 4 threads](results/summary.svg)

| Workload (5,000 images) | polars-cv eager / streaming | Best other | Next best | polars-cv peak memory |
|---|---:|---:|---:|---:|
| **w1** decode → 224² → gray → numpy | **988 / 1,025** | torchvision 701 | OpenCV 665 | **631–849 MiB** (OpenCV 992, torchvision 1,094) |
| **w2** thumbnail → JPEG | 690 / 728 | **OpenCV 939** | torchvision 705 | 247–250 MiB (OpenCV 194) |
| **w3** size + pHash → duplicate groups | 1,098 / 1,048 | **OpenCV 1,180** | PIL 650 | 138–143 MiB (OpenCV 120) |
| **w4** per-row crop + resize | **1,119 / 1,132** | OpenCV 751 | torchvision 704 | **678–686 MiB** (OpenCV 2,238, torchvision 1,879) |

**Where polars-cv wins**

- **w1 and w4 run 1.4–1.5× faster than the best alternative, in less
  memory**: w1 in up to 40% less than torchvision, w4 in under a third of
  OpenCV's. Decode, resize and grayscale run in Rust on one
  pool with no per-image Python, and w4's per-row crop box and size are just
  column expressions. The process-pool libraries pickle every result back to
  the parent; polars-cv writes straight into one Arrow buffer.
- **It holds up at 50,000 images.** w1 runs at 1,035 img/s (eager) and 1,062
  (streaming), against 757 for torchvision and 603 for OpenCV, and peaks at
  4.9–5.1 GiB, the lowest of all (torchvision 5.3, Daft 5.4, PIL and OpenCV
  11.0–11.2). Most of that is the 2.3 GiB output array itself. Hugging Face
  `datasets` passed the 12 GiB cap on w1 at 50k and was stopped. On w3 at 50k
  polars-cv and OpenCV are level (1,077–1,122 against 1,119 img/s), and every
  library finds the same 5,000 duplicate groups.

**Where polars-cv loses, and why**

- **w2: OpenCV is 1.3× faster.** polars-cv decodes and resizes as fast as
  OpenCV, but its JPEG encoder, the pure-Rust `image` crate's, takes 3.2 ms
  for a 256-px thumbnail where libjpeg-turbo takes 0.27 ms (single thread,
  measured separately). OpenCV's `INTER_LINEAR` is also cheaper than an
  antialiased resize (see Caveats).
- **w3: OpenCV is 1.1× faster at 5,000 images.** A single thread spends 3.4–3.6
  ms per image in polars-cv, decode and hash together, against 3.1 ms in
  OpenCV.

**What changed since 0.37.0**

- **w3 is 2.1× faster and correct.** `perceptual_hash()` is now standard
  pHash (imagehash's algorithm, within 2 bits of it on each of the 40 parity
  images). In
  0.37.0 it ran at 491–506 img/s here, put 236 of the 5,000 distinct val2017
  images into 64 false "duplicate" groups, and found 4,828 groups instead of
  5,000 at 50k. It now finds 0 and 5,000, like every other library.
- **Elsewhere the changes are within a run-to-run spread of a few percent**:
  w4 +7% (eager and streaming), w2 +1% eager and +16% streaming, w1 -7% eager
  and +6% streaming. The 0.37.0 numbers came from another run two days
  earlier on the same hardware type, and re-measuring w1 hours apart in this
  run moved torchvision by 5-15%, so the host accounts for part of each. A
  same-session A/B of the two wheels, alternated over three rounds, put
  0.38.0 at or above 0.37.0 on w1 and w4.

**polars-cv's own pool, on and off**

`polars-cv-streaming-nopool` is the streaming engine with
`polars_cv.set_row_parallelism(False)`: each plugin call runs its rows on the
thread polars called it on, so the engine's morsels are the only parallelism.
At 4 threads it is within ±5% of `polars-cv-streaming` on every workload (w1
963 against 1,025, w2 754 against 728, w3 1,134 against 1,048, w4 1,086
against 1,132), and identical at 1 thread. While every core already runs a
morsel the pool has no idle thread to lend, so it mostly helps at the end of
a query, when the last morsels run with cores to spare. The switch is for
hosts where waking the pool costs more than that, not a speed-up here.

Every result, with both sources and both thread counts (the 1-thread cells
were measured for the 5k set), is in the tables below and in `results/*.json`.

<details>
<summary>Full tables</summary>

#### coco-50k, w1, path source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 1,035 (1,032–1,051) | 5,258 |
| polars-cv-streaming | 1,062 (1,040–1,068) | 4,975 |
| polars-cv-streaming-nopool | 1,077 (1,072–1,089) | 5,090 |
| pil-mp | 446 (441–448) | 11,424 |
| opencv-mp | 603 (579–608) | 11,284 |
| torchvision | 757 (753–763) | 5,470 |
| daft | 277 (274–286) | 5,549 |
| hf-datasets | memory-limit | — |

#### coco-50k, w3, path source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 1,077 (1,069–1,110) | 168 |
| polars-cv-streaming | 1,092 (1,092–1,095) | 158 |
| polars-cv-streaming-nopool | 1,122 (1,112–1,127) | 160 |
| pil-mp | 609 (604–616) | 297 |
| opencv-mp | 1,119 (1,108–1,140) | 173 |
| torchvision | n/a | — |
| daft | 432 (426–442) | 2,813 |
| hf-datasets | 569 (561–584) | 8,619 |

#### coco-5k, w1, bytes source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 934 (906–949) | 2,185 |
| polars-cv-streaming | 992 (984–1,007) | 2,412 |
| polars-cv-streaming-nopool | 953 (923–968) | 2,376 |
| pil-mp | 466 (454–488) | 2,010 |
| opencv-mp | 667 (666–675) | 1,971 |
| torchvision | 678 (665–716) | 1,927 |
| daft | 102 (100–102) | 10,257 |
| hf-datasets | 394 (394–406) | 5,311 |

#### coco-5k, w2, bytes source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 665 (643–671) | 1,801 |
| polars-cv-streaming | 714 (695–733) | 1,806 |
| polars-cv-streaming-nopool | 677 (657–682) | 1,795 |
| pil-mp | 510 (479–541) | 1,013 |
| opencv-mp | 833 (819–855) | 990 |
| torchvision | 719 (704–721) | 1,639 |
| daft | n/a | — |
| hf-datasets | 519 (503–548) | 3,147 |

#### coco-5k, w3, bytes source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 969 (923–978) | 1,689 |
| polars-cv-streaming | 1,015 (996–1,065) | 1,695 |
| polars-cv-streaming-nopool | 897 (847–928) | 1,684 |
| pil-mp | 601 (592–608) | 1,027 |
| opencv-mp | 1,132 (1,113–1,187) | 916 |
| torchvision | n/a | — |
| daft | 90 (90–91) | 9,341 |
| hf-datasets | 565 (552–578) | 2,053 |

#### coco-5k, w4, bytes source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 1,094 (1,056–1,103) | 2,229 |
| polars-cv-streaming | 1,132 (1,119–1,171) | 2,239 |
| polars-cv-streaming-nopool | 1,052 (1,024–1,067) | 2,228 |
| pil-mp | 560 (554–584) | 3,138 |
| opencv-mp | 800 (734–837) | 3,503 |
| torchvision | 720 (711–722) | 2,647 |
| daft | n/a | — |
| hf-datasets | 309 (299–357) | 8,243 |

#### coco-5k, w1, path source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 988 (970–1,006) | 631 |
| polars-cv-streaming | 1,025 (1,014–1,059) | 849 |
| polars-cv-streaming-nopool | 963 (921–988) | 844 |
| pil-mp | 501 (494–503) | 867 |
| opencv-mp | 665 (661–688) | 992 |
| torchvision | 701 (665–704) | 1,094 |
| daft | 303 (300–308) | 2,461 |
| hf-datasets | 492 (440–497) | 2,753 |

#### coco-5k, w2, path source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 690 (681–702) | 247 |
| polars-cv-streaming | 728 (706–738) | 250 |
| polars-cv-streaming-nopool | 754 (723–760) | 245 |
| pil-mp | 587 (577–606) | 214 |
| opencv-mp | 939 (882–968) | 194 |
| torchvision | 705 (683–706) | 824 |
| daft | n/a | — |
| hf-datasets | 538 (530–567) | 1,760 |

#### coco-5k, w3, path source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 1,098 (1,080–1,100) | 138 |
| polars-cv-streaming | 1,048 (972–1,058) | 143 |
| polars-cv-streaming-nopool | 1,134 (1,067–1,164) | 121 |
| pil-mp | 650 (625–652) | 232 |
| opencv-mp | 1,180 (1,168–1,224) | 120 |
| torchvision | n/a | — |
| daft | 478 (468–481) | 2,496 |
| hf-datasets | 577 (576–593) | 1,355 |

#### coco-5k, w4, path source

| library | 4 thr img/s (IQR) | 4 thr peak MiB |
|---|---:|---:|
| polars-cv-eager | 1,119 (1,117–1,181) | 678 |
| polars-cv-streaming | 1,132 (1,071–1,148) | 686 |
| polars-cv-streaming-nopool | 1,086 (1,072–1,132) | 677 |
| pil-mp | 593 (572–603) | 2,813 |
| opencv-mp | 751 (749–841) | 2,238 |
| torchvision | 704 (682–744) | 1,879 |
| daft | n/a | — |
| hf-datasets | 332 (300–373) | 6,708 |

wrote SVGs to /tmp/claude-0/-home-user-polars-cv/dd6d2399-d7a6-5794-b2e6-273873ba73d8/scratchpad/plotcheck

</details>

Charts: [throughput, 5k, paths](results/throughput-coco-5k-path.svg) ·
[throughput, 5k, bytes](results/throughput-coco-5k-bytes.svg) ·
[throughput, 50k](results/throughput-coco-50k-path.svg) ·
[memory, 5k](results/memory-coco-5k.svg) · [memory, 50k](results/memory-coco-50k.svg)


## Reproduce

```bash
cd benchmarks
uv sync                                   # exact versions from uv.lock
uv run python -m bench.data --scale 50000 # COCO val2017 (815 MB, checksummed) + a 50k set
uv run --group dev pytest                 # parity first; -m "not slow" skips the 5k-image check
uv run python run.py --threads 1,max --out results/coco-5k.json
uv run python run.py --dataset coco-50k --source path --workload w1,w3 \
    --mem-limit-gib 12 --out results/coco-50k.json
uv run python plot.py results/*.json
```

`run.py` takes `--workload`, `--library`, `--threads` (all cores by default; `1,max` adds the 1-thread cells),
`--source` (`bytes,path`), `--dataset`, `--runs` and `--warmup`, so a single
cell is, for example,
`uv run python run.py --workload w1 --library polars-cv-eager --threads 4 --source path`.
A manually triggered workflow (`.github/workflows/benchmark-comparison.yml`)
runs the parity tests and a 200-image smoke version on a GitHub runner.

## Methodology

### Data

COCO val2017: 5,000 JPEGs, mostly 640 px on the long side, 10 of them
single-channel. `bench/data.py` downloads `val2017.zip` from cocodataset.org
(or the fast.ai S3 mirror of the same file) and verifies COCO's published MD5
and a pinned SHA-256. `coco-5k.parquet` lists the images with their header
dimensions and workload 4's parameters, drawn from a seeded RNG. `coco-50k`
repeats the 5k images ten times under new file names (hard links), so the bytes
are real and identical copies exist for workload 3 to find.

Two sources: **bytes** means the encoded files are already in memory (a Polars
`Binary` column, a Python list of `bytes`, a memory-mapped `datasets` Arrow
file, a Daft frame), and **path** means each job reads the files. The files sit
in the page cache, so `path` measures the read path's CPU cost, not disk speed.

### Workloads

| | Work | Output |
|---|---|---|
| **w1** ML preprocessing | decode → bilinear resize to 224×224 → grayscale (BT.601) | one `(N, 224, 224)` uint8 array |
| **w2** thumbnails | decode → fit in 256×256, keep aspect, never upscale (bilinear) → JPEG quality 85 | N JPEG byte strings |
| **w3** dataset QA | width, height, channels and a 64-bit pHash per image → group equal hashes | a per-image table and the duplicate groups |
| **w4** per-row crop | crop a box and resize to a size, both read per row from columns (bilinear) | N arrays of varying shape |

### Implementations

Each library does the work the way someone who knows it would write it. The
code is in [`bench/workloads.py`](bench/workloads.py).

| Library | How |
|---|---|
| `polars-cv-eager` | `Pipeline` → `.cv.pipe(...).sink(...)` in `df.lazy()...collect(engine="in-memory")`. w1 uses the fixed-shape `array` sink and `to_numpy()`. w3 reads width/height/channels from the header (`.cv.width()` etc.) and groups with `group_by`. w4's crop box and size are `pl.col(...)` parameters. |
| `polars-cv-streaming` | The same queries, `collect(engine="streaming")`. |
| `polars-cv-streaming-nopool` | The same streaming queries with `polars_cv.set_row_parallelism(False)`, so each plugin call runs its rows on the thread polars called it on and the morsels are the only parallelism. polars-cv 0.38 or newer (N/A before). |
| `pil` / `pil-mp` | A loop over `Image.open`. `-mp` is a `fork` `multiprocessing.Pool` sent indices, with the inputs inherited through the fork, so only results are pickled. w2 uses `Image.thumbnail` (which calls `draft()`), and w3 uses `imagehash.phash`. |
| `opencv` / `opencv-mp` | `cv2.imdecode`/`imread` (EXIF orientation off, as PIL), `cv2.resize(INTER_LINEAR)`, `cvtColor`, `imencode`. w3 uses `cv2.img_hash.pHash` (opencv-contrib). Workers call `cv2.setNumThreads(1)`. |
| `torchvision` | A `Dataset` + `DataLoader(batch_size=64, num_workers=N)` (0 workers at 1 thread), `torchvision.io.decode_image`, transforms v2 `Resize(antialias=True)`/`Grayscale`, `encode_jpeg`. |
| `daft` | The native runner (`set_runner_native(num_threads=N)`) with `decode_image`, `resize`, `convert_image`, `image_hash`, and `download` for paths. w3 groups with Daft's `groupby`. |
| `hf-datasets` | A memory-mapped `Dataset` with an `Image()` feature and `map(batched=True, num_proc=N, keep_in_memory=True)` calling PIL, with caching disabled. w3 groups with `to_polars()`. w4 returns raw bytes and a shape, which avoids the per-element NumPy formatter. |

`pil` and `opencv` run at 1 thread only (a loop is one process), and `pil-mp`
and `opencv-mp` at all cores only (a pool of one is the loop plus overhead), so
each library's 1-thread and N-thread rows are the loop and the pool.

**Not applicable**, rather than replaced with something else:

| Library | Workload | Why |
|---|---|---|
| torchvision | w3 | No perceptual hash. A torchvision user would call imagehash on PIL images, which is the PIL row. |
| Daft | w2 | `resize()` takes a literal width and height, so it cannot keep each image's aspect ratio, and `encode_image()` has no JPEG quality setting. |
| Daft | w4 | `resize()` takes literal sizes. A per-row size needs a Python UDF around PIL, which is the PIL row. |
| Ray Data | all | Not run. A Ray cluster on a 4-core box adds a scheduler, an object store and per-task overhead that would dominate these workloads, and installing and tuning it fairly is a project of its own. |

### Correctness parity

`tests/test_parity.py` runs every applicable (library, workload, source) on a
40-image sample and compares it with the PIL reference. The sample holds all 10
single-channel images and 4 repeated images, so w3 has groups to find. Each
tolerance in [`bench/parity.py`](bench/parity.py) sits just above the measured
difference and states where the difference comes from. A wrong size or channel
count fails as a shape mismatch, and a flipped or un-grayed image moves the
mean by tens of levels, far above any bound. The measured differences are in
[`results/parity.json`](results/parity.json):

| Library | Measured against PIL (w1/w4: pixel levels, w2: decoded JPEG levels) | Why |
|---|---|---|
| polars-cv (all three) | w1 mean 0.17 (max 2); w2 mean 2.0, same sizes; w4 mean 0.16 (max 2); w3 within 0.15 bits of imagehash (max 2), same duplicate groups | Another JPEG decoder and encoder; the same area-weighted bilinear and BT.601 gray; the engine's own grayscale and Lanczos resize before the DCT |
| torchvision | w1 mean 0.66 (max 2); w2 mean 0.03; w4 mean 0.00 (max 1) | Antialiased bilinear with other rounding; libjpeg-turbo on both sides |
| OpenCV | w1 mean 4.2 (max 127); w2 mean 5.7; w4 mean 4.8 (max 136); w3 same sizes, same groups | `INTER_LINEAR` does not antialias when downscaling |
| Daft | w1 mean 1.2 (max 23); w3 within 0.5 bits of imagehash, same groups | Its resize filter support differs |
| PIL loop/pool, HF datasets | exact | They are the reference code |

OpenCV's pHash, and polars-cv's before 0.38, are not bit compatible with
imagehash (about 32 differing bits, i.e. uncorrelated), so their bits are not
compared. The duplicate groups are, and must match. From 0.38 polars-cv
computes imagehash's algorithm, and its bits are held to it (mean at most 0.5,
max 3); the installed version decides which bound applies.

### Measurement

- **Process per cell.** Each (workload, library, threads, source) cell runs in a
  fresh interpreter. `POLARS_MAX_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `RAYON_NUM_THREADS`,
  `DAFT_NUM_THREADS` and `VIPS_CONCURRENCY` are set to the thread count before
  anything is imported. Each library also gets its own setting: the Daft runner's
  `num_threads`, `torch.set_num_threads`, the pool, DataLoader or `num_proc`
  size, and `cv2.setNumThreads(1)`.
- **Time.** Setup (loading inputs into the library's container) is untimed. Then
  come 1 warmup run and 5 timed runs of the whole job, from inputs to a
  materialised result. That includes starting any process pool, as a one-shot
  batch job would. Reported: the median, the IQR (Q1–Q3 of the 5 runs), and
  images/s = N / median.
- **Memory cap.** With `--mem-limit-gib`, a cell whose process tree passes the
  limit is killed and recorded as `memory-limit`, rather than letting the
  kernel's OOM killer take the run down. The 50k run used 12 GiB of the
  machine's 15.7.
- **Memory.** Peak **PSS** of the process plus all its workers, sampled every
  50 ms from `/proc/<pid>/smaps_rollup` during the timed runs. PSS splits a
  shared page between the processes that map it, so forked workers are not
  charged for the parent's memory once each, as summed RSS would charge them.
  Each cell also records its baseline (after imports and input loading) and the
  amount the job added.
- **Environment.** The runs here: 4 vCPUs (Intel Xeon @ 2.80GHz), 15.7 GiB,
  Linux 6.18, Python 3.12.3, polars 2.0.0, Pillow 12.3.0, OpenCV 5.0.0
  (contrib), torch 2.14.1, torchvision 0.29.1, Daft 0.7.26, datasets 5.1.0,
  imagehash 4.3.2, polars-cv 0.38.0 (see above), commit `49c4cec`. A
  container restart stopped the 5k run after 91 of its 160 cells; the rest
  ran on the same hardware, wheel and data, and `coco-5k.json` records each
  part's environment under `resumed_environments`.
- **Every results file** has an `environment` block with the CPU
  model and count, memory, OS, Python, the installed version of every library,
  the same from `uv.lock`, the git commit and a timestamp.

## Caveats

- **One machine.** A 4-vCPU cloud container (Intel Xeon @ 2.80GHz, 15 GiB). The
  ranking at 4 cores need not hold at 32, where process pools pay more for
  pickling and polars-cv's single process may hit memory bandwidth sooner.
  Shared-tenancy noise is visible in some IQRs.
- **OpenCV does less work in w1, w2 and w4.** `INTER_LINEAR` samples 2×2 pixels
  without antialiasing when downscaling, which is cheaper and visibly aliased
  (mean difference 4–6 levels against ~0.2 for polars-cv). `INTER_AREA` would
  be the like-for-like choice and is slower: 1.65 ms against 0.39 ms for a
  640→256 resize, single thread. The bilinear filter was pinned for
  everyone, so OpenCV runs its own bilinear.
- **pHash differs by library.** imagehash, Daft and polars-cv 0.38 agree to
  within a bit or two. OpenCV's hash is another pHash variant, and its bits
  are not comparable with imagehash's.
- **w2 for polars-cv uses a full decode.** Pillow's `thumbnail()` keeps a full
  JPEG decode at these sizes (`reducing_gap=2`), so polars-cv runs without
  `decode_max_size=` (then spelled `.thumbnail()`), which would decode at half
  scale and do less work. That
  makes polars-cv's w2 numbers conservative.
- **A release, not the checkout.** These numbers measure 0.38.0 (see the
  top of this page). A change on `main` after it is not measured until it is
  released.
