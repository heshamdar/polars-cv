# polars-cv vs PIL, OpenCV, torchvision, Daft and Hugging Face datasets

A reproducible comparison on COCO val2017: four workloads, nine
implementations, two input sources, one thread and all cores. Every
implementation is held to the PIL reference by parity tests before anything is
timed, and every number below comes from `results/*.json`, which also records
the hardware and every library version.

This directory is its own uv project. It benchmarks the **published**
`polars-cv` wheel (pinned in `pyproject.toml`), which is what users install,
not the local checkout. The regression suite that measures the checkout lives
in [`polars-cv/benchmarks/`](../polars-cv/benchmarks/).

<!-- RESULTS -->

## Reproduce

```bash
cd benchmarks
uv sync                                   # exact versions from uv.lock
uv run python -m bench.data --scale 50000 # COCO val2017 (815 MB, checksummed) + a 50k set
uv run --group dev pytest                 # parity first; -m "not slow" skips the 5k-image check
uv run python run.py --out results/coco-5k.json
uv run python run.py --dataset coco-50k --source path --threads max \
    --workload w1,w2,w3 --out results/coco-50k.json
uv run python plot.py results/coco-5k.json results/coco-50k.json
```

`run.py` takes `--workload`, `--library`, `--threads` (`1,max` by default),
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

<!-- PARITY -->

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
- **Memory.** Peak **PSS** of the process plus all its workers, sampled every
  50 ms from `/proc/<pid>/smaps_rollup` during the timed runs. PSS splits a
  shared page between the processes that map it, so forked workers are not
  charged for the parent's memory once each, as summed RSS would charge them.
  Each cell also records its baseline (after imports and input loading) and the
  amount the job added.
- **Environment.** Every results file has an `environment` block with the CPU
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
  be the like-for-like choice and is slower. The bilinear filter was pinned for
  everyone, so OpenCV runs its own bilinear.
- **pHash differs by library.** imagehash and Daft agree to within a bit.
  OpenCV's and polars-cv's hashes are other pHash variants, and their bits are
  not comparable with imagehash's.
- **w2 for polars-cv uses a full decode.** Pillow's `thumbnail()` keeps a full
  JPEG decode at these sizes (`reducing_gap=2`), so polars-cv runs without
  `.thumbnail()`, which would decode at half scale and do less work. That
  makes polars-cv's w2 numbers conservative.
- **The published wheel, not the checkout.** A change on `main` after 0.37.0 is
  not measured until it is released.
