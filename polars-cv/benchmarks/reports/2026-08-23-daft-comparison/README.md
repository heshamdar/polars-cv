# Daft vs polars-cv — investigation and benchmark comparison

**Date:** 2026-08-23
**Code under test:** `claude/daft-polars-cv-benchmark-3ha1jq` @ `7970970` (polars-cv 0.20.0, release build), Daft 0.7.24
**Hardware:** 4-core Intel Xeon @ 2.10 GHz, 15 GB RAM, Linux 6.18
**Software:** Python 3.11.15, Polars 1.42.0, OpenCV 4.11.0, Pillow 12.x, NumPy 2.4.0

Raw results are in `single_ops.json`, `pipelines.json` and `e2e.json`; the
tables they produce are in `tables.md` (`analyze.py`). The API claims come from
`capability_probe.py` (output in `capability_probe.txt`) and the fairness
checks from `parallelism_probe.py` (`parallelism_probe.txt`). Both probes are
committed so a future Daft release can be re-checked rather than re-argued.

---

## 1. Verdict

**They are not really competitors, and the benchmark is what shows it.**

Daft is a multimodal data engine that can process images. polars-cv is an image
processing engine that happens to live in a dataframe. Daft ships **17 image
expressions** against polars-cv's **71 pipeline operations**, and only **five**
of polars-cv's ops have a native Daft equivalent at all (`resize`, `crop`,
`grayscale`, `perceptual_hash`, `cast`).

That gap dominates every other result:

| | polars-cv | Daft |
|---|---|---|
| Single-op benchmarks runnable natively | 20 / 20 | **3 / 20** |
| Multi-op pipelines runnable natively | 5 / 5 | **0 / 5** |
| End-to-end workflows runnable natively | 3 / 3 | **0 / 3** |

Not one of the harness's five realistic pipelines — nor any of its three
end-to-end workflows — can be expressed in Daft without dropping into Python,
because every one of them contains a `normalize` or a `flip`, and Daft has
neither. Its image columns do not even support arithmetic (`col("img") * 2`
raises `Cannot multiply types: Image[RGB], Int64`), so normalize is not a
missing convenience method, it is outside what the expression system can say.

Where both engines *can* run the same work, polars-cv is faster:

| Comparison | geomean |
|---|---|
| polars-cv streaming ÷ Daft, on Daft's three native ops | **3.06x** |
| polars-cv eager (1 core) ÷ Daft (3 cores), same three ops | 1.01x |
| polars-cv streaming ÷ Daft-with-UDFs, on the 5 pipelines | **6.71x** |
| polars-cv streaming ÷ Daft-with-UDFs, end-to-end | **3.03x** |

And this is not a thread-count artifact — see §3.

The honest summary for someone choosing between them: **if your work is images
and you want it fast, polars-cv is several times quicker and can express
vastly more. If your work spans video, audio, text and embeddings, or has to
scale across a Ray cluster, Daft does things polars-cv does not attempt** — and
you will be writing your image operations in Python UDFs.

---

## 2. What Daft actually offers for images

As of 0.7 the image methods sit directly on `Expression` (the `.image`
namespace is gone), and `@daft.udf` is deprecated in favour of `@daft.func` /
`@daft.func.batch`. The complete vision surface:

| Category | Daft |
|---|---|
| Decode / encode | `decode_image`, `encode_image`, `decode_image_file` |
| Geometry | `resize(w, h)`, `crop(bbox)` |
| Colour | `convert_image(mode)` |
| Conversion | `image_to_tensor`, `as_image` |
| Metadata | `image_attribute`, `image_width`, `image_height`, `image_channel`, `image_mode`, `image_file_metadata` |
| Hashing | `image_hash(method=…, hash_size=…)` |

Everything else in an image pipeline — blur, threshold, rotate, pad, erode,
dilate, canny, sobel, equalize, invert, contrast, brightness, sharpen, flip,
normalize — has no Daft expression. The 66 polars-cv ops with no Daft
equivalent are listed in `capability_probe.txt`.

Two things worth crediting Daft for, though:

- **The type system is genuinely good.** A column carries its shape and mode in
  the schema — `Image[RGB; 32 x 32]`, `Tensor[UInt8; [32, 32, 3]]` — and it is
  known before execution, the same plan-time guarantee polars-cv gives.
- **`resize` is a quality implementation.** It is bilinear and matches Pillow's
  BILINEAR to ±1, and polars-cv's `filter="bilinear"` likewise. The
  `resize` numbers below compare like with like.

### A silent semantic difference

`convert_image("L")` uses **ITU-R BT.709** luma; polars-cv, OpenCV and Pillow
all use **BT.601**:

```
Daft convert_image('L')   R=0.2118  G=0.7137  B=0.0706   (BT.709)
polars-cv grayscale()     R=0.3020  G=0.5843  B=0.1137   (BT.601)
```

On gradient test images this shifts pixels by up to **21/255 (~8%)**, mean 3.3.
Nothing warns you. Swapping engines mid-project silently changes every
grayscale-derived feature, which matters for anything downstream of a trained
model.

---

## 3. Performance

### The only true engine-vs-engine cells

These are the three single ops Daft runs with its own Rust kernels. Everything
else in the suite it cannot do natively.

| op | size | pcv-eager ÷ daft | pcv-stream ÷ daft | daft ÷ opencv |
|---|---:|---:|---:|---:|
| resize | 256 | 1.67x | **5.61x** | 1.23x |
| resize | 512 | 1.84x | **7.45x** | 1.53x |
| grayscale | 256 | 0.97x | 3.24x | 0.45x |
| grayscale | 512 | 0.63x | 1.70x | 0.29x |
| crop_center | 256 | 0.84x | 2.00x | 0.28x |
| crop_center | 512 | 0.66x | 1.79x | 0.17x |
| **geomean** | | **1.01x** | **3.06x** | **0.47x** |

Daft beats polars-cv's *eager* mode on grayscale and crop, and loses on resize.
Against polars-cv's streaming engine it loses everywhere.

The `crop` column deserves a caveat: OpenCV's crop is a NumPy slice plus a copy,
so `daft ÷ opencv = 0.17x` there is not an engine indictment — no dataframe
beats a slice. Notice also that Daft's crop throughput barely moves between
256² and 512² (9,833 → 9,761 img/s): the output is 128² either way, so the work
really is constant.

### This is not a thread-count artifact

The obvious objection is that the harness hands each framework one in-memory
batch, and maybe Daft was being starved of parallelism. It was not
(`parallelism_probe.txt`):

| framework | op | img/s | cores used |
|---|---|---:|---:|
| polars-cv-eager | resize | 3,818 | 0.99 |
| polars-cv-streaming | resize | 14,930 | 3.28 |
| **daft** | **resize** | **2,369** | **3.03** |
| daft-udf | blur | 2,281 | 2.35 |
| opencv | resize | 1,793 | 1.02 |
| opencv | blur | 4,638 | 3.90 |

Daft keeps **3.03 of 4 cores** busy — essentially the same as polars-cv
streaming's 3.28 — and is still 6.3x slower on resize. Per core that is
4,552 vs 782 img/s: **polars-cv's resize kernel is ~5.8x more efficient per
core.** Single-threaded polars-cv-eager (0.99 cores, 3,818 img/s) beats
three-core Daft outright.

Partitioning is not a withheld lever either: `into_partitions` is a **no-op on
Daft's NativeRunner** (it warns and tells you to use the Ray runner), and
sweeping 1/2/4/8 partitions moves throughput by at most 7%, which is noise.

One caveat in the other direction: OpenCV's `GaussianBlur` internally uses 3.90
cores, so the `opencv` column is not uniformly single-threaded, and the
`daft-udf ÷ opencv` ratios for blur-like ops flatter OpenCV.

### What the UDF escape hatch costs

For the 17 ops Daft cannot express, the real-world answer is a
`@daft.func.batch` UDF. The `daft-udf` adapter does exactly that, calling **the
same OpenCV kernels** the `opencv` adapter uses, so the ratio is pure engine
overhead:

| | geomean vs opencv |
|---|---|
| daft-udf | **0.40x** |
| polars-cv-streaming | **1.55x** |

Handing an op back to Python costs Daft ~60% of the throughput a plain
single-threaded Python loop achieves, despite using 2.35 cores — the
marshalling in and out of the Rust engine more than eats the parallelism.
polars-cv, which never leaves Rust, runs 1.55x *faster* than the same loop.

That is a **~3.9x gap on exactly the operations Daft users have no alternative
for.**

### Realistic pipelines and end-to-end workloads

Daft cannot express any of these natively, so the column is `daft-udf`.

| pipeline | size | pcv-stream | daft-udf | ratio |
|---|---:|---:|---:|---:|
| light_pipeline | 256 | 5,532 | 671 | 8.2x |
| imagenet_preprocess | 256 | 5,457 | 908 | 6.0x |
| heavy_pipeline | 256 | 4,670 | 697 | 6.7x |
| medium_pipeline | 256 | 3,872 | 651 | 5.9x |
| medical_pipeline | 256 | 3,306 | 562 | 5.9x |
| light_pipeline | 512 | 4,472 | 607 | 7.4x |
| imagenet_preprocess | 512 | 3,616 | 620 | 5.8x |
| heavy_pipeline | 512 | 3,429 | 473 | 7.2x |
| medical_pipeline | 512 | 2,756 | 314 | 8.8x |
| medium_pipeline | 512 | 2,515 | 428 | 5.9x |

| e2e workflow | size | pcv-stream | daft-udf | ratio |
|---|---:|---:|---:|---:|
| basic_preprocess | 256 | 1,783 | 665 | 2.7x |
| imagenet_workflow | 256 | 1,716 | 762 | 2.3x |
| augmentation_workflow | 256 | 1,401 | 587 | 2.4x |
| basic_preprocess | 512 | 1,593 | 402 | 4.0x |
| imagenet_workflow | 512 | 1,413 | 393 | 3.6x |
| augmentation_workflow | 512 | 1,142 | 303 | 3.8x |

The end-to-end ratios are smaller because PNG decode dominates and both engines
decode in Rust. The pipeline ratios are larger because that is where per-op UDF
overhead compounds: each of the 2–6 operations pays its own marshalling cost.

---

## 4. Ease of setup

**For an end user this is a tie, which is the opposite of what the "Rust plugin
vs pip package" framing suggests.** Both ship prebuilt abi3 wheels; timed clean
installs from PyPI with a cold cache:

| | wall time | packages pulled | site-packages |
|---|---:|---:|---:|
| `pip install daft` | 1.42 s | 6 | 303 MB |
| `pip install polars-cv` | 1.43 s | 8 | 309 MB |

Neither compiles anything. The real differences are elsewhere:

| | polars-cv 0.20.0 | Daft 0.7.24 |
|---|---|---|
| Linux x86_64 / aarch64 | ✅ | ✅ |
| macOS arm64 | ✅ | ✅ |
| macOS x86_64 | ❌ | ✅ |
| **Windows** | ❌ | ✅ |
| Python floor | 3.10 | 3.10 |
| Optional extras | — | 23 (aws, ray, transformers, …) |

**Daft wins on platform reach** — no Windows wheel is a hard blocker for a
Windows shop, and polars-cv would have to build from source there.

**Building from source is where polars-cv is genuinely harder.** It needs a
Rust toolchain ≥ 1.96 — this container's stable 1.94.1 failed outright with
"requires rustc 1.96" and needed a `rustup update` before anything would build.
A cold build of the whole polars stack then runs in tens of minutes: on this
4-core box the final `maturin develop --release` alone reported **10m 07s**
after its dependencies were already compiled, and the dependency build before
it was the larger share. Peak disk for the `target/` tree was ~20 GB.

Daft contributors face a comparable Rust build, but Daft users effectively
never do — and polars-cv users on Windows would.

First-run experience is otherwise even: both are `import`-and-go, and both
compute an image column in about five lines.

---

## 5. Robustness

Two Daft 0.7.24 defects surfaced while building the adapter, both reproducible
in `capability_probe.py`, and both **crash the process rather than raising**:

1. **`image_to_tensor()` then cast to float32 panics the Rust worker.**
   ```
   StructArray::new received an array with dtype: List[UInt8]
   but expected child field: data#List[Float32]
   ```
   This is on the path of the canonical ML preprocessing example — decode →
   resize → to_tensor → normalize to float. The fixed-shape tensor produced by
   `image_to_tensor()` cannot have its element dtype cast. (The *ragged*
   `Tensor[UInt8]` casts fine; only the fixed-shape one fails, and via
   `image_to_tensor` it fails by panic rather than by error.)

2. **Float image modes are declared but unreadable.** `DataType.image()`
   advertises `RGB32F` and `RGBA32F`, but writing float32 data under that dtype
   panics on readback:
   ```
   Attempting to downcast Float32 to DataArray<UInt8Type>
   ```

Both abort the interpreter — a `try`/`except` does not save you, which is why
the probes for them run in subprocesses. The practical consequence is that
float image data has to be carried as `DataType.tensor(...)`, which is what the
`daft-udf` adapter does.

By contrast, polars-cv's stated design rule is that a bypass must fail loudly
rather than degrade, and nothing in this exercise produced a panic from it.

---

## 6. Flexibility

This is the one axis where Daft clearly wins, and it is worth being precise
about *which* flexibility.

### Daft is broader across modalities

Daft exposes **344 functions**, spanning things polars-cv does not attempt:

| Modality | Daft |
|---|---|
| Video | `video_frames`, `video_keyframes`, `video_metadata`, `video_file` |
| Audio | `audio_file`, `audio_metadata` |
| Text / AI | `embed_text`, `embed_image`, `classify_text`, `classify_image`, `llm_generate`, `prompt`, `tokenize_encode/decode` |
| IO | `download`, `upload`, `parse_url`, `file_exists`, S3 / GCS / Iceberg / Delta / Hugging Face |
| Scale | Ray runner, Kubernetes — "start local, scale to distributed" |

polars-cv has none of this. It has no distributed story at all: it scales to
one machine's cores via the Polars streaming engine and stops there. **If the
requirement is "one engine for video frames, transcripts, embeddings and
images, across a cluster", Daft is the answer and polars-cv is not a
candidate.**

### polars-cv is deeper within images

| | polars-cv | Daft |
|---|---|---|
| Image operations | 71 | 17 (5 overlapping) |
| Arithmetic on image columns | ✅ (`add`, `multiply`, `blend`, `ratio`, …) | ❌ `Cannot multiply` |
| Contour / geometry subsystem | ✅ (12 contour ops, point/bbox namespaces) | ❌ |
| Detection metrics (AP, FROC, LROC, bootstrap) | ✅ | ❌ |
| Reductions over images | ✅ (13 `reduce_*`) | ❌ |
| Morphology, edges, filters | ✅ | ❌ |

### Per-row parameters

polars-cv's premise is that any non-structural parameter can be a Polars
expression. Daft supports this in exactly one place:

| | polars-cv | Daft |
|---|---|---|
| `resize` dimensions from a column | ✅ | ❌ rejected |
| `crop` box from a column | ✅ | ✅ (needs a `FixedSizeList[UInt32; 4]` cast) |
| Interpolation `filter` from a column | ✅ | ❌ (no filter parameter at all) |

Per-row crop boxes are real and useful — that is the detection-cropping use
case, and Daft handles it. But per-row *resize* targets and per-row
interpolation are not expressible, and Daft's `resize` exposes no filter
choice whatsoever.

### Ergonomics

Both are pleasant. Daft's flat method style reads slightly shorter for the
handful of ops it has; polars-cv's `Pipeline` object is reusable across
queries and composes into a DAG with CSE:

```python
# Daft
df.with_column("out", daft.col("img").decode_image().resize(224, 224).convert_image("L"))

# polars-cv
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
```

The asymmetry appears at op two: in Daft, the moment you need a blur you are
writing a UDF, declaring its return dtype by hand, and (per §5) discovering
which dtypes survive readback.

---

## 7. When to choose which

**Choose Daft when** the workload is multimodal beyond images (video, audio,
text, embeddings), when it must run distributed on Ray or Kubernetes, when you
need Iceberg / Delta / Hugging Face connectors, when LLM and embedding
functions belong in the same query, or when you need Windows.

**Choose polars-cv when** the workload is image processing, when you need more
than resize/crop/grayscale, when throughput matters (3–8x here on comparable
work), when parameters vary per row, or when you need contours, geometry or
detection metrics.

**Use both** for the obvious split: Daft for ingest and multimodal fan-out,
polars-cv for the vision transforms. They share Arrow, so handing a column
across is cheap.

The genuine competitive risk is narrow but real: Daft only needs enough image
ops to cover the common preprocessing path (resize, crop, grayscale, normalize,
flip, pad) to make polars-cv unnecessary for the majority of ML preprocessing
users. It has three of those six today, and no arithmetic on image columns
standing between it and the rest.

---

## 8. How to reproduce

```bash
cd polars-cv
uv sync --group bench            # daft>=0.7 is in the bench group
uv run --no-sync maturin develop --release

R=benchmarks/reports/2026-08-23-daft-comparison
for sc in single_ops pipelines e2e; do
  uv run --no-sync python -m benchmarks.run_benchmarks --scenario $sc \
    --frameworks polars-cv-eager,polars-cv-streaming,daft,daft-udf,opencv,pillow \
    --sizes 256,512 --counts 100 --iterations 10 --quiet --output json > $R/$sc.raw
done
# strip the progress preamble the runner writes to stdout, then:
PYTHONPATH=. uv run --no-sync python $R/analyze.py > $R/tables.md
PYTHONPATH=. uv run --no-sync python $R/capability_probe.py
PYTHONPATH=. uv run --no-sync python $R/parallelism_probe.py
```

### Fairness notes

- Both Daft adapters are timed on pre-decoded, materialized DataFrames, exactly
  as the polars-cv adapter is: `prepare_decoded_images` does the decode and the
  frame construction outside the timed section for every framework.
- `daft-udf`'s UDF bodies call `OpenCVAdapter`, so `daft-udf` and `opencv` run
  byte-identical kernels and the difference is engine overhead alone. Verified:
  every UDF op matches OpenCV exactly.
- Numerical agreement was checked before timing. `crop` is exact across all
  four engines; Daft's `resize` matches polars-cv's bilinear to ±1;
  `grayscale` differs by the BT.709/BT.601 convention documented in §2.
- The `daft` (native-only) adapter raises rather than silently falling back, so
  a missing number in the tables is a missing capability, not a failed run.
- `e2e_workflow.py`'s dispatch used to route on `"polars" in adapter.name`,
  which would have driven Daft through the per-image path — one DataFrame per
  image per operation. That is now a declared `columnar` flag on the adapter,
  so every dataframe engine gets its batch path.
