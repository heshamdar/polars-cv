# polars-cv vs Daft vs Pixeltable — investigation and benchmark comparison

**Date:** 2026-08-23
**Code under test:** `claude/daft-polars-cv-benchmark-3ha1jq` (polars-cv 0.20.0, release build), Daft 0.7.24, Pixeltable 0.7.2
**Hardware:** 4-core Intel Xeon @ 2.10 GHz, 15 GB RAM, Linux 6.18
**Software:** Python 3.11.15, Polars 1.42.0, OpenCV 4.11.0, Pillow 12.x, NumPy 2.4.0

Three engines that all put images in a table, compared on throughput, operation
coverage, setup and flexibility.

Raw results are in `single_ops.json`, `pipelines.json` and `e2e.json`; the
tables they produce are in `tables.md` (`analyze.py`). Four committed probes
back the claims that a throughput table cannot make, each with its `.txt`
output alongside:

| probe | question it answers |
|---|---|
| `capability_probe.py` | what can each engine actually express? |
| `parallelism_probe.py` | is the comparison a fair fight on cores? |
| `udf_path_probe.py` | what does Daft's UDF boundary cost, chained vs fused? |
| `incremental_probe.py` | does Pixeltable's caching model pay off? |

They are committed so a future release of any of the three can be re-checked
rather than re-argued.

---

## 1. Verdict

**All three put images in a table, and they are built for three different
jobs.** polars-cv is an image-processing engine that happens to live in a
dataframe. Daft is a multimodal data engine that can process images. Pixeltable
is a *database* that computes and caches image columns for you.

Operation coverage separates them before throughput does:

| | polars-cv | Daft | Pixeltable |
|---|---|---|---|
| Image operations exposed | **71** | 17 | ~20 (PIL passthrough) |
| Single-op benchmarks runnable natively | 20 / 20 | **3 / 20** | **9 / 20** |
| Multi-op pipelines runnable natively | 5 / 5 | **0 / 5** | **0 / 5** |
| End-to-end workflows runnable natively | 3 / 3 | **0 / 3** | **0 / 3** |

Pixeltable's PIL passthrough is a real advantage over Daft — it has flips,
rotation, and lookup-table operations that give it threshold, invert and
brightness — but neither engine can run a single one of the harness's five
realistic pipelines without dropping into Python, because every pipeline
contains `normalize` and neither has any float-producing arithmetic on image
columns. Daft raises `Cannot multiply types: Image[RGB], Int64`; Pixeltable's
image type is PIL-backed and therefore 8-bit, so a normalized float image
cannot even be stored in one.

On throughput, where each engine runs its own kernels:

| Comparison (geomean) | vs Daft | vs Pixeltable |
|---|---|---|
| polars-cv streaming ÷ engine, native ops | **4.25x** | **10.00x** |
| polars-cv eager (1 core) ÷ engine, native ops | 1.01x | **3.50x** |
| engine-with-UDFs ÷ a plain OpenCV loop | 0.30x | 0.16x |

polars-cv streaming is ahead of both everywhere, and single-threaded polars-cv
still beats three-core Daft and Pixeltable. Two caveats keep this honest, and
both are measured rather than asserted:

- **Daft's numbers are not a UDF-shape artifact** (§4). Fusing its per-op UDFs
  into one is worth ~2x, which corrects the pipeline gap from 6.7x to ~3.6x —
  but the single-op gap is already the fused best case.
- **Pixeltable's numbers include a decode the others exclude** (§5). Its data
  model has no in-memory pre-decoded column, so ~60% of every cell above is
  reading pixels back out of its media store. The fair like-for-like comparison
  is the end-to-end scenario, where it is still 5–17x behind.

And Pixeltable's actual pitch — incremental computed columns — does work
(§6). Caching pays for itself after **2 reads** and appending rows is **9.4x**
cheaper than a rebuild. But the ceiling is low: **polars-cv recomputes from
files at 3,917 rows/s, faster than Pixeltable reads from its cache at 1,663.**
Caching is a fix for expensive recomputation, and polars-cv's recomputation is
not expensive.

The honest summary: **if the work is images and throughput matters, polars-cv
wins on both speed and expressiveness by a wide margin. Choose Daft when the
workload spans video, audio, text and embeddings or must run on a Ray cluster.
Choose Pixeltable when you want a system of record that keeps derived columns
up to date and you would otherwise be hand-rolling a cache** — and in both
cases expect to write your image operations in Python.

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

### The cells where each engine runs its own kernels

Daft implements 3 of the 20 single ops itself; Pixeltable implements 9. These
are the comparisons where both sides do comparable work. Full table in
`tables.md`.

| | vs Daft | vs Pixeltable |
|---|---:|---:|
| polars-cv streaming ÷ engine (geomean) | **4.25x** | **10.00x** |
| polars-cv eager, 1 core ÷ engine (geomean) | 1.01x | **3.50x** |
| resize @256 | 7.70x | 21.99x |
| grayscale @256 | 3.77x | 13.13x |
| crop @256 | 3.02x | 14.55x |

Read Pixeltable's column with §5's caveat in mind: its figures include a decode
from the media store that every other adapter excludes, which is roughly 60% of
the work. Even discounted for that it stays several times behind.

### This is not a thread-count artifact

The obvious objection is that the harness hands each framework one in-memory
batch and starves it of parallelism. It does not (`parallelism_probe.txt`):

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
Daft's NativeRunner** (it warns and points at the Ray runner), and sweeping
1/2/4/8 partitions moves throughput by at most 7%, which is noise.

One caveat in the other direction: OpenCV's `GaussianBlur` internally uses 3.90
cores, so the `opencv` column is not uniformly single-threaded, and the
`÷ opencv` ratios for blur-like ops flatter OpenCV.

### What the UDF escape hatch costs

For the ops they cannot express, both engines fall back to a UDF. Both
adapters' UDF bodies call **the same OpenCV kernels** the `opencv` adapter
uses, so the ratio is pure engine overhead:

| | geomean vs a plain OpenCV loop |
|---|---|
| daft-udf | **0.30x** |
| pixeltable-udf | **0.16x** |
| polars-cv-streaming | **1.21x** |

Handing an operation back to Python costs Daft ~70% and Pixeltable ~84% of what
a single-threaded Python loop achieves. polars-cv, which never leaves Rust,
runs 1.21x *faster* than the same loop. Note this is measured on **single**
ops, where there is exactly one UDF and no chaining penalty — see §4.

### Realistic pipelines and end-to-end workloads

Neither Daft nor Pixeltable can express any of these natively, so both columns
are the UDF variants.

| pipeline | size | pcv-stream | daft-udf | pxt-udf | ÷daft | ÷pxt |
|---|---:|---:|---:|---:|---:|---:|
| light_pipeline | 256 | 4,700 | 580 | 486 | 8.1x | 9.7x |
| imagenet_preprocess | 256 | 4,755 | 905 | 882 | 5.3x | 5.4x |
| heavy_pipeline | 256 | 4,891 | 668 | 816 | 7.3x | 6.0x |
| medium_pipeline | 256 | 3,478 | 560 | 761 | 6.2x | 4.6x |
| medical_pipeline | 256 | 3,155 | 515 | 926 | 6.1x | 3.4x |
| light_pipeline | 512 | 4,123 | 525 | 225 | 7.9x | 18.3x |
| imagenet_preprocess | 512 | 2,835 | 577 | 214 | 4.9x | 13.3x |
| heavy_pipeline | 512 | 3,270 | 478 | 195 | 6.8x | 16.7x |
| medium_pipeline | 512 | 2,844 | 400 | 204 | 7.1x | 13.9x |
| medical_pipeline | 512 | 2,788 | 262 | 387 | 10.6x | 7.2x |
| **geomean** | | | | | **6.88x** | **8.51x** |

| e2e workflow | size | pcv-stream | daft-udf | pxt-udf | ÷daft | ÷pxt |
|---|---:|---:|---:|---:|---:|---:|
| basic_preprocess | 256 | 1,711 | 646 | 282 | 2.6x | 6.1x |
| imagenet_workflow | 256 | 3,023 | 755 | 389 | 4.0x | 7.8x |
| augmentation_workflow | 256 | 2,251 | 510 | 362 | 4.4x | 6.2x |
| basic_preprocess | 512 | 2,136 | 342 | 126 | 6.2x | 17.0x |
| imagenet_workflow | 512 | 1,986 | 330 | 122 | 6.0x | 16.3x |
| augmentation_workflow | 512 | 1,461 | 284 | 125 | 5.1x | 11.7x |
| **geomean** | | | | | **4.56x** | **9.92x** |

The end-to-end scenario is the one that is genuinely like-for-like for all
three: every framework starts from files, so nobody's decode is excluded. It is
also where Daft's gap is smallest (4.56x), because PNG decode dominates and all
three decode in Rust or C.

The pipeline gap is larger because that is where per-op UDF overhead compounds:
each of the 2–6 operations pays its own marshalling cost. §4 shows how much of
that a Daft user can win back by hand-fusing.

---

## 4. Is Daft's UDF path optimal?

The fair objection to everything above is that Daft never set out to implement
every image operation natively. Its position is that a UDF *is* the supported
way to run one, executed efficiently — so the question is not "how many native
ops does it have" but "how good is the UDF path". Measured directly
(`udf_path_probe.py`, output in `udf_path_probe.txt`).

### The UDF boundary costs ~120 µs per image

A batch UDF that does nothing at all, on 100 × 256² RGB images:

| identity UDF | img/s | µs/img | cores |
|---|---:|---:|---:|
| `return series` (no materialization) | 19,999 | **50.0** | 0.80 |
| `series.to_pylist()`, return the list | 5,964 | **167.7** | 0.89 |
| `series.to_arrow()` first | 4,328 | 231.1 | 0.95 |

This splits the cost cleanly. **Dispatch is cheap** — 50 µs/image to enter and
leave a UDF that hands the Series straight back. Daft's UDF machinery is a real
vectorized batch interface, not per-row Python calls, and that deserves credit.

**Touching the pixels is what costs.** The moment the UDF materializes the
column into NumPy objects and Daft rebuilds a column from what comes back, the
price is ~168 µs/image — about **120 µs of pure marshalling** on top of
dispatch, before a single OpenCV instruction runs. `to_arrow()` does not help;
it is slower.

For scale: polars-cv streaming runs the **entire six-operation heavy pipeline**
at 5,421 img/s — 184 µs/image. Daft's cost to hand one image into Python and
take it back is roughly the cost of polars-cv doing all six operations.

### Fusing helps, and the natural way to write it does not

Chaining image UDFs the way you chain expressions makes every operation pay
that boundary again. Hand-fusing them into one UDF pays it once:

| heavy pipeline, 6 ops | img/s | cores |
|---|---:|---:|
| Daft, one UDF per op (what an expression chain gives you) | 754 | 1.27 |
| **Daft, single fused UDF** | **1,491** | 2.16 |
| opencv, plain Python loop | 3,248 | 3.67 |
| polars-cv-eager | 1,418 | 1.01 |
| polars-cv-streaming | 5,421 | 3.69 |

**Fusing is worth 1.98x**, and the main tables above use the chained shape — so
those pipeline ratios (6.71x geomean) overstate the gap for a user who knows to
hand-fuse. Corrected to the fused best case the heavy pipeline is **3.6x**, and
fused Daft reaches **parity with single-threaded polars-cv-eager** (1.05x).

But note what fusing costs in ergonomics: it means abandoning composition.
`resize(...).blur(...).threshold(...)` as three chained UDFs is half the speed
of one opaque `do_everything(...)`. Daft gives you no kernel fusion across UDF
boundaries, so the user has to do it by hand and loses the expression model to
get it. polars-cv fuses kernels for you and keeps the chain.

### The single-op numbers are unaffected

Worth being explicit, because it is the crux: the **0.40x geomean against a
plain OpenCV loop is measured on single operations**, where there is exactly one
UDF and *no chaining penalty whatsoever*. That number is already the fused best
case. Fusing cannot improve it.

### Verdict on the UDF path

**Well-designed, genuinely batched, and still not competitive for image
payloads.** Three things stack up against it:

1. The boundary costs ~120 µs/image in marshalling that polars-cv never pays,
   because it never leaves Rust.
2. It parallelizes poorly here — 0.8–2.2 cores against polars-cv streaming's
   3.7 — so the engine does not buy back what the boundary costs.
3. Composition is penalized: the idiomatic chained form is 2x off Daft's own
   best case.

The design is defensible for what Daft is optimizing: a UDF that calls a GPU
model or an LLM does real work measured in milliseconds, and 168 µs of
marshalling disappears into the noise. **The economics only break down when the
UDF is cheap relative to the payload — which is exactly what image kernels
are.** That is the honest reason a general multimodal engine and a dedicated
vision engine land where they do, and it is not a criticism of Daft's
implementation so much as of using a UDF for a 50 µs blur.

---

## 5. What Pixeltable offers for images

Pixeltable's image expressions are a **passthrough to PIL**: `resize`, `crop`,
`rotate`, `convert`, `transpose`, `point`, `blend`, `composite`, `histogram`,
`entropy`, `thumbnail`, `quantize`, `getchannel`, `getextrema` and friends.
That is a materially better vision surface than Daft's — it has flips (via
`transpose`), rotation, and `point()`, whose arbitrary lookup table supplies
threshold, invert and brightness. Nine of the harness's twenty single ops run
natively, against Daft's three.

What it still cannot express: `normalize`, `blur`, `sharpen`, `pad`, `erode`,
`dilate`, `equalize_histogram`, `canny`, `sobel`, `adjust_contrast` (its
OpenCV form is data-dependent on the image mean, so no static LUT can do it),
and `rotate` with canvas expansion — `rotate(self, angle: Int)` takes no
`expand` and no fractional angle.

### Three API constraints worth knowing

All three were found by validating output against OpenCV, not by reading docs:

- **`resize` exposes no resampling filter.** Its signature is
  `resize(self, size)`, so it uses PIL's default, which is **bicubic** —
  verified equal to `PIL.Image.BICUBIC` exactly. Every other adapter here is
  pinned to bilinear, so Pixeltable's resize output legitimately differs
  (max 16/255 against OpenCV's INTER_AREA, 21/255 against polars-cv).
- **Its image column is 8-bit.** Pixeltable's `Image` type is PIL-backed, so it
  cannot hold the float output of `normalize` or `sobel`. Those have to become
  `Array[Float]` columns — and once a column is an array, no image expression
  applies to it, so everything downstream of a float-producing operation must
  also be a UDF. This is the same wall Daft hits from the other side.
- **UDFs must live in a named module.** Pixeltable rejects a UDF defined in a
  script's global namespace *or* constructed inside a method, and registers
  them by qualified name so a second registration raises `AlreadyExistsError`.
  The adapter's gap-fillers therefore live in a separate module
  (`benchmarks/frameworks/_pixeltable_udfs.py`) and are parameterized by a JSON
  operation spec rather than closing over one.

### Why its throughput numbers carry an asterisk

Every other adapter implements `prepare_decoded_images` by decoding once,
outside the timed section, so the scenarios measure operations alone.
**Pixeltable cannot.** Its images live in the media store as files, and every
query reads and decodes them again. Its single-op and pipeline numbers
therefore include a decode the others exclude, and no adapter change can remove
it — it is the data model.

The decode is the larger share. From `incremental_probe.txt`:

| Pixeltable, 200 x 256² | rows/s |
|---|---:|
| `select(img)` — decode only, no operation | 637 |
| `select(img.resize(224,224))` — decode + resize | 376 |

So roughly 60% of the work in any Pixeltable cell in §3 is getting pixels back
out of the store. The end-to-end scenario, where every framework starts from
files, is the one to read as like-for-like — and there Pixeltable is still
6–17x behind polars-cv streaming.

One thing the comparison genuinely does *not* capture in Pixeltable's favour:
it is the only one of the three that is a system of record. The others forget
everything when the process exits.

---

## 6. Pixeltable's incremental model — does the caching pay off?

Judging Pixeltable purely on recompute throughput measures the one thing its
design deliberately avoids. Its pitch is that you declare a computed column, it
materializes once, every later read is a lookup, and when new rows arrive only
the new rows are computed.

"A database that caches beats an engine that recomputes" is not a finding,
though, so `incremental_probe.py` gives polars-cv the cache a real user would
give it: compute once, write Parquet, read the Parquet back.

200 × 256² PNGs, resize to 224²:

| | Pixeltable | polars-cv |
|---|---:|---:|
| ad-hoc recompute | 376 rows/s | **3,917 rows/s** |
| materialize (one-time) | 248 rows/s | 1,183 rows/s |
| **cached read** | **1,663 rows/s** | **4,204 rows/s** |
| break-even | **2.0 reads** | 32 reads |

**The caching works exactly as advertised.** A cached read is 4.4x Pixeltable's
own recompute rate, and materializing pays for itself after two reads. Appending
is better still: 20 rows added to a 200-row table with a computed column took
0.095 s — only the new rows were computed, about **9.4x cheaper** than
rebuilding the column. polars-cv has no equivalent; incremental maintenance of
derived columns is something you would hand-roll.

**But the ceiling is low.** polars-cv recomputes from files at 3,917 rows/s —
**2.4x faster than Pixeltable reads from its cache.** Caching is a fix for
expensive recomputation, and polars-cv's recomputation is cheap enough that
Parquet caching barely helps it either (1.11x, break-even at 32 reads).

So the value of Pixeltable's model is not speed, it is **bookkeeping**: never
recomputing by accident, never serving a stale derived column, and getting
incremental maintenance for free. If you would otherwise build that yourself,
that is worth real money. If you just want the pixels processed quickly, it is
overhead.

---

## 7. Ease of setup

**For an end user all three are effectively a tie, which is the opposite of what
the "Rust plugin vs pip package" framing suggests.** All ship prebuilt wheels;
timed clean installs from PyPI with a cold cache:

| | wall time | packages pulled | site-packages |
|---|---:|---:|---:|
| `pip install daft` | 1.42 s | 6 | 303 MB |
| `pip install polars-cv` | 1.43 s | 8 | 309 MB |
| `pip install pixeltable` | 3.65 s | 66 | 532 MB |

None compiles anything. Pixeltable is the heaviest — 66 packages including an
**embedded Postgres server** (`pixeltable-pgserver`) and `pgvector` — but it is
still a single `pip install`, and `pxt.init()` starts the database in 0.8 s and
creates its store under `~/.pixeltable` without asking anything of you. For a
tool that is a database, that is a genuinely impressive install story.

The real differences are elsewhere:

| | polars-cv 0.20.0 | Daft 0.7.24 | Pixeltable 0.7.2 |
|---|---|---|---|
| Linux x86_64 / aarch64 | ✅ | ✅ | ✅ |
| macOS arm64 | ✅ | ✅ | ✅ |
| macOS x86_64 | ❌ | ✅ | ✅ |
| **Windows** | ❌ | ✅ | ⚠️ (via WSL) |
| Python floor | 3.10 | 3.10 | 3.10 |
| Persistent state on disk | none | none | `~/.pixeltable` |
| Optional extras | — | 23 (aws, ray, transformers, …) | many (yolox, whisper, LLM providers) |

**Daft wins on platform reach** — no Windows wheel is a hard blocker for a
Windows shop, and polars-cv would have to build from source there.

**Pixeltable is the only one that leaves state behind.** That is the point of
it, but it means an install has a footprint, a database process, and a schema
to manage; the other two are libraries you import and forget.

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

## 8. Robustness

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

## 9. Flexibility

This is the axis where the two table engines clearly win, and it is worth being
precise about *which* flexibility.

### Both are far broader across modalities

Daft exposes **344 functions**; Pixeltable's function namespace spans a similar
range plus model integrations. Neither is something polars-cv attempts:

| Modality | Daft | Pixeltable |
|---|---|---|
| Video | `video_frames`, `video_keyframes`, `video_metadata` | `video` namespace, frame iterators |
| Audio | `audio_file`, `audio_metadata` | `audio` namespace, `whisper`, `whisperx` |
| Documents | — | `document` namespace, `pypdfium2` |
| Text / AI | `embed_text`, `embed_image`, `classify_*`, `llm_generate`, `prompt` | `openai`, `anthropic`, `gemini`, `huggingface`, `ollama`, `llama_cpp`, `together`, … |
| Vision models | — | `yolox`, `vision` namespace |
| Vector search | — | `EmbeddingIndex`, `pgvector`, `BtreeIndex` |
| IO | `download`, `upload`, S3 / GCS / Iceberg / Delta / Hugging Face | media store, external stores |
| Scale | Ray runner, Kubernetes | single node |
| Persistence / lineage | none | **tables, versioning, incremental columns** |

polars-cv has none of this. It has no distributed story at all: it scales to
one machine's cores via the Polars streaming engine and stops there. **If the
requirement is "one engine for video frames, transcripts, embeddings and
images, across a cluster", Daft is the answer. If it is "a place my multimodal
data lives, with derived columns and a vector index", Pixeltable is. polars-cv
is not a candidate for either.**

Pixeltable is also the only one of the three with **built-in vector search**
and a **model zoo** — an embedding index over an image column is a couple of
lines. For a RAG-over-media application it is doing a job the other two do not
attempt.

### polars-cv is deeper within images

| | polars-cv | Daft | Pixeltable |
|---|---|---|---|
| Image operations | **71** | 17 | ~20 (PIL) |
| Arithmetic on image columns | ✅ (`add`, `multiply`, `blend`, `ratio`, …) | ❌ `Cannot multiply` | ❌ 8-bit only |
| Float image data | ✅ | ⚠️ tensors only | ❌ must become `Array` |
| Contour / geometry subsystem | ✅ (12 contour ops, point/bbox namespaces) | ❌ | ❌ |
| Detection metrics (AP, FROC, LROC, bootstrap) | ✅ | ❌ | ⚠️ `vision` has some eval helpers |
| Reductions over images | ✅ (13 `reduce_*`) | ❌ | ⚠️ `histogram`, `entropy`, `getextrema` |
| Morphology, edges, filters | ✅ | ❌ | ❌ |
| Interpolation filter choice | ✅ (nearest/bilinear/lanczos3, per row) | ❌ fixed bilinear | ❌ fixed bicubic |

### Per-row parameters

polars-cv's premise is that any non-structural parameter can be a Polars
expression. Daft supports this in exactly one place:

| | polars-cv | Daft | Pixeltable |
|---|---|---|---|
| `resize` dimensions from a column | ✅ | ❌ rejected | ✅ |
| `crop` box from a column | ✅ | ✅ (needs a `FixedSizeList[UInt32; 4]` cast) | ✅ |
| Interpolation `filter` from a column | ✅ | ❌ (no filter parameter at all) | ❌ (no filter parameter at all) |

**Pixeltable is the better of the two here**, and close to polars-cv:
`t.img.resize([t.w, t.h])` and `t.img.crop([0, 0, t.w, t.h])` both work
directly. Daft handles per-row crop boxes — the detection-cropping case — but
rejects per-row resize targets. Neither exposes an interpolation filter at all,
per row or otherwise.

### Ergonomics

All three are pleasant to write. Daft's and Pixeltable's flat method style
reads slightly shorter for the handful of ops they have; polars-cv's `Pipeline`
object is reusable across queries and composes into a DAG with CSE:

```python
# Daft
df.with_column("out", daft.col("img").decode_image().resize(224, 224).convert_image("L"))

# Pixeltable
t.add_computed_column(out=t.img.resize((224, 224)).convert("L"))

# polars-cv
pipe = Pipeline().source("image_bytes").resize(height=224, width=224).grayscale()
df.with_columns(out=pl.col("img").cv.pipe(pipe).sink("numpy"))
```

Pixeltable's line is arguably the nicest of the three *and* it persists and
stays up to date — that is the whole point of it.

The asymmetry appears at op two. The moment you need a blur, both table engines
put you in a UDF: in Daft, declaring its return dtype by hand and (per §8)
discovering which dtypes survive readback; in Pixeltable, moving the function
into a named module, giving it concrete annotations, and switching the column
to `Array[Float]` if the result is not 8-bit.

---

## 10. When to choose which

**Choose polars-cv when** the workload is image processing, when you need more
than resize/crop/grayscale, when throughput matters (4–10x here on comparable
work), when parameters vary per row, or when you need contours, geometry or
detection metrics.

**Choose Daft when** the workload is multimodal beyond images (video, audio,
text, embeddings), when it must run distributed on Ray or Kubernetes, when you
need Iceberg / Delta / Hugging Face connectors, when LLM and embedding
functions belong in the same query, or when you need a Windows wheel.

**Choose Pixeltable when** you want a *system of record* rather than a query
engine: derived columns that stay correct as rows arrive, no cache invalidation
to hand-roll, versioning and lineage over media, and a place for your images to
actually live. Its incremental maintenance is real (§6) and neither of the
others offers it.

**The dividing line for both table engines is how much work each UDF does.**
Daft's UDF boundary costs ~120 µs per 256² image (§4); Pixeltable's media-store
round trip costs about the same order. If your per-row work is a GPU model, an
embedding or an LLM call — milliseconds — that overhead disappears and their
designs are exactly right. If it is a blur or a threshold — tens of
microseconds — the boundary costs more than the operation, and no tuning fixes
that.

**Use them together** for the obvious splits: Daft for ingest and multimodal
fan-out, Pixeltable as the store of record, polars-cv for the vision transforms
in between. All three speak Arrow or NumPy, so handing a column across is
cheap.

The genuine competitive risk to polars-cv is narrow but real, and Pixeltable is
closer to it than Daft: an engine only needs enough image ops to cover the
common preprocessing path — resize, crop, grayscale, normalize, flip, pad — to
make polars-cv unnecessary for most ML preprocessing. Daft has three of those
six; **Pixeltable has four**, and only lacks `normalize` and `pad`. What stops
both is the same wall: no float image type, so no normalize, so no ML
preprocessing pipeline without Python.

---

## 11. How to reproduce

```bash
cd polars-cv
uv sync --group bench     # daft>=0.7 and pixeltable>=0.7 are in the bench group
uv run --no-sync maturin develop --release

R=benchmarks/reports/2026-08-23-engine-comparison
FW=polars-cv-eager,polars-cv-streaming,daft,daft-udf,pixeltable,pixeltable-udf,opencv,pillow
for sc in single_ops pipelines e2e; do
  uv run --no-sync python -m benchmarks.run_benchmarks --scenario $sc \
    --frameworks $FW \
    --sizes 256,512 --counts 100 --iterations 10 --quiet --output json > $R/$sc.raw
done
# strip the progress preamble the runner writes to stdout, then:
PYTHONPATH=. uv run --no-sync python $R/analyze.py > $R/tables.md
PYTHONPATH=. uv run --no-sync python $R/capability_probe.py
PYTHONPATH=. uv run --no-sync python $R/parallelism_probe.py
PYTHONPATH=. uv run --no-sync python $R/udf_path_probe.py
PYTHONPATH=. uv run --no-sync python $R/incremental_probe.py
```

### Fairness notes

- Both Daft adapters are timed on pre-decoded, materialized DataFrames, exactly
  as the polars-cv adapter is: `prepare_decoded_images` does the decode and the
  frame construction outside the timed section for every framework. **Pixeltable
  is the exception and cannot be made to match** — it has no in-memory
  pre-decoded column, so its figures include a decode (§5). Read the e2e
  scenario for the like-for-like comparison.
- Both `-udf` adapters' UDF bodies call `OpenCVAdapter`, so they and the
  `opencv` adapter run byte-identical kernels and the difference is engine
  overhead alone. Verified: every UDF op matches OpenCV exactly.
- Numerical agreement was checked before timing. `crop` is exact across all
  engines; Daft's `resize` matches polars-cv's bilinear to ±1; Pixeltable's is
  bicubic and differs by up to 21/255 (§5); `grayscale` differs between Daft
  and the rest by the BT.709/BT.601 convention (§2); Pixeltable's `rotate`
  matches polars-cv exactly.
- The harness's own `opencv` adapter rotates about `(w/2, h/2)` rather than the
  pixel centre `((w-1)/2, (h-1)/2)`, so its `rotate_90` is a half-pixel off a
  true `rot90`. polars-cv and Pixeltable agree with each other exactly and both
  differ from it; the OpenCV baseline is the outlier here, not them.
- The `daft` (native-only) adapter raises rather than silently falling back, so
  a missing number in the tables is a missing capability, not a failed run.
- `daft-udf` chains one UDF per operation, which is what an expression chain
  gives a user. That is Daft's *natural* shape, not its *best* shape: §4
  measures the hand-fused alternative and it is 1.98x faster on the heavy
  pipeline, so the multi-op ratios in §3 should be read against the fused
  figures there. Single-op ratios are unaffected — one op is one UDF either way.
- Throughput varies ~15% run to run on this box (the heavy pipeline measured
  4,670 / 4,692 / 5,421 img/s for polars-cv-streaming across three runs), so
  ratios below ~1.2x should not be read as meaningful.
- `e2e_workflow.py`'s dispatch used to route on `"polars" in adapter.name`,
  which would have driven Daft through the per-image path — one DataFrame per
  image per operation. That is now a declared `columnar` flag on the adapter,
  so every dataframe engine gets its batch path.
