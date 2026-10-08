# Regression benchmark harness

A thin layer over the existing benchmark suite that adds **baseline storage +
regression comparison** — the piece needed to prove a change improves
performance across the board with no inadvertent regressions.

It compares **polars-cv against itself** (eager + streaming), before vs after a
change. External frameworks (OpenCV/Pillow/torchvision) are *not* run here —
they belong to `benchmarks.run_benchmarks` for competitive context.

## What it does

- `config.py` — the frozen, reproducible matrix (size, count, pinned threads)
  and the regression thresholds. Single source of truth.
- `selection.py` — `scenario[:glob]` selectors: which cases a run executes.
- `relevance.py` — maps changed files to the selection that can move
  (`--changed REF`).
- `run_suite.py` — runs the selected cases for the two polars-cv adapters,
  repeats the whole suite (best-of per result), and writes a results JSON (plus
  a `.meta.json` sidecar with git SHA / config / threads).
- `compare.py` — loads two result files, computes per-result `%Δ` in
  throughput / latency / memory, classifies each IMPROVED / REGRESSED /
  NEUTRAL / MISSING / NEW, prints a table, and **exits non-zero on any
  regression or missing result** so it can gate.

## Defaults (and why they're what they are)

The default matrix runs the **`pipelines`** scenario at **count=300**, 256×256,
3 warmup + 10 timed iterations, **3 whole-suite repeats** (best-of), pinned to
1 thread. Pipelines exercise the full decode → multi-op → encode hot path
(light / medium / heavy / imagenet / medical) and run in ~3.5 min/run.

These defaults are **empirical**, from same-binary self-checks:

| count | streaming noise | eager noise | verdict |
|-------|-----------------|-------------|---------|
| 50    | up to ±12%      | up to ±7%   | false regressions — too few morsels |
| 300   | ≤3%             | ≤5%         | all NEUTRAL — trustworthy gate |

So **count=300 is the floor** for a reliable gate; smaller batches are
dominated by per-call/scheduling overhead (50 rows ÷ morsel-size-10 = only ~5
morsels). The **gate metric is throughput only** — latency is its reciprocal
(double-counting), and peak memory is whole-process RSS (advisory; enable with
`--gate-memory`). The 7% threshold sits above the ≤5% noise floor with margin.

For broader across-the-board coverage add the other scenarios (slower). The
`targeted` scenario includes the `split_` cases, which need a parallel pool
(see `@threads=N` below), so the whole suite runs on 4 threads; for the 1-thread
view, name its other prefixes:

```bash
python -m benchmarks.regression.run_suite --out candidate-t4.json --threads 4 \
    --select single_ops,pipelines,e2e,targeted,zero_copy,remote
python -m benchmarks.regression.run_suite --out candidate-t1.json \
    --select single_ops,pipelines,e2e,zero_copy,remote,targeted:codec_*,targeted:sink_*,targeted:blob_*,targeted:geom_*
```

## Running only what a change can move

`--select` takes comma-separated `scenario[:glob]` selectors. A bare scenario
runs all of its cases; a glob (`fnmatch`) picks cases by the `operation` name
their results carry — the names `compare` prints:

```bash
--select single_ops:rotate_*,pipelines:medium_pipeline
--select targeted:geom_*          # every geometry accessor case
--select targeted:sink_*,zero_copy
```

`zero_copy` and `remote` run as a unit (their own fixed matrix) and take no
glob. An unknown scenario, a glob matching no case, or a glob on a unit
scenario is an error, never a run of less than asked.

`--changed REF` derives the selection from the files that differ from REF's
merge base (committed, uncommitted and untracked), via `relevance.RULES`:
a CODEOWNERS-style map, last match wins, from source paths to the selectors
whose cases execute them. Tests, docs and the harness select nothing; a
dependency or toolchain change selects everything; a code file no rule covers
is an error (add a rule — `""` if no scenario measures it). A guard test holds
every tracked source file to a rule and parses every rule's selectors.

The base side must run the **same** selection, but after checking out the base
`--changed` sees no change. So resolve it once, on the change:

```bash
SEL=$(python -m benchmarks.regression.relevance origin/main)
```

and pass `--select "$SEL"` to both runs (the `.meta.json` records the
`selection` each run used, too).

### Code that only runs in parallel: `@threads=N`

The plugin's row splitter (`row_split.rs`) spreads a call's rows over the
pool, and on one thread it does nothing. Benchmarked at the default
`--threads 1`, a change to it times every case it selects and measures none
of itself. That is how PR #124's splitter rewrite shipped with the eager
`list` sink 15.6% slower and streaming `sobel_x` 11.5% slower at 4 threads:
the cases it ran held, and the ones that moved were neither selected nor run
in parallel.

So a selection carries the pool it needs (`Selection.min_threads`):

- the `@threads=N` marker, which `relevance` adds for the splitter, and which
  `SEL` therefore carries to both sides: `…,targeted:sink_list_u8,@threads=4`;
- a case's own requirement: `targeted`'s `split_` cases declare
  `min_threads=2`, so a selection that includes them (a bare `targeted` too)
  needs a parallel pool.

`run_suite` refuses a smaller `--threads` (and, as a backstop for programmatic
callers, a smaller actual pool), and `compare` refuses two results run on
different thread counts. The count cannot default from the selection, since
reading the cases sizes the pool, so pass it: `--threads 4` on both sides.

Noise is larger on a parallel pool. Measured in a 4-vCPU container at the
default count, a case can move ±10% between runs of one binary at 4 threads
(against ≤5% at 1). Confirm a flagged case by interleaving the sides
(base, head, head, base, …) and comparing medians, rather than trusting one
pair of runs.

### The `targeted` scenario

The adapter scenarios always decode PNG and sink to numpy, so whole subsystems
went unmeasured. `targeted` times them directly, eager, grouped by prefix:
`codec_` (PNG/JPEG decode and re-encode), `sink_` (the `array`/`list` tensor
sinks, including a transposed input), `blob_` (the blob source, plain and
through a fused op) and `geom_` (the `.contour`/`.point` accessors and contour
rasterization). Image cases use the suite's counts and sizes; geometry cases run
`count × 300` rows (point-only ones `count × 3000`, so a call is not ~5 ms) and
report `image_size` `(0, 0)`.

Each case runs in **its own process**, and reports the **median** call. In a
shared process a case's timing depended on the allocator state the cases
before it left — `geom_contour_translate` measured 588–597k rows/s alone and
545–704k inside the suite — and `--select` changes which cases precede which.
The median because a call that allocates its whole output has occasional slow
outliers that move a ten-call mean. Inputs are deterministic and built once
per machine (Arrow IPC under the temp directory, keyed by the scenario's
source), since encoding 300 PNGs alone takes ~14 s. Same-binary self-check of
all 17 cases at the defaults: every one NEUTRAL, within ±5%; a full `targeted`
run takes ~5 minutes, `targeted:geom_*` ~3.5.

### The `remote` scenario

`--select remote` measures the `file_path` **fetch** stage — the one every
`s3://`, `gs://`, `az://` and `http://` source goes through, and the one no
other scenario touches, since they are all handed bytes that are already in
memory. It serves a generated corpus over loopback HTTP, so it needs no
credentials and no network.

It reports three timings whose *differences* isolate a stage
(`remote_local_paths` as control, `remote_http_paths`, and
`remote_http_read_bytes` for fetch without decode) plus one thing that is not a
timing: **requests per connection**, counted by the server. A ratio of 1.00
means the client opens a fresh connection for every file — on loopback that
costs about half a millisecond, but on a real endpoint it is a TCP *and* TLS
handshake per file.

Run it standalone for the full report, including the connection count, which
the suite's results JSON has nowhere to put:

```bash
python -m benchmarks.scenarios.remote_source --count 300
python -m benchmarks.scenarios.remote_source --count 300 --latency-ms 20  # model a WAN link
```

## Workflow

Build both sides with the **same optimised profile**, on the **same machine**,
with the **same `--threads`** and **`--select`**. Close other heavy processes.

`maturin develop --profile benchmark` is the benchmark build: release
(`opt-level = 3`, `panic = "unwind"`) with thin LTO and 16 codegen units
instead of fat LTO and one. Measured in a web container:

| build | cold (all deps) | rebuild after a source change |
|-------|-----------------|-------------------------------|
| `--release` (fat LTO) | ~18 min | ~10 min |
| `--profile benchmark` (thin LTO) | 11 min | **3.8–4.2 min** |

The rebuild is what a base-vs-head loop pays twice. Thin LTO ran the
`pipelines` cases within ±9% of a fat-LTO build (mean −2.5%), so numbers from
the two profiles are **not comparable**: build both sides with the same one.
`--release` still works when you want wheel-identical absolute numbers.

Each results file's `.meta.json` records the extension's `build_profile`
(`_lib.__build_profile__`) and whether it was a debug build, and `compare`
refuses two different profiles or any debug build. `run_suite` refuses a debug
extension outright; `--allow-debug-build` exists only to smoke-test the
harness.

```bash
cd polars-cv
SEL=$(python -m benchmarks.regression.relevance origin/main)   # or pick by hand

# 1) Baseline: the code BEFORE your change
git stash            # or check out the base commit
maturin develop --profile benchmark
python -m benchmarks.regression.run_suite --select "$SEL" --out baseline.json

# 2) Candidate: the code WITH your change
git stash pop        # or check out your branch
maturin develop --profile benchmark
python -m benchmarks.regression.run_suite --select "$SEL" --out candidate.json

# 3) Gate: non-zero exit if anything regressed
python -m benchmarks.regression.compare baseline.json candidate.json
echo "exit=$?"   # 0 = no regressions, 1 = regression/missing
```

A change is kept only if `compare` exits 0 **and** shows at least one IMPROVED.

## Self-check (validate the harness before trusting it)

Run the suite twice on the *same* binary and compare — every result should be
NEUTRAL. If not, the noise floor exceeds the thresholds; raise `--repeats` or
widen the thresholds before using it as a gate.

```bash
python -m benchmarks.regression.run_suite --out a.json
python -m benchmarks.regression.run_suite --out b.json
python -m benchmarks.regression.compare a.json b.json   # expect all NEUTRAL, exit 0
```

## Options

`run_suite`: `--out` (required), `--select scenario[:glob],...[,@threads=N]` or
`--changed REF`, `--counts`, `--sizes`, `--threads`, `--repeats`, `--warmup`,
`--iterations`, `--quiet`, `--allow-debug-build`.

For a quick directional read while iterating, cut the repeats rather than
the count (below 300, streaming noise reaches ±12%): `--repeats 1
--iterations 5` runs 8 timed-or-warmup passes per case instead of 39. It is
noisier than the default — read it for direction, gate on the defaults.

`compare`: `baseline candidate`, `--throughput-threshold` (default 5),
`--latency-threshold` (5), `--memory-threshold` (15), `--gate-memory`
(memory is advisory unless set), `--json` (machine-readable summary).

## CI (manual, advisory)

`.github/workflows/benchmark.yml` runs this suite on demand
(`workflow_dispatch`, inputs: select / counts / threads), builds the benchmark
profile,
and uploads the results JSON as an artifact. It is **not** a PR gate —
shared CI runners are too noisy to gate on absolute timings. Download two runs'
artifacts and `compare` them locally.

## Notes

- Throughput is the only gated metric; latency is its reciprocal (shown for
  context). Peak memory is whole-process RSS (noisy) so it is advisory by
  default (`--gate-memory` to include it).
- The underlying scenarios report the mean over `--iterations`; `--repeats`
  runs the whole suite N times and keeps best-of, which is the only available
  noise-rejection lever.
- `compare.py` is pure stdlib and runs even where polars-cv isn't built.
