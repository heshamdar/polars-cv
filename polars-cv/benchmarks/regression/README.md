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
- `run_suite.py` — runs the configured scenarios for the two polars-cv adapters,
  repeats the whole suite (best-of per result), and writes a results JSON (plus
  a `.meta.json` sidecar with git SHA / config / threads).
- `compare.py` — loads two result files, computes per-result `%Δ` in
  throughput / latency / memory, classifies each IMPROVED / REGRESSED /
  NEUTRAL / MISSING / NEW, prints a table, and **exits non-zero on any
  regression or missing result** so it can gate.

## The matrix (and why it is what it is)

Three cells, each run in **its own subprocess** — `POLARS_MAX_THREADS` sizes the
pool at the first polars import and is inert afterwards, so one interpreter
cannot host two thread counts.

| cell | what only this cell can catch |
|---|---|
| `eager@1` | The sequential row loop, per-row `ViewExpr` re-planning, decode/encode, and the `Vec::with_capacity(n_rows)` buffering of every row result before the output column is built. Blind to anything concurrent. |
| `streaming@1` | Per-call fixed cost with parallelism held out: the compiled-graph cache lookup and the per-morsel source resolution, which eager amortises over a whole column. |
| `streaming@auto` | **Morsel scaling**, and the only cell that runs the concurrent path through the global compiled-graph cache `Mutex`. |

This follows from one fact about the engine: **the plugin does not parallelise
within a call.** `CompiledGraph::execute` is a plain sequential
`for row_idx in 0..len` loop, and there is no rayon anywhere in either Rust
crate. All multi-core execution comes from the polars **streaming** engine
slicing the input into morsels and invoking the plugin concurrently.

Two consequences worth stating plainly:

- **`POLARS_MAX_THREADS` does nothing to eager.** Pinning the whole suite to one
  thread — which is what this harness used to do — compared eager (unaffected)
  against streaming crippled to one morsel at a time, so no regression in
  scaling was detectable at all.
- **`eager@N` does not exist.** With a sequential row loop it is byte-for-byte
  `eager@1`, so `--cells eager@4` is rejected with that reason rather than
  quietly measured and read as a scaling result.

The derived gate metric is

```
scaling_efficiency = throughput(streaming@N) / (N × throughput(streaming@1))
```

a ratio of ratios, so it is insensitive to absolute machine speed and to a
2-vCPU runner versus a 4-vCPU one. A change can leave every individual
throughput inside its noise band and still have serialised the concurrent path;
this is where that shows up.

## Defaults

The default matrix runs the **`pipelines`** scenario at **count=300**, 256×256,
3 warmup + 10 timed iterations, **3 whole-suite repeats** (best-of). Pipelines
exercise the full decode → multi-op → encode hot path (light / medium / heavy /
imagenet / medical).

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

For broader across-the-board coverage add the other scenarios (slower):

```bash
python -m benchmarks.regression.run_suite --out candidate.json \
    --scenarios single_ops,pipelines,e2e
```

### The `remote` scenario

`--scenarios remote` measures the `file_path` **fetch** stage — the one every
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

## Results format

A results file is an envelope:

```json
{"schema_version": 2, "meta": {...}, "results": [...]}
```

`meta` carries `build_info()`, the detected build profile, the polars version,
`POLARS_CV_OPTIMIZATIONS`, the CPU model, the usable core count, and each
cell's *actual* thread-pool size and peak RSS. It rides inside the file rather
than in a `.meta.json` sidecar, because a sidecar can be renamed, lost or
dropped by a partial copy — and the one that existed was never read.

`compare` checks that provenance **before** comparing anything and exits 2 on a
mismatch in build profile, optimizations, polars version, the cells' actual
pools, or the measured matrix. Each of those silently produced a meaningless
PASS or FAIL before. `--allow-mismatch` is the explicit override.

v1 files (a bare array) still load, so the committed `reports/**` baselines
remain readable as archives — but v1-against-v2 is refused rather than coerced.
Those runs came from machines that no longer exist, so they are not comparable
regardless of format.

## Workflow

Use a **release build** for both runs, on the **same machine**. Close other
heavy processes.

```bash
cd polars-cv

# 1) Baseline: the code BEFORE your change
git stash            # or check out the base commit
maturin develop --release
python -m benchmarks.regression.run_suite --out baseline.json

# 2) Candidate: the code WITH your change
git stash pop        # or check out your branch
maturin develop --release
python -m benchmarks.regression.run_suite --out candidate.json

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

`run_suite`: `--out` (required), `--scenarios single_ops,pipelines,e2e[,zero_copy][,remote]`,
`--counts`, `--sizes`, `--cells` (e.g. `streaming@1,streaming@auto`),
`--repeats`, `--warmup`, `--iterations`, `--quiet`.

`compare`: `baseline candidate`, `--throughput-threshold`,
`--memory-threshold`, `--gate-memory` (memory is advisory unless set),
`--allow-mismatch FIELD[,FIELD]`, `--json`. Defaults come from `Thresholds` in
`config.py` — run `compare --help` to see them rather than trusting a number
written down here, which is how the previously documented 5/5/15 drifted from
the real 7/7/20.

## CI (manual, advisory)

`.github/workflows/benchmark.yml` runs this suite on demand
(`workflow_dispatch`, inputs: scenarios / counts / threads), builds release,
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
