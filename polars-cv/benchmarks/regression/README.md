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

For broader across-the-board coverage add the other scenarios (slower):

```bash
python -m benchmarks.regression.run_suite --out candidate.json \
    --scenarios single_ops,pipelines,e2e
```

## Workflow

Use a **release build** for both runs, on the **same machine**, with the
**same `--threads`**. Close other heavy processes.

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

`run_suite`: `--out` (required), `--scenarios single_ops,pipelines,e2e[,zero_copy]`,
`--counts`, `--sizes`, `--threads`, `--repeats`, `--warmup`, `--iterations`,
`--quiet`.

`compare`: `baseline candidate`, `--throughput-threshold` (default 5),
`--latency-threshold` (5), `--memory-threshold` (15), `--gate-memory`
(memory is advisory unless set), `--json` (machine-readable summary).

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
