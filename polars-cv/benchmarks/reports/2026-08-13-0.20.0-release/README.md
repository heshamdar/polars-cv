# 0.20.0 release check — no regression, and the first look at the remote source

Date: 2026-08-13
Baseline: `b9da203` (the 0.19.0 version bump) · Candidate: `5819ae8` (0.20.0)

Both arms `maturin develop --release`, same host, `--threads 1`, all five
scenarios (`single_ops,pipelines,e2e,zero_copy,remote`) at `--counts 300
--repeats 3`. 66 results per run.

Artifacts: `baseline.json`, `candidate.json`, `candidate2.json` (a second
candidate run on the same binary), `ab_results2.jsonl` (the interleaved A/B
below), plus each run's `.meta.json`.

## Verdict

**No performance regression.** The suite's gate flagged two results, and both
dissolve under scrutiny — one was reproduced by running the *same binary* twice,
the other was disproved by an interleaved A/B.

Aggregate drift, median over all 66 keys:

| comparison | median throughput ratio |
|---|---|
| candidate ÷ baseline (run 1) | 0.9935 |
| candidate ÷ baseline (run 2) | 0.9877 |
| candidate run 2 ÷ candidate run 1 (**same binary**) | 0.9907 |

The candidate-vs-candidate ratio is the same size as the candidate-vs-baseline
ratio. Whatever this machine is doing, it is not attributable to the code.

## The host is noisier than the harness's thresholds assume

`benchmarks/regression/README.md` prescribes a same-binary self-check before
trusting the gate. Running it here is why this report can say anything at all:

| | count |
|---|---|
| self-check (candidate vs candidate) | IMPROVED 1, **REGRESSED 2**, NEUTRAL 63 |
| real comparison (baseline vs candidate) | IMPROVED 1, **REGRESSED 2**, NEUTRAL 63 |

Identical shape. The same-binary spread per operation:

| statistic | spread |
|---|---|
| median | 1.8% |
| 90th percentile | 5.2% |
| worst (`streaming normalize`) | 17.2% |
| second worst (`streaming blur`) | 10.0% |

So the 7% gate produces false positives on a handful of high-variance operations
on this host, and a run-A-then-run-B design lets slow drift land entirely on one
arm. **A single `compare` run on a shared/virtualised host is not evidence of a
regression** — the README already says CI runners are too noisy to gate on; this
records that the same caution applies to any non-dedicated host.

## The one result that needed settling

`polars-cv-eager sobel_x` was the only key below baseline by >7% in *both*
candidate runs (−13.4%, −10.2%), and it is not a high-variance benchmark, so
noise was not a sufficient explanation.

Settled by an interleaved A/B: both `_lib.abi3.so` files kept on disk and swapped
between invocations, alternating arms so drift lands on both equally, driving the
exact path `scenarios/single_ops.py` measures for `sobel_x` (the adapter's
pre-decoded blob source), 12 timed iterations per invocation, 6 invocations per
arm.

| metric | baseline | candidate | delta |
|---|---|---|---|
| best-of, img/s | 2496.1 (sd 57.5) | 2534.1 (sd 51.3) | **+1.5%** |
| median, img/s | 2422.3 (sd 46.1) | 2452.9 (sd 44.3) | **+1.3%** |

The candidate is marginally *faster*. The suite's baseline sample of 2745 img/s
was an outlier high; the candidate's 2376/2465 bracket the honest value of ~2500.

## Remote source: the first measurement of the fetch path

`scenarios/remote_source.py` is new in this release — before it, every scenario
was handed bytes already in memory, so `fetch.rs` / `cloud.rs` had no coverage.
Against a loopback HTTP server, 300 images at 256×256:

| operation | img/s | ms/image |
|---|---|---|
| `remote_local_paths` (control) | 1173.7 | 0.85 |
| `remote_http_paths` | 481.9 | 2.08 |
| `remote_http_read_bytes` (fetch only, no decode) | 902.4 | 1.11 |

Baseline and candidate agree to within 1.2% on all three, so nothing in this
release changed the fetch path. What the numbers say about the path itself:

**The client opens one connection per file.** The server counted
`1.00 requests/connection` — 256 connections for 256 requests. `read_http`
constructs a `reqwest::Client` *inside* the per-file read, and `read_s3` /
`read_gcs` / `read_azure` each build a fresh backend store per file. Only the
tokio runtime is shared (`cloud::get_runtime`).

A control against the same server quantifies what that costs:

| client | ms/file | connections |
|---|---|---|
| one reused connection | 0.34 | 1 |
| fresh connection per file | 0.81 | 64 |

≈0.5 ms/file on loopback **with no TLS**. Against a real S3/GCS/Azure endpoint a
fresh connection is a TCP handshake *and* a TLS handshake — several round trips
per file rather than zero, so the penalty scales with the link's RTT rather than
staying at half a millisecond.

Concurrency behaves roughly as designed but not ideally. With 20 ms injected
per-request latency and `fetch::DEFAULT_CONCURRENCY = 16`, fetching 64 files took
2.11 ms/file (135 ms total) against an ideal of 4 waves × 20 ms = 80 ms.
`read_files_concurrent` spawns a chunk of 16 threads and **joins all of them**
before starting the next chunk, so the slowest file in each chunk stalls the
whole chunk — a work queue would not have that barrier.

**Not changed in this release.** Caching a credentialed store is a correctness
question before it is a performance one (a cached store outliving its
credentials is a different class of bug), so this is measured and reported
rather than fixed.

## Reproducing

```bash
cd polars-cv
maturin develop --release
python -m benchmarks.regression.run_suite --out candidate.json \
    --scenarios single_ops,pipelines,e2e,zero_copy,remote \
    --counts 300 --threads 1 --repeats 3

# The self-check that makes the comparison interpretable — run it first.
python -m benchmarks.regression.run_suite --out candidate2.json <same flags>
python -m benchmarks.regression.compare candidate.json candidate2.json

# The remote scenario standalone, which also prints the connection count.
python -m benchmarks.scenarios.remote_source --count 300
python -m benchmarks.scenarios.remote_source --count 300 --latency-ms 20
```

Note that `--scenarios zero_copy` did not run at all before this release; it
raised `AttributeError` in `_aggregate_best`. See the 0.20.0 changelog entry.
