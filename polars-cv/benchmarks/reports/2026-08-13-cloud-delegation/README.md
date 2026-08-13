# Delegating the cloud path to polars-io

Date: 2026-08-13 · Follows `../2026-08-13-0.20.0-release/`, which measured the
problem and deliberately did not fix it.

Both sides are `maturin develop --release` on the same host, same benchmark,
same flags. The "before" column is the measurement recorded in the 0.20.0 report;
the "after" column is this branch.

## The headline number is a count, not a timing

| | before | after |
|---|---|---|
| requests per connection | **1.00** | **∞** (1200 requests, 0 newly accepted connections) |

1.00 meant a fresh TCP handshake — and off loopback a fresh TLS handshake — for
every single image. ∞ means every request in the measured window rode a pooled
connection. Counted by the server, so it is build- and machine-independent.

Measured separately across three successive plugin calls over 64 URLs: **16
connections for the first call** (one per concurrent worker, not one per file)
and **none at all** for the calls after it. That second part is the streaming
property — a client scoped to a plugin call is rebuilt on every morsel, however
wide the batch.

## Throughput

300 images, 256×256, no injected latency:

| operation | before (img/s) | after (img/s) | |
|---|---|---|---|
| `remote_http_read_bytes` (fetch only) | 902.4 | **1681.4** | **1.86×** |
| `remote_http_paths` (fetch + decode + ops) | 481.9 | **653.6** | 1.36× |
| `remote_local_paths` (control) | 1173.7 | 1104.7 | — |

Per-file fetch overhead: **1.108 ms → 0.595 ms**.

The control is the load-bearing row. `remote_local_paths` runs the identical
pipeline over local filesystem paths and touches none of the changed code; it
did not improve. So the gains above are attributable to the fetch path rather
than to the host being in a better mood than it was in the morning — which
matters, because the 0.20.0 report established that this class of host has a
noise floor above the suite's 7% gate.

## Concurrency under latency

64 files, 20 ms injected per-request latency, `POLARS_CONCURRENCY_BUDGET=16`.
The ideal is 4 waves × 20 ms = 80 ms.

| | before | after |
|---|---|---|
| fetch-only, ms/file | 2.115 | **1.650** |
| total for 64 files | 135 ms | **106 ms** |
| over ideal | +55 ms | **+26 ms** |

Half the remaining gap closed. What is left is not the barrier — that is gone —
but the test server itself: a Python `ThreadingHTTPServer` sleeping 20 ms per
request in a thread is not free at 16-way concurrency.

## What changed

Three commits, each with its own guard:

1. **One HTTP client** (`6383e9b`). `reqwest::Client::new()` moved out of the
   per-file read into a process-wide `OnceLock`. A `Client` *is* the connection
   pool. Deliberately not delegated: polars does not cache HTTP object stores,
   which suits a few large Parquet files rather than many small images.
2. **polars-io owns the object stores** (`a76b94f`). `read_s3`/`read_gcs`/
   `read_azure` are gone; `build_object_store` provides a process-wide store
   cache keyed on bucket *and credentials*, and
   `exec_with_rebuild_retry_on_err` refreshes a store in place on any error.
   That is what makes caching a credentialed store safe — the question 0.20.0
   measured but declined to answer.
3. **A global concurrency budget** (`f1812d9`). The 16-per-call constant and its
   chunk barrier are gone. Every request takes one permit from polars'
   process-wide semaphore, so `POLARS_CONCURRENCY_BUDGET` bounds our fetches and
   polars' own scans together — the per-call cap could not bound anything under
   streaming, where the engine multiplies it by the morsels in flight.

## Two things worth knowing

**The provider `Arc` is the cache key.** `PlCredentialProvider::stable_cache_key`
returns the `Arc`'s *address*, which polars folds into the store-cache key. A
provider built fresh per call would therefore defeat the cache completely and
silently — this whole change undone, one layer down, with every measurement
above reverting to 1.00. `cloud_auth::credential_provider` memoizes on credential
identity, and `provider_identity_tracks_the_credential_and_only_the_credential`
pins both directions (same credential ⇒ same `Arc`; different ⇒ different).

**polars silently drops unknown storage options.** `parse_untyped_config` is a
`filter_map` over `from_str(..).ok()`. Delegating naively would have turned
`unknown GCS storage option 'x'` into a request that quietly did the wrong thing,
so keys are validated against each backend's vocabulary before the options are
handed over.

## Reproducing

```bash
cd polars-cv
maturin develop --release

python -m benchmarks.scenarios.remote_source --count 300
POLARS_CONCURRENCY_BUDGET=16 python -m benchmarks.scenarios.remote_source \
    --count 64 --latency-ms 20

# The properties above as pass/fail, off the server's counters rather than a clock:
pytest tests/test_remote_connection_reuse.py -v
```
