# Remote fetch as a window ahead of the rows (Phase 2 of the Polars 2.0 plan, PR #124)

A call's remote paths used to be fetched all at once before its first row
decoded. Now each row fetches its own path and those of the next rows, up to
polars' concurrency budget (`fetch::Fetcher`).

**Builds:** both `maturin develop --profile benchmark`, swapped per run (the
`.meta.json` `git_sha` is the working tree's).
- **base:** Phase 3 (`67f656b`).
- **head:** this change.

Runs interleaved base → head → base → head:
- `t{1,4}-*.json`: `run_suite --select remote --threads {1,4}`, loopback with
  no latency.
- `lat-t{1,4}-*.txt`: `python -m benchmarks.scenarios.remote_source
  --latency-ms 20 --iterations 3` with `POLARS_MAX_THREADS={1,4}`.

Throughput, Δ head vs base per pair:

| case | 1 thread | 4 threads |
|------|----------|-----------|
| `remote_http_paths`, no latency | **+50.6% / +60.8%** | −5.6% / −3.8% |
| `remote_http_paths`, 20 ms per request | **+44.8% / +42.9%** | **+6.5% / +5.5%** |
| `remote_http_read_bytes`, no latency | +5.5% / +7.8% | −5.5% / −9.5% (base's own spread 11%) |
| `remote_http_read_bytes`, 20 ms | +2.0% / +0.1% | −1.1% / −0.5% |
| `remote_local_paths` (control, no fetch) | +0.7% / −4.5% | +2.2% / −8.5% |

- **Overlap is the gain.** With one row thread, the old code downloaded
  every file before decoding any, so fetch and decode times added up. Now
  the next images download while the current one decodes. With real
  latency the gain holds at 4 threads too.
- **The one cost: 4 threads over loopback with no latency, −4 to −6% on
  `http_paths`.** This is the case most favourable to the old design. The
  corpus arrives in one burst in a fraction of the decode time, and decoding
  then had all four cores to itself. Now polars' async workers read bodies
  while four row threads decode, sharing the cores. `read_bytes` (no decode)
  moved by about its own run-to-run spread.
- **Memory:** a call now holds about a window per row thread instead of the
  whole call (`tests/test_fetch_window.py`: 3–7 bodies resident instead of
  64/128). Here the peak RSS of `http_paths` at 4 threads fell from 334 to
  308 MB. The corpus (300 small PNGs) is small next to the process, so on
  a large row group the drop is proportionally larger.
- **Connection reuse is unchanged:** every run served its 1,200 requests over
  pooled connections (`connections=0` in the window is the counter's "all
  reused" case).
