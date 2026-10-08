# One async runtime (Phase 3 of the Polars 2.0 plan, PR #124)

Remote reads moved from a tokio runtime of the plugin's own (one worker per
CPU) to polars' `ASYNC` runtime (the plugin's copy, `min(POLARS_MAX_THREADS,
32)` workers unless `POLARS_ASYNC_THREAD_COUNT` says otherwise). At the
harness default of one thread, the fetch now has one async worker instead of
four, so the `remote` scenario was checked at 1 and 4 threads.

**Builds:** both `maturin develop --profile benchmark`.
- **base:** `a2691d0`'s Rust sources.
- **head:** this change on top of Phase 1, which does not touch the fetch
  path. At 1 thread nothing splits; at 4 threads `remote_*` calls are eager
  and split as before.

The extensions were swapped per run (the `.meta.json` `git_sha` is the
working tree's). Runs interleaved base → head → base → head, `t1-*` and
`t4-*`. Throughput, images/s:

| threads | case | base1 | head1 | base2 | head2 | pair 1 / pair 2 | same-binary base / head |
|---------|------|------:|------:|------:|------:|-----------------|-------------------------|
| 1 | `remote_http_paths` | 636.5 | 688.9 | 647.9 | 674.2 | +8.2% / +4.1% | +1.8% / −2.1% |
| 1 | `remote_http_read_bytes` | 1562.4 | 1911.9 | 1665.3 | 1646.6 | +22.4% / −1.1% | +6.6% / −13.9% |
| 1 | `remote_local_paths` (control) | 1159.9 | 1270.1 | 1187.1 | 1152.4 | +9.5% / −2.9% | +2.3% / −9.3% |
| 4 | `remote_http_paths` | 1117.8 | 1158.9 | 1098.5 | 1193.3 | +3.7% / +8.6% | −1.7% / +3.0% |
| 4 | `remote_http_read_bytes` | 1679.4 | 1715.7 | 1763.9 | 1575.7 | +2.2% / −10.7% | +5.0% / −8.2% |
| 4 | `remote_local_paths` (control) | 4156.0 | 4145.2 | 3820.7 | 4085.6 | −0.3% / +6.9% | −8.1% / −1.4% |

**No regression.** `remote_http_paths` is slightly faster in all four pairs.
`read_bytes` and the local-path control move in both directions, within
their same-binary spread. One async worker is enough for loopback fetches,
which are IO-bound; the decode runs on the row threads, not on the runtime.
