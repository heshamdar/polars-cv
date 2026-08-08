# Panic audit — input-reachable `panic!`/`unwrap`/`expect` in the execution path

Scope: find `panic!` / `unreachable!` / `unwrap()` / `expect()` / unchecked
indexing that a **user can trigger with ordinary inputs** (bad params, malformed
source bytes, unexpected shapes/dtypes) — as opposed to internal invariants or
test-only code. Fixes that were safe and local were applied; the rest are listed
here for triage per the agreed scope ("fix the cheap ones, list the rest").

Counts at audit time: ~103 sites in `polars-cv/src`, ~77 in `view-buffer/src`
(both include test modules). The vast majority are **not** input-reachable.

!!! note "Line references drift"
    Every site below is named by file *and* by the expression it refers to. The
    expression is the identifier; the line number is a convenience that goes
    stale with the next edit. Re-verified against `0.19.0` — two entries from
    the original audit were found already handled and have been struck (see
    *Resolved since the audit*).

## Summary of findings

| Layer | Verdict |
|-------|---------|
| `polars-cv` param resolution (`params.rs`) | **Safe.** Typed downcasts (`series.u8().unwrap()`) are guarded by the matching `DataType` arm; `resolve_usize` rejects negatives; `resolve_i64/f64` return `Result`. |
| `polars-cv` op dispatch (`execute.rs`) | **Safe.** Every arm returns `PolarsResult`. The `warp_affine` matrix `try_into` is length-checked first; enum/binary lookups `.expect()` only after a membership guard; remaining `.expect()`s are in `#[cfg(test)]`. |
| `polars-cv` source decode (`graph/decode.rs`) | **Safe.** The blob/VIEW header parse is gated by `total_len < HEADER_SIZE` (64) before any fixed-offset read, and shape/stride loops bounds-check every slice; shape-product uses `checked_mul`. |
| `polars-cv` sink encode (`graph/encode.rs`) | **Mostly safe** (Result-returning), but a few `shape[0]` accesses assume a rank the planner is expected to guarantee — see triage list. |
| **`view-buffer` op apply fns** | **Reachable panic class.** `validate()` is defined per op but **never called on the execution path** (only in tests), and apply fns index `shape[2]` etc. without a runtime rank guard. A buffer whose rank/shape doesn't match the op (e.g. a 2-D buffer reaching an HWC color op via `reshape`/`squeeze` with a per-row expr the planner can't see) panics with an index-out-of-bounds instead of a graceful error. |

Net: the **Result-returning `polars-cv` layer is defensively written** — no cheap
input-reachable panic was found to fix there. The real class lives one layer
down in `view-buffer`, and the clean fix is a small architectural change (below),
not a scatter of local edits — so it is left for triage as agreed.

## Root cause of the reachable class

`view_buffer::ops::Op::validate(input_shapes, input_dtypes)` exists and is
correct, but the graph executor (`polars-cv/src/graph/compiled.rs` →
`view-buffer` apply fns) applies ops **without calling it**. Confirmed: the only
non-test `.validate(` call sites in the whole workspace are unit tests. So the
shape/dtype contracts are enforced at *plan* time (Python planner, `op_schema`)
but not defensively re-checked at *execution* time, and per-row expression
params (e.g. `reshape([pl.col(...), ...])`) can produce a runtime shape the
planner never saw.

### Recommended fix (triage)

Wire `Op::validate()` into the step-application layer that already returns
`PolarsResult` (`compiled.rs`, where buffer ops are collected into
`pending_buffer_ops` before execution): validate each op against its concrete
input shape/dtype and return a `ComputeError` on failure. This converts the
entire class from process-abort to graceful per-row error (honoring the existing
`on_error="null"` policy) in one place, without changing the `view-buffer` apply
signatures (which return `ViewBuffer`, not `Result`). Risks to weigh: (a) fusion
collapses the buffer-op chain, so mid-chain shapes need re-derivation to validate
each op precisely; (b) `validate()` must not be stricter than today's accepted
inputs or it will reject currently-working pipelines — needs a test pass.

## Concrete reachable sites to triage (view-buffer apply fns)

Each indexes a fixed axis assuming rank ≥ 3 (HWC) with no runtime guard:

- `view-buffer/src/ops/color.rs` (`~:160`, `~:454`, `~:481`) —
  `(h, w, c) = (shape[0], shape[1], shape[2])`
- `view-buffer/src/execution/runner.rs` (`~:115-116`) — `buf.shape()[0]/[1]`
- `view-buffer/src/execution/runner.rs` (`~:433`) — `let c = shape[2]`
- `view-buffer/src/execution/runner.rs` (`~:484-485`) — `buffers[0].shape()[0]/[1]`
  (also assumes `buffers` is non-empty for channel-merge)

Sites that already guard (`if shape.len() == 3 { shape[2] } else { 1 }`) — e.g.
`color.rs` (`~:209`, `~:255`), `compute.rs`, `pad.rs`, `runner.rs` (`~:227`,
`~:286`) — are **not** reachable panics and need no change.

## Secondary triage (polars-cv encode)

- `graph/encode.rs` (`~:396`, `~:531`, `~:549`) — `shape[0]` on the
  reduction/vector encode path; reachable only if an empty-shape buffer arrives.
  Low risk (reductions produce rank ≥ 1) but worth a defensive `first()` check
  when the validate wiring lands.

## Resolved since the audit

Two entries from the original list no longer hold and are recorded here so they
are not "re-found" and re-triaged:

- `view-buffer/src/ops/mask.rs` — `c = buf_shape[2]` was listed as unguarded. It
  is reached only inside `if mask_shape.len() == 2 && buf_shape.len() == 3`, so
  the rank is checked before the index. Not a reachable panic.
- `polars-cv/src/graph/encode.rs` — `&contours[0]` no longer exists; the contour
  encode paths iterate the set rather than indexing its first element.

## Out of scope (confirmed safe / not user input)

- Mutex `lock().unwrap()` (poisoning is unrecoverable by design).
- `results.get_mut(alias).unwrap()` in `compiled.rs` — `alias` provably present
  (populated from the same `resolved_outputs`).
- `background=True` `SendError` panic reported in the proposal: **not in this
  codebase** (no `SendError`/`send()`/`spawn` in `polars-cv/src`); it originates
  in the Polars background executor and cannot be fixed here.
