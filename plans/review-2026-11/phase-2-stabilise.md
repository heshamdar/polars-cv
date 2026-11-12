# Phase 2 — Stabilise and smooth

Implementation plan for the second group of findings from the external review
of 0.41.0 (commit `c0d77d5`, 2026-11-11). Ledger entry CR-141 in
the root [`ISSUES.md`](../../ISSUES.md). Follows
[phase 1](phase-1-first-contact.md); [phase 3](phase-3-depth.md) follows this.

**Goal:** an API a team can build on, and the same care at the edges of a
pipeline (sinks, messages, docs) that the ops already have.

**Status:** CR-145, CR-143, CR-142 and CR-144 are closed (`ISSUES.md`, Closed
findings, and the CHANGELOG). What is left is CR-141, a decision.

What the implementation found that this plan had wrong, for whoever reads
phase 3 with the same eyes:

- **CR-145.** The plan trimmed `image` in `view-buffer/Cargo.toml` only.
  `polars-cv/Cargo.toml` declared `image` too, with defaults, and features
  unify, so the plugin build would have kept every codec. The entry is now
  one `[workspace.dependencies]` line. `tga` was compiled and unreachable as
  well (no signature to detect). `rayon` was measured and kept: one EXR
  decode is about 50% slower without it. Its weak feature
  (`ravif?/threading`) keeps the uncompiled `ravif` tree in `Cargo.lock`,
  re-resolved to versions without `core2` and `paste`, so both `deny.toml`
  ignores went.
- **CR-143.** "Prefix the message with the output alias, which the graph
  already knows" holds only for a multi-output graph. The name polars gives
  a single result (`.alias()`, a `select` keyword) is applied outside the
  plugin. The message names the column decoded and the sink instead.
- **CR-142.** `pl.col(...).cv` is registered at run time, so a type checker
  does not see `sink()`'s keywords through it; it does wherever the
  `LazyPipelineExpr` type is known. A misspelled keyword is now Python's
  `TypeError`, where it was the plugin's `ValueError`.

---

## CR-141 — A stability policy  **decision**

**Re-verified: the count stands. This is the owner's call, not a code change.**

### Evidence

31 minor releases from 0.10.0 (2026-06-13) to 0.41.0 (2026-10-23).
`docs/user-guide/` holds 13 migration guides, 0.29 to 0.41, the first dated
2026-09-27: thirteen breaking minors in 26 days. The README tells users to
pin a minor version.

The churn bought real things (the typed catalogue, the closed wire, the Rust
`Plan`), and those are what make a stable surface cheap to keep: a removed
name is refused by generated signatures and `deny_unknown_fields`, so
accidental breakage is already mechanically hard. What is missing is a
statement of what will not change on purpose.

### Options

1. **Declare a stable core now.** `Pipeline`, `.cv.pipe`, `.sink`, op names
   and their required parameters, sink and source names, the output schemas.
   Breaking changes to it are batched into at most one release a quarter and
   announced one release ahead with a `DeprecationWarning`. Everything else
   (`metrics`, extension types, optimisation flags, private wire) stays free
   to move.
2. **Hold until phases 1 and 2 land, then declare.** The known signature
   changes were CR-136 (wire only), CR-138 (additive), CR-137 (additive) and
   CR-142 (`sink()`). All four have landed; what remains of phases 1 and 2
   (CR-139, CR-140) changes no signature. So this option now reads: declare
   with the next release.
3. **Stay as is until 1.0.**

Recommendation: option 2. It costs one more breaking release and avoids
promising a `sink()` signature that CR-142 is about to replace.

### If accepted

- A "Stability" page in the docs listing the stable surface, generated from
  the catalogues where it can be (op names, source/sink names) so the list
  cannot drift from the code.
- A guard: a committed snapshot of the stable surface (names and required
  parameters) that fails when it shrinks without a matching entry in a
  deprecations table. Watch it fail by removing an op name.
- A deprecation path that does not exist today: the generated builder emits
  a `DeprecationWarning` for a name marked deprecated in the catalogue
  (`#[op(deprecated = "...")]`), for one release, before the wire refuses it.
  This is the one place a tombstone is justified, and it is generated, not
  hand-written. (CR-142's `(format, {params})` tuple was removed outright
  rather than deprecated, on the owner's word, so nothing has used a
  deprecation path yet.)
- README "Project status" rewritten to say what is stable.

---
