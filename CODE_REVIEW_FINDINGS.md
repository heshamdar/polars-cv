# Code Review Findings

Structural / architecture review of `polars-cv`, `view-buffer`, and the Python
planning + metrics + geometry layers. This file is the **working ledger** for
resolving the findings progressively — it is meant to be edited as items close,
not left as a static report.

## How to use this file

- Each finding has a stable **ID** (`CR-NN`), a **severity**, a **location**, an
  **evidence** note, a **proposed fix**, and a **status**.
- When you resolve one: set status to `Resolved`, add the commit/PR that did it,
  and — if it was a *deferred design gap* rather than a plain fix — mirror it into
  `polars-cv/tests/test_known_gaps.py` as an `xfail(strict=True)` first, per the
  repo's "make the backlog executable" convention (see root `AGENTS.md`).
- Do not delete resolved entries; strike them through and keep the record.

**Status legend:** `Open` · `In progress` · `Resolved` · `Won't fix (documented)`

**Baseline at review time:** debug build green — structural lane 704 passed /
1 xfailed, fast lane 3650 passed / 1 xfailed (exit 0). All findings below are
*additions* to a green suite, i.e. gaps in coverage, not existing failures.

---

## High

### CR-01 — `ViewExpr::grayscale()` hardcodes `DType::U8`, producing silent wrong values · `Resolved`

> **Resolved** on `claude/codebase-architecture-review-jpagtw`. `apply_op`'s
> `ViewDto::Image` arm is now the single metadata authority; the seven image
> builders (`grayscale`/`threshold`/`resize`/`blur`/`erode`/`dilate`/
> `morph_gradient`) are thin wrappers over it, so both hardcoded `DType::U8`
> sites are gone. Guarded by `image_ops_track_the_dtype_their_contract_declares`
> (now covers `Grayscale`/`Threshold`/`Blur`/`Resize`),
> `typed_builders_agree_with_apply_op`, `grayscale_f32_fused_chain_matches_unfused`
> (all in `view-buffer/src/expr.rs`), and
> `test_dtype_contracts.py::TestGrayscaleFusionDtypeRegression`. All three Rust
> tests were watched failing (`fused 254.92882 != unfused 0.92882353`) before the
> fix. Follow-up for the remaining parallel surface tracked as CR-27.

- **Location:** `view-buffer/src/expr.rs:517` (`grayscale()` builder); reachable
  via `apply_op` (`expr.rs:122`) which the plugin's compiled executor uses
  (`polars-cv/src/graph/compiled.rs:809-817`).
- **What's wrong:** `grayscale()` stamps the tracked dtype as `U8` regardless of
  input, but `Grayscale`'s declared contract is `OutputDTypeRule::PreserveInput`
  (`view-buffer/src/ops/image.rs`) and the runner genuinely preserves input
  dtype. So for non-u8 input the tracked dtype (`U8`) diverges from the executed
  and *published* dtype (preserved). The published schema stays correct (it reads
  the contract via FFI), so **nothing catches the divergence** — it is a silent
  execution bug.
- **Evidence (empirically confirmed):** on normalized f32 data,
  `grayscale → invert` (unfused) yields `0.4480` (correct `1 − gray`), while
  `grayscale → invert → scale` (fused) yields `254.4480` (wrong `255 − gray`),
  off by exactly 254. Mechanism: `optimize()` keeps the grayscale node's `U8`;
  when it feeds a two-compute fusion that `U8` becomes the fused block's
  `inner_input_dtype` (`expr.rs:651`) and `invert` picks `max_val = 255.0`
  instead of `1.0` (`expr.rs:868`). Bites any non-u8 (f32/u16) chain where
  `grayscale` precedes ≥2 fusible compute ops; normalized-f32 ML preprocessing is
  the realistic trigger.
- **Root cause:** a *parallel construction surface*. `apply_op` already has a
  canonical inline block (`expr.rs:142-165`) that derives shape/strides/dtype
  from the op contract — its own comment records that `Canny`/`HistogramEqualize`
  were folded into it to kill this exact "second copy of the dtype rule" bug. The
  consolidation was left incomplete: `Threshold`/`Resize`/`Blur`/`Grayscale`
  still delegate to hand-written builders, and `grayscale`'s hand-computed dtype
  is wrong. (`threshold()` at `expr.rs:502` also hardcodes `U8`, but that happens
  to agree with its `Fixed(U8)` contract — benign, still a second copy.)
- **Proposed fix (P0 — see subplan below):** finish the consolidation. Fold the
  four remaining image ops into the canonical `apply_op` block and reduce the
  four builders to thin `apply_op` wrappers, so `apply_op` is the *single*
  metadata authority and no builder can set dtype/strides independently. Extend
  the guard `image_ops_track_the_dtype_their_contract_declares` to cover
  `Grayscale`/`Threshold` and to exercise the builder methods, and add a
  fusion-execution regression pinning the empirical case above.

---

## Medium

### CR-02 — The "known gaps" ledger drifted from its own prose · `Open`

- **Location:** root `AGENTS.md` §"What Is Left" vs `polars-cv/tests/test_known_gaps.py`.
- **What's wrong:** `AGENTS.md` states the open gaps (shear/rotate_and_scale
  auto-sizing; f64 fusion) are each "pinned executably in `test_known_gaps.py` —
  one `xfail(strict=True)` each." In reality that file holds **one** entry (the
  contour-scale default). The mechanism built specifically so a backlog "cannot
  silently become stale" has itself gone stale.
- **Proposed fix:** add the two missing `xfail(strict=True)` pins (they reference
  CR-03 and the f64-fusion known limitation), or correct the prose. The repo's
  philosophy says pin them.

### CR-03 — `shear`/`rotate_and_scale` advertise unimplemented auto-sizing · `Open`

- **Location:** `polars-cv/python/polars_cv/pipeline.py:3265` (`shear`), `:3311`
  (`rotate_and_scale`); TODOs at `:3302`, `:3361`.
- **What's wrong:** `output_size`/`center` default to `None` then immediately
  raise "auto-... not yet implemented" — the exact "accepted and ignored /
  documented as not-yet-implemented" pattern `CLAUDE.md` forbids.
- **Proposed fix:** make the params **required** (drop `| None`), or implement the
  auto-computation now that the planner tracks input shape. Pair with CR-02.

### CR-04 — `OutputDTypeRule::Configurable` is declared but never emitted · `Open`

- **Location:** `view-buffer/src/core/dtype.rs:161`; docs at
  `view-buffer/src/ops/compute.rs:70`, `expr.rs:373`, `runner.rs:158`;
  `Normalize` returns `Fixed(*out_dtype)` at `compute.rs:276`.
- **What's wrong:** no op returns `Configurable`; the variant survives only in
  match arms + its own test. `Normalize`'s doc claims it uses `Configurable(F32)`
  but the code returns `Fixed`.
- **Proposed fix:** delete the variant (and its arms/test), or wire `Normalize`
  to actually emit it; fix the three doc comments either way.

### CR-05 — Metrics helpers documented as load-bearing are orphaned · `Open`

- **Location:** `metrics/_auc.py` `partial_auc` (`:72`), `mcclish_correction`
  (`:13`), `_interp` (`:173`); `metrics/_result.py` `interpolate` (`:103`),
  `summary_table` (`:130`).
- **What's wrong:** `partial_auc`/`mcclish_correction`/`_interp` are reachable
  only via `MetricResult.auc(x_range=...)`, which nothing calls with a range;
  `interpolate`/`summary_table` have no internal caller. `metrics/AGENTS.md:21,40`
  still describes them as used.
- **Proposed fix:** delete the orphans (per "deleting is part of the work") and
  correct `metrics/AGENTS.md`.

### CR-06 — Two all-points-AP implementations kept in sync by hand · `Open`

- **Location:** `metrics/_metrics/_precision_recall.py` scalar `_all_points_ap`
  (`:330`) vs vectorized `all_points_ap_by_group` (`:404`).
- **What's wrong:** same estimator, two code paths; the second's docstring admits
  identity. `average_precision`/mAP use one, bootstrap the other.
- **Proposed fix:** collapse to one authority (express the scalar path in terms of
  the grouped one, or vice-versa).

### CR-07 — `mean_average_precision` runs an eager Python loop · `Open`

- **Location:** `metrics/_metrics/_precision_recall.py:218`.
- **What's wrong:** nested Python `for` over IoU × class, each iteration a full
  eager `average_precision().collect()` — a left-behind eager path now that the
  grouped lazy authority (`all_points_ap_by_group`) exists.
- **Proposed fix:** vectorize onto the grouped authority.

### CR-08 — `_rotation_matrix` reintroduces rotation trig in Python · `Open`

- **Location:** `polars-cv/python/polars_cv/pipeline.py:65`.
- **What's wrong:** the rotate→affine matrix has a Rust authority
  (`AffineParams::from_rotation` via the `rotate_affine_params` FFI), but the FFI
  exposes no `center`/`scale`, so Python recomputes the matrix for
  `shear`/`rotate_and_scale`. Escapes the recompute guard only because that guard
  targets the fusion helper.
- **Proposed fix:** extend `rotate_affine_params` to accept `center`+`scale` so
  Python stops doing trig, or document a sanctioned exception + add a guard.

### CR-09 — Duplicate integral families (eager vs expression) · `Open`

- **Location:** `metrics/_auc.py` (eager) vs `metrics/_auc_expr.py` (expression).
- **What's wrong:** the `trapz` split is justified (eager PR path), but the eager
  `partial_auc`/`mcclish_correction` are dead (CR-05), so those copies are pure
  redundancy alongside `partial_auc_expr`/`_mcclish_correction_expr`.
- **Proposed fix:** remove the dead eager copies with CR-05; keep only the
  expression versions.

### CR-10 — PNG-factory guard enforces a subset and is being evaded · `Open`

- **Location:** `polars-cv/tests/test_sanitation.py` (`test_no_local_png_factories`,
  `_CONFTEST_PNG_FACTORIES`); offender `polars-cv/tests/test_typed_nodes.py:33`
  (also `test_statistical_reductions.py`, which uses the shared `encode_png`).
- **What's wrong:** the guard bans local redefinition of `create_test_png` /
  `encode_png` but not the sibling conftest fixture `sample_image_bytes`, which
  `test_typed_nodes.py` redefines with a direct-PIL copy and **no**
  `except ImportError: pytest.skip` — the exact "errors instead of skips without
  Pillow" harm the guard exists to prevent.
- **Proposed fix:** broaden the guard to any local PNG-building fixture (or add
  `sample_image_bytes` to the protected set); fix/remove the offending local def.

### CR-11 — `cloud.rs` / `cloud_auth.rs` have no non-network CI coverage · `Open`

- **Location:** `polars-cv/src/cloud.rs` (937 lines), `cloud_auth.rs` (616);
  tests only under `pytest.mark.network` (`test_http_sources.py`).
- **What's wrong:** backend selection (`s3://`/`gs://`/`az://`), `cloud_options`
  parsing, and bounded-concurrency reads are never exercised in CI.
- **Proposed fix:** add table-driven URL→backend / options-parse unit tests (Rust)
  or a mock `object_store` backend so the logic runs in the default lane.

### CR-12 — Structural parity sweep self-skips without the `.so` · `Open (mitigated)`

- **Location:** `polars-cv/tests/test_sanitation.py` (~10 `pytest.skip("_lib.X not
  built")` sites) + backstop `test_lib_introspection_api_is_present`.
- **What's wrong:** a CI lane that never builds the plugin turns the whole parity
  sweep into silent skips; safety rests entirely on the one backstop test.
- **Proposed fix:** assert the plugin is present in the lane that is supposed to
  run these (fail, don't skip, when the `.so` is expected).

---

## Low

- **CR-13** — `TypedBufferData::polars_dtype()` (`polars-cv/src/graph/types.rs:304`)
  re-enumerates the `DType`→Polars mapping that `polars_dtype_for`
  (`decode.rs:613`) owns; delegate via `polars_dtype_for(self.dtype())`. · `Open`
- **CR-14** — `GraphNode.alias` (`polars-cv/src/graph/types.rs:390`) is a dead
  `#[allow(dead_code)]` field whose doc comment ("becomes the key in outputs map")
  is false; fix the comment or drop the field (keep only for wire-closure). · `Open`
- **CR-15** — Stale TODO pointer at `polars-cv/src/graph/compiled.rs:642` to an
  already-shipped path sandbox; remove. · `Open`
- **CR-16** — view-buffer dead code: `FusedKernel` helpers
  (`new/push/len/is_empty/describe/op_names`, `ops/scalar.rs`), `ScalarOp::name`,
  `ViewExpr::explain`+`explain_impl`+`node_type_name` (`expr.rs:689`),
  `ViewDto::validate_input_domain` (`ops/dto.rs:89`), and `ExprNode::Compute`'s
  always-length-1 `Vec<Arc<ViewExpr>>` (`expr.rs:25`). Delete. · `Open`
- **CR-17** — `source()` `BLOB` and `AUTO` branches are byte-identical
  (`pipeline.py:1532`); collapse. · `Open`
- **CR-18** — `contours.py:376` `label_reduce(heatmap=)` back-compat alias with no
  caller; delete. · `Open`
- **CR-19** — `metrics/_matching/_contour.py:502` `match(score_col=)` accepted and
  ignored (Matcher-protocol conformance); document at the site or restructure the
  protocol. · `Open`
- **CR-20** — `geometry/schemas.py` factory helpers
  (`validate_point`/`validate_contour`/`contour_from_points`/`bbox_from_*`) used
  only by tests; move to a test helper or re-export intentionally. · `Open`
- **CR-21** — `metrics/_metrics/_confusion.py` + `f1_at_threshold` issue several
  separate `.collect()`s instead of sharing the upstream subplan. · `Open`
- **CR-22** — ~6 test files use bare `pytest.raises(Exception)` without `match=`
  (e.g. `test_correctness_audit.py:981` uses `(ValueError, Exception)`); tighten. · `Open`
- **CR-23** — Stale docstring at `polars-cv/tests/test_typed_nodes.py:8` ("marked
  xfail until implementation complete" — no xfails exist; the seamless-pipeline
  feature landed and `TestSeamlessPipeline` runs live). · `Open`
- **CR-24** — `optimize()` transpose-merge carries a self-admitted "prototype …
  slightly inaccurate" comment (`view-buffer/src/expr.rs` ~600); derive from first
  principles or narrow the comment. · `Open`
- **CR-25** — Redundant `_ => None` catch-all after `Invert` in `working_dtype`
  (`view-buffer/src/ops/compute.rs:255`) lets a new `ComputeOp` inherit `None`
  silently; make it exhaustive. · `Open`

---

## Architectural follow-up (spun out of CR-01)

### CR-27 — Extend the single-metadata-authority collapse to `Compute` and `View` builders · `Open`

CR-01's fix makes `apply_op` the sole metadata authority for **image** ops. The
`Compute` (cast/scale/normalize/clamp/…) and `View` (transpose/reshape/crop/flip)
builders remain a parallel surface. They are **not currently buggy** (only
`grayscale` diverged), and `View` builders carry genuinely special semantics
(e.g. `reshape`'s non-contiguous panic), so this is deferred rather than bundled
into P0. The end state: every builder is a thin `apply_op` wrapper, and `apply_op`
is the one place that stamps shape/strides/dtype from op contracts.

---

## P0 subplan — CR-01 (executing now)

Goal: remove the parallel image-op construction path so dtype/strides tracking has
a single authority, then fix follows for free. TDD order:

1. **Red — extend the guard.** In `view-buffer/src/expr.rs`, add `Grayscale` and
   `Threshold` to `image_ops_track_the_dtype_their_contract_declares`'s `kinds`,
   and add a sub-check that drives the builder methods (`.grayscale()`,
   `.threshold()`, `.resize()`, `.blur()`) and asserts their tracked dtype equals
   the op contract's `resolve_output_dtype`. Watch it fail on `Grayscale`.
2. **Red — fusion-execution regression.** Add a test that builds
   `f32 source → grayscale → invert → scale` and asserts executed values equal the
   unfused / `1 − gray` reference (fails today at `255 − gray`).
3. **Green — collapse the path.** In `apply_op`, move `Threshold`/`Resize`/`Blur`/
   `Grayscale` out of the delegating arms into the canonical inline block (derives
   shape via `infer_shape`, strides via `calc_strides`, dtype via
   `resolve_output_dtype`). Rewrite the four builders as thin wrappers:
   `self.apply_op(ViewDto::Image(ImageOp { kind: … }))`. This deletes both
   hardcoded `DType::U8` sites.
4. **Verify.** `cargo test -p view-buffer`, then `maturin develop` (debug) and the
   Python regression mirroring the empirical case; then the structural + fast
   lanes stay green.
5. **Record.** Mark CR-01 `Resolved` with the commit; open CR-27 as the tracked
   follow-up.
