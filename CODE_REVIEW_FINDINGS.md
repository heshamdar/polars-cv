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

**Progress (as of the 0.26.0 pre-release review):** CR-01/02/03/05/07/13/14/15/16/25
were closed in the first review pass; the batched follow-ups then closed
CR-04 (Batch C), CR-24/CR-28 (Batch D1), CR-27 (Batch D2), CR-08 (Batch E),
CR-21 (Batch F1), CR-30 — and with it CR-06's divergence — (Batch F2),
CR-10/CR-12/CR-22 (Batch B), and CR-17/18/19/20/23 (Batch A). The only
substantive item still **Open** is **CR-11** (no non-network CI coverage for
`cloud.rs`/`cloud_auth.rs`); **CR-09** stays *Won't fix (premise corrected)*.
The 2026-09-23 performance & streaming review opened **CR-31–CR-38** (see
that section); CR-31 is a silent wrong-answer bug and should go first.
The 2026-09-24 quality review opened **CR-41–CR-44** (P0: soundness, strict
input handling, dev-loop and dependency metadata); all four are resolved.
The typed-op-protocol work is tracked as **CR-45–CR-49** (see `TYPED_OPS_PLAN.md`).

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

### CR-02 — The "known gaps" ledger drifted from its own prose · `Resolved`

- **Location:** root `AGENTS.md` §"What Is Left" vs `polars-cv/tests/test_known_gaps.py`.
- **What's wrong:** `AGENTS.md` states the open gaps (shear/rotate_and_scale
  auto-sizing; f64 fusion) are each "pinned executably in `test_known_gaps.py` —
  one `xfail(strict=True)` each." In reality that file holds **one** entry (the
  contour-scale default). The mechanism built specifically so a backlog "cannot
  silently become stale" has itself gone stale.
- **Resolution:** the AGENTS.md prose was corrected rather than back-filled with
  pins. `test_known_gaps.py` is for verified *defects* (wrong behaviour); neither
  referenced item is one — shear/rotate auto-sizing is a deliberate design
  decision (CR-03), and f64-fusion is a perf limitation already tracked in
  **Known Issues**. The false "pinned … one xfail each" claim is removed and each
  item is described where it actually lives (this ledger / Known Issues), so the
  defect file is not misused for non-defects.

### CR-03 — `shear`/`rotate_and_scale` advertise unimplemented auto-sizing · `Resolved`

- **Location:** `polars-cv/python/polars_cv/pipeline.py` (`shear`, `rotate_and_scale`).
- **What's wrong:** `output_size`/`center` defaulted to `None` then immediately
  raised "auto-... not yet implemented" — the "documented as not-yet-implemented"
  pattern `CLAUDE.md` forbids, on a signature that advertised the params as
  optional.
- **Resolution:** made them **required keyword-only arguments** (dropped `| None`,
  the manual `None`-checks, and the "not yet implemented" docstrings). This is the
  principled end state, not a placeholder: `output_size` is *structural* (it sets
  the output H/W that the plan-time schema publishes), and an image source's H/W
  is unknown at plan time, so auto-sizing cannot yield a plan-time shape for the
  common case — a "sometimes required, sometimes not" default would violate
  "explicit over implicit". The signature now enforces the requirement the three
  `test_affine_builder.py` tests already asserted (updated from `ValueError` to
  the signature-level `TypeError`). Lazy stub regenerated; parity guards green.

### CR-04 — `OutputDTypeRule::Configurable` is declared but never emitted · `Resolved`

> **Resolved** in Batch C (`5648e0e`). Deleted the `Configurable(DType)` variant,
> its `resolve()` arm and the `EVERY_RULE` test entry (`dtype.rs`); the
> `config:<dtype>` wire arm (`lib.rs`); and the whole `out_dtype_override`
> plumbing — the `matches!(rule, Configurable(_))` branch in `output_dtype_for`,
> the `out_dtype_override()` helper, and the `Option<DType>` parameter threaded
> through `resolve`/`resolve_planned`/`resolve_output_dtype` (every caller passed
> `None`; `Configurable` was the only `Some`-producer, so it was exactly the
> "plumbed deep and discarded" anti-pattern). `Normalize` keeps its structural
> `out_dtype` folded into a `Fixed(out_dtype)` rule, so plan == execution is
> unchanged. The six stale doc sites were corrected and the fold-agreement test
> dropped its now-meaningless override dimension.

- **Location:** `view-buffer/src/core/dtype.rs:161`; docs at
  `view-buffer/src/ops/compute.rs:70`, `expr.rs:373`, `runner.rs:158`;
  `Normalize` returns `Fixed(*out_dtype)` at `compute.rs:276`.
- **What's wrong:** no op returns `Configurable`; the variant survives only in
  match arms + its own test. `Normalize`'s doc claims it uses `Configurable(F32)`
  but the code returns `Fixed`.
- **Proposed fix:** delete the variant (and its arms/test), or wire `Normalize`
  to actually emit it; fix the three doc comments either way.
- **Note (deferred from the dead-code batch):** this is *not* the low-risk
  deletion it first looked like — `Configurable` is live in the FFI contract, not
  just view-buffer. `polars-cv/src/lib.rs` serializes it as `"config:<dtype>"`
  and branches on `matches!(rule, R::Configurable(_))` in the planner-facing
  `op_output_dtype` path (`execute.rs` resolves it too). Removing it means
  changing the Rust↔Python wire contract and confirming nothing on the Python
  side reads `"config:"`, so it wants its own careful commit rather than riding
  with the mechanical deletions.

### CR-05 — Metrics helpers documented as load-bearing are orphaned · `Resolved (premise partly corrected)`

- **Location:** `metrics/_auc.py` `partial_auc` (`:72`), `mcclish_correction`
  (`:13`), `_interp` (`:173`); `metrics/_result.py` `interpolate` (`:103`),
  `summary_table` (`:130`).
- **Correction to the premise (verified 2026-09-07):** the `_auc.py` trio is
  **not** orphaned. `partial_auc`/`mcclish_correction`/`_interp` back the public
  `MetricResult.auc(x_col, y_col, x_range=..., correction=...)` partial-AUC /
  McClish feature and are imported and tested directly by
  `tests/test_metric_fixes.py` (`partial_auc` extrapolation-warning test,
  `mcclish_correction` correction test, integer-bound tests). They are live and
  stay. AGENTS.md's statement that `_auc.py` keeps them "for the eager PR-curve
  `MetricResult.auc`" is accurate.
- **What was actually dead:** only the two `_result.py` methods,
  `MetricResult.interpolate` and `MetricResult.summary_table`. Zero callers in
  `python/`, `tests/`, `docs/`, `examples/` — the FROC/LROC helpers use
  `_auc_expr.interpolate_curve_lazy` directly, and no test invoked the methods.
- **Resolution:** removed `MetricResult.interpolate` / `summary_table` (and the
  now-unused `interpolate_curve_lazy` import from `_result.py`). Guarded by
  `tests/test_removed_surfaces.py::test_metric_result_interpolate_and_summary_table_are_gone`.
  Dropped the two from the `docs/api/metrics.md` autodoc member list (which also
  listed a `partial_auc` member `MetricResult` never had). Corrected
  `metrics/AGENTS.md` lines 21-23, 40, 133. The `_auc.py` helpers were left in
  place per the corrected premise.

### CR-06 — Two all-points-AP implementations kept in sync by hand · `Resolved (divergence closed; kept as two agreeing functions)`

> **Update (CR-30, `3b30751`):** the tie divergence this entry documented as a
> known gap is closed — both estimators now adopt the sklearn/COCO tie
> convention and agree exactly on ties as well as distinct scores, and
> `TestAllPointsAPAuthority` was rewritten from pinning the divergence to pinning
> the agreement. The two functions are still separate (a scalar single-curve path
> and a grouped `.over(keys)` path) but no longer merely "kept in sync by hand":
> they now genuinely agree on every input, guarded by that test.


- **Location:** `metrics/_metrics/_precision_recall.py` scalar `_all_points_ap`
  (`:330`) vs vectorized `all_points_ap_by_group` (`:404`).
- **What's wrong:** same estimator, two code paths; `average_precision` /
  `PrecisionRecallResult.auc` use the scalar one, bootstrap + (now) mAP use the
  grouped one.
- **Finding (verified 2026-09-07):** the two are **bit-identical on any curve
  with distinct scores** — pinned directly by
  `test_precision_recall.py::TestAllPointsAPAuthority` (parametrized tie-free
  curves incl. single point, all-TP, all-FP) and end-to-end by the pre-existing
  `test_bootstrap_ci_lazy.py::test_pr_point_matches_average_precision`. They
  **diverge on exact score ties** (e.g. scores `[0.5,0.5,0.5,0.5]`, TP/FP mix:
  scalar 0.25 vs grouped 0.4166…). Root cause: all-points AP is
  tie-order-sensitive, and the two paths feed differently ordered frames to an
  unstable Polars sort — the tie-break is arbitrary in *both*; neither value is
  "more correct".
- **Decision:** kept as two functions. Collapsing `_all_points_ap` onto the
  grouped path would change `PrecisionRecallResult.auc("all_points")`'s public
  (already arbitrary) output on tied curves, violating the "do not change public
  numeric output" constraint. Per the review protocol ("if they don't agree,
  keep them separate and report why"), the divergence is now documented in
  `metrics/AGENTS.md` and pinned as a known gap
  (`TestAllPointsAPAuthority::test_tie_divergence_is_the_documented_known_gap`).
  A clean future collapse is tracked as **CR-30** (adopt a canonical tie
  convention first).

### CR-07 — `mean_average_precision` runs an eager Python loop · `Resolved`

- **Location:** `metrics/_metrics/_precision_recall.py` (`mean_average_precision`,
  now `_mean_average_precision_all_points`).
- **What's wrong:** nested Python `for` over IoU × class, each iteration a full
  eager `average_precision().collect()` — a left-behind eager path now that the
  grouped lazy authority (`all_points_ap_by_group`) exists.
- **Resolution:** the `interpolation="all_points"` path is now one lazy plan with
  a single final collect: detections are stacked across thresholds (re-thresholded
  through the canonical `DetectionTable.at_iou_threshold`, so the `is_tp` rule and
  its lowering warning are not re-implemented), joined to per-class `total_gts`,
  reduced by `all_points_ap_by_group` grouped on `(_iou_t, class_id)`, then
  averaged over the full `(threshold, class)` grid (so a class with no detections
  or zero GTs still averages in as `AP = 0`, exactly as the loop did). The
  `"11_point"` (VOC) method has **no** grouped form, so it keeps the eager
  per-cell loop (that branch also preserves the original `ValueError` for an
  unknown `interpolation`). Output is bit-identical for all tie-free inputs — the
  realistic and tested domain (see CR-06 for the exact-tie caveat, which the
  scalar loop was equally subject to). Pinned by new exact-value tests in
  `test_precision_recall.py::TestMeanAveragePrecision` (single/multi-threshold,
  multi-class, all interpolations, no-detection and zero-GT classes).

### CR-08 — `_rotation_matrix` reintroduces rotation trig in Python · `Resolved`

> **Resolved** in Batch E (`9fa830a`). Added
> `AffineParams::rotation_matrix_2d(angle_deg, cx, cy, scale)` (view-buffer) — the
> 2x3 rotation+scale matrix about an arbitrary centre, now the single authority;
> `from_rotation` builds on it (adding only the expand canvas + recentering), so
> the image-centre and arbitrary-centre paths share one formula. Exposed as the
> `rotation_matrix_2d` FFI (registered + in `_REQUIRED_LIB_HOOKS`), and
> `_rotation_matrix`'s all-literal path now reads the FFI instead of the trig. The
> `pl.Expr` branch stays in Python — the engine cannot evaluate an expression at
> plan time — and is the one guard-sanctioned copy, now asserted by
> `test_the_planner_does_not_recompute_the_rotation_matrix`. Building a *literal*
> `rotate_and_scale` therefore now touches the compiled plugin (a deliberate
> trade), so the matrix-building `test_affine_builder.py` cases carry
> `@plugin_required`.

- **Location:** `polars-cv/python/polars_cv/pipeline.py:65`.
- **What's wrong:** the rotate→affine matrix has a Rust authority
  (`AffineParams::from_rotation` via the `rotate_affine_params` FFI), but the FFI
  exposes no `center`/`scale`, so Python recomputes the matrix for
  `shear`/`rotate_and_scale`. Escapes the recompute guard only because that guard
  targets the fusion helper.
- **Proposed fix:** extend `rotate_affine_params` to accept `center`+`scale` so
  Python stops doing trig, or document a sanctioned exception + add a guard.

### CR-09 — Duplicate integral families (eager vs expression) · `Won't fix (premise corrected)`

- **Location:** `metrics/_auc.py` (eager) vs `metrics/_auc_expr.py` (expression).
- **Original claim:** the eager `partial_auc`/`mcclish_correction` are dead
  (CR-05), so those copies are redundant alongside
  `partial_auc_expr`/`_mcclish_correction_expr`.
- **Correction (verified 2026-09-07):** the premise does not hold. The eager
  `partial_auc`/`mcclish_correction`/`_interp` are **live** — they back the
  public `MetricResult.auc(x_range=..., correction=...)` partial-AUC / McClish
  feature and are imported and tested by `tests/test_metric_fixes.py` (see the
  CR-05 correction). The two families serve two execution models: `_auc.py`
  (eager, Series-based) for the eager PR-curve `MetricResult.auc`, and
  `_auc_expr.py` (expression) for the lazy FROC/LROC + bootstrap plans. Neither
  is redundant. No deletion; both kept.

### CR-10 — PNG-factory guard enforces a subset and is being evaded · `Resolved`

> **Resolved** in Batch B (`a093b44`). Added a sibling guard
> `test_no_unguarded_local_image_data_fixtures` that AST-scans each suite module
> and flags an override of a conftest image-*data* fixture whose body references
> `Image`/`PIL` directly (bypassing conftest's Pillow skip), while allowing an
> override that delegates to a guarded factory. The detector is watched both ways
> with inline known-bad/known-good fixtures. The offender
> (`test_typed_nodes.py::sample_image_bytes`) now delegates to `encode_png` and
> drops its module-scope PIL/BytesIO imports.

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

### CR-12 — Structural parity sweep self-skips without the `.so` · `Resolved`

> **Resolved** in Batch B (`a093b44`). Added
> `test_plugin_is_present_when_required` (not `@plugin_required`): when
> `POLARS_CV_REQUIRE_PLUGIN=1` it asserts `_lib` is importable, else it self-skips.
> The flag is set in `ci.yml`'s Build-and-Test step and in `scripts/verify.sh`
> (both run `maturin develop` first), so a lane that is supposed to have built the
> plugin now fails — rather than silently skipping the whole `@plugin_required`
> sweep — if the extension is missing.

- **Location:** `polars-cv/tests/test_sanitation.py` (~10 `pytest.skip("_lib.X not
  built")` sites) + backstop `test_lib_introspection_api_is_present`.
- **What's wrong:** a CI lane that never builds the plugin turns the whole parity
  sweep into silent skips; safety rests entirely on the one backstop test.
- **Proposed fix:** assert the plugin is present in the lane that is supposed to
  run these (fail, don't skip, when the `.so` is expected).

---

## Low

- **CR-13** — `TypedBufferData::polars_dtype()` (`polars-cv/src/graph/types.rs`)
  re-enumerated the `DType`→Polars mapping that `polars_dtype_for`
  (`decode.rs`) owns. · `Resolved` — now delegates via
  `decode::polars_dtype_for(self.dtype())`.
- **CR-14** — `GraphNode.alias` (`polars-cv/src/graph/types.rs`) carried a false
  doc comment ("becomes the key in outputs map"). · `Resolved` — the field is
  kept (Python emits `alias` per node and `GraphNode` is `deny_unknown_fields`,
  so it is required for wire-closure); the comment now states it is
  deserialized-but-unread, matching its `domain`/`output_dtype` siblings.
- **CR-15** — Stale TODO pointer at `polars-cv/src/graph/compiled.rs` to an
  already-shipped path sandbox. · `Resolved` — reworded to name the `PathPolicy`
  sandbox instead of a TODO.
- **CR-16** — view-buffer dead code. · `Resolved (partial)` — **deleted**
  `FusedKernel::describe`/`op_names`, `ScalarOp::name`,
  `ViewExpr::explain`/`explain_impl`/`node_type_name`, and
  `ViewDto::validate_input_domain` (all confirmed zero-caller). **Corrected:** the
  subagent listed `FusedKernel::new`/`push`/`len`/`is_empty` as dead, but they are
  the builder API used throughout `view-buffer/tests/` (`clamp_fusion.rs`,
  `fused_ops.rs`, …) — kept. **Deferred:** `ExprNode::Compute`'s always-length-1
  `Vec<Arc<ViewExpr>>` → `Arc<ViewExpr>` is a type change touching every
  construction/match site; left as its own follow-up (see CR-28).
- **CR-17** — `source()` `BLOB` and `AUTO` branches are byte-identical
  (`pipeline.py:1532`); collapse. · `Resolved` — Batch A (`3d987d7`): merged into
  one `elif fmt in (BLOB, AUTO)` arm, both comments preserved.
- **CR-18** — `contours.py:376` `label_reduce(heatmap=)` back-compat alias with no
  caller; delete. · `Resolved` — Batch A (`3d987d7`): alias deleted, `image` is the
  sole input; guarded by `test_label_reduce_has_no_heatmap_alias`.
- **CR-19** — `metrics/_matching/_contour.py:502` `match(score_col=)` accepted and
  ignored (Matcher-protocol conformance); document at the site or restructure the
  protocol. · `Resolved` — Batch A (`3d987d7`): documented at the site (kept for
  `Matcher`-protocol conformance; `BBoxMatcher` genuinely reads it, contour scores
  come from heatmap peaks).
- **CR-20** — `geometry/schemas.py` factory helpers
  (`validate_point`/`validate_contour`/`contour_from_points`/`bbox_from_*`) used
  only by tests; move to a test helper or re-export intentionally. · `Resolved` —
  Batch A (`3d987d7`): the genuinely test-only `bbox_from_corners`/`bbox_from_center`
  moved to `tests/_geometry_helpers.py`; `validate_point` (production caller) and
  `contour_from_points` (docs/tests) kept, the latter marked a public helper.
- **CR-21** — `metrics/_metrics/_confusion.py` + `f1_at_threshold` issue several
  separate `.collect()`s instead of sharing the upstream subplan. · `Resolved` —
  Batch F part 1 (`f912cfd`): `f1_at_threshold` routes through
  `confusion_at_threshold` (one tp/fp/fn pass); precision/recall/f1 fall out
  arithmetically with the same edge cases. Behaviour unchanged.
- **CR-22** — ~6 test files use bare `pytest.raises(Exception)` without `match=`
  (e.g. `test_correctness_audit.py:981` uses `(ValueError, Exception)`); tighten. ·
  `Resolved` — Batch B (`a093b44`): narrowed 27 sites across 12 files to the actual
  exception type, adding `match=` where a stable message exists; dropped the
  now-unnecessary PT011/B017 noqas.
- **CR-23** — Stale docstring at `polars-cv/tests/test_typed_nodes.py:8` ("marked
  xfail until implementation complete" — no xfails exist; the seamless-pipeline
  feature landed and `TestSeamlessPipeline` runs live). · `Resolved` — Batch A
  (`3d987d7`): docstring rewritten.
- **CR-24** — `optimize()` transpose-merge carries a self-admitted "prototype …
  slightly inaccurate" comment (`view-buffer/src/expr.rs` ~600); derive from first
  principles or narrow the comment. · `Resolved` — Batch D part 1 (`865428c`):
  the hand-built merged node is replaced with `grandchild.transpose(merged)`, so
  shape/strides are recomputed from the grandchild's real layout through the
  canonical builder; the hedge is gone.
- **CR-25** — Redundant `_ => None` catch-all after `Invert` in `working_dtype`
  (`view-buffer/src/ops/compute.rs`) let a new `ComputeOp` inherit `None`
  silently. · `Resolved` — replaced with explicit
  `Cast`/`Affine`/`RotateAffine`/`Fused` arms (behaviour-preserving), so the
  match is exhaustive and a new variant must declare its working dtype.
- **CR-28** — `ExprNode::Compute` holds `Vec<Arc<ViewExpr>>` but every site uses
  exactly one child (`view-buffer/src/expr.rs`); narrow to `Arc<ViewExpr>`. Spun
  out of CR-16 as a standalone type change. · `Resolved` — Batch D part 1
  (`865428c`): narrowed to `Compute(ComputeOp, Arc<ViewExpr>)`, which makes the
  `.len() == 1` guards in `optimize`/`build_plan` unrepresentable and deletes them.
- **CR-30** — All-points AP has no canonical tie convention, so its two
  implementations (scalar `_all_points_ap`, grouped `all_points_ap_by_group` in
  `metrics/_metrics/_precision_recall.py`) can return different (both arbitrary)
  values on exact score ties, which is why CR-06 keeps them separate. Adopt a
  deterministic secondary sort key (e.g. `is_tp` descending — TP before FP at
  equal score, the COCO/sklearn-friendly convention) in *both* paths, verify it
  leaves every existing pin unchanged (all use distinct scores) and does not move
  the bootstrap CI pins, then collapse the two onto one authority. Spun out of
  CR-06. · `Resolved` — Batch F part 2 (`3b30751`): both estimators now collapse
  each tied-score block to one PR point at the cumulative counts *after* the block
  (the sklearn/COCO convention), making the AP order-independent and the two paths
  agree exactly on ties as well as distinct scores. This changes public AP output
  on tied-score inputs (documented in `CHANGELOG.md`); `TestAllPointsAPAuthority`
  is rewritten from pinning the divergence to pinning the agreement. The two
  functions remain separate but now genuinely agree (the guard pins it).

---

## Performance & streaming review (2026-09-23)

A second pass aimed at the plugin's two headline promises — high performance and
correct behaviour under Polars' lazy/streaming engines — rather than contract
drift. Timings are from the **debug** build on a 4-core container, so only the
*ratios* are meaningful; each was measured through the user-facing
`df.lazy().select(...).collect(engine=...)` entry point.

### CR-31 — Expression params are identified by `str(expr)`, which is not an identity · `Resolved` · High

> **Resolved.** `_types.expr_key` is the single identity authority, read by
> `ParamValue.__eq__`/`__hash__`/`to_dict`, `Pipeline._track_expr`, and the
> graph's expression-slot and root-column deduplication. It keeps the display
> text as the key while unambiguous and disambiguates by `Expr.meta.eq`
> (`text#n`), interning via weak references. `meta.serialize` was rejected as
> the key because it raises for Python UDFs without `cloudpickle`. Guarded by
> `tests/test_expr_param_identity.py`: all eight collision tests were watched
> failing first (e.g. `40.0 == 60.0`, root column `20.0 == 30.0`).

- **Location:** `python/polars_cv/_graph.py` `_get_expr_columns` (and
  `_build_column_bindings` / `_get_ordered_columns` for root columns);
  `_types.py` `ParamValue.to_dict` keys the slot by the same `str(expr)`.
- **What's wrong:** two *different* expressions whose `repr` is equal are
  deduplicated into one plugin input slot, so the second op silently reads the
  first op's values. Polars' `str(expr)` is a display form and truncates — e.g.
  every `pl.lit(pl.Series("f", ...))` prints as `Series[f]`.
- **Evidence (empirical):** on a `[4,4,3]` image of 10s,
  `cast("f32").scale(pl.lit(s1)).scale(pl.lit(s2))` with `s1 = 1.0`, `s2 = 3.0`
  per row returns **10.0**; the correct answer is 30.0. No error, no warning.
- **Proposed fix:** identify an expression by `expr.meta.serialize()` (or a hash
  of it) — or simply assign slots positionally and never dedupe by text. Keep a
  readable name only for error messages. Guard with the repro above as a
  regression test, watched failing first.

### CR-32 — Multi-core execution depends on how Polars happens to chunk the input · `Resolved` · Low

> **Resolved (P1, 2026-09-24): the call is parallel.** The re-scoping below
> was reversed by the quality review: eager `with_columns` is the README's
> first example, so a single-core eager path is the default experience, not a
> corner. `CompiledGraph::execute` now splits a call's rows into contiguous
> ranges (a few per thread) that run on the plugin's own `THREAD_POOL`. A
> plugin links its own polars-core, so it cannot join the host's pool; the
> plugin's pool is sized by `POLARS_MAX_THREADS` and concurrent calls share
> it, since callers block while their rows run. Each range has its own
> `ParamCtx` and scratch; results are concatenated in order; under
> `on_error="raise"` the earliest failing range's error wins, so the report
> is the one a sequential run gives. The CR-37 plan cache is now shared across
> ranges and keyed by layout (up to 16 per segment), so a segment is still
> planned once per layout per call. Measured on 4 cores (debug build,
> before → after): 400 PNG rows resize+blur, eager 11.37 s → 2.94 s,
> streaming 2.90 s → 2.95 s; 200k cheap rows, eager 2.60 s → 0.83 s,
> streaming 0.74 s → 0.75 s; 8 rows at 1024², eager 3.60 s → 0.96 s,
> streaming 0.91 s → 0.96 s. Streaming neither gains nor loses. The engine
> warning, whose advice ("ran on one thread") became false, is deleted.
> Guards: `compiled.rs::a_call_runs_its_rows_on_several_threads` (watched
> failing: "256 rows ran on 1 thread(s)"); `tests/test_parallel_rows.py`
> (order, earliest error, null alignment on a single-chunk 300-row frame —
> watched failing against two deliberate mutations: reversed range order,
> last-range error wins); the unchanged plan-count assertions in
> `static_segments_plan_once_per_source_layout`; and
> `test_removed_surfaces.py::test_the_single_thread_engine_warning_is_gone`.

> **Previously resolved (re-scoped)** by improving the warning:
> `engine_warning.rs` now decided when each call finishes. It
> warns once if that call ran longer than `POLARS_CV_ENGINE_WARN_SECONDS`
> (default 2 s) and no other plugin call overlapped it. Overlap is tracked per
> call with a global counter of overlapping starts, so one overlap earlier in
> the process no longer silences the warning for good.
> `POLARS_CV_ENGINE_WARN_ROWS` is no longer read; setting it, or an unusable
> seconds value, prints a one-time notice. Streaming runs with a 50 ms
> threshold did not warn in three trials. Guarded by
> `tests/test_engine_warning.py` (subprocess cases, watched failing) and
> `test_removed_surfaces.py::test_the_engine_warning_reads_no_row_threshold`.

> **Re-scoped (2026-09-23).** The plugin's standard regime is lazy + streaming,
> and there the morsels already spread the work across cores (3.5–3.8× on 4
> cores in the table below). Parallelising inside a call would only help the
> in-memory engine and would compete with the streaming engine's own
> scheduling, so it is not planned unless a design emerges that helps both.
> What is left is that eager users get no reliable signal: the warning counts
> rows, and image rows are expensive. The warning should be based on the call's
> elapsed time rather than its row count.

- **Location:** `graph/compiled.rs` `execute_rows` (a sequential `for row_idx in
  0..len`); `engine_warning.rs`. No `rayon`/`POOL` use anywhere in either crate.
- **What's wrong:** the plugin never parallelises within a call, so it only uses
  more than one core if the engine happens to invoke it more than once at the
  same time. The in-memory engine does that per *chunk*, and the streaming
  engine per *morsel*. A single-chunk frame (anything rechunked, or read as one
  batch) on the default engine therefore runs on one core.
- **Evidence (empirical, resize 128 + blur, 64×64 PNGs):**
  | input | in-memory | streaming |
  |---|---|---|
  | 1024 rows, 1 chunk | 12.39 s | 3.30 s (3.8×) |
  | 256 rows, 8 chunks | 1.05 s | 0.86 s |
  | 256 rows, 1 chunk | 3.10 s | 0.95 s |

  The same data runs 3× faster or slower depending only on `n_chunks()`.
  The warning cannot catch this: its threshold is **50 000 rows per call**, but
  image rows cost milliseconds each, so the 12 s single-threaded run above never
  triggers it. Its "concurrency seen ⇒ user knows" suppression also fires as soon
  as two chunks or two `with_columns` expressions run together.
- **Proposed fix:** parallelise over row ranges inside `CompiledGraph::execute`
  on Polars' own thread pool (`polars_core::POOL`, so it nests correctly with
  the engine rather than oversubscribing). Give each worker its own
  `node_outputs`/scratch, and concatenate the per-range `RowResult`s in order.
  Row semantics are already per-row, so this is behaviour-preserving. Once it
  lands, delete the engine warning instead of tuning its threshold.

### CR-33 — `list` sinks of rank ≥ 2, and `array` sinks with any null row, build one `AnyValue` per element · `Resolved` · High (perf)

> **Resolved.** `graph/encode.rs` builds both sinks directly into Arrow: one
> flat primitive buffer (`flat_values`), then one `ListArray` offsets level or
> `FixedSizeListArray` level per dimension, with row nulls as the outer
> validity. The `AnyValue` builders, the typed `ListPrimitiveChunkedBuilder`
> family, `extract_as_*`, `slice_typed_data`, the flat "fast path" and the
> now-unused `TypedBufferData::polars_dtype` are deleted. A later row of the
> wrong dtype was silently cast, an `array` row of the wrong element count was
> accepted, and an unrecognised dtype string fell back to the first row's.
> All three are now errors. Guarded by `encode.rs::tensor_sink_tests` (the
> strictness cases were watched failing) and the slow-lane
> `tests/test_sink_cost_ratio.py`, which fails at 23–27× on the old code and
> bounds each sink at 5× the numpy sink. After the fix, with streaming: numpy
> 4.3 ms, rank-3 list 5.3 ms, array with a null row 3.7 ms. The
> `to_contiguous` + `to_vec` copy in `typed_list_of`/`typed_array_of` remains.

- **Location:** `graph/encode.rs`
  `build_typed_nested_list_series_from_rows_with_dtype` /
  `build_typed_nested_list_value`, and the fallback in
  `build_typed_array_series_from_rows_with_dtype` (taken whenever
  `try_build_array_series_flat` sees a single null row).
- **Evidence (empirical, 64 rows of 64×64×3 u8):** `sink("numpy")` 11.6 ms;
  `sink("array")` 10.1 ms; `sink("list")` **392.9 ms (≈34×)**; `sink("array")`
  with one null row **409.6 ms (≈40×)**. So one null (e.g. a single decode
  failure under `on_error="null"`) makes the whole column 40× slower.
- **Also:** the list/array paths copy each row three times: `to_contiguous()`,
  then `TypedBufferData::from_contiguous_buffer` (`to_vec`), then the
  concatenation into the flat builder.
- **Proposed fix:** build the Arrow arrays directly. For a list sink, one flat
  values buffer plus offsets per nesting level (the shapes are known per row).
  For an array sink, a `FixedSizeListArray` whose null rows are a validity bit
  over a zeroed slot. Write each row straight into the flat buffer so it is
  copied once. Delete the `AnyValue` fallbacks rather than keeping them as a
  "slow path".

### CR-34 — Panics are the engine's error channel, so `on_error` cannot cover them · `Resolved` · Medium

> **Remainder resolved.** Data-dependent failures are now errors, not panics.
>
> - **Every op states its requirements.** `Op::validate` is a required contract
>   method with no default, and every op implements it (the image, colour,
>   filter and view ops were added).
> - **The executor checks before running.** Buffer segments are built only
>   through `ViewExpr::try_apply_op`, which validates each op against the
>   tracked shape and dtype and refuses a reshape of a non-contiguous view.
>   Binary, `apply_mask` (`validate_mask`), `channel_merge`
>   (`validate_channel_merge`), reduction, histogram, phash and geometry steps
>   validate directly.
> - **Guard.** `tests/test_engine_no_panics.py` sweeps every buffer op, and
>   every two-operand op across all operand pairs, over 16 input
>   rank/channel/dtype cells. It found 49 panicking cases first; none remain.
>
> A before/after table of all 4,272 cells shows every change from "ok" to
> "error" is deliberate. Each one was a silent wrong result:
> - `reshape` to a different element count produced an out-of-bounds view (a
>   `[100, 200, 3]` shape over 86 bytes); `ViewBuffer::reshape` now asserts too.
> - Image ops given rank-1 or rank-4 input were a no-op or dropped an axis:
>   resize and colour conversion read `[2, 5, 4, 3]` as H=2, W=5, C=4.
> - A 3-entry `transpose` on rank 4.
>
> Found along the way:
> - `Crop`'s `infer_shape` subtracted the `usize::MAX` "to the end" sentinel,
>   so the tracked channel count was `usize::MAX`. It now resolves to the axis
>   length, and the output rank follows the input.
> - `op_infer_shape` reports an unknown input axis carried through unchanged
>   as `-1` by an explicit rule, probing unknown dims with values distinct from
>   parameter probes. Before, crop reached `-1` only by that arithmetic accident.
> - Stale `validate` rules that nothing had enforced were brought into line
>   with the kernels: minmax/z-score accept any shape (four tests pinned the
>   old rule and were rewritten), and phash no longer imposes an undocumented
>   16–1024 `hash_size` range.
> - Caught panics are labelled "internal error: the engine panicked".

> **Row policy gap closed.** `CompiledGraph::execute_rows` wraps each row in
> `catch_unwind`, so an engine panic is that row's error and `on_error` applies
> to it. The batch-level catch remains only as a backstop for series building.
> Guarded by `tests/test_on_error.py::TestEnginePanicsFollowTheRowPolicy`
> (operands that cannot broadcast, in-memory and streaming), watched failing
> first with "Pipeline batch failed". **Still open:** the engine still reports
> data errors by panicking. Each one prints a panic message to stderr, and
> `panic = "unwind"` stays load-bearing. Converting `runner.rs`/`buffer.rs` to
> return `Result` remains the end state.

- **Location:** `view-buffer` has roughly 80 `panic!`/`unwrap`/`expect`/`assert!`
  sites outside tests (29 in `execution/runner.rs`, 19 in `core/buffer.rs`). The
  only catch is the batch-level `catch_unwind` in `CompiledGraph::execute`.
- **What's wrong:** a panic on one row fails the whole call regardless of
  `on_error="null"`, which is documented in `src/AGENTS.md` but not visible to
  users. Under streaming the call is a morsel, so how much of the query a bad
  row takes down depends on the morsel size. It also forces `panic = "unwind"`
  onto the release profile.
- **Proposed fix:** convert the op implementations to return `Result` (start with
  `runner.rs`). As an interim, run `catch_unwind` per row so a panic becomes a
  row error that the row policy then handles.

### CR-35 — Published wheels ship without the SIMD code paths · `Resolved` · Low

> **Resolved.** The separable blur, the one kernel that gains from AVX2, now
> dispatches once per call to an AVX2 build of its whole body
> (`is_x86_feature_detected!` + `#[target_feature(enable = "avx2")]`).
> Release timings with the wheels' baseline flags: 1024² u8 5.02 → 3.84 ms,
> 512²×3 u8 3.81 → 3.07 ms, on par with an `x86-64-v3` build. Two approaches
> measured *worse* first, which is why the whole body is duplicated:
> - dispatching only the row axpy was slower (5.51 ms);
> - leaving the passes inside the `thread_local!` `with` closure was no faster,
>   because a closure does not inherit its caller's target features.
>
> Only `avx2` is enabled, not `fma`, so the output is bit-identical on every
> CPU (`blur_dispatch_is_bit_identical`, run against a baseline build).

> **Measured (release, view-buffer kernels, baseline flags vs
> `-C target-cpu=x86-64-v3`):**
>
> | kernel | baseline | v3 |
> |---|---|---|
> | grayscale u8 1024×1024×3 | 1.06 ms | 1.27 ms |
> | threshold u8 1024×1024 | 0.08 ms | 0.10 ms |
> | fused invert·scale·clamp f32 | 3.15 ms | 3.24 ms |
> | cast u8→f32 | 0.73 ms | 0.74 ms |
> | blur σ=2 u8 1024×1024 | 5.16 ms | 3.58 ms |
> | resize → 224×224 (`fast_image_resize`) | 1.16 ms | 1.33 ms |
>
> Only the separable blur gains (1.44×). The other kernels already
> auto-vectorise well at the SSE2 baseline, and `fast_image_resize` dispatches
> at runtime. The config comment's claim is corrected here rather than acted
> on. If blur-heavy workloads matter, a `#[target_feature(enable = "avx2,fma")]`
> clone of the blur inner loop behind `is_x86_feature_detected!` is the
> contained fix.

- **Location:** `.cargo/config.toml` enables `x86-64-v3` for dev builds only;
  `ci.yml`, `publish.yml` and `benchmark.yml` all clear it with `RUSTFLAGS=""`.
- **What's wrong:** the AVX2/FMA auto-vectorisation that the config's comment
  credits for `grayscale_u8`, `threshold_simd`, `FusedKernel` and the blur loops
  is only present in local builds. Users get baseline SSE2 wheels, and the
  regression benchmarks also measure baseline, so nothing reports the gap.
- **Proposed fix:** dispatch at runtime in the hot kernels
  (`is_x86_feature_detected!` + `#[target_feature(enable = "avx2,fma")]`, or the
  `multiversion` crate). The alternative is to publish a v3 wheel variant, as
  Polars does with `polars` / `polars-lts-cpu`.

### CR-36 — Geometry I/O goes point by point through `AnyValue` · `Resolved` · Low (measured)

> **Resolved.** `geom_schema::contour_array` builds a contour column straight
> into Arrow: flat `x`/`y` buffers plus offsets for the exteriors, holes and
> hole rings, with each field matched by name to `contour_fields()`. It backs
> both outputs:
> - the graph's contour sink (`encode.rs::contour_set_series`). This deleted
>   `encode.rs::contour_struct_dtype`, a second declaration of the layout.
> - every `.contour` accessor. `map_contours*`/`zip_contours` are now generic
>   over `ContourOutput`: measures keep the `AnyValue` path, and transforms
>   return a `Contour`.
>
> The declared dtype still governs the result. With streaming on 64 masks
> (~90k points): contour sink overhead ~31 → ~3 ms, and `.contour.translate`
> 39 → 6.2 ms. `contour_to_anyvalue` is now test-only, as the oracle for
> `encode.rs::contour_sink_tests` and `geom_arity.rs::contour_output_tests`.
> Follow-up: the graph sink published an *empty* contour set as null, while
> the `.contour` transforms kept `[]`. It now publishes `[]`, and null means
> only a null input or a failed row (`tests/test_contour_empty_set.py`,
> watched failing).

> **Measured (debug, streaming, 64 masks of 256², ~1,400 points/row):**
> `extract_contours → area` 47 ms; the same extraction with contours as the
> output 79 ms, so the `AnyValue` *encode* costs ~30 ms per ~90k points.
> The *decode* half is not a hotspot. Contour source → rasterize (219 ms) is
> no slower than extract → rasterize without any parse (242 ms), because
> rasterizing dominates. `.contour.area()` on the column takes 5 ms.
> Downgraded; only the encode is worth rewriting.

- **Location:** `src/contour.rs` (89 `AnyValue` uses: `parse_contour_list` →
  `series.get(i)` per contour and per point), contour encoding in
  `graph/encode.rs`, and the `contour` source / `LabelReduce` reads in
  `graph/compiled.rs` (`input_series.get(row_idx)`, `get_any`).
- **What's wrong:** this is the same pattern as CR-33, applied to the
  `List[Struct{x, y}]` geometry columns. Every point becomes an enum value, and
  every row goes through a sub-`Series` allocation.
- **Proposed fix:** read the `ListArray` offsets and the struct's `x`/`y`
  `Float64` buffers directly, and write outputs the same way (offsets plus flat
  coordinate buffers). Before changing anything, add a benchmark in
  `benchmarks/regression/` to confirm the cost.

### CR-37 — Per-row executor overhead from stringly-typed dispatch · `Resolved` · Low

> **Remainder resolved.**
>
> - **Plan cache.** A run of static (all-literal) buffer ops is planned once
>   per source dtype/shape/strides per call, and its `PlanStep`s are replayed
>   on later rows (`CachedPlan`, `run_segment`). Those three facts are the
>   only things planning reads from the source. Segments with a per-row
>   parameter are planned every row. Pending ops are borrowed and cloned only
>   on a miss. Guarded by `static_segments_plan_once_per_source_layout`
>   (watched failing: 4 plans vs 2) and `dynamic_segments_are_not_cached`.
>   A 4-op chain went from 2.62 to ~1.85 µs/row (debug).
> - **Source dispatch.** `SourceFormat` is parsed at compile time, and `auto`
>   is resolved per batch to the enum. `decode_source` became
>   `decode_image_bytes`, which takes no format, so `file_path` and `auto` rows
>   no longer clone their `SourceSpec` to overwrite a string. Guarded by
>   `source_format_names_match_the_vocabulary`, watched failing.

> **Id-keyed maps removed from the row loop.** Profiling showed `SipHash` over
> node-id `String`s at about a third of the executor's per-row instructions.
> `CompiledGraph` now compiles a `plan: Vec<NodePlan>` in topological order.
> Each entry holds the node's column binding, upstream position, source spec,
> `on_error`, cloud options, path policy and op resolvers. `node_outputs`,
> prefetched batches, auto-format resolutions and per-output results are
> indexed by position. Only cross-node operand reads still look a name up in
> `node_index`. No-op rows went from 1.16 to 0.72 µs (debug, 100k 8×8 rows).
> **Still open:** the static op chain is re-optimised and re-planned
> (`ViewExpr::plan_with`) on every row, which is most of the ~1 µs a short op
> chain adds. The source format is still dispatched by string comparison.

> **Measured after CR-40 (debug, 100k rows of 8×8 u8 `array`, streaming):**
> no ops 1.2 µs/row, `invert` 2.2 µs/row, a three-op fused chain 2.9 µs/row.
> Native `arr.eval(255 - x)` takes 15 ns/row. The remaining per-row cost is
> this finding. Also, the `blob`/`raw` "zero-copy" sources copy each row
> (`get_binary_row_buffer`: `bytes.to_vec()`). That cost is linear, not
> quadratic, but it contradicts the name. A true view into a `BinaryView`
> data buffer needs an alignment check, because rows sit at arbitrary byte
> offsets.

- **Location:** `graph/compiled.rs` `run_row_nodes`.
- **What's wrong:** every row, for every node:
  - `node_outputs: HashMap<String, _>` is populated with `node_id.clone()`;
  - the source format is dispatched by comparing strings (`"contour"`,
    `"file_path"`, ...);
  - `node.source.clone()` runs for `file_path` and `"auto"` sources;
  - the static op chain is re-optimised and re-planned (`ViewExpr::plan_with`),
    even when nothing in it is per-row.

  Decode cost hides all of this for encoded images. It matters for
  `blob`/`raw`/`list` sources feeding cheap ops, which is the plugin's zero-copy
  fast path.
- **Proposed fix:** at compile time, resolve node ids to indices
  (`Vec<Option<NodeOutput>>`) and the source format to an enum. Cache the
  planned `ExecutionPlan` step list per (static chain, input dtype, rank).

### CR-38 — Row/sink kind mismatch silently becomes a null row · `Resolved` · Low

> **Resolved.** Each sink kind in `build_series_from_spec` now lists the row
> variants it accepts (`convert_rows`). `None` of an accepted variant is a null
> row, and any other variant is an internal error (`foreign_row`). Guarded by
> `sink_kind.rs::a_row_of_another_kind_is_an_error_not_a_null`, which covers
> every `(domain, format)` pair and was watched failing on `(buffer, numpy)`.

- **Location:** `graph/decode.rs` `build_series_from_spec`. Every arm maps
  unexpected `RowResult` variants with `_ => None`.
- **What's wrong:** if `encode_node_output` and `SinkKind` ever disagree, the row
  is published as null rather than raising an error. That is the "degrade, not
  fail" pattern `CLAUDE.md` rejects.
- **Proposed fix:** return an internal error on a variant mismatch. Better still,
  make `RowResult` generic over, or indexed by, `SinkKind` so the mismatch
  cannot be constructed at all.

### CR-39 — A null row in the numpy/ndarray sink is a struct of nulls, not a null · `Resolved` · Low

> **Resolved.** `build_numpy_series` sets the outer struct validity as well as
> the field nulls, so a null row of `numpy`/`torch`/`ndarray` is a null value.
> `numpy_from_struct(None)` raises a clear `ValueError`. Guarded by
> `tests/test_numpy_sink_nulls.py` (10 cases, watched failing). 36 existing
> assertions of the form `row["data"] is None` pinned the removed
> representation. They were rewritten to `row is None`, which is strictly
> stronger, as was `test_on_error.py::test_numpy_sink_with_on_error`.

- **Location:** `src/output.rs` `build_numpy_series`. The validity bitmap is
  applied to the struct's *fields*.
- **What's wrong:** every other sink publishes a null row as a null value. A
  numpy/ndarray null row is `{data: null, dtype: null, …}`, so
  `is_null()` is `False` and `drop_nulls()`/`null_count()` do not see it.
  `tests/test_ndarray_sink.py::_is_null_row` accepts both representations, so
  nothing pins either one. Found while writing the CR-38 guard.
- **Proposed fix:** set the outer struct validity as well. This is a
  user-visible behaviour change (`is_null()` starts returning `True`), so it
  needs a CHANGELOG entry and a decision on whether the field-level nulls stay.

### CR-40 — The "zero-copy" `array` source copied the whole column on every row · `Resolved` · High (perf)

- **Location:** `src/graph/decode.rs` `get_primitive_buffer`, reached through
  `try_decode_array_zero_copy` for every `source("array")` row.
- **What was wrong:** to take one row's window it built a new `Buffer` from
  `values.as_slice().to_vec()`, which is the chunk's **entire** values buffer,
  once per row. A batch was therefore quadratic: 35 µs per 64-byte row at 100k
  rows. Found by profiling (callgrind: 99% of plugin time in `to_vec` under
  `get_primitive_buffer`) while measuring CR-37.
- **Fix:** `Buffer<T>::try_transmute::<u8>()` reinterprets the values buffer in
  place, so each row is a view into the column. Sharing is safe because
  view-buffer only writes in place through uniquely owned `Rust` storage.
  1.2 µs/row after the fix (~30×).
- **Guard:** `decode.rs::array_source_view_tests` asserts that each decoded
  row's data pointer lies inside the column's own values buffer, for plain,
  sliced, multi-chunk and nested f32 columns. It was watched failing on the
  pointer check first.

---

## Quality review, P0 (2026-09-24)

An independent review against a Polars-plugin quality bar. The P0 items below
are the small, urgent ones: an unsoundness, silent acceptance of bad input, a
dev-loop footgun and dependency metadata. Architectural findings from the same
review (typed op enum, Rust-side planner, symbolic shapes, intra-call
parallelism) are tracked for later phases, not here.

### CR-41 — A misaligned blob reaches `slice::from_raw_parts` in release builds · `Resolved` · Critical

> **Resolved.** `view_buffer::parse_blob` (`protocol.rs`) is now the one blob
> parser: bounds, overflow, stride reach and **alignment** (offset and every
> stride a multiple of the element size). The plugin's zero-copy decode and
> `ViewBuffer::from_blob` both read through it, so the second parser — which
> also never checked stored strides, and copied only the logical bytes of a
> strided blob while keeping its strides (an out-of-bounds read) — is gone.
> Binary rows are copied into `Vec<u64>`-backed storage so a blob's first byte
> is 8-aligned by construction, the decode re-checks the absolute address, and
> `ViewBuffer::as_ptr`'s alignment check is an `assert!` in every build.
> Guards: `tests/test_blob_protocol.py` (user entry point, watched failing as
> "the engine panicked: … not aligned") and five `decode.rs` unit tests, four
> watched failing against the old code. `binary_rows_are_copied_to_an_aligned_address`
> could not be watched failing — the old `Vec<u8>` copy was aligned by the
> allocator's choice, which is exactly what it no longer relies on.

- **Location:** `polars-cv/src/graph/decode.rs` `decode_blob_zero_copy`;
  `view-buffer/src/core/buffer.rs` `as_ptr` / `as_slice`.
- **What's wrong:** the blob header's `data_offset` and stored strides are
  bounds-checked but never checked for **alignment**. `as_slice` is a *safe*
  function whose only alignment check is a `debug_assert!` inside `as_ptr`, so
  in the release wheels a user-supplied blob with `data_offset` not a multiple
  of the element size builds a misaligned `&[f32]` — undefined behaviour
  reachable from column data.
- **Evidence:** shifting a valid f32 blob's `data_offset` by 1 makes the debug
  build fail with `engine panicked: ViewBuffer pointer is not aligned for type
  f32`; the release build compiles that assert out.
- **Proposed fix:** reject a misaligned offset/stride at decode with a proper
  error (the `raw` source path too), and make the alignment check in
  `as_ptr`/`as_slice` unconditional so no other constructor can reintroduce it.

### CR-42 — `crop` and the `raw` source silently accept out-of-range input · `Resolved` · High

> **Resolved.** The `crop` arm rejects a negative bound (so a literal fails
> while the pipeline is built) and resolves `height`/`width` independently —
> the review also found that giving only one of them discarded it.
> `ViewOp::Crop::validate` rejects a window outside the input, so it is a row
> error under `on_error`. `raw` rejects a byte length that is not a multiple of
> the element size. Guard: `tests/test_strict_input_bounds.py` (11 cases watched
> failing). Two existing tests used overrunning crops as fixtures and were
> updated: `test_offset_crop_with_full_extent_is_not_eliminated` now asserts
> the offset crop errors under every flag subset (deleting it would turn the
> error into a success), and `test_a_continuation_carries_its_own_parameter`
> uses in-bounds heights. `examples/02_image_transforms.py` cropped rows
> 14..90 of a 72-row image (it silently got 58 rows); its crop now fits. That
> surfaced only in the slow lane (`test_examples_run`), after the P0 commit,
> because only the fast lane was run for it.

- **Location:** `polars-cv/src/execute.rs` `"crop"` arm;
  `polars-cv/src/graph/decode.rs` `decode_binary_zero_copy` (`"raw"`).
- **What's wrong:**
  - `crop` clamps a negative `top`/`left` to 0 but keeps the height, so
    `top=-5, height=10` returns rows 0–9 — a *shifted* window. A window that
    overruns the image is silently shrunk (`top=15, height=10` on a 20-row
    image returns 5 rows). The comment claims NumPy/OpenCV conventions; it
    matches neither.
  - `raw` computes `len / element_size`, so 10 bytes read as f32 become 2
    elements and 2 bytes are dropped.
- **Proposed fix:** both raise. Negative offsets/extents and windows outside
  the image are errors (a row error, so `on_error` applies); a byte length that
  is not a multiple of the element size is an error.

### CR-43 — `uv run` builds the release-LTO extension · `Resolved` · Medium

> **Resolved.** `[tool.uv] package = false`: uv treats the project as virtual,
> so no `uv` command builds it; `maturin develop` remains the one build.
> Verified: `uv run python …` no longer starts a build and imports the
> `maturin develop` extension. Guard:
> `test_build_efficiency.py::test_uv_never_builds_the_project` (watched failing
> on the old `pyproject.toml`).

- **Location:** `polars-cv/pyproject.toml`; documented commands in `CLAUDE.md`.
- **What's wrong:** after `uv sync --no-install-project` + `maturin develop`,
  `uv run <anything>` re-syncs the project and builds it through the maturin
  backend at `[profile.release]` (fat LTO) — the exact trap `CLAUDE.md` warns
  about, reached by its own documented `uv run pytest` command. Observed while
  reviewing.
- **Proposed fix:** make uv never build/install the project
  (`[tool.uv] package = false`), so `maturin develop` stays the only extension
  build, and guard it in `test_build_efficiency.py`.

### CR-44 — Declared dependencies do not describe what the package needs · `Resolved` · Medium

> **Resolved.** Floors measured by running the fast suite against released
> versions: polars 1.30/1.34/1.35.2 fail at import (no extension-type API),
> 1.36.1–1.40.1 fail tests (a polars-internal streaming panic in the metrics
> paths; the engine-warning timing), 1.41.1, 1.41.2 and 1.44.2 are fully
> green. numpy 2.0.2 is green; 1.26 runs the package but not its reference
> tests (their scipy/imagehash oracles need numpy 2). Now `polars>=1.41.1,<2.0`,
> `numpy>=2.0.2`; `networkx`/`graphviz`/`pydot` are the `viz` extra;
> `pyo3/abi3-py310` matches `requires-python>=3.10`. The `dependency-floors`
> CI job installs `scripts/dependency_floors.py`'s output (read from
> `[project] dependencies`) on Python 3.10 — verified locally on 3.10 at those
> floors (4211 passed). Guards: `tests/test_dependency_metadata.py` (floor
> parsing with fixtures, viz extra, abi3 parity — watched failing on
> `abi3-py39` — and a subprocess run with the viz libraries blocked).

- **Location:** `polars-cv/pyproject.toml` `[project] dependencies`.
- **What's wrong:** `polars>=1.0,<2.0` although the package uses `Expr.ext`
  (recent polars) and is only tested against the locked version;
  `networkx`/`graphviz`/`pydot` are hard dependencies of an optional,
  lazily-imported visualisation feature; `numpy>=2.2.6` excludes NumPy 1.x
  without a stated reason; `requires-python>=3.10` disagrees with the
  `abi3-py39` wheel tag.
- **Proposed fix:** set the polars floor to the oldest version that actually
  works, move the visualisation libraries to a `viz` extra with a clear
  ImportError, justify or lower the numpy floor, and align the abi3 tag with
  `requires-python`.

---

## Typed op protocol (2026-09-24)

The quality review's architectural findings (#6–#9: string-typed op protocol,
planner split across the FFI, probe-based shape inference, creation-order
expression keys), planned in [`TYPED_OPS_PLAN.md`](TYPED_OPS_PLAN.md). That file
carries the phase-by-phase work, the transition discipline and the deletion
matrix; the entries here track status only.

### CR-45 — Ops cross the boundary as a name plus an untyped param map · `Open` · Medium (design)

- **Location:** `polars-cv/src/pipeline.rs` (`OpSpec`), `execute.rs`
  (`KNOWN_OPS`, `resolve_op_inner`), `params.rs` (`OpParams`), `pipeline.py`
  (`OP_NAMES`, hand-written builders), 19 hand-written enum mirrors.
- **What's wrong:** nothing structural ties Python and Rust together, so
  agreement is kept by registries, parity tests, source scans and a runtime
  read-tracker — each a second copy of a fact.
- **Fix:** one `define_op!` definition per op (typed `Param<T>` / `Literal<T>`
  fields, serde-enforced), a generated Python builder, and deletion of every
  check the types make structural. Plan phases P1–P3, P6.
- **Progress:** P0 (safety net) done — golden corpus, signature snapshot,
  removed-symbol gate, `tests/_plan_view.py` seam, baselines in
  `benchmarks/reports/2026-09-24-typed-ops-baseline/`, and the never-read
  `GraphNode` fields `alias`/`domain`/`output_dtype` deleted.

### CR-46 — The planner is split across the FFI and folded twice · `Open` · Medium (design)

- **Location:** `pipeline.py` planner state and `_append_op`/`_push_op`/`_update_*`;
  `lib.rs` `op_schema`/`op_contract`/`op_infer_shape`/`op_output_channels`/
  `op_identity_rule`; `graph/compiled.rs` `fold_output_rank`/`fold_output_dtype`.
- **Fix:** a Rust `Plan` pyclass owns the fold; Python becomes a thin recorder.
  Plan phase P7.

### CR-47 — Source/sink params are policed by applicability tables · `Open` · Low (design)

- **Location:** `_types.py` `SOURCE_PARAM_APPLIES`/`SINK_PARAM_APPLIES`;
  `pipeline.rs` `SourceSpec`/`SinkSpec` (`format: String`).
- **Fix:** tagged enums per format. Plan phase P4.

### CR-48 — Geometry accessors carry a second per-row parameter mechanism · `Open` · Low (design)

- **Location:** `geom_params.rs` (`InputSlots` by name), `contour.rs`/`point.rs`
  kwargs, `_namespace.py` `_ArgBinder`.
- **Fix:** the same `Param<T>` + positional slots. Plan phase P5.

### CR-49 — Plan-time shapes are inferred by probing four magic values · `Open` · Low (design)

- **Location:** `lib.rs` `op_infer_shape` (probes 7, 13, 90, 180),
  `unknown_dim_probe`, `PRESERVED_DIM`, `ParamCtx::probe`.
- **Fix:** symbolic `Dim` in a required `Op::infer_dims`. Plan phase P9.

---

## Architectural follow-up (spun out of CR-01)

### CR-27 — Extend the single-metadata-authority collapse to `Compute` and `View` builders · `Resolved`

> **Resolved** in Batch D part 2 (`bb1f184`). Added two private authorities,
> `compute_node` and `view_node`, that build a Compute/View node purely from the
> op's contract (shape via `infer_shape`, strides via `calc_strides`, dtype via
> `resolve_output_dtype`), and routed the builders through them: affine, scale,
> relu, fused, normalize, clamp, adjust_contrast, adjust_gamma, invert (and
> `apply_op`'s `RotateAffine` arm) → `compute_node`; transpose, crop, flip,
> channel_select → `view_node`. Behaviour is identical. `cast` keeps its own
> builder (a same-dtype cast passes strides through untouched) and `reshape` keeps
> its own (its non-contiguous-input panic), both documented at the helpers. No
> builder can now stamp shape/strides/dtype independently of an op contract.

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
