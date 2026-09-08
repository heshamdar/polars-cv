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
