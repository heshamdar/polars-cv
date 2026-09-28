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
The typed-op-protocol work is tracked as **CR-45–CR-49**, all resolved in 0.29.0.
The 2026-09-27 kernel performance work (`PERFORMANCE_PLAN.md`) is tracked from
**CR-50** on.

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
  `tests/test_removed_surfaces.py::test_metric_result_interpolate_and_summary_table_are_gone`
  (since deleted with the other removal tombstones; the generated signatures
  refuse the names).
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
> `test_removed_surfaces.py::test_the_single_thread_engine_warning_is_gone`
> (since deleted with the other removal tombstones).

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
> `test_removed_surfaces.py::test_the_engine_warning_reads_no_row_threshold`
> — all deleted with the warning itself (see the resolution above).

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
expression keys). The work shipped in 0.29.0 (see its CHANGELOG entry); the
entries here track status only.

### CR-45 — Ops cross the boundary as a name plus an untyped param map · `Fixed` · Medium (design)

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
  `GraphNode` fields `alias`/`domain`/`output_dtype` deleted. P1 (positional
  slots) done — expression params are `{"$slot": n}` from a graph-wide
  `SlotTable` (`Expr.meta.eq` identity); `expr_key`, `expr_column_names` and
  the Rust name binding are deleted. P2 (catalogue + spike) done — `crop`,
  `resize`, `warp_affine`, `histogram` are `#[derive(Op)]` structs with
  generated Python builders; `LEGACY_OPS` holds the remaining 81. P3 (every
  op typed) done — all 85 ops are typed, `LEGACY_OPS` is empty, and the
  name-keyed resolution (`resolve_op_inner`, `OpParams`, the `get::*` readers,
  the arm-scan guards) is deleted.
- **Resolution (P6):** `TypedOp` is the wire op; `LegacyOpSpec`, `LEGACY_OPS`,
  the dispatcher, the untyped `ParamValue`, `known_ops` and `OP_NAMES` are
  deleted, and the Python enums are generated from the Rust registries
  (`enum_catalog`), taking their parity tests and the `enum_variants` FFI with
  them. The Python planner's own `OpSpec`/`ParamValue` go with CR-46 (P7).

### CR-46 — The planner is split across the FFI and folded twice · `Fixed` · Medium (design)

> **Fixed** (0.29.0): the plan is Rust's — `plan_step`/`plan_source`/
> `plan_assert`/`plan_sink` return an immutable `PlanState`, `node_pass` runs
> the node-scope passes, each output carries its `planned` state, and shapes
> are symbolic (`OpShape`). The FFIs listed below and the Python folds are
> deleted; `fold_output_rank`/`fold_output_dtype` became one `fold_lineage`
> (the column-resolved rank/dtype of a list/array source, known only at
> execution). A later consolidation replaced the Python op list and `_replay`
> with the Rust `Plan` pyclass (`src/plan.rs`), which owns a pipeline's ops.

- **Location:** `pipeline.py` planner state and `_append_op`/`_push_op`/`_update_*`;
  `lib.rs` `op_schema`/`op_contract`/`op_infer_shape`/`op_output_channels`/
  `op_identity_rule`; `graph/compiled.rs` `fold_output_rank`/`fold_output_dtype`.
- **Fix:** a Rust `Plan` pyclass owns the fold; Python becomes a thin recorder.
  Plan phase P7.

### CR-47 — Source/sink params are policed by applicability tables · `Fixed` · Low (design)

- **Location:** `_types.py` `SOURCE_PARAM_APPLIES`/`SINK_PARAM_APPLIES`;
  `pipeline.rs` `SourceSpec`/`SinkSpec` (`format: String`).
- **Fix:** tagged enums per format. Plan phase P4.
- **Resolution (P4):** every source and sink format is a typed struct in
  `polars-cv/src/formats/` (`formats!` registry, `io_catalog.json`); the
  builder validates through the same deserializer (`io_check`), and both
  applicability tables, their checker and hint table are deleted.

### CR-48 — Geometry accessors carry a second per-row parameter mechanism · `Fixed` · Low (design)

- **Location:** `geom_params.rs` (`InputSlots` by name), `contour.rs`/`point.rs`
  kwargs, `_namespace.py` `_ArgBinder`.
- **Fix:** the same `Param<T>` + positional slots. Plan phase P5.
- **Resolution (P5):** `ContourKwargs`/`PointKwargs` are `#[derive(Op)]`
  structs of `Param<T>` and `ColumnRef` fields; `_ArgBinder` writes
  `{"$slot": n}` into each expression kwarg, and `GeomParams` checks the
  derived slots against the inputs. The name map and its readers are deleted.

### CR-49 — Plan-time shapes are inferred by probing four magic values · `Fixed` · Low (design)

> **Fixed** (0.29.0): `Op::shape() -> OpShape` is the one shape
> authority, evaluated symbolically by the planner from each typed op's
> `OpDef::shape`; the probe, `unknown_dim_probe` and `PRESERVED_DIM` are
> deleted. The probe had been planning unknown images as square (see the
> CHANGELOG's Fixed entry). `ParamCtx::probe` survives as
> `ParamCtx::planning`, one placeholder for the value-independent rules.

- **Location:** `lib.rs` `op_infer_shape` (probes 7, 13, 90, 180),
  `unknown_dim_probe`, `PRESERVED_DIM`, `ParamCtx::probe`.
- **Fix:** symbolic `Dim` in a required `Op::infer_dims`. Plan phase P9.

### CR-50 — Graph node ids are random, so equal pipelines never share a compiled graph · `Open` · Low (performance)

- **Location:** `lazy.py` `_generate_node_id` (`node_{uuid4}`), `_graph.py`
  CSE `shared_id` (`_cse_{uuid4}`).
- **What's wrong:** the graph JSON is the compiled-graph cache key, and every
  node id in it is random per construction. Two identical pipelines built
  separately (e.g. once per loop iteration or per request) serialize
  differently, so each compiles afresh and occupies its own cache slot. Found
  while pinning the positional-slot JSON determinism (`test_positional_slots.py` normalizes
  the ids to test the expression encoding alone).
- **Fix:** derive node ids from content (the node's canonical spec and its
  upstream ids) rather than `uuid4`; aliases stay user-facing names. Still
  open after 0.29.0: node ids are still `uuid4` (`lazy.py` `_generate_node_id`, `_graph.py` CSE `shared_id`).

---

## Performance review (2026-09-27)

Kernel-level follow-up to CR-31–40, planned in `PERFORMANCE_PLAN.md` and measured
by `view-buffer/benches/kernels.rs` (baseline:
`polars-cv/benchmarks/reports/2026-09-27-kernel-baseline/`). One entry per
phase of that plan, closed as each lands.

### CR-54 — Wheels run most kernels without SIMD; float → int casts call `roundf` per element · `Resolved` · Medium (perf)

> Filed as a second CR-50 in `bdce7ee`, which duplicated the existing CR-50
> above; renumbered to the next free id.

- **What was wrong:** the wheels target x86-64 (SSE2). There, `f32::round` is a
  libcall per element, so every float → int cast (`cast`, a fused chain's
  integer output) ran 2.1–2.2x slower than an AVX2 build. The u8 grayscale
  loop pushed into a `Vec` and did not vectorise at all, on any target, and
  the u8 threshold allocated a `Vec` per row for strided input. CR-35's
  runtime dispatch covered blur only, as a hand-rolled pair of functions.
- **Resolution:**
  - `core::dispatch` (`SimdKernel` + `dispatch()`) is now the one way a kernel
    gets an AVX2 build. Blur moved onto it.
  - `core::convert` (`CastFrom` + `convert_slice`) is the one element-conversion
    rule. `cast_to` and `finish_fused_output` both use it.
  - Grayscale and threshold are dispatched kernels over `ViewBuffer::dense_rows`,
    so contiguous, cropped and vertically flipped inputs are read where they lie.
- **Guards:**
  - In every debug build, `dispatch` asserts the two builds' outputs are
    byte-identical (`the_parity_check_rejects_outputs_that_differ`).
  - `convert::tests` was watched failing against a truncating rule.
  - `grayscale_threshold_parity_tests`, including an exhaustive `luma_u8`
    check, was watched failing against `+127` rounding, a nudged coefficient
    and `>=`. Its first input pattern could not reach a rounding boundary and
    was replaced.
  - The view-buffer suite passes under `RUSTFLAGS="-C target-cpu=x86-64"`.
- **Measured:** `polars-cv/benchmarks/reports/2026-09-27-phase1-dispatch/`. On the
  wheel target, u8 grayscale is 6.7–10x faster, u8 threshold 1.8–3.2x, and f32 → u8
  casts 1.8–2.1x. A first candidate zero-filled its outputs and lost ~15% on a 1024²
  threshold; it now writes into spare capacity (`map_pixel_rows`).

### CR-51 — Non-u8 gray + alpha grayscale mixed alpha into the intensity · `Resolved` · Medium

- **What was wrong:** `grayscale_typed` read a `[H, W, 2]` pixel as
  (gray, alpha, alpha) and returned `0.299·gray + 0.701·alpha`. The u8 kernel
  (and the `SingleChannel` contract) take the gray channel. A u16 gray + alpha
  pixel of (1000, 65535) came out 46239.
- **Resolution:** a two-channel pixel's grayscale is its gray channel for
  every dtype.
- **Guard:** `gray_alpha_grayscale_is_the_gray_channel_for_every_dtype`,
  watched failing on the old code.

### CR-52 — Per-value ops ran through five near-duplicate paths, each with a full f32 copy · `Resolved` · High (perf)

- **What was wrong:**
  - The scalar family, scale/relu/clamp, invert, gamma, contrast, normalize and
    fused chains each had their own function. Most cast a u8 image to f32 (one
    image-sized copy) and then mapped into a second buffer.
  - u8 gamma computed a `powf` per pixel and u8 preset normalize a divide per
    pixel: ~50 ms at 1024²×3.
  - Only f32 → f32 ever ran in place.
- **Resolution:** `view-buffer/src/ops/elementwise/` is the one engine.
  - Every op lowers to a `FusedKernel` through `lower_to_scalars`, which moved
    out of `expr.rs` so fusion and standalone execution share it. Statistics
    come first for normalize/contrast.
  - `strategy` picks how the kernel runs:
    - integer arithmetic when the kernel is exactly `clamp(±x + c)` over
      8/16-bit input into the same dtype (`invert`, integer shifts), as fast
      as the op written natively;
    - a lookup table, only for work that cannot vectorise (`powf`) or that
      differs per channel, over 8-bit input (16-bit with enough values);
    - blocked streaming through an L1 f32 scratch for an integer result;
    - one f32 pass for a float result.

    Two candidates were measured and rejected on the way. Tabling every 8-bit
    kernel made u8 `invert` 18x slower than `255 - x` (a table read vs a vector
    subtract). Streaming it through f32 blocks was still 10x slower. Hence the
    rule by cost, and the integer strategy.
  - The float → 8/16-bit integer conversion (`convert::CastFrom`) was
    rewritten to vectorise, with identical results, pinned by
    `narrow_conversion_equals_round_then_saturate`. `x.round() as u8` stayed
    scalar even in an AVX2 build (saturating `as` + no x86 round-half-away).
  - `ViewBuffer::unique_contiguous_mut` decides in place. The u8 threshold uses
    it too.
  - Eleven functions were deleted, along with
    `ViewBuffer::try_apply_fused_kernel_inplace`.
- **Deliberate change:** z-score statistics are exact (integer sums for 8/16-bit,
  f64 otherwise). They had been sequential f32 sums.
- **Guards:**
  - `elementwise::tests` compares every op × all 10 dtypes × {contiguous, crop,
    flip_v, flip_h, transpose} × {shared, sole-owned} with the pre-engine code,
    kept verbatim in `legacy.rs`, bit for bit (any NaN equals any NaN). It covers
    16-bit on both sides of the table threshold. It was watched failing against
    eight mutations: the i8 table index, the in-place table write, per-channel
    passes, the contrast mean, the blocked path's passes and store index, and
    the integer path's sign and clamp.
  - Four `copy_counts.rs` allocation cases (u8 invert / u8 → u8 chain / u8
    threshold in place, preset normalize with only its f32 output) failed on the
    old code first.
  - The suite also passes under `RUSTFLAGS="-C target-cpu=x86-64"`.

- **Measured** (`polars-cv/benchmarks/reports/2026-09-27-phase2-elementwise/`,
  against Phase 1, median of three interleaved rounds):
  - u8 gamma: 22–27x faster;
  - u8 preset normalize: 5–12x;
  - u8 `scale`: 1.3–9.5x;
  - u8 contrast: 2–5.5x;
  - u8 z-score: 2.2–4x;
  - f32 preset: 2.5–3.3x;
  - fused u8 chain: 1.8–2.5x;
  - u8 invert: 1.3–1.9x;
  - f32 → u8 cast: 1.6–2.3x.

  Every other kernel is within noise.

### CR-55 — Materialising a strided view copied a few bytes at a time · `Resolved` · High (perf)

- **What was wrong:**
  - `ViewBuffer::copy_elements_into` merged only the innermost axis. A u8 HWC
    flip or crop copied 3 bytes per `memcpy` call and recomputed the N-d offset
    for each: 5–7 ms to materialise a 1024² RGB flip, where one copy is 0.3 ms.
    Every `to_contiguous`, list/array sink and encode of a view paid it.
  - `gather_strided_f32` (the element-wise engine's read of a view)
    recomputed the offset per element.
  - `cast_to` on a view packed it first, then converted: two passes.
  - Grayscale kept its own per-pixel strided loop.
- **Resolution:** `view-buffer/src/core/strided.rs`, `Walk`, the one walk over
  a view's memory. It coalesces the layout once into packed units, evenly
  spaced rows and odometer-walked outer axes (size-1 axes dropped, evenly
  spaced outer axes merged).
  - `copy_to` packs, with constant-size unit copies. It is `copy_elements_into`.
  - `for_each_run` hands out runs: long units in place, short ones packed into
    an 8 KiB stack scratch. `convert::convert_view` converts a view in one
    dispatched pass from them, for `cast_to` and the engine's f32 read.
  - `gather_strided_f32` and grayscale's per-pixel fallback are deleted.
- **Guards** (`strided::tests`):
  - the coalescing fixture table;
  - a random-view property test (rank 1–4, permutes/flips/slices,
    u8/u16/f32/f64) against a per-element reference, over `copy_to`,
    `for_each_run`, `to_contiguous` and `cast_to`.

  Both were watched failing against three under-merging mutations (the fixture
  table) and two output-corrupting ones (the property test). A debug-build
  bound check on every walk turns a coalescing fault into a panic naming the
  geometry, instead of a wild read. The suite also passes on the wheels'
  x86-64 target.
- **Measured** (`polars-cv/benchmarks/reports/2026-09-28-phase3-strided/`,
  1024², wheel target):
  - flip_v materialise: 21.7x;
  - flip_h materialise: 5.2x;
  - transpose materialise: 5.5x;
  - grayscale of a flipped image: 4.7x;
  - crop then resize: 4.5x;
  - cast of a flipped view: 3.4x;
  - scale of a transposed f32 view: 2.8x.

  Up to 35x at 256². Everything else is within noise.
- **Tried and rejected:** a tiled transpose (`80d3b52`, reverted in `c4475b6`).
  It made u8 transpose 1.7x slower at 1024² and only helped 12-byte units at
  1024² (1.27x).
- **Open follow-up (not planned):** transpose is still ~8x a vertical flip at
  1024² (2.5 vs 0.3 ms). A kernel specialised for small units (an in-register
  3-/4-byte block transpose) is the remaining lever.

### CR-56 — Resize packed every crop and vertical flip into a new image first · `Resolved` · Medium (perf)

- **What was wrong:**
  - `ImageOp::Resize` (and every resize variant, and `Letterbox`) declared
    `MemoryEffect::RequiresContiguous`, so `build_plan` put a
    `MaterializeContiguous` in front of it whenever its input was a view.
  - `ExternalLayout::FastImageResize` was `is_contiguous()`, and the kernel
    handed fast_image_resize one packed slice. fast_image_resize reads its
    source row by row, so it never needed that.
  - The pack was a quarter of a 1024² RGB u8 crop or flip then resize to 224².
  - The kernel was written three times (`resize_typed_u8/u16/f32`).
- **Resolution:**
  - `view-buffer/src/interop/fir.rs`: `FirViewAdapter<P>` gives fast_image_resize
    a view whose rows come from `ViewBuffer::dense_rows`, any row stride
    included.
  - `FastImageResize` is `is_dense_rows()`.
  - One generic `resize_pixels::<P>` packs only what the adapter refuses (a
    transpose, a horizontal flip).
  - The resize family declares `StridePreserving`.
  - Output is byte-identical.
- **Guards:**
  - `tests/resize_views.rs` compares crop / flip_v / both, over u8/u16/f32 ×
    rank 2 and 1–4 channels × three filters, against fir's own `ImageRef`
    over the packed pixels. It was watched failing against two mutated row
    reads (rows from 0, rows reversed). A first version compared against the
    engine's own packed path and passed both mutations, since both sides
    went through the adapter.
  - `copy_counts.rs` (`resizing_a_view_with_packed_rows_allocates_only_its_output`)
    was watched failing at 2 allocations before the change.
  - The fixture `fast_image_resize_takes_any_view_with_packed_rows`
    (`core/layout.rs`) was watched failing on the crop.
- **Measured** (`polars-cv/benchmarks/reports/2026-09-28-phase4-resize/`,
  1024², wheel target): crop then resize 1.34x, flip_v then resize 1.52x;
  contiguous input within noise.

### CR-57 — The planner packed a crop or vertical flip before grayscale, which reads it in place · `Resolved` · Low (perf)

- **Location:** `ImageOpKind::Grayscale`'s `memory_effect`
  (`RequiresContiguous`, `view-buffer/src/ops/image.rs`) and `grayscale_u8`
  (`execution/runner.rs`).
- **What is wrong:**
  - `grayscale_u8` reads dense rows where they lie (CR-54, Phase 1), and the
    CHANGELOG says a cropped or flipped input is not copied first.
  - But `build_plan` inserts `MaterializeContiguous` before grayscale whenever
    its input is a view, so every planned pipeline, the plugin included, still
    packs it: `flip(0).grayscale()` plans as
    `[View(Flip), MaterializeContiguous, Image(Grayscale)]`.
  - The `grayscale_u8_flip_h` benchmark cannot show it, since a horizontal
    flip has to be packed anyway.
- **Why it is not the one-line fix resize got:** `grayscale_strided` returns
  1-channel input unchanged. Declared `StridePreserving`, it would hand a
  view onward while the planner (`infer_strides` → `None`) records a
  contiguous output, and ops after it decide whether to materialise from that
  record.
- **Resolution:** `grayscale_strided` returns 1-channel input packed
  (`to_contiguous()`, free when it already is), and `Grayscale` declares
  `StridePreserving`.
- **Guards:**
  - `copy_counts.rs` (`grayscale_of_a_view_with_packed_rows_allocates_only_its_output`)
    was watched failing at 2 allocations before the change.
  - `strided_ops.rs` (`test_grayscale_of_a_one_channel_view_is_packed`) was
    watched failing ("flip_v: not packed") with the contract changed and the
    1-channel case not yet.
- **Measured** (`polars-cv/benchmarks/reports/2026-09-28-cr57-grayscale/`,
  wheel target): grayscale of a vertically flipped RGB u8 image 1.65x at
  1024², 1.5x at 512².

### CR-58 — A cheap row spent most of its time allocating layout copies · `Resolved` · Medium (perf)

- **What was wrong:** `Layout` kept its shape and strides in two heap `Vec`s.
  Every `ViewBuffer::clone`, every `to_contiguous()` of a contiguous buffer
  (a clone) and every `is_contiguous()` (which rebuilt a `LayoutFacts` from
  copies, plus a scratch vector) allocated, several times per row. `invert`
  on an 8×8 u8 row made 27 allocations and spent 58% of its instructions in
  the allocator; the kernel was ~2%. The plan-cache hit added three more (a
  key and a step-list copy), and the executor copied every row result into
  one call-sized vector before building the column.
- **Resolution:**
  - `core::layout::{Dims, Strides}` (`SmallVec<[_; 4]>`): layouts of rank ≤ 4
    are inline; `is_c_contiguous` is the one contiguity rule, allocation-free.
  - A plan-cache hit compares slices and shares an `Arc<[PlanStep]>`
    (`ExecutionPlan::execute_steps`).
  - `PendingSegment::ops` is inline; row results reach the column builder
    as per-range parts (`RowParts`), not concatenated.
- **Guards:** `copy_counts.rs` (`layout_bookkeeping_allocates_nothing`,
  watched failing at 20 allocations) and `compiled.rs`
  (`a_cache_hit_allocates_nothing`, watched failing at 1, the step-list copy).
- **Measured** (`polars-cv/benchmarks/reports/2026-09-28-phase5-per-row/`):
  instructions per 8×8 `invert` row 10,400 → 5,750; `invert` 1.29x eager and
  1.38x streaming at 8×8, 4 threads.

### CR-59 — The blob of a contiguous part of a buffer carried the whole buffer · `Resolved` · High

- **Location:** `ViewBuffer::write_blob_into` (`to_blob`, the `blob` sink).
- **What was wrong:** the payload length was the storage's length
  (`data.len()`), not the view's. A contiguous view of part of a buffer — a
  `crop` of whole rows — wrote every byte of the storage after its offset:
  too long for leading rows, and **a read past the end of the allocation**
  for rows further down. Found while profiling Phase 5.
- **Resolution:** the payload is `logical_len_bytes()`, the view's elements.
- **Guards:** `tests/blob_write.rs` gains leading- and middle-row slices (the
  length check failed on them); `test_zero_copy_encode.py`
  (`test_blob_of_a_full_width_crop_holds_only_the_crop`) checks the blob
  length and pixels through `crop` → `sink("blob")`, and failed against the
  unfixed extension (292 bytes for a 172-byte blob).

### CR-62 — The affine warp stored a 64-bit maximum as 0 · `Resolved` · Low

- **Location:** the warp's store (`runner.rs`'s `affine_warp_typed`, now
  `execution/warp.rs`): `clamp_for_dtype` then `NumCast`.
- **What was wrong:** `clamp_for_dtype` clamps a u64 to `u64::MAX as f64`,
  which is 2^64, one past the range (and an i64 to 2^63); `NumCast` then
  refuses the value and the store falls back to 0. A `warp_affine`/`rotate`
  of a u64/i64 image at the top of its range came back 0 there.
- **Resolution:** the warp stores through M5 (`CastFrom<f64>`,
  round-then-saturate), as every other float→integer store does. For every
  other dtype and value that is the old result (the parity tests compare).
- **Guards:** `sixty_four_bit_values_saturate_rather_than_become_zero`
  (Rust) and `test_a_warped_64_bit_maximum_stays_the_maximum` (Python), both
  watched failing.
- **Still latent:** `clamp_for_dtype` + `NumCast` remains in the typed
  grayscale (`luma_typed`) and the Gaussian blur's store. Neither reaches 2^64
  (their weights sum below 1: 1,200 blur sigma/shape cases and a white u64
  image all stayed in range), so there is no failing test to fix them against.

### CR-63 — `rotate(0)` spread NaN and infinity into neighbouring pixels · `Resolved` · Low

- **What was wrong:** a 0° rotation lowered to a full bilinear warp. For
  finite pixels it returned the input, but each output pixel blends its right
  and lower neighbours with weight 0, and `NaN * 0` and `inf * 0` are NaN: one
  NaN and one infinity in a 3×3 image came back as five NaNs. It also cost a
  whole warp (36 ms at 1024² RGB).
- **Resolution:** `execution::warp::rotate` returns the (packed) input for
  the 0° lowering, sharing its data. `Rotate` declares `RequiresContiguous`,
  so a planned input is already packed and the planner's record holds.
- **Guards:** `a_zero_degree_rotation_shares_its_input`,
  `a_zero_degree_rotation_leaves_every_value_where_it_was` (Rust, through
  the engine) and `test_rotate_zero_leaves_every_value_where_it_was`
  (Python), all watched failing.

### CR-61 — A `list` value the declared dtype cannot hold became 0 · `Resolved` · Medium

- **Location:** `graph/decode.rs`, the `list`/`array` source's converting
  path (`series_to_bytes`, now `convert_row_values`).
- **What was wrong:** a row whose values were not the declared dtype was
  converted by polars' non-strict cast, which nulls a value the target cannot
  hold, and the nulls were then read as the values under them: `300`, `-1`,
  `NaN` and `1e10` declared `u8` all became `0`, silently. Found while
  replacing the path in Phase 6.
- **Resolution:** the conversion is polars' strict cast of the row's values:
  such a value is an error naming it (and the row is null under
  `on_error="null"`). Float → integer still truncates, as polars' cast does.
- **Guards:** `a_value_the_declared_dtype_cannot_hold_is_refused` (Rust) and
  `test_a_value_the_declared_dtype_cannot_hold_is_refused` (Python), both
  watched failing on the old path.

### CR-60 — The plugin allocated with the system `malloc`, not polars' allocator · `Resolved` · Low (perf)

- **Location:** `polars-cv/src/lib.rs` declared no `#[global_allocator]`.
- **What was wrong:** pyo3-polars (0.27) provides `PolarsAllocator` so that a
  plugin shares polars' allocator. Without it, glibc decided when row buffers
  went back to the OS. Once CR-58 removed the small allocations that happened
  to pin glibc's heap, a call holding many large rows until the column was
  built (a 64×64 f32 `array` sink over 50k rows, one thread) had its freed
  memory trimmed and re-faulted by the next call: ~40% slower from the second
  call on.
- **Resolution:** `polars-cv/src/allocator.rs`: `PolarsAllocator`, wrapped to
  record that it is in use, is the global allocator outside the lib's unit
  tests (which keep `test_alloc`'s counting allocator). The installed polars
  ships its binary as `_polars_runtime_32`, but the capsule still resolves
  under its old name `polars.polars._allocator`: checked, not assumed.
- **Guard:** `_lib.__allocator__` and `tests/test_allocator.py`, watched
  failing both ways `PolarsAllocator` could quietly not apply: the
  `#[global_allocator]` line removed, and the capsule name changed.
- **Measured** (`polars-cv/benchmarks/reports/2026-09-28-cr60-allocator/`):
  the `array` case is back to 0.68–0.89 s from call 2 on (0.76–1.19 s on
  glibc); a `blob` sink of the same rows is 2x faster from call 2 on; the
  64×64 plugin cases are 1.15–1.66x faster at 4 threads, and nothing is
  slower. Peak RSS is ~10% higher.

### CR-53 — `invert` on other integer dtypes returns f32 against a `PreserveInput` contract · `Resolved` · Low

- **Location:** `ComputeOp::Invert` (`output_dtype_rule` = `PreserveInput`).
  Execution for i8/i16/u32/i32/u64/i64 read as f32 and returned `1 - x` as
  **f32**, the engine's `Invert` fallback in `elementwise::lower`.
- **What was wrong:**
  - The planner published the input dtype, and execution produced f32. Through
    the plugin this was not a silent wrong dtype but an error:
    `planned dtype i8 but execution produced F32`.
  - The value was also meaningless for those dtypes: `1 - x` rather than the
    range's mirror image.
- **Resolution (owner's decision: the `MAX - x` family):** every integer dtype
  inverts as `MAX + MIN - x`, in its own dtype. That is `255 - x` for u8
  (unchanged), `-1 - x` for a signed dtype (a literal `MAX - x` overflows
  there), and the bitwise complement `!x` for all of them.
  - `lower_to_scalars` lowers i8/i16 like u8/u16 (`-x + (MAX + MIN)`, exact in
    f32), so they fuse and run on the integer affine path.
  - u32/i32/u64/i64 are not exact in f32 and lower to `Lowered::IntNot` (`!x`,
    in place for a sole owner). The f32 fallback arm is deleted.
- **Guards:** `elementwise::tests::integer_invert_keeps_its_dtype`, the parity
  matrix (which checks those dtypes against `!x` instead of the legacy oracle),
  `fused_ops.rs::signed_invert_fuses_in_its_own_dtype`, and
  `reference/test_phase1_ref.py::test_invert_integer_keeps_dtype` through the
  plugin. Each was watched failing on the old code.

### CR-64 — Each kernel re-solved how it reads a view and where it writes · `Resolved` · Medium (perf, design)

> Found reviewing Phases 1–8 as a whole (2026-09-28).

- **What was wrong:** "read a view where it lies, write in place when the
  buffer is yours" was implemented five ways, one per phase:
  - `core::strided::Walk` (any view): packing, `cast`, the engine's f32 pass;
  - `ViewBuffer::dense_rows` (crop and vertical flip only, a `Vec<&[T]>`
    per call): u8 grayscale, u8 threshold, resize;
  - `unique_contiguous_mut` plus a hand-written in-place loop: six engine
    paths and threshold;
  - `to_contiguous()` then a slice loop: four of the engine's six execution
    paths, the statistics, typed threshold and typed grayscale;
  - the planner's `RequiresContiguous`: `normalize`, `adjust_contrast`.

  Whether a view was copied therefore depended on op × dtype × strategy: a
  u8 `scale` of a transpose was read in place, a u8 `invert` of the same view
  was packed first; u8 grayscale read a crop in place, u16 packed it. Each
  in-place / into / pack triple was its own place for a bug.
- **Two latent faults in the walk itself:**
  - `for_each_run` handed runs to a closure and flushed its scratch through
    `&mut dyn FnMut`. Both are functions of their own, compiled without the
    AVX2 build unless LLVM inlines them, which is the rule `core::dispatch`
    states. It went unnoticed because its one consumer (a u8 → f32 cast)
    vectorises on SSE2 anyway; the first version of the fix reproduced it at
    scale (f32 → u8 cast 4.5x slower, u8 grayscale 3x on the wheels).
  - A run could split a pixel, so pixel kernels could not use the walk.
    That is why `dense_rows` existed.
- **Resolution:** `view-buffer/src/core/map.rs`, the one traversal. A kernel
  is an `ElementMap` (one value to one value, told where its run starts, for
  per-channel maps) or a `PixelMap` (a `C`-channel pixel to one value).
  `map_owned` writes in place when `unique_contiguous_mut` allows; otherwise
  `map_new`/`map_pixels` write into spare capacity, reading a contiguous
  input as one run and any other view in `Walk` runs (whole pixels for a
  pixel map). Everything runs through `dispatch`.
  - The engine's strategies (integer affine, table, blocked f32 passes; the
    f32 "pass" merged into "blocked"), `cast`, `threshold`, `grayscale` and
    the f16 sink's conversion (CR-65) are maps. The hand-written
    traversals, `map_pixel_rows` and the typed grayscale and threshold
    kernels are deleted, and grayscale stores by `CastFrom` (so
    `clamp_for_dtype` is left only in the warp and blur oracles).
  - Statistics read runs (`map::for_each_run`), so `normalize` and
    `adjust_contrast` declare `StridePreserving` and the planner no longer
    packs before them.
  - `Walk` yields rows from an iterator and hands runs to a `RunSink`, whose
    `#[inline(always)]` method keeps the consumer inside the dispatched
    build. Runs are whole multiples of a grain (a pixel's channels), and the
    coalesced geometry is inline (a walk allocates nothing).
  - A unit's address is opaque to the optimiser (`black_box`). Otherwise
    LLVM hoisted a consumer's vector-loop overlap check out of the per-unit
    loop, and a negative row step (a vertical flip) failed the hoisted check
    for the whole image: u8 grayscale of a flipped image ran the scalar loop,
    5x slower. The old code avoided it by accident (row slices loaded from a
    `Vec`).
- **Guards:**
  - `copy_counts.rs`: `per_value_ops_on_a_view_allocate_only_their_output`
    (9 ops, 3 layouts), `threshold_of_any_view_allocates_only_its_output`
    and `grayscale_of_any_view_allocates_only_its_output`. Each was watched
    failing at 2 allocations: the view's pack and the output.
  - `strided::tests::a_run_holds_whole_grains_in_every_layout`, and a
    channel-first layout in the element-wise and grayscale parity tests,
    watched failing with the grain ignored.
  - `elementwise::tests::per_channel_kernels_match_the_legacy_code_across_blocks_and_runs`.
    No existing parity test had a per-channel kernel cross a 2,048-element
    block or a scratch-sized run. Each of the two position mutations
    (blocked, table) passed every other test and failed this one, at element
    2,048 and 8,192.
  - `typed_threshold_matches_the_comparison_reference` (typed threshold had
    no Rust test), watched failing against `>=`.
- **Measured** (`polars-cv/benchmarks/reports/2026-09-28-review-traversal/`,
  wheel target, three interleaved rounds): see the report. Views of per-value
  ops are 1.1–1.3x faster and non-u8 grayscale 1.6–2.2x. u8 preset normalize
  is 1.6–1.7x and u8 `adjust_contrast` 1.1–1.3x. An in-place f32 chain is
  1.2–1.75x and f32 z-score up to 2.1x. Contiguous u8 kernels are
  unchanged.

### CR-65 — The half-precision sink converted in the serial column build, one element at a time · `Resolved` · Low (perf)

- **Location:** `NumpyRowOutput::from_buffer_f16` (`polars-cv/src/output.rs`),
  called from `build_numpy_series`.
- **What was wrong:**
  - `.sink("numpy"|"torch", dtype="f16")` converted each row after all rows
    had been computed, in the column build, which runs on one thread. It was
    a per-row O(pixels) job in the serial tail of every call (the
    handover's open item 1, eager calls scaling poorly, in miniature).
  - The conversion made three passes: a cast to an f32 buffer, a pack, and
    `f16::from_f32` plus a 2-byte `extend_from_slice` per element. On the
    wheels, `from_f32` also checks for F16C at run time per element.
- **Resolution:** `ViewBuffer::to_f16_bits`, an element map (CR-64) that
  reads any view once, converts each 2,048-element block to f32 by the
  conversion rule, then to f16 with `half`'s bulk conversion (F16C eight at a
  time, detected once per block). The encode half calls it on the row's own
  thread; the column build only labels the rows `float16`
  (`NumpyRowOutput::from_f16_bits`). The output is bit-identical: the same
  instruction, or the same software fallback, per value.
- **Guards:** `convert::tests::f16_bits_are_the_per_element_conversion`
  (binary16's edges: NaN payloads, ±0, ±inf, the overflow tie at 65,520,
  subnormals and the underflow tie, ties to even, 5,000 random bit patterns)
  and `f16_bits_read_every_dtype_and_layout`. In Python,
  `test_f16_matches_numpy_rounding_on_edge_values` and
  `test_f16_of_an_integer_image_reads_it_as_f32`, against NumPy's own
  conversion, through the plugin. All four were watched failing against a
  truncating conversion.
- **Measured** (same report, wheel target): the conversion is 4.4–6x faster
  for f32 rows, 6.5–7x for u8 rows and 3–4x for a transposed view, and it
  now runs in parallel with the other rows.

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
