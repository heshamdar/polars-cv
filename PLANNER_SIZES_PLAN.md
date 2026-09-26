# Plan: rank-N planned shapes, and validation over partially known sizes

> Follow-up to PR #100 (typed-op consolidation, C0–C8). It closes the two open
> deviations `HANDOVER.md` records:
>
> 1. `plan::check_rank` passes `1` for an unknown size to `Op::validate`: the
>    last placeholder value in the planner.
> 2. `plan::State.dims` holds three sizes, so the sizes of a rank-4+ buffer
>    are not planned.
>
> It also fixes four defects found while reviewing #100, each reproduced at the
> user entry point (section 1). The same rules as the consolidation apply:
> delete first, test-first at the entry point, every guard watched failing,
> `scripts/verify.sh` green at each phase exit.

| Phase | Status |
|---|---|
| S0 — Failing tests for the review findings | **done** — 15 strict gaps in `tests/test_known_gaps.py::TestPlannedSizes` (each watched failing with `--runxfail`); 6 controls in `tests/test_plan_claims.py` (3 watched failing, 3 forward guards for S1/S2) |
| S1 — The planned shape is one rank-N value | **done** — `PlannedShape` (`Ranked`/`Unranked`) replaces `ndim` + `dims[3]`; a buffer's shape publishes at every rank; `assert_shape` is `{dims, exact}`; one `apply_declaration` for plan and row; R2/R3/R6 gaps moved into the suite; see *S1 as done* for deviations |
| S2 — One `validate`, over symbolic sizes | **done** — `Op::validate(&[&[Dim]], &[PlannedDType])`, every error a verdict; `check_rank`, the `1` placeholder and `depends_only_on_rank` deleted; `validate_concrete` for execution; per-axis `broadcast_dims`; binary ops validated at plan time; soundness property over the catalogue; R1/R4/R5 gaps moved into the suite; see *S2 as done* |
| S3 — Operand reads are one mechanism | **done** — `GraphStep::operands()` is the one list of nodes a step reads; the planner passes each one's planned shape to `shape()`/`validate()`; rasterize's canvas is `OpShape::Canvas`; `apply_mask`/`channel_merge` validated and planned against their operands; see *S3 as done* |

---

## 1. Why: what the two deviations cost today

Each row below was run against the debug build of `e131709`.

| # | Entry point | Today | Cause |
|---|---|---|---|
| R1 | `Pipeline().source("raw", dtype="u8").blur(sigma=1.0)` (also `resize`, `perceptual_hash`, `channel_select`) | builds; every row then fails with "a [H, W] or [H, W, C] buffer. Got shape [16]" | `depends_only_on_rank()` classifies by error *variant*. `require_hw_or_hwc` (`validation.rs:105`) and phash's rank check return `ShapeRequirement`/`InvalidParameter`, so a verdict decided by a **known rank** is dropped as if it were size-level |
| R2 | `Array(u8, (4, 5))`, `(6,)`, `(2, 3, 4, 5)` column → `source("array")` → `.sink("array")` | `.sink()` accepts (`ColumnFacts::Pending { sizes: true }`), then `collect_schema()` fails with "needs the full output shape" | `OutputSpec::planned` publishes a shape only when `ndim == Some(3)` (`graph/types.rs:64`); `refine_by_column` zips a rank-4 column's sizes into three slots. The CHANGELOG entry "a fixed-size `Array` column's shape is planned" holds only at rank 3 |
| R3 | `source("list", dtype="f32").assert_shape(dims=[4, 5]).sink("array")`; `…resize(8, 8).channel_select(0).sink("array")` | refused, "needs the full output shape", although every size is known | same gate; pinned as deliberate by `_KNOWN_BUT_UNEXPRESSIBLE` (`tests/test_schema_parity_array_sink.py:102`). The `assert_shape` docstring promises that `dims=` "lets a list/array source reach an `array` sink" |
| R4 | two `Array` columns `(4, 5, 3)` and `(2, 5, 3)`, `x.add(y).sink("array")` | **plans `Array(u8, (4, 5, 3))`**, then every row fails "cannot be broadcast together" | `OpShape::Broadcast` over two known, incompatible shapes falls back with `.unwrap_or(a)` (`shape_rule.rs:306`), and the planner never validates graph ops (`check_rank` returns `Ok` for `GraphStep::Graph`). This is a published schema that execution cannot produce. It became reachable when e47ae7f started planning Array-column sizes |
| R5 | two image pipelines resized to 8×8, `a.add(c)` | planned `dims=[None, None, None]` | `Broadcast` needs *every* size known on both sides, or it gives up on all of them. Broadcast is per-axis, so the known `8`s are dropped for no reason |
| R6 | `assert_shape(dims=[2, 3, 4, 5])` | refused: "supports 1 to 3 dimensions … pass the shape to the sink" | `AssertShape.dims: [_; 3]`, `declare` |

R1 and R4/R5 are deviation 1 (validation over partially known sizes). R2, R3
and R6 are deviation 2 (three slots).

---

## 2. Target end state

1. **The planned shape is one value whose rank is its length.** `State` loses
   `ndim` and `dims: [Option<usize>; 3]` and gains:

   ```rust
   pub enum PlannedShape {
       /// Rank known: one entry per axis, `None` where the size is unknown.
       Ranked(Vec<Option<usize>>),
       /// Rank unknown; sizes declared for leading axes (`assert_shape(height=)`
       /// before the rank is known). Never longer than `DIM_NAMES`.
       Unranked { leading: Vec<Option<usize>> },
   }
   ```

   A size past the rank, or a rank that disagrees with the sizes, cannot be
   represented. The clipping line `dims.iter_mut().skip(n)` in `step`, the
   `ndim == Some(3)` gate, and `known_sizes`/`input_dims`'s positional
   re-derivation all go. A shape is published for **every** rank whose sizes
   are all known. `DIM_NAMES` survives only as the spelling of the
   `height=`/`width=`/`channels=` keywords.

   (The consolidation plan's own target said `Vec<Dim>`. Storing
   `Option<usize>` rather than `Dim` is deliberate: an `Input(k)` symbol is
   only meaningful relative to one node's entry, so persisting it across
   nodes and binary operands needs node-qualified symbols. See section 6.)

2. **One `validate`, symbolic, like `OpShape::dims`/`concrete`.**

   ```rust
   fn validate(&self, inputs: &[&[Dim]], dtypes: &[PlannedDType]) -> Result<(), ValidationError>;
   ```

   Every check reads a size through `Dim::known()` and **says nothing about a
   size it does not know**, the convention per-row parameters already follow
   (`M::sym`/`known`). A rank check always decides, because a ranked input's
   length *is* its rank. Execution calls a provided wrapper that passes
   all-`Known` dims and known dtypes, the way `OpShape::concrete` wraps
   `dims`:

   ```rust
   fn validate_concrete(&self, shapes: &[&[usize]], dtypes: &[DType]) -> Result<(), ValidationError>;
   ```

   The planner raises every error `validate` returns, with no filtering.
   `check_rank`, the `1` placeholder, `fully_known` and
   `ValidationError::depends_only_on_rank` are deleted: nothing is left to
   classify, because an error is only ever returned for a fact that is known.

3. **Broadcast is per-axis and never falls back.** `OpShape::Broadcast`
   aligns from the right: each axis is known when both sides are known and
   compatible, or one side is a known `1`. Two known, incompatible sizes are a
   `validate` verdict at plan time, not a fallback to the left shape.

---

## 3. Phases

### S0 — Failing tests for the review findings (test-first) — done

Each finding is a strict gap in `tests/test_known_gaps.py::TestPlannedSizes`,
at the user entry point. `_gap` now pins `raises=AssertionError`: a gap that
breaks for any other reason (a renamed helper, say) fails the suite instead of
reading as the defect (watched: a `NameError` injected into one gap reports
FAILED). The module-wide `structural` mark was dropped; the gaps are runtime
tests and carry `plugin_required`.

| Gap | Finding | Closed by |
|---|---|---|
| `test_a_known_rank_refuses_an_image_op_at_build[blur, channel_select, grayscale, perceptual_hash, resize, threshold]` | R1 (each op confirmed to fail every rank-1 row, asserted in the test) | S2 |
| `test_an_array_column_plans_its_shape_at_every_rank[(6,), (4, 5), (2, 3, 4, 5)]` | R2 | S1 |
| `test_a_declared_shape_of_any_rank_reaches_an_array_sink[(20,), (4, 5)]` | R3 | S1 |
| `test_assert_shape_declares_a_rank_4_shape` | R6 | S1 |
| `test_incompatible_known_operands_refuse_at_build` | R4: two `assert_shape`d image operands, `.sink()` publishes `(4, 5, 3)` | S2 |
| `test_incompatible_array_columns_refuse_at_plan` | R4 over Array columns, whose sizes are known only once Polars plans, so the refusal is at `collect_schema()`, not `.sink()` | S2 |
| `test_broadcast_keeps_the_sizes_both_operands_know` | R5 | S2 |

**Controls** (`tests/test_plan_claims.py::test_an_unknown_fact_is_never_refused_at_build`):
builder calls whose refusal would rest on an unknown fact must build. With
`check_rank` forced to raise every placeholder verdict, the `channel_select`,
`channel_swap` and `crop` cases fail. `threshold` (the placeholder 1 happens
to pass), `blur` over an unknown rank (S1's `Unranked` path) and `add` with
one operand unknown (S2's binary validation) guard code that does not exist
yet. **Each must be watched failing when its path lands.**

`_KNOWN_BUT_UNEXPRESSIBLE` is merged into `_KNOWN_SHAPE` in S1 (agreed).

### S1 — The planned shape is one rank-N value (deviation 2; fixes R2, R3, R6)

**Delete first:**

| What | Where | Replaced by |
|---|---|---|
| `State.ndim`, `State.dims: [Option<usize>; 3]` | `src/plan.rs:45-49` | `State.shape: PlannedShape` |
| `known_sizes`, the `[None; 3]` arms and the clipping in `step` | `plan.rs:294-308, 329-335` | `PlannedShape::from_dims(Option<Vec<Dim>>)` |
| `sizes_over_any_rank`'s fixed `1..=3` | `plan.rs:340-354` | evaluate over ranks `1..=DIM_NAMES.len() + 1` from `Unranked.leading`, justified by a guard (below) |
| the `ndim == Some(3)` gate | `graph/types.rs:64-66` | a shape published whenever `Ranked` and all known, for the buffer domain. Vector outputs keep their current `VectorList`/`VectorArray` treatment, to be checked explicitly in this phase |
| `refine_by_column`'s `zip` over three slots | `graph/compiled.rs:1513-1535` | `Ranked(sizes)` from `peel_nesting`, any depth |
| `AssertShape { rank, dims: [_; 3] }`, which can express contradictory states (`rank: 2` with `dims[2]` set) | `src/ops/graph.rs:212`, `GraphOpRef` | `AssertShape(Declared<M>)` with `Declared::Full(Vec<Option<M::V<u32>>>)` (`dims=`, pins the rank) and `Declared::Leading { height, width, channels }` (the keywords). "A parameter read only under some branch becomes an enum variant" (CLAUDE.md) |
| `declare` (plan) and `check_declared_shape` (execution): the same match written twice | `plan.rs:172-233`, `compiled.rs:1614-1660` | `Declared::check(&[Dim]) -> Result<PlannedShape, String>`, called over planned dims by the planner and over all-known dims per row, with one error text. Same pattern as S2 |
| "supports 1 to 3 dimensions" error; `PlanState.dims` 3-tuple getter; `DIM_NAMES` classattr as a width | `plan.rs:70-102, 180-188` | `PlanState.dims` returns a tuple of length rank, or the leading sizes; `ndim` stays a derived getter. Removed surface pinned in `tests/test_removed_surfaces.py` |

**Wire and seams:** the pickled `PlanState` form changes. `tests/_plan_view.py`
absorbs this (its `height`/`width`/`channels` become `dims[0..3]`, and it gains
`dims`). The hand-built graph JSON form of `assert_shape` changes: add a note
to `docs/user-guide/migration.md` and a CHANGELOG entry correcting the
rank-3-only claim.

**Guards:**

- `a_shape_is_rank_stable_past_its_patterns`: a view-buffer property test
  that, for every `OpShape` variant, `dims` over rank *n* and rank *n + 1*
  (n ≥ 3) agree on the leading axes and carry the extra axis through. It
  justifies `sizes_over_any_rank`'s bound. Watch it fail by bounding at
  `1..=2`.
- The R2/R3/R6 tests from S0 go green; `_KNOWN_BUT_UNEXPRESSIBLE` is merged
  into `_KNOWN_SHAPE` (with agreement).
- `validate_output_schema` now compares shapes at every rank, which widens the
  existing `plan == data` guard for free. Run the full suite plus
  `gen_golden_corpus.py` in check mode and list every corpus entry that
  changes (rank-1/2 outputs gain a shape).

### S1 as done — deviations and findings

- **`assert_shape` is `{dims: [...], exact: bool}`, not `Declared::Full` /
  `Declared::Leading`.** A nested object field has no catalogue type (the
  encoder tells `OneOf` options apart only as sequence vs scalar), so the
  enum would have needed a new `TypeDesc` kind; a registered enum for the
  flag would have generated a public Python class for an internal op. Both
  fields are read in every branch and every combination means something
  (`exact: false` is the keywords' leading dimensions, of any count), so no
  contradictory state remains. `exact` defaults to `true`, so hand-built JSON
  writes `{"op": "assert_shape", "dims": [...]}` for a whole shape.
- **Channel shapes have no output at a rank their ops refuse.**
  `SingleChannel`, `ColorChannels`, `DropChannelAxis` and `StackChannels`
  returned "unchanged" (or three unknowns) for any rank but their patterns,
  an output no op ever produces: every op behind them runs only on `[H, W]`
  or `[H, W, C]`. Evaluated over an unknown rank, that invented rank-4 output
  hid a known channel count (`assert_shape(channels=3).grayscale()` lost
  `C = 1`; `test_assert_shape_survives_a_continuation` caught it). They now
  give no output (`None`) there, and `dims_over_unknown_rank` skips such a
  rank. `a_rank_known_without_the_input_holds_over_every_input` now holds
  over every rank that has an output (it pinned `StackChannels`'s invented
  three unknowns).
- **The bound is `DISTINGUISHED_RANK = 3`, evaluated up to it (no `+ 1`).**
  With the channel shapes honest, no variant tells a rank above 3 apart.
  `a_shape_is_rank_stable_past_its_patterns` was watched failing at 2
  (`ColorChannels` at rank 3) and passes at 3. `sizes_over_any_rank` moved
  into view-buffer as `OpShape::dims_over_unknown_rank`, beside its bound.
- **A per-row declaration keeps a known size.** `declare` used to overwrite
  a known size with `None` for a per-row `assert_shape` entry; the known size
  still holds (the row check adds to it), so it is kept
  (`a_per_row_declaration_keeps_a_known_size`).
- **Tests changed on purpose** (each pinned removed behaviour):
  `test_dims_rejects_what_it_cannot_track` (its "1 to 3 dimensions" case;
  renamed `test_assert_shape_rejects_a_malformed_declaration`),
  `test_assert_shape_has_no_batch_parameter` (its "exactly three sizes"
  line), `_KNOWN_BUT_UNEXPRESSIBLE` merged into `_KNOWN_SHAPE` (agreed), the
  `assert_shape` wire in `test_serialization.py` and the Rust fixtures, and
  `output_facts_are_planned_from_the_ops` (a rank-2 output now publishes
  `[4, 5]`). Removed names are in `check_removed_symbols.py`; `PlanState.DIM_NAMES`
  and `_assert_shape(rank=)` are pinned in `test_removed_surfaces.py`.
- **Golden corpus unchanged**: no recorded case sinks a rank-1/2 buffer to
  `array`, so no entry's plan or output moved.
- The S0 control "blur, rank unknown" guards a path S1 did not add (the
  planner still validates nothing over an unknown rank); it moves to S2.

### S2 — One `validate`, over symbolic sizes (deviation 1; fixes R1, R4, R5)

**Delete first:**

| What | Where | Replaced by |
|---|---|---|
| `Op::validate(&[&[usize]], &[DType])` | `view-buffer/src/ops/traits.rs:154` | `validate(&[&[Dim]], &[PlannedDType])` plus the provided `validate_concrete`. The signature change makes the compiler list every impl |
| `ValidationError::depends_only_on_rank` | `ops/validation.rs:68` | nothing: every returned error is a verdict |
| `check_rank`: the `1` placeholder, `fully_known`, the variant filter, and `validate(…, &[])` dropping planned dtypes | `src/plan.rs:440-461` | `step` calls `op.validate(&[input], &[state.dtype])` and raises whatever it returns |
| `OpShape::Broadcast`'s all-or-nothing `known(a).zip(known(b))` and `.unwrap_or(a)` | `shape_rule.rs:302-315` | per-axis broadcast over `Dim`; a known incompatibility is `validate`'s verdict, and `dims` never invents a shape for it |
| `GraphStep::Graph(_) => Ok(())` in `check_rank` | `plan.rs:449` | binary ops validate over both operands' planned dims (`BinaryOp::validate` is already written for two inputs) |

**Per-impl audit** (11 impls plus the helpers). A check that reads a size
returns an error only when that size is `Known`:

| Impl | Size reads to make `Dim`-aware |
|---|---|
| `validation.rs` helpers | `require_hw_or_hwc`/`require_spatial` are rank-only. `require_single_channel`/`is_2d_like` read `C`; `require_channels_at_least` reads `C`; `require_axes` is rank-only. `ShapeRequirement.got` becomes `Vec<Option<usize>>`, rendered with `?` for unknown sizes |
| `ImageOp` (`image.rs:350`) | resize `C > 4`; `ChannelSwap` `order.len() == C` and each index `< C` |
| `ColorConvertOp` (`color.rs:86`) | channels at least the source space's |
| `ConvolveOp` (`filter.rs:85`) | rank only |
| `ComputeOp` (`compute.rs:647`) | preset `Normalize` mean/std length vs `C`; dtype check (now fed the planned dtype) |
| `ViewOp` (`view.rs:197`) | `Reshape` element count (decides only when all known); `Crop`/`Slice` per-axis bound (decides per axis that is known, which is a gain over today's all-or-nothing) |
| `ReductionOp`, `PerceptualHashOp`, `HistogramOp`, `GeometryOp`, `BinaryOp`, `mask.rs` | phash rank and `C > 4`; broadcast per axis; the rest are rank or parameter-only |

**Guards:**

- **Soundness property** (view-buffer, table of every catalogue sample
  × random known shapes at ranks 1–4): for any shape *S* and any mask that
  turns some sizes `Unknown`, `validate(masked).is_err()` implies
  `validate_concrete(S).is_err()`. A plan-time refusal is always a real
  verdict. Watch it fail by re-introducing a placeholder `1` in one helper.
  Fixtures: one known-bad impl (reads an unknown `C` as 1) that it must
  reject, and one known-good.
- **Completeness at full knowledge:** `validate` over all-`Known` dims equals
  `validate_concrete`. This holds by construction (a wrapper), so it gets no
  test; the wrapper is the mechanism.
- R1, R4 and R5 go green. The executor call sites (`expr.rs:122`,
  `compiled.rs:833/862/882/909`, `encode.rs:29`) change only to the wrapper's
  name.

### S2 as done — deviations and findings

- **`validate_concrete` is a free function**, not a provided trait method:
  a provided method could be overridden and drift from `validate`; a free
  function over `&dyn Op` cannot, so "execution calls the same `validate`" is
  structural.
- **`GraphStep::validate` routes every engine family**, not only `Buffer` and
  `Geometry`: `check_rank` returned `Ok` for reductions, histograms and
  perceptual hashes, so their rank checks (a reduction axis past the rank, a
  hash of a rank-1 buffer) never ran at plan time. `GraphOp::validate` is an
  exhaustive match over `Role`; a binary op validates both operands' planned
  dims, and the other node readers wait for S3.
- **`ValidationError::ShapeMismatch` deleted**: nothing constructed it.
  `ShapeRequirement.got` is `Vec<Dim>`, rendered `[16, ?, 3]`; a fully known
  shape renders exactly as before. `Normalize`'s mean/std mismatch now names
  the input shape (it reported `[mean_len, std_len, channels]` as a "shape").
- **An input of unknown rank is not validated** (it gives `validate` nothing
  to decide), so the S0 control "blur, rank unknown" still guards a path the
  planner does not take.
- **Soundness property** `plan::validation_soundness`: every catalogue sample
  plus seven size-sensitive ops, over every shape of rank 1–4 with sizes
  {1, 2, 3, 5} (binary: rank 1–3, sizes {1, 2, 3}) and every mask of unknown
  sizes. Watched failing by reading an unknown channel count as 1 in
  `require_channels_at_least` (`cvt_color` refused `[1, 1, ?]`, though
  `[1, 1, 3]` runs). Fixtures: a three-channel check that reads an unknown
  channel count as 1 (caught) and the same check saying nothing (passes).
- **Controls watched failing**: `threshold` (with `require_single_channel`
  reading an unknown channel count as 3) and `add` with one operand unknown
  (with `broadcast_dims` refusing a known size against an unknown one).
- **Tests changed because the behaviour they pinned was removed**:
  `test_offset_crop_with_full_extent_is_kept` and
  `test_offset_crop_with_full_extent_is_not_eliminated` built a pipeline whose
  crop runs past a known edge (every row failed); it is now refused at build,
  which both assert, and each keeps its optimizer property over a size only
  the data states. The R1 gap's "every row fails" precondition went with its
  marker (building now raises first).

### S3 — Operand reads are one mechanism (optional, elegance)

`step` special-cases the ops that read another node: `binary()` gets the other
operand's state, and `Rasterize { size: FromNode }` has a hand-written arm
that copies the canvas H/W (`plan.rs:310-318`) outside the op's `OpShape`.
Replace both with one `GraphStep::operands() -> Vec<&NodeRef>`. The planner
passes each operand's planned dims to `shape()` and `validate()` as further
inputs. Rasterize's shape becomes `OpShape::HwOf(1)` (H/W read from input 1),
and `apply_mask`/`channel_merge` can then validate and plan against their
operands' sizes too. This removes the last per-op arm in `step`. In scope
(user decision); it changes nothing user-visible beyond earlier errors.

### S3 as done — deviations and findings

- **One list, three readers.** `GraphStep::operands()` (a binary op's other
  operand, `apply_mask`'s mask, `channel_merge`'s channels, rasterize's canvas
  node) feeds the planner's inputs, and `reads_other_nodes()` (the spatial
  pushdown's barrier) is now "has operands" rather than a second hand-kept
  list (`GraphOp::reads_other_nodes` deleted). Rasterize from a node counts as
  one now, which only makes the barrier more conservative (it never hoisted
  across rasterize, a domain change).
- **The rasterize arm in `step` is gone**, and with it
  `PlannedShape::with_size`. Its shape is `OpShape::Canvas { of: 1 }`.
- **Operands of unknown rank.** `OpShape::dims_over(&[PlannedInput])`
  replaces `dims_over_unknown_rank`: every input is ranked or unranked, and
  an unranked one is evaluated over the ranks it may have. With several
  inputs, sizes are claimed only when the output rank does not depend on the
  unranked ones (`Broadcast` aligns from the last axis, so a longer operand
  shifts every position). So a canvas of unknown rank still gives its
  declared H and W, as the special arm did. Guarded by
  `an_unranked_operand_claims_only_what_every_rank_keeps`, watched failing
  without the rule (`Broadcast` claimed a size a rank-4 operand shifts).
- **`StackChannels` reads every input's H and W**, not only its own: an input
  of unknown size takes them from the channels that know them, and two
  different known sizes have no output (as `validate` refuses them).
- **`validate_mask` / `validate_channel_merge` take `Dim`s and planned
  dtypes**, so the planner calls them; execution goes through
  `GraphStep::validate_concrete` (the same `validate`) for binary ops, masks
  and merges alike. The soundness property enumerates `1 + operands()` inputs
  (a three-input merge included), watched failing with
  `validate_channel_merge` reading an unknown H as a clash.
- **Test changed**: `test_crop_does_not_cross_apply_mask` built `apply_mask`
  over a bare node id with no planned state, which the planner now refuses
  as it already did for binary ops and rasterize; its mask is a real node, and
  the barrier it pins is unchanged.

---

## 4. Decisions (answered)

1. **Move `_KNOWN_BUT_UNEXPRESSIBLE` into `_KNOWN_SHAPE`:** yes, in S1.
2. **`assert_shape` wire becomes `Declared::Full` / `Declared::Leading`:**
   yes, in S1 (hand-built graph JSON migration note and CHANGELOG entry).
3. **S3:** in scope, after S2.

## 5. Order and size

S0 is small. S1 is medium: `plan.rs`, `compiled.rs`, `types.rs`, `graph.rs`,
the plan-view seam, and the golden corpus. S2 is the large one: every `Op`
impl and the validation helpers, but the compiler enumerates the work and the
property test checks it. S1 comes first because S2's property test needs
rank-4 inputs to be plannable, and S1's user-visible fixes do not depend on
S2.

## 6. Out of scope, noted

- Persisting `Dim::Input(k)` symbols in `State`, so a pass could prove that
  `transpose ∘ transpose` preserves an unknown shape, needs node-qualified
  symbols across binary operands. Revisit only if a pass needs it.
