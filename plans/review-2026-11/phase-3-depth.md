# Phase 3 — Depth

Implementation plan for the third group of findings from the external review
of 0.41.0 (commit `c0d77d5`, 2026-11-11). Ledger entries CR-146 to CR-149,
plus the existing CR-118 and CR-79, in the root
[`ISSUES.md`](../../ISSUES.md). Follows [phase 1](phase-1-first-contact.md)
and [phase 2](phase-2-stabilise.md).

**Goal:** widen what the tool does well. Nothing here blocks adoption; do it
after the first two phases.

**Order:** CR-148, CR-149 (small, independent), CR-147, CR-146, then CR-118
if the owner decides for it.

The working rules at the top of phase 1 apply here unchanged.

---

## Withdrawn: a faster JPEG encoder

The review recommended an optional faster encoder, because the thumbnail
benchmark is lost to OpenCV on encode time (3.2 ms against 0.27 ms per
256-pixel thumbnail, from `benchmarks/README.md`).

**Withdrawn on re-verification.** `ISSUES.md` records an owner decision dated
2026-09-28 under "Decisions (do not relitigate)": the `image` crate's encoder
stays. `jpeg-encoder` was evaluated (1.62–1.65x faster, but 5–13% larger
files on smooth content) and rejected, and the IJG licence stays out of
`deny.toml`, which rules out libjpeg-turbo bindings. The eval is
`view-buffer/tests/jpeg_encode_eval.rs`. No ledger entry is opened.

---

## CR-148 — No text `explain()` on the lazy expression

**Re-verified: confirmed.**

### Evidence

`explain()`, `output_dtype()` and `current_domain()` exist on `Pipeline`
only. `LazyPipelineExpr`, where graphs are built, has a one-line `repr`.
Seeing a graph takes `sink(..., return_expr=False).show_graph()`, which needs
the `viz` extra and Graphviz's `dot`. `PipelineGraph`'s public methods are
`add_alias`, `add_node`, `check`, `optimize`, `set_multi_output`,
`set_output`, `show_graph`, `to_expr`, `topological_order`.

`Pipeline.explain()` shows node-scope passes only (identity elimination,
window pushdown). For `flip_h().flip_h()` and for a fusable scalar chain it
printed the same text optimised and unoptimised; the engine-level rewrites
(`view_flip_involution`, `scalar_fusion`, `cast_chain_collapse`) are not
visible anywhere.

### Approach

1. `PipelineGraph.explain() -> str`: the nodes in topological order, each
   with its input (column or node), its ops, and the planned state leaving
   it (domain, dtype, shape with unknowns shown), then the outputs with
   their sinks and planned Polars dtypes. All of this is already computed
   for `check()`; render it in Rust beside the planner (one `#[pyfunction]`
   over the graph JSON) so the text cannot disagree with what runs.
2. `LazyPipelineExpr.explain(format="native", *, optimized=True, **sink_kw)`
   builds the same graph `sink()` builds and returns its text. With
   `optimized=False` it shows the graph as written, as `Pipeline.explain`
   does.
3. Engine-level passes: show the fused segments the executor will form
   (`PendingSegment`, `graph/compiled/run.rs`; check first whether the
   grouping is fixed at compile time or per row). If that is not cheaply
   available as text,
   say in the output which passes are engine-level and not shown, so
   identical text is not read as "nothing was optimised".
4. `output_dtype`/`current_domain` on the lazy expression: forward to the
   node's planned state.
5. Then `sink(return_expr=False)` has one remaining use (`show_graph`); add
   `LazyPipelineExpr.show_graph(...)` and deprecate the flag under CR-141's
   path.

### Tests first

- The README cut-out example: `explain()` shows six nodes and one source
  optimised, eight nodes and three sources with `optimized=False`.
- Text names every output alias and its planned dtype, equal to
  `collect_schema()`.
- No Graphviz needed: the test runs without the `viz` extra.

---

## CR-149 — Published benchmarks are three releases old

**Re-verified: confirmed. See also the existing CR-79.**

### Evidence

`benchmarks/README.md` and the README table report 0.38.0, measured from a
locally built wheel before it reached PyPI, with `uv.lock` naming 0.37.0.
The checkout is 0.41.0. My own run on 0.41.0 (400 synthetic JPEGs, 4 cores)
agrees in ratio with the published w1 result (1.4x a threaded OpenCV loop),
so nothing suggests the claims are wrong, only unrefreshed. CR-79 (the
pycocotools parity test never runs in CI) is open in the ledger.

### Approach

1. Re-run `benchmarks/` against the published 0.41.x wheel; update
   `results/*.json`, the figures and both README tables in one commit.
2. Make staleness visible: `benchmarks/tests` asserts the version recorded in
   `results/*.json` is within one minor of `pyproject.toml`'s, or the README
   table carries the version it was measured on in its header (it does in
   prose today). Prefer the test; watch it fail before the refresh.
3. Close CR-79 as its ledger entry describes.
4. Add the two cases this review measured that the suite lacks: thread
   scaling at fixed work (row parallelism off against on), and
   multi-output sharing (one graph against separate expressions).

---

## CR-147 — `.bbox` has two accessors

**Re-verified: confirmed. No recorded reason for the size.**

### Evidence

`geom_catalog.json`: 35 contour accessors, 17 point, 2 bbox
(`bbox_pairwise_iou`, `bbox_correspond`). `BBOX_SCHEMA` is
`Struct{x, y, width, height: Float64}`. The geometry `AGENTS.md` describes
`.bbox` as existing for `BBoxMatcher`. Boxes are the commonest annotation
type, and the README's own second example computes a crop box by hand.

A box is four floats, so most of this is expressible as plain Polars struct
arithmetic. That is the argument for *not* adding it; the argument for is
that the per-row crop feature, the tool's differentiator, takes boxes as
input and every user writes the same conversions first.

### Approach

Add `BBoxFn` variants (`namespaces/fns.rs`), each a mode-generic definition
like the existing two, generated into `.bbox`:

| Accessor | Notes |
|---|---|
| `area()` | scalar per box |
| `to_xyxy()`, `from_xyxy()`, `to_cxcywh()`, `from_cxcywh()` | one conversion authority; `BBOX_SCHEMA` stays the stored form |
| `clip(width=, height=)` | per-row sizes, so `this`-free expressions from `image_info()` |
| `scale(sx=, sy=)`, `translate(dx=, dy=)` | mirror the contour/point accessors' names and `on_null` |
| `to_absolute`/`normalize(width=, height=)` | as `.point` and `.contour` have |
| `pad(px=)` / `expand(ratio=)` | the "crop with context" case |
| `to_contour()` | bridges to the 35 contour accessors and to `rasterize` |
| `nms(scores=, iou_threshold=)` | greedy, per list; reuse `pairwise::bbox_iou_matrix` |

Then a `crop(box=pl.col("bbox"))` convenience on `Pipeline` as `@_sugar`
over `crop`, declaring the op it lowers to.

Resolve first: negative-size boxes are accepted and score as false positives
(CR-76, open). Every new accessor needs the same answer, so decide CR-76
before adding ten more places for it to matter.

### Tests first

- Each accessor against a NumPy definition; `nms` against
  `torchvision.ops.nms` or `cv2.dnn.NMSBoxes` on random boxes, in the parity
  report.
- Round trips between the four formats are exact on integer boxes.
- Null boxes, empty lists, and a null per-row parameter under each
  `on_null` policy: these go through the shared `GeomParams` path, so the
  existing arity and null sweeps should pick the new functions up from the
  catalogue without per-function tests. Check that they do.

---

## CR-146 — Crop pushdown stops at neighbourhood ops

**Re-verified: confirmed, and harder than the review implied.**

### Evidence

`blur(sigma=2.0).crop(top=10, left=10, height=32, width=32)` over 400 JPEGs
took 503 ms with optimisation on and 494 ms off, and `explain()` printed the
same plan both ways. `hoist_spatial_windows` (`polars-cv/src/passes.rs:149`)
moves a crop left across a run of ops whose rule is
`SpatialDependency::Pointwise` and stops at anything else.
`SpatialDependency::Neighborhood(NeighborhoodSupport)` already carries each
filter's radius, known or per-row.

Decode-only for the same images was about 200 ms, so roughly 300 ms of the
503 is a blur over pixels that are then discarded.

### Why it is not a small change

1. **Passes answer with a permutation or a subset of the existing ops.**
   `run_pass` returns an order (`Vec<usize>`). Crossing a neighbourhood op
   needs two *new* ops: a wider crop before the filter and a trim after it.
   The pass interface has to grow an "insert" answer, and `Plan` has to plan
   the inserted ops through `plan::step` like any other.
2. **The halo must be clipped to the image**, whose size is unknown at plan
   time for a decoded source. `crop_clip` exists for exactly this. The trim's
   offset is then `min(top, r)` and `min(left, r)`, which is a plan-time
   constant when the crop's `top`/`left` are literals. A crop with per-row
   parameters is not crossed.
3. **It must be bit-identical**, per the ledger's first decision, and this
   area has form: `affine_fusion` was removed because it changed pixels, and
   CR-134 fixed optimisations that changed results on `this` and padded
   crops. Identity holds only if the filter's result at a pixel depends on
   nothing but the values within its radius and the border rule at the true
   image edge: no position-dependent rounding, no whole-image statistic, no
   SIMD tail that differs by alignment.

### Approach

1. Extend the pass result to allow replacing one op by a short sequence, and
   re-plan the node. Guard: the planned state leaving the node is unchanged
   (domain, dtype, shape), checked inside `run_pass`.
2. Add an explicit opt-in to the op contract, not an inference from the
   radius: a `window_exact()` rule on the neighbourhood ops, required with no
   default like the other contract methods, returning whether a halo of the
   declared radius reproduces the full-image result exactly. Start with
   `false` everywhere.
3. Turn it on one op at a time, each behind a bit-identity sweep: random
   crops, including ones touching each edge and corner, every dtype the op
   accepts, every border mode, optimised against unoptimised, compared
   byte for byte. Candidates in order of likely exactness: `erode`/`dilate`,
   `median_filter`/`rank_filter`, `convolve2d`, `sobel`, then `blur`.
   `canny` (hysteresis is global), `equalize_histogram` and
   `distance_transform` are never eligible.
4. The pass refuses to cross when the crop reads `this`, when the op reads
   another node, when the radius is per-row, or when the op sits before an
   `assert_shape` — the same exclusions the pointwise pass has today.
5. Add the blur-then-crop case to the regression suite and to
   `OptFlags`' documentation.

### Tests first

- The sweep in step 3, written before any op is switched on, passing
  trivially (nothing crosses), then failing for an op deliberately marked
  exact that is not (`canny`) — that is the "watched failing" for the guard.
- `explain()` shows `crop_clip → blur → crop` for an enabled op.
- `opt_flags=False` output equals optimised output across the golden corpus.

### Risks

This is the highest-risk item in the three phases for the least certain
gain: it helps only pipelines that filter and then crop, and the user can
already write the crop first. If the sweep fails for `blur`, stop there and
record which ops are exact; a partial result is still correct.

---

## CR-118 — CSE in Rust  **decision** (existing entry)

**Re-verified: the ledger already holds this, with the option and its cost.**

The review recommended moving common-subexpression elimination out of
Python. CR-118 says the same and marks it **decision**: it "moves the
optimiser across the FFI and changes the wire again". Nothing in the review
changes that analysis. Two additions:

- Measured benefit of CSE as it stands: three outputs from one graph ran in
  400 ms against 782 ms with optimisation off. It works; this is about where
  it lives, not whether.
- If CR-141 declares a stable surface, the per-node wire is private and can
  change freely, which removes the main cost CR-118 names. Decide CR-141
  first.

No new ledger entry. If the owner decides for it, the plan is the option
CR-118 already describes, with one guard added: the eager/lazy parity sweep
(`tests/test_append_contract.py`) and the CSE interaction tests
(`test_cse_interactions.py`, `test_cse_optimization.py`) run unchanged
against the Rust implementation before the Python one is deleted.
