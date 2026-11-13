# Phase 3 — Depth

Implementation plan for the third group of findings from the external review
of 0.41.0 (commit `c0d77d5`, 2026-11-11). Ledger entries CR-146 to CR-149,
plus the existing CR-118 and CR-79, in the root
[`ISSUES.md`](../../ISSUES.md). Follows [phase 1](phase-1-first-contact.md)
and phase 2 (complete; its plan is in git history).

**Goal:** widen what the tool does well. Nothing here blocks adoption; do it
after the first two phases.

**Order:** CR-148, CR-149's guard with CR-79, then CR-76 and CR-147,
CR-146; CR-118 is not scheduled. CR-149's refresh runs with
the next release (below).

**Status:** CR-148, CR-79 and CR-76 are closed (`ISSUES.md`, Closed findings), and
CR-149's guard is pinned in `test_benchmark_freshness.py`. CR-148 went further than
its plan after review (owner, 2026-11-13): `expr.graph(format)` is the one way
to a graph, `explain()` and `show_graph()` render one Rust description of it,
and `sink(return_expr=False)`, `Pipeline.explain` and `Pipeline.to_graph` are
removed. A sink with settings is a `Sink` value, not keywords. The optimised README cut-out is six nodes and
one decode against eight and three as written, as first measured.

The working rules at the top of phase 1 apply here unchanged.

### Re-verification before implementation (2026-11-12, at `42e08b3`)

Phases 1 and 2 have landed except CR-139's remainder and CR-141, both
decisions, and CR-140 (a Windows wheel, Python 3.14), which is not started. Every finding's evidence below still holds against the code;
what the second reading changed:

- **CR-148.** Two corrections. Segment boundaries are fixed at compile time
  (a step is `GraphStep::Buffer` or it is not, `run.rs`), but the engine
  rewrites inside a segment are planned per source layout (`run_segment`,
  `CachedPlan`), so the text can show the segments and which engine passes
  will run over each, not what they rewrite. And `sink(return_expr=False)`
  is not one use: about twenty test files and the multi-output guide read
  the `PipelineGraph` through it. It stays; whether it is part of the stable
  surface is CR-141's list, and no deprecation path exists to put it on
  (CR-142's tuple was removed outright).
- **CR-149.** `images.cocodataset.org` does not resolve from a web session,
  so the refresh runs on the owner's machine. `benchmarks/` pins the
  published wheel, and the next release replaces `sink("numpy")` with
  `sink("tensor")` (CR-81), so the harness changes for that release anyway:
  refresh at that release, not on 0.41.x. The staleness guard goes in now as
  a strict xfail, the ledger's mechanism for a verified defect.
- **CR-147.** `.bbox`'s two accessors take box sets (`List[BBOX_SCHEMA]`),
  but its reader, `BBoxColumn`, already reads one box or a set per row
  (`Arity`), so a new accessor maps over whichever it is given, as the
  contour accessors do. `bbox_from_coords` (`geometry/coords.py`) is already the one
  conversion *into* `BBOX_SCHEMA`; `from_xyxy`/`from_cxcywh` would be a
  second, and the `to_*` conversions belong beside it as its inverse. `crop`
  takes `u32` offsets and a box is `Float64`, so the `crop(box=)` sugar needs
  a stated rounding rule. CR-76 is still open and still comes first.
- **CR-146.** Unchanged. `hoist_spatial_windows` is now `passes.rs:155`.
  One addition: a crop out of bounds must still be refused by a message that
  names the user's crop, not the inserted trim.
- **CR-118.** Unchanged. (Since decided: CR-141 keeps everything free to
  change until 1.0, and CR-118 is not scheduled.)

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

1. Re-run `benchmarks/` against the next published wheel (the release that
   carries CR-81's `tensor` sink, which the harness has to adopt anyway);
   update `results/*.json`, the figures and both README tables in one
   commit. Needs COCO val2017, which a web session cannot download: on the
   owner's machine.
2. **Done** (`tests/test_benchmark_freshness.py`, every `coco-*.json`). Make staleness visible: a test asserts the version recorded in
   `results/coco-5k.json` (`environment.libraries["polars-cv"]`) is within
   one minor of `polars-cv/pyproject.toml`'s. It fails today (0.38 against
   0.41), so it lands as a strict xfail in `test_known_gaps.py` and the
   refresh turns it into a pass.
3. **Done.** Close CR-79 as its ledger entry describes.
4. Add the two cases this review measured that the suite lacks: thread
   scaling at fixed work (row parallelism off against on), and
   multi-output sharing (one graph against separate expressions). These
   measure the checkout, so they go to `polars-cv/benchmarks/regression`,
   not the cross-library tree.

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
like the existing two, generated into `.bbox`. `BBoxColumn` reads one box or
a set per row (`Arity`), so each per-box accessor keeps the arity it is
given (`area()` on a box column is `Float64`, on a set `List[Float64]`), as
the contour accessors do; `nms` takes a set.

| Accessor | Notes |
|---|---|
| `area()` | per box |
| `to_xyxy()`, `to_cxcywh()` | the inverse of `bbox_from_coords`, beside it in `geometry/coords.py` (plain Polars over the set); `bbox_from_coords(format=)` stays the only way in, so no `from_*`; `BBOX_SCHEMA` stays the stored form |
| `clip(width=, height=)` | per-row sizes, so `this`-free expressions from `image_info()` |
| `scale(sx=, sy=)`, `translate(dx=, dy=)` | mirror the contour/point accessors' names and `on_null` |
| `to_absolute`/`normalize(width=, height=)` | as `.point` and `.contour` have |
| `pad(px=)` / `expand(ratio=)` | the "crop with context" case |
| `to_contour()` | bridges to the 35 contour accessors and to `rasterize` |
| `nms(scores=, iou_threshold=)` | greedy, per list; reuse `pairwise::bbox_iou_matrix` |

Then a `crop(box=pl.col("bbox"))` convenience on `Pipeline` as `@_sugar`
over `crop`, declaring the op it lowers to. It takes one box per row (a
`BBOX_SCHEMA` struct, not the set), and `crop` takes `u32` offsets, so it
states its rounding: the smallest pixel window containing the box (floor
`x`/`y`, ceil the far edges), lowered to `crop_clip` if the box may leave
the image. Decide which, and say it in the docstring.

Resolved first: CR-76 is closed. The bbox reader every accessor goes
through refuses a negative size and reads a zero one, so a new accessor
inherits that answer and needs no check of its own.

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
same plan both ways. `hoist_spatial_windows` (`polars-cv/src/passes.rs:155`)
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
   A crop out of bounds is still refused, and its message still names the
   user's crop and window, not the inserted trim (add the case to
   `test_message_hygiene.py`).
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

## CR-118 — CSE in Rust  (existing entry; not scheduled)

**Re-verified: the ledger already holds this, with the option and its cost.**

The review recommended moving common-subexpression elimination out of
Python. CR-118 says the same and marks it **decision**: it "moves the
optimiser across the FFI and changes the wire again". Nothing in the review
changes that analysis. Two additions:

- Measured benefit of CSE as it stands: three outputs from one graph ran in
  400 ms against 782 ms with optimisation off. It works; this is about where
  it lives, not whether.
- CR-141 is decided (2026-11-13): nothing is stable until 1.0, so the wire
  may change freely and the cost CR-118 names is only the work. The owner's
  priority is that CSE as it stands is correct; this is not scheduled.

No new ledger entry. If it is taken up, the plan is the option
CR-118 already describes, with one guard added: the eager/lazy parity sweep
(`tests/test_append_contract.py`) and the CSE interaction tests
(`test_cse_interactions.py`, `test_cse_optimization.py`) run unchanged
against the Rust implementation before the Python one is deleted.
