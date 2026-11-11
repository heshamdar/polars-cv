# Phase 2 — Stabilise and smooth

Implementation plan for the second group of findings from the external review
of 0.41.0 (commit `c0d77d5`, 2026-11-11). Ledger entries CR-141 to CR-145 in
the root [`ISSUES.md`](../../ISSUES.md). Follows
[phase 1](phase-1-first-contact.md); [phase 3](phase-3-depth.md) follows this.

**Goal:** an API a team can build on, and the same care at the edges of a
pipeline (sinks, messages, docs) that the ops already have.

**Order:** CR-145 (a one-line change, verified), CR-143, CR-142, CR-144,
then CR-141, which is a decision and frames everything after it. CR-142
changes a public signature, so land it *before* any stability promise.

The working rules at the top of phase 1 apply here unchanged.

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
2. **Hold until phases 1 and 2 land, then declare.** The remaining known
   signature changes are CR-136 (wire only), CR-138 (additive), CR-137
   (additive) and CR-142 (`sink()`); after them, option 1.
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
  hand-written.
- README "Project status" rewritten to say what is stable.

---

## CR-142 — `sink()` is hand-written; `source()` is generated

**Re-verified: confirmed. The catalogue already has what generation needs.**

### Evidence

`io_catalog.json` describes every sink with typed fields:

| Sink | Fields |
|---|---|
| `array` | `shape: list[int] \| None` |
| `jpeg` | `quality: int` |
| `long` | `names: list[str]`, `where: str \| None` |
| `numpy`, `torch`, `ndarray` | `dtype: SinkDType \| None`, `compact: bool` |
| `blob`, `list`, `native`, `png`, `tiff`, `webp` | none |

`scripts/gen_ops.py` already emits the `SinkFormat` enum from it (line 591)
and the whole of `source()` with a typed keyword per field. `sink()`
(`lazy.py:224`) takes `format: str | dict[...]` and `**kwargs: Any`. Rust
validates every key, so nothing is silently accepted; the cost is that
editors and `ty` see nothing, and the multi-output form is a dict of
`(format, {params})` tuples.

### Approach

1. Generate `sink()`'s single-output signature from the catalogue exactly as
   `source()` is: `format: SinkFormat | str = "native"`, then one
   keyword-only parameter per field across all sinks, defaulting to `None`,
   with the docstring naming the formats each applies to. A keyword that
   does not apply is still refused by Rust; the signature only makes the set
   visible.
2. Keep `return_expr` and `opt_flags` hand-written around the generated part,
   or move graph inspection off `sink()` (see CR-148 in phase 3, which makes
   `return_expr=False` unnecessary).
3. Multi-output: replace the tuple form with a small generated value, one
   constructor per sink (`Sink.array(shape=[...])`, `Sink.jpeg(quality=85)`),
   accepted wherever a format string is. The dict stays; its values become
   `str | Sink`. Keep the tuple form working for one release under the
   deprecation path from CR-141.
4. `OP_FIELDS`-style table for sinks (`SINK_FIELDS`), so the Python side
   reads field types from the generated authority and
   `set_output`/`set_multi_output` in `_graph.py` stop carrying their own
   notion of which keys exist.

### Tests first

- Signature parity: every catalogue sink field is a `sink()` keyword and a
  `Sink.<format>` argument, and nothing else is. Watch it fail by adding a
  field to a sink in Rust without regenerating.
- `test_the_committed_catalog_is_the_built_one` already covers freshness;
  extend it to the sink section.
- Behaviour: every existing `sink(...)` call in `tests/` passes unchanged.
- `ty check` over an example file that passes a wrong keyword type must
  fail (run once by hand; it is the point of the change).

### Risks

- `sink()` is the most-called method in user code. The single-output call
  shape must not change; only the multi-output tuple form is deprecated.
- A parameter named the same on two sinks with different types would not fit
  one flat signature. None exists today; make the generator refuse one.

### Done when

`sink()` has no `**kwargs`, and the canonical-paths table in `AGENTS.md`
lists sinks beside sources.

---

## CR-143 — Messages that leak internals

**Re-verified: confirmed, with exact sites.**

### Evidence and fix per case

| Case | Today | Site | Fix |
|---|---|---|---|
| Corrupt or unsupported image | `Failed to decode image: Unsupported(UnsupportedError { format: Unknown, kind: Format(Unknown) })` | `execute.rs:64` formats `{:?}` | `{}`; `ImageError` implements `Display`. Same for `execute.rs:120` (encode). |
| Wrong sink name or option | correct diagnosis followed by `at line 1 column 203` | `graph/types.rs:367` prints the `serde_json` error | Strip the position: the wire is machine-written, so a line and column mean nothing to the user. Keep the path (`serde_path_to_error` already supplies it for ops). |
| `.cv.pipe(Pipeline().resize(...))` with no source | `root node 'node_e16ec35b' has no source`, raised at `.sink()` | `plan.rs:874` | Raise in `CvNamespace.pipe` (`expressions.py`), where a root is created: a pipeline given to `.cv.pipe` reads a column, so it must have a source. Message: "this pipeline has no source; start it with `.source()`". Continuations (`LazyPipelineExpr.pipe`) are unaffected. |
| `LazyPipelineExpr` used as an expression | Polars' "cannot create expression literal for value of type LazyPipelineExpr" | Polars | Cannot be intercepted cleanly. Document it in the quickstart's troubleshooting, and check whether Polars 2.x consults a protocol on unknown objects before adding anything. Do not monkey-patch. |
| Null in a parameter column | names the column and row | `params.rs` | Append the two remedies: `Pipeline.on_null_param("null")` or `fill_null` on the column. |
| Any runtime failure in a multi-pipeline query | Polars prints `col("img").ext.storage()()./path/_lib.abi3.so:vb_graph()` | Polars | Prefix the plugin's own message with the output alias(es) of the failing graph, which the graph already knows. |

The remaining `{:?}` in error strings (`graph/decode.rs:35,55,353`,
`lib.rs:384`) print Polars `DataType`s, whose `Debug` is the readable form.
Leave them.

### Tests first

`tests/test_error_messages.py` already pins message text. Add one case per
row: assert the new text is present and that `UnsupportedError {`,
`at line` and `node_` are absent from user-facing messages. A general guard
is worth having here: a test that runs a table of deliberate failures and
rejects any message matching `\w+ \{ \w+:` (a Rust `Debug` struct) or
`line \d+ column \d+`. Give it fixtures, as the other scanners have.

### Risks

Tests and downstream code that match on message text. `ResolveError`
replaced text matching inside the plugin (CR-130); these are the
user-visible strings, so list each change in the CHANGELOG.

---

## CR-144 — Documentation gaps and drift

**Re-verified against the docs tree.**

### What is missing

1. **Which sink to use.** `concepts/pipelines.md` has a table describing each
   sink and nothing on choosing. Add a short decision section: fixed shape
   and a batch wanted → `array`; ragged → `list`; per-row tensors for a
   loader → `numpy`/`torch`; persisting → `ndarray` with `compact=True` or
   an encoder; feeding another pipeline → `blob` (and, after CR-138, the
   tensor sinks).
2. **Training-time augmentation.** The only mention is one sentence under
   `warp_affine`. Add a guide: seeded per-row parameters as Polars columns
   (`pl.int_range`, a hash of row id and epoch, or a NumPy generator
   materialised into columns), feeding `crop`, `flip(when=)`, `rotate`,
   `adjust_contrast`, `warp_affine`; a random-resized-crop written out; and
   how an epoch maps to a query. Say plainly what this is not: it does not
   replace a shuffling loader.
3. **EXIF orientation** (with CR-137).
4. **Troubleshooting**: the forgotten `.sink()` and forgotten `.source()`
   messages.

### What has drifted

- `AGENTS.md` principle 6 and its architecture diagram say schema inference
  and validation live in Python. They live in Rust (`plan::step`). Rewrite
  both to describe what is true: Python appends, Rust plans.
- The quickstart says "naming the source format is optional". The call is
  required; only its argument is optional. One sentence.
- `benchmarks/README.md` reports 0.38.0 (see CR-149, phase 3).

### Tests first

`mkdocs build --strict` checks links and nav only; code blocks in the docs
are not executed. `tests/test_examples.py` runs every script in `examples/`.
So back each new guide with a numbered example (`14_augmentation.py`,
`15_choosing_a_sink.py`) that the guide quotes, so an API change breaks the
example in CI rather than leaving the guide stale.

---

## CR-145 — The `image` crate's default features

**Re-verified by building it: confirmed, with one correction.**

### Evidence

`view-buffer/Cargo.toml` takes `image = { version = "0.25" }`. Its defaults
are `rayon` plus fifteen formats, one of which, `avif`, is encode-only
(`ravif`, which pulls `rav1e`). No sink reaches it: the sinks are PNG, JPEG,
WebP and TIFF, and AVIF *decoding* (`avif-native`) is not enabled, so an AVIF
file is refused.

Tried on this checkout:

```toml
image = { version = "0.25", default-features = false, features = [
    "bmp", "dds", "exr", "ff", "gif", "hdr", "ico", "jpeg", "png",
    "pnm", "qoi", "tga", "tiff", "webp",
] }
```

`cargo check -p view-buffer --all-features --tests` passed (1m 24s) and
neither `ravif` nor `rav1e` was compiled.

**Correction to the review:** `Cargo.lock` does not change (it resolves all
features: still 504 packages), so "one fewer ignored advisory" is not
automatic. Whether `cargo deny` stops seeing `rav1e` depends on the graph it
evaluates; check, and remove the `RUSTSEC-2026-0105` ignore from `deny.toml`
only if it reports the advisory as no longer detected.

### Approach

1. Apply the manifest change above. It keeps every format that decodes
   today, so no behaviour changes.
2. Decide `rayon` separately. The list above drops it; `image` then decodes
   on the calling thread only, which is what a plugin with its own row
   parallelism wants (the same reasoning `geo`'s manifest entry gives for
   dropping `multithreading`). Benchmark EXR and large PNG decode before and
   after, single call and under the row splitter.
3. Decide whether the long tail (`dds`, `ff`, `ico`, `qoi`, `tga`) is
   supported or accidental. If accidental, do not drop it silently: that is a
   behaviour change needing a CHANGELOG entry. If supported, add each to the
   documented list and give it a decode test.
4. Measure the wheel: record the release `.so` size before and after in the
   CHANGELOG entry.

### Tests first

A decode test per format the manifest lists, from a committed fixture, so
the feature list and the supported-formats doc cannot disagree. Watch one
fail by removing its feature.

### Done when

No unreachable codec is compiled, and the docs list exactly the formats the
manifest enables.
