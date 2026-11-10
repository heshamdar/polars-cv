# Metrics separation: review follow-up plan

Working checklist for the fixes found in the review of
`claude/metric-layer-separation-e5ksp3`. Work happens on
`ccr-647b8b64-2a5wf5`, which is fast-forwarded onto that branch's tip
(`2f68562d`). **This file is deleted in the last step; it must not be merged.**

Rules for every phase:

- Test or guard first, watched failing for the reason it claims, then the change.
- A moved thing is deleted at its old location in the same commit. No
  re-export, alias or shim left behind "for now".
- One commit per phase (or per lettered step where noted), each green on its own.
- Tick a box only after the stated check has been run and read in full.

## Decisions (recommended defaults; change here before the phase starts)

- **D1. Layout.** CV code moves to `metrics/_cv/`. Tier = path: everything
  under `metrics/` outside `_cv/` is generic, except the facade
  `metrics/__init__.py`. `CV_ADAPTERS` is deleted.
- **D2. `MATCH_SCHEMA` policy.** Keep the exact dtype check in
  `matched_table` (its producers are matcher code, which can cast), keep
  `from_matched` names-only (its frames are caller-assembled), and state that
  distinction in both places. Alternative: names + strict cast in
  `matched_table`, matching `from_matched`.
- **D3. `FloatOrExpr`.** Export core's alias from `polars_cv` and delete the
  metrics copy. Alternative: keep a metrics-owned alias and say why.
- **D4. `ty`.** Fix the renamed ignore comments and pin the `ty` version in
  the three places that run it. It fails on `main` too; done here as its own
  commit so it can be cherry-picked.

## Phase 0: baseline

- [x] Working branch fast-forwarded to `2f68562d`.
- [x] `maturin develop` (debug); `polars_cv.build_info()` hashes agree.
      Repeat after every session resume: the SessionStart hook's
      `uv sync --no-install-project` uninstalls the editable package, and the
      subprocess-import tests then fail with `No module named 'polars_cv'`
      (seen once already: 56 failures in the pre-commit structural lane).
- [x] Baseline recorded: default pytest lane count (was 7716 passed, 34
      skipped, 4 xfailed), `ruff`, `mkdocs build --strict`.

## Phase 1: `ty` (D4), own commit, **first**

The pre-commit `ty` hook exits 1 on `main` and here, so no commit can land
until this is done (`--no-verify` is not an option). This file is committed
together with this phase.

- [x] Rename the ignore comments (7 sites: the 6 below plus
      `tests/test_engine_no_panics.py:49`; `ty` now exits 0): 5 × `unresolved-attribute` →
      `missing-attribute` (`_inputs.py:430`, `_bbox.py:125`,
      `_contour.py:154`, `_contour.py:849`, `_segmentation.py:81`), 1 ×
      `invalid-method-override` → `incompatible-method-override`
      (`_metrics/_precision_recall.py:73`). Grep the whole repo for other
      uses of both old names.
- [x] Pin `ty` to the version that passes, identically in
      `scripts/verify.sh:166`, `.github/workflows/ci.yml:313` and
      `.pre-commit-config.yaml:47`. Check whether an existing guard
      (`test_structural_lane.py`, `test_build_efficiency.py`) pins these
      command strings and update it rather than leave it stale.
- [x] `uvx ty@<pin> check --project polars-cv` exits 0 with 0 diagnostics.

## Phase 2: guard fixtures assert the reason (before the guard is rewritten)

- [x] `test_metrics_boundary_fixtures.py`: `BAD` maps each case to the
      message fragment it must produce (`core imports`, `depends on
      polars-cv`, `imports a CV adapter`, `accessor`, `not in … __all__`,
      `not a public surface`, `import polars-cv absolutely`, `CV_ADAPTERS
      names`). `test_scanner_rejects` asserts that fragment is in a reported
      line, and that no *other* rule fired.
- [x] Watched failing: swap two cases' expected fragments, see both fail,
      swap back.
- [x] Same treatment for the two layering fixtures this branch added
      (`test_import_layering_fixtures.py`: downstream BAD/GOOD), if that
      file's existing cases can take it without a wider rewrite; otherwise
      note it and leave the file's convention alone. *Done as one targeted
      test (the re-exporting root is reported as a cycle); the file's other
      cases keep their convention.*

## Phase 3: tiers become layout (D1): the main change

### 3a. New run-time guard, written first

- [ ] New test (`test_metrics_boundary.py`):
      `test_the_generic_tier_runs_without_polars_cv`. In a subprocess: copy
      every generic module (tier read from the scanner, not a second list)
      into a temp package under another name, make `polars_cv` unimportable
      (`sys.modules["polars_cv"] = None`), import every copied module, then
      run a smoke path (`PreMatchedAdapter().match(...)` → `froc_auc`,
      `average_precision`) and compare against values computed normally.
- [ ] Watched failing on the current tree for the right reason (a generic
      module's package `__init__` pulls the CV matchers).

### 3b. Move generic code out of the CV files

- [ ] `_confidence_order` → `_matching/_table.py` as `confidence_order`
      (it produces the order aligned with the scores the tail reads).
      Deleted from `_contour.py`; `_bbox.py`'s `from ._contour import
      _confidence_order` deleted.
- [ ] `_reports.py` split: `ReportMetric`, `DetectionReport`,
      `_materialized` stay in generic `_reports.py`;
      `evaluate_detections` / `evaluate_heatmaps` move to `_cv/`. The
      `TYPE_CHECKING` `BoxFormat` import leaves `_reports.py`.
- [ ] `_inputs.py` split: `group_objects`, `GT_ROW`, `PRED_ROW`,
      `_require`, `_image_ids` stay generic; `match_detections`,
      `_image_columns`, `_geometry_kind`, `_kind_of`, `_instance_outlines`
      and the `polars_cv` imports move to `_cv/`. Confirm each helper's tier
      by what it actually calls, not by this list.
- [ ] `_segmentation.py` split: `SegmentationReport` (and `MEASURES` if the
      report reads it) generic; `segmentation_measures`,
      `evaluate_segmentation` to `_cv/`.

### 3c. Create `metrics/_cv/` and move the CV code

- [ ] `_cv/__init__.py`, `_cv/_contour.py`, `_cv/_bbox.py`, plus modules for
      the detection entry points and segmentation moved in 3b (names chosen
      when the split is visible; one module per concern).
- [ ] `_matching/__init__.py` exports only `Matcher` and
      `PreMatchedAdapter`; it imports nothing from `_cv`.
- [ ] `metrics/__init__.py` (the facade) is the only module outside `_cv/`
      that imports `_cv`; its `__all__` is unchanged (diff it: same 31+
      names, same objects).
- [ ] Function-level imports of `ContourMatcher` / `BBoxMatcher` / the
      `_contour` helpers hoisted to module level inside `_cv/`
      (`_inputs.py:267`, `_inputs.py:416`, `_reports.py:544` today). If one
      really needs deferring, a comment says which cycle.
- [ ] **Deleted, not left behind:** `_matching/_contour.py`,
      `_matching/_bbox.py`, the moved functions in `_inputs.py` /
      `_reports.py` / `_segmentation.py`, and any now-unused imports.
      `git grep` for each old dotted path returns nothing.

### 3d. Rewrite the scanner to read the layout

- [ ] `_metrics_boundary.py`: `CV_ADAPTERS` deleted; `CV_PACKAGE =
      "polars_cv.metrics._cv"`; generic = downstream, not under
      `CV_PACKAGE`, not the facade (defined as the downstream package's own
      `__init__`, not by name in a set). The "adapter list names a missing
      module" rule and its fixture go with the list.
- [ ] Rule (b) additionally: a generic module does not import `CV_PACKAGE`
      (replaces "imports a CV adapter").
- [ ] `violations()` signature, fixtures (`ADAPTERS` → the synthetic
      package gets a `metrics._cv`), and the docstring's rules and limits
      updated. The limit about a generic module inside an adapter
      subpackage is deleted: it no longer exists.
- [ ] Phase-2 reason fragments updated; every BAD still fails for its
      stated reason, every GOOD still passes.

### 3e. Everything that names the old paths

- [ ] Tests: `test_matchers.py:19,545`, `test_metric_fixes.py:45`,
      `test_dtype_spellings.py:280,294`, `test_match_schema.py:14`,
      `test_contour_source.py:99` (prose).
- [ ] Docs: `AGENTS.md:309` and the boundary row of the canonical-paths
      table, `CLAUDE.md` module table, `metrics/AGENTS.md` (Boundary
      section, layer list, file tree), `polars_cv/AGENTS.md`.
- [ ] Other path-keyed mechanisms, each checked rather than assumed:
      `benchmarks/regression` relevance rules, `tests/_join_ratchet.py`,
      `test_engine_choice.py`, `test_no_python_udfs.py`, `_discovery.py`,
      `mkdocs.yml` / `docs/api/*` `:::` targets, `examples/`.
- [ ] 3a's test now passes; default pytest lane count is baseline + new
      tests, with no new skips; `mkdocs build --strict`; `ruff`.

## Phase 4: `MATCH_SCHEMA` (D2)

- [ ] Docstrings state the two policies and why they differ: `_check_match`
      / `MATCH_SCHEMA` (exact; producers are matcher code) and
      `_validate_schema` (names only; caller-assembled frames), each
      pointing at the other.
- [ ] `explode_matches` reads its dtypes from `MATCH_SCHEMA`: the
      `pl.lit(None, dtype=pl.UInt32)` / `0.0` / `False` fill literals take
      the field's inner dtype, and the redundant `.cast(pl.UInt32)` on
      `_gt_idx` is deleted (the check already guarantees it). The
      `COL_GT_IDX` output dtype is still asserted by an existing test; if
      none does, add one.
- [ ] Tie test at the entry point, in `test_match_schema.py`: for
      `.contour.correspond`, `.contour.correspond_by_coverage` and
      `.bbox.correspond`, the *lazy* schema of the expression equals
      `MATCH_SCHEMA`. Watched failing by temporarily changing one field of
      `MATCH_SCHEMA`.
- [ ] `_check_match` runs once, not once per threshold, if the threshold
      cannot change the dtype; otherwise leave it and say so in the
      docstring.
- [ ] CHANGELOG and the commit claim corrected: a signed `right_idx` gave
      correct results (or a cast error on a negative), so the entry says the
      check moves the refusal to plan time, not that it fixes a misread.
- [ ] `metrics/AGENTS.md`: say plainly that polars-cv's matchers are today's
      only in-package producers and `PreMatchedAdapter` builds its table
      through `DetectionTable.from_matched`.

## Phase 5: remaining guard gaps

- [ ] Accessor rule: in a generic module, *any* attribute named after a
      registered namespace is reported (catches `ns = expr.contour`).
      Fixtures: bound-to-a-name BAD, Polars' own namespaces GOOD. Limits
      docstring updated (`getattr` still invisible).
- [ ] Rule (c): a name imported from a public surface must not have a
      private attribute read off it or off a call of it (`Pipeline._x`,
      `Pipeline()._x`). Fixtures both ways. Stated limit: an instance bound
      to another name is not followed.
- [ ] `is_downstream` deleted; the scanner uses one predicate exported from
      `_import_layering` (rename `_consumer` to a public name). The
      `(DOWNSTREAM,) = _CONSUMERS` unpacking gets a message if it ever holds
      more than one.
- [ ] Each new rule watched failing on a mutation of the real tree, as in
      the review probes.

## Phase 6: leftovers

- [ ] `FloatOrExpr` (D3): added to `polars_cv.__all__` and the API docs (and
      whatever test requires every exported name to be documented);
      `_cv/_contour.py` imports it; the local alias and its `TypeAlias`
      import are deleted.
- [ ] `_matching/_prematched.py`: dead `if TYPE_CHECKING: pass` and the
      then-unused `TYPE_CHECKING` import deleted.
- [ ] `geometry/AGENTS.md:163`: the sentence justifying
      `CORRESPONDENCE_SCHEMA`'s top-level export by the metrics matchers
      reading it is rewritten; they do not import it.
- [ ] `examples/06_detection_metrics.py`: `COL_CLASS_ID` / `COL_IMAGE_ID`
      come from `polars_cv.metrics._types` (private). Either export them
      from `polars_cv.metrics` or use the literal column names; no private
      import in an example.

## Phase 7: close out

- [ ] CHANGELOG (Unreleased): the layout change, the corrected
      `MATCH_SCHEMA` wording, `FloatOrExpr` exported, `ty` pinned.
- [ ] Review findings re-read one by one against the tree; each is either
      fixed or has a written reason here.
- [ ] `scripts/verify.sh` from the repo root prints PASS (all lanes,
      including the Rust checks and `ty`); output read unfiltered.
- [ ] `examples/06_detection_metrics.py` runs.
- [ ] Scratch files removed; **this plan file deleted** in the final commit.
- [ ] Pushed to `ccr-647b8b64-2a5wf5`. No PR unless asked.
