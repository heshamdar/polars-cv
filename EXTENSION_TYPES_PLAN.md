# Plan: first-class Arrow extension types — remaining work

Scope: the numpy/torch sink struct (`polars_cv.ndarray`) and the geometry
family (`polars_cv.point` / `.contour` / `.bbox`).

## Status

Shipped (0.29.0; see the CHANGELOG):

| Phase | What landed |
|---|---|
| 1 — Foundation | `ExtType` in Rust, `NdArrayType`/`PointType`/`ContourType`/`BBoxType` registered at `import polars_cv`, `_plugin.call` the only way into the plugin |
| 2 — `sink("ndarray")` | the numpy sink struct tagged `polars_cv.ndarray`; `sink("numpy")` unchanged |
| 3 — Tagged inputs | every accessor and `vb_graph` source accepts a tagged column; loose "looks-like" contour matching removed |
| 6 — Spike deleted | — |

How it works, and why tagging must stay opt-in (tagging breaks polars'
struct operations), is in `polars-cv/python/polars_cv/AGENTS.md`
("Extension Types").

## Remaining

### Phase 4: Opt-in tagged geometry output
- Geometry-producing paths (contour extraction `sink("native")`, bbox/point
  constructors) gain a tagged form. Prefer a **distinct sink format or
  constructor** over a global config flag:
  - a flag would have to reach every plugin call and enter op identity;
  - `CLAUDE.md` rejects per-call policy keywords that alter the output schema.
- Candidates: `sink("contour")` for tagged contours, and `polars_cv.point(x, y)`
  / `polars_cv.bbox(...)` constructors as zero-copy `.ext.to()` relabels.
  Users can already tag with `.ext.to(PointType())`.
- Tests mirror `tests/test_ndarray_sink.py`, plus metrics end-to-end on
  tagged inputs.

### Phase 5: Defaults (major version; separate decision)
- Decide from Phase 2–4 usage whether `sink("numpy")` and the untagged
  geometry outputs become tagged by default, or stay as the struct
  interchange formats. Recommendation: **keep both permanently** — `numpy`
  stays the plain-struct interchange format and `ndarray` the typed one,
  which avoids a flag day.
- If a default flips: CHANGELOG migration notes with the struct-operations
  table from the AGENTS.md section.

### Open decision: `polars_cv.ndarray` metadata
Should the type carry metadata (`{"dtype": "uint8", "ndim": 3}`)?
- The planner knows the dtype and rank at plan time, so the schema could
  carry them, enabling type-level dispatch and checks before any data flows,
  much as `Array` does for shape.
- Cost: metadata is part of dtype equality, so columns with different dtypes
  stop concatenating, and the factory must parse and validate metadata.
- Recommendation: stay without metadata; adding it later is a dtype change,
  so decide before Phase 5.

### Upstream
- Report polars' `register_extension_type` duplicate check (it tests the
  literal `"ext_name"`).
- Report that `ext_from_params` raising panics polars instead of surfacing
  the error.

### Risks to keep watching
| Risk | Mitigation |
|---|---|
| Upstream API changes (unstable) | one module per side; pinned tests |
| Polars 2.0 flips the unknown-type default to `load_as_extension` | test both behaviours via `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR` |
| Users break on struct operations | tagging opt-in until Phase 5; `.ext.storage()` in docs and examples |
