# AGENTS.md — Geometry Subsystem (`polars_cv.geometry`)

> Read the [root AGENTS.md](../../../../AGENTS.md) and [Python API AGENTS.md](../AGENTS.md) first.
> Update this file when you change schemas, geometry namespaces, or validation.

## Purpose

This subpackage defines the **structured geometry types** (points, contours, bounding boxes) and **expression namespaces** (`.point`, `.contour`) for operating on them within Polars DataFrames.

Geometry data is represented as Polars Struct columns with well-defined schemas. Operations are implemented as Polars plugin functions (Rust-backed) registered via expression namespaces.

## Key Files

| File | Responsibility |
|------|---------------|
| `__init__.py` | Re-exports schemas and `BBoxNamespace` |
| `bbox.py` | `BBoxNamespace` (`.bbox`) — pairwise IoU, match detections for bounding boxes |
| `schemas.py` | Schema constants (`POINT_SCHEMA`, `CONTOUR_SCHEMA`, `CONTOUR_SET_SCHEMA`, `CORRESPONDENCE_SCHEMA`, `BBOX_SCHEMA`, etc.), validation helpers, factory functions |
| `contours.py` | `ContourNamespace` (`.contour`) — area, perimeter, centroid, bounding_box, IoU/Dice/Hausdorff, set-level correspondence (`pairwise_iou`, `correspond`), and heatmap scoring (`label_reduce`) |
| `points.py` | `PointNamespace` (`.point`) — normalize, to_absolute, translate, scale, rotate, distance, angle_to, etc. |

## Schemas

All geometry data uses Float64 coordinates. This is deliberate — it avoids precision issues and matches the Polars Struct system naturally.

### Coordinate System

- **Origin:** Top-left of image
- **X:** Increases rightward
- **Y:** Increases downward
- **Normalized:** [0, 1] range relative to image dimensions

### Schema Definitions

| Schema | Structure | Usage |
|--------|-----------|-------|
| `POINT_SCHEMA` | `Struct({x: Float64, y: Float64})` | Single 2D point |
| `ANNOTATED_POINT_SCHEMA` | `Struct({x, y, label: String, confidence: Float64})` | Point with metadata |
| `POINT_SET_SCHEMA` | `List(POINT_SCHEMA)` | Multiple points |
| `RING_SCHEMA` | `List(POINT_SCHEMA)` | Ordered closed ring of points |
| `CONTOUR_SCHEMA` | `Struct({exterior: RING_SCHEMA, holes: List(RING_SCHEMA), is_closed: Boolean})` | Polygon with optional holes |
| `CONTOUR_SET_SCHEMA` | `List(CONTOUR_SCHEMA)` | Multiple contours |
| `BBOX_SCHEMA` | `Struct({x, y, width, height: Float64})` | Axis-aligned bounding box |

### Holes and Winding

Hole-ness is **structural**: a ring is a hole because it sits in `CONTOUR_SCHEMA`'s
`holes` field, never because of how it is wound. Every operation — area, centroid,
`contains_point`, IoU, Dice, rasterization — is winding-independent, so `flip()` and
`ensure_winding()` change what `winding()` reports without changing the region the
contour describes. Do not reintroduce a "CW means hole" rule: an earlier version of
this doc stated one, no code ever honoured it, and the one place that accidentally
depended on winding (the old Sutherland-Hodgman IoU clipper) returned 0.0 for a
CW contour matched against itself.

Winding is **computed from point order** (the sign of `geo`'s signed area), not stored:
- Counter-clockwise (CCW) = positive signed area
- Clockwise (CW) = negative signed area

`is_closed` is reserved — written unconditionally as `true`, never read back. Rings
are implicitly closed; the first point is not repeated.

## Expression Namespaces

### `.contour` (ContourNamespace)

Registered on `pl.Expr` for columns matching `CONTOUR_SCHEMA`. Each method calls `_plugin.call` (via `_PluginNamespace._plugin`) with a specific Rust function name (e.g., `contour_area`, `contour_iou`).

Set-level detection helpers also live here and operate on `CONTOUR_SET_SCHEMA`:
- `pairwise_iou(other)` -> `List[List[Float64]]`
- `correspond(other, threshold, order)` -> `CORRESPONDENCE_SCHEMA`
- `label_reduce(heatmap, reduction, region_mode)` -> `List[Float64]`

### `.bbox` (BBoxNamespace)

Registered on `pl.Expr` for columns containing `List[BBOX_SCHEMA]`. Methods:
- `pairwise_iou(other)` -> `List[List[Float64]]`
- `correspond(other, threshold, order)` -> `CORRESPONDENCE_SCHEMA`

These delegate to Rust functions `bbox_pairwise_iou` and `bbox_correspond`
which internally convert bounding boxes to rectangular contours and reuse the
existing contour matching logic. Used by `BBoxMatcher` in the metrics subsystem.

### `.point` (PointNamespace)

Registered on `pl.Expr` for columns matching `POINT_SCHEMA`. Each method calls `_plugin.call` (via `_PluginNamespace._plugin`) with a specific Rust function name (e.g., `point_normalize`, `point_distance`).

### Important: These bypass the pipeline/graph system

Point and contour namespace operations go directly through `_plugin.call` to dedicated Rust functions. Every accessor also accepts the `PointType`/`ContourType`/`BBoxType` extension types: `_plugin.call` hands the plugin `.ext.storage()`, so a tagged column computes exactly as its plain struct (`test_accessors_accept_tagged_inputs`). Accessors that work on the struct in Python (`.point.x`/`.y`) must read `.ext.storage()` themselves. They do **not** go through the `vb_graph` pipeline path. This is a design distinction — they operate on Struct columns directly rather than on binary image data.

### Parameter policy: one typed definition per function, like every op

Each plugin function is described the way an op is: a variant of a
mode-generic family in `polars-cv/src/geom_fns.rs` (`ContourFn`, `PointFn`,
`BBoxFn`) with the wire's fields — `M::V<T>` for a value that may be per-row,
`ColumnRef` / `Option<ColumnRef>` for a data operand — its doc comment as the
Python docstring, and its defaults declared once with `#[param(default = ...)]`.
A function that is also a pipeline op (`.contour.area`, `translate`, `scale`,
…) *is* that `GeometryOp` variant (`geom_fns::OP_ACCESSORS`), so the two
surfaces cannot drift in fields, defaults or docs.

Every accessor method is generated (`scripts/gen_ops.py`, from
`tests/golden/geom_catalog.json`) as one `self._call("<fn>", {...})`.
`_GeomNamespace._call` (`_namespace.py`) appends each `pl.Expr` as a plugin
input and writes its position (`{"$slot": n}`) into its field; a literal is
the value itself. It asks the plugin to parse the literals against the
definition as the expression is built (`_lib.check_geom_call`), so a
misspelled enum raises `ValueError` where it was written. In Rust,
`GeomParams::parse` reads the call's arguments strictly as the function's own
definition and checks that every input is claimed exactly once; each row
resolves a field through `params.value(field, row)` and `params::ParamCol`, so
these namespaces share the graph engine's dtype coverage, scalar broadcasting
and null policy.

**Operands are read through their references, never by position.** Optional
operands (`correspond`'s `order`, `point.rotate`'s `origin`) and per-row
parameters both occupy input slots; each field names its own
(`params.column(field)` / `params.optional_column(field)`).

Every enum parameter here is per-row capable (none changes an output schema);
`test_non_structural_geometry_enums_accept_an_expression` reads the catalogue.

Validation that can no longer happen once per batch moves into the row loop and
names the offending row — see the `threshold` range check in
`contour_correspond` and the zero-dimension guard in `point_normalize`.

## Adding a Geometry Operation

1. **Definition:** add a variant to `ContourFn`/`PointFn`/`BBoxFn` in
   `src/geom_fns.rs` — `#[op(name = "<fn>", python = "<method>", sample =
   {...})]`, a doc comment (the docstring, with its `Returns:`), a doc comment
   per field, and `#[param(default = ...)]` where the method has a default.
   (If the method is a pipeline op, add it to `OP_ACCESSORS` instead.)
2. **Function:** add the `#[polars_expr]` function in `src/contour.rs` or
   `src/point.rs` (a `contour_accessor!` arm for a contour one), parsing its
   definition with `GeomParams::parse` and reading each field per row.
3. **Regenerate:** re-bless the catalogues (`POLARS_CV_BLESS=1
   scripts/with-pyo3-env.sh cargo test -p polars-cv catalog_matches`), run
   `scripts/gen_ops.py` and `scripts/gen_signature_snapshot.py`, and
   `maturin develop`. The Python method appears on the namespace; write none by
   hand.
4. **Tests:** Add to `tests/test_contour_plugin.py` or create a reference test. For a per-row parameter, assert two rows with *different* values produce *different* outputs (`tests/test_expression_params.py`) — a call that merely succeeds cannot distinguish "resolved per row" from "silently dropped"

## Schema export policy

- `ANNOTATED_POINT_SCHEMA` is a deliberately geometry-only export (`polars_cv.geometry.ANNOTATED_POINT_SCHEMA`): it describes an internal structure, unlike the user-facing geometry schemas re-exported at the package top level. Import it from `polars_cv.geometry` when needed. `CORRESPONDENCE_SCHEMA` is *not* in that category and is exported at the top level: keeping the old match-result schema geometry-only is why the metrics matchers read its fields by string literal instead of from the declaration.

## Arity: one contour or a set of them

A `.contour` column carries either a `CONTOUR_SCHEMA` struct per row or the
`CONTOUR_SET_SCHEMA` list of them that `extract_contours()` produces, and
**every accessor takes both**. The rule:

| input | result |
|-------|--------|
| `CONTOUR_SCHEMA` | the element type (`Float64`, `POINT_SCHEMA`, a contour, …) |
| `List(CONTOUR_SCHEMA)` | `List(<element type>)`, one entry per contour, in input order |

Two-operand accessors (`iou`, `dice`, `hausdorff_distance`) **broadcast**: a set
on one side and a single contour on the other gives one result per contour,
whichever side the set is on. A set on *both* sides **raises** — it could mean
the N×M matrix (`pairwise_iou`) or an index-wise pairing (`.explode()` one
side), and guessing between two different answers is the fallback behaviour this
codebase removes. The set-level accessors (`pairwise_iou`, `correspond`,
`label_reduce`) run the same rule backwards: a lone contour is read as a set of
one, via `parse_contour_set`.

### Why it is a mechanism and not a per-accessor `if`

Each accessor has two halves that must agree — the `output_type_func` (the dtype
published at plan time) and the body (the Series produced) — and nothing in
`#[polars_expr]` forces them to. Fifteen hand-written `output_type=Float64`
attributes would have been fifteen chances to declare `Float64` and build
`List(Float64)`.

So the arity is **one value, read from the column dtype** (never from a row —
`output_type_func` only sees `Field`s, so a row-level decision is one the
declaration could not have made), and `src/geom_arity.rs` drives both halves
from it:

- `Arity::of` reads it, using `point_dtype_fields()` — the same field names the
  point parser reads, so the dispatch cannot admit a struct the parser rejects.
- `elementwise_field` / `binary_field` wrap the element type for the declaration.
- `map_contours` / `zip_contours` wrap the results
  with the same `Arity::wrap`, and are the only decode path the accessors use.
- `contour_accessor!` emits both halves from a single `-> <elem>` declaration.

`map_contours` also owns the null-parameter policy: it wraps each
*row* in `GeomParams::row`, so `on_null("null")` nulls the row rather than each
contour. That is the job `contour_row` used to do, moved so it cannot be
forgotten.

`contour_contains_point` is the one accessor with its own loop: its second
operand is a point, so neither the `map` arm (one operand) nor the `zip` arm
(two contour operands) describes it. It still reads `Arity::of` and wraps
through `elementwise_field`/`pack_row`, so only the loop is local.

### Adding an accessor

Use a `contour_accessor!` arm — `map` or `zip`, each naming the definition it parses. Do not write a
bare `#[polars_expr(output_type=...)]` for a contour accessor: the case table in
`tests/test_schema_parity_namespaces.py` is completeness-asserted against the
namespace's real methods *and* swept in both arities, so an accessor that skips
the macro fails `test_contour_accessors_over_a_contour_set` rather than shipping
a schema its data contradicts.

`.point` has the identical single-only limitation over `POINT_SET_SCHEMA`;
`geom_arity.rs` is written to fit it, but wiring it up is not done.
