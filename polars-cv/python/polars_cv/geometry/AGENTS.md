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
| `schemas.py` | Schema constants (`POINT_SCHEMA`, `CONTOUR_SCHEMA`, `CONTOUR_SET_SCHEMA`, `MATCH_RESULT_SCHEMA`, `BBOX_SCHEMA`, etc.), validation helpers, factory functions |
| `contours.py` | `ContourNamespace` (`.contour`) — area, perimeter, centroid, bounding_box, IoU/Dice/Hausdorff, set-level matching (`pairwise_iou`, `match_detections`), and heatmap scoring (`label_reduce`) |
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

Registered on `pl.Expr` for columns matching `CONTOUR_SCHEMA`. Each method calls `register_plugin_function` with a specific Rust function name (e.g., `contour_area`, `contour_iou`).

Set-level detection helpers also live here and operate on `CONTOUR_SET_SCHEMA`:
- `pairwise_iou(other)` -> `List[List[Float64]]`
- `match_detections(other, threshold, scores)` -> `MATCH_RESULT_SCHEMA`
- `label_reduce(heatmap, reduction, region_mode)` -> `List[Float64]`

### `.bbox` (BBoxNamespace)

Registered on `pl.Expr` for columns containing `List[BBOX_SCHEMA]`. Methods:
- `pairwise_iou(other)` -> `List[List[Float64]]`
- `match_detections(other, threshold, scores)` -> `MATCH_RESULT_SCHEMA`

These delegate to Rust functions `bbox_pairwise_iou` and `bbox_match_detections`
which internally convert bounding boxes to rectangular contours and reuse the
existing contour matching logic. Used by `BBoxMatcher` in the metrics subsystem.

### `.point` (PointNamespace)

Registered on `pl.Expr` for columns matching `POINT_SCHEMA`. Each method calls `register_plugin_function` with a specific Rust function name (e.g., `point_normalize`, `point_distance`).

### Important: These bypass the pipeline/graph system

Point and contour namespace operations go directly through `register_plugin_function` to dedicated Rust functions. They do **not** go through the `vb_graph` pipeline path. This is a design distinction — they operate on Struct columns directly rather than on binary image data.

### Parameter policy: per-row via input slots, not `ParamValue`

Because these bypass `vb_graph`, they have none of `ParamValue`'s literal-vs-
expression machinery. Their per-row channel is instead the plugin's **input
series**: `_ArgBinder` (`_namespace.py`) appends an expression-valued parameter
as an extra plugin argument and records it in an `input_slots` name→index map
passed as a kwarg. Rust reads it back through `GeomParams`
(`polars-cv/src/geom_params.rs`), which delegates to `params::ParamCol` — so
these namespaces inherit the graph engine's dtype coverage, scalar broadcasting
and null-as-error policy for free.

**Look inputs up by name, never by position.** Several of these functions take
*optional* data operands (`match_detections`' `scores`, `point.rotate`'s
`origin`). With per-row parameters also occupying input slots, an appended
parameter is otherwise indistinguishable from an omitted operand. Register the
data operands in the map too (`binder.add_data("scores", scores)`) and read them
via `params.slot("scores")`.

Numeric parameters here are per-row capable; parameters that *select behaviour*
rather than carry a value stay literal kwargs (`scale`'s `origin`,
`ensure_winding`'s `direction`).

Validation that can no longer happen once per batch moves into the row loop and
names the offending row — see the `threshold` range check in
`contour_match_detections` and the zero-dimension guard in `point_normalize`.

**Keep signatures honest.** These namespaces have no generated stub, and
their *annotations* have no parity test — the schema they publish does
(`tests/test_schema_parity_namespaces.py`, swept in both arities and
completeness-asserted against the real method list, as the section below
describes). So a hand-written annotation can still drift from behaviour
unnoticed —
which is exactly how four `.contour` methods came to advertise `int | pl.Expr`
while unconditionally raising `TypeError` on it. `mkdocs.yml` sets
`show_signature_annotations: true`, so a wrong annotation is published in the
API reference.

## Adding a Geometry Operation

1. **Rust:** Add the function in `polars-cv/src/point.rs` or `polars-cv/src/contour.rs` with `#[polars_expr]`
2. **Python:** Add a method to `PointNamespace` or `ContourNamespace` that builds an `_ArgBinder` and calls `binder.call(self, "<rust_fn_name>")` (or `self._plugin(...)` when it takes no parameters)
3. **Per-row params:** register each with `binder.add_param(...)`, add the matching field to `ContourKwargs`/`PointKwargs`, and resolve it inside the row loop with `GeomParams`
4. **Tests:** Add to `tests/test_contour_plugin.py` or create a reference test. For a per-row parameter, assert two rows with *different* values produce *different* outputs (`tests/test_expression_params.py`) — a call that merely succeeds cannot distinguish "resolved per row" from "silently dropped"

## Schema export policy

- `ANNOTATED_POINT_SCHEMA` and `MATCH_RESULT_SCHEMA` are deliberately geometry-only exports (`polars_cv.geometry.ANNOTATED_POINT_SCHEMA`): they describe internal match-result structures, unlike the six user-facing geometry schemas re-exported at the package top level. Import them from `polars_cv.geometry` when needed.

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
codebase removes. The set-level accessors (`pairwise_iou`, `match_detections`,
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
- `map_contours` / `map_contours_with_params` / `zip_contours` wrap the results
  with the same `Arity::wrap`, and are the only decode path the accessors use.
- `contour_accessor!` emits both halves from a single `-> <elem>` declaration.

`map_contours_with_params` also owns the null-parameter policy: it wraps each
*row* in `GeomParams::row`, so `on_null("null")` nulls the row rather than each
contour. That is the job `contour_row` used to do, moved so it cannot be
forgotten.

`contour_contains_point` is the one accessor with its own loop: its second
operand is a point, so neither the `map` arm (one operand) nor the `zip` arm
(two contour operands) describes it. It still reads `Arity::of` and wraps
through `elementwise_field`/`pack_row`, so only the loop is local.

### Adding an accessor

Use a `contour_accessor!` arm — `map`, `map_params` or `zip`. Do not write a
bare `#[polars_expr(output_type=...)]` for a contour accessor: the case table in
`tests/test_schema_parity_namespaces.py` is completeness-asserted against the
namespace's real methods *and* swept in both arities, so an accessor that skips
the macro fails `test_contour_accessors_over_a_contour_set` rather than shipping
a schema its data contradicts.

`.point` has the identical single-only limitation over `POINT_SET_SCHEMA`;
`geom_arity.rs` is written to fit it, but wiring it up is not done.
