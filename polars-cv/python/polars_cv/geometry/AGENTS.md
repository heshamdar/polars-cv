# AGENTS.md — Geometry Subsystem (`polars_cv.geometry`)

> Read the [root AGENTS.md](../../../../AGENTS.md) and [Python API AGENTS.md](../AGENTS.md) first.
> Update this file when you change schemas, geometry namespaces, or validation.

## Purpose

This subpackage defines the **structured geometry types** (points, contours, bounding boxes) and **expression namespaces** (`.point`, `.contour`) for operating on them within Polars DataFrames.

Geometry data is represented as Polars Struct columns with well-defined schemas. Operations are implemented as Polars plugin functions (Rust-backed) registered via expression namespaces.

## Key Files

| File | Responsibility |
|------|---------------|
| `__init__.py` | Re-exports schemas and validation errors |
| `schemas.py` | Schema constants (`POINT_SCHEMA`, `CONTOUR_SCHEMA`, `CONTOUR_SET_SCHEMA`, `MATCH_RESULT_SCHEMA`, `BBOX_SCHEMA`, etc.), validation helpers, factory functions |
| `contours.py` | `ContourNamespace` (`.contour`) — area, perimeter, centroid, bounding_box, IoU/Dice/Hausdorff, set-level matching (`pairwise_iou`, `match_detections`), and heatmap scoring (`label_reduce`) |
| `points.py` | `PointNamespace` (`.point`) — normalize, to_absolute, translate, scale, rotate, distance, angle_to, etc. |
| `validation.py` | Error classes: `GeometryValidationError`, `OpenContourError`, `CoordinateRangeError`, `InvalidContourError` |

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

### Winding Direction Convention

Winding is **computed from point order**, not stored:
- Counter-clockwise (CCW) = positive signed area = exterior boundary
- Clockwise (CW) = negative signed area = hole
- Uses the Shoelace formula

## Expression Namespaces

### `.contour` (ContourNamespace)

Registered on `pl.Expr` for columns matching `CONTOUR_SCHEMA`. Each method calls `register_plugin_function` with a specific Rust function name (e.g., `contour_area`, `contour_iou`).

Set-level detection helpers also live here and operate on `CONTOUR_SET_SCHEMA`:
- `pairwise_iou(other)` -> `List[List[Float64]]`
- `match_detections(other, threshold, scores, strategy)` -> `MATCH_RESULT_SCHEMA`
- `label_reduce(heatmap, reduction, region_mode)` -> `List[Float64]`

### `.point` (PointNamespace)

Registered on `pl.Expr` for columns matching `POINT_SCHEMA`. Each method calls `register_plugin_function` with a specific Rust function name (e.g., `point_normalize`, `point_distance`).

### Important: These bypass the pipeline/graph system

Point and contour namespace operations go directly through `register_plugin_function` to dedicated Rust functions. They do **not** go through the `vb_graph` pipeline path. This is a design distinction — they operate on Struct columns directly rather than on binary image data.

## Adding a Geometry Operation

1. **Rust:** Add the function in `polars-cv/src/point.rs` or `polars-cv/src/contour.rs` with `#[polars_expr]`
2. **Python:** Add a method to `PointNamespace` or `ContourNamespace` that calls `register_plugin_function`
3. **Tests:** Add to `tests/test_contour_plugin.py` or create a reference test

## Known Issue

`ANNOTATED_POINT_SCHEMA` is exported by `geometry/__init__.py` but not included in the main `polars_cv/__init__.py` `__all__`. Decide whether it should be part of the top-level public API.
