//! The geometry namespaces' functions: one typed definition per plugin
//! function, as the ops are defined.
//!
//! `.contour`, `.point` and `.bbox` are standalone `#[polars_expr]` functions
//! rather than graph nodes, but each is described the way an op is: a variant
//! of a mode-generic family deriving `Ops`, with the wire's fields (a per-row
//! `M::V<T>` or a data operand's `ColumnRef`), its doc comment as the Python
//! docstring, its defaults declared once with `#[param(default = ...)]`, and a
//! sample. A function that is also a pipeline op (`.contour.area` is
//! `contour_area`) is not redeclared here: it *is* that [`GeometryOp`]
//! variant ([`OP_ACCESSORS`]). Each plugin function parses its own definition
//! strictly by name ([`GeomParams::parse`](crate::geom_params::GeomParams::parse)),
//! and the Python accessor methods are generated from [`geom_catalog`].

use serde::Serialize;
use view_buffer::geometry::contour::Winding;
use view_buffer::geometry::label::{LabelReduction, LabelRegionMode};
use view_buffer::mode::{ColumnRef, Exec, Mode, OpDesc, Wire, WireOps};
use view_buffer::GeometryOp;

use polars_cv_macros::Ops;

/// The `.contour` accessor's functions that are not pipeline ops.
#[derive(Debug, Clone, PartialEq, Ops)]
pub enum ContourFn<M: Mode = Exec> {
    /// Convert pixel coordinates to normalized [0,1] range.
    ///
    /// Returns:
    ///     Contour with coordinates in [0,1] range, or a list of them for a
    ///     contour-set column.
    #[op(name = "contour_normalize", python = "normalize",
         sample = {"width": 640.0, "height": 480.0})]
    Normalize {
        /// Reference width for normalization (literal or expression).
        width: M::V<f64>,
        /// Reference height for normalization (literal or expression).
        height: M::V<f64>,
    },
    /// Convert normalized coordinates to pixel coordinates.
    ///
    /// Returns:
    ///     Contour with pixel coordinates, or a list of them for a
    ///     contour-set column.
    #[op(name = "contour_to_absolute", python = "to_absolute",
         sample = {"width": 640.0, "height": 480.0})]
    ToAbsolute {
        /// Reference width for scaling (literal or expression).
        width: M::V<f64>,
        /// Reference height for scaling (literal or expression).
        height: M::V<f64>,
    },
    /// Compute the exterior ring's winding direction from point order.
    ///
    /// Returns:
    ///     String 'ccw' for counter-clockwise, 'cw' for clockwise — a list of
    ///     them for a contour-set column.
    ///
    /// Note:
    ///     Winding is computed using the Shoelace formula:
    ///     - Positive signed area = CCW
    ///     - Negative signed area = CW
    ///
    ///     This is purely a report on point order. Winding does not mark a ring as
    ///     a hole — the `holes` field does — and no other operation consults it.
    #[op(name = "contour_winding", python = "winding", sample = {})]
    Winding,
    /// Check if contour is convex.
    ///
    /// Returns:
    ///     Boolean indicating convexity, or `List(Boolean)` for a contour-set
    ///     column.
    #[op(name = "contour_is_convex", python = "is_convex", sample = {})]
    IsConvex,
    /// Reverse point order (flips winding direction).
    ///
    /// Exterior and holes are reversed together. This changes only what
    /// `winding()` reports — the region the contour describes is unaffected,
    /// because no operation reads winding.
    ///
    /// Returns:
    ///     Contour with reversed point order, or a list of them for a
    ///     contour-set column.
    #[op(name = "contour_flip", python = "flip", sample = {})]
    Flip,
    /// Ensure contour has specified winding direction.
    ///
    /// Flips the contour if needed to match target winding. Use this when handing
    /// contours to an external consumer that expects a convention; polars-cv's own
    /// operations never require one.
    ///
    /// Returns:
    ///     Contour with guaranteed winding direction, or a list of them for a
    ///     contour-set column.
    #[op(name = "contour_ensure_winding", python = "ensure_winding",
         sample = {"direction": "cw"})]
    EnsureWinding {
        /// Target winding direction. Accepts a Polars expression for a per-row
        /// choice — rewinding a ring reorders its vertices and leaves the output
        /// schema untouched.
        direction: M::V<Winding>,
    },
    /// Compute Intersection over Union with another contour.
    ///
    /// Overlap is exact for arbitrary simple polygons — concave shapes and holes
    /// included, in either winding direction. Each contour is measured as its
    /// exterior minus the union of its hole rings, the same region `area()`,
    /// `contains_point()` and rasterization use.
    ///
    /// Returns:
    ///     Float64 IoU value in [0, 1] — `List(Float64)` when either side is a
    ///     contour set (see the class docstring on broadcasting).
    #[op(name = "contour_iou", python = "iou", sample = {"other": {"$slot": 1}})]
    Iou {
        /// Another contour column to compare with.
        other: ColumnRef,
    },
    /// Compute Dice coefficient with another contour.
    ///
    /// Dice = 2 * intersection / (area1 + area2)
    ///
    /// Returns:
    ///     Float64 Dice coefficient in [0, 1] — `List(Float64)` when either
    ///     side is a contour set.
    #[op(name = "contour_dice", python = "dice", sample = {"other": {"$slot": 1}})]
    Dice {
        /// Another contour column to compare with.
        other: ColumnRef,
    },
    /// Compute Hausdorff distance to another contour.
    ///
    /// The maximum, over every *vertex* of either contour, of the distance to the
    /// nearest vertex of the other. This is a vertex-to-vertex measure, not
    /// point-to-edge: two contours tracing the same outline with different vertex
    /// spacing have a non-zero distance. Hole vertices are included. An empty
    /// contour gives `inf`.
    ///
    /// Returns:
    ///     Float64 Hausdorff distance — `List(Float64)` when either side is a
    ///     contour set.
    #[op(name = "contour_hausdorff", python = "hausdorff_distance",
         sample = {"other": {"$slot": 1}})]
    Hausdorff {
        /// Another contour column to compare with.
        other: ColumnRef,
    },
    /// Test if contour contains a point.
    ///
    /// Returns:
    ///     Boolean indicating if point is inside contour — `List(Boolean)`, one
    ///     per contour, for a contour-set column.
    #[op(name = "contour_contains_point", python = "contains_point",
         sample = {"point": {"$slot": 1}})]
    ContainsPoint {
        /// Point column to test.
        point: ColumnRef,
    },
    /// Compute full pairwise IoU matrix between contour sets.
    ///
    /// Returns:
    ///     A nested list (`List[List[Float64]]`) representing an N x M IoU matrix.
    #[op(name = "contour_pairwise_iou", python = "pairwise_iou",
         sample = {"other": {"$slot": 1}})]
    PairwiseIou {
        /// Ground-truth contour-set expression (`List[Contour]`).
        other: ColumnRef,
    },
    /// Pair each contour with at most one contour in *other*, by overlap.
    ///
    /// Greedy and exclusive: contours are visited in *order* and each takes the
    /// highest-IoU partner not already taken, keeping it only if that IoU is at
    /// least *threshold*. A partner claimed earlier is unavailable later, so the
    /// order is what resolves contention. Exact ties go to the lowest index in
    /// *other*; the threshold bound is inclusive.
    ///
    /// The relation carries no interpretation. Contours can represent anything,
    /// and whether a pairing means a "detection" is the caller's business --
    /// as is deriving *order* from confidence, and counting pairings across a
    /// population.
    ///
    /// Returns:
    ///     A struct matching :data:`polars_cv.CORRESPONDENCE_SCHEMA`:
    ///     ``right_idx`` (index into *other*, null where unpaired) and
    ///     ``overlap`` (the IoU of the chosen pair, 0.0 where unpaired), both
    ///     positionally aligned with this expression's contours.
    #[op(name = "contour_correspond", python = "correspond",
         sample = {"other": {"$slot": 1}, "threshold": 0.5, "order": {"$slot": 2}})]
    Correspond {
        /// Contour-set expression to pair against (`List[Contour]`).
        other: ColumnRef,
        /// Minimum IoU for a pairing, in [0, 1]. Accepts a Polars expression
        /// for a per-row threshold.
        #[param(default = 0.5)]
        threshold: M::V<f64>,
        /// Optional per-row list of indices giving the visit sequence, a
        /// permutation of ``0..n``. Defaults to natural order.
        order: Option<ColumnRef>,
    },
    /// Score each contour from an image/array expression with configurable reduction.
    ///
    /// Runs the same engine routine as :meth:`polars_cv.Pipeline.label_reduce`,
    /// so the two agree on every reduction, region mode and edge case; this
    /// accessor differs only in taking an already-materialized contour column
    /// rather than extracting one inside a pipeline.
    ///
    /// Pixels are sampled at their centres. A contour whose region catches no
    /// pixel centre — a sub-pixel detection — is scored at its centroid rather
    /// than as 0.0.
    ///
    /// Returns:
    ///     A list of float scores, aligned to the input contour order.
    #[op(name = "contour_label_reduce", python = "label_reduce",
         sample = {"image": {"$slot": 1}, "reduction": "mean", "region_mode": "bbox"})]
    LabelReduce {
        /// Image/array expression aligned by row with contour sets.
        image: ColumnRef,
        /// Aggregation method over pixels in each contour region. Accepts a
        /// Polars expression for a per-row choice.
        #[param(default = "max")]
        reduction: M::V<LabelReduction>,
        /// Region selector - ``"interior"`` (pixels strictly inside),
        /// ``"boundary"`` (interior plus the contour boundary) or ``"bbox"``
        /// (everything in the bounding box). Accepts a Polars expression.
        #[param(default = "interior")]
        region_mode: M::V<LabelRegionMode>,
    },
}

/// The `.point` accessor's functions.
#[derive(Debug, Clone, PartialEq, Ops)]
pub enum PointFn<M: Mode = Exec> {
    /// Convert pixel coordinates to normalized [0,1] range.
    ///
    /// Returns:
    ///     Point with coordinates in [0,1] range.
    #[op(name = "point_normalize", python = "normalize",
         sample = {"width": 640.0, "height": 480.0})]
    Normalize {
        /// Reference width for normalization; non-zero (literal or expression).
        width: M::V<f64>,
        /// Reference height for normalization; non-zero (literal or expression).
        height: M::V<f64>,
    },
    /// Convert normalized coordinates to pixel coordinates.
    ///
    /// Returns:
    ///     Point with pixel coordinates.
    #[op(name = "point_to_absolute", python = "to_absolute",
         sample = {"width": 640.0, "height": 480.0})]
    ToAbsolute {
        /// Reference width for scaling (literal or expression).
        width: M::V<f64>,
        /// Reference height for scaling (literal or expression).
        height: M::V<f64>,
    },
    /// Translate point by offset.
    ///
    /// Returns:
    ///     Translated point.
    #[op(name = "point_translate", python = "translate", sample = {"dx": 1.0, "dy": -2.0})]
    Translate {
        /// X offset (literal or expression).
        dx: M::V<f64>,
        /// Y offset (literal or expression).
        dy: M::V<f64>,
    },
    /// Scale point coordinates about the coordinate origin.
    ///
    /// Returns:
    ///     Scaled point.
    #[op(name = "point_scale", python = "scale", sample = {"sx": 2.0, "sy": 0.5})]
    Scale {
        /// X scale factor (literal or expression).
        sx: M::V<f64>,
        /// Y scale factor (literal or expression).
        sy: M::V<f64>,
    },
    /// Compute Euclidean distance to another point.
    ///
    /// Returns:
    ///     Float64 distance.
    #[op(name = "point_distance", python = "distance", sample = {"other": {"$slot": 1}})]
    Distance {
        /// Another point column.
        other: ColumnRef,
    },
    /// Compute Manhattan (L1) distance to another point.
    ///
    /// Returns:
    ///     Float64 distance.
    #[op(name = "point_manhattan_distance", python = "manhattan_distance",
         sample = {"other": {"$slot": 1}})]
    ManhattanDistance {
        /// Another point column.
        other: ColumnRef,
    },
    /// Compute the minimum distance from the point to the contour boundary.
    ///
    /// Returns:
    ///     Float64 distance to the nearest edge of the contour.
    #[op(name = "point_distance_to_contour", python = "distance_to_contour",
         sample = {"contour": {"$slot": 1}})]
    DistanceToContour {
        /// Contour column.
        contour: ColumnRef,
    },
    /// Compute the signed distance from the point to the contour boundary.
    ///
    /// Returns:
    ///     Float64 distance: negative if the point is inside the contour,
    ///     positive if outside.
    #[op(name = "point_signed_distance_to_contour", python = "signed_distance_to_contour",
         sample = {"contour": {"$slot": 1}})]
    SignedDistanceToContour {
        /// Contour column.
        contour: ColumnRef,
    },
    /// Find the nearest point on the contour boundary.
    ///
    /// Returns:
    ///     The nearest point on the contour, or null for an empty contour.
    #[op(name = "point_nearest_on_contour", python = "nearest_point_on_contour",
         sample = {"contour": {"$slot": 1}})]
    NearestOnContour {
        /// Contour column.
        contour: ColumnRef,
    },
    /// Compute the angle from this point to another, in radians.
    ///
    /// Returns:
    ///     Float64 angle in radians, from ``atan2(dy, dx)``.
    #[op(name = "point_angle_to", python = "angle_to", sample = {"other": {"$slot": 1}})]
    AngleTo {
        /// Target point column.
        other: ColumnRef,
    },
    /// Rotate the point about an origin by an angle in radians.
    ///
    /// Returns:
    ///     Rotated point.
    #[op(name = "point_rotate", python = "rotate",
         sample = {"angle": 1.5, "origin": {"$slot": 1}})]
    Rotate {
        /// Rotation angle in radians (literal or expression); positive is
        /// counter-clockwise in a y-up frame.
        angle: M::V<f64>,
        /// Point column to rotate about; the coordinate origin ``(0, 0)`` when
        /// omitted (or null in a row).
        origin: Option<ColumnRef>,
    },
    /// Compute the midpoint between two points.
    ///
    /// Returns:
    ///     The point halfway between.
    #[op(name = "point_midpoint", python = "midpoint", sample = {"other": {"$slot": 1}})]
    Midpoint {
        /// Another point column.
        other: ColumnRef,
    },
    /// Linearly interpolate between this point and another.
    ///
    /// Returns:
    ///     ``self + t * (other - self)``.
    #[op(name = "point_interpolate", python = "interpolate",
         sample = {"other": {"$slot": 1}, "t": 0.25})]
    Interpolate {
        /// Another point column.
        other: ColumnRef,
        /// Interpolation parameter: 0 gives this point, 1 gives *other*
        /// (literal or expression).
        #[param(default = 0.5)]
        t: M::V<f64>,
    },
    /// Check whether the point lies within a bounding box (edges included).
    ///
    /// Returns:
    ///     Boolean.
    #[op(name = "point_within_bbox", python = "within_bbox", sample = {"bbox": {"$slot": 1}})]
    WithinBbox {
        /// Bounding-box column.
        bbox: ColumnRef,
    },
}

/// The `.bbox` accessor's functions.
#[derive(Debug, Clone, PartialEq, Ops)]
pub enum BBoxFn<M: Mode = Exec> {
    /// Compute the full pairwise IoU matrix between two bounding-box sets.
    ///
    /// Returns:
    ///     A nested list (`List[List[Float64]]`) representing an N x M IoU matrix.
    #[op(name = "bbox_pairwise_iou", python = "pairwise_iou", sample = {"other": {"$slot": 1}})]
    PairwiseIou {
        /// Ground-truth bounding-box set expression.
        other: ColumnRef,
    },
    /// Pair each box with at most one box in *other*, by IoU.
    ///
    /// The bounding-box form of :meth:`ContourNamespace.correspond` — the same
    /// greedy, exclusive rule, threshold and ordering, over box overlap.
    ///
    /// Returns:
    ///     A struct matching :data:`polars_cv.CORRESPONDENCE_SCHEMA`.
    #[op(name = "bbox_correspond", python = "correspond",
         sample = {"other": {"$slot": 1}, "threshold": 0.5, "order": {"$slot": 2}})]
    Correspond {
        /// Bounding-box set expression to pair against.
        other: ColumnRef,
        /// Minimum IoU for a pairing, in [0, 1]. Accepts a Polars expression
        /// for a per-row threshold.
        #[param(default = 0.5)]
        threshold: M::V<f64>,
        /// Optional per-row list of indices giving the visit sequence, a
        /// permutation of ``0..n``. Defaults to natural order.
        order: Option<ColumnRef>,
    },
}

// The families' per-row values are read field by field where each function
// runs; a family with a per-row field reads it through `M::V`, so the mode
// parameter is always used.
const _: fn(&ContourFn<Wire>, &PointFn<Wire>, &BBoxFn<Wire>) = |_, _, _| ();

/// The `.contour` accessor methods that are pipeline ops: `(op, method)`.
///
/// Each is that [`GeometryOp`] variant — its fields, defaults and doc — and
/// its plugin function has the op's wire name.
pub const OP_ACCESSORS: &[(&str, &str)] = &[
    ("contour_area", "area"),
    ("contour_perimeter", "perimeter"),
    ("contour_centroid", "centroid"),
    ("contour_bounding_box", "bounding_box"),
    ("contour_convex_hull", "convex_hull"),
    ("contour_translate", "translate"),
    ("contour_scale", "scale"),
    ("contour_simplify", "simplify"),
];

/// One accessor method: its namespace and its function's description.
#[derive(Debug, Serialize)]
pub struct GeomDesc {
    pub namespace: &'static str,
    #[serde(flatten)]
    pub function: OpDesc,
}

/// Every geometry accessor method, by namespace then method name — what the
/// generated `.contour`/`.point`/`.bbox` methods are rendered from.
pub fn geom_catalog() -> Vec<GeomDesc> {
    let ops = GeometryOp::<Wire>::catalog();
    let op_backed = OP_ACCESSORS.iter().map(|(name, method)| {
        let mut desc = ops
            .iter()
            .find(|d| d.name == *name)
            .cloned()
            .unwrap_or_else(|| panic!("OP_ACCESSORS names '{name}', which is not an op"));
        desc.python = method;
        desc.visibility = "public";
        ("contour", desc)
    });
    let families = <ContourFn<Wire> as WireOps>::catalog()
        .into_iter()
        .map(|d| ("contour", d))
        .chain(
            <PointFn<Wire> as WireOps>::catalog()
                .into_iter()
                .map(|d| ("point", d)),
        )
        .chain(
            <BBoxFn<Wire> as WireOps>::catalog()
                .into_iter()
                .map(|d| ("bbox", d)),
        );
    let mut catalog: Vec<GeomDesc> = op_backed
        .chain(families)
        .map(|(namespace, function)| GeomDesc {
            namespace,
            function,
        })
        .collect();
    catalog.sort_by_key(|d| (d.namespace, d.function.python));
    catalog
}

/// Parse `args` as the accessor function `name`: every literal checked
/// against its definition (a per-row slot is read when its row runs). The
/// generated Python accessor calls this as the expression is built, so a
/// misspelled enum or a wrong type is refused where it was written.
pub fn check_call(name: &str, args: serde_json::Value) -> Result<(), String> {
    fn parsed<F>(r: Option<Result<F, String>>) -> Option<Result<(), String>> {
        r.map(|r| r.map(|_| ()))
    }
    let op_backed = OP_ACCESSORS.iter().any(|(op, _)| *op == name);
    parsed(<ContourFn<Wire> as WireOps>::from_wire(name, args.clone()))
        .or_else(|| parsed(<PointFn<Wire> as WireOps>::from_wire(name, args.clone())))
        .or_else(|| parsed(<BBoxFn<Wire> as WireOps>::from_wire(name, args.clone())))
        .or_else(|| {
            op_backed
                .then(|| parsed(<GeometryOp<Wire> as WireOps>::from_wire(name, args)))
                .flatten()
        })
        .unwrap_or_else(|| Err(format!("'{name}' is not a geometry accessor function")))
        .map_err(|e| format!("{name}: {e}"))
}

/// The catalogue as committed in `tests/golden/geom_catalog.json`.
pub fn geom_catalog_json() -> String {
    let mut text = serde_json::to_string_pretty(&geom_catalog()).expect("the catalogue serializes");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every function's sample, re-read from its own wire fields, is itself.
    fn round_trip<F: WireOps + std::fmt::Debug + PartialEq>(
        samples: Vec<F>,
        fields: impl Fn(&F) -> serde_json::Map<String, serde_json::Value>,
    ) {
        for sample in samples {
            let name = sample.wire_name().expect("a function has a wire name");
            let back = F::from_wire(name, serde_json::Value::Object(fields(&sample)))
                .expect("registered")
                .unwrap();
            assert_eq!(back, sample, "{name}");
        }
    }

    #[test]
    fn samples_round_trip_through_the_wire() {
        round_trip(ContourFn::<Wire>::samples(), |f| f.wire_fields().unwrap());
        round_trip(PointFn::<Wire>::samples(), |f| f.wire_fields().unwrap());
        round_trip(BBoxFn::<Wire>::samples(), |f| f.wire_fields().unwrap());
    }

    /// Parse `fields` as the function `name` of whichever family has it.
    fn parse_any(name: &str, fields: serde_json::Value) -> Result<(), String> {
        fn ok<F>(r: Option<Result<F, String>>) -> Option<Result<(), String>> {
            r.map(|r| r.map(|_| ()))
        }
        ok(ContourFn::<Wire>::from_wire(name, fields.clone()))
            .or_else(|| ok(PointFn::<Wire>::from_wire(name, fields.clone())))
            .or_else(|| ok(BBoxFn::<Wire>::from_wire(name, fields.clone())))
            .or_else(|| ok(GeometryOp::<Wire>::from_wire(name, fields)))
            .unwrap_or_else(|| Err(format!("'{name}' is not catalogued")))
    }

    /// A declared default becomes the Python signature's default, so it must
    /// parse as its field.
    #[test]
    fn every_declared_default_is_a_valid_value() {
        for desc in geom_catalog() {
            let d = &desc.function;
            for field in d.fields.iter().filter(|f| f.default.is_some()) {
                let mut fields = serde_json::Map::new();
                for f in &d.fields {
                    let value = f.default.clone().unwrap_or(serde_json::json!({"$slot": 1}));
                    fields.insert(f.name.into(), value);
                }
                fields.insert(field.name.into(), field.default.clone().unwrap());
                parse_any(d.name, serde_json::Value::Object(fields))
                    .unwrap_or_else(|e| panic!("{}.{} default: {e}", d.name, field.name));
            }
        }
    }

    #[test]
    fn method_names_are_unique_per_namespace() {
        let catalog = geom_catalog();
        let mut seen = std::collections::BTreeSet::new();
        for d in &catalog {
            assert!(
                seen.insert((d.namespace, d.function.python)),
                "{} {}",
                d.namespace,
                d.function.python
            );
        }
    }

    #[test]
    fn geom_catalog_matches_the_committed_file() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/geom_catalog.json"
        );
        let current = geom_catalog_json();
        if std::env::var_os("POLARS_CV_BLESS").is_some() {
            std::fs::write(path, &current).unwrap();
        }
        let committed = std::fs::read_to_string(path).unwrap_or_default();
        assert!(
            committed == current,
            "tests/golden/geom_catalog.json is stale; regenerate with \
             POLARS_CV_BLESS=1 cargo test -p polars-cv catalog_matches, then \
             python scripts/gen_ops.py"
        );
    }
}
