//! The typed op catalogue: one Rust definition per operation.
//!
//! An op is one variant of a *family* — an engine enum in view-buffer
//! (`ImageOpKind`, `ComputeOp`, …) or [`graph::GraphOp`] — generic over a
//! [`Mode`](view_buffer::mode::Mode) and deriving `Ops`/`Resolve`; the family
//! is one line in [`typed_ops!`]. From that:
//!
//! - **the derived wire** rejects an unknown op, an unknown or missing field, a
//!   wrong type, an out-of-range value and a per-row value for a structural
//!   field;
//! - **the compiler** holds the executed op to the wire's fields: the `Exec`
//!   variant *is* the `Wire` variant with each value resolved, so there is no
//!   field to forget;
//! - **the catalogue** ([`catalog_json`], committed as
//!   `tests/golden/op_catalog.json`) is what `scripts/gen_ops.py` generates
//!   the Python builder methods from.
//!
//! [`TypedOp`] *is* the wire op: `{"op": <name>, <field>: <value>, ...}`,
//! deserialized strictly by name. A name no op registers is an error.

pub mod graph;
pub mod param;

use polars::prelude::*;
use serde::Serialize;

use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
pub use param::{ColumnRef, FieldType, Literal, NodeRef, Param, ParamExt};
pub use view_buffer::mode::OpDesc;

/// Register the typed op families.
///
/// Each line is a family — an enum deriving `Ops`/`Resolve`, generic over
/// the [`Mode`](view_buffer::mode::Mode) — and the position its ops take in a
/// [`GraphStep`], written once and used both to build and to match one. The
/// typed op is `GraphStep<Wire>`: registering is the whole act, since the
/// line is what makes the family's ops deserializable, described and — through
/// the samples — covered by the registry-driven tests; resolving and every
/// plan-time rule are `GraphStep`'s own, generic over the mode.
macro_rules! typed_ops {
    ($($($family:ident)::+ => |$op:ident| $($at:tt)::+ ($($inner:tt)*);)*) => {
        /// The typed op: a graph step whose parameters are literals or
        /// per-row slots (see the module docs).
        pub type TypedOp = GraphStep<view_buffer::mode::Wire>;

        impl TypedOp {
            /// Every typed op's wire name, sorted.
            #[cfg(test)]
            pub fn names() -> Vec<&'static str> {
                let mut names: Vec<&'static str> = Vec::new();
                $(names.extend_from_slice(
                    <$($family)::+<view_buffer::mode::Wire>>::WIRE_NAMES,
                );)*
                names.sort_unstable();
                names
            }

            /// The op's wire name.
            pub fn name(&self) -> &'static str {
                match self {
                    $($($at)::+($($inner)*) => {
                        $op.wire_name().expect("a typed op has a wire name")
                    })*
                }
            }

            /// Deserialize the op `name` from its fields (the wire object
            /// without `"op"`). `None` when `name` is not a typed op.
            pub fn from_fields(
                name: &str,
                fields: serde_json::Value,
            ) -> Option<Result<Self, String>> {
                $(
                    if <$($family)::+<view_buffer::mode::Wire>>::WIRE_NAMES.contains(&name) {
                        return <$($family)::+<view_buffer::mode::Wire>>::from_wire(name, fields)
                            .map(|r| r.map(|$op| $($at)::+($($inner)*)));
                    }
                )*
                let _ = fields;
                None
            }

            /// The op's fields as a wire object (without `"op"`).
            pub fn fields_json(&self) -> serde_json::Value {
                match self {
                    $($($at)::+($($inner)*) => serde_json::Value::Object(
                        $op.wire_fields().expect("a typed op has wire fields"),
                    ),)*
                }
            }

            /// Call `f(field, slot)` for every slot a field reads.
            pub fn visit_slots(&self, f: &mut dyn FnMut(&'static str, usize)) {
                match self {
                    $($($at)::+($($inner)*) => $op.visit_slots(f),)*
                }
            }

            /// One valid instance of every typed op, in `names()` order.
            #[cfg(test)]
            pub fn samples() -> Vec<TypedOp> {
                let mut samples: Vec<TypedOp> = Vec::new();
                $(samples.extend(
                    <$($family)::+<view_buffer::mode::Wire>>::samples()
                        .into_iter()
                        .map(|$op| $($at)::+($($inner)*)),
                );)*
                samples.sort_by_key(TypedOp::name);
                samples
            }

            /// Every typed op's description, in `names()` order.
            pub fn catalog() -> Vec<OpDesc> {
                let mut catalog = Vec::new();
                $(catalog.extend(<$($family)::+<view_buffer::mode::Wire>>::catalog());)*
                catalog.sort_by_key(|d| d.name);
                catalog
            }
        }
    };
}

typed_ops! {
    view_buffer::ImageOpKind => |op| GraphStep::Buffer(
        view_buffer::ViewDto::Image(view_buffer::ImageOp { kind: op })
    );
    view_buffer::ComputeOp => |op| GraphStep::Buffer(view_buffer::ViewDto::Compute(op));
    view_buffer::ViewOp => |op| GraphStep::Buffer(view_buffer::ViewDto::View(op));
    view_buffer::ColorConvertOp => |op| GraphStep::Buffer(view_buffer::ViewDto::Color(op));
    view_buffer::ops::filter::ConvolveOp => |op| GraphStep::Buffer(view_buffer::ViewDto::Filter(op));
    view_buffer::GeometryOp => |op| GraphStep::Geometry(op);
    view_buffer::ops::ReductionOp => |op| GraphStep::Reduction(op);
    view_buffer::ops::histogram::HistogramOp => |op| GraphStep::Histogram(op);
    view_buffer::ops::phash::PerceptualHashOp => |op| GraphStep::PerceptualHash(op);
    graph::GraphOp => |op| GraphStep::Graph(op);
}

impl Serialize for TypedOp {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut value = self.fields_json();
        value
            .as_object_mut()
            .expect("an op struct serializes to a JSON object")
            .insert("op".into(), self.name().into());
        value.serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for TypedOp {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        // One parse into a map, consumed without a second copy: this runs
        // several times per builder append, via the planning FFIs.
        let mut fields = serde_json::Map::<String, serde_json::Value>::deserialize(d)?;
        let name = match fields.remove("op") {
            Some(serde_json::Value::String(name)) => name,
            _ => return Err(D::Error::custom("an operation needs a string \"op\" name")),
        };
        TypedOp::from_fields(&name, serde_json::Value::Object(fields))
            .ok_or_else(|| D::Error::custom(format!("Unknown operation: '{name}'")))?
            .map_err(|e| D::Error::custom(format!("operation '{name}': {e}")))
    }
}

impl TypedOp {
    /// The step for `row`: each per-row parameter read from its column, then
    /// checked with every value known. An op with no per-row parameter is
    /// resolved once at graph compile time.
    pub fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let step: GraphStep =
            view_buffer::mode::Resolve::resolve(self, &param::RowValues { row, ctx })?;
        step.check()
            .map_err(|e| polars_err!(ComputeError: "{}", e))?;
        Ok(step)
    }

    /// Whether any field is per-row. An op with none resolves once.
    pub fn is_static(&self) -> bool {
        let mut any = false;
        self.visit_slots(&mut |_, _| any = true);
        !any
    }

    /// One past the highest slot the op reads (0 when none).
    pub fn min_inputs(&self) -> usize {
        let mut n = 0;
        self.visit_slots(&mut |_, slot| n = n.max(slot + 1));
        n
    }
}

/// The catalogue as committed in `tests/golden/op_catalog.json`.
pub fn catalog_json() -> String {
    let mut text =
        serde_json::to_string_pretty(&TypedOp::catalog()).expect("the catalogue serializes");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: serde_json::Value) -> Result<TypedOp, String> {
        serde_json::from_value(v).map_err(|e| e.to_string())
    }

    fn parse_err(v: serde_json::Value) -> String {
        parse(v).expect_err("expected the spec to be rejected")
    }

    #[test]
    fn typed_names_are_sorted_and_unique() {
        assert!(TypedOp::names().windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn an_unknown_field_is_rejected_naming_op_and_field() {
        let err = parse_err(json!({"op": "resize", "height": 4, "width": 4,
                                   "filter": "nearest", "antialias": true}));
        assert!(
            err.contains("operation 'resize'") && err.contains("antialias"),
            "{err}"
        );
    }

    #[test]
    fn the_op_tag_is_not_an_unknown_field() {
        let op = parse(json!({"op": "crop", "top": 0, "left": 0})).unwrap();
        assert_eq!(op.name(), "crop");
        assert!(op.is_static(), "an all-literal op resolves once");
    }

    #[test]
    fn a_missing_required_field_is_rejected() {
        let err = parse_err(json!({"op": "resize", "height": 4, "filter": "nearest"}));
        assert!(err.contains("width"), "{err}");
    }

    /// A default is declared once, `#[param(default = ...)]`, and the wire
    /// applies it: the Python signature and a hand-built graph agree on what
    /// an absent field means. It used to reach only the Python signature, so
    /// the wire refused the field as missing.
    #[test]
    fn an_absent_field_takes_its_declared_default() {
        let op = parse(json!({"op": "resize", "height": 4, "width": 4})).unwrap();
        assert_eq!(op.fields_json()["filter"], "lanczos3");
        let op = parse(json!({"op": "rasterize", "size": [4, 4]})).unwrap();
        assert_eq!(op.fields_json()["fill_value"], 255);
        assert_eq!(op.fields_json()["background"], 0);
    }

    #[test]
    fn a_value_error_names_the_field() {
        let err = parse_err(json!({"op": "crop", "top": -5, "left": 0}));
        assert!(
            err.contains("'top'") && err.contains("cannot be negative"),
            "{err}"
        );
    }

    #[test]
    fn integers_are_exact_and_floats_accept_integers() {
        let err = parse_err(json!({"op": "resize", "height": 3.0, "width": 4,
                                   "filter": "nearest"}));
        assert!(
            err.contains("'height'") && err.contains("expected an integer"),
            "{err}"
        );
        // A float field takes `1` as 1.0.
        parse(json!({"op": "warp_affine", "matrix": [1, 0, 0, 0, 1, 0],
                     "output_size": [4, 4], "interpolation": "bilinear",
                     "border_value": 0}))
        .unwrap();
    }

    #[test]
    fn a_slot_in_a_structural_field_is_rejected() {
        let err = parse_err(json!({"op": "histogram", "bins": 8, "closed": "left",
                                   "output": {"$slot": 1}}));
        assert!(
            err.contains("'output'") && err.contains("structural"),
            "{err}"
        );
    }

    #[test]
    fn an_array_field_has_an_exact_length() {
        let err = parse_err(json!({"op": "warp_affine", "matrix": [1, 0, 0, 0, 1],
                                   "output_size": [4, 4], "interpolation": "bilinear",
                                   "border_value": 0}));
        assert!(err.contains("'matrix'") && err.contains("length"), "{err}");
    }

    #[test]
    fn an_enum_value_outside_its_named_table_is_rejected() {
        let err = parse_err(json!({"op": "warp_affine", "matrix": [1, 0, 0, 0, 1, 0],
                                   "output_size": [4, 4], "interpolation": "cubic",
                                   "border_value": 0}));
        assert!(
            err.contains("'interpolation'") && err.contains("nearest") && err.contains("bilinear"),
            "{err}"
        );
        // "triangle" (a former parser-only alias, deleted) is not a spelling: the
        // `NAMED` table is the one list of names (typed-op P2).
        let err = parse_err(json!({"op": "resize", "height": 4, "width": 4,
                                   "filter": "triangle"}));
        assert!(err.contains("'filter'"), "{err}");
    }

    /// A present but invalid value is an error naming the field — never read
    /// as a default. Ported from the legacy strict-parameter tests as each op
    /// migrated; the serde error is now what enforces it.
    #[test]
    fn an_invalid_value_is_rejected_naming_its_field() {
        let cases = [
            (
                json!({"op": "perceptual_hash", "algorithm": "phash", "hash_size": 64}),
                "'algorithm'",
                "perceptual",
            ),
            (
                json!({"op": "perceptual_hash", "algorithm": "average", "hash_size": "large"}),
                "'hash_size'",
                "",
            ),
            (json!({"op": "reduce_max", "axis": "rows"}), "'axis'", ""),
            (json!({"op": "reduce_min", "axis": "rows"}), "'axis'", ""),
            (json!({"op": "reduce_mean", "axis": "rows"}), "'axis'", ""),
            (
                json!({"op": "reduce_std", "axis": null, "ddof": "one"}),
                "'ddof'",
                "",
            ),
            (
                json!({"op": "reduce_std", "axis": null, "ddof": 300}),
                "'ddof'",
                "out of range",
            ),
            (
                json!({"op": "rotate", "angle": 45.0, "expand": "yes",
                    "interpolation": "bilinear", "border_value": 0.0}),
                "'expand'",
                "",
            ),
            (
                json!({"op": "rotate", "angle": 45.0, "expand": false,
                    "interpolation": "cubic", "border_value": 0.0}),
                "'interpolation'",
                "nearest",
            ),
            (
                json!({"op": "convolve2d", "kernel": [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    "normalize": "yes", "border": "replicate"}),
                "'normalize'",
                "",
            ),
            (
                json!({"op": "extract_contours", "mode": "outer", "method": "simple"}),
                "'mode'",
                "external",
            ),
            (
                json!({"op": "extract_contours", "mode": "external", "method": "fancy"}),
                "'method'",
                "simple",
            ),
            (
                json!({"op": "contour_area", "signed": "yes"}),
                "'signed'",
                "",
            ),
            (
                json!({"op": "contour_scale", "sx": 1.0, "sy": 1.0, "origin": "middle"}),
                "'origin'",
                "centroid",
            ),
            (
                json!({"op": "contour_translate", "dx": "far", "dy": 0.0}),
                "'dx'",
                "",
            ),
            (json!({"op": "contour_simplify"}), "tolerance", ""),
            (
                json!({"op": "apply_mask", "mask": "m", "invert": "yes"}),
                "'invert'",
                "",
            ),
            (
                json!({"op": "channel_merge", "others": [1, 2]}),
                "'others[0]'",
                "graph node id",
            ),
            (
                json!({"op": "add", "other": {"$slot": 1}}),
                "'other'",
                "graph node id",
            ),
            (
                json!({"op": "label_reduce", "contours": [[0, 0]],
                    "reduction": "max", "region_mode": "interior"}),
                "'contours'",
                "expression",
            ),
            (
                json!({"op": "rasterize", "size": [8], "fill_value": 255, "background": 0}),
                "'size'",
                "",
            ),
            (
                json!({"op": "rasterize", "size": [8, 8], "fill_value": "red", "background": 0}),
                "'fill_value'",
                "",
            ),
            (
                json!({"op": "rasterize", "size": [8, 8], "fill_value": 300, "background": 0}),
                "'fill_value'",
                "",
            ),
            (
                json!({"op": "rasterize", "size": [8, 8], "fill_value": 255, "background": -1}),
                "'background'",
                "",
            ),
        ];
        for (spec, field, also) in cases {
            let err = parse_err(spec.clone());
            assert!(err.contains(field) && err.contains(also), "{spec}: {err}");
        }
        // An absent (null) axis is a global reduction, not an error.
        for op in ["reduce_max", "reduce_min", "reduce_mean"] {
            parse(json!({"op": op, "axis": null})).unwrap();
        }
    }

    /// The binary ops are one typed op per `BinaryOp::NAMED` entry, each
    /// resolving to its own variant — so the table stays the one list of
    /// names, and a new `BinaryOp` without an op fails here.
    #[test]
    fn binary_ops_are_exactly_the_named_table() {
        use view_buffer::BinaryOp;
        for (name, op) in BinaryOp::NAMED {
            let typed = TypedOp::from_fields(name, json!({"other": "n0"}))
                .unwrap_or_else(|| panic!("BinaryOp '{name}' is not a typed op"))
                .unwrap();
            let step = typed.resolve(0, &ParamCtx::empty()).unwrap();
            match &step {
                GraphStep::Graph(graph) => match graph.binary() {
                    Some((got, other)) => {
                        assert_eq!((got, other.0.as_str()), (*op, "n0"), "{name}")
                    }
                    None => panic!("'{name}' resolves to {step:?}"),
                },
                step => panic!("'{name}' resolves to {step:?}"),
            }
        }
        let binary = TypedOp::samples()
            .into_iter()
            .filter(|op| {
                matches!(
                    op.resolve(0, &ParamCtx::empty()),
                    Ok(GraphStep::Graph(graph)) if graph.binary().is_some()
                )
            })
            .count();
        assert_eq!(binary, BinaryOp::NAMED.len());
    }

    /// `rasterize(shape=<node>)` takes its canvas from another node's buffer,
    /// which only the graph executor has. Resolving it invents no size: the
    /// node reference survives, and executing the op without the canvas the
    /// executor sets from that node is refused.
    #[test]
    fn a_node_sized_rasterize_is_never_given_an_invented_canvas() {
        use view_buffer::geometry::ops::RasterSize;
        let op = TypedOp::from_fields(
            "rasterize",
            json!({"size": "n0", "fill_value": 255, "background": 0}),
        )
        .unwrap()
        .unwrap();
        let GraphStep::Geometry(geo) = op.resolve(0, &ParamCtx::empty()).unwrap() else {
            panic!("rasterize resolves to a geometry step");
        };
        assert!(matches!(
            &geo,
            view_buffer::GeometryOp::Rasterize {
                size: RasterSize::FromNode(_),
                ..
            }
        ));
        // The executor sets the canvas from the node it names; run without
        // one, the op is refused rather than given a size.
        let contours = view_buffer::ops::NodeOutput::from_contours(Vec::new());
        let err = crate::graph::encode::execute_geometry_op(contours, &geo).unwrap_err();
        assert!(err.contains("canvas"), "{err}");
        assert_eq!(geo.with_canvas(4, 6).canvas(), Some((4, 6)));
    }

    #[test]
    fn channel_merge_needs_another_channel() {
        let op = TypedOp::from_fields("channel_merge", json!({"others": []}))
            .unwrap()
            .unwrap();
        let err = op.resolve(0, &ParamCtx::empty()).unwrap_err().to_string();
        assert!(err.contains("at least one"), "{err}");
    }

    #[test]
    fn a_measure_with_no_parameters_refuses_a_stray_one() {
        // The legacy unread-parameter tracker pinned this with
        // `contour_perimeter`; the typed form refuses the key at the boundary.
        for op in [
            "contour_perimeter",
            "contour_centroid",
            "contour_bounding_box",
            "contour_convex_hull",
        ] {
            let err = parse_err(json!({"op": op, "sigma": "u8"}));
            assert!(
                err.contains(&format!("operation '{op}'")) && err.contains("sigma"),
                "{err}"
            );
        }
    }

    #[test]
    fn scale_and_clamp_refuse_an_out_dtype() {
        // The two ops that shipped an accepted-but-unread `out_dtype`: it
        // entered the op's identity and reached no code path.
        for op in [
            json!({"op": "scale", "factor": 2.0, "out_dtype": "u8"}),
            json!({"op": "clamp", "min": 0.0, "max": 1.0, "out_dtype": "u8"}),
        ] {
            let err = parse_err(op);
            assert!(err.contains("out_dtype"), "{err}");
        }
    }

    #[test]
    fn normalize_statistics_belong_to_the_preset_only() {
        let resolve =
            |v: serde_json::Value| parse(v).unwrap().resolve(0, &ParamCtx::empty()).map(|_| ());
        let err = resolve(json!({"op": "normalize", "method": "minmax",
                                 "mean": [0.5], "std": [0.5]}))
        .unwrap_err()
        .to_string();
        assert!(err.contains("only valid for method='preset'"), "{err}");
        let err = resolve(json!({"op": "normalize", "method": "preset", "mean": [0.5]}))
            .unwrap_err()
            .to_string();
        assert!(err.contains("requires both"), "{err}");
        assert!(resolve(json!({"op": "normalize", "method": "zscore"})).is_ok());
    }

    #[test]
    fn the_legacy_wire_form_is_not_accepted_for_a_typed_op() {
        // No fallback: a typed op never takes the untyped path.
        let err = parse_err(json!({"op": "crop",
                                   "top": {"type": "literal", "value": 0},
                                   "left": {"type": "literal", "value": 0}}));
        assert!(err.contains("operation 'crop'"), "{err}");
    }

    /// A slot is exactly `{"$slot": n}` with `n >= 0`; the removed name-keyed
    /// and legacy literal forms are not values of any field.
    #[test]
    fn a_malformed_slot_is_rejected() {
        for (top, expected) in [
            (json!({"$slot": -1}), "slot index"),
            (json!({"$slot": 1, "type": "literal"}), "a slot is exactly"),
            (json!({"type": "expr", "col": "h"}), "'top'"),
            (json!({"type": "literal", "value": 1}), "'top'"),
        ] {
            let err = parse_err(json!({"op": "crop", "top": top, "left": 0}));
            assert!(err.contains(expected), "{top}: {err}");
        }
    }

    #[test]
    fn an_unknown_operation_is_rejected() {
        let err = parse_err(json!({"op": "definitely_not_a_real_op"}));
        assert!(err.contains("Unknown operation"), "{err}");
    }

    #[test]
    fn samples_round_trip_through_the_wire() {
        for op in TypedOp::samples() {
            let wire = serde_json::to_value(&op).unwrap();
            assert_eq!(wire["op"], op.name());
            assert_eq!(parse(wire).unwrap(), op);
        }
    }

    #[test]
    fn every_field_default_is_a_valid_value() {
        // The catalogue's defaults become Python signature defaults; each must
        // deserialize as its field, or the generated method's default call
        // would fail.
        for sample in TypedOp::samples() {
            let desc = TypedOp::catalog()
                .into_iter()
                .find(|d| d.name == sample.name())
                .unwrap();
            for field in desc.fields.iter().filter(|f| f.default.is_some()) {
                let mut wire = sample.fields_json();
                wire[field.name] = field.default.clone().unwrap();
                TypedOp::from_fields(sample.name(), wire)
                    .unwrap()
                    .unwrap_or_else(|e| panic!("{}.{} default: {e}", desc.name, field.name));
            }
        }
    }

    #[test]
    fn slots_are_visited_by_field_name() {
        let op = parse(json!({"op": "warp_affine",
                                    "matrix": [1, 0, {"$slot": 3}, 0, 1, 0],
                                    "output_size": [{"$slot": 1}, 4],
                                    "interpolation": "bilinear", "border_value": 0}))
        .unwrap();
        let mut seen = Vec::new();
        op.visit_slots(&mut |name, slot| seen.push((name, slot)));
        assert_eq!(seen, [("matrix", 3), ("output_size", 1)]);
        assert!(!op.is_static());
        assert_eq!(op.min_inputs(), 4);
    }

    #[test]
    fn per_row_values_resolve_from_their_column() {
        let op = parse(json!({"op": "resize", "height": {"$slot": 1}, "width": 4,
                                    "filter": {"$slot": 2}}))
        .unwrap();
        let inputs = [
            Series::new("img".into(), &[0i32, 0]),
            Series::new("h".into(), &[6i64, -1]),
            Series::new("f".into(), &["nearest", "nearest"]),
        ];
        let ctx = ParamCtx::with_null_policy(&inputs, Default::default());
        let step = op.resolve(0, &ctx).unwrap();
        assert!(format!("{step:?}").contains("height: 6"), "{step:?}");
        let err = op.resolve(1, &ctx).unwrap_err().to_string();
        assert!(
            err.contains("'h'") && err.contains("cannot be negative"),
            "{err}"
        );
    }

    #[test]
    fn warp_affine_rejects_a_singular_matrix() {
        // The runner used to substitute the identity for a singular matrix, so
        // a degenerate transform returned the input with no signal.
        let warp = |m: [f64; 6]| {
            let op = parse(json!({"op": "warp_affine", "matrix": m,
                                        "output_size": [8, 8],
                                        "interpolation": "bilinear",
                                        "border_value": 0.0}))
            .unwrap();
            op.resolve(0, &ParamCtx::empty())
        };
        let err = warp([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("singular") && err.contains("determinant"),
            "{err}"
        );
        // Poorly conditioned but invertible is fine.
        assert!(warp([1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0]).is_ok());
    }

    /// The committed catalogue is what `scripts/gen_ops.py` generates Python
    /// from. Regenerate with `POLARS_CV_BLESS=1 cargo test -p polars-cv
    /// catalog_matches`.
    #[test]
    fn catalog_matches_the_committed_file() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/op_catalog.json");
        let current = catalog_json();
        if std::env::var_os("POLARS_CV_BLESS").is_some() {
            std::fs::write(path, &current).unwrap();
        }
        let committed = std::fs::read_to_string(path).unwrap_or_default();
        assert!(
            committed == current,
            "tests/golden/op_catalog.json is stale; regenerate with \
             POLARS_CV_BLESS=1 cargo test -p polars-cv catalog_matches, then \
             python scripts/gen_ops.py"
        );
    }
}
