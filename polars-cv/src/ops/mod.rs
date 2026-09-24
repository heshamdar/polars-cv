//! The typed op catalogue: one Rust definition per operation.
//!
//! An op is a struct deriving [`Op`](polars_cv_macros::Op) (its fields are
//! [`Param`]/[`Literal`] compositions) plus an [`OpDef`] impl that resolves it
//! to a [`GraphStep`], and one line in [`typed_ops!`]. From that:
//!
//! - **serde** rejects an unknown op, an unknown or missing field, a wrong
//!   type, an out-of-range value and a per-row value for a structural field;
//! - **the compiler** rejects an op with no `OpDef`, and a field the `OpDef`
//!   does not use (each impl opens with an exhaustive destructure, and an
//!   unused binding is a `-D warnings` error);
//! - **the catalogue** ([`catalog_json`], committed as
//!   `tests/golden/op_catalog.json`) is what `scripts/gen_ops.py` generates
//!   the Python builder methods from.
//!
//! Migration (typed-op plan P2–P6): ops not yet listed here are still resolved
//! by name through `execute::resolve_op`'s legacy table (`LEGACY_OPS`).
//! [`crate::pipeline::OpSpec`]'s deserializer picks the path by name, and a
//! name in neither set is an error.

pub mod affine;
pub mod binary;
pub mod channel;
pub mod color;
pub mod compute;
pub mod filter;
pub mod geometry;
pub mod histogram;
pub mod image;
pub mod label;
pub mod param;
pub mod phash;
pub mod reduce;
pub mod view;

use polars::prelude::*;
use serde::Serialize;

use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
pub use param::{ColumnRef, FieldType, Literal, NodeRef, Param, TypeDesc};

/// One field of an op, as the catalogue describes it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FieldDesc {
    pub name: &'static str,
    pub doc: &'static str,
    /// Positional-or-keyword in Python (otherwise keyword-only).
    pub positional: bool,
    /// The Python signature default; absent means required (or `None` for an
    /// optional field).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(rename = "type")]
    pub ty: TypeDesc,
}

/// What `#[derive(Op)]` emits for an op struct.
pub trait OpFields {
    /// The op's doc comment: the generated docstring's body.
    const DOC: &'static str;
    /// The Python method name, when it differs from the wire name.
    const PYTHON_NAME: Option<&'static str>;
    /// `public`, `lazy_only` or `internal`.
    const VISIBILITY: &'static str;
    /// Every field, in declaration (= Python signature) order.
    fn fields() -> Vec<FieldDesc>;
    /// Call `f(field, slot)` for every slot any field reads.
    fn visit_slots(&self, f: &mut dyn FnMut(&'static str, usize));
}

/// An op's execution: resolve it, for one row, to the step the engine runs.
///
/// No default methods (see CLAUDE.md, "No defaulted contract methods").
pub trait OpDef: OpFields {
    /// The step for `row`. Per-row parameters read their column here; an op
    /// with none is resolved once at graph compile time.
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep>;
}

/// One op in the catalogue.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OpDesc {
    /// The wire name.
    pub name: &'static str,
    /// The Python method name.
    pub python: &'static str,
    pub visibility: &'static str,
    pub doc: &'static str,
    pub fields: Vec<FieldDesc>,
}

impl OpDesc {
    fn of<T: OpFields>(name: &'static str) -> Self {
        OpDesc {
            name,
            python: T::PYTHON_NAME.unwrap_or(name),
            visibility: T::VISIBILITY,
            doc: T::DOC,
            fields: T::fields(),
        }
    }
}

/// Register the typed ops: wire name, `TypedOp` variant, struct, and a valid
/// sample of its wire fields.
///
/// Registering is the whole act: the line is what makes the op
/// deserializable, resolvable, described and — through the sample — covered by
/// the registry-driven tests, so there is no second list.
macro_rules! typed_ops {
    ($($wire:literal => $variant:ident($ty:ty) $sample:tt),+ $(,)?) => {
        /// A typed operation (see the module docs).
        #[derive(Debug, Clone, PartialEq)]
        pub enum TypedOp {
            $($variant($ty)),+
        }

        impl TypedOp {
            /// Every typed op's wire name, sorted.
            pub const NAMES: &'static [&'static str] = &[$($wire),+];

            /// The op's wire name.
            pub fn name(&self) -> &'static str {
                match self {
                    $(TypedOp::$variant(_) => $wire),+
                }
            }

            /// Deserialize the op `name` from its fields (the wire object
            /// without `"op"`). `None` when `name` is not a typed op.
            pub fn from_fields(
                name: &str,
                fields: serde_json::Value,
            ) -> Option<Result<Self, String>> {
                match name {
                    $($wire => Some(
                        serde_path_to_error::deserialize::<_, $ty>(fields)
                            .map(TypedOp::$variant)
                            .map_err(|e| path_error(&e)),
                    ),)+
                    _ => None,
                }
            }

            /// The op's fields as a wire object (without `"op"`).
            pub fn fields_json(&self) -> serde_json::Value {
                match self {
                    $(TypedOp::$variant(op) => serde_json::to_value(op),)+
                }
                .expect("an op struct serializes to a JSON object")
            }

            /// See [`OpDef::resolve`].
            pub fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                match self {
                    $(TypedOp::$variant(op) => OpDef::resolve(op, row, ctx),)+
                }
            }

            /// See [`OpFields::visit_slots`].
            pub fn visit_slots(&self, f: &mut dyn FnMut(&'static str, usize)) {
                match self {
                    $(TypedOp::$variant(op) => op.visit_slots(f),)+
                }
            }

            /// One valid instance of every typed op, in `NAMES` order.
            #[cfg(test)]
            pub fn samples() -> Vec<TypedOp> {
                vec![$(
                    TypedOp::from_fields($wire, serde_json::json!($sample))
                        .expect("registered")
                        .unwrap_or_else(|e| panic!("sample for '{}': {e}", $wire)),
                )+]
            }

            /// Every typed op's description, in `NAMES` order.
            pub fn catalog() -> Vec<OpDesc> {
                vec![$(OpDesc::of::<$ty>($wire)),+]
            }
        }
    };
}

typed_ops! {
    "abs" => Abs(compute::Abs) {},
    "add" => Add(binary::Add) {"other": "n0"},
    "add_constant" => AddConstant(compute::AddConstant) {"value": 1.0},
    "adjust_contrast" => AdjustContrast(compute::AdjustContrast) {"factor": 1.5},
    "adjust_gamma" => AdjustGamma(compute::AdjustGamma) {"gamma": 0.5},
    "apply_mask" => ApplyMask(binary::ApplyMask) {"mask": "n0", "invert": true},
    "bitwise_and" => BitwiseAnd(binary::BitwiseAnd) {"other": "n0"},
    "bitwise_or" => BitwiseOr(binary::BitwiseOr) {"other": "n0"},
    "bitwise_xor" => BitwiseXor(binary::BitwiseXor) {"other": "n0"},
    "blend" => Blend(binary::Blend) {"other": "n0"},
    "blur" => Blur(image::Blur) {"sigma": 1.0},
    "canny" => Canny(image::Canny) {"low_threshold": 50.0, "high_threshold": 150.0},
    "cast" => Cast(compute::Cast) {"dtype": "f32"},
    "ceil" => Ceil(compute::Ceil) {},
    "channel_merge" => ChannelMerge(binary::ChannelMerge) {"others": ["n0", "n1"]},
    "channel_select" => ChannelSelect(channel::ChannelSelect) {"index": 0},
    "channel_swap" => ChannelSwap(channel::ChannelSwap) {"order": [2, 1, 0]},
    "clamp" => Clamp(compute::Clamp) {"min": 0.0, "max": 1.0},
    "clamp_max" => ClampMax(compute::ClampMax) {"value": 1.0},
    "clamp_min" => ClampMin(compute::ClampMin) {"value": 0.0},
    "contour_area" => ContourArea(geometry::ContourArea) {"signed": false},
    "contour_bounding_box" => ContourBoundingBox(geometry::ContourBoundingBox) {},
    "contour_centroid" => ContourCentroid(geometry::ContourCentroid) {},
    "contour_convex_hull" => ContourConvexHull(geometry::ContourConvexHull) {},
    "contour_perimeter" => ContourPerimeter(geometry::ContourPerimeter) {},
    "contour_scale" => ContourScale(geometry::ContourScale) {"sx": 2.0, "sy": 0.5, "origin": "bbox_center"},
    "contour_simplify" => ContourSimplify(geometry::ContourSimplify) {"tolerance": 1.5},
    "contour_translate" => ContourTranslate(geometry::ContourTranslate) {"dx": 1.0, "dy": -2.0},
    "convolve2d" => Convolve2d(filter::Convolve2d)
        {"kernel": [0, 0, 0, 0, 1, 0, 0, 0, 0], "ksize": 3, "normalize": false, "border": "replicate"},
    "crop" => Crop(view::Crop) {"top": 1, "left": 1, "height": 2, "width": 2},
    "cvt_color" => CvtColor(color::CvtColor) {"from_space": "rgb", "to_space": "hsv"},
    "dilate" => Dilate(image::Dilate) {"ksize": 3, "iterations": 1},
    "divide" => Divide(binary::Divide) {"other": "n0"},
    "equalize_histogram" => EqualizeHistogram(image::EqualizeHistogram) {},
    "erode" => Erode(image::Erode) {"ksize": 3, "iterations": 1},
    "extract_contours" => ExtractContours(geometry::ExtractContours)
        {"mode": "tree", "method": "none", "min_area": 2.0},
    "extract_shape" => ExtractShape(reduce::ExtractShape) {},
    "flip" => Flip(view::Flip) {"axes": [1]},
    "floor" => Floor(compute::Floor) {},
    "grayscale" => Grayscale(image::Grayscale) {},
    "histogram" => Histogram(histogram::Histogram)
        {"bins": 8, "range": null, "closed": "left", "output": "counts"},
    "invert" => Invert(compute::Invert) {},
    "label_reduce" => LabelReduce(label::LabelReduce)
        {"contours": {"$slot": 1}, "reduction": "mean", "region_mode": "bbox"},
    "letterbox" => Letterbox(image::Letterbox)
        {"height": 4, "width": 4, "value": 0.0, "filter": "bilinear"},
    "maximum" => Maximum(binary::Maximum) {"other": "n0"},
    "minimum" => Minimum(binary::Minimum) {"other": "n0"},
    "morphology_gradient" => MorphologyGradient(image::MorphologyGradient) {"ksize": 3},
    "multiply" => Multiply(binary::Multiply) {"other": "n0"},
    "neg" => Neg(compute::Neg) {},
    "normalize" => Normalize(compute::Normalize)
        {"method": "preset", "mean": [0.5], "std": [0.25], "out_dtype": "f32"},
    "pad" => Pad(image::Pad)
        {"top": 1, "bottom": 1, "left": 1, "right": 1, "value": 0.0, "mode": "constant"},
    "pad_to_size" => PadToSize(image::PadToSize)
        {"height": 4, "width": 4, "position": "center", "value": 0.0},
    "perceptual_hash" => PerceptualHash(phash::PerceptualHash) {"algorithm": "perceptual", "hash_size": 64},
    "rasterize" => Rasterize(geometry::Rasterize) {"size": [8, 6], "fill_value": 1, "background": 0},
    "ratio" => Ratio(binary::Ratio) {"other": "n0"},
    "reciprocal" => Reciprocal(compute::Reciprocal) {},
    "reduce_argmax" => ReduceArgmax(reduce::ReduceArgmax) {"axis": 0},
    "reduce_argmin" => ReduceArgmin(reduce::ReduceArgmin) {"axis": 0},
    "reduce_max" => ReduceMax(reduce::ReduceMax) {"axis": null},
    "reduce_mean" => ReduceMean(reduce::ReduceMean) {"axis": 1},
    "reduce_min" => ReduceMin(reduce::ReduceMin) {"axis": null},
    "reduce_percentile" => ReducePercentile(reduce::ReducePercentile) {"q": 50.0},
    "reduce_popcount" => ReducePopcount(reduce::ReducePopcount) {},
    "reduce_std" => ReduceStd(reduce::ReduceStd) {"axis": null, "ddof": 1},
    "reduce_sum" => ReduceSum(reduce::ReduceSum) {},
    "relu" => Relu(compute::Relu) {},
    "reshape" => Reshape(view::Reshape) {"shape": [2, 2, 1]},
    "resize" => Resize(image::Resize) {"height": 4, "width": 4, "filter": "bilinear"},
    "resize_max" => ResizeMax(image::ResizeMax) {"max_size": 4, "filter": "bilinear"},
    "resize_min" => ResizeMin(image::ResizeMin) {"min_size": 4, "filter": "bilinear"},
    "resize_scale" => ResizeScale(image::ResizeScale) {"scale_x": 0.5, "scale_y": 0.5, "filter": "bilinear"},
    "resize_to_height" => ResizeToHeight(image::ResizeToHeight) {"height": 4, "filter": "bilinear"},
    "resize_to_width" => ResizeToWidth(image::ResizeToWidth) {"width": 4, "filter": "bilinear"},
    "rotate" => Rotate(affine::Rotate)
        {"angle": 30.0, "expand": true, "interpolation": "nearest", "border_value": 0.0},
    "round" => Round(compute::Round) {},
    "scale" => Scale(compute::Scale) {"factor": 2.0},
    "sign" => Sign(compute::Sign) {},
    "sqrt" => Sqrt(compute::Sqrt) {},
    "square" => Square(compute::Square) {},
    "subtract" => Subtract(binary::Subtract) {"other": "n0"},
    "subtract_constant" => SubtractConstant(compute::SubtractConstant) {"value": 1.0},
    "threshold" => Threshold(image::Threshold) {"value": 128.0},
    "transpose" => Transpose(view::Transpose) {"axes": [1, 0, 2]},
    "trunc" => Trunc(compute::Trunc) {},
    "warp_affine" => WarpAffine(affine::WarpAffine) {
        "matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "output_size": [4, 4],
        "interpolation": "bilinear",
        "border_value": 0.0
    },
}

impl TypedOp {
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

fn path_error(e: &serde_path_to_error::Error<serde_json::Error>) -> String {
    let path = e.path().to_string();
    if path == "." {
        e.inner().to_string()
    } else {
        format!("'{path}': {}", e.inner())
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
    use crate::execute::LEGACY_OPS;
    use crate::pipeline::OpSpec;
    use polars_cv_macros::Op;
    use serde::Deserialize;
    use serde_json::json;
    use std::collections::BTreeSet;

    fn parse(v: serde_json::Value) -> Result<OpSpec, String> {
        serde_json::from_value(v).map_err(|e| e.to_string())
    }

    fn parse_err(v: serde_json::Value) -> String {
        parse(v).expect_err("expected the spec to be rejected")
    }

    /// The ops executable before the migration (P0's name registry). Typed and
    /// legacy must partition exactly this set, so migrating an op moves it
    /// rather than dropping or duplicating it. An op added on purpose is
    /// added here too.
    const OP_SET: &[&str] = &[
        "abs",
        "add",
        "add_constant",
        "adjust_contrast",
        "adjust_gamma",
        "apply_mask",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "blend",
        "blur",
        "canny",
        "cast",
        "ceil",
        "channel_merge",
        "channel_select",
        "channel_swap",
        "clamp",
        "clamp_max",
        "clamp_min",
        "contour_area",
        "contour_bounding_box",
        "contour_centroid",
        "contour_convex_hull",
        "contour_perimeter",
        "contour_scale",
        "contour_simplify",
        "contour_translate",
        "convolve2d",
        "crop",
        "cvt_color",
        "dilate",
        "divide",
        "equalize_histogram",
        "erode",
        "extract_contours",
        "extract_shape",
        "flip",
        "floor",
        "grayscale",
        "histogram",
        "invert",
        "label_reduce",
        "letterbox",
        "maximum",
        "minimum",
        "morphology_gradient",
        "multiply",
        "neg",
        "normalize",
        "pad",
        "pad_to_size",
        "perceptual_hash",
        "rasterize",
        "ratio",
        "reciprocal",
        "reduce_argmax",
        "reduce_argmin",
        "reduce_max",
        "reduce_mean",
        "reduce_min",
        "reduce_percentile",
        "reduce_popcount",
        "reduce_std",
        "reduce_sum",
        "relu",
        "reshape",
        "resize",
        "resize_max",
        "resize_min",
        "resize_scale",
        "resize_to_height",
        "resize_to_width",
        "rotate",
        "round",
        "scale",
        "sign",
        "sqrt",
        "square",
        "subtract",
        "subtract_constant",
        "threshold",
        "transpose",
        "trunc",
        "warp_affine",
    ];

    #[test]
    fn typed_and_legacy_ops_partition_the_op_set() {
        let typed: BTreeSet<&str> = TypedOp::NAMES.iter().copied().collect();
        let legacy: BTreeSet<&str> = LEGACY_OPS.iter().copied().collect();
        let both: Vec<_> = typed.intersection(&legacy).collect();
        assert!(both.is_empty(), "ops both typed and legacy: {both:?}");
        let all: BTreeSet<&str> = typed.union(&legacy).copied().collect();
        let expected: BTreeSet<&str> = OP_SET.iter().copied().collect();
        assert_eq!(all, expected, "typed ∪ legacy must be the op set");
    }

    #[test]
    fn typed_names_are_sorted_and_unique() {
        assert!(TypedOp::NAMES.windows(2).all(|w| w[0] < w[1]));
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
    }

    #[test]
    fn a_missing_required_field_is_rejected() {
        let err = parse_err(json!({"op": "resize", "height": 4, "filter": "nearest"}));
        assert!(err.contains("width"), "{err}");
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
                json!({"op": "convolve2d", "kernel": [0, 0, 0, 0, 1, 0, 0, 0, 0], "ksize": 3,
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
            match typed.resolve(0, &ParamCtx::empty()).unwrap() {
                GraphStep::Binary { op: got, other } => {
                    assert_eq!((got, other.as_str()), (*op, "n0"), "{name}")
                }
                step => panic!("'{name}' resolves to {step:?}"),
            }
        }
        let binary = TypedOp::samples()
            .into_iter()
            .filter(|op| {
                matches!(
                    op.resolve(0, &ParamCtx::empty()),
                    Ok(GraphStep::Binary { .. })
                )
            })
            .count();
        assert_eq!(binary, BinaryOp::NAMED.len());
    }

    /// `rasterize(shape=<node>)` takes its canvas from another node's buffer,
    /// which only the graph executor has. A plan-time probe sees dimensions
    /// that vary with the probe (so the planner reports them unknown); any
    /// other resolution is a compile path that skipped the executor's
    /// special case, and must fail rather than invent a size.
    #[test]
    fn a_node_sized_rasterize_resolves_only_under_a_probe() {
        let op = TypedOp::from_fields(
            "rasterize",
            json!({"size": "n0", "fill_value": 255, "background": 0}),
        )
        .unwrap()
        .unwrap();
        let err = op.resolve(0, &ParamCtx::empty()).unwrap_err().to_string();
        assert!(err.contains("graph executor"), "{err}");
        let dims = |probe: i64| match op.resolve(0, &ParamCtx::probe(&[], probe)).unwrap() {
            GraphStep::Geometry(view_buffer::GeometryOp::Rasterize { width, height, .. }) => {
                (width, height)
            }
            step => panic!("{step:?}"),
        };
        assert_ne!(dims(3), dims(5));
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
        let resolve = |v: serde_json::Value| match parse(v).unwrap() {
            OpSpec::Typed(op) => op.resolve(0, &ParamCtx::empty()).map(|_| ()),
            OpSpec::Legacy(_) => unreachable!(),
        };
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

    #[test]
    fn an_unknown_operation_is_rejected() {
        let err = parse_err(json!({"op": "definitely_not_a_real_op"}));
        assert!(err.contains("Unknown operation"), "{err}");
    }

    #[test]
    fn an_extra_key_on_a_no_field_op_is_rejected() {
        /// A test op with no parameters.
        #[derive(Debug, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        struct NoFields {}
        assert!(serde_json::from_value::<NoFields>(json!({})).is_ok());
        assert!(serde_json::from_value::<NoFields>(json!({"x": 1})).is_err());
        assert!(NoFields::fields().is_empty());
    }

    #[test]
    fn samples_round_trip_through_the_wire() {
        for op in TypedOp::samples() {
            let wire = serde_json::to_value(OpSpec::Typed(op.clone())).unwrap();
            assert_eq!(wire["op"], op.name());
            match parse(wire).unwrap() {
                OpSpec::Typed(back) => assert_eq!(back, op),
                OpSpec::Legacy(_) => panic!("{} came back legacy", op.name()),
            }
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
        let op = match parse(json!({"op": "warp_affine",
                                    "matrix": [1, 0, {"$slot": 3}, 0, 1, 0],
                                    "output_size": [{"$slot": 1}, 4],
                                    "interpolation": "bilinear", "border_value": 0}))
        .unwrap()
        {
            OpSpec::Typed(op) => op,
            OpSpec::Legacy(_) => unreachable!(),
        };
        let mut seen = Vec::new();
        op.visit_slots(&mut |name, slot| seen.push((name, slot)));
        assert_eq!(seen, [("matrix", 3), ("output_size", 1)]);
        assert!(!op.is_static());
        assert_eq!(op.min_inputs(), 4);
    }

    #[test]
    fn per_row_values_resolve_from_their_column() {
        let op = match parse(json!({"op": "resize", "height": {"$slot": 1}, "width": 4,
                                    "filter": {"$slot": 2}}))
        .unwrap()
        {
            OpSpec::Typed(op) => op,
            OpSpec::Legacy(_) => unreachable!(),
        };
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
            let op = match parse(json!({"op": "warp_affine", "matrix": m,
                                        "output_size": [8, 8],
                                        "interpolation": "bilinear",
                                        "border_value": 0.0}))
            .unwrap()
            {
                OpSpec::Typed(op) => op,
                OpSpec::Legacy(_) => unreachable!(),
            };
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
