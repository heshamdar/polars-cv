//! Geometry operation enum for pipeline integration.

use crate::mode::{known, Exec, FieldType, Mode, NodeRef, Param, TypeDesc, Wire};
use polars_cv_macros::{Ops, Resolve};

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use crate::ops::Domain;

/// Origin point for scale operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleOrigin {
    /// Scale around the contour's centroid.
    Centroid,
    /// Scale around the bounding box center.
    BBoxCenter,
    /// Scale around the coordinate origin (0, 0).
    Origin,
}

/// The geometry ops reachable from a `Pipeline` graph — one variant per wire
/// op (see `crate::mode`).
///
/// This enum is the *graph* vocabulary, not a catalogue of the geometry the crate
/// can do: contour operations that only make sense on an already-materialized
/// contour column — winding, flip, normalize, contains_point, IoU, Dice,
/// Hausdorff and friends — are standalone `.contour` namespace plugin functions
/// that call [`super::measures`], [`super::predicates`], [`super::pairwise`] and
/// [`super::transforms`] directly.
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
pub enum GeometryOp<M: Mode = Exec> {
    /// Compute the area of the contour using the Shoelace formula.
    ///
    /// The area of the region the contour describes: the exterior minus the
    /// union of its hole rings, in either winding direction. Overlapping or
    /// nested hole rings are not double-subtracted. One value per contour for
    /// a contour set.
    #[op(name = "contour_area", python = "area", sample = {"signed": false})]
    Area {
        /// If True, return signed area (negative for CW winding).
        #[param(default = false)]
        signed: M::V<bool>,
    },
    /// Compute the perimeter (arc length) of the contour.
    #[op(name = "contour_perimeter", python = "perimeter", sample = {})]
    Perimeter,
    /// Compute the centroid (center of mass) of the contour.
    ///
    /// Measured on the same region as `area()` — the exterior minus the union
    /// of the hole rings — so overlapping or nested holes are not subtracted
    /// twice.
    ///
    /// Returns ``[x, y]``.
    #[op(name = "contour_centroid", python = "centroid", sample = {})]
    Centroid,
    /// Compute the axis-aligned bounding box of the contour.
    ///
    /// Returns ``[x, y, width, height]``.
    #[op(name = "contour_bounding_box", python = "bounding_box", sample = {})]
    BoundingBox,
    /// Translate the contour by an offset.
    #[op(name = "contour_translate", python = "translate", sample = {"dx": 1.0, "dy": -2.0})]
    Translate {
        /// X offset (horizontal translation).
        dx: M::V<f64>,
        /// Y offset (vertical translation).
        dy: M::V<f64>,
    },
    /// Scale the contour about *origin*.
    #[op(name = "contour_scale", python = "scale_contour",
         sample = {"sx": 2.0, "sy": 0.5, "origin": "bbox_center"})]
    Scale {
        /// X scale factor.
        sx: M::V<f64>,
        /// Y scale factor.
        sy: M::V<f64>,
        /// Point to scale about — ``"centroid"`` (center of mass),
        /// ``"bbox_center"`` (bounding-box center) or ``"origin"`` (the
        /// coordinate origin ``(0, 0)``). Accepts an expression for a per-row
        /// choice: which point the scale is measured from changes no output
        /// shape, rank or dtype.
        #[param(default = "centroid")]
        origin: M::V<ScaleOrigin>,
    },
    /// Simplify the contour using the Douglas-Peucker algorithm.
    #[op(name = "contour_simplify", python = "simplify", sample = {"tolerance": 1.5})]
    Simplify {
        /// Maximum distance from the original contour.
        tolerance: M::V<f64>,
    },
    /// Compute the convex hull of the contour.
    #[op(name = "contour_convex_hull", python = "convex_hull", sample = {})]
    ConvexHull,
    /// Rasterize contours to a mask.
    ///
    /// The builder is ``Pipeline.rasterize``, whose ``width``/``height`` or
    /// ``shape`` arguments become ``size``; it also records the shape reference's
    /// graph dependency and its canvas assertion.
    #[op(name = "rasterize", visibility = Internal,
         sample = {"size": [8, 6], "fill_value": 1, "background": 0})]
    Rasterize {
        /// ``[height, width]`` of the mask (each may be a Polars expression), or
        /// another node whose buffer's height and width the mask takes.
        size: RasterSize<M>,
        /// Inside value (default 255). Accepts a Polars expression for per-row
        /// dynamic values.
        #[param(default = 255)]
        fill_value: M::V<u8>,
        /// Outside value (default 0). Accepts a Polars expression for per-row
        /// dynamic values.
        #[param(default = 0)]
        background: M::V<u8>,
    },
    /// Extract contours from binary mask.
    ///
    /// The traced outline passes through the **centres** of the boundary pixels,
    /// so it sits half a pixel inside the region it describes: a blob filling
    /// ``w x h`` pixels comes back bounding ``(w-1) x (h-1)``. Rasterizing the
    /// result therefore erodes it by a pixel per round trip.
    ///
    /// Borders come back as a flat list with no hierarchy. ``mode="all"`` yields
    /// the exterior plus one border for each enclosed background region — holes
    /// that touch or nest enclose one region between them — and reassembling a
    /// holed contour from those is the caller's job. ``mode="external"`` keeps
    /// only the outermost, discarding hole borders.
    #[op(name = "extract_contours", sample = {"mode": "tree", "method": "none", "min_area": 2.0})]
    ExtractContours {
        /// "external" (outer only), "tree" (full hierarchy), "all".
        #[param(default = "external")]
        mode: M::V<ExtractMode>,
        /// "simple" (remove redundant), "none" (all points), "approx".
        #[param(default = "simple")]
        method: M::V<ApproxMethod>,
        /// Filter small contours. Accepts a Polars expression for per-row dynamic
        /// thresholds.
        min_area: Option<M::V<f64>>,
    },
}

/// Where a rasterized mask's canvas size comes from.
///
/// Two variants rather than optional width/height plus an optional node, so a
/// spec cannot carry both and have one ignored. On the wire a string is a node
/// id, anything else the `[height, width]` pair.
#[derive(Debug, Clone, PartialEq, Resolve)]
pub enum RasterSize<M: Mode = Exec> {
    /// Explicit `[height, width]`.
    Fixed([M::V<u32>; 2]),
    /// The height and width of another node's buffer, known only when the
    /// graph executor has run that node: it sets the canvas
    /// ([`GeometryOp::with_canvas`]) before the op executes.
    FromNode(NodeRef),
}

impl serde::Serialize for RasterSize<Wire> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            RasterSize::Fixed(dims) => dims.serialize(s),
            RasterSize::FromNode(node) => node.serialize(s),
        }
    }
}

impl<'de> serde::Deserialize<'de> for RasterSize<Wire> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let value = serde_json::Value::deserialize(d)?;
        if value.is_string() {
            serde_json::from_value(value).map(RasterSize::FromNode)
        } else {
            serde_json::from_value(value).map(RasterSize::Fixed)
        }
        .map_err(D::Error::custom)
    }
}

impl FieldType for RasterSize<Wire> {
    fn describe() -> TypeDesc {
        TypeDesc::OneOf {
            options: vec![
                <[Param<u32>; 2] as FieldType>::describe(),
                <NodeRef as FieldType>::describe(),
            ],
        }
    }
    fn visit_slots(&self, f: &mut dyn FnMut(usize)) {
        match self {
            RasterSize::Fixed(dims) => dims.visit_slots(f),
            RasterSize::FromNode(node) => node.visit_slots(f),
        }
    }
}

impl<M: Mode> GeometryOp<M> {
    /// Refuse a parameter combination no row can execute: none (the
    /// per-value checks — a positive canvas, a non-negative tolerance — are
    /// `validate`'s, on the values).
    pub fn check(&self) -> Result<(), String> {
        Ok(())
    }

    /// How this op's output shape follows from its input — the one
    /// definition.
    pub fn shape(&self) -> OpShape {
        match self {
            // A measure runs over the row's contour set: one value (a
            // centroid's two, a box's four) per member, so the length is the
            // set's size — known only with the data.
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::Centroid
            | GeometryOp::BoundingBox => OpShape::Dynamic,
            // Contour transforms preserve the point list; `Simplify` and
            // `ConvexHull` may shorten it, which is not knowable statically, so
            // the input shape stands in for both.
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull => OpShape::Preserve,
            GeometryOp::Rasterize { size, .. } => {
                let (h, w) = match size {
                    RasterSize::Fixed([h, w]) => {
                        (crate::mode::size::<M>(h), crate::mode::size::<M>(w))
                    }
                    // Another node's canvas: known only once that node has run.
                    RasterSize::FromNode(_) => (Sym::PerRow, Sym::PerRow),
                };
                OpShape::Fixed(vec![h, w, Sym::Known(1)])
            }
            // ExtractContours output shape is data-dependent
            GeometryOp::ExtractContours { .. } => OpShape::Dynamic,
        }
    }
}

impl GeometryOp {
    /// This rasterize op with its canvas set to `height` x `width` (another
    /// node's buffer's, read by the executor). Any other op is unchanged.
    pub fn with_canvas(self, height: u32, width: u32) -> GeometryOp {
        match self {
            GeometryOp::Rasterize {
                fill_value,
                background,
                ..
            } => GeometryOp::Rasterize {
                size: RasterSize::Fixed([height, width]),
                fill_value,
                background,
            },
            other => other,
        }
    }

    /// A rasterize op's canvas, `(height, width)`, once it is known.
    pub fn canvas(&self) -> Option<(u32, u32)> {
        match self {
            GeometryOp::Rasterize {
                size: RasterSize::Fixed([h, w]),
                ..
            } => Some((*h, *w)),
            _ => None,
        }
    }
}

/// Mode for contour extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractMode {
    /// Only outermost contours (no nesting).
    External,
    /// Full hierarchy with parent-child relationships.
    Tree,
    /// All contours flattened (no hierarchy).
    All,
}

/// Contour approximation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApproxMethod {
    /// Keep all boundary points.
    None,
    /// Remove redundant points on straight lines.
    Simple,
    /// Douglas-Peucker approximation.
    Approx,
}

crate::naming::named_variants!(ScaleOrigin: "Point a contour scale operation is measured from (``.contour.scale``)." {
    "centroid" => Centroid,
    "bbox_center" => BBoxCenter,
    "origin" => Origin,
});

crate::naming::named_variants!(ExtractMode: "Contour retrieval mode for ``extract_contours``.\n\n- EXTERNAL: Outermost contours only (default).\n- TREE: Full nesting hierarchy.\n- ALL: Every contour, without hierarchy." {
    "external" => External,
    "tree" => Tree,
    "all" => All,
});

crate::naming::named_variants!(ApproxMethod: "Contour point-approximation method for ``extract_contours``.\n\n- NONE: Keep every boundary point.\n- SIMPLE: Drop redundant collinear points (default).\n- APPROX: Douglas-Peucker style approximation." {
    "none" => None,
    "simple" => Simple,
    "approx" => Approx,
});

impl<M: Mode> Op for GeometryOp<M> {
    fn name(&self) -> &'static str {
        match self {
            GeometryOp::Area { .. } => "Area",
            GeometryOp::Perimeter => "Perimeter",
            GeometryOp::Centroid => "Centroid",
            GeometryOp::BoundingBox => "BoundingBox",
            GeometryOp::Translate { .. } => "Translate",
            GeometryOp::Scale { .. } => "Scale",
            GeometryOp::Simplify { .. } => "Simplify",
            GeometryOp::ConvexHull => "ConvexHull",
            GeometryOp::Rasterize { .. } => "Rasterize",
            GeometryOp::ExtractContours { .. } => "ExtractContours",
        }
    }

    fn shape(&self) -> OpShape {
        GeometryOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Every geometry op materializes a fresh contour, measure or mask.
        MemoryEffect::RequiresContiguous
    }

    fn identity_rule(&self) -> IdentityRule {
        // Computes / combines / reduces — never a removable no-op.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        // Geometry ops work in the contour/measure/mask domains, not on an
        // image-space H/W window.
        false
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // Geometry ops work in the contour/measure/mask domains, not on an
        // image-space window: extraction reads the whole image, measures reduce
        // a whole contour, transforms and rasterization remap coordinates. None
        // admits a buffer-space crop commuting through it, so the conservative,
        // reorder-blocking classification is Global for every variant. Matched
        // exhaustively (not a blanket) so a new variant must reconfirm this
        // rather than silently inherit Global.
        match self {
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::Centroid
            | GeometryOp::BoundingBox
            | GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull
            | GeometryOp::Rasterize { .. }
            | GeometryOp::ExtractContours { .. } => SpatialDependency::Global,
        }
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        // Geometry ops don't preserve strides
        None
    }

    fn validate(
        &self,
        _input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        match self {
            // A canvas another node sets is known only once that node has
            // run (the executor sets it, and refuses a rasterize without
            // one); a per-row size is checked per row.
            GeometryOp::Rasterize {
                size: RasterSize::Fixed([h, w]),
                ..
            } => {
                let zero = |d: &M::V<u32>| known::<M, u32>(d) == Some(0);
                if zero(h) || zero(w) {
                    return Err(ValidationError::InvalidParameter {
                        param: "width/height".to_string(),
                        reason: "Dimensions must be > 0".to_string(),
                    });
                }
                Ok(())
            }

            GeometryOp::Simplify { tolerance } => {
                if known::<M, f64>(tolerance).is_some_and(|t| t < 0.0) {
                    return Err(ValidationError::InvalidParameter {
                        param: "tolerance".to_string(),
                        reason: "Tolerance must be >= 0".to_string(),
                    });
                }
                Ok(())
            }

            _ => Ok(()),
        }
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Any
    }

    fn working_dtype(&self) -> Option<DType> {
        Some(DType::F64)
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            GeometryOp::Rasterize { .. } => OutputDTypeRule::Fixed(DType::U8),
            _ => OutputDTypeRule::Fixed(DType::F64),
        }
    }
}

impl<M: Mode> GeometryOp<M> {
    /// Get the input domain this geometry operation expects.
    pub fn input_domain(&self) -> Domain {
        match self {
            // Extraction: Buffer → Contour
            GeometryOp::ExtractContours { .. } => Domain::Buffer,

            // Rasterization: Contour → Buffer
            GeometryOp::Rasterize { .. } => Domain::Contour,

            // Measures and contour→contour transforms alike read a contour.
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::Centroid
            | GeometryOp::BoundingBox
            | GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull => Domain::Contour,
        }
    }

    /// Get the output domain this geometry operation produces.
    pub fn output_domain(&self) -> Domain {
        match self {
            // Extraction: Buffer → Contour
            GeometryOp::ExtractContours { .. } => Domain::Contour,

            // Rasterization: Contour → Buffer
            GeometryOp::Rasterize { .. } => Domain::Buffer,

            // Per-contour measures: one value (or coordinate group) per
            // extracted contour. Execution iterates every contour, so these
            // are vector outputs — Area/Perimeter previously declared Scalar
            // here, which silently nulled lazily-chained measures (the eager
            // Python builder masked it with a manual domain override).
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::Centroid
            | GeometryOp::BoundingBox => Domain::Vector,

            // Contour transforms preserve contour domain
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull => Domain::Contour,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The rules are generic over the mode; these tests read executed ops.
    type GeometryOp = super::GeometryOp<Exec>;

    #[test]
    fn test_op_names() {
        assert_eq!(GeometryOp::Area { signed: false }.name(), "Area");
        assert_eq!(GeometryOp::Perimeter.name(), "Perimeter");
        assert_eq!(
            GeometryOp::Rasterize {
                size: RasterSize::Fixed([100, 100]),
                fill_value: 255,
                background: 0,
            }
            .name(),
            "Rasterize"
        );
    }

    #[test]
    fn test_rasterize_shape() {
        let op: GeometryOp = GeometryOp::Rasterize {
            size: RasterSize::Fixed([100, 200]),
            fill_value: 255,
            background: 0,
        };
        let shape = op.shape().concrete(&[]);
        assert_eq!(shape, vec![100, 200, 1]);
    }

    #[test]
    fn test_validate_rasterize() {
        let op = GeometryOp::Rasterize {
            size: RasterSize::Fixed([100, 0]),
            fill_value: 255,
            background: 0,
        };
        assert!(op.validate(&[], &[]).is_err());
    }

    #[test]
    fn test_geometry_op_domains() {
        // ExtractContours: Buffer → Contour
        let extract = GeometryOp::ExtractContours {
            mode: ExtractMode::External,
            method: ApproxMethod::Simple,
            min_area: None,
        };
        assert_eq!(extract.input_domain(), Domain::Buffer);
        assert_eq!(extract.output_domain(), Domain::Contour);

        // Rasterize: Contour → Buffer
        let rasterize = GeometryOp::Rasterize {
            size: RasterSize::Fixed([100, 100]),
            fill_value: 255,
            background: 0,
        };
        assert_eq!(rasterize.input_domain(), Domain::Contour);
        assert_eq!(rasterize.output_domain(), Domain::Buffer);

        // Area: Contour → Vector (one area per extracted contour)
        let area = GeometryOp::Area { signed: false };
        assert_eq!(area.input_domain(), Domain::Contour);
        assert_eq!(area.output_domain(), Domain::Vector);

        // Translate: Contour → Contour
        let translate = GeometryOp::Translate { dx: 10.0, dy: 20.0 };
        assert_eq!(translate.input_domain(), Domain::Contour);
        assert_eq!(translate.output_domain(), Domain::Contour);

        // Centroid: Contour → Vector
        assert_eq!(GeometryOp::Centroid.input_domain(), Domain::Contour);
        assert_eq!(GeometryOp::Centroid.output_domain(), Domain::Vector);
    }
}
