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
    /// A region's rings as open lines: the exterior, then each hole, each
    /// returning to its first point. The result is a set (holes add rings);
    /// an open contour, already a line, is refused.
    #[op(name = "contour_boundary", python = "boundary", sample = {})]
    Boundary,
    /// Join an open line's ends into the region it bounds. A line that
    /// returns to its first point (as ``boundary()`` draws one) drops the
    /// repeat, so ``close()`` undoes ``boundary()`` on a region without holes.
    /// A closed contour is refused.
    #[op(name = "contour_close", python = "close", sample = {})]
    Close,
    /// Grow the set's regions by *distance* (negative: shrink them): every
    /// point within *distance* of their union, or inside it and at least
    /// ``-distance`` from its edge.
    ///
    /// The set is read as one union, so growing can merge regions and
    /// shrinking can split them or leave none: the result is always a set.
    /// Convex corners turn by *join*; concave ones stay sharp. Round arcs are
    /// polygons within 1/64 pixel of the true arc, or about ``distance / 2000``
    /// past a distance of 32 pixels. An open contour is refused.
    #[op(name = "contour_offset", python = "offset", sample = {"distance": -1.5, "join": "miter"})]
    Offset {
        /// How far to move the edge, in pixels: positive grows, negative
        /// shrinks (literal or expression).
        distance: M::V<f64>,
        /// ``"round"`` (an arc about each corner), ``"miter"`` (the edges
        /// extended to meet, cut off past 4 times *distance*) or ``"bevel"``
        /// (cut straight across). Accepts an expression.
        #[param(default = "round")]
        join: M::V<LineJoin>,
    },
    /// The union of the set's regions, as disjoint regions: overlapping
    /// members merge and a hole another member covers is filled. The result
    /// is always a set. An open contour is refused.
    #[op(name = "contour_union_all", python = "union_all", sample = {})]
    UnionAll,
    /// Round off corners by Chaikin's corner cutting: each pass replaces every
    /// edge by its points a quarter and three quarters along, doubling the
    /// point count. A region's rings are cut all the way round; an open line
    /// keeps its two ends, unless it returns to its first point (as
    /// ``boundary()`` draws one), when it is cut through it as a loop.
    #[op(name = "contour_smooth", python = "smooth", sample = {"iterations": 2})]
    Smooth {
        /// Passes to make; 0 returns the contour as it is (literal or
        /// expression).
        #[param(default = 1)]
        iterations: M::V<u32>,
    },
    /// Add points so no edge is longer than *max_segment_length*: each edge is
    /// split into ``ceil(length / max_segment_length)`` equal parts, and every
    /// existing point is kept. A region's closing edges are split too.
    #[op(name = "contour_densify", python = "densify", sample = {"max_segment_length": 2.5})]
    Densify {
        /// The longest edge allowed, a finite length > 0 (literal or
        /// expression).
        max_segment_length: M::V<f64>,
    },
    /// The region the set's paths cover when stroked, as disjoint regions:
    /// every point within ``width / 2`` of a path, its ends shaped by *cap*
    /// and its corners by *join*. Every member is read as its outline: an
    /// open line, capped at its ends (one returning to its first point, as
    /// ``boundary()`` draws one, is a closed path), and a region's rings,
    /// each a closed path with no ends. A ring of fewer than two points has
    /// no edge and covers nothing; a one-point line is a dot (its caps). The result is always a set, filled
    /// by ``rasterize()``: ``stroke(width=2).rasterize(...)`` draws lines and
    /// outlines. Round arcs are polygons within 1/64 pixel of the true arc,
    /// or about ``width / 4000`` past a width of 64 pixels.
    #[op(name = "contour_stroke", python = "stroke",
         sample = {"width": 2.5, "cap": "square", "join": "bevel"})]
    Stroke {
        /// The stroke's full width in pixels, a finite width > 0 (literal or
        /// expression).
        #[param(default = 1.0)]
        width: M::V<f64>,
        /// How an open line ends: ``"round"`` (a half disc), ``"square"`` (a
        /// half square) or ``"butt"`` (flat at its end point). Accepts an
        /// expression.
        #[param(default = "round")]
        cap: M::V<LineCap>,
        /// How a path turns a corner: ``"round"``, ``"miter"`` (cut off past 4
        /// half-widths) or ``"bevel"``. Accepts an expression.
        #[param(default = "round")]
        join: M::V<LineJoin>,
    },
    /// Keep the ``k`` largest contours of the set, by area: largest first,
    /// equal areas in their input order. A set of fewer keeps them all.
    #[op(name = "contour_largest", python = "largest", sample = {"k": 2})]
    Largest {
        /// How many contours to keep, >= 1 (literal or expression).
        #[param(default = 1)]
        k: M::V<u32>,
    },
    /// Rasterize contours to a mask: each region filled. An open line has no
    /// region to fill and is refused; ``stroke()`` it first to draw it.
    ///
    /// The builder is ``Pipeline.rasterize``, whose ``width``/``height`` or
    /// ``shape`` arguments become ``size``; it also records the shape reference's
    /// graph dependency and its canvas assertion.
    #[op(name = "rasterize", visibility = Internal,
         sample = {"size": [8, 6], "fill_value": 1, "background": 0, "supersample": 3})]
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
        /// Samples per pixel along each axis, 1 to 16: a pixel takes the
        /// fraction of its ``n x n`` sub-pixel centres inside, blended between
        /// ``background`` and ``fill_value`` (anti-aliasing). 1 (the default)
        /// samples the pixel centre alone, so a pixel is one value or the
        /// other. Accepts a Polars expression for per-row dynamic values.
        #[param(default = 1)]
        supersample: M::V<u32>,
    },
    /// Extract contours from binary mask.
    ///
    /// The traced outline runs along the **edges** of the boundary pixels (pixel
    /// ``(x, y)`` is the square ``[x, x+1] x [y, y+1]``), so it bounds exactly
    /// the pixels of its region: a blob filling ``w x h`` pixels comes back
    /// bounding ``w x h``, a single pixel as its unit square, and every region's
    /// area is its pixel count. Rasterizing the result gives back the mask.
    ///
    /// Regions are 8-connected: pixels touching only at a corner are one region,
    /// whose outline passes through that corner twice.
    ///
    /// Borders come back as a flat list with no hierarchy. ``mode="all"`` yields
    /// the exterior plus one border for each enclosed background region — holes
    /// that touch or nest enclose one region between them — and reassembling a
    /// holed contour from those is the caller's job. ``mode="external"`` keeps
    /// only the outermost, discarding hole borders.
    #[op(name = "extract_contours", sample = {"mode": "all", "method": "none", "min_area": 2.0})]
    ExtractContours {
        /// "external" (outer borders only) or "all" (hole borders too, as
        /// a flat list).
        #[param(default = "external")]
        mode: M::V<ExtractMode>,
        /// "simple" (remove redundant), "none" (all points), "approx".
        #[param(default = "simple")]
        method: M::V<ApproxMethod>,
        /// Filter small contours: the minimum area in pixels. Accepts a Polars
        /// expression for per-row dynamic thresholds.
        min_area: Option<M::V<f64>>,
        /// Filter small contours relative to the image: the minimum area as a
        /// fraction of its height x width, in (0, 1], resolved per image as it
        /// runs (a speck threshold that scales with resolution). Applies
        /// together with ``min_area``: a contour must pass both. Accepts a
        /// Polars expression.
        min_area_fraction: Option<M::V<f64>>,
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
    fn visit_inputs(&self, f: &mut dyn FnMut(crate::mode::RowInput)) {
        match self {
            RasterSize::Fixed(dims) => dims.visit_inputs(f),
            RasterSize::FromNode(node) => node.visit_inputs(f),
        }
    }
    fn bind_this(&mut self, planned: &crate::mode::PlannedThis<'_>) -> Result<(), String> {
        match self {
            RasterSize::Fixed(dims) => dims.bind_this(planned),
            RasterSize::FromNode(node) => node.bind_this(planned),
        }
    }
}

impl<M: Mode> GeometryOp<M> {
    /// Refuse a literal no row can execute — checked whatever the plan
    /// knows of the input (`validate` runs only over a known rank, which a
    /// contour set never has). The checks on per-row values run with the row.
    pub fn check(&self) -> Result<(), String> {
        match self {
            GeometryOp::Rasterize { supersample, .. }
                if known::<M, u32>(supersample).is_some_and(|n| !(1..=16).contains(&n)) =>
            {
                Err(format!(
                    "rasterize: supersample must be between 1 and 16, got {}",
                    known::<M, u32>(supersample).unwrap_or_default()
                ))
            }
            GeometryOp::Stroke { width, .. } => {
                known::<M, f64>(width).map_or(Ok(()), super::transforms::Stroke::check_width)
            }
            GeometryOp::Offset { distance, .. } => {
                known::<M, f64>(distance).map_or(Ok(()), super::transforms::check_offset_distance)
            }
            GeometryOp::Densify { max_segment_length } => known::<M, f64>(max_segment_length)
                .map_or(Ok(()), super::transforms::check_max_segment_length),
            GeometryOp::Largest { k } if known::<M, u32>(k) == Some(0) => {
                Err("largest: k must be >= 1".to_string())
            }
            GeometryOp::ExtractContours {
                min_area_fraction: Some(fraction),
                ..
            } => match known::<M, f64>(fraction) {
                Some(f) if !(f > 0.0 && f <= 1.0) => Err(format!(
                    "extract_contours: min_area_fraction must be in (0, 1], got {f}"
                )),
                _ => Ok(()),
            },
            _ => Ok(()),
        }
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
            // How many contours are kept is `min(k, set size)`: the set's
            // size is known only with the data, as is how many rings a
            // boundary draws.
            // So is how many regions an offset or a union leaves.
            GeometryOp::Largest { .. }
            | GeometryOp::Boundary
            | GeometryOp::Offset { .. }
            | GeometryOp::UnionAll
            | GeometryOp::Stroke { .. } => OpShape::Dynamic,
            // Contour transforms preserve the point list; `Simplify` and
            // `ConvexHull` may shorten it, which is not knowable statically, so
            // the input shape stands in for both.
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull
            | GeometryOp::Close
            | GeometryOp::Smooth { .. }
            | GeometryOp::Densify { .. } => OpShape::Preserve,
            GeometryOp::Rasterize { size, .. } => match size {
                RasterSize::Fixed([h, w]) => OpShape::Fixed(vec![
                    crate::mode::size::<M>(h),
                    crate::mode::size::<M>(w),
                    Sym::Known(1),
                ]),
                // Another node's canvas: its H and W, read from that node as
                // the step's second input (the planner passes its planned
                // shape; the executor sets the canvas before the op runs).
                RasterSize::FromNode(_) => OpShape::Canvas { of: 1, channels: 1 },
            },
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
                supersample,
                ..
            } => GeometryOp::Rasterize {
                size: RasterSize::Fixed([height, width]),
                fill_value,
                background,
                supersample,
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
    /// Every border, hole borders included, as a flat list (no hierarchy).
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

/// How a stroke or an offset turns a corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineJoin {
    /// An arc about the corner.
    Round,
    /// The edges extended until they meet, cut off past the miter limit
    /// ([`super::transforms::MITER_LIMIT`]).
    Miter,
    /// The corner cut straight across.
    Bevel,
}

/// How a stroke ends a line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineCap {
    /// A half disc past each end.
    Round,
    /// A half square past each end.
    Square,
    /// Flat at the end point itself.
    Butt,
}

crate::naming::named_variants!(LineJoin: "How ``stroke`` and ``offset`` turn a corner.\n\n- ROUND: An arc about the corner (default).\n- MITER: The edges extended to meet, cut off where the point would reach more than 4 half-widths out (SVG's default miter limit).\n- BEVEL: The corner cut straight across." {
    "round" => Round,
    "miter" => Miter,
    "bevel" => Bevel,
});

crate::naming::named_variants!(LineCap: "How ``stroke`` ends an open line.\n\n- ROUND: A half disc past each end (default).\n- SQUARE: A half square past each end.\n- BUTT: Flat at the end point itself." {
    "round" => Round,
    "square" => Square,
    "butt" => Butt,
});

crate::naming::named_variants!(ScaleOrigin: "Point a contour scale operation is measured from (``.contour.scale``)." {
    "centroid" => Centroid,
    "bbox_center" => BBoxCenter,
    "origin" => Origin,
});

crate::naming::named_variants!(ExtractMode: "Contour retrieval mode for ``extract_contours``.\n\n- EXTERNAL: Outermost contours only (default).\n- ALL: Every border, hole borders included, without hierarchy." {
    "external" => External,
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
            GeometryOp::Largest { .. } => "Largest",
            GeometryOp::Boundary => "Boundary",
            GeometryOp::Close => "Close",
            GeometryOp::Offset { .. } => "Offset",
            GeometryOp::UnionAll => "UnionAll",
            GeometryOp::Smooth { .. } => "Smooth",
            GeometryOp::Densify { .. } => "Densify",
            GeometryOp::Stroke { .. } => "Stroke",
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
            | GeometryOp::Largest { .. }
            | GeometryOp::Boundary
            | GeometryOp::Close
            | GeometryOp::Offset { .. }
            | GeometryOp::UnionAll
            | GeometryOp::Smooth { .. }
            | GeometryOp::Densify { .. }
            | GeometryOp::Stroke { .. }
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
        _input_shapes: &[&[crate::ops::Dim]],
        _input_dtypes: &[crate::PlannedDType],
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
            | GeometryOp::ConvexHull
            | GeometryOp::Largest { .. }
            | GeometryOp::Boundary
            | GeometryOp::Close
            | GeometryOp::Offset { .. }
            | GeometryOp::UnionAll
            | GeometryOp::Smooth { .. }
            | GeometryOp::Densify { .. }
            | GeometryOp::Stroke { .. } => Domain::Contour,
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

            // Contour transforms (and a selection of the set) stay contours
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull
            | GeometryOp::Largest { .. }
            | GeometryOp::Boundary
            | GeometryOp::Close
            | GeometryOp::Offset { .. }
            | GeometryOp::UnionAll
            | GeometryOp::Smooth { .. }
            | GeometryOp::Densify { .. }
            | GeometryOp::Stroke { .. } => Domain::Contour,
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
                supersample: 1,
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
            supersample: 1,
        };
        let shape = op.shape().concrete(&[]);
        assert_eq!(shape, Some(vec![100, 200, 1]));
    }

    #[test]
    fn test_validate_rasterize() {
        let op = GeometryOp::Rasterize {
            size: RasterSize::Fixed([100, 0]),
            fill_value: 255,
            background: 0,
            supersample: 1,
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
            min_area_fraction: None,
        };
        assert_eq!(extract.input_domain(), Domain::Buffer);
        assert_eq!(extract.output_domain(), Domain::Contour);

        // Rasterize: Contour → Buffer
        let rasterize = GeometryOp::Rasterize {
            size: RasterSize::Fixed([100, 100]),
            fill_value: 255,
            background: 0,
            supersample: 1,
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
