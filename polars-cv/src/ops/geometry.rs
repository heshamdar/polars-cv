//! Contour extraction (buffer → contour), rasterization (contour → buffer),
//! measures and transforms (contour → scalar, vector or contour).

#[allow(unused_imports)]
use crate::ops::ParamExt as _;
use polars::prelude::*;
use polars_cv_macros::Op;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use view_buffer::geometry::ops::{ApproxMethod, ExtractMode, ScaleOrigin};
use view_buffer::GeometryOp;

use super::{FieldType, NodeRef, OpDef, Param, TypeDesc};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::{Op as _, OpShape, Sym};

fn geometry(op: GeometryOp) -> PolarsResult<GraphStep> {
    Ok(GraphStep::Geometry(op))
}

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
///
/// Domain transition: buffer → contour
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct ExtractContours {
    /// "external" (outer only), "tree" (full hierarchy), "all".
    #[param(default = "external")]
    pub mode: Param<ExtractMode>,
    /// "simple" (remove redundant), "none" (all points), "approx".
    #[param(default = "simple")]
    pub method: Param<ApproxMethod>,
    /// Filter small contours. Accepts a Polars expression for per-row dynamic
    /// thresholds.
    pub min_area: Option<Param<f64>>,
}

impl OpDef for ExtractContours {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Dynamic)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ExtractContours {
            mode,
            method,
            min_area,
        } = self;
        geometry(GeometryOp::ExtractContours {
            mode: mode.resolve(row, ctx)?,
            method: method.resolve(row, ctx)?,
            min_area: min_area.map(|a| a.resolve(row, ctx)).transpose()?,
        })
    }
}

/// Compute the area of the contour using the Shoelace formula.
///
/// Domain transition: contour → scalar
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(python = "area")]
pub struct ContourArea {
    /// If True, return signed area (negative for CW winding).
    #[param(default = false)]
    pub signed: Param<bool>,
}

impl OpDef for ContourArea {
    fn shape(&self) -> Option<OpShape> {
        Some(GeometryOp::Area { signed: false }.shape())
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ContourArea { signed } = self;
        geometry(GeometryOp::Area {
            signed: signed.resolve(row, ctx)?,
        })
    }
}

/// Declare a contour op with no parameters.
macro_rules! contour_measures {
    ($($ty:ident $python:literal $doc:literal $domain:literal => $variant:ident;)+) => {$(
        #[doc = $doc]
        ///
        #[doc = $domain]
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
        #[serde(deny_unknown_fields)]
        #[op(python = $python)]
        pub struct $ty {}

        impl OpDef for $ty {
            fn shape(&self) -> Option<OpShape> {
                Some(GeometryOp::$variant.shape())
            }

            fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
                let $ty {} = self;
                geometry(GeometryOp::$variant)
            }
        }
    )+};
}

contour_measures! {
    ContourPerimeter "perimeter" "Compute the perimeter (arc length) of the contour."
        "Domain transition: contour → scalar" => Perimeter;
    ContourCentroid "centroid" "Compute the centroid (center of mass) of the contour."
        "Domain transition: contour → vector (returns [x, y])" => Centroid;
    ContourBoundingBox "bounding_box" "Compute the axis-aligned bounding box of the contour."
        "Domain transition: contour → vector (returns [x, y, width, height])" => BoundingBox;
    ContourConvexHull "convex_hull" "Compute the convex hull of the contour."
        "Domain: contour → contour" => ConvexHull;
}

/// Translate the contour by an offset.
///
/// Domain: contour → contour
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(python = "translate")]
pub struct ContourTranslate {
    /// X offset (horizontal translation).
    pub dx: Param<f64>,
    /// Y offset (vertical translation).
    pub dy: Param<f64>,
}

impl OpDef for ContourTranslate {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ContourTranslate { dx, dy } = self;
        geometry(GeometryOp::Translate {
            dx: dx.resolve(row, ctx)?,
            dy: dy.resolve(row, ctx)?,
        })
    }
}

/// Scale the contour about *origin*.
///
/// Domain: contour → contour
///
/// Note:
///     The default is ``"centroid"``, which is what this method has always
///     done — it previously hardcoded it with no way to choose. The
///     ``.contour.scale`` accessor defaults to ``"origin"`` instead; pass
///     *origin* explicitly if you need the two to agree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(python = "scale_contour", visibility = "internal")]
pub struct ContourScale {
    /// X scale factor.
    pub sx: Param<f64>,
    /// Y scale factor.
    pub sy: Param<f64>,
    /// Point to scale about — ``"centroid"`` (the default), ``"bbox_center"``
    /// or ``"origin"``. Accepts an expression for a per-row choice: which
    /// point the scale is measured from changes no output shape, rank or
    /// dtype, so it meets the eligibility rule for a per-row parameter.
    #[param(default = "centroid")]
    pub origin: Param<ScaleOrigin>,
}

impl OpDef for ContourScale {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ContourScale { sx, sy, origin } = self;
        geometry(GeometryOp::Scale {
            sx: sx.resolve(row, ctx)?,
            sy: sy.resolve(row, ctx)?,
            origin: origin.resolve(row, ctx)?,
        })
    }
}

/// Simplify the contour using the Douglas-Peucker algorithm.
///
/// Domain: contour → contour
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(python = "simplify")]
pub struct ContourSimplify {
    /// Maximum distance from the original contour.
    pub tolerance: Param<f64>,
}

impl OpDef for ContourSimplify {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ContourSimplify { tolerance } = self;
        geometry(GeometryOp::Simplify {
            tolerance: tolerance.resolve(row, ctx)?,
        })
    }
}

/// Rasterize contours to a mask.
///
/// The builder is ``Pipeline.rasterize``, whose ``width``/``height`` or
/// ``shape`` arguments become ``size``; it also records the shape reference's
/// graph dependency and its canvas assertion.
///
/// Domain transition: contour → buffer
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
#[op(visibility = "internal")]
pub struct Rasterize {
    /// ``[height, width]`` of the mask (each may be a Polars expression), or
    /// another node whose buffer's height and width the mask takes.
    pub size: RasterSize,
    /// Inside value (default 255). Accepts a Polars expression for per-row
    /// dynamic values.
    #[param(default = 255)]
    pub fill_value: Param<u8>,
    /// Outside value (default 0). Accepts a Polars expression for per-row
    /// dynamic values.
    #[param(default = 0)]
    pub background: Param<u8>,
}

/// Where a rasterized mask's canvas size comes from.
///
/// Two variants rather than optional width/height plus an optional node, so a
/// spec cannot carry both and have one ignored. On the wire a string is a node
/// id, anything else the `[height, width]` pair.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RasterSize {
    /// Explicit `[height, width]`.
    Fixed([Param<u32>; 2]),
    /// The height and width of another node's buffer, known only when the
    /// graph executor has run that node (`CompiledGraph`'s
    /// `OpResolver::RasterizeShapeRef`).
    FromNode(NodeRef),
}

impl<'de> Deserialize<'de> for RasterSize {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        if value.is_string() {
            serde_json::from_value(value).map(RasterSize::FromNode)
        } else {
            serde_json::from_value(value).map(RasterSize::Fixed)
        }
        .map_err(D::Error::custom)
    }
}

impl FieldType for RasterSize {
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

impl Rasterize {
    /// The engine op for a canvas of `width` x `height`, with this op's fill
    /// and background resolved for `row`. The one place both size forms
    /// become a `GeometryOp`.
    pub(crate) fn with_size(
        &self,
        width: u32,
        height: u32,
        row: usize,
        ctx: &ParamCtx,
    ) -> PolarsResult<GeometryOp> {
        Ok(GeometryOp::Rasterize {
            width,
            height,
            fill_value: self.fill_value.resolve(row, ctx)?,
            background: self.background.resolve(row, ctx)?,
        })
    }
}

impl OpDef for Rasterize {
    fn shape(&self) -> Option<OpShape> {
        let (height, width) = match &self.size {
            RasterSize::Fixed([height, width]) => (height.size(), width.size()),
            // Another node's canvas: known only once that node has run.
            RasterSize::FromNode(_) => (Sym::PerRow, Sym::PerRow),
        };
        Some(OpShape::Fixed(vec![height, width, Sym::Known(1)]))
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Rasterize {
            size,
            fill_value: _,
            background: _,
        } = self;
        let (width, height) = match size {
            RasterSize::Fixed([height, width]) => {
                (width.resolve(row, ctx)?, height.resolve(row, ctx)?)
            }
            // At plan time any canvas stands in: the size is symbolic
            // (`shape`), and the rules this resolution is read for do not
            // depend on it.
            RasterSize::FromNode(_) if ctx.is_planning() => (1, 1),
            RasterSize::FromNode(_) => polars_bail!(ComputeError:
                "rasterize(shape=<node>) takes its size from another node's \
                 buffer, which only the graph executor has; this spec reached \
                 a path that resolves it without one"),
        };
        Ok(GraphStep::Geometry(
            self.with_size(width, height, row, ctx)?,
        ))
    }
}
