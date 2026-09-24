//! Contour extraction (buffer → contour), measures and transforms (contour →
//! scalar, vector or contour).

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::geometry::ops::{ApproxMethod, ExtractMode, ScaleOrigin};
use view_buffer::GeometryOp;

use super::{OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;

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
    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let ContourSimplify { tolerance } = self;
        geometry(GeometryOp::Simplify {
            tolerance: tolerance.resolve(row, ctx)?,
        })
    }
}
