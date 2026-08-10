//! Geometry operation enum for pipeline integration.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use crate::ops::Domain;

/// Origin point for scale operations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ScaleOrigin {
    /// Scale around the contour's centroid.
    Centroid,
    /// Scale around the bounding box center.
    BBoxCenter,
    /// Scale around the coordinate origin (0, 0).
    Origin,
}

/// Geometry operations reachable from a `Pipeline` graph.
///
/// This enum is the *graph* vocabulary, not a catalogue of the geometry the crate
/// can do. Every variant here has a `resolve_op` arm in the polars-cv plugin and a
/// `GraphStep::Geometry` encoding; contour operations that only make sense on an
/// already-materialized contour column — winding, flip, normalize, contains_point,
/// IoU, Dice, Hausdorff and friends — are standalone `.contour` namespace plugin
/// functions that call [`super::measures`], [`super::predicates`],
/// [`super::pairwise`] and [`super::transforms`] directly. Adding a variant here
/// without a `resolve_op` arm makes it unconstructible; adding one without an
/// encoding is a compile error at the plugin's exhaustive match.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GeometryOp {
    // --- Measures (contour -> scalar) ---
    /// Compute the area of the region the contour describes.
    /// If `signed` is true, returns signed area (negative for CW).
    Area { signed: bool },

    /// Compute perimeter (arc length).
    Perimeter,

    /// Compute centroid (center of mass).
    Centroid,

    /// Compute axis-aligned bounding box.
    BoundingBox,

    // --- Transforms (contour -> contour) ---
    /// Translate by offset.
    Translate { dx: f64, dy: f64 },

    /// Scale relative to an origin point.
    Scale {
        sx: f64,
        sy: f64,
        origin: ScaleOrigin,
    },

    /// Simplify using Douglas-Peucker algorithm.
    Simplify { tolerance: f64 },

    /// Compute convex hull.
    ConvexHull,

    // --- Rasterization (contour -> image) ---
    /// Rasterize contour to binary mask.
    Rasterize {
        width: u32,
        height: u32,
        fill_value: u8,
        background: u8,
    },

    // --- Extraction (image -> contour) ---
    /// Extract contours from binary image.
    ExtractContours {
        mode: ExtractMode,
        method: ApproxMethod,
        min_area: Option<f64>,
    },
}

/// Mode for contour extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ApproxMethod {
    /// Keep all boundary points.
    None,
    /// Remove redundant points on straight lines.
    Simple,
    /// Douglas-Peucker approximation.
    Approx,
}

crate::naming::named_variants!(ScaleOrigin {
    "centroid" => Centroid,
    "bbox_center" => BBoxCenter,
    "origin" => Origin,
});

crate::naming::named_variants!(ExtractMode {
    "external" => External,
    "tree" => Tree,
    "all" => All,
});

crate::naming::named_variants!(ApproxMethod {
    "none" => None,
    "simple" => Simple,
    "approx" => Approx,
});

impl Op for GeometryOp {
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

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        match self {
            // Scalar outputs
            GeometryOp::Area { .. } | GeometryOp::Perimeter => vec![1],

            // Centroid returns (x, y)
            GeometryOp::Centroid => vec![2],

            // BoundingBox returns (x, y, width, height)
            GeometryOp::BoundingBox => vec![4],

            // Contour transforms preserve the point list; `Simplify` and
            // `ConvexHull` may shorten it, which is not knowable statically, so
            // the input shape stands in for both.
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull => {
                if !inputs.is_empty() {
                    inputs[0].to_vec()
                } else {
                    vec![]
                }
            }

            // Rasterize produces an image
            GeometryOp::Rasterize { width, height, .. } => {
                vec![*height as usize, *width as usize, 1]
            }

            // ExtractContours output shape is dynamic
            GeometryOp::ExtractContours { .. } => {
                // Variable-length output, placeholder
                vec![]
            }
        }
    }

    fn output_rank_rule(&self) -> OutputRankRule {
        match self {
            // Scalar/vector measures emit a fixed-length 1-D result.
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::Centroid
            | GeometryOp::BoundingBox => OutputRankRule::Fixed(1),
            // Contour→contour transforms preserve the point-list rank.
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull => OutputRankRule::PreserveRank,
            // Rasterize emits an [H, W, 1] image.
            GeometryOp::Rasterize { .. } => OutputRankRule::Fixed(3),
            // Extraction produces a variable-length contour set.
            GeometryOp::ExtractContours { .. } => OutputRankRule::Unknown,
        }
    }

    fn output_channel_rule(&self) -> OutputChannelRule {
        match self {
            // Rasterize produces a single-channel mask.
            GeometryOp::Rasterize { .. } => OutputChannelRule::Fixed(1),
            // Everything else is scalar/vector/contour data, not an image.
            _ => OutputChannelRule::NotApplicable,
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Every geometry op materializes a fresh contour, measure or mask.
        MemoryEffect::RequiresContiguous
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
            GeometryOp::Rasterize { width, height, .. } => {
                if *width == 0 || *height == 0 {
                    return Err(ValidationError::InvalidParameter {
                        param: "width/height".to_string(),
                        reason: "Dimensions must be > 0".to_string(),
                    });
                }
                Ok(())
            }

            GeometryOp::Simplify { tolerance } => {
                if *tolerance < 0.0 {
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

impl GeometryOp {
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

    #[test]
    fn test_op_names() {
        assert_eq!(GeometryOp::Area { signed: false }.name(), "Area");
        assert_eq!(GeometryOp::Perimeter.name(), "Perimeter");
        assert_eq!(
            GeometryOp::Rasterize {
                width: 100,
                height: 100,
                fill_value: 255,
                background: 0,
            }
            .name(),
            "Rasterize"
        );
    }

    #[test]
    fn test_rasterize_shape() {
        let op = GeometryOp::Rasterize {
            width: 200,
            height: 100,
            fill_value: 255,
            background: 0,
        };
        let shape = op.infer_shape(&[]);
        assert_eq!(shape, vec![100, 200, 1]);
    }

    #[test]
    fn test_validate_rasterize() {
        let op = GeometryOp::Rasterize {
            width: 0,
            height: 100,
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
            width: 100,
            height: 100,
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
