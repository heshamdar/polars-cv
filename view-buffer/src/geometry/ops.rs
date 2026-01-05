//! Geometry operation enum for pipeline integration.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::traits::{MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use crate::ops::Domain;

use super::contour::Winding;

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

/// Geometry operations for the pipeline system.
///
/// These operations work on contour data or produce contours/masks.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum GeometryOp {
    // --- Measures (contour -> scalar) ---
    /// Compute area using the Shoelace formula.
    /// If `signed` is true, returns signed area (negative for CW).
    Area { signed: bool },

    /// Compute perimeter (arc length).
    Perimeter,

    /// Compute centroid (center of mass).
    Centroid,

    /// Compute axis-aligned bounding box.
    BoundingBox,

    /// Determine winding direction.
    Winding,

    // --- Transforms (contour -> contour) ---
    /// Translate by offset.
    Translate { dx: f64, dy: f64 },

    /// Scale relative to an origin point.
    Scale {
        sx: f64,
        sy: f64,
        origin: ScaleOrigin,
    },

    /// Flip (reverse) point order, changing winding direction.
    Flip,

    /// Ensure a specific winding direction.
    EnsureWinding { direction: Winding },

    /// Simplify using Douglas-Peucker algorithm.
    Simplify { tolerance: f64 },

    /// Compute convex hull.
    ConvexHull,

    /// Normalize coordinates to [0, 1] range.
    Normalize { ref_width: f64, ref_height: f64 },

    /// Convert normalized coords to absolute pixel coords.
    ToAbsolute { ref_width: f64, ref_height: f64 },

    // --- Predicates (contour -> bool or scalar) ---
    /// Check if contour is convex.
    IsConvex,

    /// Check if a point is inside the contour.
    ContainsPoint { x: f64, y: f64 },

    // --- Pairwise (contour, contour -> scalar) ---
    /// Compute Intersection over Union with another contour.
    IoU,

    /// Compute Dice coefficient with another contour.
    Dice,

    /// Compute Hausdorff distance to another contour.
    HausdorffDistance,

    // --- Rasterization (contour -> image) ---
    /// Rasterize contour to binary mask.
    Rasterize {
        width: u32,
        height: u32,
        fill_value: u8,
        background: u8,
        anti_alias: bool,
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

impl Op for GeometryOp {
    fn name(&self) -> &'static str {
        match self {
            GeometryOp::Area { .. } => "Area",
            GeometryOp::Perimeter => "Perimeter",
            GeometryOp::Centroid => "Centroid",
            GeometryOp::BoundingBox => "BoundingBox",
            GeometryOp::Winding => "Winding",
            GeometryOp::Translate { .. } => "Translate",
            GeometryOp::Scale { .. } => "Scale",
            GeometryOp::Flip => "Flip",
            GeometryOp::EnsureWinding { .. } => "EnsureWinding",
            GeometryOp::Simplify { .. } => "Simplify",
            GeometryOp::ConvexHull => "ConvexHull",
            GeometryOp::Normalize { .. } => "Normalize",
            GeometryOp::ToAbsolute { .. } => "ToAbsolute",
            GeometryOp::IsConvex => "IsConvex",
            GeometryOp::ContainsPoint { .. } => "ContainsPoint",
            GeometryOp::IoU => "IoU",
            GeometryOp::Dice => "Dice",
            GeometryOp::HausdorffDistance => "HausdorffDistance",
            GeometryOp::Rasterize { .. } => "Rasterize",
            GeometryOp::ExtractContours { .. } => "ExtractContours",
        }
    }

    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        match self {
            // Scalar outputs
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::IsConvex
            | GeometryOp::ContainsPoint { .. }
            | GeometryOp::IoU
            | GeometryOp::Dice
            | GeometryOp::HausdorffDistance => {
                vec![1]
            }

            // Winding returns a single value (encoded as int)
            GeometryOp::Winding => vec![1],

            // Centroid returns (x, y)
            GeometryOp::Centroid => vec![2],

            // BoundingBox returns (x, y, width, height)
            GeometryOp::BoundingBox => vec![4],

            // Contour transforms preserve the number of points (mostly)
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Flip
            | GeometryOp::EnsureWinding { .. }
            | GeometryOp::Normalize { .. }
            | GeometryOp::ToAbsolute { .. } => {
                if !inputs.is_empty() {
                    inputs[0].to_vec()
                } else {
                    vec![]
                }
            }

            // These may reduce points
            GeometryOp::Simplify { .. } | GeometryOp::ConvexHull => {
                // Can't know output size statically, return placeholder
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

    fn infer_dtype(&self, _inputs: &[DType]) -> DType {
        match self {
            // Scalar measures are float
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::IoU
            | GeometryOp::Dice
            | GeometryOp::HausdorffDistance => DType::F64,

            // Centroid, BoundingBox are float
            GeometryOp::Centroid | GeometryOp::BoundingBox => DType::F64,

            // Boolean results
            GeometryOp::IsConvex | GeometryOp::ContainsPoint { .. } => DType::U8,

            // Winding direction (0 = CCW, 1 = CW)
            GeometryOp::Winding => DType::U8,

            // Contour transforms preserve F64 coordinates
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Flip
            | GeometryOp::EnsureWinding { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull
            | GeometryOp::Normalize { .. }
            | GeometryOp::ToAbsolute { .. } => DType::F64,

            // Rasterize produces U8 mask
            GeometryOp::Rasterize { .. } => DType::U8,

            // ExtractContours produces F64 coordinates
            GeometryOp::ExtractContours { .. } => DType::F64,
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        match self {
            // In-place transformations could be view-like but we allocate for safety
            GeometryOp::Flip | GeometryOp::EnsureWinding { .. } => MemoryEffect::StridePreserving,

            // Everything else allocates
            _ => MemoryEffect::RequiresContiguous,
        }
    }

    fn intrinsic_cost(&self) -> OpCost {
        match self {
            // Simple scalar measures are cheap
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::Centroid
            | GeometryOp::BoundingBox
            | GeometryOp::Winding
            | GeometryOp::IsConvex
            | GeometryOp::ContainsPoint { .. } => OpCost::Allocating,

            // Transforms allocate new contour
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Flip
            | GeometryOp::EnsureWinding { .. }
            | GeometryOp::Normalize { .. }
            | GeometryOp::ToAbsolute { .. } => OpCost::Allocating,

            // These are O(n log n) or more
            GeometryOp::Simplify { .. } | GeometryOp::ConvexHull => OpCost::Allocating,

            // Pairwise ops require rasterization or polygon clipping
            GeometryOp::IoU | GeometryOp::Dice | GeometryOp::HausdorffDistance => {
                OpCost::Allocating
            }

            // Rasterization is expensive
            GeometryOp::Rasterize { .. } => OpCost::Allocating,

            // Contour extraction is expensive
            GeometryOp::ExtractContours { .. } => OpCost::Allocating,
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
        input_shapes: &[&[usize]],
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

            GeometryOp::Normalize {
                ref_width,
                ref_height,
            }
            | GeometryOp::ToAbsolute {
                ref_width,
                ref_height,
            } => {
                if *ref_width <= 0.0 || *ref_height <= 0.0 {
                    return Err(ValidationError::InvalidParameter {
                        param: "ref_width/ref_height".to_string(),
                        reason: "Reference dimensions must be > 0".to_string(),
                    });
                }
                Ok(())
            }

            // Pairwise ops need two inputs
            GeometryOp::IoU | GeometryOp::Dice | GeometryOp::HausdorffDistance => {
                if input_shapes.len() < 2 {
                    return Err(ValidationError::InsufficientInputs {
                        expected: 2,
                        got: input_shapes.len(),
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
            GeometryOp::IsConvex | GeometryOp::ContainsPoint { .. } | GeometryOp::Winding => {
                OutputDTypeRule::Fixed(DType::U8)
            }
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

            // Measures: Contour → Scalar/Vector
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::Centroid
            | GeometryOp::BoundingBox
            | GeometryOp::Winding
            | GeometryOp::IsConvex
            | GeometryOp::ContainsPoint { .. } => Domain::Contour,

            // Contour transforms: Contour → Contour
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Flip
            | GeometryOp::EnsureWinding { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull
            | GeometryOp::Normalize { .. }
            | GeometryOp::ToAbsolute { .. } => Domain::Contour,

            // Pairwise operations: Contour (+ Contour) → Scalar
            GeometryOp::IoU | GeometryOp::Dice | GeometryOp::HausdorffDistance => Domain::Contour,
        }
    }

    /// Get the output domain this geometry operation produces.
    pub fn output_domain(&self) -> Domain {
        match self {
            // Extraction: Buffer → Contour
            GeometryOp::ExtractContours { .. } => Domain::Contour,

            // Rasterization: Contour → Buffer
            GeometryOp::Rasterize { .. } => Domain::Buffer,

            // Scalar measures
            GeometryOp::Area { .. }
            | GeometryOp::Perimeter
            | GeometryOp::IsConvex
            | GeometryOp::ContainsPoint { .. }
            | GeometryOp::Winding
            | GeometryOp::IoU
            | GeometryOp::Dice
            | GeometryOp::HausdorffDistance => Domain::Scalar,

            // Vector measures (multi-value)
            GeometryOp::Centroid | GeometryOp::BoundingBox => Domain::Vector,

            // Contour transforms preserve contour domain
            GeometryOp::Translate { .. }
            | GeometryOp::Scale { .. }
            | GeometryOp::Flip
            | GeometryOp::EnsureWinding { .. }
            | GeometryOp::Simplify { .. }
            | GeometryOp::ConvexHull
            | GeometryOp::Normalize { .. }
            | GeometryOp::ToAbsolute { .. } => Domain::Contour,
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
                anti_alias: false
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
            anti_alias: false,
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
            anti_alias: false,
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
            anti_alias: false,
        };
        assert_eq!(rasterize.input_domain(), Domain::Contour);
        assert_eq!(rasterize.output_domain(), Domain::Buffer);

        // Area: Contour → Scalar
        let area = GeometryOp::Area { signed: false };
        assert_eq!(area.input_domain(), Domain::Contour);
        assert_eq!(area.output_domain(), Domain::Scalar);

        // Translate: Contour → Contour
        let translate = GeometryOp::Translate { dx: 10.0, dy: 20.0 };
        assert_eq!(translate.input_domain(), Domain::Contour);
        assert_eq!(translate.output_domain(), Domain::Contour);

        // Centroid: Contour → Vector
        assert_eq!(GeometryOp::Centroid.input_domain(), Domain::Contour);
        assert_eq!(GeometryOp::Centroid.output_domain(), Domain::Vector);
    }
}
