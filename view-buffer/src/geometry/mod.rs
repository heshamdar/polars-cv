//! Geometry operations for contours, points, and polygons.
//!
//! This module provides geometric types and operations that integrate
//! with the view-buffer pipeline system. Contours can be rasterized
//! to masks, extracted from binary images, and various geometric
//! measures computed.
//!
//! # Key Types
//!
//! - [`Contour`] - A polygon with optional holes (exterior ring + interior rings)
//! - [`Point`] - A 2D point with f64 coordinates
//! - [`BoundingBox`] - Axis-aligned bounding box
//! - [`GeometryOp`] - Enum of geometry operations for pipeline integration
//!
//! # Operations
//!
//! ## Measures
//! - Area (signed and unsigned, using Shoelace formula)
//! - Perimeter (arc length)
//! - Centroid (center of mass)
//! - Bounding box
//! - Winding direction (CW/CCW)
//!
//! ## Transforms
//! - Translate, Scale, Flip (reverse winding)
//! - Simplify (Douglas-Peucker)
//! - Convex hull
//!
//! ## Predicates
//! - Point-in-polygon
//! - Is convex
//!
//! ## Pairwise
//! - IoU (Intersection over Union)
//! - Dice coefficient
//! - Hausdorff distance
//!
//! ## Rasterization
//! - Contour to mask (rasterize)
//! - Mask to contours (extract)

pub mod contour;
pub mod extract;
pub mod measures;
pub mod ops;
pub mod pairwise;
pub mod predicates;
pub mod rasterize;
pub mod transforms;

// Re-exports
pub use contour::{BoundingBox, Contour, Point, Winding};
pub use ops::GeometryOp;
