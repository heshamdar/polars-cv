//! A graph node's value for one row, in whichever domain it produced.

use std::sync::Arc;

use view_buffer::geometry::contour::Outline;
use view_buffer::ops::Domain;
use view_buffer::ViewBuffer;

/// A graph node's output for one row, in any of the four domains.
///
/// Uses `Arc` for cheap cloning and efficient graph caching
/// without copying data.
#[derive(Debug, Clone)]
pub(crate) enum NodeOutput {
    /// Array/image data (reference-counted for cheap cloning).
    Buffer(Arc<ViewBuffer>),
    /// Extracted contours (geometry domain).
    Contours(Arc<Vec<Outline>>),
    /// Single scalar value.
    Scalar(f64),
    /// Multiple scalar values (bbox, centroid, etc.).
    Vector(Arc<Vec<f64>>),
}

impl NodeOutput {
    /// Get the domain of this output.
    pub(crate) fn domain(&self) -> Domain {
        match self {
            NodeOutput::Buffer(_) => Domain::Buffer,
            NodeOutput::Contours(_) => Domain::Contour,
            NodeOutput::Scalar(_) => Domain::Scalar,
            NodeOutput::Vector(_) => Domain::Vector,
        }
    }

    /// Try to extract a buffer from this output.
    pub(crate) fn as_buffer(&self) -> Option<&Arc<ViewBuffer>> {
        match self {
            NodeOutput::Buffer(buf) => Some(buf),
            _ => None,
        }
    }

    /// Try to extract contours from this output.
    pub(crate) fn as_contours(&self) -> Option<&Arc<Vec<Outline>>> {
        match self {
            NodeOutput::Contours(contours) => Some(contours),
            _ => None,
        }
    }

    /// Create a Buffer output from a ViewBuffer (takes ownership).
    pub(crate) fn from_buffer(buf: ViewBuffer) -> Self {
        NodeOutput::Buffer(Arc::new(buf))
    }

    /// Create a Contours output from a Vec of contours.
    pub(crate) fn from_contours(contours: Vec<Outline>) -> Self {
        NodeOutput::Contours(Arc::new(contours))
    }

    /// Create a Scalar output.
    pub(crate) fn from_scalar(val: f64) -> Self {
        NodeOutput::Scalar(val)
    }

    /// Create a Vector output.
    pub(crate) fn from_vector(vals: Vec<f64>) -> Self {
        NodeOutput::Vector(Arc::new(vals))
    }
}
