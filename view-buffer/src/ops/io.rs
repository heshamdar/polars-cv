//! I/O operations for pipeline sources and sinks.
//!
//! Defines the formats that can be used to start and end a pipeline,
//! enabling composition of pipelines and integration with external formats.

use crate::core::dtype::DType;
use crate::ops::cost::OpCost;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Format for pipeline input sources.
///
/// These define how raw data enters the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SourceFormat {
    /// Zero-cost from existing ViewBuffer (for meta-pipelines).
    ViewBuffer,

    /// Deserialize from VIEW protocol blob.
    /// This is zero-cost after the initial header parse.
    Blob,

    /// Decode from image bytes (PNG, JPEG, etc.).
    /// Auto-detects format from magic bytes.
    #[cfg(feature = "image_interop")]
    ImageBytes,

    /// Zero-copy from Arrow array.
    #[cfg(feature = "arrow_interop")]
    Arrow,
}

impl SourceFormat {
    /// Returns the cost of this source operation.
    pub fn cost(&self) -> OpCost {
        match self {
            SourceFormat::ViewBuffer => OpCost::ZeroCopy,
            SourceFormat::Blob => OpCost::IO, // Parsing required
            #[cfg(feature = "image_interop")]
            SourceFormat::ImageBytes => OpCost::IO,
            #[cfg(feature = "arrow_interop")]
            SourceFormat::Arrow => OpCost::ZeroCopy,
        }
    }

    /// Returns the name of this source format for display.
    pub fn name(&self) -> &'static str {
        match self {
            SourceFormat::ViewBuffer => "ViewBuffer",
            SourceFormat::Blob => "Blob",
            #[cfg(feature = "image_interop")]
            SourceFormat::ImageBytes => "ImageBytes",
            #[cfg(feature = "arrow_interop")]
            SourceFormat::Arrow => "Arrow",
        }
    }
}

/// Format for pipeline output sinks.
///
/// These define how data exits the pipeline and what format it takes.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SinkFormat {
    /// Keep as ViewBuffer (for composing pipelines).
    /// This is zero-cost if no materialization is needed.
    ViewBuffer,

    /// Serialize to VIEW protocol blob.
    /// Contains header with dtype/shape and contiguous data.
    Blob,

    /// Export as raw bytes with numpy-compatible header.
    /// Contains dtype, shape, and contiguous C-order data.
    #[cfg(feature = "numpy_interop")]
    Numpy,

    /// Export as raw bytes compatible with torch.frombuffer.
    /// Contains dtype, shape, and contiguous data.
    #[cfg(feature = "torch_interop")]
    Torch,

    /// Encode as PNG image bytes.
    #[cfg(feature = "image_interop")]
    Png,

    /// Encode as JPEG image bytes with specified quality.
    #[cfg(feature = "image_interop")]
    Jpeg {
        /// JPEG quality (1-100).
        quality: u8,
    },
    #[cfg(feature = "image_interop")]
    WebP {
        /// WebP quality (1-100).
        quality: u8,
    },

    /// Encode as TIFF image bytes.
    /// Supports floating-point data types for medical imaging.
    #[cfg(feature = "image_interop")]
    Tiff,

    /// Export as Arrow array.
    #[cfg(feature = "arrow_interop")]
    Arrow {
        /// If true, create nested FixedSizeListArray.
        /// If false, create flat array.
        nested: bool,
    },
}

impl SinkFormat {
    /// Returns the cost of this sink operation.
    pub fn cost(&self) -> OpCost {
        match self {
            SinkFormat::ViewBuffer => OpCost::ZeroCopy,
            SinkFormat::Blob => OpCost::Allocating, // Serialization
            #[cfg(feature = "numpy_interop")]
            SinkFormat::Numpy => OpCost::Allocating,
            #[cfg(feature = "torch_interop")]
            SinkFormat::Torch => OpCost::Allocating,
            #[cfg(feature = "image_interop")]
            SinkFormat::Png => OpCost::IO,
            #[cfg(feature = "image_interop")]
            SinkFormat::Jpeg { .. } => OpCost::IO,
            #[cfg(feature = "image_interop")]
            SinkFormat::WebP { .. } => OpCost::IO,
            #[cfg(feature = "image_interop")]
            SinkFormat::Tiff => OpCost::IO,
            #[cfg(feature = "arrow_interop")]
            SinkFormat::Arrow { .. } => OpCost::Allocating,
        }
    }

    /// Returns the name of this sink format for display.
    pub fn name(&self) -> &'static str {
        match self {
            SinkFormat::ViewBuffer => "ViewBuffer",
            SinkFormat::Blob => "Blob",
            #[cfg(feature = "numpy_interop")]
            SinkFormat::Numpy => "Numpy",
            #[cfg(feature = "torch_interop")]
            SinkFormat::Torch => "Torch",
            #[cfg(feature = "image_interop")]
            SinkFormat::Png => "Png",
            #[cfg(feature = "image_interop")]
            SinkFormat::Jpeg { .. } => "Jpeg",
            #[cfg(feature = "image_interop")]
            SinkFormat::WebP { .. } => "WebP",
            #[cfg(feature = "image_interop")]
            SinkFormat::Tiff => "Tiff",
            #[cfg(feature = "arrow_interop")]
            SinkFormat::Arrow { .. } => "Arrow",
        }
    }
}

/// Metadata for a placeholder source.
///
/// Used when defining pipelines without concrete data,
/// allowing shape/dtype to be specified or inferred later.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlaceholderMeta {
    /// Expected shape, if known.
    pub expected_shape: Option<Vec<usize>>,

    /// Expected dtype, if known.
    pub expected_dtype: Option<DType>,
}

impl PlaceholderMeta {
    /// Creates a new placeholder with no constraints.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a placeholder with an expected shape.
    pub fn with_shape(shape: Vec<usize>) -> Self {
        Self {
            expected_shape: Some(shape),
            expected_dtype: None,
        }
    }

    /// Creates a placeholder with an expected dtype.
    pub fn with_dtype(dtype: DType) -> Self {
        Self {
            expected_shape: None,
            expected_dtype: Some(dtype),
        }
    }

    /// Creates a placeholder with both shape and dtype constraints.
    pub fn with_shape_and_dtype(shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            expected_shape: Some(shape),
            expected_dtype: Some(dtype),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_format_costs() {
        assert_eq!(SourceFormat::ViewBuffer.cost(), OpCost::ZeroCopy);
        assert_eq!(SourceFormat::Blob.cost(), OpCost::IO);
    }

    #[test]
    fn test_sink_format_costs() {
        assert_eq!(SinkFormat::ViewBuffer.cost(), OpCost::ZeroCopy);
        assert_eq!(SinkFormat::Blob.cost(), OpCost::Allocating);
    }

    #[test]
    fn test_placeholder_meta() {
        let empty = PlaceholderMeta::new();
        assert!(empty.expected_shape.is_none());
        assert!(empty.expected_dtype.is_none());

        let with_shape = PlaceholderMeta::with_shape(vec![100, 100, 3]);
        assert_eq!(with_shape.expected_shape, Some(vec![100, 100, 3]));
    }
}
