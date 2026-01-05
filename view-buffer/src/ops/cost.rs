//! Operation cost tracking for zero-copy verification and pipeline analysis.

use crate::core::dtype::DType;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Categorizes the memory/performance cost of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum OpCost {
    /// True zero-copy: only metadata changes (offset, strides).
    /// No data is read or written; the underlying buffer is shared.
    ZeroCopy,

    /// Reads/writes data element-wise, allocates a new buffer.
    /// This includes scalar operations, type casts, and any compute.
    Allocating,

    /// External I/O operation (file read, image decode, network, etc.).
    /// These operations have unpredictable latency and always allocate.
    IO,
}

impl OpCost {
    /// Returns a short symbol for display in cost reports.
    pub fn symbol(&self) -> &'static str {
        match self {
            OpCost::ZeroCopy => "0",
            OpCost::Allocating => "A",
            OpCost::IO => "IO",
        }
    }

    /// Returns true if this operation allocates new memory.
    pub fn allocates(&self) -> bool {
        !matches!(self, OpCost::ZeroCopy)
    }
}

/// Detailed cost report for a single operation in a pipeline.
#[derive(Debug, Clone)]
pub struct OpCostReport {
    /// Name of the operation (e.g., "Flip", "Scale", "Resize").
    pub op_name: &'static str,

    /// Extended description for fused operations (e.g., "Fused(Scale(2.0), Relu)").
    pub op_description: Option<String>,

    /// The intrinsic cost of this operation.
    pub intrinsic_cost: OpCost,

    /// If the operation changes the dtype, records (from, to).
    /// None if dtype is preserved.
    pub dtype_change: Option<(DType, DType)>,

    /// True if this operation preserves the input dtype.
    pub preserves_dtype: bool,

    /// Details of fused operations (if this is a Fused op).
    pub fused_ops: Option<Vec<String>>,

    /// Input dtype for this operation.
    pub input_dtype: DType,

    /// Output dtype for this operation.
    pub output_dtype: DType,

    /// Estimated bytes allocated by this operation (if known).
    pub estimated_bytes: Option<usize>,
}

impl OpCostReport {
    /// Creates a new cost report for an operation that preserves dtype.
    pub fn new(op_name: &'static str, cost: OpCost, dtype: DType) -> Self {
        Self {
            op_name,
            op_description: None,
            intrinsic_cost: cost,
            dtype_change: None,
            preserves_dtype: true,
            fused_ops: None,
            input_dtype: dtype,
            output_dtype: dtype,
            estimated_bytes: None,
        }
    }

    /// Creates a new cost report for an operation that changes dtype.
    pub fn with_dtype_change(op_name: &'static str, cost: OpCost, from: DType, to: DType) -> Self {
        Self {
            op_name,
            op_description: None,
            intrinsic_cost: cost,
            dtype_change: Some((from, to)),
            preserves_dtype: false,
            fused_ops: None,
            input_dtype: from,
            output_dtype: to,
            estimated_bytes: None,
        }
    }

    /// Creates a cost report for a fused operation with details of constituent ops.
    pub fn fused(
        fused_op_names: Vec<String>,
        cost: OpCost,
        input_dtype: DType,
        output_dtype: DType,
    ) -> Self {
        let description = format!("Fused({})", fused_op_names.join(", "));
        let dtype_change = if input_dtype != output_dtype {
            Some((input_dtype, output_dtype))
        } else {
            None
        };
        Self {
            op_name: "Fused",
            op_description: Some(description),
            intrinsic_cost: cost,
            dtype_change,
            preserves_dtype: input_dtype == output_dtype,
            fused_ops: Some(fused_op_names),
            input_dtype,
            output_dtype,
            estimated_bytes: None,
        }
    }

    /// Sets the estimated bytes for this operation.
    pub fn with_estimated_bytes(mut self, bytes: usize) -> Self {
        self.estimated_bytes = Some(bytes);
        self
    }

    /// Returns the display name (uses description if available, otherwise op_name).
    pub fn display_name(&self) -> &str {
        self.op_description.as_deref().unwrap_or(self.op_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_symbols() {
        assert_eq!(OpCost::ZeroCopy.symbol(), "0");
        assert_eq!(OpCost::Allocating.symbol(), "A");
        assert_eq!(OpCost::IO.symbol(), "IO");
    }

    #[test]
    fn test_allocates() {
        assert!(!OpCost::ZeroCopy.allocates());
        assert!(OpCost::Allocating.allocates());
        assert!(OpCost::IO.allocates());
    }
}
