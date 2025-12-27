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

    /// The intrinsic cost of this operation.
    pub intrinsic_cost: OpCost,

    /// If the operation changes the dtype, records (from, to).
    /// None if dtype is preserved.
    pub dtype_change: Option<(DType, DType)>,

    /// True if this operation preserves the input dtype.
    pub preserves_dtype: bool,
}

impl OpCostReport {
    /// Creates a new cost report for an operation that preserves dtype.
    pub fn new(op_name: &'static str, cost: OpCost) -> Self {
        Self {
            op_name,
            intrinsic_cost: cost,
            dtype_change: None,
            preserves_dtype: true,
        }
    }

    /// Creates a new cost report for an operation that changes dtype.
    pub fn with_dtype_change(op_name: &'static str, cost: OpCost, from: DType, to: DType) -> Self {
        Self {
            op_name,
            intrinsic_cost: cost,
            dtype_change: Some((from, to)),
            preserves_dtype: false,
        }
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
