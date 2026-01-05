#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Elementary scalar operations that can be fused into a single kernel.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ScalarOp {
    Add(f32),
    Mul(f32),
    Relu,
    Clamp(f32, f32),
}

impl ScalarOp {
    /// Returns a human-readable name for this operation with parameters.
    pub fn name(&self) -> String {
        match self {
            ScalarOp::Add(v) => format!("Add({v:.2})"),
            ScalarOp::Mul(v) => format!("Mul({v:.2})"),
            ScalarOp::Relu => "Relu".to_string(),
            ScalarOp::Clamp(min, max) => format!("Clamp({min:.2}, {max:.2})"),
        }
    }

    /// Returns the operation type name without parameters.
    pub fn op_type(&self) -> &'static str {
        match self {
            ScalarOp::Add(_) => "Add",
            ScalarOp::Mul(_) => "Mul",
            ScalarOp::Relu => "Relu",
            ScalarOp::Clamp(_, _) => "Clamp",
        }
    }
}

/// A sequence of scalar operations to be executed element-wise in a single pass.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FusedKernel {
    pub ops: Vec<ScalarOp>,
}

impl FusedKernel {
    /// Creates a new empty fused kernel.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Adds an operation to the kernel.
    pub fn push(&mut self, op: ScalarOp) {
        self.ops.push(op);
    }

    /// Returns the number of operations in the kernel.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns true if the kernel has no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Returns a human-readable description of the fused operations.
    /// Example: "Fused(Mul(2.00), Add(1.00), Relu)"
    pub fn describe(&self) -> String {
        let op_names: Vec<String> = self.ops.iter().map(|op| op.name()).collect();
        format!("Fused({})", op_names.join(", "))
    }

    /// Returns a list of operation names for detailed reporting.
    pub fn op_names(&self) -> Vec<String> {
        self.ops.iter().map(|op| op.name()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_construction() {
        let mut kernel = FusedKernel::new();
        kernel.push(ScalarOp::Mul(2.0));
        kernel.push(ScalarOp::Add(5.0));
        kernel.push(ScalarOp::Relu);

        assert_eq!(kernel.len(), 3);
        assert_eq!(kernel.ops[0], ScalarOp::Mul(2.0));
        assert_eq!(kernel.ops[1], ScalarOp::Add(5.0));
        assert_eq!(kernel.ops[2], ScalarOp::Relu);
    }
}
