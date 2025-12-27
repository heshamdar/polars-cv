#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Elementary scalar operations that can be fused into a single kernel.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ScalarOp {
    Add(f32),
    Mul(f32),
    Relu,
    // We can add Clamp, Sigmoid, etc. later
}

/// A sequence of scalar operations to be executed element-wise in a single pass.
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FusedKernel {
    pub ops: Vec<ScalarOp>,
}

impl FusedKernel {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn push(&mut self, op: ScalarOp) {
        self.ops.push(op);
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
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
