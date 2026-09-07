use crate::core::DType;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Elementary scalar operations that can be fused into a single kernel.
///
/// All ops compute in `f32` regardless of the kernel's input/output dtype —
/// the kernel converts on read and write (see [`FusedKernel::out_dtype`]).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ScalarOp {
    Add(f32),
    Mul(f32),
    /// Division by a constant. Kept distinct from `Mul(1/c)` so lowered ops
    /// (e.g. gamma's `x / max`) stay bit-identical to their unfused form.
    Div(f32),
    /// Power-law: `x.powf(c)`.
    Pow(f32),
    Relu,
    Clamp(f32, f32),
}

#[cfg(feature = "serde")]
fn default_out_dtype() -> DType {
    DType::F32
}

/// A sequence of scalar operations executed element-wise in a single pass.
///
/// The kernel reads any numeric input dtype (converting to `f32` during the
/// read, like a fused leading `Cast`), applies `ops` in `f32`, and writes the
/// result as [`out_dtype`](Self::out_dtype) (converting during the write,
/// like a fused trailing `Cast` — `round()`-then-saturate for integer
/// targets, matching `ViewBuffer::cast_to`). This removes the separate
/// cast materializations that used to bracket every fused chain.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FusedKernel {
    pub ops: Vec<ScalarOp>,
    /// Output dtype the f32 result is converted to in the same pass.
    #[cfg_attr(feature = "serde", serde(default = "default_out_dtype"))]
    pub out_dtype: DType,
}

impl Default for FusedKernel {
    fn default() -> Self {
        Self {
            ops: Vec::new(),
            out_dtype: DType::F32,
        }
    }
}

impl FusedKernel {
    /// Creates a new empty fused kernel with `f32` output.
    pub fn new() -> Self {
        Self::default()
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
        assert_eq!(kernel.out_dtype, DType::F32);
    }
}
