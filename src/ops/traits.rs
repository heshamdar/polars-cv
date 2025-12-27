//! Core operation traits and types.

use crate::core::dtype::DType;
use crate::ops::cost::OpCost;
use crate::ops::validation::ValidationError;

/// Legacy memory effect enum - kept for backwards compatibility.
/// Prefer using `Op::intrinsic_cost()` which returns `OpCost`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEffect {
    View,
    StridePreserving,
    RequiresContiguous,
}

impl From<MemoryEffect> for OpCost {
    fn from(effect: MemoryEffect) -> Self {
        match effect {
            MemoryEffect::View => OpCost::ZeroCopy,
            MemoryEffect::StridePreserving => OpCost::Allocating,
            MemoryEffect::RequiresContiguous => OpCost::Allocating,
        }
    }
}

/// Trait for all operations in the pipeline.
///
/// Operations must provide shape/dtype inference, cost information,
/// and optional validation for plan-time error checking.
pub trait Op {
    /// Returns the name of this operation for display/debugging.
    fn name(&self) -> &'static str;

    /// Infers the output shape given input shapes.
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize>;

    /// Infers the output dtype given input dtypes.
    fn infer_dtype(&self, inputs: &[DType]) -> DType;

    /// Returns the legacy memory effect. Prefer `intrinsic_cost()`.
    fn memory_effect(&self) -> MemoryEffect;

    /// Returns the intrinsic cost of this operation.
    fn intrinsic_cost(&self) -> OpCost {
        self.memory_effect().into()
    }

    /// Infers output strides given input shape and strides.
    ///
    /// Returns None if strides cannot be inferred or if the operation
    /// requires materialization that makes input strides irrelevant.
    fn infer_strides(&self, input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>>;

    /// Validates the operation at plan time.
    ///
    /// Returns Ok(()) if the operation is valid for the given inputs,
    /// or Err with a description of why validation failed.
    fn validate(
        &self,
        _input_shapes: &[&[usize]],
        _input_dtypes: &[DType],
    ) -> Result<(), ValidationError> {
        // Default: no validation requirements
        Ok(())
    }
}
