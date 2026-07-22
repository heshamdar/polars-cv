//! Core operation traits and types.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::cost::OpCost;
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::validation::ValidationError;
use crate::ops::{Domain, NodeOutput};

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
///
/// ## Dtype Contract
///
/// Operations declare their dtype requirements through three methods:
/// - `accepted_input_dtypes()`: What input types the operation can work with
/// - `working_dtype()`: The dtype used for internal computation (accumulator)
/// - `output_dtype_rule()`: How the output dtype is determined
///
/// This separates semantic operations from dtype mechanics, allowing the
/// execution layer to handle automatic casting.
pub trait Op {
    /// Returns the name of this operation for display/debugging.
    fn name(&self) -> &'static str;

    /// Infers the output shape given input shapes.
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize>;

    /// Declares how this operation transforms the input *rank* (number of
    /// dimensions).
    ///
    /// This is the plan-time-inspectable, structural counterpart to
    /// [`infer_shape`](Op::infer_shape): it states the rank effect abstractly
    /// (and can say [`Unknown`](OutputRankRule::Unknown)) without a concrete
    /// input shape. `infer_shape` stays the concrete authority; the two are
    /// bound by a parity test so they cannot diverge.
    ///
    /// Required (no default): every op must state its rank transform so a new
    /// op cannot silently inherit `PreserveRank` and lie about its structure.
    fn output_rank_rule(&self) -> OutputRankRule;

    /// Declares how this operation transforms the input *channel count* (the
    /// trailing dimension of an `[H, W, C]` buffer).
    ///
    /// The plan-time-inspectable, structural counterpart to
    /// [`infer_shape`](Op::infer_shape) for the channel dimension. Replaces the
    /// Python-side alpha/channel contract as the single authority.
    ///
    /// Required (no default): every op must state its channel transform.
    fn output_channel_rule(&self) -> OutputChannelRule;

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

    // --- Dtype Contract Methods ---

    /// Returns the categories of dtypes this operation accepts as input.
    ///
    /// The execution layer will automatically cast inputs to the working dtype
    /// if the input dtype is accepted but different from the working dtype.
    ///
    /// Default: Accept all types.
    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Any
    }

    /// Returns the dtype used for internal computation (accumulator).
    ///
    /// If Some(dtype), the execution layer will cast input to this dtype
    /// before performing the operation. This ensures numerical stability
    /// (e.g., using f32 for accumulation to avoid integer overflow).
    ///
    /// If None, the operation works directly with the input dtype.
    ///
    /// Default: None (preserve input dtype).
    fn working_dtype(&self) -> Option<DType> {
        None
    }

    /// Returns the rule for determining output dtype.
    ///
    /// This allows operations to declare whether they:
    /// - Preserve input dtype
    /// - Always output a fixed dtype
    /// - Have a configurable output dtype
    /// - Promote integers to floats
    ///
    /// Required (no default): every op must state its dtype rule so a new op
    /// cannot silently inherit `PreserveInput` and mis-report its output dtype.
    fn output_dtype_rule(&self) -> OutputDTypeRule;

    /// Resolves the actual output dtype given input dtype and optional override.
    ///
    /// This is a convenience method that uses `output_dtype_rule()`.
    fn resolve_output_dtype(&self, input_dtype: DType, out_dtype_override: Option<DType>) -> DType {
        self.output_dtype_rule()
            .resolve(input_dtype, out_dtype_override)
    }

    /// Validate that a produced buffer matches this operation's dtype contract.
    ///
    /// Returns `Ok(())` when the output dtype matches the declared
    /// [`output_dtype_rule`](Op::output_dtype_rule) for the given
    /// `input_dtype`, or an `Err` with a human-readable description of the
    /// mismatch.
    ///
    /// This is intended to be called after execution as a runtime guardrail.
    fn validate_output_dtype(&self, input_dtype: DType, output_dtype: DType) -> Result<(), String> {
        let expected = self.output_dtype_rule().resolve(input_dtype, None);
        if output_dtype != expected {
            return Err(format!(
                "{}: expected output dtype {:?} (rule {:?} with input {:?}), but got {:?}",
                self.name(),
                expected,
                self.output_dtype_rule(),
                input_dtype,
                output_dtype,
            ));
        }
        Ok(())
    }
}

// ============================================================
// Domain-Aware Operation Trait
// ============================================================

/// Trait for operations with domain type information.
///
/// This extends the basic Op trait with domain-aware execution,
/// enabling typed pipelines that cross between different data domains
/// (buffer, contour, scalar, vector).
///
/// # Example
///
/// ```ignore
/// // ExtractContours: Buffer → Contour
/// impl DomainOp for ExtractContoursOp {
///     fn input_domain(&self) -> Domain { Domain::Buffer }
///     fn output_domain(&self) -> Domain { Domain::Contour }
///
///     fn execute_typed(&self, input: NodeOutput) -> Result<NodeOutput, String> {
///         let buffer = input.as_buffer()
///             .ok_or("Expected Buffer input")?;
///         let contours = extract_contours(buffer, ...);
///         Ok(NodeOutput::from_contours(contours))
///     }
/// }
/// ```
pub trait DomainOp {
    /// What domain this operation expects as input.
    fn input_domain(&self) -> Domain;

    /// What domain this operation produces.
    fn output_domain(&self) -> Domain;

    /// Validate that the input domain is compatible.
    ///
    /// Returns an error with a helpful message if incompatible.
    fn validate_input_domain(&self, input: Domain) -> Result<(), String> {
        let expected = self.input_domain();
        if expected.accepts(input) {
            Ok(())
        } else {
            Err(format!(
                "{} expects {} input but received {}. Add a domain-converting operation.",
                std::any::type_name::<Self>()
                    .rsplit("::")
                    .next()
                    .unwrap_or("operation"),
                expected.name(),
                input.name()
            ))
        }
    }

    /// Execute with typed input/output.
    ///
    /// Implementations should first validate the input domain,
    /// then perform the operation and return the correctly-typed output.
    fn execute_typed(&self, input: NodeOutput) -> Result<NodeOutput, String>;
}
