//! Core operation traits and types.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::execution::tiling::TilePolicy;
use crate::ops::cost::OpCost;
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

    /// Infers the output dtype given input dtypes.
    ///
    /// This is the legacy method. For new operations, prefer implementing
    /// `output_dtype_rule()` and using `resolve_output_dtype()` instead.
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
    /// Default: PreserveInput.
    fn output_dtype_rule(&self) -> OutputDTypeRule {
        OutputDTypeRule::PreserveInput
    }

    /// Resolves the actual output dtype given input dtype and optional override.
    ///
    /// This is a convenience method that uses `output_dtype_rule()`.
    fn resolve_output_dtype(&self, input_dtype: DType, out_dtype_override: Option<DType>) -> DType {
        self.output_dtype_rule()
            .resolve(input_dtype, out_dtype_override)
    }

    // --- Tiling Support ---

    /// Returns the tiling policy for this operation.
    ///
    /// The tiling policy determines whether and how the operation can be
    /// executed in tiles for improved cache efficiency.
    ///
    /// Default: [`TilePolicy::Global`] (cannot be tiled) for safety.
    /// Operations should override this to enable tiled execution.
    ///
    /// # Policies
    ///
    /// - [`TilePolicy::PointWise`]: No pixel dependencies (halo=0)
    /// - [`TilePolicy::LocalNeighborhood`]: Needs pixel radius (e.g., blur)
    /// - [`TilePolicy::Global`]: Cannot be tiled (e.g., resize, normalize)
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn tile_policy(&self) -> TilePolicy {
    ///     match self {
    ///         MyOp::Threshold => TilePolicy::PointWise,
    ///         MyOp::Blur { sigma } => TilePolicy::LocalNeighborhood {
    ///             halo: (*sigma * 3.0).ceil() as usize,
    ///         },
    ///         MyOp::Resize { .. } => TilePolicy::Global,
    ///     }
    /// }
    /// ```
    fn tile_policy(&self) -> TilePolicy {
        TilePolicy::Global
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
