//! Core operation traits and types.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::ops::shape_rule::{OutputChannelRule, OutputRankRule};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::validation::ValidationError;

/// What an operation needs from its input's memory layout.
///
/// **This is the materialisation authority**, not a legacy label: `build_plan`
/// matches on it to decide whether to insert a `MaterializeContiguous` step
/// before an op, and `calc_strides` reads it to decide whether an output can
/// keep its input's strides. It was previously documented as legacy in favour
/// of an `OpCost` that could not replace it — the `MemoryEffect -> OpCost`
/// conversion collapsed `StridePreserving` and `RequiresContiguous` into a
/// single `Allocating`, losing exactly the distinction the planner needs. That
/// cost-reporting surface had no consumer and has been removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEffect {
    /// Metadata-only: the output is a view over the input's buffer.
    View,
    /// Allocates, but can read a strided input directly.
    StridePreserving,
    /// Allocates, and needs its input contiguous first.
    RequiresContiguous,
}

/// Under what condition an operation is a no-op — value-, dtype-, shape- and
/// channel-preserving, so a plan-time pass may delete it without changing any
/// output byte.
///
/// This is the **algebraic-identity authority**, the counterpart to
/// [`SpatialDependency`](crate::ops::spatial_rule::SpatialDependency) for
/// "does this op do anything at all". Value-preservation is op-*semantic* and
/// cannot be read off the shape/dtype rules — a `resize` to the same size still
/// resamples, a `flip` of a square still flips — so each op declares the
/// condition under which it collapses to a copy, and the planner evaluates that
/// condition against the op's entering shape/dtype. The two contextual variants
/// are strictly opt-in: an op that changes pixel positions must stay
/// [`Never`](IdentityRule::Never) even when it happens to preserve shape.
///
/// [`Always`](IdentityRule::Always) is decided by *literal* parameter values, so
/// it names the parameters it inspected (`deciding_params`). That declaration is
/// what makes the verdict structurally safe across the FFI: an op is resolved
/// with every expression parameter neutralized to a placeholder, so an `Always`
/// op whose deciding parameter is actually per-row would otherwise be spoofed by
/// the placeholder happening to equal the identity value. The plugin's
/// identity-elimination pass treats the op as [`Never`](IdentityRule::Never)
/// whenever any named deciding parameter
/// was expression-bound, so soundness rests on the declaration rather than on
/// which placeholder value the resolver used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentityRule {
    /// The op computes; it is never removable. The conservative answer for any
    /// op that transforms its input.
    Never,
    /// The op is a no-op purely by its (literal) parameters, for any input —
    /// e.g. `pad(0, 0, 0, 0)`. `deciding_params` names the parameters whose
    /// literal values were read to reach this verdict (e.g. `pad`'s four
    /// amounts); if any of them is per-row the op cannot be proven a no-op at
    /// plan time and is treated as [`Never`](IdentityRule::Never). List every
    /// parameter the identity condition inspects, and no others — a fill `value`
    /// behind zero amounts is not a deciding param.
    Always {
        deciding_params: &'static [&'static str],
    },
    /// The op is an identity exactly when its output shape equals its input
    /// shape. Sound only for ops that move no pixels when shape is preserved —
    /// a pure view (`reshape`, a crop anchored at the origin) or a pad that
    /// added nothing. An op whose candidacy also rests on a literal parameter
    /// value (a crop's origin being `(0, 0)`) names those parameters in
    /// `deciding_params`, gated exactly as for [`Always`](IdentityRule::Always).
    WhenShapePreserved {
        deciding_params: &'static [&'static str],
    },
    /// The op is an identity exactly when its output dtype equals its input
    /// dtype — the same-dtype `cast`, which copies rather than converts.
    WhenDtypePreserved,
}

impl IdentityRule {
    /// The parameters whose literal values this verdict was decided by. A
    /// planner must treat the op as [`Never`](IdentityRule::Never) when any of
    /// them is per-row: the verdict was reached against a placeholder.
    pub fn deciding_params(&self) -> &'static [&'static str] {
        match self {
            IdentityRule::Always { deciding_params }
            | IdentityRule::WhenShapePreserved { deciding_params } => deciding_params,
            IdentityRule::Never | IdentityRule::WhenDtypePreserved => &[],
        }
    }
}

/// Trait for all operations in the pipeline.
///
/// Operations must provide shape/dtype inference, their memory effect,
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

    /// Declares what this operation needs from its input's memory layout.
    ///
    /// Required (no default): an op that allocates but claims `View` would be
    /// planned as zero-copy.
    fn memory_effect(&self) -> MemoryEffect;

    /// Declares how this operation's output depends on the *spatial* extent of
    /// its input — the plan-time authority for whether a spatial window (a crop
    /// / ROI) may commute with the op.
    ///
    /// The structural, plan-time-inspectable counterpart to
    /// [`infer_shape`](Op::infer_shape) for spatial locality, in the same spirit
    /// as [`output_channel_rule`](Op::output_channel_rule) is for the channel
    /// dimension. See [`SpatialDependency`] for the four closed variants.
    ///
    /// Required (no default): an op that omits it would silently inherit a
    /// dependency it does not have. Unlike the rank/channel/dtype rules there is
    /// no `infer_shape`-style authority to parity-check this against, so the
    /// conservative, always-correct answer for any op whose dependence cannot be
    /// reasoned about is [`SpatialDependency::Global`] (it permits no reorder).
    fn spatial_dependency(&self) -> SpatialDependency;

    /// Declares under what condition this op is a no-op — the plan-time
    /// authority for whether an identity-elimination pass may delete it.
    ///
    /// See [`IdentityRule`] for the four closed variants. Required (no default):
    /// an op that omits it would silently inherit a claim it may not deserve.
    /// The conservative, always-correct answer for any computing op is
    /// [`IdentityRule::Never`]. The two contextual variants are value-semantic
    /// opt-ins — an op must return them only when preserving the named property
    /// genuinely means it copied its input unchanged.
    fn identity_rule(&self) -> IdentityRule;

    /// Whether this op is a *spatial window* — an axis-aligned crop/ROI over the
    /// H/W plane that the spatial-window pushdown may hoist earlier past ops it
    /// commutes with. The plan-time authority for "which op is a hoistable
    /// window", the counterpart to [`spatial_dependency`](Op::spatial_dependency)
    /// (which decides what a window may *cross*).
    ///
    /// Required (no default): "is a window" is an op-identity fact the pushdown
    /// must not re-decide by name in the planner. Only an op that selects an
    /// H/W sub-rectangle **without touching the channel axis** — so it commutes
    /// with channel-changing pointwise ops — may return `true`; every other op,
    /// including geometric resamplers and channel slicers, returns `false`. The
    /// conservative answer is `false`.
    fn is_spatial_window(&self) -> bool;

    /// Infers output strides given input shape and strides.
    ///
    /// Returns None if strides cannot be inferred or if the operation
    /// requires materialization that makes input strides irrelevant.
    fn infer_strides(&self, input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>>;

    /// Validates the operation at plan time.
    ///
    /// Returns Ok(()) if the operation is valid for the given inputs,
    /// or Err with a description of why validation failed.
    ///
    /// **Required, with no default** (CR-34): the executor calls this against
    /// the concrete input before running the op, so a shape the plan could not
    /// see becomes a row error instead of an index-out-of-bounds panic. An op
    /// that accepts anything says so with an explicit `Ok(())`; a new op cannot
    /// inherit "accepts anything" by omission.
    ///
    /// The planner also calls it with no dtypes and placeholder sizes for the
    /// dimensions it cannot know (see `ValidationError::depends_only_on_rank`),
    /// so an implementation must not assume `input_dtypes` is non-empty.
    fn validate(
        &self,
        input_shapes: &[&[usize]],
        input_dtypes: &[DType],
    ) -> Result<(), ValidationError>;

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

    /// Resolves the actual output dtype given input dtype.
    ///
    /// This is a convenience method that uses `output_dtype_rule()`.
    fn resolve_output_dtype(&self, input_dtype: DType) -> DType {
        self.output_dtype_rule().resolve(input_dtype)
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
        let expected = self.output_dtype_rule().resolve(input_dtype);
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
