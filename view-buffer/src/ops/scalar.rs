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
    /// Subtraction of a constant. Kept distinct from `Add(-c)` so lowered ops
    /// stay bit-identical to their unfused form, mirroring `Div` vs `Mul`.
    Sub(f32),
    Mul(f32),
    /// Division by a constant. Kept distinct from `Mul(1/c)` so lowered ops
    /// (e.g. gamma's `x / max`) stay bit-identical to their unfused form.
    Div(f32),
    /// Power-law: `x.powf(c)`.
    Pow(f32),
    /// Negate: `-x`.
    Neg,
    /// Absolute value: `x.abs()`.
    Abs,
    /// Square root: `x.sqrt()`.
    Sqrt,
    /// Square: `x * x` (cheaper and exact vs `Pow(2.0)`).
    Square,
    /// Reciprocal: `1.0 / x`.
    Recip,
    /// Minimum with a constant ceiling: `x.min(c)`.
    Min(f32),
    /// Maximum with a constant floor: `x.max(c)` (`Relu == Max(0.0)`).
    Max(f32),
    /// Sign: `-1`, `0`, `+1`; `0` for `±0`, `NaN` for `NaN` (matches numpy.sign).
    Sign,
    /// Round toward negative infinity.
    Floor,
    /// Round toward positive infinity.
    Ceil,
    /// Round to nearest, ties to even (`f32::round_ties_even`; matches Polars/numpy).
    Round,
    /// Round toward zero (drop the fractional part).
    Trunc,
    Relu,
    Clamp(f32, f32),
}

/// numpy-compatible sign: `0` for `±0`, `NaN` for `NaN`, else `±1`.
///
/// `f32::signum`/`f64::signum` return `±1` for zero and would mislabel `0`, so
/// the sign ops below cannot use it. Written once here (generic over the float
/// type) so the f32 kernel arm and the f64 cold path share one definition.
#[inline]
pub(crate) fn signum_numpy<F: num_traits::Float>(x: F) -> F {
    if x.is_nan() {
        x
    } else if x > F::zero() {
        F::one()
    } else if x < F::zero() {
        -F::one()
    } else {
        F::zero()
    }
}

impl ScalarOp {
    /// Stable identifier for the op, used for `ComputeOp::Scalar` naming and
    /// debugging. Distinct from the Python-facing method name.
    pub fn name(&self) -> &'static str {
        match self {
            ScalarOp::Add(_) => "Add",
            ScalarOp::Sub(_) => "Sub",
            ScalarOp::Mul(_) => "Mul",
            ScalarOp::Div(_) => "Div",
            ScalarOp::Pow(_) => "Pow",
            ScalarOp::Neg => "Neg",
            ScalarOp::Abs => "Abs",
            ScalarOp::Sqrt => "Sqrt",
            ScalarOp::Square => "Square",
            ScalarOp::Recip => "Recip",
            ScalarOp::Min(_) => "Min",
            ScalarOp::Max(_) => "Max",
            ScalarOp::Sign => "Sign",
            ScalarOp::Floor => "Floor",
            ScalarOp::Ceil => "Ceil",
            ScalarOp::Round => "Round",
            ScalarOp::Trunc => "Trunc",
            ScalarOp::Relu => "Relu",
            ScalarOp::Clamp(..) => "Clamp",
        }
    }

    /// Evaluate the op on a single `f64` value.
    ///
    /// This is the **f64 cold path** authority: an unfused scalar op on an
    /// `f64` buffer computes here so `PromoteToFloat`'s f64-preserving contract
    /// holds (the bulk f32 kernel in `apply_fused_op_passes` handles every
    /// other dtype). The two must agree; `scalar_f64_matches_f32_kernel` in the
    /// buffer tests pins that. Constants are `f32` on the wire and widen to
    /// `f64` here, matching the unfused compute paths.
    #[inline]
    pub fn apply_f64(&self, x: f64) -> f64 {
        match self {
            ScalarOp::Add(c) => x + *c as f64,
            ScalarOp::Sub(c) => x - *c as f64,
            ScalarOp::Mul(c) => x * *c as f64,
            ScalarOp::Div(c) => x / *c as f64,
            ScalarOp::Pow(c) => x.powf(*c as f64),
            ScalarOp::Neg => -x,
            ScalarOp::Abs => x.abs(),
            ScalarOp::Sqrt => x.sqrt(),
            ScalarOp::Square => x * x,
            ScalarOp::Recip => 1.0 / x,
            ScalarOp::Min(c) => x.min(*c as f64),
            ScalarOp::Max(c) => x.max(*c as f64),
            ScalarOp::Sign => signum_numpy(x),
            ScalarOp::Floor => x.floor(),
            ScalarOp::Ceil => x.ceil(),
            ScalarOp::Round => x.round_ties_even(),
            ScalarOp::Trunc => x.trunc(),
            ScalarOp::Relu => x.max(0.0),
            ScalarOp::Clamp(lo, hi) => x.clamp(*lo as f64, *hi as f64),
        }
    }
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
