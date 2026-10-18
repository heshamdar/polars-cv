//! Binary operations between two arrays.
//!
//! This module provides element-wise operations between two ViewBuffers,
//! including arithmetic operations and bitwise operations for mask manipulation.
//!
//! # Operation Semantics
//!
//! Operations have type-dependent semantics to match common library expectations:
//!
//! ## For integer types (u8, u16):
//! - `Add`/`Subtract`: Saturating arithmetic (clamps to valid range)
//! - `Multiply`: Saturating multiplication (clamps to max value)
//! - `Blend`: Normalized multiplication ((a/max) * (b/max) * max)
//! - `Divide`: True division — integer operands promote to float and `a / b`
//!   is computed in float with IEEE semantics (`x / 0` is ±inf, `0 / 0` NaN),
//!   so the result dtype is `f32` (or `f64` when an operand is already `f64`).
//!
//! ## For float types (f32, f64):
//! - All operations use standard IEEE 754 arithmetic

use crate::core::buffer::ViewBuffer;
use crate::core::convert::CastFrom;
use crate::core::dtype::{
    with_dtype, DType, DTypeCategory, OutputDTypeRule, PlannedDType, ViewType,
};
use crate::core::strided::{Cursor, Walk};
use crate::ops::shape_rule::{show_dims, Dim, OpShape};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use std::mem::MaybeUninit;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Binary operations between two arrays.
///
/// All operations are element-wise over two operands of one shape, which
/// is the output's. Nothing broadcasts: an operand of another shape is
/// refused (`validate`), and a rank is changed explicitly (`squeeze`,
/// `expand_dims`).
///
/// Operations have type-dependent semantics:
/// - For `u8`/`u16`: Image-processing semantics (saturating, normalized)
/// - For `f32`/`f64`: Standard numerical semantics
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BinaryOp {
    /// Element-wise addition.
    ///
    /// For u8/u16: Saturating addition (clamps to max value).
    /// For f32/f64: Standard addition.
    Add,
    /// Element-wise subtraction.
    ///
    /// For u8/u16: Saturating subtraction (clamps to 0).
    /// For f32/f64: Standard subtraction.
    Subtract,
    /// Element-wise multiplication.
    ///
    /// For u8/u16: Saturating multiplication (clamps to max value).
    /// For f32/f64: Standard multiplication.
    Multiply,
    /// Normalized blend (element-wise).
    ///
    /// For u8: (a/255) * (b/255) * 255
    /// For u16: (a/65535) * (b/65535) * 65535
    /// For f32/f64: Standard multiplication (same as Multiply).
    Blend,
    /// Element-wise division (true division).
    ///
    /// The operands' promotion ([`DType::promote`]) divides in its
    /// [`DType::accumulator`] with IEEE semantics (`x / 0` is ±inf, `0 / 0`
    /// NaN), as NumPy's `true_divide`.
    Divide,
    /// Element-wise maximum.
    Maximum,
    /// Element-wise minimum.
    Minimum,
    /// Bitwise AND (useful for combining masks).
    BitwiseAnd,
    /// Bitwise OR (useful for combining masks).
    BitwiseOr,
    /// Bitwise XOR.
    BitwiseXor,
}

// The Python-facing name of every two-buffer operation.
//
// This table used to be `BINARY_OPS` in the polars-cv crate, which made
// `BinaryOp` the one enum-shaped vocabulary the registry could not hold — and
// so the one that needed a hand-written FFI arm and a parity-test exemption.
// Nothing required it to live there: the enum is
// this crate's, and the names describe engine semantics, not plugin ones.
//
// Declaring it here puts it under the same exhaustiveness guard as every other
// vocabulary: a new `BinaryOp` variant now fails to compile until it is named.
crate::naming::named_variants!(BinaryOp {
    "add" => Add,
    "subtract" => Subtract,
    "multiply" => Multiply,
    "divide" => Divide,
    "blend" => Blend,
    "maximum" => Maximum,
    "minimum" => Minimum,
    "bitwise_and" => BitwiseAnd,
    "bitwise_or" => BitwiseOr,
    "bitwise_xor" => BitwiseXor,
});

impl BinaryOp {
    /// Execute the binary operation on two buffers.
    ///
    /// Both buffers have one shape (`validate`). The operands combine in
    /// their common dtype ([`DType::promote`], NumPy's promotion), with the
    /// per-dtype semantics of [`BinaryElem`]; true division computes in that
    /// dtype's [`DType::accumulator`]. Either way the result is
    /// [`output_dtype`](Self::output_dtype), the dtype planning reads.
    pub fn execute(&self, a: &ViewBuffer, b: &ViewBuffer) -> ViewBuffer {
        assert_eq!(
            a.shape(),
            b.shape(),
            "a binary op's operands have one shape (`validate`)"
        );
        let output_shape = a.shape();
        let out_dtype = self.output_dtype(a.dtype(), b.dtype());
        let common = a.dtype().promote(b.dtype());
        // Each operand is read in its own dtype and converted exactly to the
        // dtype the op computes in (`zip_with`): the promotion, or for true
        // division its float (`out_dtype`, the promotion's accumulator).
        match self {
            BinaryOp::Divide => match out_dtype {
                DType::F64 => divide::<f64>(a, b, output_shape),
                _ => divide::<f32>(a, b, output_shape),
            },
            _ => with_dtype!(common, T => self.execute_typed::<T>(a, b, output_shape)),
        }
    }

    /// The output dtype of this binary op for the given operand dtypes: its
    /// [`output_dtype_rule`](Op::output_dtype_rule) applied to the two
    /// operands' common dtype ([`DType::promote`], NumPy's promotion).
    ///
    /// This is what planning (the plugin's `plan::step`, given both operand
    /// dtypes) and execution ([`execute`](BinaryOp::execute)) both read, and
    /// it derives from the rule rather than restating it, so the two cannot
    /// disagree. Divide uses *true division*, into the promotion's
    /// [`DType::accumulator`].
    pub fn output_dtype(&self, left: DType, right: DType) -> DType {
        self.output_dtype_rule().resolve(left.promote(right))
    }

    /// Every op but division, in the operands' own dtype `T`. The op is
    /// matched once, outside the loop, so each arm is its own monomorphic
    /// loop.
    fn execute_typed<T: BinaryElem + FromAny>(
        &self,
        a: &ViewBuffer,
        b: &ViewBuffer,
        output_shape: &[usize],
    ) -> ViewBuffer {
        match self {
            BinaryOp::Add => zip_with(a, b, output_shape, T::sat_add),
            BinaryOp::Subtract => zip_with(a, b, output_shape, T::sat_sub),
            BinaryOp::Multiply => zip_with(a, b, output_shape, T::sat_mul),
            BinaryOp::Blend => zip_with(a, b, output_shape, T::blend),
            BinaryOp::Maximum => zip_with(a, b, output_shape, crate::ops::util::maximum::<T>),
            BinaryOp::Minimum => zip_with(a, b, output_shape, crate::ops::util::minimum::<T>),
            BinaryOp::BitwiseAnd => zip_with(a, b, output_shape, T::bit_and),
            BinaryOp::BitwiseOr => zip_with(a, b, output_shape, T::bit_or),
            BinaryOp::BitwiseXor => zip_with(a, b, output_shape, T::bit_xor),
            BinaryOp::Divide => {
                unreachable!("division is computed in float by `execute`")
            }
        }
    }
}

/// `a / b` element-wise in the float `F`, IEEE: `x / 0` is ±inf, `0 / 0`
/// NaN.
fn divide<F: ViewType + num_traits::Float + FromAny>(
    a: &ViewBuffer,
    b: &ViewBuffer,
    output_shape: &[usize],
) -> ViewBuffer {
    zip_with(a, b, output_shape, |x: F, y: F| x / y)
}

/// A dtype every element dtype converts to ([`CastFrom`]): the dtype a
/// binary op computes in reads either operand, whatever its own dtype.
pub(crate) trait FromAny:
    ViewType
    + Default
    + CastFrom<u8>
    + CastFrom<i8>
    + CastFrom<u16>
    + CastFrom<i16>
    + CastFrom<u32>
    + CastFrom<i32>
    + CastFrom<u64>
    + CastFrom<i64>
    + CastFrom<f32>
    + CastFrom<f64>
{
}

impl<T> FromAny for T where
    T: ViewType
        + Default
        + CastFrom<u8>
        + CastFrom<i8>
        + CastFrom<u16>
        + CastFrom<i16>
        + CastFrom<u32>
        + CastFrom<i32>
        + CastFrom<u64>
        + CastFrom<i64>
        + CastFrom<f32>
        + CastFrom<f64>
{
}

/// Elements per block of [`zip_with`]'s operand reads.
const BLOCK: usize = 1024;

/// `f(a, b)` element-wise in the dtype `T`, over two operands of
/// `output_shape`.
///
/// Two contiguous operands of dtype `T` are zipped slice against slice.
/// Otherwise each operand is read a block at a time through its [`Walk`]'s
/// cursor: a block inside one packed unit (a contiguous operand, a row of a
/// crop or a vertical flip) where it lies, any other (a transpose, a
/// horizontal flip) packed into a stack scratch by the walk's unit copies,
/// then converted to `T` (which holds every value of both:
/// [`DType::promote`]) unless already in it. No operand is copied whole.
pub(crate) fn zip_with<T: FromAny, Fun: Fn(T, T) -> T>(
    a: &ViewBuffer,
    b: &ViewBuffer,
    output_shape: &[usize],
    f: Fun,
) -> ViewBuffer {
    let direct = |x: &ViewBuffer| x.dtype() == T::DTYPE && x.layout.is_contiguous();
    let out: Vec<T> = if direct(a) && direct(b) {
        let (sa, sb) = (a.as_slice::<T>(), b.as_slice::<T>());
        sa.iter().zip(sb).map(|(&x, &y)| f(x, y)).collect()
    } else {
        let total: usize = output_shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let (wa, wb) = (Walk::of(a), Walk::of(b));
        let (mut ca, mut cb) = (wa.cursor(), wb.cursor());
        let (mut ba, mut bb) = (Block::<T>::new(), Block::<T>::new());
        let mut start = 0;
        while start < total {
            let n = BLOCK.min(total - start);
            let xa = ba.read(&mut ca, a.dtype(), n);
            let xb = bb.read(&mut cb, b.dtype(), n);
            out.extend(xa.iter().zip(xb).map(|(&x, &y)| f(x, y)));
            start += n;
        }
        out
    };
    ViewBuffer::from_vec_with_shape(out, output_shape.to_vec())
}

/// One operand's block in `T`, and the stack scratch it is packed and
/// converted in.
struct Block<T> {
    scratch: [MaybeUninit<u64>; BLOCK],
    converted: [T; BLOCK],
}

impl<T: FromAny> Block<T> {
    fn new() -> Self {
        Block {
            scratch: [MaybeUninit::uninit(); BLOCK],
            converted: [T::default(); BLOCK],
        }
    }

    /// The operand's next `n` elements (of `dtype`) in `T`: as the cursor
    /// gives them when already in `T`, else converted.
    fn read<'s>(&'s mut self, cursor: &'s mut Cursor<'_>, dtype: DType, n: usize) -> &'s [T] {
        if dtype == T::DTYPE {
            cursor.next::<T>(n, &mut self.scratch)
        } else {
            with_dtype!(dtype, S => {
                let src = cursor.next::<S>(n, &mut self.scratch);
                for (d, &x) in self.converted[..n].iter_mut().zip(src) {
                    *d = T::cast_from(x);
                }
            });
            &self.converted[..n]
        }
    }
}

/// The per-dtype semantics of the two-buffer ops, the one definition for
/// every dtype:
///
/// - integers: `add`/`subtract`/`multiply` saturate to the dtype's range;
///   `blend` is the normalized product `(a/MAX)(b/MAX)MAX = round(a·b/MAX)`
///   (MAX is odd for every integer dtype, so there are no ties), saturated;
///   the bitwise ops are exact;
/// - floats: IEEE arithmetic, and `blend` is a plain product (MAX is 1). The
///   bitwise ops are integer-only by contract (`accepted_input_dtypes`,
///   `validate`), so a float reaching one is a caller that skipped it.
pub(crate) trait BinaryElem: ViewType + PartialOrd {
    fn sat_add(self, other: Self) -> Self;
    fn sat_sub(self, other: Self) -> Self;
    fn sat_mul(self, other: Self) -> Self;
    fn blend(self, other: Self) -> Self;
    fn bit_and(self, other: Self) -> Self;
    fn bit_or(self, other: Self) -> Self;
    fn bit_xor(self, other: Self) -> Self;
}

macro_rules! binary_int {
    ($($t:ty => $wide:ty),+) => {$(
        impl BinaryElem for $t {
            #[inline(always)]
            fn sat_add(self, other: Self) -> Self {
                self.saturating_add(other)
            }
            #[inline(always)]
            fn sat_sub(self, other: Self) -> Self {
                self.saturating_sub(other)
            }
            #[inline(always)]
            fn sat_mul(self, other: Self) -> Self {
                self.saturating_mul(other)
            }
            #[inline(always)]
            fn blend(self, other: Self) -> Self {
                // `round(a·b / MAX)`, half away from zero, in a width that
                // holds the product (i128 for signed, u128 for unsigned).
                let max = <$t>::MAX as $wide;
                let p = self as $wide * other as $wide;
                #[allow(unused_comparisons)]
                let q = if p >= 0 { (p + max / 2) / max } else { (p - max / 2) / max };
                q.clamp(<$t>::MIN as $wide, max) as $t
            }
            #[inline(always)]
            fn bit_and(self, other: Self) -> Self {
                self & other
            }
            #[inline(always)]
            fn bit_or(self, other: Self) -> Self {
                self | other
            }
            #[inline(always)]
            fn bit_xor(self, other: Self) -> Self {
                self ^ other
            }
        }
    )+};
}
binary_int!(u8 => u128, u16 => u128, u32 => u128, u64 => u128,
            i8 => i128, i16 => i128, i32 => i128, i64 => i128);

macro_rules! binary_float {
    ($($t:ty),+) => {$(
        impl BinaryElem for $t {
            #[inline(always)]
            fn sat_add(self, other: Self) -> Self {
                self + other
            }
            #[inline(always)]
            fn sat_sub(self, other: Self) -> Self {
                self - other
            }
            #[inline(always)]
            fn sat_mul(self, other: Self) -> Self {
                self * other
            }
            #[inline(always)]
            fn blend(self, other: Self) -> Self {
                self * other
            }
            #[inline(always)]
            fn bit_and(self, _: Self) -> Self {
                unreachable!("bitwise ops are integer-only")
            }
            #[inline(always)]
            fn bit_or(self, _: Self) -> Self {
                unreachable!("bitwise ops are integer-only")
            }
            #[inline(always)]
            fn bit_xor(self, _: Self) -> Self {
                unreachable!("bitwise ops are integer-only")
            }
        }
    )+};
}
binary_float!(f32, f64);

impl Op for BinaryOp {
    fn name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "Add",
            BinaryOp::Subtract => "Subtract",
            BinaryOp::Multiply => "Multiply",
            BinaryOp::Blend => "Blend",
            BinaryOp::Divide => "Divide",
            BinaryOp::Maximum => "Maximum",
            BinaryOp::Minimum => "Minimum",
            BinaryOp::BitwiseAnd => "BitwiseAnd",
            BinaryOp::BitwiseOr => "BitwiseOr",
            BinaryOp::BitwiseXor => "BitwiseXor",
        }
    }

    fn shape(&self) -> OpShape {
        OpShape::Elementwise
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Reads each operand where it lies, a block at a time through its
        // walk's cursor (`zip_with`): no layout is packed whole.
        MemoryEffect::StridePreserving
    }

    fn identity_rule(&self) -> IdentityRule {
        // Computes / combines / reduces — never a removable no-op.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        false // A binary op combines two buffers, not an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // Element-wise combination of two aligned buffers: output at (y, x)
        // depends only on both inputs at (y, x). Matched exhaustively (not a
        // blanket) so a new variant must reconfirm this rather than silently
        // inherit Pointwise — the classification a spatial-window reorder trusts.
        match self {
            BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::Blend
            | BinaryOp::Divide
            | BinaryOp::Maximum
            | BinaryOp::Minimum
            | BinaryOp::BitwiseAnd
            | BinaryOp::BitwiseOr
            | BinaryOp::BitwiseXor => SpatialDependency::Pointwise,
        }
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        // Binary ops produce new contiguous output
        None
    }

    fn validate(
        &self,
        input_shapes: &[&[Dim]],
        input_dtypes: &[PlannedDType],
    ) -> Result<(), ValidationError> {
        let [a, b, ..] = input_shapes else {
            return Err(ValidationError::InsufficientInputs {
                expected: 2,
                got: input_shapes.len(),
            });
        };
        // The bitwise ops need a common integer dtype: u64 with a signed
        // integer promotes to f64, which has no bits to combine (NumPy
        // refuses it too).
        if let (
            BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor,
            [PlannedDType::Known(l), PlannedDType::Known(r), ..],
        ) = (self, input_dtypes)
        {
            if !DTypeCategory::Integer.accepts(l.promote(*r)) {
                return Err(ValidationError::Generic {
                    message: format!(
                        "{l:?} and {r:?} have no common integer dtype (they promote to {:?}), \
                         so they have no bits to combine; cast one first",
                        l.promote(*r)
                    ),
                });
            }
        }
        // Refused where the ranks differ or two known sizes do; an unknown
        // size is checked per row.
        if elementwise_dims(a, b).is_none() {
            return Err(ValidationError::Generic {
                message: format!(
                    "operands must have the same shape, got {} and {} (nothing \
                     broadcasts: change a rank with squeeze() or expand_dims())",
                    show_dims(a),
                    show_dims(b)
                ),
            });
        }
        Ok(())
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        match self {
            BinaryOp::BitwiseAnd | BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => {
                DTypeCategory::Integer
            }
            _ => DTypeCategory::Numeric,
        }
    }

    fn working_dtype(&self) -> Option<DType> {
        None // Work with promoted input dtype
    }

    /// Over the operands' common dtype ([`DType::promote`]).
    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            BinaryOp::Divide => OutputDTypeRule::PromoteToFloat,
            _ => OutputDTypeRule::PreserveInput,
        }
    }
}

/// The one shape two operands share, over what is known of them; `None`
/// when their ranks differ or two known sizes on an axis do.
///
/// Each axis: a known size on either side gives it (an unknown one must be
/// that size for the row to run); two unknown sizes stay unknown — each
/// operand numbers its own `Input(k)`, so equal symbols on the two sides are
/// not the same size.
pub fn elementwise_dims(a: &[Dim], b: &[Dim]) -> Option<Vec<Dim>> {
    if a.len() != b.len() {
        return None;
    }
    a.iter()
        .zip(b)
        .map(|pair| match pair {
            (Dim::Known(x), Dim::Known(y)) => (x == y).then_some(Dim::Known(*x)),
            (Dim::Known(n), _) | (_, Dim::Known(n)) => Some(Dim::Known(*n)),
            _ => Some(Dim::Unknown),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An operand read through any layout gives the value at the same
    /// position as its packed copy: across `zip_with`'s blocks (every case
    /// spans several), in a dtype other than the one computed in.
    #[test]
    fn a_strided_operand_reads_as_its_packed_copy() {
        let a = ViewBuffer::from_vec_with_shape(
            (0..37 * 53 * 3).map(|i| i as u16).collect(),
            vec![37, 53, 3],
        );
        let b = ViewBuffer::from_vec_with_shape(
            (0..53 * 37 * 3).map(|i| (i % 251) as u8).collect(),
            vec![53, 37, 3],
        )
        .permute(&[1, 0, 2])
        .flip(&[1]);
        let packed = b.to_contiguous();
        let out = BinaryOp::Add.execute(&a, &b);
        let want = BinaryOp::Add.execute(&a, &packed);
        assert_eq!(out.shape(), &[37, 53, 3]);
        assert_eq!(out.as_slice::<u16>(), want.as_slice::<u16>());
    }

    #[test]
    fn elementwise_dims_decides_each_axis_on_what_it_knows() {
        use crate::ops::shape_rule::Dim::{Input, Known, Unknown};
        // A known size on either side gives the axis.
        assert_eq!(
            elementwise_dims(&[Known(8), Input(1)], &[Input(0), Known(3)]),
            Some(vec![Known(8), Known(3)])
        );
        // Each operand numbers its own symbols: equal ones are not one size.
        assert_eq!(
            elementwise_dims(&[Input(0)], &[Input(0)]),
            Some(vec![Unknown])
        );
        // A size of 1 stretches nothing.
        assert_eq!(
            elementwise_dims(&[Known(4), Known(1)], &[Known(4), Known(3)]),
            None
        );
        // Different ranks have no shared shape, whatever they know: the
        // `[H, W, 1]` against `[H, W]` that broadcast to `[H, W, W]`.
        assert_eq!(
            elementwise_dims(&[Known(4), Known(4), Known(1)], &[Known(4), Known(4)]),
            None
        );
        assert_eq!(elementwise_dims(&[Input(0), Input(1)], &[Input(0)]), None);
    }

    #[test]
    fn validate_refuses_operands_of_different_shapes() {
        use crate::ops::shape_rule::Dim::{Input, Known};
        let f32s = [PlannedDType::Known(DType::F32); 2];
        let ok = |a: &[Dim], b: &[Dim]| BinaryOp::Add.validate(&[a, b], &f32s);
        assert!(ok(&[Known(4), Known(5)], &[Known(4), Known(5)]).is_ok());
        assert!(ok(&[Known(4), Input(1)], &[Input(0), Known(5)]).is_ok());
        let err = ok(&[Known(4), Known(5), Known(1)], &[Known(4), Known(5)]).unwrap_err();
        assert!(err.to_string().contains("same shape"), "{err}");
        assert!(ok(
            &[Known(4), Known(5), Known(1)],
            &[Known(4), Known(5), Known(3)]
        )
        .is_err());
    }

    #[test]
    fn test_u8_saturating_add() {
        let a = ViewBuffer::from_vec_with_shape(vec![200u8, 100, 50], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![100u8, 50, 10], vec![3]);
        let result = BinaryOp::Add.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 255); // 200 + 100 = 255 (saturated)
        assert_eq!(data[1], 150); // 100 + 50 = 150
        assert_eq!(data[2], 60); // 50 + 10 = 60
    }

    #[test]
    fn test_u8_saturating_subtract() {
        let a = ViewBuffer::from_vec_with_shape(vec![50u8, 100, 200], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![100u8, 50, 50], vec![3]);
        let result = BinaryOp::Subtract.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 0); // 50 - 100 = 0 (saturated)
        assert_eq!(data[1], 50); // 100 - 50 = 50
        assert_eq!(data[2], 150); // 200 - 50 = 150
    }

    #[test]
    fn test_u8_saturating_multiply() {
        let a = ViewBuffer::from_vec_with_shape(vec![10u8, 16, 20], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![10u8, 16, 20], vec![3]);
        let result = BinaryOp::Multiply.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 100); // 10 * 10 = 100
        assert_eq!(data[1], 255); // 16 * 16 = 256 -> 255 (saturated)
        assert_eq!(data[2], 255); // 20 * 20 = 400 -> 255 (saturated)
    }

    #[test]
    fn test_u8_blend() {
        let a = ViewBuffer::from_vec_with_shape(vec![255u8, 128, 0], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![255u8, 128, 255], vec![3]);
        let result = BinaryOp::Blend.execute(&a, &b);
        let data = result.as_slice::<u8>();
        assert_eq!(data[0], 255); // (255/255) * (255/255) * 255 = 255
        assert_eq!(data[1], 64); // (128/255) * (128/255) * 255 ≈ 64
        assert_eq!(data[2], 0); // (0/255) * (255/255) * 255 = 0
    }

    #[test]
    fn test_u8_divide_is_true_division() {
        // divide(u8, u8) promotes to f32 and computes true division, not the
        // truncating integer division it used to.
        let a = ViewBuffer::from_vec_with_shape(vec![130u8, 128, 1], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![64u8, 64, 0], vec![3]);
        let result = BinaryOp::Divide.execute(&a, &b);
        assert_eq!(result.dtype(), DType::F32);
        let data = result.as_slice::<f32>();
        assert!((data[0] - (130.0 / 64.0)).abs() < 1e-6); // ~2.031, not 2
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert_eq!(data[2], f32::INFINITY); // IEEE: 1 / 0
    }

    #[test]
    fn test_output_dtype_authority() {
        // Standard promotion for non-dividing ops.
        assert_eq!(BinaryOp::Add.output_dtype(DType::U8, DType::U8), DType::U8);
        assert_eq!(
            BinaryOp::Add.output_dtype(DType::U8, DType::U16),
            DType::U16
        );
        assert_eq!(
            BinaryOp::Add.output_dtype(DType::U8, DType::F32),
            DType::F32
        );
        // True division always lands on a float.
        assert_eq!(
            BinaryOp::Divide.output_dtype(DType::U8, DType::U8),
            DType::F32
        );
        assert_eq!(
            BinaryOp::Divide.output_dtype(DType::F64, DType::F64),
            DType::F64
        );
        assert_eq!(
            BinaryOp::Divide.output_dtype(DType::U8, DType::F64),
            DType::F64
        );
    }

    #[test]
    fn test_mixed_dtype_add_matches_authority() {
        // A mixed add computes in, and produces, the promoted dtype (u16),
        // matching planning.
        let a = ViewBuffer::from_vec_with_shape(vec![200u8, 100], vec![2]);
        let b = ViewBuffer::from_vec_with_shape(vec![400u16, 50], vec![2]);
        let result = BinaryOp::Add.execute(&a, &b);
        assert_eq!(result.dtype(), DType::U16);
        let data = result.as_slice::<u16>();
        assert_eq!(data[0], 600);
        assert_eq!(data[1], 150);
    }

    /// The bitwise ops have no float semantics: a float operand is refused by
    /// the contract, and the engine fails rather than inventing bits for one
    /// that reaches it.
    #[test]
    #[should_panic(expected = "integer-only")]
    fn bitwise_on_a_float_fails() {
        let a = ViewBuffer::from_vec_with_shape(vec![3.0f64], vec![1]);
        BinaryOp::BitwiseAnd.execute(&a, &a);
    }

    #[test]
    fn test_f32_standard_arithmetic() {
        let a = ViewBuffer::from_vec_with_shape(vec![1.0f32, 2.0, 3.0], vec![3]);
        let b = ViewBuffer::from_vec_with_shape(vec![0.5f32, 0.5, 0.5], vec![3]);

        let add_result = BinaryOp::Add.execute(&a, &b);
        let add_data = add_result.as_slice::<f32>();
        assert!((add_data[0] - 1.5).abs() < 1e-6);

        let mul_result = BinaryOp::Multiply.execute(&a, &b);
        let mul_data = mul_result.as_slice::<f32>();
        assert!((mul_data[0] - 0.5).abs() < 1e-6);
        assert!((mul_data[1] - 1.0).abs() < 1e-6);
    }
}
