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
use crate::ops::shape_rule::{show_dims, Dim, OpShape};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Binary operations between two arrays.
///
/// All operations are element-wise and support broadcasting.
/// The output shape is the broadcast result of both input shapes.
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
    /// Both buffers must have broadcastable shapes. The operands combine in
    /// their common dtype ([`DType::promote`], NumPy's promotion), with the
    /// per-dtype semantics of [`BinaryElem`]; true division computes in that
    /// dtype's [`DType::accumulator`]. Either way the result is
    /// [`output_dtype`](Self::output_dtype), the dtype planning reads.
    pub fn execute(&self, a: &ViewBuffer, b: &ViewBuffer) -> ViewBuffer {
        let output_shape =
            broadcast_shapes(a.shape(), b.shape()).expect("Shapes must be broadcastable");
        let out_dtype = self.output_dtype(a.dtype(), b.dtype());
        let common = a.dtype().promote(b.dtype());
        // Each operand is read in its own dtype and converted exactly to the
        // dtype the op computes in (`zip_with`): the promotion, or for true
        // division its float (`out_dtype`, the promotion's accumulator).
        match self {
            BinaryOp::Divide => match out_dtype {
                DType::F64 => divide::<f64>(a, b, &output_shape),
                _ => divide::<f32>(a, b, &output_shape),
            },
            _ => with_dtype!(common, T => self.execute_typed::<T>(a, b, &output_shape)),
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

/// Elements `start..start + dst.len()` of `buf` broadcast to `shape`, in
/// logical order, each converted to `T`.
fn read_as<T: FromAny>(buf: &ViewBuffer, shape: &[usize], start: usize, dst: &mut [T]) {
    with_dtype!(buf.dtype(), S => {
        let src = buf.as_slice::<S>();
        if buf.shape() == shape {
            let end = start + dst.len();
            for (d, &x) in dst.iter_mut().zip(&src[start..end]) {
                *d = T::cast_from(x);
            }
        } else {
            // Walk the output positions with an odometer, carrying the source
            // index along: each step adds the axis's broadcast step (0 where
            // the source axis is 1), with no division or allocation per element.
            let steps = broadcast_steps(buf.shape(), shape);
            let mut coords = linear_to_coords(start, shape);
            let mut at: usize = coords.iter().zip(&steps).map(|(c, s)| c * s).sum();
            for d in dst.iter_mut() {
                *d = T::cast_from(src[at]);
                for ax in (0..shape.len()).rev() {
                    coords[ax] += 1;
                    at += steps[ax];
                    if coords[ax] < shape[ax] {
                        break;
                    }
                    at -= steps[ax] * shape[ax];
                    coords[ax] = 0;
                }
            }
        }
    })
}

/// `f(a, b)` element-wise in the dtype `T`, broadcast to `output_shape`.
///
/// Operands already of dtype `T` and of the output's shape are read where
/// they lie. Otherwise each is read in its own dtype and converted to `T`
/// (which holds every value of both: [`DType::promote`]) a block at a time,
/// so mixed operands cost two blocks of scratch rather than a converted copy
/// of each.
fn zip_with<T: FromAny, Fun: Fn(T, T) -> T>(
    a: &ViewBuffer,
    b: &ViewBuffer,
    output_shape: &[usize],
    f: Fun,
) -> ViewBuffer {
    let (ca, cb) = (a.to_contiguous(), b.to_contiguous());
    let direct = |x: &ViewBuffer| x.dtype() == T::DTYPE && x.shape() == output_shape;
    let out: Vec<T> = if direct(&ca) && direct(&cb) {
        let (sa, sb) = (ca.as_slice::<T>(), cb.as_slice::<T>());
        sa.iter().zip(sb).map(|(&x, &y)| f(x, y)).collect()
    } else {
        const BLOCK: usize = 1024;
        let total: usize = output_shape.iter().product();
        let mut out = Vec::with_capacity(total);
        let (mut xa, mut xb) = ([T::default(); BLOCK], [T::default(); BLOCK]);
        let mut start = 0;
        while start < total {
            let n = BLOCK.min(total - start);
            read_as(&ca, output_shape, start, &mut xa[..n]);
            read_as(&cb, output_shape, start, &mut xb[..n]);
            out.extend(xa[..n].iter().zip(&xb[..n]).map(|(&x, &y)| f(x, y)));
            start += n;
        }
        out
    };
    ViewBuffer::from_vec_with_shape(out, output_shape.to_vec())
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
        OpShape::Broadcast
    }

    fn memory_effect(&self) -> MemoryEffect {
        // Binary ops require contiguous input for efficient SIMD
        MemoryEffect::RequiresContiguous
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
        // Refused only where two known sizes cannot broadcast.
        if broadcast_dims(a, b).is_none() {
            return Err(ValidationError::Generic {
                message: format!(
                    "shapes {} and {} cannot be broadcast together",
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

/// Two shapes broadcast together over what is known of them, aligned from the
/// last axis; `None` when two known sizes cannot broadcast (neither is 1 and
/// they differ).
///
/// Each axis on its own: two known sizes give theirs; a known 1 gives the
/// other side; a known size `n > 1` against an unknown one gives `n` (the
/// unknown one must be 1 or `n` for the row to run at all); two unknown sizes
/// stay unknown — each operand numbers its own `Input(k)`, so equal symbols
/// on the two sides are not the same size.
pub fn broadcast_dims(a: &[Dim], b: &[Dim]) -> Option<Vec<Dim>> {
    let rank = a.len().max(b.len());
    let at = |s: &[Dim], i: usize| {
        (i + s.len())
            .checked_sub(rank)
            .map_or(Dim::Known(1), |j| s[j])
    };
    (0..rank)
        .map(|i| match (at(a, i), at(b, i)) {
            (Dim::Known(1), d) | (d, Dim::Known(1)) => Some(d),
            (Dim::Known(x), Dim::Known(y)) => (x == y).then_some(Dim::Known(x)),
            (Dim::Known(n), _) | (_, Dim::Known(n)) => Some(Dim::Known(n)),
            _ => Some(Dim::Unknown),
        })
        .collect()
}

/// Compute the broadcast shape of two shapes.
///
/// Returns None if shapes are not broadcastable.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return None; // Not broadcastable
        }
    }

    result.reverse();
    Some(result)
}

use super::util::linear_to_coords;

/// Each axis of `out`'s step through a packed `src` broadcast to it: the
/// source's stride, or 0 where the source axis is 1 or absent (axes align from
/// the last).
fn broadcast_steps(src: &[usize], out: &[usize]) -> Vec<usize> {
    let offset = out.len() - src.len();
    let mut steps = vec![0; out.len()];
    let mut stride = 1;
    for i in (0..src.len()).rev() {
        if src[i] != 1 {
            steps[offset + i] = stride;
        }
        stride *= src[i];
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_shapes_same() {
        let result = broadcast_shapes(&[3, 4], &[3, 4]);
        assert_eq!(result, Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        let result = broadcast_shapes(&[3, 4], &[1]);
        assert_eq!(result, Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_different_ndim() {
        let result = broadcast_shapes(&[3, 4], &[4]);
        assert_eq!(result, Some(vec![3, 4]));
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        let result = broadcast_shapes(&[3, 4], &[3, 5]);
        assert_eq!(result, None);
    }

    /// A broadcast operand is read at the position NumPy's broadcasting
    /// gives every output element, checked against a reference that decodes
    /// each position independently: across `zip_with`'s blocks (every case
    /// spans several), a carry over more than one axis, size-1 axes on both
    /// sides and an operand of lower rank.
    #[test]
    fn broadcast_reads_every_position_numpy_gives() {
        let cases: &[(&[usize], &[usize])] = &[
            (&[37, 53, 3], &[37, 53, 1]),
            (&[37, 53, 3], &[1, 53, 3]),
            (&[37, 53, 3], &[37, 1, 3]),
            (&[37, 53, 3], &[3]),
            (&[37, 53, 3], &[53, 1]),
            (&[37, 1, 3], &[1, 53, 1]),
            (&[1], &[45, 29, 4]),
            (&[45, 29, 4], &[1, 1, 1]),
            (&[2, 3, 5, 7, 11], &[3, 1, 7, 1]),
        ];
        // The packed index of output position `pos` in an operand of `shape`.
        let index = |pos: &[usize], shape: &[usize]| {
            let offset = pos.len() - shape.len();
            shape.iter().enumerate().fold(0, |acc, (i, &d)| {
                acc * d + if d == 1 { 0 } else { pos[offset + i] }
            })
        };
        for &(sa, sb) in cases {
            let (na, nb): (usize, usize) = (sa.iter().product(), sb.iter().product());
            let a = ViewBuffer::from_vec_with_shape((0..na as u64).collect(), sa.to_vec());
            let b = ViewBuffer::from_vec_with_shape(
                (0..nb as u64).map(|i| i * 1_000_000).collect(),
                sb.to_vec(),
            );
            let out = BinaryOp::Add.execute(&a, &b);
            let shape = out.shape().to_vec();
            for (k, &got) in out.as_slice::<u64>().iter().enumerate() {
                let mut pos = vec![0; shape.len()];
                let mut rest = k;
                for ax in (0..shape.len()).rev() {
                    pos[ax] = rest % shape[ax];
                    rest /= shape[ax];
                }
                let want = index(&pos, sa) as u64 + index(&pos, sb) as u64 * 1_000_000;
                assert_eq!(got, want, "{sa:?} + {sb:?} at {pos:?}");
            }
        }
    }

    #[test]
    fn broadcast_dims_decides_each_axis_on_what_it_knows() {
        use crate::ops::shape_rule::Dim::{Input, Known, Unknown};
        // Two known equal sizes, and a known 1 against anything.
        assert_eq!(
            broadcast_dims(&[Known(8), Known(1)], &[Known(8), Input(1)]),
            Some(vec![Known(8), Input(1)])
        );
        // A known size > 1 against an unknown one: the row runs only if the
        // unknown one is 1 or that size.
        assert_eq!(
            broadcast_dims(&[Known(8), Input(1)], &[Input(0), Input(1)]),
            Some(vec![Known(8), Unknown])
        );
        // Each operand numbers its own symbols: equal ones are not one size.
        assert_eq!(
            broadcast_dims(&[Input(0)], &[Input(0)]),
            Some(vec![Unknown])
        );
        // Missing leading axes broadcast as 1.
        assert_eq!(
            broadcast_dims(&[Known(4), Known(5), Known(3)], &[Known(3)]),
            Some(vec![Known(4), Known(5), Known(3)])
        );
        // Two known sizes that cannot broadcast have no output.
        assert_eq!(
            broadcast_dims(&[Known(4), Input(1)], &[Known(2), Input(1)]),
            None
        );
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
