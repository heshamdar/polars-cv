//! Binary-op kernels: the saturating, blending and bitwise element loops, and
//! the blocked two-operand zip they run through.

use crate::core::buffer::ViewBuffer;
use crate::core::convert::CastFrom;
use crate::core::dtype::{with_dtype, DType, ViewType};
use crate::core::strided::{Cursor, Walk};
use crate::ops::binary::BinaryOp;
use std::mem::MaybeUninit;

impl BinaryOp {
    /// Every op but division, in the operands' own dtype `T`. The op is
    /// matched once, outside the loop, so each arm is its own monomorphic
    /// loop.
    pub(crate) fn execute_typed<T: BinaryElem + FromAny>(
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
pub(crate) fn divide<F: ViewType + num_traits::Float + FromAny>(
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
