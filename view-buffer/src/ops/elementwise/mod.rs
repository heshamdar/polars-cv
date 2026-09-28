//! The element-wise engine: **the one way a per-value compute op runs.**
//!
//! Every op whose output element depends only on its input element (and, for
//! a per-channel `normalize`, the element's channel) executes here: the scalar
//! family, `scale`, `relu`, `clamp`, `invert`, `adjust_gamma`,
//! `adjust_contrast`, `normalize` and fused chains. `cast` is the conversion
//! rule itself (`core::convert`).
//!
//! An op is first **lowered** ([`lower_to_scalars`], shared with fusion) to a
//! [`FusedKernel`]: read each element as f32, run the kernel's scalar passes
//! ([`apply_fused_op_passes`], the one f32 arithmetic authority), write
//! `out_dtype` through the conversion rule. Ops with statistics (`normalize`,
//! `adjust_contrast`) compute them first ([`stats`]) and lower to a kernel
//! over the resulting constants. f64 input to the float-preserving ops
//! computes in f64 instead ([`F64Step`], via `ScalarOp::apply_f64`).
//!
//! How a kernel computes is decided once, here ([`strategy`]):
//! - **Integer affine** when the kernel is exactly `clamp(±x + c)` over
//!   8/16-bit input into the same dtype (`invert`, an integer shift).
//! - **Lookup table** for 8-bit input, and 16-bit input with at least 65,536
//!   elements per table, when the work cannot vectorise (`powf`) or differs
//!   per channel: the kernel runs over every possible input value once and
//!   each element becomes a table read. The table is built by the same
//!   passes and conversion, so the result is identical by construction.
//! - **Blocked** otherwise: blocks of elements read as f32 into an L1
//!   scratch, run through the passes, stored by the conversion rule.
//!
//! Each strategy is an element map; *where* it reads and writes is the one
//! traversal's decision (`core::map`): **in place** when the buffer is its
//! own sole owner and the output dtype is the input dtype, else into a new
//! buffer, reading a view (a crop, a flip, a transpose) where it lies.
//! Statistics are read the same way.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use crate::core::buffer::ViewBuffer;
use crate::core::convert::{convert_slice, CastFrom};
use crate::core::dispatch::{dispatch_mut, SimdKernelMut};
use crate::core::dtype::{with_dtype, DType, ViewType};
use crate::core::map::{map_new, map_owned, ElementMap, ElementMapInPlace};
use crate::ops::compute::{ComputeOp, Normalization};
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::ops::traits::Op;

#[cfg(test)]
mod legacy;
#[cfg(test)]
mod tests;

/// Run the per-value compute op `op` on `buf`.
///
/// # Panics
/// Panics if `op` is not a per-value op (`cast`, the affine family).
pub(crate) fn apply(buf: ViewBuffer, op: &ComputeOp) -> ViewBuffer {
    match lower(op, &buf) {
        Lowered::Kernels(kernels) => run_kernels(buf, &kernels),
        Lowered::F64(steps) => map_owned::<f64, _>(buf, &F64Map(&steps)),
        Lowered::Zeros(out) => zeros(out, buf.shape()),
        Lowered::IntNot => match buf.dtype() {
            DType::U32 => map_owned::<u32, _>(buf, &Not),
            DType::I32 => map_owned::<i32, _>(buf, &Not),
            DType::U64 => map_owned::<u64, _>(buf, &Not),
            DType::I64 => map_owned::<i64, _>(buf, &Not),
            other => {
                unreachable!("invert lowers to a complement only for 32/64-bit, not {other:?}")
            }
        },
    }
}

/// Run one fused kernel on `buf`: [`apply`] for a kernel already built.
pub(crate) fn run_kernel(buf: ViewBuffer, kernel: &FusedKernel) -> ViewBuffer {
    run_kernels(buf, std::slice::from_ref(kernel))
}

/// What a per-value op computes once its statistics are known.
#[derive(Debug)]
enum Lowered {
    /// Read as f32, run the passes, write the kernels' (shared) `out_dtype`.
    /// One kernel applies to every element; `C` kernels apply per channel of
    /// a channels-last buffer.
    Kernels(Vec<FusedKernel>),
    /// f64 input to a float-preserving op: compute and write f64.
    F64(Vec<F64Step>),
    /// Every element is zero of this dtype (a normalize with no spread).
    Zeros(DType),
    /// `invert` of a 32/64-bit integer: `MAX + MIN - x`, the bitwise
    /// complement, in the input dtype. Those dtypes are not exact in f32, so
    /// they cannot take the scalar lowering the 8/16-bit ones do.
    IntNot,
}

/// One step of the f64 path. `ScalarOp` constants are f32; contrast's mean
/// is an f64 statistic and must not be rounded to f32 first.
#[derive(Debug, Clone)]
pub(crate) enum F64Step {
    Scalar(ScalarOp),
    Sub(f64),
    Mul(f64),
    Add(f64),
}

impl F64Step {
    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        match self {
            F64Step::Scalar(op) => op.apply_f64(x),
            F64Step::Sub(c) => x - c,
            F64Step::Mul(c) => x * c,
            F64Step::Add(c) => x + c,
        }
    }
}

fn lower(op: &ComputeOp, buf: &ViewBuffer) -> Lowered {
    let dtype = buf.dtype();
    let one = |ops: Vec<ScalarOp>, out_dtype: DType| {
        Lowered::Kernels(vec![FusedKernel { ops, out_dtype }])
    };
    match op {
        ComputeOp::Fused(kernel) => Lowered::Kernels(vec![kernel.clone()]),
        ComputeOp::Normalize {
            method,
            mean,
            std,
            out_dtype,
        } => lower_normalize(
            &ComputeOp::normalization(*method, mean, std),
            out_dtype.unwrap_or(DType::F32),
            buf,
        ),
        ComputeOp::AdjustContrast { factor } => {
            if dtype == DType::F64 {
                let mean = stats::mean_f64(buf);
                Lowered::F64(vec![
                    F64Step::Sub(mean),
                    F64Step::Mul(*factor as f64),
                    F64Step::Add(mean),
                ])
            } else {
                let mean = stats::contrast_mean_f32(buf);
                one(
                    vec![
                        ScalarOp::Sub(mean),
                        ScalarOp::Mul(*factor),
                        ScalarOp::Add(mean),
                    ],
                    DType::F32,
                )
            }
        }
        ComputeOp::Cast { .. }
        | ComputeOp::Affine(_)
        | ComputeOp::RotateAffine { .. }
        | ComputeOp::WarpAffine { .. }
        | ComputeOp::Rotate { .. } => {
            panic!("internal: `{}` is not a per-value op", op.name())
        }
        // The scalar family, scale, relu, clamp, gamma and invert: the
        // shared lowering.
        _ => {
            let out_dtype = op.output_dtype_rule().resolve(dtype);
            let mut ops = Vec::new();
            if dtype == DType::F64 {
                // Every one of these preserves f64 and lowers for f64 as for
                // any float (gamma's range and invert's maximum are 1).
                assert!(
                    lower_to_scalars(op, DType::F32, true, &mut ops),
                    "internal: `{}` has no scalar lowering",
                    op.name()
                );
                return Lowered::F64(ops.into_iter().map(F64Step::Scalar).collect());
            }
            if lower_to_scalars(op, dtype, true, &mut ops) {
                return one(ops, out_dtype);
            }
            match (op, dtype) {
                (ComputeOp::Invert, DType::U32 | DType::I32 | DType::U64 | DType::I64) => {
                    Lowered::IntNot
                }
                _ => panic!(
                    "internal: `{}` has no scalar lowering for {dtype:?}",
                    op.name()
                ),
            }
        }
    }
}

fn lower_normalize(normalization: &Normalization, out_dtype: DType, buf: &ViewBuffer) -> Lowered {
    let affine = |sub: f32, div: f32| FusedKernel {
        ops: vec![ScalarOp::Sub(sub), ScalarOp::Div(div)],
        out_dtype,
    };
    match normalization {
        Normalization::MinMax => {
            let (min, max) = stats::min_max_f32(buf);
            let range = max - min;
            if range == 0.0 {
                Lowered::Zeros(out_dtype)
            } else {
                Lowered::Kernels(vec![affine(min, range)])
            }
        }
        Normalization::ZScore => {
            let (mean, var) = stats::mean_var_f64(buf);
            let std = var.sqrt() as f32;
            if std == 0.0 {
                Lowered::Zeros(out_dtype)
            } else {
                Lowered::Kernels(vec![affine(mean as f32, std)])
            }
        }
        Normalization::Preset { mean, std } => {
            Lowered::Kernels(mean.iter().zip(std).map(|(&m, &s)| affine(m, s)).collect())
        }
    }
}

fn zeros(out_dtype: DType, shape: &[usize]) -> ViewBuffer {
    let n = shape.iter().product::<usize>();
    with_dtype!(out_dtype, T => ViewBuffer::from_vec_with_shape(vec![0 as T; n], shape.to_vec()))
}

/// How a kernel computes each element. Every strategy is an element map run
/// by the one traversal (`core::map`), which decides whether it writes in
/// place and how a view is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Strategy {
    /// The kernel is `clamp(sign·x + offset)` exactly: integer adds,
    /// subtracts and negations over 8/16-bit input returning the same dtype
    /// (`invert`, an integer brightness shift). Runs in integers at the speed
    /// of the op written natively.
    IntAffine { sign: i32, offset: i32 },
    /// Evaluate the kernel once per possible input value, then read a table.
    Lut,
    /// Stream fixed-size blocks through an f32 scratch in L1: convert in,
    /// run the passes, convert out. No image-sized f32 intermediate.
    Blocked,
}

/// **The one decision** of how a kernel runs.
///
/// A kernel that is an integer affine map of 8/16-bit input into the same
/// dtype runs in integers ([`int_affine`]).
///
/// A lookup table pays only when the per-value work cannot vectorise, or
/// differs per channel: a `powf` per element (gamma) or a per-channel
/// normalize. It needs an enumerable input (8-bit; 16-bit once there are at
/// least as many elements per table as entries). A vectorisable kernel such
/// as `255 - x` streams faster than a table read (measured 0.23 vs 4.1 ms
/// for a 1024² RGB u8 invert). Everything else streams in blocks.
pub(crate) fn strategy(in_dtype: DType, n_elems: usize, kernels: &[FusedKernel]) -> Strategy {
    let tables = kernels.len();
    if let [kernel] = kernels {
        if kernel.out_dtype == in_dtype
            && matches!(in_dtype, DType::U8 | DType::I8 | DType::U16 | DType::I16)
        {
            if let Some((sign, offset)) = int_affine(&kernel.ops) {
                return Strategy::IntAffine { sign, offset };
            }
        }
    }
    let enumerable = match in_dtype {
        DType::U8 | DType::I8 => true,
        DType::U16 | DType::I16 => n_elems >= tables * (1 << 16),
        _ => false,
    };
    let costly = tables > 1
        || kernels
            .iter()
            .any(|k| k.ops.iter().any(scalar_op_is_costly));
    if enumerable && costly {
        Strategy::Lut
    } else {
        Strategy::Blocked
    }
}

/// `(sign, offset)` when `ops` compute exactly `sign·x + offset` for every
/// 8/16-bit `x`: only integer `Add`/`Sub`, `Neg` and `Mul(±1)`, with every
/// intermediate below 2^24 in magnitude so each f32 step is exact. The f32
/// result is then an integer, which the output conversion only saturates,
/// so `clamp(sign·x + offset)` in integers is the same value.
fn int_affine(ops: &[ScalarOp]) -> Option<(i32, i32)> {
    const LIMIT: i64 = 1 << 21;
    let int = |c: f32| (c.fract() == 0.0 && (c.abs() as i64) <= LIMIT).then_some(c as i64);
    let (mut sign, mut offset) = (1i64, 0i64);
    for op in ops {
        match *op {
            ScalarOp::Add(c) => offset += int(c)?,
            ScalarOp::Sub(c) => offset -= int(c)?,
            ScalarOp::Neg => (sign, offset) = (-sign, -offset),
            ScalarOp::Mul(-1.0) => (sign, offset) = (-sign, -offset),
            ScalarOp::Mul(1.0) => {}
            _ => return None,
        }
        if offset.abs() > LIMIT {
            return None;
        }
    }
    Some((sign as i32, offset as i32))
}

/// An op with no vector instruction: `powf` is a libm call per element.
fn scalar_op_is_costly(op: &ScalarOp) -> bool {
    matches!(op, ScalarOp::Pow(_))
}

fn run_kernels(buf: ViewBuffer, kernels: &[FusedKernel]) -> ViewBuffer {
    let out_dtype = kernels[0].out_dtype;
    assert!(
        kernels.iter().all(|k| k.out_dtype == out_dtype),
        "internal: per-channel kernels must share an output dtype"
    );
    let n = buf.shape().iter().product::<usize>();
    assert!(
        kernels.len() == 1 || n % kernels.len() == 0,
        "internal: {} per-channel kernels over {n} elements",
        kernels.len()
    );
    let in_dtype = buf.dtype();
    match strategy(in_dtype, n, kernels) {
        Strategy::IntAffine { sign, offset } => {
            let map = IntAffine { sign, offset };
            match in_dtype {
                DType::U8 => map_owned::<u8, _>(buf, &map),
                DType::I8 => map_owned::<i8, _>(buf, &map),
                DType::U16 => map_owned::<u16, _>(buf, &map),
                DType::I16 => map_owned::<i16, _>(buf, &map),
                other => {
                    unreachable!("strategy picks integer affine only for 8/16-bit, not {other:?}")
                }
            }
        }
        Strategy::Lut => match in_dtype {
            DType::U8 => run_lut::<u8>(buf, kernels),
            DType::I8 => run_lut::<i8>(buf, kernels),
            DType::U16 => run_lut::<u16>(buf, kernels),
            DType::I16 => run_lut::<i16>(buf, kernels),
            other => unreachable!("strategy picks a table only for 8/16-bit, not {other:?}"),
        },
        Strategy::Blocked if in_dtype == out_dtype => {
            with_dtype!(in_dtype, S => map_owned::<S, _>(buf, &Blocked::<S, S>::new(kernels)))
        }
        Strategy::Blocked => with_dtype!(in_dtype, S => with_dtype!(out_dtype, D => {
            map_new(&buf, &Blocked::<S, D>::new(kernels))
        })),
    }
}

/// An 8/16-bit element type the integer affine strategy runs over.
trait SmallInt: ViewType {
    /// `data[i] = clamp(sign·data[i] + offset)`.
    fn affine_in_place(data: &mut [Self], sign: i32, offset: i32);
    /// `dst[i] = clamp(sign·src[i] + offset)`.
    fn affine_into(src: &[Self], dst: &mut [MaybeUninit<Self>], sign: i32, offset: i32);
}

/// The affine map on one element type, in the narrowest lanes that are
/// exact:
/// - `x → MAX + MIN − x` (`invert`) cannot saturate, so it is native wrapping
///   arithmetic in the element type itself (as many lanes as bytes allow);
/// - otherwise in `$wide` (i16 for 8-bit, i32 for 16-bit), after clamping the
///   offset to `±2·(MAX − MIN)`: beyond that every element saturates to the
///   same bound whatever the offset, so the clamp changes no result and keeps
///   `sign·x + offset` inside `$wide`.
macro_rules! small_int {
    ($t:ty, $wide:ty) => {
        impl SmallInt for $t {
            #[inline(always)]
            fn affine_in_place(data: &mut [Self], sign: i32, offset: i32) {
                let (min, max) = (i32::from(<$t>::MIN), i32::from(<$t>::MAX));
                if sign == -1 && offset == max + min {
                    let o = offset as $t;
                    for x in data.iter_mut() {
                        *x = o.wrapping_sub(*x);
                    }
                    return;
                }
                let span = 2 * (max - min);
                let (s, o) = (sign as $wide, offset.clamp(-span, span) as $wide);
                let (lo, hi) = (min as $wide, max as $wide);
                for x in data.iter_mut() {
                    *x = (s * (*x as $wide) + o).clamp(lo, hi) as $t;
                }
            }

            #[inline(always)]
            fn affine_into(src: &[Self], dst: &mut [MaybeUninit<Self>], sign: i32, offset: i32) {
                let (min, max) = (i32::from(<$t>::MIN), i32::from(<$t>::MAX));
                if sign == -1 && offset == max + min {
                    let o = offset as $t;
                    for (d, &x) in dst.iter_mut().zip(src) {
                        d.write(o.wrapping_sub(x));
                    }
                    return;
                }
                let span = 2 * (max - min);
                let (s, o) = (sign as $wide, offset.clamp(-span, span) as $wide);
                let (lo, hi) = (min as $wide, max as $wide);
                for (d, &x) in dst.iter_mut().zip(src) {
                    d.write((s * (x as $wide) + o).clamp(lo, hi) as $t);
                }
            }
        }
    };
}
small_int!(u8, i16);
small_int!(i8, i16);
small_int!(u16, i32);
small_int!(i16, i32);

/// The integer affine strategy's map.
#[derive(Debug, Clone, Copy)]
struct IntAffine {
    sign: i32,
    offset: i32,
}

// SAFETY: `affine_into` writes every element of `dst`.
unsafe impl<S: SmallInt> ElementMap<S, S> for IntAffine {
    #[inline(always)]
    fn map_into(&self, src: &[S], dst: &mut [MaybeUninit<S>], _at: usize) {
        S::affine_into(src, dst, self.sign, self.offset);
    }
}

impl<S: SmallInt> ElementMapInPlace<S> for IntAffine {
    #[inline(always)]
    fn map_in_place(&self, data: &mut [S]) {
        S::affine_in_place(data, self.sign, self.offset);
    }
}

/// The integer complement `!x` (`invert` of a 32/64-bit integer).
struct Not;

// SAFETY: `map_into` writes every element of `dst`.
unsafe impl<T: ViewType + std::ops::Not<Output = T>> ElementMap<T, T> for Not {
    #[inline(always)]
    fn map_into(&self, src: &[T], dst: &mut [MaybeUninit<T>], _at: usize) {
        for (d, &x) in dst.iter_mut().zip(src) {
            d.write(!x);
        }
    }
}

impl<T: ViewType + std::ops::Not<Output = T>> ElementMapInPlace<T> for Not {
    #[inline(always)]
    fn map_in_place(&self, data: &mut [T]) {
        for x in data.iter_mut() {
            *x = !*x;
        }
    }
}

/// Elements per block of the blocked strategy: an 8 KiB f32 scratch on the
/// stack, well inside L1.
const BLOCK: usize = 2048;

/// The blocked strategy's map: each block of elements is read as f32 into a
/// stack scratch, run through the kernels' passes, and stored as `D`. One
/// kernel applies to every element; `C` kernels apply per channel of a
/// channels-last buffer, element `i` of the buffer being channel `i % C`.
struct Blocked<'a, S, D> {
    kernels: &'a [FusedKernel],
    _types: PhantomData<fn(S) -> D>,
}

impl<'a, S, D> Blocked<'a, S, D> {
    fn new(kernels: &'a [FusedKernel]) -> Self {
        Blocked {
            kernels,
            _types: PhantomData,
        }
    }

    /// The kernels' passes over one block of f32 values, the first being
    /// element `at` of the buffer.
    #[inline(always)]
    fn passes(&self, acc: &mut [f32], at: usize) {
        let kernels = self.kernels;
        if let [kernel] = kernels {
            apply_fused_op_passes(acc, &kernel.ops);
            return;
        }
        // Per channel: channel `c`'s values in the block, gathered into a
        // second scratch, run, and scattered back, so the passes stay the
        // single f32 arithmetic.
        let channels = kernels.len();
        let mut lane = [0.0f32; BLOCK];
        for (c, kernel) in kernels.iter().enumerate() {
            let first = (c + channels - at % channels) % channels;
            let values = acc.iter().skip(first).step_by(channels);
            let len = values.len();
            for (l, &v) in lane.iter_mut().zip(values) {
                *l = v;
            }
            apply_fused_op_passes(&mut lane[..len], &kernel.ops);
            for (v, &l) in acc.iter_mut().skip(first).step_by(channels).zip(&lane) {
                *v = l;
            }
        }
    }
}

// SAFETY: the blocks of `dst` pair up with those of `src`, and each block
// writes one value per element.
unsafe impl<S, D> ElementMap<S, D> for Blocked<'_, S, D>
where
    S: ViewType,
    D: ViewType + CastFrom<f32>,
    f32: CastFrom<S>,
{
    #[inline(always)]
    fn map_into(&self, src: &[S], dst: &mut [MaybeUninit<D>], at: usize) {
        for (k, (src, dst)) in src.chunks(BLOCK).zip(dst.chunks_mut(BLOCK)).enumerate() {
            let mut acc = [0.0f32; BLOCK];
            let acc = &mut acc[..src.len()];
            for (a, &x) in acc.iter_mut().zip(src) {
                *a = f32::cast_from(x);
            }
            self.passes(acc, at + k * BLOCK);
            for (d, &a) in dst.iter_mut().zip(acc.iter()) {
                d.write(D::cast_from(a));
            }
        }
    }
}

impl<S> ElementMapInPlace<S> for Blocked<'_, S, S>
where
    S: ViewType + CastFrom<f32>,
    f32: CastFrom<S>,
{
    #[inline(always)]
    fn map_in_place(&self, data: &mut [S]) {
        if let Some(data) = as_f32_mut(data) {
            // Already f32: the passes run on the buffer's own blocks, with no
            // copy into the scratch and back (the conversions would be the
            // identity). Blocked, so a chain of passes stays in L1.
            for (k, block) in data.chunks_mut(BLOCK).enumerate() {
                self.passes(block, k * BLOCK);
            }
            return;
        }
        for (k, block) in data.chunks_mut(BLOCK).enumerate() {
            let mut acc = [0.0f32; BLOCK];
            let acc = &mut acc[..block.len()];
            for (a, &x) in acc.iter_mut().zip(block.iter()) {
                *a = f32::cast_from(x);
            }
            self.passes(acc, k * BLOCK);
            for (x, &a) in block.iter_mut().zip(acc.iter()) {
                *x = S::cast_from(a);
            }
        }
    }
}

/// `data` as f32, when that is its element type.
#[inline(always)]
fn as_f32_mut<S: ViewType>(data: &mut [S]) -> Option<&mut [f32]> {
    (S::DTYPE == DType::F32).then(|| {
        // SAFETY: `S::DTYPE` is `F32` only for `S = f32` (`ViewType`'s one
        // implementation per dtype), so this is the same slice.
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<f32>(), data.len()) }
    })
}

/// An element type small enough to enumerate: a table index per value.
trait LutIndex: ViewType {
    /// How many values the type has.
    const DOMAIN: usize;
    /// The value's table index (its bit pattern).
    fn index(self) -> usize;
    /// The value at table index `i`, as the kernel reads it (`x as f32`).
    fn value_f32(i: usize) -> f32;
}

macro_rules! lut_index {
    ($t:ty, $bits:ty) => {
        impl LutIndex for $t {
            const DOMAIN: usize = 1 << (8 * std::mem::size_of::<$t>());
            #[inline(always)]
            fn index(self) -> usize {
                self as $bits as usize
            }
            #[inline(always)]
            fn value_f32(i: usize) -> f32 {
                (i as $bits as $t) as f32
            }
        }
    };
}
lut_index!(u8, u8);
lut_index!(i8, u8);
lut_index!(u16, u16);
lut_index!(i16, u16);

/// Table strategy: the kernels over every possible value of `S`, then a
/// table read per element, in place when the output dtype is `S`.
fn run_lut<S: LutIndex + CastFrom<f32>>(buf: ViewBuffer, kernels: &[FusedKernel]) -> ViewBuffer {
    let out_dtype = kernels[0].out_dtype;
    if out_dtype == S::DTYPE {
        return map_owned::<S, _>(buf, &Lut::<S, S>::new(kernels));
    }
    with_dtype!(out_dtype, D => map_new(&buf, &Lut::<S, D>::new(kernels)))
}

/// A lookup table: `table[c * S::DOMAIN + x.index()]` is the kernel for
/// channel `c` applied to value `x`, computed by the kernel's own passes and
/// conversion, so a read equals the computation by construction.
struct Lut<S, D> {
    table: Vec<D>,
    channels: usize,
    _source: PhantomData<fn(S)>,
}

impl<S: LutIndex, D: ViewType + CastFrom<f32>> Lut<S, D> {
    fn new(kernels: &[FusedKernel]) -> Self {
        let domain: Vec<f32> = (0..S::DOMAIN).map(S::value_f32).collect();
        let mut table: Vec<D> = Vec::with_capacity(kernels.len() * S::DOMAIN);
        for kernel in kernels {
            let mut values = domain.clone();
            run_passes(&mut values, &kernel.ops);
            table.extend(convert_slice::<f32, D>(&values));
        }
        Lut {
            table,
            channels: kernels.len(),
            _source: PhantomData,
        }
    }
}

// SAFETY: every branch writes one value per element of `dst`.
unsafe impl<S: LutIndex, D: ViewType> ElementMap<S, D> for Lut<S, D> {
    #[inline(always)]
    fn map_into(&self, src: &[S], dst: &mut [MaybeUninit<D>], at: usize) {
        // A slice, not `&Vec`: the lookups then read its pointer from a
        // register rather than reloading it past every store.
        let (table, channels): (&[D], usize) = (&self.table, self.channels);
        if channels == 1 {
            for (d, &x) in dst.iter_mut().zip(src) {
                d.write(lookup(table, x.index()));
            }
        } else if at.is_multiple_of(channels) && src.len().is_multiple_of(channels) {
            // Whole pixels: the channel is the position in the pixel, a
            // constant for 3 or 4 channels (a loop over a run-time count
            // compiled 20% slower in the AVX2 build).
            match channels {
                3 => lut_pixels::<S, D, 3>(table, src, dst),
                4 => lut_pixels::<S, D, 4>(table, src, dst),
                _ => {
                    for (d, px) in dst
                        .chunks_exact_mut(channels)
                        .zip(src.chunks_exact(channels))
                    {
                        for (c, (d, &x)) in d.iter_mut().zip(px).enumerate() {
                            d.write(lookup(table, c * S::DOMAIN + x.index()));
                        }
                    }
                }
            }
        } else {
            let mut c = at % channels;
            for (d, &x) in dst.iter_mut().zip(src) {
                d.write(lookup(table, c * S::DOMAIN + x.index()));
                c += 1;
                if c == channels {
                    c = 0;
                }
            }
        }
    }
}

impl<S: LutIndex> ElementMapInPlace<S> for Lut<S, S> {
    #[inline(always)]
    fn map_in_place(&self, data: &mut [S]) {
        let (table, channels): (&[S], usize) = (&self.table, self.channels);
        if channels == 1 {
            for x in data.iter_mut() {
                *x = lookup(table, x.index());
            }
        } else {
            // As in `map_into`: a constant channel count for 3 or 4.
            // Inlined, so it is compiled into the dispatched build.
            #[inline(always)]
            fn pixels<S: LutIndex, const C: usize>(table: &[S], data: &mut [S]) {
                for px in data.as_chunks_mut::<C>().0 {
                    for (c, x) in px.iter_mut().enumerate() {
                        *x = lookup(table, c * S::DOMAIN + x.index());
                    }
                }
            }
            match channels {
                3 => pixels::<S, 3>(table, data),
                4 => pixels::<S, 4>(table, data),
                _ => {
                    for px in data.chunks_exact_mut(channels) {
                        for (c, x) in px.iter_mut().enumerate() {
                            *x = lookup(table, c * S::DOMAIN + x.index());
                        }
                    }
                }
            }
        }
    }
}

/// Table reads over whole `C`-channel pixels: `dst[i] = table[c * DOMAIN +
/// src[i]]` for element `i`, channel `c = i % C`.
#[inline(always)]
fn lut_pixels<S: LutIndex, D: Copy, const C: usize>(
    table: &[D],
    src: &[S],
    dst: &mut [MaybeUninit<D>],
) {
    let (src, _) = src.as_chunks::<C>();
    let (dst, _) = dst.as_chunks_mut::<C>();
    for (d, p) in dst.iter_mut().zip(src) {
        for c in 0..C {
            d[c].write(lookup(table, c * S::DOMAIN + p[c].index()));
        }
    }
}

/// `table[i]`, for an index built from a `LutIndex` value and a channel.
///
/// Unchecked: `index()` is below `S::DOMAIN` by construction and the channel
/// below the kernel count, and the table holds `kernels × S::DOMAIN`
/// entries. Checked in debug builds.
#[inline(always)]
fn lookup<D: Copy>(table: &[D], i: usize) -> D {
    debug_assert!(i < table.len());
    // SAFETY: see above; every caller indexes as `c * DOMAIN + index()`.
    unsafe { *table.get_unchecked(i) }
}

/// The scalar passes, dispatched so the wheels get AVX2 (vector rounding for
/// floor/ceil/round/trunc on an SSE2 baseline).
fn run_passes(data: &mut [f32], ops: &[ScalarOp]) {
    dispatch_mut(&Passes { ops }, data);
}

struct Passes<'a> {
    ops: &'a [ScalarOp],
}

impl SimdKernelMut<f32> for Passes<'_> {
    #[inline(always)]
    fn run_mut(&self, data: &mut [f32]) {
        apply_fused_op_passes(data, self.ops);
    }
}

/// f64 input to a float-preserving op: each element through the steps in
/// order, in f64.
struct F64Map<'a>(&'a [F64Step]);

impl F64Map<'_> {
    #[inline(always)]
    fn apply(&self, x: f64) -> f64 {
        self.0.iter().fold(x, |v, step| step.apply(v))
    }
}

// SAFETY: `map_into` writes every element of `dst`.
unsafe impl ElementMap<f64, f64> for F64Map<'_> {
    #[inline(always)]
    fn map_into(&self, src: &[f64], dst: &mut [MaybeUninit<f64>], _at: usize) {
        for (d, &x) in dst.iter_mut().zip(src) {
            d.write(self.apply(x));
        }
    }
}

impl ElementMapInPlace<f64> for F64Map<'_> {
    #[inline(always)]
    fn map_in_place(&self, data: &mut [f64]) {
        for x in data.iter_mut() {
            *x = self.apply(*x);
        }
    }
}

/// Statistics a per-value op needs before it can lower to a kernel, read in
/// logical element order from the buffer's runs, never from a packed copy.
pub(crate) mod stats {
    use crate::core::buffer::ViewBuffer;
    use crate::core::convert::CastFrom;
    use crate::core::dtype::{with_dtype, DType, ViewType};
    use crate::core::map::for_each_run;

    /// The starting value of `f64`'s `Sum`, so a sum carried across runs
    /// adds exactly what one `sum()` over the elements would.
    fn sum_start() -> f64 {
        std::iter::empty::<f64>().sum()
    }

    /// Minimum and maximum of the elements as f32 read them. Integers take
    /// their integer extremes (`as f32` is monotone, so this equals a fold
    /// over the converted values); floats fold in element order with
    /// `f32::min`/`max`, which skip NaN. `(inf, -inf)` for no elements.
    pub(crate) fn min_max_f32(buf: &ViewBuffer) -> (f32, f32) {
        fn int_extremes<T: ViewType + Ord>(buf: &ViewBuffer) -> Option<(T, T)> {
            let mut extremes: Option<(T, T)> = None;
            for_each_run::<T>(buf, |run| {
                if let (Some(&lo), Some(&hi)) = (run.iter().min(), run.iter().max()) {
                    extremes = Some(match extremes {
                        Some((a, b)) => (a.min(lo), b.max(hi)),
                        None => (lo, hi),
                    });
                }
            });
            extremes
        }
        fn float_extremes<T: ViewType>(buf: &ViewBuffer) -> (f32, f32)
        where
            f32: CastFrom<T>,
        {
            let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
            for_each_run::<T>(buf, |run| {
                for &x in run {
                    let v = f32::cast_from(x);
                    min = min.min(v);
                    max = max.max(v);
                }
            });
            (min, max)
        }
        macro_rules! ints {
            ($t:ty) => {
                int_extremes::<$t>(buf).map_or((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi)| {
                    (lo as f32, hi as f32)
                })
            };
        }
        match buf.dtype() {
            DType::U8 => ints!(u8),
            DType::I8 => ints!(i8),
            DType::U16 => ints!(u16),
            DType::I16 => ints!(i16),
            DType::U32 => ints!(u32),
            DType::I32 => ints!(i32),
            DType::U64 => ints!(u64),
            DType::I64 => ints!(i64),
            DType::F32 => float_extremes::<f32>(buf),
            DType::F64 => float_extremes::<f64>(buf),
        }
    }

    /// The exact sum of an 8/16-bit buffer.
    fn small_int_sum(buf: &ViewBuffer) -> Option<i64> {
        fn sum<T: ViewType + Into<i64>>(buf: &ViewBuffer) -> i64 {
            let mut sum = 0i64;
            for_each_run::<T>(buf, |run| sum += run.iter().map(|&x| x.into()).sum::<i64>());
            sum
        }
        match buf.dtype() {
            DType::U8 => Some(sum::<u8>(buf)),
            DType::I8 => Some(sum::<i8>(buf)),
            DType::U16 => Some(sum::<u16>(buf)),
            DType::I16 => Some(sum::<i16>(buf)),
            _ => None,
        }
    }

    /// Exact sums of an 8/16-bit buffer: `(n, Σx, Σx²)`.
    fn small_int_sums(buf: &ViewBuffer) -> Option<(usize, i128, i128)> {
        fn sums<T: ViewType + Into<i64>>(buf: &ViewBuffer) -> (usize, i128, i128) {
            let (mut n, mut sum, mut sum_sq) = (0usize, 0i64, 0u128);
            for_each_run::<T>(buf, |run| {
                n += run.len();
                sum += run.iter().map(|&x| x.into()).sum::<i64>();
                sum_sq += run
                    .iter()
                    .map(|&x| {
                        let x: i64 = x.into();
                        (x * x) as u64
                    })
                    .fold(0u128, |acc, sq| acc + u128::from(sq));
            });
            (n, i128::from(sum), sum_sq as i128)
        }
        match buf.dtype() {
            DType::U8 => Some(sums::<u8>(buf)),
            DType::I8 => Some(sums::<i8>(buf)),
            DType::U16 => Some(sums::<u16>(buf)),
            DType::I16 => Some(sums::<i16>(buf)),
            _ => None,
        }
    }

    /// The f64 sum of the elements read as f32, in element order: what
    /// `as_f32(buf).iter().map(|&x| x as f64).sum()` computes, without the
    /// f32 copy.
    fn f32_read_sum(buf: &ViewBuffer, term: impl Fn(f64) -> f64) -> f64 {
        let mut sum = sum_start();
        with_dtype!(buf.dtype(), T => for_each_run::<T>(buf, |run| {
            for &x in run {
                sum += term(f32::cast_from(x) as f64);
            }
        }));
        sum
    }

    /// `adjust_contrast`'s mean, as it has always been computed: the f64 sum
    /// of the elements read as f32, rounded to f32, over the count as f32.
    /// An 8/16-bit sum is taken exactly in integers, which equals that f64
    /// sum (every partial sum is an integer below 2^53).
    pub(crate) fn contrast_mean_f32(buf: &ViewBuffer) -> f32 {
        let n = buf.shape().iter().product::<usize>();
        if n == 0 {
            return 0.0;
        }
        let sum = match small_int_sum(buf) {
            Some(sum) => sum as f64,
            None => f32_read_sum(buf, |x| x),
        };
        sum as f32 / n as f32
    }

    /// The f64 mean of an f64 buffer, summed in element order.
    pub(crate) fn mean_f64(buf: &ViewBuffer) -> f64 {
        let n = buf.shape().iter().product::<usize>();
        if n == 0 {
            return 0.0;
        }
        let mut sum = sum_start();
        for_each_run::<f64>(buf, |run| {
            for &x in run {
                sum += x;
            }
        });
        sum / n as f64
    }

    /// Mean and (population) variance for `normalize(method="zscore")`, in
    /// f64. 8/16-bit input is exact: integer `Σx` and `Σx²`, and
    /// `var = (n·Σx² − (Σx)²) / n²` from an exact integer numerator. Other
    /// input is read as f32 (as the kernel reads it) and summed in f64 in two
    /// passes, in element order.
    pub(crate) fn mean_var_f64(buf: &ViewBuffer) -> (f64, f64) {
        if let Some((n, sum, sum_sq)) = small_int_sums(buf) {
            if n == 0 {
                return (f64::NAN, f64::NAN);
            }
            let n_i = n as i128;
            let numerator = n_i * sum_sq - sum * sum;
            let n_f = n as f64;
            return (sum as f64 / n_f, numerator as f64 / (n_f * n_f));
        }
        let n = buf.shape().iter().product::<usize>() as f64;
        let mean = f32_read_sum(buf, |x| x) / n;
        let var = f32_read_sum(buf, |x| {
            let d = x - mean;
            d * d
        }) / n;
        (mean, var)
    }
}

/// Lowers one `ComputeOp` into the scalar ops a `FusedKernel` runs.
///
/// **The one lowering**: fusion (`expr.rs::try_fuse`) and standalone
/// execution ([`apply`]) both read it, so a fused chain and the same ops run
/// one at a time compute the same thing.
///
/// `input_dtype` is needed because some lowerings are dtype-dependent:
/// `Invert` and the gamma family use the dtype's value range, so the same op
/// becomes different scalar work for `u8` than for `f32`.
///
/// `is_outer` marks the op at the end of the chain. An outer `Cast` lowers to
/// *no* scalar ops at all: the kernel already converts its `f32` result to
/// `FusedKernel::out_dtype` on write, and `try_fuse` pins that dtype to what
/// the unfused chain would have produced — so emitting a cast here would apply
/// the conversion twice.
pub(crate) fn lower_to_scalars(
    op: &ComputeOp,
    input_dtype: DType,
    is_outer: bool,
    list: &mut Vec<ScalarOp>,
) -> bool {
    // The float-promoting scalar family is excluded for f64 inputs: the
    // dtype contract preserves f64 (and the unfused runtime now computes in
    // f64), but the fused kernel computes in f32 — fusing would silently
    // drop precision. f64 chains simply stay unfused.
    let promote_family_fusable = input_dtype != DType::F64;
    match op {
        ComputeOp::Scale { factor } if promote_family_fusable => {
            list.push(ScalarOp::Mul(*factor));
            true
        }
        ComputeOp::Relu if promote_family_fusable => {
            list.push(ScalarOp::Relu);
            true
        }
        ComputeOp::Clamp { min, max } if promote_family_fusable => {
            list.push(ScalarOp::Clamp(*min, *max));
            true
        }
        // The core math primitives: each is already a single `ScalarOp`, so
        // lowering is a direct push. f64 stays unfused (the kernel is f32),
        // via the same promote-family gate as the ops above.
        op if promote_family_fusable && op.scalar().is_some() => {
            list.extend(op.scalar());
            true
        }
        // Gamma is scan-free and lowers exactly to its unfused formula:
        // `((x / max).clamp(0, 1)).powf(g) * max`, max = the input dtype's
        // value range for integers, 1 for float inputs (matching
        // `apply_adjust_gamma` via the same `norm_range_max_f32`).
        ComputeOp::AdjustGamma { gamma: g } if promote_family_fusable => {
            let max_val: f32 = input_dtype.norm_range_max_f32();
            if max_val != 1.0 {
                list.push(ScalarOp::Div(max_val));
            }
            list.push(ScalarOp::Clamp(0.0, 1.0));
            list.push(ScalarOp::Pow(*g));
            if max_val != 1.0 {
                list.push(ScalarOp::Mul(max_val));
            }
            true
        }
        // Invert maps the value range onto itself: `MAX + MIN - x` for an
        // integer dtype (`255 - x` for u8, `-1 - x` for a signed one, i.e.
        // `!x`), `1 - x` for a float. Written `-x + (MAX + MIN)`: exact in f32
        // for the 8/16-bit dtypes and bit-identical in IEEE for f32. f64 would
        // lose precision in the f32 kernel, and 32/64-bit integers are not
        // exact in f32, so those stay unfused (`Lowered::IntNot`).
        ComputeOp::Invert => {
            let max_val: f32 = match input_dtype {
                DType::U8 => 255.0,
                DType::U16 => 65535.0,
                DType::I8 => -1.0,
                DType::I16 => -1.0,
                DType::F32 => 1.0,
                _ => return false,
            };
            list.push(ScalarOp::Mul(-1.0));
            list.push(ScalarOp::Add(max_val));
            true
        }
        // A cast is the kernel's own read/write conversion:
        // - as the chain's last op, the kernel's out_dtype performs it;
        // - mid-chain, only cast-to-f32 is a no-op (the kernel computes in
        //   f32 anyway); other mid-chain casts quantize and must materialize.
        ComputeOp::Cast { dtype: target } => is_outer || *target == DType::F32,
        // An existing kernel can be extended only while its result is still
        // raw f32 — a non-f32 out_dtype is a quantization step that later
        // ops must observe.
        ComputeOp::Fused(k) => {
            if !is_outer && k.out_dtype != DType::F32 {
                return false;
            }
            list.extend(k.ops.iter().cloned());
            true
        }
        _ => false,
    }
}

/// Apply each fused scalar op as a full-array pass over `f32` data.
///
/// One pass per op (vs one pass total) costs slightly more bandwidth but lets
/// LLVM auto-vectorize each inner loop with AVX/AVX2/NEON — the inner loop is
/// a simple scalar operation with no enum dispatch. The bandwidth tradeoff
/// breaks even at ~2 ops for typical L2-resident sizes.
#[inline(always)]
pub(crate) fn apply_fused_op_passes(data: &mut [f32], ops: &[ScalarOp]) {
    for op in ops {
        match op {
            ScalarOp::Add(c) => {
                for x in data.iter_mut() {
                    *x += *c;
                }
            }
            ScalarOp::Sub(c) => {
                for x in data.iter_mut() {
                    *x -= *c;
                }
            }
            ScalarOp::Mul(c) => {
                for x in data.iter_mut() {
                    *x *= *c;
                }
            }
            ScalarOp::Div(c) => {
                for x in data.iter_mut() {
                    *x /= *c;
                }
            }
            ScalarOp::Pow(c) => {
                for x in data.iter_mut() {
                    *x = x.powf(*c);
                }
            }
            ScalarOp::Neg => {
                for x in data.iter_mut() {
                    *x = -*x;
                }
            }
            ScalarOp::Abs => {
                for x in data.iter_mut() {
                    *x = x.abs();
                }
            }
            ScalarOp::Sqrt => {
                for x in data.iter_mut() {
                    *x = x.sqrt();
                }
            }
            ScalarOp::Square => {
                for x in data.iter_mut() {
                    *x *= *x;
                }
            }
            ScalarOp::Recip => {
                for x in data.iter_mut() {
                    *x = 1.0 / *x;
                }
            }
            ScalarOp::Min(c) => {
                for x in data.iter_mut() {
                    *x = x.min(*c);
                }
            }
            ScalarOp::Max(c) => {
                for x in data.iter_mut() {
                    *x = x.max(*c);
                }
            }
            ScalarOp::Sign => {
                for x in data.iter_mut() {
                    *x = crate::ops::scalar::signum_numpy(*x);
                }
            }
            ScalarOp::Floor => {
                for x in data.iter_mut() {
                    *x = x.floor();
                }
            }
            ScalarOp::Ceil => {
                for x in data.iter_mut() {
                    *x = x.ceil();
                }
            }
            ScalarOp::Round => {
                for x in data.iter_mut() {
                    *x = x.round_ties_even();
                }
            }
            ScalarOp::Trunc => {
                for x in data.iter_mut() {
                    *x = x.trunc();
                }
            }
            ScalarOp::Relu => {
                for x in data.iter_mut() {
                    *x = x.max(0.0);
                }
            }
            ScalarOp::Clamp(lo, hi) => {
                for x in data.iter_mut() {
                    *x = x.clamp(*lo, *hi);
                }
            }
        }
    }
}
