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
//! How a kernel runs is decided once, here ([`strategy`]):
//! - **Lookup table** for 8-bit input, and 16-bit input with at least 65,536
//!   elements per table: the kernel runs over every possible input value once
//!   and each element becomes a table read. The table is built by the same
//!   passes and conversion, so the result is identical by construction, and
//!   a `powf` per pixel becomes one per possible value.
//! - **Passes** otherwise: gather to f32, run the passes (dispatched, so the
//!   wheels get AVX2), convert.
//!
//! Either writes **in place** when the buffer is its own sole owner
//! ([`ViewBuffer::unique_contiguous_mut`]) and the output dtype is the input
//! dtype, as a u8 `invert` or a u8 → u8 chain is.

use std::any::Any;
use std::sync::Arc;

use crate::core::buffer::{BufferStorage, ViewBuffer};
use crate::core::bytes::AlignedBytes;
use crate::core::convert::{convert_slice, convert_view, CastFrom};
use crate::core::dispatch::{dispatch, dispatch_mut, SimdKernel, SimdKernelMut};
use crate::core::dtype::{with_dtype, DType, ViewType};
use crate::core::layout::Layout;
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
        Lowered::F64(steps) => run_f64(buf, &steps),
        Lowered::Zeros(out) => zeros(out, buf.shape()),
        Lowered::IntNot => match buf.dtype() {
            DType::U32 => run_not::<u32>(buf),
            DType::I32 => run_not::<i32>(buf),
            DType::U64 => run_not::<u64>(buf),
            DType::I64 => run_not::<i64>(buf),
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

/// How a kernel runs over a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Strategy {
    /// The kernel is `clamp(sign·x + offset)` exactly: integer adds,
    /// subtracts and negations over 8/16-bit input returning the same dtype
    /// (`invert`, an integer brightness shift). Runs in integers at the speed
    /// of the op written natively.
    IntAffine { sign: i32, offset: i32 },
    /// Evaluate the kernel once per possible input value, then read a table.
    Lut,
    /// Stream fixed-size blocks through an f32 scratch: convert in, run the
    /// passes, convert out, into the input (sole owner, same dtype) or a new
    /// output of the integer dtype. No image-sized f32 intermediate.
    Blocked,
    /// Gather every element to f32, run the passes over the whole buffer,
    /// convert: the f32 buffer *is* the output for a float result.
    Pass,
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
/// for a 1024² RGB u8 invert). Among streaming kernels, an integer result
/// runs in blocks and a float result in one f32 pass.
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
    } else if tables == 1 && !matches!(kernels[0].out_dtype, DType::F32 | DType::F64) {
        Strategy::Blocked
    } else {
        Strategy::Pass
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
    match (strategy(in_dtype, n, kernels), in_dtype) {
        (Strategy::IntAffine { sign, offset }, _) => {
            let map = IntAffine { sign, offset };
            match in_dtype {
                DType::U8 => run_int_affine::<u8>(buf, map),
                DType::I8 => run_int_affine::<i8>(buf, map),
                DType::U16 => run_int_affine::<u16>(buf, map),
                DType::I16 => run_int_affine::<i16>(buf, map),
                other => {
                    unreachable!("strategy picks integer affine only for 8/16-bit, not {other:?}")
                }
            }
        }
        (Strategy::Lut, DType::U8) => run_lut_from::<u8>(buf, kernels),
        (Strategy::Lut, DType::I8) => run_lut_from::<i8>(buf, kernels),
        (Strategy::Lut, DType::U16) => run_lut_from::<u16>(buf, kernels),
        (Strategy::Lut, DType::I16) => run_lut_from::<i16>(buf, kernels),
        (Strategy::Lut, other) => {
            unreachable!("strategy picks a table only for 8/16-bit, not {other:?}")
        }
        (Strategy::Blocked, _) if in_dtype == out_dtype => {
            with_dtype!(in_dtype, S => run_blocked_same::<S>(buf, &kernels[0]))
        }
        (Strategy::Blocked, _) => {
            with_dtype!(in_dtype, S => with_dtype!(out_dtype, D => {
                run_blocked_into::<S, D>(&buf, &kernels[0])
            }))
        }
        (Strategy::Pass, _) => run_pass(buf, kernels),
    }
}

/// An 8/16-bit element type the integer affine strategy runs over.
trait SmallInt: ViewType {
    /// `data[i] = clamp(sign·data[i] + offset)`.
    fn affine_in_place(data: &mut [Self], sign: i32, offset: i32);
    /// `clamp(sign·x + offset)` for every `x` of `src`.
    fn affine_into(src: &[Self], sign: i32, offset: i32) -> Vec<Self>;
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
            fn affine_into(src: &[Self], sign: i32, offset: i32) -> Vec<Self> {
                let (min, max) = (i32::from(<$t>::MIN), i32::from(<$t>::MAX));
                if sign == -1 && offset == max + min {
                    let o = offset as $t;
                    return src.iter().map(|&x| o.wrapping_sub(x)).collect();
                }
                let span = 2 * (max - min);
                let (s, o) = (sign as $wide, offset.clamp(-span, span) as $wide);
                let (lo, hi) = (min as $wide, max as $wide);
                src.iter()
                    .map(|&x| (s * (x as $wide) + o).clamp(lo, hi) as $t)
                    .collect()
            }
        }
    };
}
small_int!(u8, i16);
small_int!(i8, i16);
small_int!(u16, i32);
small_int!(i16, i32);

#[derive(Debug, Clone, Copy)]
struct IntAffine {
    sign: i32,
    offset: i32,
}

impl<S: SmallInt> SimdKernelMut<S> for IntAffine {
    #[inline(always)]
    fn run_mut(&self, data: &mut [S]) {
        S::affine_in_place(data, self.sign, self.offset);
    }
}

#[derive(Clone)]
struct IntAffineInto<'a, S> {
    src: &'a [S],
    map: IntAffine,
}

impl<S: SmallInt> SimdKernel for IntAffineInto<'_, S> {
    type Output = Vec<S>;

    #[inline(always)]
    fn run(self) -> Vec<S> {
        S::affine_into(self.src, self.map.sign, self.map.offset)
    }
}

/// Integer affine strategy: in place for a sole owner, else into a new buffer.
fn run_int_affine<S: SmallInt>(mut buf: ViewBuffer, map: IntAffine) -> ViewBuffer {
    if let Some(data) = buf.unique_contiguous_mut::<S>() {
        dispatch_mut(&map, data);
        return buf;
    }
    let shape = buf.shape().to_vec();
    let packed = buf.to_contiguous();
    let out = dispatch(IntAffineInto {
        src: packed.as_slice::<S>(),
        map,
    });
    ViewBuffer::from_vec_with_shape(out, shape)
}

/// `!x` over a slice, in place.
struct Not;

impl<T: ViewType + std::ops::Not<Output = T>> SimdKernelMut<T> for Not {
    #[inline(always)]
    fn run_mut(&self, data: &mut [T]) {
        for x in data.iter_mut() {
            *x = !*x;
        }
    }
}

#[derive(Clone)]
struct NotInto<'a, T>(&'a [T]);

impl<T: ViewType + std::ops::Not<Output = T>> SimdKernel for NotInto<'_, T> {
    type Output = Vec<T>;

    #[inline(always)]
    fn run(self) -> Vec<T> {
        self.0.iter().map(|&x| !x).collect()
    }
}

/// The integer complement (`invert` of a 32/64-bit integer): in place for a
/// sole owner, else into a new buffer.
fn run_not<T: ViewType + std::ops::Not<Output = T>>(mut buf: ViewBuffer) -> ViewBuffer {
    if let Some(data) = buf.unique_contiguous_mut::<T>() {
        dispatch_mut(&Not, data);
        return buf;
    }
    let shape = buf.shape().to_vec();
    let packed = buf.to_contiguous();
    let out = dispatch(NotInto(packed.as_slice::<T>()));
    ViewBuffer::from_vec_with_shape(out, shape)
}

/// Elements per block of the blocked strategy: an 8 KiB f32 scratch on the
/// stack, well inside L1.
const BLOCK: usize = 2048;

/// Blocked strategy, output dtype = input dtype: in place when the buffer is
/// its sole owner, into a new buffer otherwise.
fn run_blocked_same<S>(mut buf: ViewBuffer, kernel: &FusedKernel) -> ViewBuffer
where
    S: ViewType + CastFrom<f32>,
    f32: CastFrom<S>,
{
    if let Some(data) = buf.unique_contiguous_mut::<S>() {
        dispatch_mut(&BlockedInPlace { ops: &kernel.ops }, data);
        return buf;
    }
    run_blocked_into::<S, S>(&buf, kernel)
}

/// Blocked strategy into a new buffer of the kernel's output dtype.
fn run_blocked_into<S, D>(buf: &ViewBuffer, kernel: &FusedKernel) -> ViewBuffer
where
    S: ViewType,
    D: ViewType + CastFrom<f32>,
    f32: CastFrom<S>,
{
    let shape = buf.shape().to_vec();
    let packed = buf.to_contiguous();
    let out = dispatch(BlockedInto::<S, D> {
        src: packed.as_slice::<S>(),
        ops: &kernel.ops,
        _to: std::marker::PhantomData,
    });
    ViewBuffer::from_vec_with_shape(out, shape)
}

/// One block: `dst[i] = D(passes(f32(src[i])))`, the same per-element
/// conversions and passes as a whole-buffer pass.
#[inline(always)]
fn run_block<S, D>(src: &[S], ops: &[ScalarOp], mut store: impl FnMut(usize, D))
where
    S: ViewType,
    D: ViewType + CastFrom<f32>,
    f32: CastFrom<S>,
{
    let mut acc = [0.0f32; BLOCK];
    let acc = &mut acc[..src.len()];
    for (a, &x) in acc.iter_mut().zip(src) {
        *a = f32::cast_from(x);
    }
    apply_fused_op_passes(acc, ops);
    for (i, &a) in acc.iter().enumerate() {
        store(i, D::cast_from(a));
    }
}

struct BlockedInPlace<'a> {
    ops: &'a [ScalarOp],
}

impl<S> SimdKernelMut<S> for BlockedInPlace<'_>
where
    S: ViewType + CastFrom<f32>,
    f32: CastFrom<S>,
{
    #[inline(always)]
    fn run_mut(&self, data: &mut [S]) {
        for block in data.chunks_mut(BLOCK) {
            let mut acc = [0.0f32; BLOCK];
            let acc = &mut acc[..block.len()];
            for (a, &x) in acc.iter_mut().zip(block.iter()) {
                *a = f32::cast_from(x);
            }
            apply_fused_op_passes(acc, self.ops);
            for (x, &a) in block.iter_mut().zip(acc.iter()) {
                *x = S::cast_from(a);
            }
        }
    }
}

struct BlockedInto<'a, S, D> {
    src: &'a [S],
    ops: &'a [ScalarOp],
    _to: std::marker::PhantomData<D>,
}

// Derived `Clone` would demand `D: Clone` of the marker's parameter.
impl<S, D> Clone for BlockedInto<'_, S, D> {
    fn clone(&self) -> Self {
        BlockedInto {
            src: self.src,
            ops: self.ops,
            _to: std::marker::PhantomData,
        }
    }
}

impl<S, D> SimdKernel for BlockedInto<'_, S, D>
where
    S: ViewType,
    D: ViewType + CastFrom<f32>,
    f32: CastFrom<S>,
{
    type Output = Vec<D>;

    #[inline(always)]
    fn run(self) -> Vec<D> {
        let n = self.src.len();
        let mut out: Vec<D> = Vec::with_capacity(n);
        let spare = &mut out.spare_capacity_mut()[..n];
        for (dst, src) in spare.chunks_mut(BLOCK).zip(self.src.chunks(BLOCK)) {
            run_block::<S, D>(src, self.ops, |i, v| {
                dst[i].write(v);
            });
        }
        // SAFETY: the blocks cover all `n` slots, and `run_block` stores one
        // value per source element of its block.
        unsafe { out.set_len(n) };
        out
    }
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

fn run_lut_from<S: LutIndex>(buf: ViewBuffer, kernels: &[FusedKernel]) -> ViewBuffer {
    with_dtype!(kernels[0].out_dtype, D => run_lut::<S, D>(buf, kernels))
}

/// Table strategy: `tables[c * S::DOMAIN + x.index()]` is the kernel for
/// channel `c` applied to value `x`, computed by the kernel's own passes and
/// conversion.
fn run_lut<S: LutIndex, D: ViewType + CastFrom<f32>>(
    mut buf: ViewBuffer,
    kernels: &[FusedKernel],
) -> ViewBuffer {
    let domain: Vec<f32> = (0..S::DOMAIN).map(S::value_f32).collect();
    let mut table: Vec<D> = Vec::with_capacity(kernels.len() * S::DOMAIN);
    for kernel in kernels {
        let mut values = domain.clone();
        run_passes(&mut values, &kernel.ops);
        table.extend(convert_slice::<f32, D>(&values));
    }
    let channels = kernels.len();

    // Same dtype in and out, sole owner: rewrite the elements where they are.
    if let Some(same) = (&table as &dyn Any).downcast_ref::<Vec<S>>() {
        if let Some(data) = buf.unique_contiguous_mut::<S>() {
            if channels == 1 {
                for x in data.iter_mut() {
                    *x = lookup(same, x.index());
                }
            } else {
                for px in data.chunks_exact_mut(channels) {
                    for (c, x) in px.iter_mut().enumerate() {
                        *x = lookup(same, c * S::DOMAIN + x.index());
                    }
                }
            }
            return buf;
        }
    }

    let shape = buf.shape().to_vec();
    let packed = buf.to_contiguous();
    let src = packed.as_slice::<S>();
    let mut out: Vec<D> = Vec::with_capacity(src.len());
    let spare = &mut out.spare_capacity_mut()[..src.len()];
    if channels == 1 {
        for (d, &x) in spare.iter_mut().zip(src) {
            d.write(lookup(&table, x.index()));
        }
    } else {
        for (dst, px) in spare
            .chunks_exact_mut(channels)
            .zip(src.chunks_exact(channels))
        {
            for (c, (d, &x)) in dst.iter_mut().zip(px).enumerate() {
                d.write(lookup(&table, c * S::DOMAIN + x.index()));
            }
        }
    }
    // SAFETY: `src.len()` is a whole number of `channels`-element pixels
    // (asserted in `run_kernels`), so the loop wrote every slot.
    unsafe { out.set_len(src.len()) };
    ViewBuffer::from_vec_with_shape(out, shape)
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

/// Pass strategy: gather to f32, run the passes, convert to `out_dtype`.
fn run_pass(mut buf: ViewBuffer, kernels: &[FusedKernel]) -> ViewBuffer {
    let out_dtype = kernels[0].out_dtype;
    if buf.dtype() == DType::F32 && out_dtype == DType::F32 {
        if let Some(data) = buf.unique_contiguous_mut::<f32>() {
            run_kernel_passes(data, kernels);
            return buf;
        }
    }
    let shape = buf.shape().to_vec();
    let mut acc = gather_f32(&buf);
    run_kernel_passes(&mut acc, kernels);
    finish_fused_output(acc, shape, out_dtype)
}

/// Run each kernel's passes over its elements: all of them, or channel `c`
/// of a channels-last buffer for the `c`th of several kernels. A channel is
/// gathered into a scratch, run and scattered back, so the passes stay the
/// single f32 arithmetic.
fn run_kernel_passes(data: &mut [f32], kernels: &[FusedKernel]) {
    if let [kernel] = kernels {
        run_passes(data, &kernel.ops);
        return;
    }
    let channels = kernels.len();
    let mut scratch: Vec<f32> = Vec::with_capacity(data.len() / channels);
    for (c, kernel) in kernels.iter().enumerate() {
        scratch.clear();
        scratch.extend(data.iter().skip(c).step_by(channels));
        run_passes(&mut scratch, &kernel.ops);
        for (d, &v) in data.iter_mut().skip(c).step_by(channels).zip(&scratch) {
            *d = v;
        }
    }
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

/// f64 path: each element through the steps in order, in f64.
fn run_f64(mut buf: ViewBuffer, steps: &[F64Step]) -> ViewBuffer {
    let apply = |x: f64| steps.iter().fold(x, |v, step| step.apply(v));
    if let Some(data) = buf.unique_contiguous_mut::<f64>() {
        for x in data.iter_mut() {
            *x = apply(*x);
        }
        return buf;
    }
    let shape = buf.shape().to_vec();
    let packed = buf.to_contiguous();
    let out: Vec<f64> = packed.as_slice::<f64>().iter().map(|&x| apply(x)).collect();
    ViewBuffer::from_vec_with_shape(out, shape)
}

/// Every element as f32, in logical (row-major) order, by the conversion rule.
fn gather_f32(buf: &ViewBuffer) -> Vec<f32> {
    with_dtype!(buf.dtype(), S => convert_view::<S, f32>(buf))
}

/// Statistics a per-value op needs before it can lower to a kernel, read in
/// logical element order.
pub(crate) mod stats {
    use crate::core::buffer::ViewBuffer;
    use crate::core::dtype::DType;

    /// The elements as f32, contiguous (the value the kernel reads).
    fn as_f32(buf: &ViewBuffer) -> Vec<f32> {
        super::gather_f32(buf)
    }

    /// Minimum and maximum of the elements as f32 read them. Integers take
    /// their integer extremes (`as f32` is monotone, so this equals a fold
    /// over the converted values); floats fold in element order with
    /// `f32::min`/`max`, which skip NaN. `(inf, -inf)` for no elements.
    pub(crate) fn min_max_f32(buf: &ViewBuffer) -> (f32, f32) {
        macro_rules! int_extremes {
            ($t:ty) => {{
                let packed = buf.to_contiguous();
                let src = packed.as_slice::<$t>();
                match (src.iter().min(), src.iter().max()) {
                    (Some(&lo), Some(&hi)) => (lo as f32, hi as f32),
                    _ => (f32::INFINITY, f32::NEG_INFINITY),
                }
            }};
        }
        match buf.dtype() {
            DType::U8 => int_extremes!(u8),
            DType::I8 => int_extremes!(i8),
            DType::U16 => int_extremes!(u16),
            DType::I16 => int_extremes!(i16),
            DType::U32 => int_extremes!(u32),
            DType::I32 => int_extremes!(i32),
            DType::U64 => int_extremes!(u64),
            DType::I64 => int_extremes!(i64),
            DType::F32 | DType::F64 => {
                let values = as_f32(buf);
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (min, max)
            }
        }
    }

    /// The exact sum of an 8/16-bit buffer.
    fn small_int_sum(buf: &ViewBuffer) -> Option<i64> {
        macro_rules! sum {
            ($t:ty) => {{
                let packed = buf.to_contiguous();
                Some(packed.as_slice::<$t>().iter().map(|&x| i64::from(x)).sum())
            }};
        }
        match buf.dtype() {
            DType::U8 => sum!(u8),
            DType::I8 => sum!(i8),
            DType::U16 => sum!(u16),
            DType::I16 => sum!(i16),
            _ => None,
        }
    }

    /// Exact sums of an 8/16-bit buffer: `(n, Σx, Σx²)`.
    fn small_int_sums(buf: &ViewBuffer) -> Option<(usize, i128, i128)> {
        macro_rules! sums {
            ($t:ty) => {{
                let packed = buf.to_contiguous();
                let src = packed.as_slice::<$t>();
                let sum: i64 = src.iter().map(|&x| i64::from(x)).sum();
                let sum_sq: u128 = src
                    .iter()
                    .map(|&x| {
                        let x = i64::from(x);
                        (x * x) as u64
                    })
                    .fold(0u128, |acc, sq| acc + u128::from(sq));
                Some((src.len(), i128::from(sum), sum_sq as i128))
            }};
        }
        match buf.dtype() {
            DType::U8 => sums!(u8),
            DType::I8 => sums!(i8),
            DType::U16 => sums!(u16),
            DType::I16 => sums!(i16),
            _ => None,
        }
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
            None => as_f32(buf).iter().map(|&x| x as f64).sum::<f64>(),
        };
        sum as f32 / n as f32
    }

    /// The f64 mean of an f64 buffer, summed in element order.
    pub(crate) fn mean_f64(buf: &ViewBuffer) -> f64 {
        let packed = buf.to_contiguous();
        let src = packed.as_slice::<f64>();
        if src.is_empty() {
            0.0
        } else {
            src.iter().sum::<f64>() / src.len() as f64
        }
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
        let values = as_f32(buf);
        let n = values.len() as f64;
        let mean = values.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = values
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n;
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

/// Materialize the kernel's f32 result as a contiguous buffer of `out_dtype`.
///
/// Other targets convert through [`convert_slice`], the rule
/// [`ViewBuffer::cast_to`] uses, so a fused trailing cast and a standalone
/// cast cannot round differently; `F32` reuses the accumulator allocation
/// without copying.
pub(crate) fn finish_fused_output(
    acc: Vec<f32>,
    shape: Vec<usize>,
    out_dtype: DType,
) -> ViewBuffer {
    if out_dtype == DType::F32 {
        // Reuse the accumulator allocation: AlignedBytes takes it over and
        // deallocates with f32 alignment.
        return ViewBuffer {
            data: BufferStorage::Rust(Arc::new(AlignedBytes::from_typed_vec(acc))),
            layout: Layout::new_contiguous(shape, DType::F32),
        };
    }
    with_dtype!(out_dtype, T => {
        ViewBuffer::from_vec_with_shape(convert_slice::<f32, T>(&acc), shape)
    })
}
