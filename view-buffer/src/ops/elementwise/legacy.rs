//! **Test oracle: the element-wise ops as they were before the engine.**
//!
//! Every function here is today's implementation, moved verbatim (only
//! `self` became `this` where a method became a free function, and the
//! kernel's in-place/allocating pair calls the fused arithmetic by path). The
//! engine is checked against it bit for bit, so "bit-identical" is measured
//! against the code that shipped rather than restated from memory.
#![allow(dead_code, clippy::all)]

use crate::core::buffer::{BufferStorage, ViewBuffer};
use crate::core::dtype::DType;
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::ops::ComputeOp;
use std::sync::Arc;

#[cfg(feature = "ndarray_interop")]
use crate::interop::ndarray::{AsNdarray, FromNdarray};

/// Today's `apply_compute_inner` for the per-value ops.
pub(super) fn apply(buf: ViewBuffer, op: ComputeOp) -> ViewBuffer {
    if let Some(op) = op.scalar() {
        return apply_scalar_op(buf, op);
    }
    match op {
        ComputeOp::Scale { factor } => apply_scalar_owned_with(
            buf,
            move |x: f32| x * factor,
            move |x: f64| x * factor as f64,
        ),
        ComputeOp::Relu => apply_scalar_owned_with(
            buf,
            |x: f32| if x > 0.0 { x } else { 0.0 },
            |x: f64| if x > 0.0 { x } else { 0.0 },
        ),
        ComputeOp::Fused(ref kernel) => {
            let mut buf = buf;
            if try_apply_fused_kernel_inplace(&mut buf, kernel) {
                buf
            } else {
                apply_fused_kernel(&buf, kernel)
            }
        }
        ComputeOp::Normalize {
            method,
            ref mean,
            ref std,
            out_dtype,
        } => apply_normalize(
            &buf,
            &ComputeOp::normalization(method, mean, std),
            out_dtype.unwrap_or(DType::F32),
        ),
        ComputeOp::Clamp { min, max } => apply_scalar_owned_with(
            buf,
            move |x: f32| x.clamp(min, max),
            move |x: f64| x.clamp(min as f64, max as f64),
        ),
        ComputeOp::AdjustContrast { factor } => apply_adjust_contrast(&buf, factor),
        ComputeOp::AdjustGamma { gamma } => apply_adjust_gamma(&buf, gamma),
        ComputeOp::Invert => apply_invert(&buf),
        other => panic!("not a per-value op: {other:?}"),
    }
}

/// A lone scalar op, routed through the fused kernel so it and a fused one
/// share the identical f32 arithmetic (the "route through the kernel"
/// design). f64 preserves precision on its own cold path (`PromoteToFloat`
/// keeps f64; the kernel is f32-only), mirroring how the promote-family ops
/// keep f64 unfused.
fn apply_scalar_op(buf: ViewBuffer, op: ScalarOp) -> ViewBuffer {
    if buf.dtype() == DType::F64 {
        return apply_scalar_op_f64(&buf, &op);
    }
    let kernel = FusedKernel {
        ops: vec![op],
        out_dtype: DType::F32,
    };
    let mut buf = buf;
    if try_apply_fused_kernel_inplace(&mut buf, &kernel) {
        buf
    } else {
        apply_fused_kernel(&buf, &kernel)
    }
}

/// Apply a single scalar op to an `f64` buffer, computing in `f64`.
///
/// The unfused f64 cold path for [`ComputeOp::Scalar`]: `PromoteToFloat`
/// preserves f64, but the bulk kernel computes in f32, so f64 evaluates here
/// through [`ScalarOp::apply_f64`] — the shared f64 arithmetic authority.
fn apply_scalar_op_f64(buf: &ViewBuffer, op: &ScalarOp) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let src = contig.as_slice::<f64>();
    let new_data: Vec<f64> = src.iter().map(|&x| op.apply_f64(x)).collect();
    ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec())
}

/// Apply normalization and cast the f32 result to the configured output dtype.
///
/// Normalization always computes in f32 (see [`apply_normalize_f32`]); the
/// `out_dtype` is the target the planner resolved from the op's `Fixed`
/// rule (defaulting to f32). Casting here is what keeps the produced dtype equal to the planned
/// dtype — without it the planner could declare, say, `u8` while execution
/// emitted `f32` (a plan/execution contract violation guarded by the
/// dtype-contract tests).
fn apply_normalize(
    buf: &ViewBuffer,
    method: &crate::ops::Normalization,
    out_dtype: DType,
) -> ViewBuffer {
    let normalized = apply_normalize_f32(buf, method);
    if out_dtype == DType::F32 {
        normalized
    } else {
        normalized.cast(out_dtype)
    }
}

/// Apply normalization to a buffer, accepting any numeric input type.
///
/// This function automatically casts the input to f32 for computation,
/// as per the dtype promotion contract. The output is always f32.
///
/// ## Edge Case Behavior
/// - **Constant array (min == max)**: Returns 0.0 for all elements (MinMax) or 0.0 (ZScore)
/// - **NaN values**: Propagated according to IEEE 754 semantics
/// - **Inf values**: Handled naturally by min/max/mean calculations
fn apply_normalize_f32(buf: &ViewBuffer, method: &crate::ops::Normalization) -> ViewBuffer {
    use crate::ops::Normalization;

    // Cast to f32 working dtype if needed (dtype promotion)
    let work_buf = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };

    let shape = work_buf.shape().to_vec();

    // Try ndarray path first (handles negative strides via invert_axis in ndarray 0.17+)
    #[cfg(feature = "ndarray_interop")]
    {
        if let Ok(view) = work_buf.as_array_view::<f32>() {
            match method {
                Normalization::MinMax => {
                    let min = view.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = view.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let range = max - min;
                    if range == 0.0 {
                        let result: ndarray::ArrayD<f32> = ndarray::Array::zeros(view.raw_dim());
                        return ViewBuffer::from_array(result);
                    }
                    let result = view.mapv(|x| (x - min) / range);
                    return ViewBuffer::from_array(result.into_owned());
                }
                Normalization::ZScore => {
                    let n = view.len() as f32;
                    let mean = view.iter().sum::<f32>() / n;
                    let variance = view.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
                    let std_val = variance.sqrt();
                    if std_val == 0.0 {
                        let result: ndarray::ArrayD<f32> = ndarray::Array::zeros(view.raw_dim());
                        return ViewBuffer::from_array(result);
                    }
                    let result = view.mapv(|x| (x - mean) / std_val);
                    return ViewBuffer::from_array(result.into_owned());
                }
                Normalization::Preset { mean, std } => {
                    // Channel-wise normalization - need to iterate with channel awareness
                    let channels = if shape.len() == 3 { shape[2] } else { 1 };
                    assert_eq!(
                        mean.len(),
                        channels,
                        "Mean length {} must match channel count {}",
                        mean.len(),
                        channels
                    );
                    assert_eq!(
                        std.len(),
                        channels,
                        "Std length {} must match channel count {}",
                        std.len(),
                        channels
                    );

                    // Collect all values with channel-wise normalization
                    let new_data: Vec<f32> = view
                        .iter()
                        .enumerate()
                        .map(|(i, &x)| {
                            let c = i % channels;
                            (x - mean[c]) / std[c]
                        })
                        .collect();
                    return ViewBuffer::from_vec(new_data).reshape(shape);
                }
            }
        }
    }

    // Fallback: use contiguous buffer
    let contig = work_buf.to_contiguous();
    let count = contig.layout.num_elements();
    let src = contig.as_slice::<f32>();

    let new_data: Vec<f32> = match method {
        Normalization::MinMax => {
            let min = src.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            if range == 0.0 {
                vec![0.0; count]
            } else {
                src.iter().map(|&x| (x - min) / range).collect()
            }
        }
        Normalization::ZScore => {
            let n = count as f32;
            let mean = src.iter().sum::<f32>() / n;
            let variance = src.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
            let std_val = variance.sqrt();
            if std_val == 0.0 {
                vec![0.0; count]
            } else {
                src.iter().map(|&x| (x - mean) / std_val).collect()
            }
        }
        Normalization::Preset { mean, std } => {
            let channels = if shape.len() == 3 { shape[2] } else { 1 };
            assert_eq!(
                mean.len(),
                channels,
                "Mean length {} must match channel count {}",
                mean.len(),
                channels
            );
            assert_eq!(
                std.len(),
                channels,
                "Std length {} must match channel count {}",
                std.len(),
                channels
            );
            src.iter()
                .enumerate()
                .map(|(i, &x)| {
                    let c = i % channels;
                    (x - mean[c]) / std[c]
                })
                .collect()
        }
    };

    ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec())
}

/// Adjust contrast: `(pixel - mean) * factor + mean`.
///
/// Computes the global mean, then scales each pixel's deviation from it.
/// Input is cast to f32; output is f32.
fn apply_adjust_contrast(buf: &ViewBuffer, factor: f32) -> ViewBuffer {
    // f64 input computes (and stays) in f64, per the PromoteToFloat contract.
    if buf.dtype() == DType::F64 {
        let contig = buf.to_contiguous();
        let count = contig.layout.num_elements();
        let src = contig.as_slice::<f64>();
        let mean: f64 = if count > 0 {
            src.iter().sum::<f64>() / count as f64
        } else {
            0.0
        };
        let factor = factor as f64;
        let new_data: Vec<f64> = src.iter().map(|&x| (x - mean) * factor + mean).collect();
        return ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec());
    }
    let work_buf = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = work_buf.to_contiguous();
    let count = contig.layout.num_elements();
    let src = contig.as_slice::<f32>();

    let mean: f32 = if count > 0 {
        src.iter().map(|&x| x as f64).sum::<f64>() as f32 / count as f32
    } else {
        0.0
    };
    let new_data: Vec<f32> = src.iter().map(|&x| (x - mean) * factor + mean).collect();
    ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec())
}

/// Adjust gamma (power-law): normalize to [0,1], apply `pixel^gamma`, denormalize.
///
/// For u8 input the [0,255] range is used; for float [0,1] is assumed.
/// Integer inputs promote to f32; f64 input computes (and stays) in f64,
/// per the PromoteToFloat contract.
fn apply_adjust_gamma(buf: &ViewBuffer, gamma: f32) -> ViewBuffer {
    let input_dtype = buf.dtype();
    if input_dtype == DType::F64 {
        let contig = buf.to_contiguous();
        let src = contig.as_slice::<f64>();
        let gamma = gamma as f64;
        let new_data: Vec<f64> = src.iter().map(|&x| x.clamp(0.0, 1.0).powf(gamma)).collect();
        return ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec());
    }
    let work_buf = if input_dtype != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = work_buf.to_contiguous();
    let src = contig.as_slice::<f32>();

    // Normalize by the input dtype's value range (255 for u8, 65535 for
    // u16, ...; 1.0 for floats). A hardcoded 255 would clamp every u16+
    // pixel above 255 to the ceiling and flatten the image to a constant.
    let max_val: f32 = input_dtype.norm_range_max_f32();

    let new_data: Vec<f32> = src
        .iter()
        .map(|&x| {
            let normalized = (x / max_val).clamp(0.0, 1.0);
            normalized.powf(gamma) * max_val
        })
        .collect();
    ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec())
}

/// Invert pixel values: `max_val - pixel`.
///
/// For u8: `255 - pixel`. For float: `1.0 - pixel`. Preserves input dtype.
fn apply_invert(buf: &ViewBuffer) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let shape = contig.shape().to_vec();

    match buf.dtype() {
        DType::U8 => {
            let src = contig.as_slice::<u8>();
            let new_data: Vec<u8> = src.iter().map(|&x| 255u8 - x).collect();
            ViewBuffer::from_vec_with_shape(new_data, shape)
        }
        DType::U16 => {
            let src = contig.as_slice::<u16>();
            let new_data: Vec<u16> = src.iter().map(|&x| 65535u16 - x).collect();
            ViewBuffer::from_vec_with_shape(new_data, shape)
        }
        DType::F32 => {
            let src = contig.as_slice::<f32>();
            let new_data: Vec<f32> = src.iter().map(|&x| 1.0f32 - x).collect();
            ViewBuffer::from_vec_with_shape(new_data, shape)
        }
        DType::F64 => {
            let src = contig.as_slice::<f64>();
            let new_data: Vec<f64> = src.iter().map(|&x| 1.0f64 - x).collect();
            ViewBuffer::from_vec_with_shape(new_data, shape)
        }
        _ => {
            // For other dtypes, cast to f32, invert as 1.0 - x, return f32
            let f32_buf = buf.cast(DType::F32);
            apply_invert(&f32_buf)
        }
    }
}

/// Apply a scalar operation element-wise, honoring the float-promotion
/// dtype contract (`OutputDTypeRule::PromoteToFloat`):
///
/// - f64 input computes in f64 and returns f64 (the rule preserves f64);
/// - every other numeric input is promoted to f32, computed in f32, and
///   returned as f32.
///
/// This follows the pattern used by NumPy, PyTorch, and other numeric
/// libraries. Both closures must implement the same operation at their
/// respective precision.
fn apply_scalar_op_with<F32Op, F64Op>(buf: &ViewBuffer, op32: F32Op, op64: F64Op) -> ViewBuffer
where
    F32Op: Fn(f32) -> f32,
    F64Op: Fn(f64) -> f64,
{
    if buf.dtype() == DType::F64 {
        // Try to use ndarray if available for efficient strided iteration
        #[cfg(feature = "ndarray_interop")]
        {
            if let Ok(view) = buf.as_array_view::<f64>() {
                let result_array = view.mapv(&op64);
                return ViewBuffer::from_array(result_array);
            }
        }
        let contig = buf.to_contiguous();
        let src = contig.as_slice::<f64>();
        let new_data: Vec<f64> = src.iter().map(|&x| op64(x)).collect();
        return ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec());
    }

    // Cast to f32 working dtype if needed (dtype promotion)
    let work_buf = if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };

    // Try to use ndarray if available for efficient strided iteration
    // (ndarray 0.17+ handles negative strides via invert_axis)
    #[cfg(feature = "ndarray_interop")]
    {
        if let Ok(view) = work_buf.as_array_view::<f32>() {
            let result_array = view.mapv(&op32);
            return ViewBuffer::from_array(result_array);
        }
    }

    // Fallback: use contiguous buffer
    let contig = work_buf.to_contiguous();
    let src = contig.as_slice::<f32>();
    let new_data: Vec<f32> = src.iter().map(|&x| op32(x)).collect();
    ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec())
}

/// Scalar op applied to an owned buffer.
///
/// When the buffer is already a contiguous float (f32 or f64) with a sole
/// strong reference (refcount == 1), the data is mutated in-place — no heap
/// allocation. Falls back to [`apply_scalar_op_with`] for all other cases
/// (dtype promotion, non-contiguous layout, or shared Arc).
fn apply_scalar_owned_with<F32Op, F64Op>(
    mut buf: ViewBuffer,
    op32: F32Op,
    op64: F64Op,
) -> ViewBuffer
where
    F32Op: Fn(f32) -> f32 + Copy,
    F64Op: Fn(f64) -> f64 + Copy,
{
    use crate::core::buffer::BufferStorage;
    use std::sync::Arc;

    let dtype = buf.dtype();
    if matches!(dtype, DType::F32 | DType::F64) && buf.layout.is_contiguous() {
        if let BufferStorage::Rust(ref mut arc) = buf.data {
            if let Some(vec) = Arc::get_mut(arc) {
                let count = buf.layout.num_elements();
                let base = unsafe { vec.as_mut_ptr().add(buf.layout.offset) };
                match dtype {
                    DType::F32 => {
                        let data =
                            unsafe { std::slice::from_raw_parts_mut(base as *mut f32, count) };
                        for x in data.iter_mut() {
                            *x = op32(*x);
                        }
                    }
                    _ => {
                        let data =
                            unsafe { std::slice::from_raw_parts_mut(base as *mut f64, count) };
                        for x in data.iter_mut() {
                            *x = op64(*x);
                        }
                    }
                }
                return buf;
            }
        }
    }
    apply_scalar_op_with(&buf, op32, op64)
}

/// Applies a fused kernel of scalar operations in-place, without allocation.
///
/// Succeeds only when no dtype conversion is involved on either end
/// (F32 buffer, F32 kernel output), the buffer is contiguous, and this
/// `Arc` has exactly one strong reference (i.e. the caller holds exclusive
/// ownership). In that case the inner `Vec<u8>` is mutated directly —
/// zero heap allocation.
///
/// Returns `true` if the in-place path was taken, `false` if the caller should
/// fall back to the allocating [`apply_fused_kernel`] path.
pub(super) fn try_apply_fused_kernel_inplace(this: &mut ViewBuffer, kernel: &FusedKernel) -> bool {
    if this.dtype() != DType::F32 || kernel.out_dtype != DType::F32 || !this.layout.is_contiguous()
    {
        return false;
    }
    let BufferStorage::Rust(ref mut arc) = this.data else {
        return false;
    };
    let Some(vec) = Arc::get_mut(arc) else {
        return false;
    };

    let total_elems: usize = this.layout.shape.iter().product();
    let data = unsafe {
        std::slice::from_raw_parts_mut(
            vec.as_mut_ptr().add(this.layout.offset) as *mut f32,
            total_elems,
        )
    };

    // One full-array pass per op — the inner loop is a simple scalar
    // operation that LLVM can auto-vectorize with SIMD (the closure is
    // known at compile time within each match arm, unlike the old
    // chunk-then-ops order which blocked auto-vectorization).
    crate::ops::elementwise::apply_fused_op_passes(data, &kernel.ops);
    true
}

/// Applies a fused kernel of scalar operations element-wise.
///
/// Accepts any numeric input dtype: the input is converted to `f32`
/// during the gather (equivalent to a fused leading `Cast`), the ops run
/// as SIMD-friendly full-array `f32` passes, and the result is converted
/// to `kernel.out_dtype` while writing the output buffer (equivalent to a
/// fused trailing `Cast`, matching [`ViewBuffer::cast_to`] semantics).
/// Compared to bracketing the kernel with separate casts, this removes
/// the intermediate materializations.
pub(super) fn apply_fused_kernel(this: &ViewBuffer, kernel: &FusedKernel) -> ViewBuffer {
    let total_elems: usize = this.layout.shape.iter().product();

    // Gather to f32 (handles dtype conversion and striding in one pass).
    let mut acc: Vec<f32> = gather_to_f32(this, total_elems);

    // One full-array pass per op (auto-vectorized; see inplace docs).
    crate::ops::elementwise::apply_fused_op_passes(&mut acc, &kernel.ops);

    // Convert to the kernel's output dtype while writing the result.
    finish_fused_output(acc, this.layout.shape.clone(), kernel.out_dtype)
}

/// The engine's former `finish_fused_output`, which the pre-engine code
/// called and the engine no longer has (its blocked strategy stores as it
/// goes): moved here verbatim so this oracle stays the code it was.
fn finish_fused_output(
    acc: Vec<f32>,
    shape: impl Into<crate::core::layout::Dims>,
    out_dtype: DType,
) -> ViewBuffer {
    let shape = shape.into();
    if out_dtype == DType::F32 {
        // Reuse the accumulator allocation: AlignedBytes takes it over and
        // deallocates with f32 alignment.
        return ViewBuffer {
            data: BufferStorage::Rust(Arc::new(crate::core::bytes::AlignedBytes::from_typed_vec(
                acc,
            ))),
            layout: crate::core::layout::Layout::new_contiguous(shape, DType::F32),
        };
    }
    crate::core::dtype::with_dtype!(out_dtype, T => {
        ViewBuffer::from_vec_with_shape(crate::core::convert::convert_slice::<f32, T>(&acc), shape)
    })
}

/// Read every element as `f32`, in logical (row-major) order.
///
/// Contiguous buffers convert with a monomorphic per-dtype loop; strided
/// buffers gather through the stride walk (also monomorphic per dtype).
fn gather_to_f32(this: &ViewBuffer, total_elems: usize) -> Vec<f32> {
    if this.layout.is_contiguous() {
        macro_rules! convert_contig {
            ($t:ty) => {{
                let src_ptr = unsafe { this.data.as_ptr().add(this.layout.offset) as *const $t };
                let src = unsafe { std::slice::from_raw_parts(src_ptr, total_elems) };
                src.iter().map(|&x| x as f32).collect()
            }};
        }
        return match this.dtype() {
            DType::F32 => {
                let src_ptr = unsafe { this.data.as_ptr().add(this.layout.offset) as *const f32 };
                let src = unsafe { std::slice::from_raw_parts(src_ptr, total_elems) };
                src.to_vec()
            }
            DType::U8 => convert_contig!(u8),
            DType::I8 => convert_contig!(i8),
            DType::U16 => convert_contig!(u16),
            DType::I16 => convert_contig!(i16),
            DType::U32 => convert_contig!(u32),
            DType::I32 => convert_contig!(i32),
            DType::U64 => convert_contig!(u64),
            DType::I64 => convert_contig!(i64),
            DType::F64 => convert_contig!(f64),
        };
    }

    // Strided gather: walk logical indices, reading each element at its
    // byte offset. The walk is monomorphized per dtype so the inner read
    // has no per-element dispatch.
    macro_rules! gather_strided {
        ($t:ty) => {{
            let mut out: Vec<f32> = Vec::with_capacity(total_elems);
            let mut indices = vec![0; this.layout.shape.len()];
            let shape = &this.layout.shape;
            let strides = &this.layout.strides;
            let ptr = this.data.as_ptr();
            let base_offset = this.layout.offset;
            let data_len = this.data.len();
            for _ in 0..total_elems {
                let mut offset = base_offset as isize;
                for (dim, &idx) in indices.iter().enumerate() {
                    offset += (idx as isize) * strides[dim];
                }
                debug_assert!(
                    offset >= 0 && (offset as usize) + std::mem::size_of::<$t>() <= data_len,
                    "Fused kernel read OOB"
                );
                let value = unsafe { *(ptr.offset(offset) as *const $t) };
                out.push(value as f32);
                for dim in (0..shape.len()).rev() {
                    indices[dim] += 1;
                    if indices[dim] < shape[dim] {
                        break;
                    }
                    indices[dim] = 0;
                }
            }
            out
        }};
    }
    match this.dtype() {
        DType::U8 => gather_strided!(u8),
        DType::I8 => gather_strided!(i8),
        DType::U16 => gather_strided!(u16),
        DType::I16 => gather_strided!(i16),
        DType::U32 => gather_strided!(u32),
        DType::I32 => gather_strided!(i32),
        DType::U64 => gather_strided!(u64),
        DType::I64 => gather_strided!(i64),
        DType::F32 => gather_strided!(f32),
        DType::F64 => gather_strided!(f64),
    }
}
