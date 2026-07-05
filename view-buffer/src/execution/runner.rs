//! Pure operation runners — apply one op to one full buffer.
//!
//! These functions have no tiling or strategy logic; that lives in
//! [`execution::tiling`] and [`execution::plan`].  Everything here is
//! `pub(crate)` so both the full-image path and the tiling path can call the
//! same implementations.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::expr::ViewExpr;
use crate::ops::affine::AffineParams;
use crate::ops::dto::ViewDto;
use crate::ops::traits::Op;
use crate::ops::{ComputeOp, ImageOp, ViewOp};

#[cfg(feature = "image_interop")]
use crate::ops::{FilterType, ImageOpKind};

#[cfg(feature = "image_interop")]
use crate::interop::image::AsImageView;

#[cfg(feature = "ndarray_interop")]
use crate::interop::ndarray::{AsNdarray, FromNdarray};

#[cfg(feature = "image_interop")]
use image::Luma;

#[cfg(feature = "image_interop")]
use fast_image_resize as fir;

/// High-level entry point to execute a plan described by a sequence of ViewDto operations.
/// This acts as the bridge between the serialized plan (e.g. from Python) and the
/// execution engine.
pub fn execute_plan(source: ViewBuffer, ops: Vec<ViewDto>) -> ViewBuffer {
    let mut expr = ViewExpr::new_source(source);
    for op in ops {
        expr = expr.apply_op(op);
    }
    // plan() performs optimization (fusion, etc) before execution
    expr.plan().execute()
}

/// Applies a view operation to a buffer.
pub fn apply_view(buf: ViewBuffer, op: ViewOp) -> ViewBuffer {
    match op {
        ViewOp::Transpose(perm) => buf.permute(&perm),
        ViewOp::Reshape(shape) => {
            if !buf.layout.is_contiguous() {
                panic!("Reshape on non-contiguous view not supported without copy");
            }
            buf.reshape(shape)
        }
        ViewOp::Flip(axes) => buf.flip(&axes),
        ViewOp::Crop { start, end } => buf.slice(&start, &end),
        ViewOp::Rotate90 => {
            // Rotate90: transpose [1,0] then flip axis 1 (width)
            // For HWC layout: transpose swaps H and W, then flip W
            let shape = buf.shape();
            if shape.len() < 2 {
                return buf; // Can't rotate 1D or 0D
            }
            let perm = if shape.len() == 2 {
                vec![1, 0] // [H, W] -> [W, H]
            } else {
                vec![1, 0, 2] // [H, W, C] -> [W, H, C]
            };
            let transposed = buf.permute(&perm);
            transposed.flip(&[1]) // Flip width axis
        }
        ViewOp::Rotate180 => {
            // Rotate180: flip both height (axis 0) and width (axis 1)
            buf.flip(&[0, 1])
        }
        ViewOp::Rotate270 => {
            // Rotate270: transpose [1,0] then flip axis 0 (height)
            // For HWC layout: transpose swaps H and W, then flip H
            let shape = buf.shape();
            if shape.len() < 2 {
                return buf; // Can't rotate 1D or 0D
            }
            let perm = if shape.len() == 2 {
                vec![1, 0] // [H, W] -> [W, H]
            } else {
                vec![1, 0, 2] // [H, W, C] -> [W, H, C]
            };
            let transposed = buf.permute(&perm);
            transposed.flip(&[0]) // Flip height axis
        }
        ViewOp::ChannelSelect { index } => {
            let shape = buf.shape();
            if shape.len() != 3 {
                return buf;
            }
            let h = shape[0];
            let w = shape[1];
            // Slice to [H, W, 1] then materialize (slice is non-contiguous in HWC)
            let sliced = buf.slice(&[0, 0, index], &[h, w, index + 1]);
            sliced.to_contiguous().reshape(vec![h, w])
        }
    }
}

/// Applies a compute operation to a buffer.
#[inline]
pub(crate) fn apply_compute_inner(buf: ViewBuffer, op: ComputeOp) -> ViewBuffer {
    match op {
        ComputeOp::Cast(dtype) => buf.cast(dtype),
        ComputeOp::Affine(params) => apply_affine_warp(buf, params),
        ComputeOp::RotateAffine {
            angle_deg,
            expand,
            interpolation,
            border_value,
        } => {
            let h = buf.shape()[0] as u32;
            let w = buf.shape()[1] as u32;
            let params =
                AffineParams::from_rotation(angle_deg, h, w, expand, interpolation, border_value);
            apply_affine_warp(buf, params)
        }
        ComputeOp::Scale(factor) => apply_scalar_owned_with(
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
            // The kernel reads any numeric dtype (converting to f32 during
            // the gather) and converts to its out_dtype while writing — no
            // separate cast materializations. The in-place path applies when
            // no conversion is involved on either end (f32 → f32).
            let mut buf = buf;
            if buf.try_apply_fused_kernel_inplace(kernel) {
                buf
            } else {
                buf.apply_fused_kernel(kernel)
            }
        }
        ComputeOp::Normalize(ref method) => apply_normalize(&buf, method),
        ComputeOp::Clamp { min, max } => apply_scalar_owned_with(
            buf,
            move |x: f32| x.clamp(min, max),
            move |x: f64| x.clamp(min as f64, max as f64),
        ),
        ComputeOp::AdjustContrast(factor) => apply_adjust_contrast(&buf, factor),
        ComputeOp::AdjustGamma(gamma) => apply_adjust_gamma(&buf, gamma),
        ComputeOp::Invert => apply_invert(&buf),
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
fn apply_normalize(buf: &ViewBuffer, method: &crate::ops::NormalizeMethod) -> ViewBuffer {
    use crate::ops::NormalizeMethod;

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
                NormalizeMethod::MinMax => {
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
                NormalizeMethod::ZScore => {
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
                NormalizeMethod::Preset { mean, std } => {
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
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };

    let new_data: Vec<f32> = match method {
        NormalizeMethod::MinMax => {
            let min = src.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            if range == 0.0 {
                vec![0.0; count]
            } else {
                src.iter().map(|&x| (x - min) / range).collect()
            }
        }
        NormalizeMethod::ZScore => {
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
        NormalizeMethod::Preset { mean, std } => {
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
        let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f64>(), count) };
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
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };

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
        let count = contig.layout.num_elements();
        let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f64>(), count) };
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
    let count = contig.layout.num_elements();
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };

    let is_integer = matches!(
        input_dtype,
        DType::U8
            | DType::U16
            | DType::I8
            | DType::I16
            | DType::U32
            | DType::I32
            | DType::U64
            | DType::I64
    );
    let max_val: f32 = if is_integer { 255.0 } else { 1.0 };

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
    let count = contig.layout.num_elements();
    let shape = contig.shape().to_vec();

    match buf.dtype() {
        DType::U8 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), count) };
            let new_data: Vec<u8> = src.iter().map(|&x| 255u8 - x).collect();
            ViewBuffer::from_vec_with_shape(new_data, shape)
        }
        DType::U16 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u16>(), count) };
            let new_data: Vec<u16> = src.iter().map(|&x| 65535u16 - x).collect();
            ViewBuffer::from_vec_with_shape(new_data, shape)
        }
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };
            let new_data: Vec<f32> = src.iter().map(|&x| 1.0f32 - x).collect();
            ViewBuffer::from_vec_with_shape(new_data, shape)
        }
        DType::F64 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f64>(), count) };
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

/// Execute a channel swap operation: reorder channels in a [H, W, C] buffer.
pub fn apply_channel_swap(buf: &ViewBuffer, order: &[usize]) -> ViewBuffer {
    let shape = buf.shape();
    assert!(shape.len() == 3, "ChannelSwap requires 3D [H, W, C] input");
    let h = shape[0];
    let w = shape[1];
    let c = shape[2];
    assert!(
        order.len() == c,
        "ChannelSwap order length {} must match channel count {}",
        order.len(),
        c
    );

    let contig = buf.to_contiguous();
    match buf.dtype() {
        DType::U8 => {
            let src = contig.as_slice::<u8>();
            let mut output = vec![0u8; h * w * c];
            for y in 0..h {
                for x in 0..w {
                    let base_src = (y * w + x) * c;
                    let base_dst = (y * w + x) * c;
                    for (dst_c, &src_c) in order.iter().enumerate() {
                        output[base_dst + dst_c] = src[base_src + src_c];
                    }
                }
            }
            ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
        }
        DType::F32 => {
            let src = contig.as_slice::<f32>();
            let mut output = vec![0.0f32; h * w * c];
            for y in 0..h {
                for x in 0..w {
                    let base_src = (y * w + x) * c;
                    let base_dst = (y * w + x) * c;
                    for (dst_c, &src_c) in order.iter().enumerate() {
                        output[base_dst + dst_c] = src[base_src + src_c];
                    }
                }
            }
            ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
        }
        _ => {
            let f32_buf = buf.cast(DType::F32);
            apply_channel_swap(&f32_buf, order)
        }
    }
}

/// Merge multiple single-channel [H, W] buffers into a [H, W, C] buffer.
pub fn apply_channel_merge(buffers: &[&ViewBuffer]) -> ViewBuffer {
    assert!(
        !buffers.is_empty(),
        "ChannelMerge requires at least one input"
    );
    let h = buffers[0].shape()[0];
    let w = buffers[0].shape()[1];
    let c = buffers.len();

    // All buffers must be 2D [H, W] with matching dimensions
    for (i, buf) in buffers.iter().enumerate() {
        let s = buf.shape();
        assert!(
            s.len() == 2 && s[0] == h && s[1] == w,
            "ChannelMerge input {} has shape {:?}, expected [{}, {}]",
            i,
            s,
            h,
            w
        );
    }

    match buffers[0].dtype() {
        DType::U8 => {
            let mut output = vec![0u8; h * w * c];
            let contigs: Vec<_> = buffers.iter().map(|b| b.to_contiguous()).collect();
            let slices: Vec<&[u8]> = contigs.iter().map(|b| b.as_slice::<u8>()).collect();
            for y in 0..h {
                for x in 0..w {
                    let pixel_idx = y * w + x;
                    let base_dst = pixel_idx * c;
                    for (ch, slice) in slices.iter().enumerate() {
                        output[base_dst + ch] = slice[pixel_idx];
                    }
                }
            }
            ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
        }
        DType::F32 => {
            let mut output = vec![0.0f32; h * w * c];
            let contigs: Vec<_> = buffers.iter().map(|b| b.to_contiguous()).collect();
            let slices: Vec<&[f32]> = contigs.iter().map(|b| b.as_slice::<f32>()).collect();
            for y in 0..h {
                for x in 0..w {
                    let pixel_idx = y * w + x;
                    let base_dst = pixel_idx * c;
                    for (ch, slice) in slices.iter().enumerate() {
                        output[base_dst + ch] = slice[pixel_idx];
                    }
                }
            }
            ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
        }
        _ => {
            let f32_bufs: Vec<_> = buffers.iter().map(|b| b.cast(DType::F32)).collect();
            let refs: Vec<&ViewBuffer> = f32_bufs.iter().collect();
            apply_channel_merge(&refs)
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
        let count = contig.layout.num_elements();
        let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f64>(), count) };
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
    let count = contig.layout.num_elements();
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };
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

/// Convert a buffer to U8 for image operations.
///
/// This handles dtype promotion for image operations:
/// - F32/F64 in [0.0, 1.0] range: scale to [0, 255]
/// - F32/F64 outside range: clamp then scale
/// - Other integer types: cast directly
/// - U8: pass through
#[cfg(feature = "image_interop")]
fn convert_to_u8_for_image(buf: ViewBuffer) -> ViewBuffer {
    if buf.dtype() == DType::U8 {
        return buf;
    }

    let contig = buf.to_contiguous();
    let count = contig.layout.num_elements();
    let shape = contig.shape().to_vec();

    match contig.dtype() {
        DType::F32 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };
            // Scale from [0.0, 1.0] to [0, 255], clamping values outside range
            let new_data: Vec<u8> = src
                .iter()
                .map(|&x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::F64 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f64>(), count) };
            let new_data: Vec<u8> = src
                .iter()
                .map(|&x| (x.clamp(0.0, 1.0) * 255.0).round() as u8)
                .collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::U16 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u16>(), count) };
            // Scale from [0, 65535] to [0, 255]
            let new_data: Vec<u8> = src.iter().map(|&x| (x >> 8) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::I16 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<i16>(), count) };
            let new_data: Vec<u8> = src.iter().map(|&x| x.clamp(0, 255) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::U32 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u32>(), count) };
            let new_data: Vec<u8> = src.iter().map(|&x| (x.min(255)) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::I32 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<i32>(), count) };
            let new_data: Vec<u8> = src.iter().map(|&x| x.clamp(0, 255) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        DType::I8 => {
            let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<i8>(), count) };
            let new_data: Vec<u8> = src.iter().map(|&x| x.max(0) as u8).collect();
            ViewBuffer::from_vec(new_data).reshape(shape)
        }
        _ => {
            // For other types, use the cast method
            contig.cast(DType::U8)
        }
    }
}

/// Map our filter types to fast_image_resize algorithm types.
#[cfg(feature = "image_interop")]
#[inline]
fn to_fir_algorithm(filter: &FilterType) -> fir::ResizeAlg {
    match filter {
        FilterType::Nearest => fir::ResizeAlg::Nearest,
        FilterType::Triangle => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        FilterType::CatmullRom => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
        FilterType::Gaussian => fir::ResizeAlg::Convolution(fir::FilterType::Gaussian),
        FilterType::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
    }
}

/// Map (channels, element-size) to the appropriate `fir::PixelType`.
///
/// `fast_image_resize` natively supports U8, U16, and F32 pixel types
/// (with 1, 2, 3, and 4 channel variants for U8/U16/F32 plus a single-channel I32).
#[cfg(feature = "image_interop")]
fn pixel_type_for(dtype: DType, channels: usize) -> fir::PixelType {
    match (dtype, channels) {
        (DType::U8, 1) => fir::PixelType::U8,
        (DType::U8, 2) => fir::PixelType::U8x2,
        (DType::U8, 3) => fir::PixelType::U8x3,
        (DType::U8, 4) => fir::PixelType::U8x4,
        (DType::U16, 1) => fir::PixelType::U16,
        (DType::U16, 2) => fir::PixelType::U16x2,
        (DType::U16, 3) => fir::PixelType::U16x3,
        (DType::U16, 4) => fir::PixelType::U16x4,
        (DType::F32, 1) => fir::PixelType::F32,
        (DType::F32, 2) => fir::PixelType::F32x2,
        (DType::F32, 3) => fir::PixelType::F32x3,
        (DType::F32, 4) => fir::PixelType::F32x4,
        (DType::I32, 1) => fir::PixelType::I32,
        _ => panic!("fast_image_resize does not support dtype {dtype:?} with {channels} channels"),
    }
}

/// Resize using fast_image_resize with SIMD optimization.
///
/// Supports U8, U16, and F32 dtypes natively via ``fast_image_resize``.
/// For other dtypes the buffer is cast to F32, resized, then cast back to
/// the original dtype so that the output always preserves the input dtype.
///
/// Non-contiguous inputs are materialized first as fast_image_resize
/// requires contiguous memory.
#[cfg(feature = "image_interop")]
fn resize_strided(
    buf: ViewBuffer,
    target_width: u32,
    target_height: u32,
    filter: FilterType,
) -> ViewBuffer {
    // Ensure contiguous input (fast_image_resize requires contiguous memory)
    let contig_buf = if buf.layout.is_contiguous() {
        buf
    } else {
        buf.to_contiguous()
    };

    let dtype = contig_buf.dtype();
    let shape = contig_buf.shape();
    let input_rank = shape.len();
    let (h, w) = (shape[0], shape[1]);
    let c = shape.get(2).copied().unwrap_or(1);

    // Dtypes natively supported by fast_image_resize
    let resized = match dtype {
        DType::U8 => resize_typed_u8(&contig_buf, h, w, c, target_height, target_width, &filter),
        DType::U16 => resize_typed_u16(&contig_buf, h, w, c, target_height, target_width, &filter),
        DType::F32 => resize_typed_f32(&contig_buf, h, w, c, target_height, target_width, &filter),
        // Unsupported by fast_image_resize: cast to F32, resize, cast back.
        other => {
            let f32_buf = contig_buf.cast(DType::F32);
            let r = resize_typed_f32(&f32_buf, h, w, c, target_height, target_width, &filter);
            r.cast(other)
        }
    };

    // Preserve input rank: 2D input must produce 2D output.  The typed
    // resize functions always produce [H, W, C]; squeeze the trailing
    // dimension when the input was 2D.
    if input_rank == 2 {
        resized.reshape(vec![target_height as usize, target_width as usize])
    } else {
        resized
    }
}

#[cfg(feature = "image_interop")]
thread_local! {
    /// Reused across resize calls on the same worker thread to avoid
    /// reallocating fast_image_resize's internal scratch buffers on every row.
    /// `fir::Resizer` adapts to the pixel type and dimensions on each `resize`
    /// call, so a single instance safely serves U8/U16/F32 at any size. It is
    /// thread-local, so streaming morsel workers never share one.
    static FIR_RESIZER: std::cell::RefCell<fir::Resizer> =
        std::cell::RefCell::new(fir::Resizer::new());

    /// Scratch buffer for the horizontal pass of separable Gaussian blur.
    /// Holds f32 values for the current image (h × w × channels). Grows to
    /// fit the largest image seen per thread and is never shrunk — amortised
    /// O(1) allocation like TILE_EXTRACT_BUF in tiling.rs.
    static BLUR_HORIZ_BUF: std::cell::RefCell<Vec<f32>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Perform a typed resize for U8 data.
#[cfg(feature = "image_interop")]
fn resize_typed_u8(
    contig_buf: &ViewBuffer,
    h: usize,
    w: usize,
    c: usize,
    target_height: u32,
    target_width: u32,
    filter: &FilterType,
) -> ViewBuffer {
    let fir_filter = to_fir_algorithm(filter);
    let pixel_type = pixel_type_for(DType::U8, c);
    let src_len = h * w * c;
    let dst_size = (target_height as usize) * (target_width as usize) * c;
    let mut dst_data = vec![0u8; dst_size];

    let src_slice = unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u8>(), src_len) };
    let src_image = fir::images::ImageRef::new(w as u32, h as u32, src_slice, pixel_type)
        .expect("Failed to create source image");
    let mut dst_image =
        fir::images::Image::from_slice_u8(target_width, target_height, &mut dst_data, pixel_type)
            .expect("Failed to create dest image");

    FIR_RESIZER.with(|cell| {
        cell.borrow_mut()
            .resize(
                &src_image,
                &mut dst_image,
                &fir::ResizeOptions::new().resize_alg(fir_filter),
            )
            .expect("Resize failed");
    });

    ViewBuffer::from_vec(dst_data).reshape(vec![target_height as usize, target_width as usize, c])
}

/// Perform a typed resize for U16 data.
#[cfg(feature = "image_interop")]
fn resize_typed_u16(
    contig_buf: &ViewBuffer,
    h: usize,
    w: usize,
    c: usize,
    target_height: u32,
    target_width: u32,
    filter: &FilterType,
) -> ViewBuffer {
    let fir_filter = to_fir_algorithm(filter);
    let pixel_type = pixel_type_for(DType::U16, c);
    let src_len = h * w * c;
    let dst_size = (target_height as usize) * (target_width as usize) * c;
    let mut dst_data: Vec<u16> = vec![0u16; dst_size];

    let src_slice = unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u16>(), src_len) };
    // Reinterpret the u16 slices as byte slices for fir API
    let src_bytes =
        unsafe { std::slice::from_raw_parts(src_slice.as_ptr() as *const u8, src_len * 2) };
    let dst_bytes =
        unsafe { std::slice::from_raw_parts_mut(dst_data.as_mut_ptr() as *mut u8, dst_size * 2) };

    let src_image = fir::images::ImageRef::new(w as u32, h as u32, src_bytes, pixel_type)
        .expect("Failed to create source image");
    let mut dst_image =
        fir::images::Image::from_slice_u8(target_width, target_height, dst_bytes, pixel_type)
            .expect("Failed to create dest image");

    FIR_RESIZER.with(|cell| {
        cell.borrow_mut()
            .resize(
                &src_image,
                &mut dst_image,
                &fir::ResizeOptions::new().resize_alg(fir_filter),
            )
            .expect("Resize failed");
    });

    ViewBuffer::from_vec(dst_data).reshape(vec![target_height as usize, target_width as usize, c])
}

/// Perform a typed resize for F32 data.
#[cfg(feature = "image_interop")]
fn resize_typed_f32(
    contig_buf: &ViewBuffer,
    h: usize,
    w: usize,
    c: usize,
    target_height: u32,
    target_width: u32,
    filter: &FilterType,
) -> ViewBuffer {
    let fir_filter = to_fir_algorithm(filter);
    let pixel_type = pixel_type_for(DType::F32, c);
    let src_len = h * w * c;
    let dst_size = (target_height as usize) * (target_width as usize) * c;
    let mut dst_data: Vec<f32> = vec![0.0f32; dst_size];

    let src_slice = unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<f32>(), src_len) };
    // Reinterpret as byte slices for fir API
    let src_bytes =
        unsafe { std::slice::from_raw_parts(src_slice.as_ptr() as *const u8, src_len * 4) };
    let dst_bytes =
        unsafe { std::slice::from_raw_parts_mut(dst_data.as_mut_ptr() as *mut u8, dst_size * 4) };

    let src_image = fir::images::ImageRef::new(w as u32, h as u32, src_bytes, pixel_type)
        .expect("Failed to create source image");
    let mut dst_image =
        fir::images::Image::from_slice_u8(target_width, target_height, dst_bytes, pixel_type)
            .expect("Failed to create dest image");

    FIR_RESIZER.with(|cell| {
        cell.borrow_mut()
            .resize(
                &src_image,
                &mut dst_image,
                &fir::ResizeOptions::new().resize_alg(fir_filter),
            )
            .expect("Resize failed");
    });

    ViewBuffer::from_vec(dst_data).reshape(vec![target_height as usize, target_width as usize, c])
}

/// Strided grayscale conversion that works on non-contiguous buffers.
///
/// Uses ndarray for strided access when available, falling back to manual
/// strided iteration. This avoids the need to call `to_contiguous()` for
/// flipped, cropped, or transposed buffers.
///
/// Uses BT.601 coefficients: Y = 0.299*R + 0.587*G + 0.114*B
/// Implemented with fixed-point math: Y = (77*R + 150*G + 29*B + 128) >> 8
///
/// This implementation uses direct pointer arithmetic which is faster than
/// ndarray per-pixel indexing due to avoiding bounds checks.
#[cfg(feature = "image_interop")]
fn grayscale_strided(buf: ViewBuffer) -> ViewBuffer {
    let shape = buf.shape();
    let channels = shape.get(2).copied().unwrap_or(1);

    if channels == 1 {
        // Already grayscale
        return buf;
    }

    let dtype = buf.dtype();

    // U8 fast path: uses fixed-point integer math (original optimized path)
    if dtype == DType::U8 {
        return grayscale_u8(buf);
    }

    // Generic path for all other dtypes: uses f64 BT.601 coefficients
    match dtype {
        DType::U8 => unreachable!(), // Handled above
        DType::I8 => grayscale_typed::<i8>(buf),
        DType::U16 => grayscale_typed::<u16>(buf),
        DType::I16 => grayscale_typed::<i16>(buf),
        DType::U32 => grayscale_typed::<u32>(buf),
        DType::I32 => grayscale_typed::<i32>(buf),
        DType::F32 => grayscale_typed::<f32>(buf),
        DType::F64 => grayscale_typed::<f64>(buf),
        DType::U64 => grayscale_typed::<u64>(buf),
        DType::I64 => grayscale_typed::<i64>(buf),
    }
}

/// Fast U8 grayscale using fixed-point BT.601 coefficients.
///
/// This preserves the original optimized path for the common u8 case.
/// Supports both contiguous and strided input buffers.
#[cfg(feature = "image_interop")]
fn grayscale_u8(buf: ViewBuffer) -> ViewBuffer {
    let shape = buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let channels = shape.get(2).copied().unwrap_or(1);
    let strides = buf.strides_bytes();

    // Fast path: contiguous 3-channel or 4-channel (RGBA) buffer
    if buf.layout.is_contiguous() && (channels == 3 || channels == 4) {
        let data = unsafe { std::slice::from_raw_parts(buf.as_ptr::<u8>(), h * w * channels) };
        let mut gray_data: Vec<u8> = Vec::with_capacity(h * w);

        for pixel in data.chunks_exact(channels) {
            let r = pixel[0] as u32;
            let g = pixel[1] as u32;
            let b = pixel[2] as u32;
            let gray = ((77 * r + 150 * g + 29 * b + 128) >> 8).min(255) as u8;
            gray_data.push(gray);
        }

        return ViewBuffer::from_vec(gray_data).reshape(vec![h, w, 1]);
    }

    // Fast path: contiguous 2-channel (GrayA) — take the intensity channel
    if buf.layout.is_contiguous() && channels == 2 {
        let data = unsafe { std::slice::from_raw_parts(buf.as_ptr::<u8>(), h * w * 2) };
        let mut gray_data: Vec<u8> = Vec::with_capacity(h * w);

        for pixel in data.chunks_exact(2) {
            gray_data.push(pixel[0]);
        }

        return ViewBuffer::from_vec(gray_data).reshape(vec![h, w, 1]);
    }

    // Strided path: handles non-contiguous buffers (crop, flip, etc.)
    let (stride_h, stride_w, stride_c) =
        (strides[0], strides[1], strides.get(2).copied().unwrap_or(1));
    let base_ptr = unsafe { buf.as_ptr::<u8>() };

    let mut gray_data: Vec<u8> = Vec::with_capacity(h * w);

    if channels == 2 {
        // GrayA: take the intensity channel only
        for y in 0..h {
            for x in 0..w {
                let pixel_offset = y as isize * stride_h + x as isize * stride_w;
                unsafe {
                    let pixel_ptr = base_ptr.offset(pixel_offset);
                    gray_data.push(*pixel_ptr);
                }
            }
        }
    } else {
        // RGB or RGBA: BT.601 on first 3 channels, alpha ignored
        for y in 0..h {
            for x in 0..w {
                let pixel_offset = y as isize * stride_h + x as isize * stride_w;
                unsafe {
                    let pixel_ptr = base_ptr.offset(pixel_offset);
                    let r = *pixel_ptr as u32;
                    let g = *pixel_ptr.offset(stride_c) as u32;
                    let b = *pixel_ptr.offset(2 * stride_c) as u32;
                    let gray = ((77 * r + 150 * g + 29 * b + 128) >> 8).min(255) as u8;
                    gray_data.push(gray);
                }
            }
        }
    }

    ViewBuffer::from_vec(gray_data).reshape(vec![h, w, 1])
}

/// Dtype-generic grayscale using float BT.601 coefficients.
///
/// Reads multichannel data as type `T`, applies `Y = 0.299*R + 0.587*G + 0.114*B`
/// in `f64` arithmetic, then casts back to `T`. Preserves the input dtype.
#[cfg(feature = "image_interop")]
fn grayscale_typed<T>(buf: ViewBuffer) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + num_traits::NumCast,
{
    use num_traits::NumCast;

    // BT.601 luma coefficients
    const R_COEFF: f64 = 0.299;
    const G_COEFF: f64 = 0.587;
    const B_COEFF: f64 = 0.114;

    let shape = buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let channels = shape.get(2).copied().unwrap_or(1);

    // Ensure contiguous for typed slice access
    let contig_buf = if buf.layout.is_contiguous() {
        buf
    } else {
        buf.to_contiguous()
    };

    let src_data: &[T] = contig_buf.as_slice::<T>();
    let mut gray_data: Vec<T> = Vec::with_capacity(h * w);

    for pixel in src_data.chunks_exact(channels) {
        let r: f64 = NumCast::from(pixel[0]).unwrap_or(0.0);
        let g: f64 = if channels > 1 {
            NumCast::from(pixel[1]).unwrap_or(0.0)
        } else {
            r
        };
        let b: f64 = if channels > 2 {
            NumCast::from(pixel[2]).unwrap_or(0.0)
        } else {
            g
        };

        let luma = R_COEFF * r + G_COEFF * g + B_COEFF * b;

        // For integer types, clamp to valid range
        let is_float = matches!(T::DTYPE, DType::F32 | DType::F64);
        let clamped = if is_float {
            luma
        } else {
            clamp_for_dtype(luma, T::DTYPE)
        };

        gray_data.push(NumCast::from(clamped).unwrap_or(T::default()));
    }

    ViewBuffer::from_vec(gray_data).reshape(vec![h, w, 1])
}

/// Applies an image operation to a buffer.
///
/// The conversion strategy depends on the operation's ``working_dtype()``:
///
/// - ``Some(DType::U8)``: the operation requires U8 data (blur).
///   Float inputs in [0.0, 1.0] are scaled to [0, 255].
/// - ``None``: the operation works on the input's native dtype
///   (resize, rotate, grayscale, threshold).
///   The buffer is passed through unchanged.
///
/// Applies an image operation to a buffer.
///
/// Handles dtype promotion (converting to the op's working dtype before
/// dispatch) and validates the output dtype contract in debug builds.
#[cfg(feature = "image_interop")]
#[inline]
pub(crate) fn apply_image_inner(buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    let input_dtype = buf.dtype();

    // Only convert to U8 when the operation requires it.
    let work_buf = match op.working_dtype() {
        Some(DType::U8) => convert_to_u8_for_image(buf),
        Some(target) => buf.cast(target),
        None => buf, // Resize: use the input's native dtype
    };

    let result = apply_image_dispatch(work_buf, op.clone());

    debug_assert!(
        op.validate_output_dtype(input_dtype, result.dtype())
            .is_ok(),
        "ImageOp contract violation: {}",
        op.validate_output_dtype(input_dtype, result.dtype())
            .unwrap_err()
    );

    result
}

/// SIMD-friendly threshold implementation for contiguous u8 data.
///
/// Processes in chunks of 32 bytes (256 bits = AVX) to enable auto-vectorization.
/// The compiler can vectorize the comparison and conditional select operations.
#[cfg(feature = "image_interop")]
#[inline]
fn threshold_simd(src: &[u8], thresh: u8) -> Vec<u8> {
    let count = src.len();
    let mut new_data: Vec<u8> = Vec::with_capacity(count);

    // Process in chunks of 32 for SIMD (u8 x 32 = 256 bits = AVX)
    const CHUNK_SIZE: usize = 32;
    let chunks = count / CHUNK_SIZE;
    let remainder = count % CHUNK_SIZE;

    // Process main chunks - compiler can auto-vectorize this pattern
    for chunk_idx in 0..chunks {
        let base = chunk_idx * CHUNK_SIZE;
        let chunk = &src[base..base + CHUNK_SIZE];

        // Fixed-size array enables SIMD optimization
        let mut out = [0u8; CHUNK_SIZE];
        for (i, &p) in chunk.iter().enumerate() {
            // Simple comparison that vectorizes well
            out[i] = if p > thresh { 255 } else { 0 };
        }
        new_data.extend_from_slice(&out);
    }

    // Handle remainder elements
    let remainder_start = chunks * CHUNK_SIZE;
    for i in 0..remainder {
        let p = src[remainder_start + i];
        new_data.push(if p > thresh { 255 } else { 0 });
    }

    new_data
}

/// Dtype-generic threshold that compares each element against `f64` threshold.
///
/// For each element, outputs `255u8` if the element (cast to `f64`) exceeds
/// the threshold, else `0u8`. The output is always `Vec<u8>` regardless of
/// input dtype.
///
/// For U8 input where the threshold fits in u8 range, delegates to the
/// SIMD-optimized `threshold_simd` fast path.
#[cfg(feature = "image_interop")]
fn threshold_generic(buf: ViewBuffer, thresh: f64) -> ViewBuffer {
    let shape = buf.shape();

    // Validate: threshold only works on single-channel data
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Threshold requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() first to convert multi-channel images to grayscale."
        );
    }

    let dtype = buf.dtype();

    // U8 SIMD fast path: when input is u8 and threshold fits in u8 range
    if dtype == DType::U8 && (0.0..=255.0).contains(&thresh) {
        let thresh_u8 = thresh as u8;

        // Try image view path for strided u8 data
        if let Ok(view) = buf.as_image_view::<Luma<u8>>() {
            let total_pixels = (view.width * view.height) as usize;
            let mut new_data: Vec<u8> = Vec::with_capacity(total_pixels);

            for y in 0..view.height {
                let row_start = (y as usize) * view.row_stride;
                let row_slice = &view.data[row_start..row_start + view.width as usize];
                let thresholded = threshold_simd(row_slice, thresh_u8);
                new_data.extend_from_slice(&thresholded);
            }

            return ViewBuffer::from_vec(new_data).reshape(vec![
                view.height as usize,
                view.width as usize,
                1,
            ]);
        }

        // Fallback contiguous u8 path
        let contig_buf = if buf.layout.is_contiguous() {
            buf
        } else {
            buf.to_contiguous()
        };
        let count = contig_buf.layout.num_elements();
        let src_slice = unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u8>(), count) };
        let new_data = threshold_simd(src_slice, thresh_u8);
        return ViewBuffer::from_vec(new_data).reshape(contig_buf.shape().to_vec());
    }

    // Generic path: dispatch by dtype, compare in f64 space
    match dtype {
        DType::U8 => threshold_typed::<u8>(buf, thresh),
        DType::I8 => threshold_typed::<i8>(buf, thresh),
        DType::U16 => threshold_typed::<u16>(buf, thresh),
        DType::I16 => threshold_typed::<i16>(buf, thresh),
        DType::U32 => threshold_typed::<u32>(buf, thresh),
        DType::I32 => threshold_typed::<i32>(buf, thresh),
        DType::F32 => threshold_typed::<f32>(buf, thresh),
        DType::F64 => threshold_typed::<f64>(buf, thresh),
        DType::U64 => threshold_typed::<u64>(buf, thresh),
        DType::I64 => threshold_typed::<i64>(buf, thresh),
    }
}

/// Typed threshold helper: reads elements as `T`, compares against `f64` threshold,
/// outputs `Vec<u8>` with 0 or 255 values.
#[cfg(feature = "image_interop")]
fn threshold_typed<T>(buf: ViewBuffer, thresh: f64) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + num_traits::NumCast,
{
    use num_traits::NumCast;

    let contig_buf = if buf.layout.is_contiguous() {
        buf
    } else {
        buf.to_contiguous()
    };

    let out_shape = contig_buf.shape().to_vec();
    let src_data: &[T] = contig_buf.as_slice::<T>();

    let new_data: Vec<u8> = src_data
        .iter()
        .map(|x| {
            let v: f64 = NumCast::from(*x).unwrap_or(0.0);
            if v > thresh {
                255u8
            } else {
                0u8
            }
        })
        .collect();

    ViewBuffer::from_vec(new_data).reshape(out_shape)
}

/// Check if a shape represents a single-channel image.
///
/// Valid single-channel shapes:
/// - `[H, W]` - 2D array
/// - `[H, W, 1]` - 3D with 1 channel
///
/// Invalid (multi-channel):
/// - `[H, W, C]` where C > 1
#[cfg(feature = "image_interop")]
#[inline]
fn is_single_channel(shape: &[usize]) -> bool {
    match shape.len() {
        2 => true,          // [H, W] - 2D is single channel
        3 => shape[2] == 1, // [H, W, 1] - explicit single channel
        _ => false,         // Other ranks not supported
    }
}

/// Get the number of channels from a shape.
#[cfg(feature = "image_interop")]
#[inline]
fn get_channel_count(shape: &[usize]) -> usize {
    match shape.len() {
        2 => 1,        // [H, W] - implicit single channel
        3 => shape[2], // [H, W, C]
        _ => 0,        // Invalid
    }
}

/// Apply a 2D affine warp to a `ViewBuffer`.
///
/// Dispatches to `affine_warp_typed` based on the buffer's dtype.
fn apply_affine_warp(buf: ViewBuffer, params: crate::ops::affine::AffineParams) -> ViewBuffer {
    let shape = buf.shape();
    if shape.len() < 2 {
        return buf;
    }

    let dtype = buf.dtype();
    match dtype {
        DType::U8 => affine_warp_typed::<u8>(buf, &params),
        DType::I8 => affine_warp_typed::<i8>(buf, &params),
        DType::U16 => affine_warp_typed::<u16>(buf, &params),
        DType::I16 => affine_warp_typed::<i16>(buf, &params),
        DType::U32 => affine_warp_typed::<u32>(buf, &params),
        DType::I32 => affine_warp_typed::<i32>(buf, &params),
        DType::F32 => affine_warp_typed::<f32>(buf, &params),
        DType::F64 => affine_warp_typed::<f64>(buf, &params),
        DType::U64 => affine_warp_typed::<u64>(buf, &params),
        DType::I64 => affine_warp_typed::<i64>(buf, &params),
    }
}

/// Typed affine warp: bilinear or nearest interpolation in `f64` arithmetic.
fn affine_warp_typed<T>(buf: ViewBuffer, params: &crate::ops::affine::AffineParams) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + num_traits::NumCast,
{
    use crate::ops::affine::InterpolationType;
    use num_traits::NumCast;

    let shape = buf.shape();
    let in_h = shape[0];
    let in_w = shape[1];
    let channels = shape.get(2).copied().unwrap_or(1);

    let out_h = params.output_height as usize;
    let out_w = params.output_width as usize;

    // The user-facing matrix follows OpenCV convention (forward mapping).
    // Invert the 2x3 matrix for inverse-mapping interpolation.
    let [a_fwd, b_fwd, tx_fwd, c_fwd, d_fwd, ty_fwd] = params.matrix;
    let det = a_fwd * d_fwd - b_fwd * c_fwd;
    let (a, b, tx, c, d, ty) = if det.abs() < 1e-15 {
        // Singular matrix — fall back to identity (no-op)
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    } else {
        let inv_det = 1.0 / det;
        let ai = d_fwd * inv_det;
        let bi = -b_fwd * inv_det;
        let ci = -c_fwd * inv_det;
        let di = a_fwd * inv_det;
        let txi = -(ai * tx_fwd + bi * ty_fwd);
        let tyi = -(ci * tx_fwd + di * ty_fwd);
        (ai, bi, txi, ci, di, tyi)
    };

    let contig_buf = if buf.layout.is_contiguous() {
        buf
    } else {
        buf.to_contiguous()
    };
    let src_data: &[T] = contig_buf.as_slice::<T>();

    let output_size = out_h * out_w * channels;
    let border_val: T = NumCast::from(params.border_value).unwrap_or(T::default());
    let mut dst_data: Vec<T> = vec![border_val; output_size];

    let is_float = matches!(T::DTYPE, DType::F32 | DType::F64);

    for y_dst in 0..out_h {
        for x_dst in 0..out_w {
            let x_src = a * x_dst as f64 + b * y_dst as f64 + tx;
            let y_src = c * x_dst as f64 + d * y_dst as f64 + ty;

            match params.interpolation {
                InterpolationType::Nearest => {
                    let sx = x_src.round() as i64;
                    let sy = y_src.round() as i64;
                    if sx >= 0 && sy >= 0 && (sx as usize) < in_w && (sy as usize) < in_h {
                        let src_idx = (sy as usize * in_w + sx as usize) * channels;
                        let dst_idx = (y_dst * out_w + x_dst) * channels;
                        dst_data[dst_idx..dst_idx + channels]
                            .copy_from_slice(&src_data[src_idx..src_idx + channels]);
                    }
                }
                InterpolationType::Bilinear => {
                    let x0 = x_src.floor() as i64;
                    let y0 = y_src.floor() as i64;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;

                    // Fully out of bounds — dst already filled with border_val
                    if x1 < 0 || y1 < 0 || x0 >= in_w as i64 || y0 >= in_h as i64 {
                        continue;
                    }

                    let dx = x_src - x0 as f64;
                    let dy = y_src - y0 as f64;

                    let bv: f64 = params.border_value;

                    let in_bounds = |px: i64, py: i64| -> bool {
                        px >= 0 && py >= 0 && (px as usize) < in_w && (py as usize) < in_h
                    };

                    let dst_idx = (y_dst * out_w + x_dst) * channels;
                    for ch in 0..channels {
                        let sample = |px: i64, py: i64| -> f64 {
                            if in_bounds(px, py) {
                                let idx = (py as usize * in_w + px as usize) * channels + ch;
                                NumCast::from(src_data[idx]).unwrap_or(bv)
                            } else {
                                bv
                            }
                        };

                        let v00 = sample(x0, y0);
                        let v10 = sample(x1, y0);
                        let v01 = sample(x0, y1);
                        let v11 = sample(x1, y1);

                        let v0 = v00 * (1.0 - dx) + v10 * dx;
                        let v1 = v01 * (1.0 - dx) + v11 * dx;
                        let v = v0 * (1.0 - dy) + v1 * dy;

                        let clamped = if is_float {
                            v
                        } else {
                            clamp_for_dtype(v, T::DTYPE)
                        };
                        dst_data[dst_idx + ch] = NumCast::from(clamped).unwrap_or(T::default());
                    }
                }
            }
        }
    }

    let output_shape = if channels == 1 {
        vec![out_h, out_w]
    } else {
        vec![out_h, out_w, channels]
    };

    ViewBuffer::from_vec(dst_data).reshape(output_shape)
}

/// Clamp an `f64` value to the representable range of the given integer `DType`.
///
/// `#[inline(always)]`: hot loops call this with a compile-time-constant
/// dtype from monomorphized contexts, so inlining folds the match away.
#[inline(always)]
fn clamp_for_dtype(v: f64, dtype: DType) -> f64 {
    match dtype {
        // Round before clamping so that e.g. 127.9999 → 128, not 127.
        // NumCast::from truncates (floor) for positive floats, so without rounding
        // Gaussian convolution on a solid-128 image gives 127.
        DType::U8 => v.round().clamp(0.0, u8::MAX as f64),
        DType::I8 => v.round().clamp(i8::MIN as f64, i8::MAX as f64),
        DType::U16 => v.round().clamp(0.0, u16::MAX as f64),
        DType::I16 => v.round().clamp(i16::MIN as f64, i16::MAX as f64),
        DType::U32 => v.round().clamp(0.0, u32::MAX as f64),
        DType::I32 => v.round().clamp(i32::MIN as f64, i32::MAX as f64),
        DType::U64 => v.round().clamp(0.0, u64::MAX as f64),
        DType::I64 => v.round().clamp(i64::MIN as f64, i64::MAX as f64),
        DType::F32 | DType::F64 => v,
    }
}

/// Inner implementation of image operations (without tiling logic).
#[cfg(feature = "image_interop")]
#[inline]
fn apply_image_dispatch(work_buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    match op.kind {
        ImageOpKind::Threshold(thresh) => threshold_generic(work_buf, thresh),
        ImageOpKind::Grayscale => grayscale_strided(work_buf),
        ImageOpKind::Resize {
            width,
            height,
            filter,
        } => resize_strided(work_buf, width, height, filter),
        ImageOpKind::Canny {
            low_threshold,
            high_threshold,
        } => apply_canny(work_buf, low_threshold, high_threshold),
        ImageOpKind::HistogramEqualize => apply_histogram_equalize(work_buf),
        ImageOpKind::Erode { ksize, iterations } => apply_erode(work_buf, ksize, iterations),
        ImageOpKind::Dilate { ksize, iterations } => apply_dilate(work_buf, ksize, iterations),
        ImageOpKind::MorphGradient { ksize } => apply_morph_gradient(work_buf, ksize),
        ImageOpKind::Blur { sigma } => {
            let contig_buf = if work_buf.layout.is_contiguous() {
                work_buf
            } else {
                work_buf.to_contiguous()
            };
            match contig_buf.dtype() {
                DType::U8 => separable_gaussian_blur_typed::<u8>(&contig_buf, sigma),
                DType::U16 => separable_gaussian_blur_typed::<u16>(&contig_buf, sigma),
                DType::F32 => separable_gaussian_blur_typed::<f32>(&contig_buf, sigma),
                other => {
                    let f32_buf = contig_buf.cast(DType::F32);
                    separable_gaussian_blur_typed::<f32>(&f32_buf, sigma).cast(other)
                }
            }
        }
        // Deferred resizes: dimensions come from output_hw — the same
        // authority infer_shape declared at plan time — then the shared
        // resize kernel runs.
        ref kind @ (ImageOpKind::ResizeScale { .. }
        | ImageOpKind::ResizeToHeight { .. }
        | ImageOpKind::ResizeToWidth { .. }
        | ImageOpKind::ResizeMax { .. }
        | ImageOpKind::ResizeMin { .. }) => {
            let shape = work_buf.shape();
            let (h, w) = kind
                .output_hw(shape[0], shape[1])
                .expect("deferred resize kinds always produce output dims");
            let filter = match kind {
                ImageOpKind::ResizeScale { filter, .. }
                | ImageOpKind::ResizeToHeight { filter, .. }
                | ImageOpKind::ResizeToWidth { filter, .. }
                | ImageOpKind::ResizeMax { filter, .. }
                | ImageOpKind::ResizeMin { filter, .. } => *filter,
                _ => unreachable!(),
            };
            resize_strided(work_buf, w as u32, h as u32, filter)
        }
        ImageOpKind::Pad {
            top,
            bottom,
            left,
            right,
            value,
            mode,
        } => crate::ops::pad::pad(&work_buf, top, bottom, left, right, value, mode),
        ImageOpKind::PadToSize {
            height,
            width,
            position,
            value,
        } => crate::ops::pad::pad_to_size(&work_buf, height, width, position, value),
        ImageOpKind::Letterbox {
            height,
            width,
            value,
            filter,
        } => {
            let shape = work_buf.shape();
            let (fit_h, fit_w) =
                crate::ops::image::letterbox_fit(shape[0], shape[1], height, width);
            let resized = resize_strided(work_buf, fit_w as u32, fit_h as u32, filter);
            crate::ops::pad::pad_to_size(
                &resized,
                height,
                width,
                crate::ops::pad::PadPosition::Center,
                value,
            )
        }
        ImageOpKind::ChannelSwap { ref order } => apply_channel_swap(&work_buf, order),
    }
}

/// Builds a normalised 1-D Gaussian kernel of radius `ceil(3σ)`.
/// The radius formula matches the halo declaration in `ops/image.rs`.
#[cfg(feature = "image_interop")]
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let radius = (sigma * 3.0).ceil() as usize;
    let size = 2 * radius + 1;
    let mut k: Vec<f32> = (0..size)
        .map(|i| {
            let x = i as f32 - radius as f32;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Separable Gaussian blur: one 1-D horizontal pass followed by one 1-D
/// vertical pass.  Mathematically equivalent to 2-D Gaussian convolution but
/// O(k) per pixel instead of O(k²) — ~6.5× fewer multiply-adds for σ=2.
///
/// Border handling: replicate (edge pixels clamped to image boundary), matching
/// the behaviour of `morph_minmax_typed` in the same file.
///
/// Structured for auto-vectorization:
/// - the input is converted to f32 once up front (n casts instead of k·n
///   per-tap `NumCast` calls);
/// - the horizontal pass accumulates each kernel tap as a contiguous
///   shifted-slice multiply-add over the row interior, with clamped-index
///   gathers only for the `radius` columns at each border;
/// - the vertical pass accumulates whole rows (`row_out += k·row_src`), so
///   border handling is a per-row clamp, not a per-element one;
/// - the f32 → T conversion happens once per element after the passes.
///
/// The horizontal-pass intermediate is stored in a thread-local f32 slab
/// (`BLUR_HORIZ_BUF`) that grows to fit the largest image seen per thread and
/// is never shrunk — zero allocator round-trip on warm paths.
#[cfg(feature = "image_interop")]
fn separable_gaussian_blur_typed<T>(contig_buf: &ViewBuffer, sigma: f32) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + Copy + num_traits::NumCast,
{
    use num_traits::NumCast;

    let shape = contig_buf.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);
    let n = h * w * c;
    let wc = w * c;

    let kernel = gaussian_kernel_1d(sigma);
    let radius = kernel.len() / 2;
    let src: &[T] = contig_buf.as_slice::<T>();

    // ── Convert input to f32 once ────────────────────────────────────────
    let mut input_f32: Vec<f32> = Vec::with_capacity(n);
    input_f32.extend(src.iter().map(|&v| {
        let v: f32 = NumCast::from(v).unwrap_or(0.0);
        v
    }));

    BLUR_HORIZ_BUF.with(|cell| {
        let mut slab = cell.borrow_mut();
        if slab.len() < n {
            slab.resize(n, 0.0f32);
        }
        let horiz = &mut slab[..n];

        // ── Horizontal pass (f32 → f32) ──────────────────────────────────
        for y in 0..h {
            let row_in = &input_f32[y * wc..(y + 1) * wc];
            let row_out = &mut horiz[y * wc..(y + 1) * wc];

            if w <= 2 * radius {
                // Image narrower than the kernel: clamped gather everywhere.
                for x in 0..w {
                    for ch in 0..c {
                        let mut sum = 0.0f32;
                        for (ki, &kw) in kernel.iter().enumerate() {
                            let sx = (x as i64 + ki as i64 - radius as i64).clamp(0, w as i64 - 1)
                                as usize;
                            sum += kw * row_in[sx * c + ch];
                        }
                        row_out[x * c + ch] = sum;
                    }
                }
                continue;
            }

            // Interior: per-tap shifted-slice multiply-add over contiguous
            // memory — the inner zip vectorizes.
            let lo = radius * c;
            let hi = (w - radius) * c;
            row_out[lo..hi].fill(0.0);
            for (ki, &kw) in kernel.iter().enumerate() {
                let shift = (ki as i64 - radius as i64) * c as i64;
                let src_start = (lo as i64 + shift) as usize;
                let src_slice = &row_in[src_start..src_start + (hi - lo)];
                for (o, &v) in row_out[lo..hi].iter_mut().zip(src_slice) {
                    *o += kw * v;
                }
            }
            // Borders: clamped gather for `radius` columns on each side.
            for x in (0..radius).chain(w - radius..w) {
                for ch in 0..c {
                    let mut sum = 0.0f32;
                    for (ki, &kw) in kernel.iter().enumerate() {
                        let sx =
                            (x as i64 + ki as i64 - radius as i64).clamp(0, w as i64 - 1) as usize;
                        sum += kw * row_in[sx * c + ch];
                    }
                    row_out[x * c + ch] = sum;
                }
            }
        }

        // ── Vertical pass (f32 → f32 row accumulation) → T ───────────────
        let is_float = matches!(T::DTYPE, DType::F32 | DType::F64);
        let mut acc_row = vec![0.0f32; wc];
        let mut out: Vec<T> = Vec::with_capacity(n);
        for y in 0..h {
            acc_row.fill(0.0);
            for (ki, &kw) in kernel.iter().enumerate() {
                let sy = (y as i64 + ki as i64 - radius as i64).clamp(0, h as i64 - 1) as usize;
                let src_row = &horiz[sy * wc..(sy + 1) * wc];
                for (o, &v) in acc_row.iter_mut().zip(src_row) {
                    *o += kw * v;
                }
            }
            if is_float {
                out.extend(
                    acc_row
                        .iter()
                        .map(|&v| NumCast::from(v).unwrap_or(T::default())),
                );
            } else {
                out.extend(acc_row.iter().map(|&v| {
                    NumCast::from(clamp_for_dtype(v as f64, T::DTYPE)).unwrap_or(T::default())
                }));
            }
        }

        ViewBuffer::from_vec_with_shape(out, shape.to_vec())
    })
}

#[cfg(not(feature = "image_interop"))]
pub(crate) fn apply_image_inner(_buf: ViewBuffer, _op: ImageOp) -> ViewBuffer {
    panic!("Image operations require the 'image_interop' feature");
}

// ============================================================
// Morphological Operations (Erode / Dilate / Gradient)
// ============================================================

/// Morphological erosion: output = local minimum over ksize×ksize neighborhood.
///
/// Uses replicate border handling (edge pixels are replicated).
/// Operates on single-channel data only; panics for multi-channel input.
/// Dtype-generic: dispatches per element type.
#[cfg(feature = "image_interop")]
fn apply_erode(buf: ViewBuffer, ksize: u32, iterations: u32) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Erode requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    let mut result = buf;
    for _ in 0..iterations {
        result = morph_minmax_pass(&result, ksize, MorphKind::Min);
    }
    result
}

/// Morphological dilation: output = local maximum over ksize×ksize neighborhood.
///
/// Uses replicate border handling (edge pixels are replicated).
/// Operates on single-channel data only; panics for multi-channel input.
/// Dtype-generic: dispatches per element type.
#[cfg(feature = "image_interop")]
fn apply_dilate(buf: ViewBuffer, ksize: u32, iterations: u32) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "Dilate requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    let mut result = buf;
    for _ in 0..iterations {
        result = morph_minmax_pass(&result, ksize, MorphKind::Max);
    }
    result
}

/// Morphological gradient: dilate(input) − erode(input), clamped to valid range.
///
/// Produces an edge outline by computing the difference between dilation and
/// erosion. Output dtype matches input.
#[cfg(feature = "image_interop")]
fn apply_morph_gradient(buf: ViewBuffer, ksize: u32) -> ViewBuffer {
    let shape = buf.shape();
    if !is_single_channel(shape) {
        let channels = get_channel_count(shape);
        panic!(
            "MorphGradient requires single-channel input, but got {channels} channels (shape: {shape:?}). \
             Consider using .grayscale() or .threshold() first."
        );
    }
    let dilated = morph_minmax_pass(&buf, ksize, MorphKind::Max);
    let eroded = morph_minmax_pass(&buf, ksize, MorphKind::Min);

    morph_subtract(&dilated, &eroded)
}

#[cfg(feature = "image_interop")]
#[derive(Clone, Copy)]
enum MorphKind {
    Min,
    Max,
}

/// Single-pass min or max filter over a ksize×ksize rectangular structuring element.
///
/// Uses a separable approach (row pass then column pass) for efficiency.
/// Border handling: replicate edge pixels.
#[cfg(feature = "image_interop")]
fn morph_minmax_pass(buf: &ViewBuffer, ksize: u32, kind: MorphKind) -> ViewBuffer {
    if ksize <= 1 {
        return buf.clone();
    }
    let dtype = buf.dtype();
    match dtype {
        DType::U8 => morph_dispatch::<u8>(buf, ksize, kind),
        DType::I8 => morph_dispatch::<i8>(buf, ksize, kind),
        DType::U16 => morph_dispatch::<u16>(buf, ksize, kind),
        DType::I16 => morph_dispatch::<i16>(buf, ksize, kind),
        DType::U32 => morph_dispatch::<u32>(buf, ksize, kind),
        DType::I32 => morph_dispatch::<i32>(buf, ksize, kind),
        DType::F32 => morph_dispatch::<f32>(buf, ksize, kind),
        DType::F64 => morph_dispatch::<f64>(buf, ksize, kind),
        DType::U64 => morph_dispatch::<u64>(buf, ksize, kind),
        DType::I64 => morph_dispatch::<i64>(buf, ksize, kind),
    }
}

/// Select the const-generic Min/Max instantiation for one element type.
#[cfg(feature = "image_interop")]
fn morph_dispatch<T>(buf: &ViewBuffer, ksize: u32, kind: MorphKind) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + Copy + PartialOrd,
{
    match kind {
        MorphKind::Min => morph_minmax_typed::<T, true>(buf, ksize),
        MorphKind::Max => morph_minmax_typed::<T, false>(buf, ksize),
    }
}

/// One Min/Max fold step with the comparison kind monomorphized out of the
/// loop.
///
/// The comparison expression is kept literally as `candidate < val` /
/// `candidate > val` (NOT `f32::min`/`max`): when `val` is NaN every
/// comparison is false so NaN is kept, and when `candidate` is NaN it is
/// ignored — the exact semantics of the original per-element implementation,
/// preserved for float dtypes (including signed-zero tie behavior, where the
/// incumbent `val` wins).
#[cfg(feature = "image_interop")]
#[inline(always)]
fn morph_select<T: PartialOrd + Copy, const IS_MIN: bool>(val: T, candidate: T) -> T {
    let take = if IS_MIN {
        candidate < val
    } else {
        candidate > val
    };
    if take {
        candidate
    } else {
        val
    }
}

/// Typed separable min/max filter: row pass then column pass.
///
/// Structured for auto-vectorization (interior/border split, same pattern as
/// the separable Gaussian blur above):
/// - row pass: the output row is initialized from the source row (the
///   original code's `val = center` seed), then each tap `kx ∈ [-r, r]` is a
///   contiguous shifted-slice elementwise min/max over the in-bounds segment,
///   with the clamped (replicate) border lanes folding the constant edge
///   pixel — identical comparison order to the original per-element loop;
/// - column pass: whole-row elementwise min/max against the clamped source
///   row per tap (the index clamp happens once per row, not per element).
///
/// Bit-exact vs the pre-split implementation (see tests/morph_ref.rs).
#[cfg(feature = "image_interop")]
fn morph_minmax_typed<T, const IS_MIN: bool>(buf: &ViewBuffer, ksize: u32) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + Copy + PartialOrd,
{
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let h = shape[0];
    let w = shape[1];
    let radius = (ksize / 2) as usize;
    let src: &[T] = contig.as_slice::<T>();

    // ── Row pass ─────────────────────────────────────────────────────────
    let mut row_out: Vec<T> = vec![T::default(); h * w];
    for y in 0..h {
        let row_in = &src[y * w..(y + 1) * w];
        let row_dst = &mut row_out[y * w..(y + 1) * w];

        if w <= 2 * radius {
            // Row narrower than the kernel: clamped gather everywhere.
            for (x, dst) in row_dst.iter_mut().enumerate() {
                let mut val = row_in[x];
                for kx in -(radius as i64)..=radius as i64 {
                    let sx = (x as i64 + kx).clamp(0, w as i64 - 1) as usize;
                    val = morph_select::<T, IS_MIN>(val, row_in[sx]);
                }
                *dst = val;
            }
            continue;
        }

        // Seed with the center pixel, then fold taps in ascending kx order.
        row_dst.copy_from_slice(row_in);
        for kx in -(radius as i64)..=radius as i64 {
            // In-bounds segment for this tap: x ∈ [lo, hi) ⇒ x+kx ∈ [0, w).
            let lo = (-kx).max(0) as usize;
            let hi = w - kx.max(0) as usize;
            let src_seg = &row_in[(lo as i64 + kx) as usize..(hi as i64 + kx) as usize];
            for (dst, &cand) in row_dst[lo..hi].iter_mut().zip(src_seg) {
                *dst = morph_select::<T, IS_MIN>(*dst, cand);
            }
            // Border lanes: the clamped index degenerates to the edge pixel.
            let left_edge = row_in[0];
            for dst in row_dst[..lo].iter_mut() {
                *dst = morph_select::<T, IS_MIN>(*dst, left_edge);
            }
            let right_edge = row_in[w - 1];
            for dst in row_dst[hi..].iter_mut() {
                *dst = morph_select::<T, IS_MIN>(*dst, right_edge);
            }
        }
    }

    // ── Column pass: whole-row folds with a per-row index clamp ──────────
    // Seeded from the row-pass output (the original code's `val = center`),
    // then folded in ascending ky order; src rows live in `row_out`, the
    // destination in `col_out`, so the borrows are disjoint.
    let mut col_out: Vec<T> = row_out.clone();
    for y in 0..h {
        let dst_row = &mut col_out[y * w..(y + 1) * w];
        for ky in -(radius as i64)..=radius as i64 {
            let sy = (y as i64 + ky).clamp(0, h as i64 - 1) as usize;
            let src_row = &row_out[sy * w..(sy + 1) * w];
            for (dst, &cand) in dst_row.iter_mut().zip(src_row) {
                *dst = morph_select::<T, IS_MIN>(*dst, cand);
            }
        }
    }

    ViewBuffer::from_vec_with_shape(col_out, shape.to_vec())
}

/// Element-wise saturating subtraction: result = a − b, clamped to valid range.
#[cfg(feature = "image_interop")]
fn morph_subtract(a: &ViewBuffer, b: &ViewBuffer) -> ViewBuffer {
    let dtype = a.dtype();
    let ca = a.to_contiguous();
    let cb = b.to_contiguous();
    let shape = ca.shape().to_vec();
    let count = ca.layout.num_elements();

    match dtype {
        DType::U8 => {
            let sa = unsafe { std::slice::from_raw_parts(ca.as_ptr::<u8>(), count) };
            let sb = unsafe { std::slice::from_raw_parts(cb.as_ptr::<u8>(), count) };
            let out: Vec<u8> = sa
                .iter()
                .zip(sb)
                .map(|(&a, &b)| a.saturating_sub(b))
                .collect();
            ViewBuffer::from_vec_with_shape(out, shape)
        }
        DType::U16 => {
            let sa = unsafe { std::slice::from_raw_parts(ca.as_ptr::<u16>(), count) };
            let sb = unsafe { std::slice::from_raw_parts(cb.as_ptr::<u16>(), count) };
            let out: Vec<u16> = sa
                .iter()
                .zip(sb)
                .map(|(&a, &b)| a.saturating_sub(b))
                .collect();
            ViewBuffer::from_vec_with_shape(out, shape)
        }
        DType::F32 => {
            let sa = unsafe { std::slice::from_raw_parts(ca.as_ptr::<f32>(), count) };
            let sb = unsafe { std::slice::from_raw_parts(cb.as_ptr::<f32>(), count) };
            let out: Vec<f32> = sa.iter().zip(sb).map(|(&a, &b)| (a - b).max(0.0)).collect();
            ViewBuffer::from_vec_with_shape(out, shape)
        }
        DType::F64 => {
            let sa = unsafe { std::slice::from_raw_parts(ca.as_ptr::<f64>(), count) };
            let sb = unsafe { std::slice::from_raw_parts(cb.as_ptr::<f64>(), count) };
            let out: Vec<f64> = sa.iter().zip(sb).map(|(&a, &b)| (a - b).max(0.0)).collect();
            ViewBuffer::from_vec_with_shape(out, shape)
        }
        _ => {
            // For other integer types, cast to f32, subtract, cast back
            let fa = a.cast(DType::F32);
            let fb = b.cast(DType::F32);
            let result = morph_subtract(&fa, &fb);
            result.cast(dtype)
        }
    }
}

// ============================================================
// Canny Edge Detection
// ============================================================

/// The 5×5 σ≈1.4 Gaussian used by Canny. NOT separable (center 15/159 differs
/// from the outer product of any 1-D kernel), so it runs through the
/// vectorized 2-D convolution rather than a two-pass separable blur.
#[cfg(feature = "image_interop")]
#[rustfmt::skip]
const CANNY_GAUSSIAN_5X5: [f32; 25] = [
    2.0/159.0,  4.0/159.0,  5.0/159.0,  4.0/159.0,  2.0/159.0,
    4.0/159.0,  9.0/159.0, 12.0/159.0,  9.0/159.0,  4.0/159.0,
    5.0/159.0, 12.0/159.0, 15.0/159.0, 12.0/159.0,  5.0/159.0,
    4.0/159.0,  9.0/159.0, 12.0/159.0,  9.0/159.0,  4.0/159.0,
    2.0/159.0,  4.0/159.0,  5.0/159.0,  4.0/159.0,  2.0/159.0,
];

/// Quantize a gradient direction into the 4 Canny bins without `atan2`.
///
/// Replaces `atan2(gy, gx).to_degrees()` binning (bin0 = horizontal
/// [0°,22.5°)∪[157.5°,180°), bin1 = 45° [22.5°,67.5°), bin2 = vertical
/// [67.5°,112.5°), bin3 = 135° [112.5°,157.5°)) with sign/ratio comparisons
/// against tan(22.5°) = √2−1 and tan(67.5°) = √2+1.
///
/// Boundary inclusivity differs by sign case because the fold θ = 180°−θ'
/// flips which side of each boundary is inclusive:
/// - same sign (θ = θ'): bin0 ⇔ ay <  t22·ax; bin2 ⇔ ay ≥ t67·ax; else bin1
/// - opposite     (θ = 180−θ'): bin0 ⇔ ay ≤ t22·ax; bin2 ⇔ ay > t67·ax; else bin3
///
/// `gx == 0 && gy == 0` is special-cased to bin0 (atan2(0,0) = 0). Agreement
/// with the atan2 quantizer is exact except where floating-point rounding
/// puts the computed angle within ~1 ulp of a bin boundary, where the old
/// answer was itself rounding-determined (see tests/canny_ref.rs grid test).
#[cfg(feature = "image_interop")]
#[inline(always)]
fn canny_direction(gx: f32, gy: f32) -> u8 {
    const T22: f32 = 0.41421356; // tan(22.5°) = √2 − 1
    const T67: f32 = 2.4142137; // tan(67.5°) = √2 + 1

    if gx == 0.0 && gy == 0.0 {
        return 0;
    }
    let ax = gx.abs();
    let ay = gy.abs();
    let same_sign = (gx >= 0.0) == (gy >= 0.0);
    let t22ax = T22 * ax;
    let t67ax = T67 * ax;
    if same_sign {
        if ay < t22ax {
            0
        } else if ay >= t67ax {
            2
        } else {
            1
        }
    } else if ay <= t22ax {
        0
    } else if ay > t67ax {
        2
    } else {
        3
    }
}

/// Canny edge detection: Gaussian blur → Sobel gradients → NMS → double-threshold hysteresis.
///
/// Operates on a single-channel image. For multi-channel input, converts to
/// grayscale first. Output is always U8 (0 or 255).
///
/// Optimized from the original naive implementation (preserved in
/// tests/canny_ref.rs) while keeping identical output:
/// - the 5×5 blur runs through the vectorized `apply_convolve2d` (same tap
///   order and replicate border ⇒ bit-exact, ~10–20× faster);
/// - direction quantization avoids the per-pixel `atan2` (see
///   [`canny_direction`]);
/// - magnitude keeps `sqrt` (squared-threshold comparisons would not be
///   bit-exact: f32 sqrt rounding can flip the tie-inclusive `>=` NMS
///   comparisons);
/// - double-thresholding is folded into the NMS loop (the `nms` plane is
///   never materialized; non-maxima and border pixels classify the implicit
///   0.0 exactly as before — including the degenerate `high ≤ 0` case where
///   borders become STRONG);
/// - hysteresis is a single-pass worklist flood fill from STRONG seeds
///   instead of whole-image sweeps to a fixpoint (identical transitive
///   closure: a WEAK interior pixel is promoted iff it is 8-connected to a
///   STRONG pixel through WEAK interior pixels).
#[cfg(feature = "image_interop")]
fn apply_canny(buf: ViewBuffer, low_threshold: f32, high_threshold: f32) -> ViewBuffer {
    use crate::ops::filter::{apply_convolve2d, BorderMode, ConvolveOp};

    let shape = buf.shape();
    let channels = shape.get(2).copied().unwrap_or(1);

    // Convert to single-channel grayscale f32
    let gray = if channels > 1 {
        let gs = grayscale_strided(buf);
        if gs.dtype() != DType::F32 {
            gs.cast(DType::F32)
        } else {
            gs
        }
    } else if buf.dtype() != DType::F32 {
        buf.cast(DType::F32)
    } else {
        buf.clone()
    };
    let contig = gray.to_contiguous();
    let gray_shape = contig.shape();
    let gh = gray_shape[0];
    let gw = gray_shape[1];
    let count = gh * gw;

    // Step 1: Gaussian blur (5×5, sigma ≈ 1.4) via the vectorized 2-D
    // convolution. Replicate border matches the original's clamped gather.
    let blurred_buf = apply_convolve2d(
        &contig,
        &ConvolveOp {
            kernel: CANNY_GAUSSIAN_5X5.to_vec(),
            ksize: 5,
            normalize: false,
            border: BorderMode::Replicate,
        },
    );
    let blurred = blurred_buf.as_slice::<f32>();

    // Step 2: Sobel gradients
    let (gx, gy) = sobel_gradients(blurred, gh, gw);

    // Step 3: Magnitude and direction (no atan2)
    let mut magnitude = vec![0.0f32; count];
    let mut direction = vec![0u8; count]; // quantized to 4 directions (0,1,2,3)
    for i in 0..count {
        magnitude[i] = (gx[i] * gx[i] + gy[i] * gy[i]).sqrt();
        direction[i] = canny_direction(gx[i], gy[i]);
    }

    const STRONG: u8 = 255;
    const WEAK: u8 = 128;

    // Classification of a suppressed (0.0) magnitude — applies to border
    // pixels and non-maxima, exactly like the original's zero-filled `nms`
    // plane fed through the threshold loop.
    let zero_class = if 0.0 >= high_threshold {
        STRONG
    } else if 0.0 >= low_threshold {
        WEAK
    } else {
        0
    };

    // Steps 4+5 fused: non-maximum suppression classifying directly into the
    // edge map (same `>=` comparisons; the nms plane is never materialized).
    let mut edges = vec![zero_class; count];
    for y in 1..gh.saturating_sub(1) {
        for x in 1..gw - 1 {
            let idx = y * gw + x;
            let mag = magnitude[idx];
            let (n1, n2) = match direction[idx] {
                0 => (magnitude[idx - 1], magnitude[idx + 1]), // horizontal: left, right
                1 => (
                    magnitude[(y - 1) * gw + x + 1],
                    magnitude[(y + 1) * gw + x - 1],
                ), // 45°
                2 => (magnitude[(y - 1) * gw + x], magnitude[(y + 1) * gw + x]), // vertical
                _ => (
                    magnitude[(y - 1) * gw + x - 1],
                    magnitude[(y + 1) * gw + x + 1],
                ), // 135°
            };
            if mag >= n1 && mag >= n2 {
                edges[idx] = if mag >= high_threshold {
                    STRONG
                } else if mag >= low_threshold {
                    WEAK
                } else {
                    0
                };
            }
        }
    }

    // Step 6: Hysteresis — single-pass worklist flood fill from STRONG seeds.
    // Only WEAK pixels at interior coordinates are ever promoted (the
    // original sweep iterated `1..h-1 × 1..w-1`); STRONG border pixels still
    // act as seeds, as they did as neighbors in the original sweep.
    if gh > 2 && gw > 2 {
        let mut stack: Vec<u32> = edges
            .iter()
            .enumerate()
            .filter(|(_, &e)| e == STRONG)
            .map(|(i, _)| i as u32)
            .collect();
        while let Some(idx) = stack.pop() {
            let idx = idx as usize;
            let y = idx / gw;
            let x = idx % gw;
            let y0 = y.saturating_sub(1).max(1);
            let y1 = (y + 1).min(gh - 2);
            let x0 = x.saturating_sub(1).max(1);
            let x1 = (x + 1).min(gw - 2);
            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let nidx = ny * gw + nx;
                    if edges[nidx] == WEAK {
                        edges[nidx] = STRONG;
                        stack.push(nidx as u32);
                    }
                }
            }
        }
    }
    // Suppress remaining weak edges
    for e in &mut edges {
        if *e == WEAK {
            *e = 0;
        }
    }

    ViewBuffer::from_vec_with_shape(edges, vec![gh, gw, 1])
}

/// Compute Sobel gradients (Gx, Gy) for single-channel image.
#[cfg(feature = "image_interop")]
fn sobel_gradients(src: &[f32], h: usize, w: usize) -> (Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; h * w];
    let mut gy = vec![0.0f32; h * w];

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let tl = src[(y - 1) * w + x - 1];
            let tc = src[(y - 1) * w + x];
            let tr = src[(y - 1) * w + x + 1];
            let ml = src[y * w + x - 1];
            let mr = src[y * w + x + 1];
            let bl = src[(y + 1) * w + x - 1];
            let bc = src[(y + 1) * w + x];
            let br = src[(y + 1) * w + x + 1];

            gx[y * w + x] = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
            gy[y * w + x] = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
        }
    }
    (gx, gy)
}

// ============================================================
// Histogram Equalization
// ============================================================

/// Histogram equalization for contrast enhancement.
///
/// Computes the histogram, cumulative distribution, then maps each pixel
/// through the normalized CDF. Operates per-channel for multi-channel images.
/// Input must be U8 (enforced by `working_dtype`). Output is U8.
#[cfg(feature = "image_interop")]
fn apply_histogram_equalize(buf: ViewBuffer) -> ViewBuffer {
    let contig = buf.to_contiguous();
    let shape = contig.shape();
    let h = shape[0];
    let w = shape[1];
    let c = shape.get(2).copied().unwrap_or(1);
    let count = contig.layout.num_elements();
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<u8>(), count) };

    let total_pixels = h * w;
    let mut output = vec![0u8; count];

    for ch in 0..c {
        // Compute histogram for this channel
        let mut hist = [0u32; 256];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * c + ch;
                hist[src[idx] as usize] += 1;
            }
        }

        // Compute CDF
        let mut cdf = [0u32; 256];
        cdf[0] = hist[0];
        for i in 1..256 {
            cdf[i] = cdf[i - 1] + hist[i];
        }

        // Find minimum non-zero CDF value
        let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);

        // Map pixels through equalized CDF
        let denominator = (total_pixels as f64 - cdf_min as f64).max(1.0);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * c + ch;
                let val = src[idx] as usize;
                let equalized =
                    ((cdf[val] as f64 - cdf_min as f64) / denominator * 255.0).round() as u8;
                output[idx] = equalized;
            }
        }
    }

    if c == 1 && shape.len() == 2 {
        ViewBuffer::from_vec_with_shape(output, vec![h, w])
    } else {
        ViewBuffer::from_vec_with_shape(output, vec![h, w, c])
    }
}

// ============================================================
// Perceptual Hash Operations
// ============================================================

#[cfg(feature = "perceptual_hash")]
use crate::ops::phash::PerceptualHashOp;

/// Applies a perceptual hash operation to a buffer.
///
/// Perceptual hashing requires the buffer to be in image format.
/// The output is a 1D u8 buffer containing the hash bytes.
#[cfg(feature = "perceptual_hash")]
pub fn apply_perceptual_hash(buf: ViewBuffer, op: PerceptualHashOp) -> ViewBuffer {
    // Convert to U8 if needed (perceptual hash expects image format)
    let work_buf = convert_to_u8_for_image(buf);

    // Execute the perceptual hash operation
    op.execute(&work_buf)
}

#[cfg(not(feature = "perceptual_hash"))]
pub fn apply_perceptual_hash(
    _buf: ViewBuffer,
    _op: crate::ops::phash::PerceptualHashOp,
) -> ViewBuffer {
    panic!("Perceptual hash operations require the 'perceptual_hash' feature");
}

#[cfg(all(test, feature = "image_interop"))]
mod canny_direction_tests {
    use super::canny_direction;

    /// The original atan2-based quantizer, kept verbatim as the reference.
    fn atan2_direction(gx: f32, gy: f32) -> u8 {
        let angle = gy.atan2(gx).to_degrees();
        let angle = if angle < 0.0 { angle + 180.0 } else { angle };
        if !(22.5..157.5).contains(&angle) {
            0
        } else if angle < 67.5 {
            1
        } else if angle < 112.5 {
            2
        } else {
            3
        }
    }

    /// Degrees-distance from the nearest direction-bin boundary, in f64.
    fn boundary_distance_deg(gx: f32, gy: f32) -> f64 {
        let angle = (gy as f64).atan2(gx as f64).to_degrees();
        let angle = if angle < 0.0 { angle + 180.0 } else { angle };
        [22.5f64, 67.5, 112.5, 157.5]
            .iter()
            .map(|b| (angle - b).abs())
            .fold(f64::INFINITY, f64::min)
    }

    fn check(gx: f32, gy: f32, divergences: &mut Vec<(f32, f32, u8, u8)>) {
        let got = canny_direction(gx, gy);
        let want = atan2_direction(gx, gy);
        if got != want {
            // Divergence is only acceptable where the angle sits within
            // rounding distance of a bin boundary, where the atan2 answer
            // was itself rounding-determined.
            assert!(
                boundary_distance_deg(gx, gy) < 1e-4,
                "direction mismatch away from a bin boundary: \
                 gx={gx:?} gy={gy:?} got={got} want={want}"
            );
            divergences.push((gx, gy, got, want));
        }
    }

    #[test]
    fn direction_quantizer_matches_atan2_reference() {
        let mut divergences = Vec::new();

        // Exhaustive grid (65×65 including signed values and zeros).
        let mut v = -8.0f32;
        let mut grid = Vec::new();
        while v <= 8.0 {
            grid.push(v);
            v += 0.25;
        }
        for &gx in &grid {
            for &gy in &grid {
                check(gx, gy, &mut divergences);
            }
        }

        // Constructed boundary ratios, exact diagonals, axes, signed zeros.
        const T22: f32 = 0.41421356;
        const T67: f32 = 2.4142137;
        for a in [0.5f32, 1.0, 3.0, 100.0, 1e-3] {
            for (gx, gy) in [
                (a, T22 * a),
                (a, -(T22 * a)),
                (-a, T22 * a),
                (-a, -(T22 * a)),
                (a, T67 * a),
                (a, -(T67 * a)),
                (-a, T67 * a),
                (-a, -(T67 * a)),
                (a, a),
                (a, -a),
                (-a, a),
                (-a, -a),
                (a, 0.0),
                (-a, 0.0),
                (0.0, a),
                (0.0, -a),
                (a, -0.0),
                (-0.0, a),
            ] {
                check(gx, gy, &mut divergences);
            }
        }
        check(0.0, 0.0, &mut divergences);
        check(-0.0, 0.0, &mut divergences);
        check(0.0, -0.0, &mut divergences);
        check(-0.0, -0.0, &mut divergences);

        // Bound the rounding-determined divergence set: the grid contains no
        // exact-boundary ratios, so only the constructed boundary points may
        // diverge (and only within the 1e-4° band asserted above).
        assert!(
            divergences.len() <= 40,
            "too many boundary divergences: {divergences:?}"
        );
    }
}
