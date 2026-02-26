//! Execution runners for applying operations.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::execution::tiling::{get_tile_config, is_tiling_enabled, maybe_tiled};
use crate::expr::ViewExpr;
use crate::ops::dto::ViewDto;
use crate::ops::traits::Op;
use crate::ops::{ComputeOp, ImageOp, ViewOp};

#[cfg(feature = "image_interop")]
use crate::core::layout::ExternalLayout;
#[cfg(feature = "image_interop")]
use crate::ops::{FilterType, ImageOpKind};

#[cfg(feature = "image_interop")]
use crate::interop::image::AsImageView;

#[cfg(feature = "ndarray_interop")]
use crate::interop::ndarray::{AsNdarray, FromNdarray};

#[cfg(feature = "image_interop")]
use image::imageops;
#[cfg(feature = "image_interop")]
use image::{ImageBuffer, Luma, Rgb};

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
///
/// If tiling is enabled (via environment variable or [`with_tile_config`]),
/// tileable operations will be executed tile-by-tile for improved cache efficiency.
#[inline]
pub fn apply_compute(buf: ViewBuffer, op: ComputeOp) -> ViewBuffer {
    // Fast path: atomic check avoids TLS access when tiling is disabled
    if !is_tiling_enabled() {
        return apply_compute_inner(buf, op);
    }

    // Slow path: tiling might be enabled, check TLS and policy
    let tile_config = get_tile_config();
    if let Some(ref config) = tile_config {
        let policy = op.tile_policy();
        if policy.is_tileable() {
            let halo = policy.halo();
            let op_clone = op.clone();
            return maybe_tiled(buf, halo, Some(config), move |tile| {
                apply_compute_inner(tile, op_clone.clone())
            });
        }
    }

    apply_compute_inner(buf, op)
}

/// Inner implementation of compute operations (without tiling logic).
#[inline]
fn apply_compute_inner(buf: ViewBuffer, op: ComputeOp) -> ViewBuffer {
    match op {
        ComputeOp::Cast(dtype) => buf.cast(dtype),
        ComputeOp::Affine(_params) => unimplemented!("Affine transform compute"),
        ComputeOp::Scale(factor) => apply_scalar_op(&buf, |x: f32| x * factor),
        ComputeOp::Relu => apply_scalar_op(&buf, |x: f32| if x > 0.0 { x } else { 0.0 }),
        ComputeOp::Fused(ref kernel) => buf.apply_fused_kernel(kernel),
        ComputeOp::Normalize(ref method) => apply_normalize(&buf, method),
        ComputeOp::Clamp { min, max } => apply_scalar_op(&buf, move |x: f32| x.clamp(min, max)),
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
/// Output is always f32.
fn apply_adjust_gamma(buf: &ViewBuffer, gamma: f32) -> ViewBuffer {
    let input_dtype = buf.dtype();
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

/// Apply a scalar operation element-wise, accepting any numeric input type.
///
/// This function automatically casts the input to f32 for computation,
/// as per the dtype promotion contract. The output is always f32.
///
/// This follows the pattern used by NumPy, PyTorch, and other numeric libraries:
/// - Accept any numeric input dtype
/// - Perform computation in f32 for numerical stability
/// - Return f32 (can be cast to desired output type afterward)
fn apply_scalar_op<F>(buf: &ViewBuffer, op: F) -> ViewBuffer
where
    F: Fn(f32) -> f32,
{
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
            let result_array = view.mapv(&op);
            return ViewBuffer::from_array(result_array);
        }
    }

    // Fallback: use contiguous buffer
    let contig = work_buf.to_contiguous();
    let count = contig.layout.num_elements();
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };
    let new_data: Vec<f32> = src.iter().map(|&x| op(x)).collect();
    ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec())
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

    let mut resizer = fir::Resizer::new();
    resizer
        .resize(
            &src_image,
            &mut dst_image,
            &fir::ResizeOptions::new().resize_alg(fir_filter),
        )
        .expect("Resize failed");

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

    let mut resizer = fir::Resizer::new();
    resizer
        .resize(
            &src_image,
            &mut dst_image,
            &fir::ResizeOptions::new().resize_alg(fir_filter),
        )
        .expect("Resize failed");

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

    let mut resizer = fir::Resizer::new();
    resizer
        .resize(
            &src_image,
            &mut dst_image,
            &fir::ResizeOptions::new().resize_alg(fir_filter),
        )
        .expect("Resize failed");

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

    // Fast path: contiguous RGB buffer with standard layout
    if buf.layout.is_contiguous() && channels == 3 {
        let data = unsafe { std::slice::from_raw_parts(buf.as_ptr::<u8>(), h * w * 3) };
        let mut gray_data: Vec<u8> = Vec::with_capacity(h * w);

        for pixel in data.chunks_exact(3) {
            let r = pixel[0] as u32;
            let g = pixel[1] as u32;
            let b = pixel[2] as u32;
            // BT.601 fixed-point
            let gray = ((77 * r + 150 * g + 29 * b + 128) >> 8).min(255) as u8;
            gray_data.push(gray);
        }

        return ViewBuffer::from_vec(gray_data).reshape(vec![h, w, 1]);
    }

    // Strided path: handles non-contiguous buffers (crop, flip, etc.)
    let (stride_h, stride_w, stride_c) =
        (strides[0], strides[1], strides.get(2).copied().unwrap_or(1));
    let base_ptr = unsafe { buf.as_ptr::<u8>() };

    let mut gray_data: Vec<u8> = Vec::with_capacity(h * w);

    for y in 0..h {
        for x in 0..w {
            let pixel_offset = y as isize * stride_h + x as isize * stride_w;
            unsafe {
                let pixel_ptr = base_ptr.offset(pixel_offset);
                let r = *pixel_ptr as u32;
                let g = *pixel_ptr.offset(stride_c) as u32;
                let b = *pixel_ptr.offset(2 * stride_c) as u32;
                // BT.601 fixed-point
                let gray = ((77 * r + 150 * g + 29 * b + 128) >> 8).min(255) as u8;
                gray_data.push(gray);
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
/// If tiling is enabled (via environment variable or [`with_tile_config`]),
/// tileable operations will be executed tile-by-tile for improved cache efficiency.
#[cfg(feature = "image_interop")]
#[inline]
pub fn apply_image(buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    let input_dtype = buf.dtype();

    // Only convert to U8 when the operation requires it.
    let work_buf = match op.working_dtype() {
        Some(DType::U8) => convert_to_u8_for_image(buf),
        Some(target) => buf.cast(target),
        None => buf, // Resize: use the input's native dtype
    };

    // Execute (tiled or not)
    let result = if !is_tiling_enabled() {
        apply_image_inner(work_buf, op.clone())
    } else {
        let tile_config = get_tile_config();
        if let Some(ref config) = tile_config {
            let policy = op.tile_policy();
            if policy.is_tileable() {
                let halo = policy.halo();
                let op_clone = op.clone();
                maybe_tiled(work_buf, halo, Some(config), move |tile| {
                    apply_image_inner(tile, op_clone.clone())
                })
            } else {
                apply_image_inner(work_buf, op.clone())
            }
        } else {
            apply_image_inner(work_buf, op.clone())
        }
    };

    // Runtime contract validation: the produced dtype must match the
    // operation's declared output_dtype_rule for the given input dtype.
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

/// Rotate buffer by arbitrary angle using bilinear interpolation.
///
/// Dtype-generic: works on any numeric type via `f64` interpolation space.
/// Supports strided input buffers and handles both expand and non-expand modes.
/// When expand=false, the output has the same dimensions as input (corners may be cropped).
/// When expand=true, the output dimensions are calculated to fit the rotated image.
#[cfg(feature = "image_interop")]
fn rotate_arbitrary(buf: ViewBuffer, angle: f32, expand: bool) -> ViewBuffer {
    let shape = buf.shape();
    if shape.len() < 2 {
        return buf; // Can't rotate 1D or 0D
    }

    let dtype = buf.dtype();

    match dtype {
        DType::U8 => rotate_typed::<u8>(buf, angle, expand),
        DType::I8 => rotate_typed::<i8>(buf, angle, expand),
        DType::U16 => rotate_typed::<u16>(buf, angle, expand),
        DType::I16 => rotate_typed::<i16>(buf, angle, expand),
        DType::U32 => rotate_typed::<u32>(buf, angle, expand),
        DType::I32 => rotate_typed::<i32>(buf, angle, expand),
        DType::F32 => rotate_typed::<f32>(buf, angle, expand),
        DType::F64 => rotate_typed::<f64>(buf, angle, expand),
        DType::U64 => rotate_typed::<u64>(buf, angle, expand),
        DType::I64 => rotate_typed::<i64>(buf, angle, expand),
    }
}

/// Typed rotate helper: performs bilinear interpolation in `f64` arithmetic,
/// reading and writing elements of type `T`.
///
/// For integer types the interpolated value is clamped to the representable
/// range before casting back to `T`.
#[cfg(feature = "image_interop")]
fn rotate_typed<T>(buf: ViewBuffer, angle: f32, expand: bool) -> ViewBuffer
where
    T: crate::core::dtype::ViewType + Default + num_traits::NumCast,
{
    use num_traits::NumCast;

    let shape = buf.shape();
    let h = shape[0] as f64;
    let w = shape[1] as f64;
    let channels = shape.get(2).copied().unwrap_or(1);
    let shape_vec = shape.to_vec();

    let angle_rad = (angle as f64).to_radians();
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();

    // Calculate output dimensions
    let (out_h, out_w) = if expand {
        let new_h = (h * cos_a.abs() + w * sin_a.abs()).ceil() as usize;
        let new_w = (h * sin_a.abs() + w * cos_a.abs()).ceil() as usize;
        (new_h, new_w)
    } else {
        (shape_vec[0], shape_vec[1])
    };

    // Center points
    let center_x_in = (w - 1.0) * 0.5;
    let center_y_in = (h - 1.0) * 0.5;
    let center_x_out = (out_w as f64 - 1.0) * 0.5;
    let center_y_out = (out_h as f64 - 1.0) * 0.5;

    // Ensure contiguous input for efficient access
    let contig_buf = if buf.layout.is_contiguous() {
        buf
    } else {
        buf.to_contiguous()
    };

    let src_data: &[T] = contig_buf.as_slice::<T>();

    // Allocate output buffer
    let output_size = out_h * out_w * channels;
    let mut dst_data: Vec<T> = vec![T::default(); output_size];

    // For floats, we skip clamping entirely (no range limit needed).
    let is_float = matches!(T::DTYPE, DType::F32 | DType::F64);

    // Inverse rotation: for each output pixel, find source pixel
    for y_out in 0..out_h {
        for x_out in 0..out_w {
            let x_rel = x_out as f64 - center_x_out;
            let y_rel = y_out as f64 - center_y_out;

            // Apply inverse rotation (counter-clockwise to get source)
            let x_src = x_rel * cos_a + y_rel * sin_a + center_x_in;
            let y_src = -x_rel * sin_a + y_rel * cos_a + center_y_in;

            // Bilinear interpolation coordinates
            let x0 = x_src.floor() as i64;
            let y0 = y_src.floor() as i64;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let dx = x_src - x0 as f64;
            let dy = y_src - y0 as f64;

            // Check bounds — out of bounds pixels are zero-filled (default)
            if x0 < 0 || y0 < 0 || x1 >= w as i64 || y1 >= h as i64 {
                // dst_data is already zero-initialized via T::default()
                continue;
            }

            // Get four corner pixels
            let get_pixel = |x: i64, y: i64| -> &[T] {
                let idx = (y as usize * shape_vec[1] + x as usize) * channels;
                &src_data[idx..idx + channels]
            };

            let p00 = get_pixel(x0, y0);
            let p10 = get_pixel(x1, y0);
            let p01 = get_pixel(x0, y1);
            let p11 = get_pixel(x1, y1);

            // Bilinear interpolation per channel in f64 space
            for c in 0..channels {
                let v00: f64 = NumCast::from(p00[c]).unwrap_or(0.0);
                let v10: f64 = NumCast::from(p10[c]).unwrap_or(0.0);
                let v01: f64 = NumCast::from(p01[c]).unwrap_or(0.0);
                let v11: f64 = NumCast::from(p11[c]).unwrap_or(0.0);

                let v0 = v00 * (1.0 - dx) + v10 * dx;
                let v1 = v01 * (1.0 - dx) + v11 * dx;
                let v = v0 * (1.0 - dy) + v1 * dy;

                // For integer types, clamp to valid range before casting back
                let clamped = if is_float {
                    v
                } else {
                    clamp_for_dtype(v, T::DTYPE)
                };

                dst_data[(y_out * out_w + x_out) * channels + c] =
                    NumCast::from(clamped).unwrap_or(T::default());
            }
        }
    }

    // Build output shape
    let output_shape = if channels == 1 {
        vec![out_h, out_w]
    } else {
        vec![out_h, out_w, channels]
    };

    ViewBuffer::from_vec(dst_data).reshape(output_shape)
}

/// Clamp an `f64` value to the representable range of the given integer `DType`.
#[cfg(feature = "image_interop")]
#[inline]
fn clamp_for_dtype(v: f64, dtype: DType) -> f64 {
    match dtype {
        DType::U8 => v.clamp(0.0, u8::MAX as f64),
        DType::I8 => v.clamp(i8::MIN as f64, i8::MAX as f64),
        DType::U16 => v.clamp(0.0, u16::MAX as f64),
        DType::I16 => v.clamp(i16::MIN as f64, i16::MAX as f64),
        DType::U32 => v.clamp(0.0, u32::MAX as f64),
        DType::I32 => v.clamp(i32::MIN as f64, i32::MAX as f64),
        DType::U64 => v.clamp(0.0, u64::MAX as f64),
        DType::I64 => v.clamp(i64::MIN as f64, i64::MAX as f64),
        DType::F32 | DType::F64 => v, // No clamping for floats
    }
}

/// Inner implementation of image operations (without tiling logic).
#[cfg(feature = "image_interop")]
#[inline]
fn apply_image_inner(work_buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    match op.kind {
        ImageOpKind::Threshold(thresh) => threshold_generic(work_buf, thresh),
        ImageOpKind::Grayscale => grayscale_strided(work_buf),
        ImageOpKind::Resize {
            width,
            height,
            filter,
        } => resize_strided(work_buf, width, height, filter),
        ImageOpKind::Rotate { angle, expand } => rotate_arbitrary(work_buf, angle, expand),
        ImageOpKind::Blur { sigma } => {
            let contig_buf = if work_buf.is_compatible_with(ExternalLayout::ImageCrate)
                && work_buf.layout.is_contiguous()
            {
                work_buf
            } else {
                work_buf.to_contiguous()
            };

            let shape = contig_buf.shape();
            let (h, w, c) = (
                shape[0] as u32,
                shape[1] as u32,
                *shape.get(2).unwrap_or(&1) as u32,
            );
            let count = contig_buf.layout.num_elements();
            let raw_vec =
                unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u8>(), count).to_vec() };

            if c == 3 {
                let img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(w, h, raw_vec).unwrap();
                let blurred = imageops::blur(&img_buf, sigma);
                ViewBuffer::from_vec(blurred.into_raw()).reshape(vec![h as usize, w as usize, 3])
            } else {
                let img_buf: ImageBuffer<Luma<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(w, h, raw_vec).unwrap();
                let blurred = imageops::blur(&img_buf, sigma);
                ViewBuffer::from_vec(blurred.into_raw()).reshape(vec![h as usize, w as usize, 1])
            }
        }
    }
}

#[cfg(not(feature = "image_interop"))]
pub fn apply_image(_buf: ViewBuffer, _op: ImageOp) -> ViewBuffer {
    panic!("Image operations require the 'image_interop' feature");
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
