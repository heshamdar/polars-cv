//! Execution runners for applying operations.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::expr::ViewExpr;
use crate::ops::dto::ViewDto;
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
    }
}

/// Applies a compute operation to a buffer.
pub fn apply_compute(buf: ViewBuffer, op: ComputeOp) -> ViewBuffer {
    match op {
        ComputeOp::Cast(dtype) => buf.cast(dtype),
        ComputeOp::Affine(_params) => unimplemented!("Affine transform compute"),
        ComputeOp::Scale(factor) => apply_scalar_op(&buf, |x: f32| x * factor),
        ComputeOp::Relu => apply_scalar_op(&buf, |x: f32| if x > 0.0 { x } else { 0.0 }),
        ComputeOp::Fused(ref kernel) => buf.apply_fused_kernel(kernel),
        ComputeOp::Normalize(ref method) => apply_normalize(&buf, method),
        ComputeOp::Clamp { min, max } => apply_scalar_op(&buf, move |x: f32| x.clamp(min, max)),
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

    let contig = work_buf.to_contiguous();
    let count = contig.layout.num_elements();
    let shape = contig.shape();
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };

    let new_data: Vec<f32> = match method {
        NormalizeMethod::MinMax => {
            let min = src.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            if range == 0.0 {
                // Edge case: constant array - return 0.0 for all elements
                // This avoids division by zero and gives a predictable result
                vec![0.0; count]
            } else {
                src.iter().map(|&x| (x - min) / range).collect()
            }
        }
        NormalizeMethod::ZScore => {
            let n = count as f32;
            let mean = src.iter().sum::<f32>() / n;
            let variance = src.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
            let std = variance.sqrt();
            if std == 0.0 {
                // Edge case: constant array - return 0.0 for all elements
                vec![0.0; count]
            } else {
                src.iter().map(|&x| (x - mean) / std).collect()
            }
        }
        NormalizeMethod::Preset { mean, std } => {
            // Channel-wise normalization with preset values
            // Assumes HWC layout where last dimension is channels
            let channels = if shape.len() == 3 { shape[2] } else { 1 };

            // Validate channel count matches mean/std length
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

            // Apply (x - mean[c]) / std[c] for each element
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

/// Applies an image operation to a buffer.
///
/// Image operations accept any numeric input dtype and automatically convert
/// to U8 as needed. For float inputs in [0.0, 1.0], values are scaled to [0, 255].
#[cfg(feature = "image_interop")]
pub fn apply_image(buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    // Convert to U8 if needed (dtype promotion for image ops)
    let work_buf = convert_to_u8_for_image(buf);

    match op.kind {
        ImageOpKind::Threshold(thresh) => {
            if let Ok(view) = work_buf.as_image_view::<Luma<u8>>() {
                let mut new_data: Vec<u8> = Vec::with_capacity((view.width * view.height) as usize);
                for y in 0..view.height {
                    let row_start = (y as usize) * view.row_stride;
                    let row_slice = &view.data[row_start..row_start + view.width as usize];
                    for &p in row_slice {
                        new_data.push(if p > thresh { 255 } else { 0 });
                    }
                }
                ViewBuffer::from_vec(new_data).reshape(vec![
                    view.height as usize,
                    view.width as usize,
                    1,
                ])
            } else {
                let contig_buf = if work_buf.layout.is_contiguous() {
                    work_buf
                } else {
                    work_buf.to_contiguous()
                };
                let count = contig_buf.layout.num_elements();
                let src_slice =
                    unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u8>(), count) };

                let new_data: Vec<u8> = src_slice
                    .iter()
                    .map(|&p| if p > thresh { 255 } else { 0 })
                    .collect();

                ViewBuffer::from_vec(new_data).reshape(contig_buf.shape().to_vec())
            }
        }
        ImageOpKind::Grayscale => {
            let shape = work_buf.shape();
            let channels = *shape.get(2).unwrap_or(&1);

            if channels == 1 {
                // Already grayscale, just ensure it's U8
                return work_buf;
            }

            let contig_buf = if work_buf.is_compatible_with(ExternalLayout::ImageCrate)
                && work_buf.layout.is_contiguous()
            {
                work_buf
            } else {
                work_buf.to_contiguous()
            };

            let (h, w) = (
                contig_buf.shape()[0] as usize,
                contig_buf.shape()[1] as usize,
            );
            let count = contig_buf.layout.num_elements();
            let raw_slice = unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u8>(), count) };

            // Use BT.601 coefficients (same as OpenCV, Pillow, etc.)
            // Y = 0.299*R + 0.587*G + 0.114*B
            // Using fixed-point math for speed: Y = (77*R + 150*G + 29*B) >> 8
            let mut gray_data: Vec<u8> = Vec::with_capacity(h * w);
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) * channels;
                    let r = raw_slice[idx] as u32;
                    let g = raw_slice[idx + 1] as u32;
                    let b = raw_slice[idx + 2] as u32;
                    // BT.601: 0.299*R + 0.587*G + 0.114*B
                    // Fixed-point: (77*R + 150*G + 29*B + 128) >> 8
                    let gray = ((77 * r + 150 * g + 29 * b + 128) >> 8).min(255) as u8;
                    gray_data.push(gray);
                }
            }
            ViewBuffer::from_vec(gray_data).reshape(vec![h, w, 1])
        }
        ImageOpKind::Resize {
            width,
            height,
            filter,
        } => {
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

            let img_filter = match filter {
                FilterType::Nearest => imageops::FilterType::Nearest,
                FilterType::Triangle => imageops::FilterType::Triangle,
                FilterType::CatmullRom => imageops::FilterType::CatmullRom,
                FilterType::Gaussian => imageops::FilterType::Gaussian,
                FilterType::Lanczos3 => imageops::FilterType::Lanczos3,
            };

            let count = contig_buf.layout.num_elements();
            let raw_vec =
                unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u8>(), count).to_vec() };

            if c == 3 {
                let img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(w, h, raw_vec).expect("Failed to create ImageBuffer");
                let resized = imageops::resize(&img_buf, width, height, img_filter);
                ViewBuffer::from_vec(resized.into_raw()).reshape(vec![
                    height as usize,
                    width as usize,
                    3,
                ])
            } else if c == 1 {
                let img_buf: ImageBuffer<Luma<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(w, h, raw_vec).expect("Failed to create ImageBuffer");
                let resized = imageops::resize(&img_buf, width, height, img_filter);
                ViewBuffer::from_vec(resized.into_raw()).reshape(vec![
                    height as usize,
                    width as usize,
                    1,
                ])
            } else {
                panic!("Resize only supports 1 or 3 channels");
            }
        }
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
