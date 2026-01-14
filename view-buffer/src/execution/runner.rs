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

/// Resize using fast_image_resize with SIMD optimization.
///
/// Uses SIMD-optimized resize algorithms for high performance.
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

    let shape = contig_buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let c = shape.get(2).copied().unwrap_or(1);

    // Map our filter types to fast_image_resize types
    let fir_filter = match filter {
        FilterType::Nearest => fir::ResizeAlg::Nearest,
        FilterType::Triangle => fir::ResizeAlg::Convolution(fir::FilterType::Bilinear),
        FilterType::CatmullRom => fir::ResizeAlg::Convolution(fir::FilterType::CatmullRom),
        FilterType::Gaussian => fir::ResizeAlg::Convolution(fir::FilterType::Gaussian),
        FilterType::Lanczos3 => fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3),
    };

    // Allocate destination buffer
    let dst_size = (target_height as usize) * (target_width as usize) * c;
    let mut dst_data = vec![0u8; dst_size];

    // Get source buffer slice
    let src_len = h * w * c;
    let src_slice = unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u8>(), src_len) };

    // Map channel count to pixel type
    let pixel_type = match c {
        1 => fir::PixelType::U8,
        3 => fir::PixelType::U8x3,
        4 => fir::PixelType::U8x4,
        _ => panic!("Resize only supports 1, 3, or 4 channels, got {c}"),
    };

    // Create source image (read-only reference)
    let src_image = fir::images::ImageRef::new(w as u32, h as u32, src_slice, pixel_type)
        .expect("Failed to create source image");

    // Create destination image (mutable)
    let mut dst_image =
        fir::images::Image::from_slice_u8(target_width, target_height, &mut dst_data, pixel_type)
            .expect("Failed to create dest image");

    // Perform resize
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
    let (h, w) = (shape[0], shape[1]);
    let channels = shape.get(2).copied().unwrap_or(1);

    if channels == 1 {
        // Already grayscale
        return buf;
    }

    let strides = buf.strides_bytes();

    // Fast path: contiguous RGB buffer with standard layout
    // Strides should be [w*3, 3, 1] for contiguous HWC layout
    if buf.layout.is_contiguous() && channels == 3 {
        let data = unsafe { std::slice::from_raw_parts(buf.as_ptr::<u8>(), h * w * 3) };
        let mut gray_data: Vec<u8> = Vec::with_capacity(h * w);

        // Process 1 pixel at a time with direct slice access (no bounds check per channel)
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
    // Uses pointer arithmetic which handles both positive and negative strides
    let (stride_h, stride_w, stride_c) =
        (strides[0], strides[1], strides.get(2).copied().unwrap_or(1));
    let base_ptr = unsafe { buf.as_ptr::<u8>() };

    let mut gray_data: Vec<u8> = Vec::with_capacity(h * w);

    for y in 0..h {
        for x in 0..w {
            // Use pointer offset arithmetic to handle negative strides properly
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

/// Applies an image operation to a buffer.
///
/// Image operations accept any numeric input dtype and automatically convert
/// to U8 as needed. For float inputs in [0.0, 1.0], values are scaled to [0, 255].
///
/// If tiling is enabled (via environment variable or [`with_tile_config`]),
/// tileable operations will be executed tile-by-tile for improved cache efficiency.
#[cfg(feature = "image_interop")]
#[inline]
pub fn apply_image(buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    // Convert to U8 if needed (dtype promotion for image ops)
    let work_buf = convert_to_u8_for_image(buf);

    // Fast path: atomic check avoids TLS access when tiling is disabled
    if !is_tiling_enabled() {
        return apply_image_inner(work_buf, op);
    }

    // Slow path: tiling might be enabled, check TLS and policy
    let tile_config = get_tile_config();
    if let Some(ref config) = tile_config {
        let policy = op.tile_policy();
        if policy.is_tileable() {
            let halo = policy.halo();
            let op_clone = op.clone();
            return maybe_tiled(work_buf, halo, Some(config), move |tile| {
                apply_image_inner(tile, op_clone.clone())
            });
        }
    }

    apply_image_inner(work_buf, op)
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

/// Inner implementation of image operations (without tiling logic).
#[cfg(feature = "image_interop")]
#[inline]
fn apply_image_inner(work_buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    match op.kind {
        ImageOpKind::Threshold(thresh) => {
            // Fast path: try grayscale image view (for HW1 layout with positive strides)
            if let Ok(view) = work_buf.as_image_view::<Luma<u8>>() {
                // For image view, rows may have padding but are contiguous within
                // Process each row using SIMD-friendly threshold
                let total_pixels = (view.width * view.height) as usize;
                let mut new_data: Vec<u8> = Vec::with_capacity(total_pixels);

                for y in 0..view.height {
                    let row_start = (y as usize) * view.row_stride;
                    let row_slice = &view.data[row_start..row_start + view.width as usize];
                    // Use SIMD threshold for each row
                    let thresholded = threshold_simd(row_slice, thresh);
                    new_data.extend_from_slice(&thresholded);
                }

                ViewBuffer::from_vec(new_data).reshape(vec![
                    view.height as usize,
                    view.width as usize,
                    1,
                ])
            } else {
                // Fallback: ensure contiguous and use SIMD threshold
                let contig_buf = if work_buf.layout.is_contiguous() {
                    work_buf
                } else {
                    work_buf.to_contiguous()
                };
                let count = contig_buf.layout.num_elements();
                let src_slice =
                    unsafe { std::slice::from_raw_parts(contig_buf.as_ptr::<u8>(), count) };

                // Use SIMD-friendly threshold
                let new_data = threshold_simd(src_slice, thresh);

                ViewBuffer::from_vec(new_data).reshape(contig_buf.shape().to_vec())
            }
        }
        ImageOpKind::Grayscale => grayscale_strided(work_buf),
        ImageOpKind::Resize {
            width,
            height,
            filter,
        } => resize_strided(work_buf, width, height, filter),
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
