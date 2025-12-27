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
        ComputeOp::Normalize(method) => apply_normalize(&buf, method),
        ComputeOp::Clamp { min, max } => apply_scalar_op(&buf, move |x: f32| x.clamp(min, max)),
    }
}

fn apply_normalize(buf: &ViewBuffer, method: crate::ops::NormalizeMethod) -> ViewBuffer {
    use crate::ops::NormalizeMethod;

    if buf.dtype() != DType::F32 {
        panic!("Normalize requires F32 dtype");
    }

    let contig = buf.to_contiguous();
    let count = contig.layout.num_elements();
    let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };

    let new_data: Vec<f32> = match method {
        NormalizeMethod::MinMax => {
            let min = src.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = src.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            if range == 0.0 {
                src.to_vec()
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
                src.iter().map(|_| 0.0).collect()
            } else {
                src.iter().map(|&x| (x - mean) / std).collect()
            }
        }
    };

    ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec())
}

/// Apply a scalar operation element-wise.
fn apply_scalar_op<F>(buf: &ViewBuffer, op: F) -> ViewBuffer
where
    F: Fn(f32) -> f32,
{
    // Try to use ndarray if available for efficient strided iteration
    #[cfg(feature = "ndarray_interop")]
    {
        if buf.dtype() == DType::F32 {
            if let Ok(view) = buf.as_array_view::<f32>() {
                let result_array = view.mapv(&op);
                return ViewBuffer::from_array(result_array);
            }
        }
    }

    // Fallback: use contiguous buffer
    if buf.dtype() == DType::F32 {
        let contig = buf.to_contiguous();
        let count = contig.layout.num_elements();
        let src = unsafe { std::slice::from_raw_parts(contig.as_ptr::<f32>(), count) };
        let new_data: Vec<f32> = src.iter().map(|&x| op(x)).collect();
        ViewBuffer::from_vec(new_data).reshape(contig.shape().to_vec())
    } else {
        unimplemented!("Scalar ops only implemented for F32");
    }
}

/// Applies an image operation to a buffer.
#[cfg(feature = "image_interop")]
pub fn apply_image(buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    match op.kind {
        ImageOpKind::Threshold(thresh) => {
            if let Ok(view) = buf.as_image_view::<Luma<u8>>() {
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
                let work_buf = if buf.layout.is_contiguous() {
                    buf
                } else {
                    buf.to_contiguous()
                };
                if work_buf.dtype() != DType::U8 {
                    panic!("Threshold op requires U8 dtype");
                }
                let count = work_buf.layout.num_elements();
                let src_slice =
                    unsafe { std::slice::from_raw_parts(work_buf.as_ptr::<u8>(), count) };

                let new_data: Vec<u8> = src_slice
                    .iter()
                    .map(|&p| if p > thresh { 255 } else { 0 })
                    .collect();

                ViewBuffer::from_vec(new_data).reshape(work_buf.shape().to_vec())
            }
        }
        ImageOpKind::Grayscale => {
            let shape = buf.shape();
            let channels = *shape.get(2).unwrap_or(&1);

            if channels == 1 {
                return buf;
            }

            let work_buf = if buf.is_compatible_with(ExternalLayout::ImageCrate)
                && buf.layout.is_contiguous()
            {
                buf
            } else {
                buf.to_contiguous()
            };

            if work_buf.dtype() != DType::U8 {
                panic!("Grayscale op requires U8 dtype");
            }

            let (h, w) = (work_buf.shape()[0] as u32, work_buf.shape()[1] as u32);
            let count = work_buf.layout.num_elements();
            let raw_slice = unsafe { std::slice::from_raw_parts(work_buf.as_ptr::<u8>(), count) };

            if let Some(img_buf) = ImageBuffer::<Rgb<u8>, &[u8]>::from_raw(w, h, raw_slice) {
                let gray = imageops::grayscale(&img_buf);
                ViewBuffer::from_vec(gray.into_raw()).reshape(vec![h as usize, w as usize, 1])
            } else {
                panic!("Failed to create ImageBuffer for grayscale operation");
            }
        }
        ImageOpKind::Resize {
            width,
            height,
            filter,
        } => {
            let work_buf = if buf.is_compatible_with(ExternalLayout::ImageCrate)
                && buf.layout.is_contiguous()
            {
                buf
            } else {
                buf.to_contiguous()
            };

            let shape = work_buf.shape();
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

            let count = work_buf.layout.num_elements();
            let raw_vec =
                unsafe { std::slice::from_raw_parts(work_buf.as_ptr::<u8>(), count).to_vec() };

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
            let work_buf = if buf.is_compatible_with(ExternalLayout::ImageCrate)
                && buf.layout.is_contiguous()
            {
                buf
            } else {
                buf.to_contiguous()
            };

            let shape = work_buf.shape();
            let (h, w, c) = (
                shape[0] as u32,
                shape[1] as u32,
                *shape.get(2).unwrap_or(&1) as u32,
            );
            let count = work_buf.layout.num_elements();
            let raw_vec =
                unsafe { std::slice::from_raw_parts(work_buf.as_ptr::<u8>(), count).to_vec() };

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
