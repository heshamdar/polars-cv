//! Execution planning and running for ViewExpr graphs.

use std::sync::Arc;

use crate::buffer::ViewBuffer;
use crate::dtype::DType;
use crate::expr::{ExprNode, ViewExpr};
use crate::ops::{ComputeOp, ImageOp, MemoryEffect, Op, ViewOp};

#[cfg(feature = "image_interop")]
use crate::layout::ExternalLayout;
#[cfg(feature = "image_interop")]
use crate::ops::{FilterType, ImageOpKind};

#[cfg(feature = "image_interop")]
use crate::image_view::AsImageView;

#[cfg(feature = "ndarray_interop")]
use crate::ndarray_view::{AsNdarray, FromNdarray};

#[cfg(feature = "image_interop")]
use image::imageops;
#[cfg(feature = "image_interop")]
use image::{ImageBuffer, Luma, Rgb};

/// A step in the execution plan.
#[derive(Debug, Clone)]
pub enum PlanStep {
    View(ViewOp),
    Compute(ComputeOp),
    Image(ImageOp),
    MaterializeContiguous,
}

/// An execution plan built from a ViewExpr graph.
#[derive(Debug)]
pub struct ExecutionPlan {
    pub source: ViewBuffer,
    pub steps: Vec<PlanStep>,
}

impl ExecutionPlan {
    /// Executes the plan and returns the resulting ViewBuffer.
    pub fn execute(self) -> ViewBuffer {
        let mut current_buffer = self.source;

        for step in self.steps {
            match step {
                PlanStep::View(op) => {
                    current_buffer = apply_view(current_buffer, op);
                }
                PlanStep::Compute(op) => {
                    current_buffer = apply_compute(current_buffer, op);
                }
                PlanStep::Image(op) => {
                    current_buffer = apply_image(current_buffer, op);
                }
                PlanStep::MaterializeContiguous => {
                    current_buffer = current_buffer.to_contiguous();
                }
            }
        }
        current_buffer
    }
}

fn apply_view(buf: ViewBuffer, op: ViewOp) -> ViewBuffer {
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

fn apply_compute(buf: ViewBuffer, op: ComputeOp) -> ViewBuffer {
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

fn apply_normalize(
    buf: &ViewBuffer,
    method: crate::ops::NormalizeMethod,
) -> ViewBuffer {
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

#[cfg(feature = "image_interop")]
fn apply_image(buf: ViewBuffer, op: ImageOp) -> ViewBuffer {
    match op.kind {
        ImageOpKind::Threshold(thresh) => {
            if let Ok(view) = buf.as_image_view::<Luma<u8>>() {
                let mut new_data: Vec<u8> =
                    Vec::with_capacity((view.width * view.height) as usize);
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
                let img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, raw_vec)
                    .expect("Failed to create ImageBuffer");
                let resized = imageops::resize(&img_buf, width, height, img_filter);
                ViewBuffer::from_vec(resized.into_raw()).reshape(vec![
                    height as usize,
                    width as usize,
                    3,
                ])
            } else if c == 1 {
                let img_buf: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, raw_vec)
                    .expect("Failed to create ImageBuffer");
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
fn apply_image(_buf: ViewBuffer, _op: ImageOp) -> ViewBuffer {
    panic!("Image operations require the 'image_interop' feature");
}

impl ViewExpr {
    /// Builds and returns an execution plan from the expression graph.
    pub fn plan(self: &Arc<Self>) -> ExecutionPlan {
        let optimized_expr = self.optimize();
        optimized_expr.build_plan()
    }

    fn build_plan(&self) -> ExecutionPlan {
        match &self.node {
            ExprNode::Source(buf) => ExecutionPlan {
                source: buf.as_ref().clone(),
                steps: Vec::new(),
            },
            ExprNode::LazySource { .. } => {
                panic!("LazySource must be resolved before building plan");
            }
            ExprNode::Placeholder(_) => {
                panic!("Placeholder must be bound to data before building plan");
            }
            ExprNode::View(op, child) => {
                let mut plan = child.build_plan();
                plan.steps.push(PlanStep::View(op.clone()));
                plan
            }
            ExprNode::Compute(op, children) => {
                let mut plan = children[0].build_plan();

                match op.memory_effect() {
                    MemoryEffect::RequiresContiguous => {
                        if plan_ends_in_view(&plan) || !plan.source.layout.is_contiguous() {
                            plan.steps.push(PlanStep::MaterializeContiguous);
                        }
                    }
                    MemoryEffect::StridePreserving => {}
                    MemoryEffect::View => unreachable!(),
                }

                plan.steps.push(PlanStep::Compute(op.clone()));
                plan
            }
            ExprNode::Image(op, child) => {
                let mut plan = child.build_plan();

                match op.memory_effect() {
                    MemoryEffect::RequiresContiguous => {
                        if plan_ends_in_view(&plan) || !plan.source.layout.is_contiguous() {
                            plan.steps.push(PlanStep::MaterializeContiguous);
                        }
                    }
                    MemoryEffect::StridePreserving => {}
                    MemoryEffect::View => unreachable!(),
                }

                plan.steps.push(PlanStep::Image(op.clone()));
                plan
            }
            ExprNode::Sink { input, .. } => {
                // Sink doesn't add steps; the format is handled after execution
                input.build_plan()
            }
        }
    }
}

fn plan_ends_in_view(plan: &ExecutionPlan) -> bool {
    matches!(plan.steps.last(), Some(PlanStep::View(_)))
}
