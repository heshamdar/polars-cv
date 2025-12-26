use crate::buffer::TensorBuffer;
use crate::dtype::DType;
use crate::expr::{ExprNode, TensorExpr};
use crate::image_view::AsImageView;
use crate::interop::{AsNdarray, FromNdarray};
use crate::layout::ExternalLayout; // NEW: Needed for compatibility checks
use crate::ops::{ComputeOp, FilterType, ImageOp, ImageOpKind, MemoryEffect, Op, ViewOp};
use image::imageops;
use image::{ImageBuffer, Luma, Rgb};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum PlanStep {
    View(ViewOp),
    Compute(ComputeOp),
    Image(ImageOp),
    MaterializeContiguous,
}

#[derive(Debug)]
pub struct ExecutionPlan {
    pub source: TensorBuffer,
    pub steps: Vec<PlanStep>,
}

impl ExecutionPlan {
    pub fn execute(self) -> TensorBuffer {
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

fn apply_view(buf: TensorBuffer, op: ViewOp) -> TensorBuffer {
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

fn apply_compute(buf: TensorBuffer, op: ComputeOp) -> TensorBuffer {
    match op {
        ComputeOp::Cast(dtype) => buf.cast(dtype),
        ComputeOp::Affine(_params) => unimplemented!("Affine transform compute"),
        ComputeOp::Scale(factor) => apply_ndarray_op(&buf, |x: f32| x * factor),
        ComputeOp::Relu => apply_ndarray_op(&buf, |x: f32| if x > 0.0 { x } else { 0.0 }),
    }
}

fn apply_image(buf: TensorBuffer, op: ImageOp) -> TensorBuffer {
    match op.kind {
        ImageOpKind::Threshold(thresh) => {
            // Optimization: Try to use zero-copy view to read input (Stride Preserving)
            if let Ok(view) = buf.as_image_view::<Luma<u8>>() {
                let mut new_data = Vec::with_capacity((view.width * view.height) as usize);
                for y in 0..view.height {
                    let row_start = (y as usize) * view.row_stride;
                    let row_slice = &view.data[row_start..row_start + view.width as usize];
                    for &p in row_slice {
                        new_data.push(if p > thresh { 255 } else { 0 });
                    }
                }
                TensorBuffer::from_vec(new_data).reshape(vec![
                    view.height as usize,
                    view.width as usize,
                    1,
                ])
            } else {
                // Decision: Materialize if not compatible or if view creation failed
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

                TensorBuffer::from_vec(new_data).reshape(work_buf.shape().to_vec())
            }
        }
        ImageOpKind::Grayscale => {
            // Optimization: Use zero-copy view to read strided RGB input
            if let Ok(view) = buf.as_image_view::<Rgb<u8>>() {
                let mut new_data = Vec::with_capacity((view.width * view.height) as usize);
                for y in 0..view.height {
                    let row_start = (y as usize) * view.row_stride;
                    let row_end = row_start + (view.width as usize * 3);
                    let row_slice = &view.data[row_start..row_end];

                    for chunk in row_slice.chunks_exact(3) {
                        let (r, g, b) = (chunk[0] as f32, chunk[1] as f32, chunk[2] as f32);
                        let luma = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                        new_data.push(luma);
                    }
                }
                TensorBuffer::from_vec(new_data).reshape(vec![
                    view.height as usize,
                    view.width as usize,
                    1,
                ])
            } else {
                panic!("Grayscale requires RGB u8 image layout");
            }
        }
        ImageOpKind::Resize {
            width,
            height,
            filter,
        } => {
            // Decision: Check compatibility with ImageCrate requirements before proceeding.
            // If compatible, we use the buffer directly (assuming here it's dense enough for ImageBuffer or we handle it).
            // If not, we MUST materialize to contiguous layout.
            let work_buf = if buf.is_compatible_with(ExternalLayout::ImageCrate)
                && buf.layout.is_contiguous()
            {
                // Note: We currently add `&& is_contiguous()` because standard `ImageBuffer::from_raw`
                // assumes dense packing without row padding.
                // A more advanced backend adapter would handle padding if ImageCrate layout allows it.
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

            // Safe now because work_buf is guaranteed compatible/contiguous
            let count = work_buf.layout.num_elements();
            let raw_vec =
                unsafe { std::slice::from_raw_parts(work_buf.as_ptr::<u8>(), count).to_vec() };

            if c == 3 {
                let img_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, raw_vec)
                    .expect("Failed to create ImageBuffer from compatible buffer");
                let resized = imageops::resize(&img_buf, width, height, img_filter);
                TensorBuffer::from_vec(resized.into_raw()).reshape(vec![
                    height as usize,
                    width as usize,
                    3,
                ])
            } else if c == 1 {
                let img_buf: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(w, h, raw_vec)
                    .expect("Failed to create ImageBuffer from compatible buffer");
                let resized = imageops::resize(&img_buf, width, height, img_filter);
                TensorBuffer::from_vec(resized.into_raw()).reshape(vec![
                    height as usize,
                    width as usize,
                    1,
                ])
            } else {
                panic!("Resize only supports 1 or 3 channels currently");
            }
        }
        ImageOpKind::Blur { sigma } => {
            // Decision: Enforce layout compatibility
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
                TensorBuffer::from_vec(blurred.into_raw()).reshape(vec![h as usize, w as usize, 3])
            } else {
                let img_buf: ImageBuffer<Luma<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(w, h, raw_vec).unwrap();
                let blurred = imageops::blur(&img_buf, sigma);
                TensorBuffer::from_vec(blurred.into_raw()).reshape(vec![h as usize, w as usize, 1])
            }
        }
    }
}

fn apply_ndarray_op<F>(buf: &TensorBuffer, op: F) -> TensorBuffer
where
    F: Fn(f32) -> f32,
{
    match buf.dtype() {
        DType::F32 => {
            let view = buf.as_array_view::<f32>().expect("Failed to create view");
            let result_array = view.mapv(op);
            TensorBuffer::from_array(result_array)
        }
        _ => unimplemented!("Math ops only implemented for F32 in this prototype"),
    }
}

impl TensorExpr {
    /// Generates the execution plan.
    /// implicitly runs optimization pass before planning.
    pub fn plan(self: &Arc<Self>) -> ExecutionPlan {
        let optimized_expr = self.optimize();
        optimized_expr.build_plan()
    }

    /// Internal recursive planner builder
    fn build_plan(&self) -> ExecutionPlan {
        match &self.node {
            ExprNode::Source(buf) => ExecutionPlan {
                source: buf.as_ref().clone(),
                steps: Vec::new(),
            },
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
                        // Image ops like Resize almost always require contiguous inputs
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
        }
    }
}

fn plan_ends_in_view(plan: &ExecutionPlan) -> bool {
    matches!(plan.steps.last(), Some(PlanStep::View(_)))
}
