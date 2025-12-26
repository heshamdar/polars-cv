use crate::dtype::DType;
use crate::ops::affine::AffineParams;

/// Describes how an operation interacts with memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEffect {
    View,
    StridePreserving,
    RequiresContiguous,
}

pub trait Op {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize>;
    fn infer_dtype(&self, inputs: &[DType]) -> DType;
    fn memory_effect(&self) -> MemoryEffect;
}

#[derive(Debug, Clone, PartialEq)]
pub enum ViewOp {
    Transpose(Vec<usize>),
    Reshape(Vec<usize>),
    Flip(Vec<usize>),
    Crop { start: Vec<usize>, end: Vec<usize> },
}

impl Op for ViewOp {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input_shape = inputs[0]; 
        match self {
            ViewOp::Transpose(perm) => perm.iter().map(|&i| input_shape[i]).collect(),
            ViewOp::Reshape(new_shape) => new_shape.clone(),
            ViewOp::Flip(_) => input_shape.to_vec(),
            ViewOp::Crop { start, end } => start.iter().zip(end.iter()).map(|(s, e)| e - s).collect(),
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        inputs[0]
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::View
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComputeOp {
    Cast(DType),
    Affine(AffineParams),
    Scale(f32),
    Relu,
}

impl Op for ComputeOp {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        // All current compute ops preserve shape (element-wise or resampling within bounds)
        inputs[0].to_vec()
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        match self {
            ComputeOp::Cast(target) => *target,
            ComputeOp::Affine(_) => inputs[0],
            ComputeOp::Scale(_) => inputs[0], // Assuming scale preserves float type
            ComputeOp::Relu => inputs[0],
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        match self {
            ComputeOp::Cast(_) => MemoryEffect::StridePreserving,
            ComputeOp::Scale(_) => MemoryEffect::StridePreserving,
            ComputeOp::Relu => MemoryEffect::StridePreserving,
            ComputeOp::Affine(_) => MemoryEffect::RequiresContiguous,
        }
    }
}