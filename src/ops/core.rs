use crate::dtype::DType;
use crate::ops::affine::AffineParams;
use crate::ops::scalar::FusedKernel;
 // Needed to calculate default strides
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

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
    
    /// meaningful only for operations that preserve/transform strides (Views).
    /// Returns None if strides cannot be inferred or if the operation requires materialization 
    /// that makes input strides irrelevant (in which case output is usually default contiguous).
    fn infer_strides(&self, input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>>;
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

    fn infer_strides(&self, input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
        match self {
            ViewOp::Transpose(perm) => {
                Some(perm.iter().map(|&i| input_strides[i]).collect())
            },
            ViewOp::Reshape(_new_shape) => {
                let layout_check = crate::layout::LayoutFacts::new(
                    input_shape, 
                    input_strides, 
                    DType::U8, // Dummy
                    0
                );
                
                if layout_check.is_contiguous() {
                    None // Defer to runtime/planner checks for now as we lack DType here
                } else {
                    None // Invalid view
                }
            },
            ViewOp::Flip(_) => {
                let axes = match self { ViewOp::Flip(a) => a, _ => unreachable!() };
                let mut new_strides = input_strides.to_vec();
                for &axis in axes {
                    new_strides[axis] = -new_strides[axis];
                }
                Some(new_strides)
            },
            ViewOp::Crop { .. } => {
                Some(input_strides.to_vec())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComputeOp {
    Cast(DType),
    Affine(AffineParams),
    Scale(f32),
    Relu,
    Fused(FusedKernel),
}

impl Op for ComputeOp {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        inputs[0].to_vec()
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        match self {
            ComputeOp::Cast(target) => *target,
            ComputeOp::Affine(_) => inputs[0],
            ComputeOp::Scale(_) => inputs[0],
            ComputeOp::Relu => inputs[0],
            ComputeOp::Fused(_) => inputs[0],
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        match self {
            ComputeOp::Cast(_) => MemoryEffect::StridePreserving,
            ComputeOp::Scale(_) => MemoryEffect::StridePreserving,
            ComputeOp::Relu => MemoryEffect::StridePreserving,
            ComputeOp::Fused(_) => MemoryEffect::StridePreserving,
            ComputeOp::Affine(_) => MemoryEffect::RequiresContiguous,
        }
    }

    fn infer_strides(&self, _input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
        match self.memory_effect() {
            MemoryEffect::StridePreserving => Some(input_strides.to_vec()),
            MemoryEffect::RequiresContiguous => None, // Output will be contiguous, calculation requires DType
            MemoryEffect::View => unreachable!(),
        }
    }
}