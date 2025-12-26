use crate::dtype::DType;
use crate::ops::affine::AffineParams;
use crate::ops::scalar::FusedKernel;
// Needed to calculate default strides

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
            ViewOp::Crop { start, end } => {
                start.iter().zip(end.iter()).map(|(s, e)| e - s).collect()
            }
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
            ViewOp::Transpose(perm) => Some(perm.iter().map(|&i| input_strides[i]).collect()),
            ViewOp::Reshape(_new_shape) => {
                // Reshape is only a valid View if the input is contiguous
                // Check contiguity using Layout logic
                // We construct a dummy layout to reuse the is_contiguous check
                // Note: DType size doesn't matter for the logic of contiguity check on strides/shape
                // but we need one.
                let layout_check = crate::layout::LayoutFacts::new(
                    input_shape,
                    input_strides,
                    DType::U8, // Dummy
                    0,
                );

                if layout_check.is_contiguous() {
                    // Result is new contiguous strides
                    // We need a dummy DType size to calculate strides in bytes.
                    // Ideally infer_strides should take element_size, but let's assume byte-strides are what we are tracking.
                    // Wait, input_strides are in bytes. We need element size to calc new strides.
                    // Limitation: Op trait doesn't pass input DType to infer_strides currently.
                    // Let's rely on standard calculation logic assuming we can get element size.
                    // Actually, Layout::new_contiguous requires DType.
                    // We will return None here if we can't reliably calculate without DType,
                    // OR we assume the caller handles the "Bytes" aspect.
                    // Let's change the trait signature?
                    // Minimally: assume dense packing based on total size?
                    // Better: To correctly calculate bytes strides, we need input DType.
                    None // Defer to runtime/planner checks for now as we lack DType here
                } else {
                    None // Invalid view
                }
            }
            ViewOp::Flip(_) => {
                // Flip negates strides. But we need to update offsets which isn't tracked in strides.
                // Strides themselves just become negative.
                // Wait, logic in buffer.rs: new_strides[axis] = -stride;
                // Here we assume axis matches input layout.
                // We need the axes.
                let axes = match self {
                    ViewOp::Flip(a) => a,
                    _ => unreachable!(),
                };
                let mut new_strides = input_strides.to_vec();
                for &axis in axes {
                    new_strides[axis] = -new_strides[axis];
                }
                Some(new_strides)
            }
            ViewOp::Crop { .. } => {
                // Crop preserves strides
                Some(input_strides.to_vec())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
