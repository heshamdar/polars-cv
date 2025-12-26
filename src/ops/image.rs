use crate::dtype::DType;
use crate::ops::core::{Op, MemoryEffect};
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ImageOpKind {
    Threshold(u8),
    Resize { width: u32, height: u32, filter: FilterType },
    Blur { sigma: f32 },
    Grayscale,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FilterType {
    Nearest,
    Triangle,
    CatmullRom,
    Gaussian,
    Lanczos3,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ImageOp {
    pub kind: ImageOpKind,
}

impl Op for ImageOp {
    fn infer_shape(&self, inputs: &[&[usize]]) -> Vec<usize> {
        let input_shape = inputs[0];
        match &self.kind {
            ImageOpKind::Threshold(_) => input_shape.to_vec(),
            ImageOpKind::Blur { .. } => input_shape.to_vec(),
            ImageOpKind::Grayscale => {
                let mut s = input_shape.to_vec();
                if s.len() == 3 {
                    s[2] = 1;
                } else if s.len() == 2 {
                    s.push(1);
                }
                s
            },
            ImageOpKind::Resize { width, height, .. } => {
                let mut s = input_shape.to_vec();
                if s.len() >= 2 {
                    s[0] = *height as usize;
                    s[1] = *width as usize;
                }
                s
            },
        }
    }

    fn infer_dtype(&self, inputs: &[DType]) -> DType {
        match &self.kind {
            ImageOpKind::Grayscale => DType::U8,
            ImageOpKind::Threshold(_) => DType::U8,
            _ => inputs[0],
        }
    }

    fn memory_effect(&self) -> MemoryEffect {
        match &self.kind {
            ImageOpKind::Threshold(_) => MemoryEffect::StridePreserving,
            ImageOpKind::Resize { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Blur { .. } => MemoryEffect::RequiresContiguous,
            ImageOpKind::Grayscale => MemoryEffect::RequiresContiguous,
        }
    }

    fn infer_strides(&self, _input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
        match self.memory_effect() {
            MemoryEffect::StridePreserving => Some(input_strides.to_vec()),
            MemoryEffect::RequiresContiguous => None,
            MemoryEffect::View => unreachable!(),
        }
    }
}