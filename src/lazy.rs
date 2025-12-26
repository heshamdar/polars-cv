use std::sync::Arc;
use crate::buffer::TensorBuffer;
use crate::dtype::DType;
use crate::ops::affine::AffineParams;

#[derive(Debug, Clone, PartialEq)]
pub struct CropParams {
    pub start: Vec<usize>,
    pub end: Vec<usize>,
}

#[derive(Debug, Clone)]
pub enum Op {
    Load(TensorBuffer),
    Transpose { input: Arc<LazyTensor>, perm: Vec<usize> },
    Reshape { input: Arc<LazyTensor>, shape: Vec<usize> },
    Flip { input: Arc<LazyTensor>, axes: Vec<usize> },
    Crop { input: Arc<LazyTensor>, params: CropParams },
    Cast { input: Arc<LazyTensor>, dtype: DType },
    Affine { input: Arc<LazyTensor>, params: AffineParams },
}

#[derive(Debug, Clone)]
pub struct LazyTensor {
    pub op: Op,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl LazyTensor {
    pub fn new(buf: TensorBuffer) -> Arc<Self> {
        Arc::new(Self {
            shape: buf.shape().to_vec(),
            dtype: buf.dtype(),
            op: Op::Load(buf),
        })
    }

    pub fn transpose(self: &Arc<Self>, perm: Vec<usize>) -> Arc<Self> {
        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        Arc::new(Self {
            op: Op::Transpose { input: self.clone(), perm },
            shape: new_shape,
            dtype: self.dtype,
        })
    }
    
    pub fn flip(self: &Arc<Self>, axes: Vec<usize>) -> Arc<Self> {
        Arc::new(Self {
            op: Op::Flip { input: self.clone(), axes },
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    pub fn crop(self: &Arc<Self>, start: Vec<usize>, end: Vec<usize>) -> Arc<Self> {
        let new_shape: Vec<usize> = start.iter().zip(end.iter()).map(|(s, e)| e - s).collect();
        Arc::new(Self {
            op: Op::Crop { input: self.clone(), params: CropParams { start, end } },
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    pub fn cast(self: &Arc<Self>, dtype: DType) -> Arc<Self> {
        Arc::new(Self {
            op: Op::Cast { input: self.clone(), dtype },
            shape: self.shape.clone(),
            dtype,
        })
    }
    
    pub fn affine(self: &Arc<Self>, params: AffineParams) -> Arc<Self> {
        Arc::new(Self {
            op: Op::Affine { input: self.clone(), params },
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    /// Optimizes the graph by merging nodes and eliminating redundancy.
    pub fn optimize(self: &Arc<Self>) -> Arc<Self> {
        // Recursive optimization: Optimize children first
        let optimized_input = match &self.op {
            Op::Load(_) => return self.clone(),
            Op::Transpose { input, .. } => input.optimize(),
            Op::Reshape { input, .. } => input.optimize(),
            Op::Flip { input, .. } => input.optimize(),
            Op::Crop { input, .. } => input.optimize(),
            Op::Cast { input, .. } => input.optimize(),
            Op::Affine { input, .. } => input.optimize(),
        };

        match (&self.op, &optimized_input.op) {
            
            // Rule: Merge Affine Chains
            // Affine(Affine(x, p1), p2) -> Affine(x, p2 * p1)
            (Op::Affine { params: p2, .. }, Op::Affine { params: p1, input: grand_child }) => {
                let merged_params = p1.combine(p2);
                Arc::new(Self {
                    op: Op::Affine { input: grand_child.clone(), params: merged_params },
                    shape: self.shape.clone(),
                    dtype: self.dtype,
                })
            },

            // Rule: Push Casts Late (Swap Cast and Crop)
            // Crop(Cast(x)) -> Cast(Crop(x))
            // We want to crop the raw data (saving bandwidth) then cast the smaller result.
            (Op::Crop { params, .. }, Op::Cast { input: grand_child, dtype: cast_dtype }) => {
                // 1. Create inner Crop on the grand_child (pre-cast data)
                let inner_crop = grand_child.crop(params.start.clone(), params.end.clone());
                // 2. Wrap with Cast
                inner_crop.cast(*cast_dtype)
            },

            // Default: Rebuild node with optimized input
            _ => self.rebuild_with_input(optimized_input),
        }
    }

    fn rebuild_with_input(&self, input: Arc<Self>) -> Arc<Self> {
        let new_op = match &self.op {
            Op::Load(_) => return input,
            Op::Transpose { perm, .. } => Op::Transpose { input, perm: perm.clone() },
            Op::Reshape { shape, .. } => Op::Reshape { input, shape: shape.clone() },
            Op::Flip { axes, .. } => Op::Flip { input, axes: axes.clone() },
            Op::Crop { params, .. } => Op::Crop { input, params: params.clone() },
            Op::Cast { dtype, .. } => Op::Cast { input, dtype: *dtype },
            Op::Affine { params, .. } => Op::Affine { input, params: params.clone() },
        };
        Arc::new(Self {
            op: new_op,
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    pub fn materialize(self: &Arc<Self>) -> TensorBuffer {
        let optimized = self.optimize();
        optimized.exec()
    }

    // Internal execution of the optimized graph
    fn exec(&self) -> TensorBuffer {
        match &self.op {
            Op::Load(buf) => buf.clone(),
            Op::Crop { input, params } => {
                let parent = input.exec();
                parent.slice(&params.start, &params.end)
            },
            Op::Transpose { input, perm } => {
                let parent = input.exec();
                parent.permute(perm)
            },
            Op::Reshape { input, shape } => {
                let parent = input.exec();
                // Minimal check for contiguous
                 if !parent.layout.is_contiguous() {
                    panic!("Reshaping non-contiguous tensors requires copy (not implemented)");
                }
                crate::buffer::TensorBuffer {
                    data: parent.data.clone(),
                    layout: crate::layout::Layout::new_contiguous(shape.clone(), parent.layout.dtype),
                }
            },
            Op::Flip { input, axes } => {
                let parent = input.exec();
                parent.flip(axes)
            },
            Op::Cast { input, dtype } => {
                let parent = input.exec();
                parent.cast(*dtype)
            },
            Op::Affine { .. } => {
                unimplemented!("Affine transform execution (resampling) required - Placeholder")
            },
        }
    }
}