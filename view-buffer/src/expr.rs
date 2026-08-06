use std::sync::Arc;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::core::layout::Layout;
use crate::execution::{ExecutionPlan, PlanStep};
use crate::ops::affine::AffineParams;
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::ops::traits::MemoryEffect;
use crate::ops::{
    ColorConvertOp, ComputeOp, ConvolveOp, FilterType, ImageOp, ImageOpKind, NormalizeMethod, Op,
    ViewDto, ViewOp,
};

/// A node in the expression graph.
#[derive(Debug, Clone)]
pub enum ExprNode {
    /// Concrete source data.
    Source(Arc<ViewBuffer>),

    /// View operation (zero-copy).
    View(ViewOp, Arc<ViewExpr>),

    /// Compute operation (allocating).
    Compute(ComputeOp, Vec<Arc<ViewExpr>>),

    /// Image processing operation.
    Image(ImageOp, Arc<ViewExpr>),

    /// Color space conversion.
    Color(ColorConvertOp, Arc<ViewExpr>),

    /// 2D convolution.
    Filter(ConvolveOp, Arc<ViewExpr>),
}

#[derive(Debug, Clone)]
pub struct ViewExpr {
    pub node: ExprNode,
    pub shape: Vec<usize>,
    pub strides: Option<Vec<isize>>, // NEW: Track strides symbolically
    pub dtype: DType,
}

impl ViewExpr {
    // --- Source Constructors ---

    /// Creates a new expression from a concrete ViewBuffer.
    pub fn new_source(buffer: ViewBuffer) -> Arc<Self> {
        Arc::new(Self {
            shape: buffer.shape().to_vec(),
            strides: Some(buffer.strides_bytes().to_vec()),
            dtype: buffer.dtype(),
            node: ExprNode::Source(Arc::new(buffer)),
        })
    }

    /// Entry point for applying a serializable operation DTO (from JSON/Plugins).
    /// Dispatches to the specific builder methods.
    pub fn apply_op(self: &Arc<Self>, op: ViewDto) -> Arc<Self> {
        match op {
            ViewDto::View(view) => match view {
                ViewOp::Transpose(perm) => self.transpose(perm),
                ViewOp::Reshape(shape) => self.reshape(shape),
                ViewOp::Flip(axes) => self.flip(axes),
                ViewOp::Crop { start, end } => self.crop(start, end),
                ViewOp::Rotate90 => {
                    if self.shape.len() < 2 {
                        return self.clone();
                    }
                    let perm = if self.shape.len() == 2 {
                        vec![1, 0]
                    } else {
                        vec![1, 0, 2]
                    };
                    self.transpose(perm).flip(vec![1])
                }
                ViewOp::Rotate180 => self.flip(vec![0, 1]),
                ViewOp::Rotate270 => {
                    if self.shape.len() < 2 {
                        return self.clone();
                    }
                    let perm = if self.shape.len() == 2 {
                        vec![1, 0]
                    } else {
                        vec![1, 0, 2]
                    };
                    self.transpose(perm).flip(vec![0])
                }
                ViewOp::ChannelSelect { index } => self.channel_select(index),
            },
            ViewDto::Compute(compute) => match compute {
                ComputeOp::Cast(dtype) => self.cast(dtype),
                ComputeOp::Affine(params) => self.affine(params),
                ComputeOp::Scale(f) => self.scale(f),
                ComputeOp::Relu => self.relu(),
                ComputeOp::Fused(kernel) => self.fused(kernel),
                ComputeOp::Normalize(method, out_dtype) => self.normalize(method, out_dtype),
                ComputeOp::Clamp { min, max } => self.clamp(min, max),
                ComputeOp::AdjustContrast(factor) => self.adjust_contrast(factor),
                ComputeOp::AdjustGamma(gamma) => self.adjust_gamma(gamma),
                ComputeOp::Invert => self.invert(),
                r @ ComputeOp::RotateAffine { .. } => {
                    let new_shape = r.infer_shape(&[&self.shape]);
                    let new_strides = self.calc_strides(&r, &new_shape);
                    Arc::new(Self {
                        node: ExprNode::Compute(r, vec![self.clone()]),
                        shape: new_shape,
                        strides: new_strides,
                        dtype: self.dtype,
                    })
                }
            },
            ViewDto::Image(img) => match img.kind {
                ImageOpKind::Threshold(val) => self.threshold(val),
                ImageOpKind::Resize {
                    width,
                    height,
                    filter,
                } => self.resize(width, height, filter),
                ImageOpKind::Blur { sigma } => self.blur(sigma),
                ImageOpKind::Grayscale => self.grayscale(),
                ImageOpKind::Canny { .. } => Arc::new(Self {
                    shape: img.infer_shape(&[&self.shape]),
                    strides: None,
                    dtype: DType::U8,
                    node: ExprNode::Image(img, self.clone()),
                }),
                ImageOpKind::HistogramEqualize => Arc::new(Self {
                    shape: img.infer_shape(&[&self.shape]),
                    strides: None,
                    dtype: DType::U8,
                    node: ExprNode::Image(img, self.clone()),
                }),
                ImageOpKind::Erode { .. }
                | ImageOpKind::Dilate { .. }
                | ImageOpKind::MorphGradient { .. }
                // Deferred resizes, padding, letterbox and channel reorder:
                // dtype-preserving allocating ops whose output shape comes
                // from infer_shape (backed by ImageOpKind::output_hw, the
                // same authority the runner executes with).
                | ImageOpKind::ResizeScale { .. }
                | ImageOpKind::ResizeToHeight { .. }
                | ImageOpKind::ResizeToWidth { .. }
                | ImageOpKind::ResizeMax { .. }
                | ImageOpKind::ResizeMin { .. }
                | ImageOpKind::Pad { .. }
                | ImageOpKind::PadToSize { .. }
                | ImageOpKind::Letterbox { .. }
                | ImageOpKind::ChannelSwap { .. } => {
                    let new_shape = img.infer_shape(&[&self.shape]);
                    let new_strides = self.calc_strides(&img, &new_shape);
                    Arc::new(Self {
                        shape: new_shape,
                        strides: new_strides,
                        dtype: self.dtype,
                        node: ExprNode::Image(img, self.clone()),
                    })
                }
            },
            ViewDto::Filter(op) => {
                let new_shape = Op::infer_shape(&op, &[&self.shape]);
                let new_strides = self.calc_strides(&op, &new_shape);
                let new_dtype = op.resolve_output_dtype(self.dtype, None);
                Arc::new(Self {
                    shape: new_shape,
                    strides: new_strides,
                    dtype: new_dtype,
                    node: ExprNode::Filter(op, self.clone()),
                })
            }
            ViewDto::Color(op) => {
                let new_shape = ColorConvertOp::infer_shape(&op, &self.shape);
                let new_dtype = Op::resolve_output_dtype(&op, self.dtype, None);
                Arc::new(Self {
                    shape: new_shape,
                    strides: None, // Color conversion always allocates
                    dtype: new_dtype,
                    node: ExprNode::Color(op, self.clone()),
                })
            }
        }
    }

    // Helper to calculate next strides
    fn calc_strides(&self, op: &impl Op, new_shape: &[usize]) -> Option<Vec<isize>> {
        if let Some(current_strides) = &self.strides {
            // Try to infer strides from the operation
            let res = op.infer_strides(&self.shape, current_strides);

            // If Op returned None, it means the operation produces contiguous output
            // (either because it requires contiguous input, or because it allocates a new buffer).
            // Calculate contiguous strides for the new shape.
            if res.is_none() {
                let new_dtype = op.resolve_output_dtype(self.dtype, None);
                let l = Layout::new_contiguous(new_shape.to_vec(), new_dtype);
                return Some(l.strides);
            }

            res
        } else {
            // If input strides are unknown, calculate contiguous strides for allocating ops
            // or ops that require contiguous input
            if op.memory_effect() != MemoryEffect::View {
                let new_dtype = op.resolve_output_dtype(self.dtype, None);
                let l = Layout::new_contiguous(new_shape.to_vec(), new_dtype);
                return Some(l.strides);
            }
            None
        }
    }

    // --- View Ops ---

    pub fn transpose(self: &Arc<Self>, perm: Vec<usize>) -> Arc<Self> {
        let op = ViewOp::Transpose(perm);
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::View(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn reshape(self: &Arc<Self>, new_shape: Vec<usize>) -> Arc<Self> {
        let op = ViewOp::Reshape(new_shape.clone());

        // Validation: Reshape on non-contiguous strided buffer is invalid as a View.
        if let Some(strides) = &self.strides {
            let facts = crate::core::layout::LayoutFacts::new(&self.shape, strides, self.dtype, 0);
            if !facts.is_contiguous() {
                // In a full implementation, we might auto-insert a Materialize op here.
                // For now, we allow the Planner to catch it (or panic) but we warn/mark strides None.
                // But since we want to "detect invalid views during definition":
                panic!("Invalid View: Cannot reshape non-contiguous view without copying. Input strides: {strides:?}");
            }
        }

        // If valid (or unknown), calculate new strides
        // Since Reshape implies contiguous -> contiguous, we generate new contiguous strides.
        let new_strides = if self.strides.is_some() {
            let l = Layout::new_contiguous(new_shape.clone(), self.dtype);
            Some(l.strides)
        } else {
            None
        };

        Arc::new(Self {
            node: ExprNode::View(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn crop(self: &Arc<Self>, start: Vec<usize>, end: Vec<usize>) -> Arc<Self> {
        let op = ViewOp::Crop { start, end };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::View(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn flip(self: &Arc<Self>, axes: Vec<usize>) -> Arc<Self> {
        let op = ViewOp::Flip(axes);
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::View(op, self.clone()),
            shape: new_shape, // Flip preserves shape
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    // --- Compute Ops ---

    pub fn cast(self: &Arc<Self>, target: DType) -> Arc<Self> {
        let op = ComputeOp::Cast(target);
        let new_shape = op.infer_shape(&[&self.shape]);

        // A same-dtype cast is an identity clone: the buffer (and its
        // strides) pass through untouched. Any real cast materializes a
        // fresh contiguous buffer in the target dtype, so input strides —
        // whatever their element size — never describe the output.
        let new_strides = if target == self.dtype {
            self.strides.clone()
        } else {
            self.strides
                .as_ref()
                .map(|_| Layout::new_contiguous(new_shape.clone(), target).strides)
        };

        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: new_shape,
            strides: new_strides,
            dtype: target,
        })
    }

    pub fn affine(self: &Arc<Self>, params: AffineParams) -> Arc<Self> {
        let op = ComputeOp::Affine(params);
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape); // RequiresContiguous -> New Layout

        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn scale(self: &Arc<Self>, factor: f32) -> Arc<Self> {
        let op = ComputeOp::Scale(factor);
        let new_shape = self.shape.clone();
        let new_strides = self.calc_strides(&op, &new_shape);
        let new_dtype = op.resolve_output_dtype(self.dtype, None);

        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: new_shape,
            strides: new_strides,
            dtype: new_dtype,
        })
    }

    pub fn relu(self: &Arc<Self>) -> Arc<Self> {
        let op = ComputeOp::Relu;
        let new_dtype = op.resolve_output_dtype(self.dtype, None);
        let new_strides = self.calc_strides(&op, &self.shape);
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: new_dtype,
        })
    }

    pub fn fused(self: &Arc<Self>, kernel: FusedKernel) -> Arc<Self> {
        let op = ComputeOp::Fused(kernel);
        // Fused kernels preserve shape but write a fresh contiguous buffer.
        let new_dtype = op.resolve_output_dtype(self.dtype, None);
        let new_strides = self.calc_strides(&op, &self.shape);
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: new_dtype,
        })
    }

    /// Normalize data using the specified method, emitting `out_dtype`.
    ///
    /// Only supports 2D (HW) or single-channel (HW1) shapes. Computation is in
    /// f32; pass `DType::F32` for the default float output, or another dtype to
    /// have the normalized result cast to it (the `Configurable` output rule).
    pub fn normalize(self: &Arc<Self>, method: NormalizeMethod, out_dtype: DType) -> Arc<Self> {
        let op = ComputeOp::Normalize(method, out_dtype);
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        let new_dtype = op.resolve_output_dtype(self.dtype, None);
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: new_shape,
            strides: new_strides,
            dtype: new_dtype,
        })
    }

    /// Clamp values to [min, max] range.
    pub fn clamp(self: &Arc<Self>, min: f32, max: f32) -> Arc<Self> {
        let op = ComputeOp::Clamp { min, max };
        let new_dtype = op.resolve_output_dtype(self.dtype, None);
        let new_strides = self.calc_strides(&op, &self.shape);
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: new_dtype,
        })
    }

    /// Adjust contrast: `(pixel - mean) * factor + mean`.
    pub fn adjust_contrast(self: &Arc<Self>, factor: f32) -> Arc<Self> {
        let op = ComputeOp::AdjustContrast(factor);
        let new_strides = self.calc_strides(&op, &self.shape);
        let new_dtype = op.resolve_output_dtype(self.dtype, None);
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: new_dtype,
        })
    }

    /// Adjust gamma (power-law transformation).
    pub fn adjust_gamma(self: &Arc<Self>, gamma: f32) -> Arc<Self> {
        let op = ComputeOp::AdjustGamma(gamma);
        let new_dtype = op.resolve_output_dtype(self.dtype, None);
        let new_strides = self.calc_strides(&op, &self.shape);
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: new_dtype,
        })
    }

    /// Invert pixel values: `max_val - pixel`.
    pub fn invert(self: &Arc<Self>) -> Arc<Self> {
        let op = ComputeOp::Invert;
        let new_dtype = op.resolve_output_dtype(self.dtype, None);
        let new_strides = self.calc_strides(&op, &self.shape);
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: new_dtype,
        })
    }

    /// Select a single channel from a [H, W, C] buffer, producing [H, W].
    pub fn channel_select(self: &Arc<Self>, index: usize) -> Arc<Self> {
        let op = ViewOp::ChannelSelect { index };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);
        Arc::new(Self {
            node: ExprNode::View(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    // --- Image Ops ---

    pub fn resize(self: &Arc<Self>, width: u32, height: u32, filter: FilterType) -> Arc<Self> {
        let op = ImageOp {
            kind: ImageOpKind::Resize {
                width,
                height,
                filter,
            },
        };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn blur(self: &Arc<Self>, sigma: f32) -> Arc<Self> {
        let op = ImageOp {
            kind: ImageOpKind::Blur { sigma },
        };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn threshold(self: &Arc<Self>, value: f64) -> Arc<Self> {
        let op = ImageOp {
            kind: ImageOpKind::Threshold(value),
        };
        // The kernel always materializes a fresh contiguous u8 mask, so
        // the output strides are contiguous u8 strides — never the input's
        // (whose element size may differ and which may be non-contiguous).
        let new_strides = self.calc_strides(&op, &self.shape);

        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: DType::U8,
        })
    }

    pub fn grayscale(self: &Arc<Self>) -> Arc<Self> {
        let op = ImageOp {
            kind: ImageOpKind::Grayscale,
        };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: DType::U8,
        })
    }

    /// Morphological erosion: local minimum over `ksize×ksize` neighborhood.
    pub fn erode(self: &Arc<Self>, ksize: u32, iterations: u32) -> Arc<Self> {
        let op = ImageOp {
            kind: ImageOpKind::Erode { ksize, iterations },
        };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);
        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    /// Morphological dilation: local maximum over `ksize×ksize` neighborhood.
    pub fn dilate(self: &Arc<Self>, ksize: u32, iterations: u32) -> Arc<Self> {
        let op = ImageOp {
            kind: ImageOpKind::Dilate { ksize, iterations },
        };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);
        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    /// Morphological gradient: dilate − erode (edge outline).
    pub fn morph_gradient(self: &Arc<Self>, ksize: u32) -> Arc<Self> {
        let op = ImageOp {
            kind: ImageOpKind::MorphGradient { ksize },
        };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);
        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    // --- Optimization ---

    pub fn optimize(self: &Arc<Self>) -> Arc<Self> {
        let optimized_node = match &self.node {
            ExprNode::Source(_) => return self.clone(),
            ExprNode::View(op, child) => ExprNode::View(op.clone(), child.optimize()),
            ExprNode::Compute(op, children) => {
                let opt_children: Vec<_> = children.iter().map(|c| c.optimize()).collect();
                ExprNode::Compute(op.clone(), opt_children)
            }
            ExprNode::Image(op, child) => ExprNode::Image(op.clone(), child.optimize()),
            ExprNode::Color(op, child) => ExprNode::Color(op.clone(), child.optimize()),
            ExprNode::Filter(op, child) => ExprNode::Filter(op.clone(), child.optimize()),
        };

        match optimized_node {
            ExprNode::View(ViewOp::Flip(axes1), child) => {
                if let ExprNode::View(ViewOp::Flip(ref axes2), ref grandchild) = &child.node {
                    if axes1 == *axes2 {
                        return grandchild.clone();
                    }
                }
                self.rebuild(ExprNode::View(ViewOp::Flip(axes1), child))
            }

            ExprNode::View(ViewOp::Transpose(p1), child) => {
                if let ExprNode::View(ViewOp::Transpose(ref p2), ref grandchild) = &child.node {
                    let merged: Vec<usize> = p1.iter().map(|&i| p2[i]).collect();
                    let is_identity = merged.iter().enumerate().all(|(i, &x)| i == x);
                    if is_identity {
                        return grandchild.clone();
                    } else {
                        return Arc::new(Self {
                            node: ExprNode::View(ViewOp::Transpose(merged), grandchild.clone()),
                            shape: self.shape.clone(),
                            // We must re-calc strides here for the optimized node in a real implementation
                            // For prototype, reusing self fields via rebuild might be slightly inaccurate if
                            // fusion changed layout semantics, but for Transpose fusion it should be consistent.
                            // Ideally, optimize() returns a new clean expression with recalculated metadata.
                            strides: self.strides.clone(),
                            dtype: self.dtype,
                        });
                    }
                }
                self.rebuild(ExprNode::View(ViewOp::Transpose(p1), child))
            }

            ExprNode::Compute(op1, children) => {
                if children.len() == 1 {
                    let child = &children[0];

                    // Cast optimization: eliminate redundant casts
                    if let ComputeOp::Cast(target_dtype) = &op1 {
                        // Optimization 1: Identity cast (cast to same dtype as child)
                        // Example: u8 input -> cast(u8) -> output
                        // Result: eliminate the cast entirely
                        if child.dtype == *target_dtype {
                            return child.clone();
                        }

                        // Optimization 2: Consecutive casts (cast(A) -> cast(B) -> cast(A))
                        // Example: cast(f32) -> cast(u8) -> cast(f32)
                        // Result: just cast(f32)
                        if let ExprNode::Compute(ComputeOp::Cast(_), ref grand_children) =
                            &child.node
                        {
                            // Skip the intermediate cast, cast directly from grandchild
                            return Arc::new(Self {
                                node: ExprNode::Compute(
                                    ComputeOp::Cast(*target_dtype),
                                    grand_children.clone(),
                                ),
                                shape: self.shape.clone(),
                                strides: self.strides.clone(),
                                dtype: *target_dtype,
                            });
                        }
                    }

                    // Try fusing scalar operations
                    if let ExprNode::Compute(ref op2, ref grand_children) = &child.node {
                        if grand_children.len() == 1 {
                            // Each op's lowering may depend on the dtype it
                            // would have received unfused; the kernel's output
                            // is pinned to the chain's planned dtype.
                            let inner_input_dtype = grand_children[0].dtype;
                            let outer_input_dtype = child.dtype;
                            if let Some(fused) = try_fuse(
                                &op1,
                                op2,
                                inner_input_dtype,
                                outer_input_dtype,
                                self.dtype,
                            ) {
                                return Arc::new(Self {
                                    node: ExprNode::Compute(fused, grand_children.clone()),
                                    shape: self.shape.clone(),
                                    strides: self.strides.clone(),
                                    dtype: self.dtype,
                                });
                            }
                        }
                    }
                }
                self.rebuild(ExprNode::Compute(op1, children))
            }

            _ => self.rebuild(optimized_node),
        }
    }

    fn rebuild(&self, node: ExprNode) -> Arc<Self> {
        Arc::new(Self {
            node,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        })
    }

    // --- Introspection ---

    /// Returns a text visualization of the execution graph.
    pub fn explain(&self) -> String {
        self.explain_impl(0)
    }

    fn explain_impl(&self, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let mut info = format!("{}Node: {:?}\n", indent, self.node_type_name());
        info.push_str(&format!("{}  Shape: {:?}\n", indent, self.shape));
        info.push_str(&format!("{}  Strides: {:?}\n", indent, self.strides));
        info.push_str(&format!("{}  DType: {:?}\n", indent, self.dtype));

        match &self.node {
            ExprNode::Source(_) => {
                info.push_str(&format!("{indent}  Source: ViewBuffer\n"));
            }
            ExprNode::View(op, child) => {
                info.push_str(&format!("{indent}  Op: {op:?}\n"));
                info.push_str(&child.explain_impl(depth + 1));
            }
            ExprNode::Compute(op, children) => {
                info.push_str(&format!("{indent}  Op: {op:?}\n"));
                for child in children {
                    info.push_str(&child.explain_impl(depth + 1));
                }
            }
            ExprNode::Image(op, child) => {
                info.push_str(&format!("{indent}  Op: {op:?}\n"));
                info.push_str(&child.explain_impl(depth + 1));
            }
            ExprNode::Color(op, child) => {
                info.push_str(&format!("{indent}  Op: {op:?}\n"));
                info.push_str(&child.explain_impl(depth + 1));
            }
            ExprNode::Filter(op, child) => {
                info.push_str(&format!("{indent}  Op: {op:?}\n"));
                info.push_str(&child.explain_impl(depth + 1));
            }
        }
        info
    }

    fn node_type_name(&self) -> &'static str {
        match &self.node {
            ExprNode::Source(_) => "Source",
            ExprNode::View(_, _) => "View",
            ExprNode::Compute(_, _) => "Compute",
            ExprNode::Image(_, _) => "Image",
            ExprNode::Color(_, _) => "Color",
            ExprNode::Filter(_, _) => "Filter",
        }
    }

    // --- Execution Planning ---

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
            ExprNode::View(op, child) => {
                let mut plan = child.build_plan();
                plan.steps.push(PlanStep::View(op.clone()));
                plan
            }
            ExprNode::Compute(op, children) => {
                let mut plan = children[0].build_plan();

                if op.memory_effect() == MemoryEffect::RequiresContiguous
                    && (plan_ends_in_view(&plan) || !plan.source.layout.is_contiguous())
                {
                    plan.steps.push(PlanStep::MaterializeContiguous);
                }

                plan.steps.push(PlanStep::Compute(op.clone()));
                plan
            }
            ExprNode::Image(op, child) => {
                let mut plan = child.build_plan();

                if op.memory_effect() == MemoryEffect::RequiresContiguous
                    && (plan_ends_in_view(&plan) || !plan.source.layout.is_contiguous())
                {
                    plan.steps.push(PlanStep::MaterializeContiguous);
                }

                plan.steps.push(PlanStep::Image(op.clone()));
                plan
            }
            ExprNode::Color(op, child) => {
                let mut plan = child.build_plan();
                if plan_ends_in_view(&plan) || !plan.source.layout.is_contiguous() {
                    plan.steps.push(PlanStep::MaterializeContiguous);
                }
                plan.steps.push(PlanStep::Color(op.clone()));
                plan
            }
            ExprNode::Filter(op, child) => {
                let mut plan = child.build_plan();
                if plan_ends_in_view(&plan) || !plan.source.layout.is_contiguous() {
                    plan.steps.push(PlanStep::MaterializeContiguous);
                }
                plan.steps.push(PlanStep::Filter(op.clone()));
                plan
            }
        }
    }
}

fn plan_ends_in_view(plan: &ExecutionPlan) -> bool {
    matches!(plan.steps.last(), Some(PlanStep::View(_)))
}

// --- Helper for Fusion ---

/// Lowers one `ComputeOp` into the scalar ops a `FusedKernel` runs.
///
/// `input_dtype` is needed because some lowerings are dtype-dependent:
/// `Invert` and the gamma family use the dtype's value range, so the same op
/// becomes different scalar work for `u8` than for `f32`.
///
/// `is_outer` marks the op at the end of the chain. An outer `Cast` lowers to
/// *no* scalar ops at all: the kernel already converts its `f32` result to
/// `FusedKernel::out_dtype` on write, and `try_fuse` pins that dtype to what
/// the unfused chain would have produced — so emitting a cast here would apply
/// the conversion twice.
fn extract_ops(
    op: &ComputeOp,
    input_dtype: DType,
    is_outer: bool,
    list: &mut Vec<ScalarOp>,
) -> bool {
    // The float-promoting scalar family is excluded for f64 inputs: the
    // dtype contract preserves f64 (and the unfused runtime now computes in
    // f64), but the fused kernel computes in f32 — fusing would silently
    // drop precision. f64 chains simply stay unfused.
    let promote_family_fusable = input_dtype != DType::F64;
    match op {
        ComputeOp::Scale(s) if promote_family_fusable => {
            list.push(ScalarOp::Mul(*s));
            true
        }
        ComputeOp::Relu if promote_family_fusable => {
            list.push(ScalarOp::Relu);
            true
        }
        ComputeOp::Clamp { min, max } if promote_family_fusable => {
            list.push(ScalarOp::Clamp(*min, *max));
            true
        }
        // Gamma is scan-free and lowers exactly to its unfused formula:
        // `((x / max).clamp(0, 1)).powf(g) * max`, max = the input dtype's
        // value range for integers, 1 for float inputs (matching
        // `apply_adjust_gamma` via the same `norm_range_max_f32`).
        ComputeOp::AdjustGamma(g) if promote_family_fusable => {
            let max_val: f32 = input_dtype.norm_range_max_f32();
            if max_val != 1.0 {
                list.push(ScalarOp::Div(max_val));
            }
            list.push(ScalarOp::Clamp(0.0, 1.0));
            list.push(ScalarOp::Pow(*g));
            if max_val != 1.0 {
                list.push(ScalarOp::Mul(max_val));
            }
            true
        }
        // Invert is `max - x`, which is `-x + max` (bit-identical in IEEE
        // arithmetic for the float case, exact integers for u8/u16 in f32).
        // Only the dtypes whose unfused output round-trips exactly through
        // the kernel's f32 compute are fused: f64 would lose precision, and
        // the remaining integer dtypes take an unfused fallback path with
        // different output-dtype behavior.
        ComputeOp::Invert => {
            let max_val: f32 = match input_dtype {
                DType::U8 => 255.0,
                DType::U16 => 65535.0,
                DType::F32 => 1.0,
                _ => return false,
            };
            list.push(ScalarOp::Mul(-1.0));
            list.push(ScalarOp::Add(max_val));
            true
        }
        // A cast is the kernel's own read/write conversion:
        // - as the chain's last op, the kernel's out_dtype performs it;
        // - mid-chain, only cast-to-f32 is a no-op (the kernel computes in
        //   f32 anyway); other mid-chain casts quantize and must materialize.
        ComputeOp::Cast(target) => is_outer || *target == DType::F32,
        // An existing kernel can be extended only while its result is still
        // raw f32 — a non-f32 out_dtype is a quantization step that later
        // ops must observe.
        ComputeOp::Fused(k) => {
            if !is_outer && k.out_dtype != DType::F32 {
                return false;
            }
            list.extend(k.ops.iter().cloned());
            true
        }
        _ => false,
    }
}

/// Try to fuse two adjacent compute ops into a single `FusedKernel`.
///
/// `inner` runs first (on data of `inner_input_dtype`), then `outer` (on
/// data of `outer_input_dtype` — inner's planned output). The kernel's
/// `out_dtype` is pinned to `planned_out_dtype`, the dtype the *unfused*
/// chain would produce, so fusion can never change the planned schema —
/// plan == exec holds by construction for any fused combination.
fn try_fuse(
    outer: &ComputeOp,
    inner: &ComputeOp,
    inner_input_dtype: DType,
    outer_input_dtype: DType,
    planned_out_dtype: DType,
) -> Option<ComputeOp> {
    let mut ops = Vec::new();

    if !extract_ops(inner, inner_input_dtype, false, &mut ops) {
        return None;
    }

    if !extract_ops(outer, outer_input_dtype, true, &mut ops) {
        return None;
    }

    Some(ComputeOp::Fused(FusedKernel {
        ops,
        out_dtype: planned_out_dtype,
    }))
}
