use std::sync::Arc;
use crate::buffer::ViewBuffer;
use crate::dtype::DType;
use crate::ops::{ViewOp, ComputeOp, ImageOp, ImageOpKind, FilterType};
use crate::ops::affine::AffineParams;
use crate::ops::scalar::{FusedKernel, ScalarOp};

#[derive(Debug, Clone)]
pub enum ExprNode {
    Source(Arc<ViewBuffer>),
    View(ViewOp, Arc<ViewExpr>),
    Compute(ComputeOp, Vec<Arc<ViewExpr>>),
    Image(ImageOp, Arc<ViewExpr>),
}

#[derive(Debug, Clone)]
pub struct ViewExpr {
    pub node: ExprNode,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl ViewExpr {
    pub fn new_source(buffer: ViewBuffer) -> Arc<Self> {
        Arc::new(Self {
            shape: buffer.shape().to_vec(),
            dtype: buffer.dtype(),
            node: ExprNode::Source(Arc::new(buffer)),
        })
    }

    // --- View Ops ---

    pub fn transpose(self: &Arc<Self>, perm: Vec<usize>) -> Arc<Self> {
        let new_shape: Vec<usize> = perm.iter().map(|&i| self.shape[i]).collect();
        Arc::new(Self {
            node: ExprNode::View(ViewOp::Transpose(perm), self.clone()),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    pub fn reshape(self: &Arc<Self>, new_shape: Vec<usize>) -> Arc<Self> {
        Arc::new(Self {
            node: ExprNode::View(ViewOp::Reshape(new_shape.clone()), self.clone()),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    pub fn crop(self: &Arc<Self>, start: Vec<usize>, end: Vec<usize>) -> Arc<Self> {
        let new_shape: Vec<usize> = start.iter().zip(end.iter()).map(|(s, e)| e - s).collect();
        Arc::new(Self {
            node: ExprNode::View(ViewOp::Crop { start, end }, self.clone()),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    pub fn flip(self: &Arc<Self>, axes: Vec<usize>) -> Arc<Self> {
        Arc::new(Self {
            node: ExprNode::View(ViewOp::Flip(axes), self.clone()),
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    // --- Compute Ops ---

    pub fn cast(self: &Arc<Self>, target: DType) -> Arc<Self> {
        Arc::new(Self {
            node: ExprNode::Compute(ComputeOp::Cast(target), vec![self.clone()]),
            shape: self.shape.clone(),
            dtype: target,
        })
    }

    pub fn affine(self: &Arc<Self>, params: AffineParams) -> Arc<Self> {
        Arc::new(Self {
            node: ExprNode::Compute(ComputeOp::Affine(params), vec![self.clone()]),
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    pub fn scale(self: &Arc<Self>, factor: f32) -> Arc<Self> {
        Arc::new(Self {
            node: ExprNode::Compute(ComputeOp::Scale(factor), vec![self.clone()]),
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    pub fn relu(self: &Arc<Self>) -> Arc<Self> {
        Arc::new(Self {
            node: ExprNode::Compute(ComputeOp::Relu, vec![self.clone()]),
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    // --- Image Ops ---

    pub fn resize(self: &Arc<Self>, width: u32, height: u32, filter: FilterType) -> Arc<Self> {
        let op = ImageOp { kind: ImageOpKind::Resize { width, height, filter } };
        let new_shape = vec![height as usize, width as usize, *self.shape.last().unwrap_or(&1)];
        
        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            dtype: self.dtype,
        })
    }

    pub fn blur(self: &Arc<Self>, sigma: f32) -> Arc<Self> {
        let op = ImageOp { kind: ImageOpKind::Blur { sigma } };
        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: self.shape.clone(),
            dtype: self.dtype,
        })
    }

    pub fn threshold(self: &Arc<Self>, value: u8) -> Arc<Self> {
        let op = ImageOp { kind: ImageOpKind::Threshold(value) };
        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: self.shape.clone(),
            dtype: DType::U8, // Thresholding always produces a u8 mask/image
        })
    }

    pub fn grayscale(self: &Arc<Self>) -> Arc<Self> {
        let op = ImageOp { kind: ImageOpKind::Grayscale };
        // Output shape inference: [H, W, 3] -> [H, W, 1]
        let mut new_shape = self.shape.clone();
        if new_shape.len() >= 3 {
            new_shape[2] = 1;
        } else if new_shape.len() == 2 {
            new_shape.push(1);
        }
        
        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            dtype: DType::U8, // Grayscale conversion always produces u8 (Luma)
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
            },
            ExprNode::Image(op, child) => ExprNode::Image(op.clone(), child.optimize()),
        };

        match optimized_node {
            ExprNode::View(ViewOp::Flip(axes1), child) => {
                if let ExprNode::View(ViewOp::Flip(ref axes2), ref grandchild) = &child.node {
                    if axes1 == *axes2 {
                        return grandchild.clone(); 
                    }
                }
                self.rebuild(ExprNode::View(ViewOp::Flip(axes1), child))
            },

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
                            dtype: self.dtype
                        });
                    }
                }
                self.rebuild(ExprNode::View(ViewOp::Transpose(p1), child))
            },

            // Rule: Fuse consecutive ComputeOps (e.g. Scale -> Relu) into one Fused kernel
            ExprNode::Compute(op1, children) => {
                if children.len() == 1 {
                    let child = &children[0];
                    if let ExprNode::Compute(ref op2, ref grand_children) = &child.node {
                        if let Some(fused) = try_fuse(&op1, op2) {
                            return Arc::new(Self {
                                node: ExprNode::Compute(fused, grand_children.clone()),
                                shape: self.shape.clone(),
                                dtype: self.dtype,
                            });
                        }
                    }
                }
                self.rebuild(ExprNode::Compute(op1, children))
            },

            _ => self.rebuild(optimized_node)
        }
    }

    fn rebuild(&self, node: ExprNode) -> Arc<Self> {
        Arc::new(Self {
            node,
            shape: self.shape.clone(),
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
        info.push_str(&format!("{}  DType: {:?}\n", indent, self.dtype));
        
        match &self.node {
            ExprNode::Source(_) => {
                info.push_str(&format!("{}  Source: ViewBuffer\n", indent));
            },
            ExprNode::View(op, child) => {
                info.push_str(&format!("{}  Op: {:?}\n", indent, op));
                info.push_str(&child.explain_impl(depth + 1));
            },
            ExprNode::Compute(op, children) => {
                info.push_str(&format!("{}  Op: {:?}\n", indent, op));
                for child in children {
                    info.push_str(&child.explain_impl(depth + 1));
                }
            },
            ExprNode::Image(op, child) => {
                info.push_str(&format!("{}  Op: {:?}\n", indent, op));
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
        }
    }
}

// --- Helper for Fusion ---

fn try_fuse(outer: &ComputeOp, inner: &ComputeOp) -> Option<ComputeOp> {
    let mut ops = Vec::new();
    
    // Helper to extract scalar ops from a ComputeOp if supported
    fn extract_ops(op: &ComputeOp, list: &mut Vec<ScalarOp>) -> bool {
        match op {
            ComputeOp::Scale(s) => {
                list.push(ScalarOp::Mul(*s));
                true
            },
            ComputeOp::Relu => {
                list.push(ScalarOp::Relu);
                true
            },
            ComputeOp::Fused(k) => {
                list.extend(k.ops.iter().cloned());
                true
            },
            // Other ops (Cast, Affine) are not supported for simple scalar fusion yet
            _ => false,
        }
    }

    // 1. Extract Inner (Applied First)
    if !extract_ops(inner, &mut ops) {
        return None;
    }

    // 2. Extract Outer (Applied Second)
    if !extract_ops(outer, &mut ops) {
        return None;
    }

    Some(ComputeOp::Fused(FusedKernel { ops }))
}