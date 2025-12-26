use std::sync::Arc;
use crate::buffer::ViewBuffer;
use crate::dtype::DType;
use crate::ops::{ViewOp, ComputeOp, ImageOp, ImageOpKind, FilterType, Op};
use crate::ops::affine::AffineParams;
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::layout::Layout;

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
    pub strides: Option<Vec<isize>>, // NEW: Track strides symbolically
    pub dtype: DType,
}

impl ViewExpr {
    pub fn new_source(buffer: ViewBuffer) -> Arc<Self> {
        Arc::new(Self {
            shape: buffer.shape().to_vec(),
            strides: Some(buffer.strides_bytes().to_vec()), // Initial strides known
            dtype: buffer.dtype(),
            node: ExprNode::Source(Arc::new(buffer)),
        })
    }

    // Helper to calculate next strides
    fn calc_strides(&self, op: &impl Op, new_shape: &[usize]) -> Option<Vec<isize>> {
        if let Some(current_strides) = &self.strides {
            // Special handling for Reshape which requires DType to calc new contiguous strides
            // if we determined it's valid.
            // Since Op trait definition was limited, we handle specific logic here or accept None.
            // For now, simple delegation.
            let res = op.infer_strides(&self.shape, current_strides);
            
            // If Op returned None, but we know it produces contiguous output (RequiresContiguous),
            // we can calculate default strides here using self.dtype (or new dtype).
            if res.is_none() {
                if op.memory_effect() == crate::ops::MemoryEffect::RequiresContiguous {
                    let new_dtype = op.infer_dtype(&[self.dtype]);
                    // Calculate default contiguous strides
                    let l = Layout::new_contiguous(new_shape.to_vec(), new_dtype);
                    return Some(l.strides);
                }
                
                // Special Check for Reshape:
                // If it is a ViewOp::Reshape, we need to check contiguity.
                // We use LayoutFacts logic.
                // If contiguous, we return new contiguous strides.
                // If NOT contiguous, Reshape as a view is INVALID.
                // We can detect this here!
                // (This check assumes we can cast Op to ViewOp to check variant, which is hard generically,
                // but we are in builder methods below).
            }
            res
        } else {
            // If input strides are unknown, output usually unknown unless forced contiguous
            if op.memory_effect() == crate::ops::MemoryEffect::RequiresContiguous {
                let new_dtype = op.infer_dtype(&[self.dtype]);
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
            let facts = crate::layout::LayoutFacts::new(&self.shape, strides, self.dtype, 0);
            if !facts.is_contiguous() {
                 // In a full implementation, we might auto-insert a Materialize op here.
                 // For now, we allow the Planner to catch it (or panic) but we warn/mark strides None.
                 // But since we want to "detect invalid views during definition":
                panic!("Invalid View: Cannot reshape non-contiguous view without copying. Input strides: {:?}", strides);
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
        
        // Stride Preserving: Strides match input (in elements).
        // But stride bytes change if element size changes!
        // calc_strides needs to account for this scaling.
        // Current op.infer_strides for StridePreserving just copies input bytes strides.
        // This is WRONG if dtype size changes.
        // We need to re-scale strides based on ratio of type sizes.
        
        let new_strides = if let Some(input_strides) = &self.strides {
            let src_size = self.dtype.size_of();
            let dst_size = target.size_of();
            if src_size == dst_size {
                Some(input_strides.clone())
            } else {
                // Check if all strides are divisible
                // We use i64 to prevent overflow during intermediate mult and handle negative strides
                let valid = input_strides.iter().all(|&s| (s as i64 * dst_size as i64) % src_size as i64 == 0);
                if valid {
                    Some(input_strides.iter().map(|&s| ((s as i64 * dst_size as i64) / src_size as i64) as isize).collect())
                } else {
                    None // Should not happen for aligned buffers
                }
            }
        } else {
            None
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
        // StridePreserving, same dtype -> same strides
        let new_strides = self.strides.clone(); 

        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn relu(self: &Arc<Self>) -> Arc<Self> {
        let op = ComputeOp::Relu;
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        })
    }

    // --- Image Ops ---

    pub fn resize(self: &Arc<Self>, width: u32, height: u32, filter: FilterType) -> Arc<Self> {
        let op = ImageOp { kind: ImageOpKind::Resize { width, height, filter } };
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
        let op = ImageOp { kind: ImageOpKind::Blur { sigma } };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn threshold(self: &Arc<Self>, value: u8) -> Arc<Self> {
        let op = ImageOp { kind: ImageOpKind::Threshold(value) };
        // Output U8, Input might be U8. StridePreserving.
        // If input was U8, strides preserved.
        let new_strides = if self.dtype == DType::U8 {
            self.strides.clone()
        } else {
            // If casting occurred (implicit or explicit in op logic), strides might scale.
            // Threshold op usually implies U8->U8 or similar.
            // Assuming U8->U8 for now.
            self.strides.clone()
        };

        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: self.shape.clone(),
            strides: new_strides,
            dtype: DType::U8,
        })
    }

    pub fn grayscale(self: &Arc<Self>) -> Arc<Self> {
        let op = ImageOp { kind: ImageOpKind::Grayscale };
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::Image(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: DType::U8,
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
                            // We must re-calc strides here for the optimized node in a real implementation
                            // For prototype, reusing self fields via rebuild might be slightly inaccurate if
                            // fusion changed layout semantics, but for Transpose fusion it should be consistent.
                            // Ideally, optimize() returns a new clean expression with recalculated metadata.
                            strides: self.strides.clone(), 
                            dtype: self.dtype
                        });
                    }
                }
                self.rebuild(ExprNode::View(ViewOp::Transpose(p1), child))
            },

            ExprNode::Compute(op1, children) => {
                if children.len() == 1 {
                    let child = &children[0];
                    if let ExprNode::Compute(ref op2, ref grand_children) = &child.node {
                        if let Some(fused) = try_fuse(&op1, op2) {
                            return Arc::new(Self {
                                node: ExprNode::Compute(fused, grand_children.clone()),
                                shape: self.shape.clone(),
                                strides: self.strides.clone(),
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
            _ => false,
        }
    }

    if !extract_ops(inner, &mut ops) {
        return None;
    }

    if !extract_ops(outer, &mut ops) {
        return None;
    }

    Some(ComputeOp::Fused(FusedKernel { ops }))
}