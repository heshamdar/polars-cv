use std::sync::Arc;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::core::layout::Layout;
use crate::execution::{ExecutionPlan, PlanStep};
use crate::ops::affine::AffineParams;
use crate::ops::cost::{OpCost, OpCostReport};
use crate::ops::io::{PlaceholderMeta, SinkFormat, SourceFormat};
use crate::ops::phash::PerceptualHashOp;
use crate::ops::scalar::{FusedKernel, ScalarOp};
use crate::ops::traits::MemoryEffect;
use crate::ops::{
    ComputeOp, FilterType, ImageOp, ImageOpKind, NormalizeMethod, Op, ViewDto, ViewOp,
};

/// A node in the expression graph.
#[derive(Debug, Clone)]
pub enum ExprNode {
    /// Concrete source data.
    Source(Arc<ViewBuffer>),

    /// Lazy source - data stored but not decoded until execution.
    LazySource {
        format: SourceFormat,
        data: Arc<[u8]>,
    },

    /// Placeholder - pipeline defined without data, shape/dtype provided at bind time.
    Placeholder(PlaceholderMeta),

    /// View operation (zero-copy).
    View(ViewOp, Arc<ViewExpr>),

    /// Compute operation (allocating).
    Compute(ComputeOp, Vec<Arc<ViewExpr>>),

    /// Image processing operation.
    Image(ImageOp, Arc<ViewExpr>),

    /// Perceptual hash operation.
    PerceptualHash(PerceptualHashOp, Arc<ViewExpr>),

    /// Terminal sink specifying output format.
    Sink {
        format: SinkFormat,
        input: Arc<ViewExpr>,
    },
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

    /// Creates a lazy source that will be decoded at execution time.
    pub fn new_lazy_source(format: SourceFormat, data: Vec<u8>, dtype: DType) -> Arc<Self> {
        Arc::new(Self {
            shape: vec![], // Unknown until execution
            strides: None,
            dtype,
            node: ExprNode::LazySource {
                format,
                data: data.into(),
            },
        })
    }

    /// Creates a placeholder for context-free pipeline definition.
    pub fn new_placeholder(meta: PlaceholderMeta) -> Arc<Self> {
        Arc::new(Self {
            shape: meta.expected_shape.clone().unwrap_or_default(),
            strides: None,
            dtype: meta.expected_dtype.unwrap_or(DType::U8),
            node: ExprNode::Placeholder(meta),
        })
    }

    // --- Sink Operations ---

    /// Terminates the pipeline with a specific output format.
    pub fn sink(self: &Arc<Self>, format: SinkFormat) -> Arc<Self> {
        Arc::new(Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            node: ExprNode::Sink {
                format,
                input: self.clone(),
            },
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
            },
            ViewDto::Compute(compute) => match compute {
                ComputeOp::Cast(dtype) => self.cast(dtype),
                ComputeOp::Affine(params) => self.affine(params),
                ComputeOp::Scale(f) => self.scale(f),
                ComputeOp::Relu => self.relu(),
                ComputeOp::Fused(kernel) => self.fused(kernel),
                ComputeOp::Normalize(method) => self.normalize(method),
                ComputeOp::Clamp { min, max } => self.clamp(min, max),
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
                ImageOpKind::Rotate { .. } => {
                    // Arbitrary rotation is handled in the execution layer
                    // Here we just create the expression node
                    Arc::new(Self {
                        shape: img.infer_shape(&[&self.shape]),
                        strides: None, // Rotation produces contiguous output
                        dtype: self.dtype,
                        node: ExprNode::Image(img, self.clone()),
                    })
                }
            },
            ViewDto::PerceptualHash(op) => self.perceptual_hash(op),
            ViewDto::Materialize => {
                // Explicit materialization handled by Planner
                self.clone()
            }
            ViewDto::Geometry(_geom) => {
                // Geometry operations are handled separately at the polars-cv level
                // They operate on contour data, not on ViewBuffers directly
                // The plugin layer converts between contour representation and buffer
                self.clone()
            }
            ViewDto::Binary { .. } | ViewDto::ApplyMask { .. } => {
                // Binary and ApplyMask operations are graph-level operations
                // They require access to other nodes' buffers and are handled
                // by the graph executor, not ViewExpr
                panic!(
                    "Binary and ApplyMask operations cannot be applied via ViewExpr. \
                     Use graph-level execution to resolve node references."
                )
            }
            ViewDto::Reduction(_) => {
                // Reduction operations change domain from Buffer to Scalar
                // They are handled by the graph executor to properly manage
                // the domain transition
                panic!(
                    "Reduction operations cannot be applied via ViewExpr. \
                     Use graph-level execution to handle domain transitions."
                )
            }
            ViewDto::Histogram(_) => {
                // Histogram operations change domain (counts/normalized/edges → Vector)
                // or preserve it (quantized → Buffer)
                // They are handled by the graph executor to properly manage
                // the domain transition and output conversion
                panic!(
                    "Histogram operations cannot be applied via ViewExpr. \
                     Use graph-level execution to handle domain transitions."
                )
            }
            ViewDto::ResizeScale { .. }
            | ViewDto::ResizeToHeight { .. }
            | ViewDto::ResizeToWidth { .. }
            | ViewDto::ResizeMax { .. }
            | ViewDto::ResizeMin { .. } => {
                // Deferred resize operations need access to input buffer dimensions
                // They are handled by the graph executor
                panic!(
                    "Deferred resize operations cannot be applied via ViewExpr. \
                     Use graph-level execution to compute dimensions."
                )
            }
            ViewDto::Pad { .. } | ViewDto::PadToSize { .. } | ViewDto::Letterbox { .. } => {
                // Padding operations are handled by the graph executor
                // to support constant padding and dimension computation
                panic!(
                    "Padding operations cannot be applied via ViewExpr. \
                     Use graph-level execution."
                )
            }
            ViewDto::ExtractShape => {
                // ExtractShape changes domain from Buffer to Vector
                // It is handled by the graph executor
                panic!(
                    "ExtractShape cannot be applied via ViewExpr. \
                     Use graph-level execution to handle domain transitions."
                )
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
                let new_dtype = op.infer_dtype(&[self.dtype]);
                let l = Layout::new_contiguous(new_shape.to_vec(), new_dtype);
                return Some(l.strides);
            }

            res
        } else {
            // If input strides are unknown, calculate contiguous strides for allocating ops
            // or ops that require contiguous input
            let effect = op.memory_effect();
            if effect == MemoryEffect::RequiresContiguous
                || op.intrinsic_cost() == crate::ops::OpCost::Allocating
            {
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
                let valid = input_strides
                    .iter()
                    .all(|&s| (s as i64 * dst_size as i64) % src_size as i64 == 0);
                if valid {
                    Some(
                        input_strides
                            .iter()
                            .map(|&s| ((s as i64 * dst_size as i64) / src_size as i64) as isize)
                            .collect(),
                    )
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

    pub fn fused(self: &Arc<Self>, kernel: FusedKernel) -> Arc<Self> {
        let op = ComputeOp::Fused(kernel);
        // Fused preserves strides and shape
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        })
    }

    /// Normalize data using the specified method.
    /// Only supports 2D (HW) or single-channel (HW1) shapes with F32 dtype.
    pub fn normalize(self: &Arc<Self>, method: NormalizeMethod) -> Arc<Self> {
        let op = ComputeOp::Normalize(method);
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    /// Clamp values to [min, max] range.
    pub fn clamp(self: &Arc<Self>, min: f32, max: f32) -> Arc<Self> {
        let op = ComputeOp::Clamp { min, max };
        Arc::new(Self {
            node: ExprNode::Compute(op, vec![self.clone()]),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
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

    pub fn threshold(self: &Arc<Self>, value: u8) -> Arc<Self> {
        let op = ImageOp {
            kind: ImageOpKind::Threshold(value),
        };
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

    /// Compute a perceptual hash of the image.
    ///
    /// Returns a 1D u8 buffer containing the hash bytes.
    /// For a 64-bit hash (default), the shape is [8].
    pub fn perceptual_hash(self: &Arc<Self>, op: PerceptualHashOp) -> Arc<Self> {
        let new_shape = op.infer_shape(&[&self.shape]);
        let new_strides = self.calc_strides(&op, &new_shape);

        Arc::new(Self {
            node: ExprNode::PerceptualHash(op, self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: DType::U8,
        })
    }

    // --- Optimization ---

    pub fn optimize(self: &Arc<Self>) -> Arc<Self> {
        let optimized_node = match &self.node {
            ExprNode::Source(_) => return self.clone(),
            ExprNode::LazySource { .. } => return self.clone(),
            ExprNode::Placeholder(_) => return self.clone(),
            ExprNode::View(op, child) => ExprNode::View(op.clone(), child.optimize()),
            ExprNode::Compute(op, children) => {
                let opt_children: Vec<_> = children.iter().map(|c| c.optimize()).collect();
                ExprNode::Compute(op.clone(), opt_children)
            }
            ExprNode::Image(op, child) => ExprNode::Image(op.clone(), child.optimize()),
            ExprNode::PerceptualHash(op, child) => {
                ExprNode::PerceptualHash(op.clone(), child.optimize())
            }
            ExprNode::Sink { format, input } => ExprNode::Sink {
                format: format.clone(),
                input: input.optimize(),
            },
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
            ExprNode::LazySource { format, .. } => {
                info.push_str(&format!("{indent}  Format: {format:?}\n"));
            }
            ExprNode::Placeholder(meta) => {
                info.push_str(&format!("{indent}  Expected: {meta:?}\n"));
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
            ExprNode::PerceptualHash(op, child) => {
                info.push_str(&format!("{indent}  Op: {op:?}\n"));
                info.push_str(&child.explain_impl(depth + 1));
            }
            ExprNode::Sink { format, input } => {
                info.push_str(&format!("{indent}  Format: {format:?}\n"));
                info.push_str(&input.explain_impl(depth + 1));
            }
        }
        info
    }

    fn node_type_name(&self) -> &'static str {
        match &self.node {
            ExprNode::Source(_) => "Source",
            ExprNode::LazySource { .. } => "LazySource",
            ExprNode::Placeholder(_) => "Placeholder",
            ExprNode::View(_, _) => "View",
            ExprNode::Compute(_, _) => "Compute",
            ExprNode::Image(_, _) => "Image",
            ExprNode::PerceptualHash(_, _) => "PerceptualHash",
            ExprNode::Sink { .. } => "Sink",
        }
    }

    // --- Cost Reporting ---

    /// Generates a cost report for the entire pipeline.
    pub fn cost_report(&self) -> PipelineCostReport {
        let mut operations = Vec::new();
        let mut dtype_flow = Vec::new();

        // Get source dtype
        let source_dtype = self.get_source_dtype();
        dtype_flow.push(source_dtype);

        self.collect_costs(&mut operations, &mut dtype_flow);

        let total_allocations = operations
            .iter()
            .filter(|r| r.intrinsic_cost == OpCost::Allocating)
            .count();

        let zero_copy_operations = operations
            .iter()
            .filter(|r| r.intrinsic_cost == OpCost::ZeroCopy)
            .count();

        let io_operations = operations
            .iter()
            .filter(|r| r.intrinsic_cost == OpCost::IO)
            .count();

        let dtype_changes: Vec<_> = operations
            .iter()
            .filter_map(|r| {
                r.dtype_change
                    .map(|(from, to)| (r.display_name().to_string(), from, to))
            })
            .collect();

        let fusion_summary: Vec<_> = operations
            .iter()
            .filter_map(|r| r.op_description.clone())
            .collect();

        // Estimate memory: sum of allocating ops
        let estimated_memory_bytes: Option<usize> = {
            let total: usize = operations.iter().filter_map(|r| r.estimated_bytes).sum();
            if total > 0 {
                Some(total)
            } else {
                None
            }
        };

        PipelineCostReport {
            operations,
            total_allocations,
            dtype_changes,
            io_operations,
            zero_copy_operations,
            estimated_memory_bytes,
            dtype_flow,
            fusion_summary,
        }
    }

    /// Gets the source dtype for this expression.
    fn get_source_dtype(&self) -> DType {
        match &self.node {
            ExprNode::Source(buf) => buf.dtype(),
            ExprNode::LazySource { .. } => DType::U8, // Default for lazy sources
            ExprNode::Placeholder(_) => DType::U8,
            ExprNode::View(_, child) => child.get_source_dtype(),
            ExprNode::Compute(_, children) => children
                .first()
                .map(|c| c.get_source_dtype())
                .unwrap_or(DType::U8),
            ExprNode::Image(_, child) => child.get_source_dtype(),
            ExprNode::PerceptualHash(_, child) => child.get_source_dtype(),
            ExprNode::Sink { input, .. } => input.get_source_dtype(),
        }
    }

    fn collect_costs(&self, ops: &mut Vec<OpCostReport>, dtype_flow: &mut Vec<DType>) {
        match &self.node {
            ExprNode::Source(_) => {}
            ExprNode::LazySource { format, .. } => {
                let dtype = DType::U8; // Lazy sources typically produce U8
                ops.push(OpCostReport::new(format.name(), format.cost(), dtype));
                dtype_flow.push(dtype);
            }
            ExprNode::Placeholder(_) => {}
            ExprNode::View(op, child) => {
                child.collect_costs(ops, dtype_flow);
                let current_dtype = *dtype_flow.last().unwrap_or(&DType::U8);
                ops.push(OpCostReport::new(
                    op.name(),
                    op.intrinsic_cost(),
                    current_dtype,
                ));
                // View ops don't change dtype
                dtype_flow.push(current_dtype);
            }
            ExprNode::Compute(op, children) => {
                for child in children {
                    child.collect_costs(ops, dtype_flow);
                }
                let input_dtype = children.first().map(|c| c.dtype).unwrap_or(DType::U8);
                let output_dtype = op.infer_dtype(&[input_dtype]);

                let report = if let ComputeOp::Fused(kernel) = op {
                    // Create detailed fused operation report
                    let fused_names: Vec<String> =
                        kernel.ops.iter().map(|s| s.name().to_string()).collect();
                    OpCostReport::fused(fused_names, op.intrinsic_cost(), input_dtype, output_dtype)
                } else if input_dtype != output_dtype {
                    OpCostReport::with_dtype_change(
                        op.name(),
                        op.intrinsic_cost(),
                        input_dtype,
                        output_dtype,
                    )
                } else {
                    OpCostReport::new(op.name(), op.intrinsic_cost(), input_dtype)
                };
                ops.push(report);
                dtype_flow.push(output_dtype);
            }
            ExprNode::Image(op, child) => {
                child.collect_costs(ops, dtype_flow);
                let input_dtype = child.dtype;
                let output_dtype = op.infer_dtype(&[input_dtype]);
                if input_dtype != output_dtype {
                    ops.push(OpCostReport::with_dtype_change(
                        op.name(),
                        op.intrinsic_cost(),
                        input_dtype,
                        output_dtype,
                    ));
                } else {
                    ops.push(OpCostReport::new(
                        op.name(),
                        op.intrinsic_cost(),
                        input_dtype,
                    ));
                }
                dtype_flow.push(output_dtype);
            }
            ExprNode::PerceptualHash(op, child) => {
                child.collect_costs(ops, dtype_flow);
                let input_dtype = child.dtype;
                let output_dtype = op.infer_dtype(&[input_dtype]);
                ops.push(OpCostReport::with_dtype_change(
                    op.name(),
                    op.intrinsic_cost(),
                    input_dtype,
                    output_dtype,
                ));
                dtype_flow.push(output_dtype);
            }
            ExprNode::Sink { format, input } => {
                input.collect_costs(ops, dtype_flow);
                let current_dtype = *dtype_flow.last().unwrap_or(&DType::U8);
                ops.push(OpCostReport::new(
                    format.name(),
                    format.cost(),
                    current_dtype,
                ));
            }
        }
    }

    /// Returns a human-readable cost explanation.
    pub fn explain_costs(&self) -> String {
        let report = self.cost_report();
        let mut output = String::new();

        output.push_str("Pipeline Cost Summary:\n");
        output.push_str(&format!(
            "  Operations: {} ({} zero-copy, {} allocating)\n",
            report.operations.len(),
            report.zero_copy_operations,
            report.total_allocations
        ));
        output.push_str(&format!(
            "  DType changes: {}\n",
            report.dtype_changes.len()
        ));
        output.push_str(&format!("  I/O operations: {}\n", report.io_operations));

        // Show memory estimate if available
        if let Some(bytes) = report.estimated_memory_bytes {
            output.push_str(&format!("  Estimated memory: {bytes} bytes\n"));
        }

        // Show dtype flow
        if report.dtype_flow.len() > 1 {
            let flow_str: Vec<String> =
                report.dtype_flow.iter().map(|d| format!("{d:?}")).collect();
            // Deduplicate consecutive duplicates
            let deduped: Vec<&str> = flow_str
                .iter()
                .enumerate()
                .filter(|(i, s)| *i == 0 || flow_str.get(i - 1).map(|p| p != *s).unwrap_or(true))
                .map(|(_, s)| s.as_str())
                .collect();
            output.push_str(&format!("  DType flow: {}\n", deduped.join(" -> ")));
        }

        // Show fusion summary if any
        if !report.fusion_summary.is_empty() {
            output.push_str("\nFusion Summary:\n");
            for fusion in &report.fusion_summary {
                output.push_str(&format!("  {fusion}\n"));
            }
        }

        output.push_str("\nDetails:\n");

        for op in &report.operations {
            let display_name = op.display_name();
            let dtype_info = format!(" {:?} -> {:?}", op.input_dtype, op.output_dtype);
            let bytes_info = op
                .estimated_bytes
                .map(|b| format!(" ({b}B)"))
                .unwrap_or_default();
            output.push_str(&format!(
                "  {} [{}]{}{}\n",
                display_name,
                op.intrinsic_cost.symbol(),
                dtype_info,
                bytes_info
            ));
        }

        output
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
            ExprNode::PerceptualHash(op, child) => {
                let mut plan = child.build_plan();

                // Perceptual hash requires contiguous data
                match op.memory_effect() {
                    MemoryEffect::RequiresContiguous => {
                        if plan_ends_in_view(&plan) || !plan.source.layout.is_contiguous() {
                            plan.steps.push(PlanStep::MaterializeContiguous);
                        }
                    }
                    MemoryEffect::StridePreserving => {}
                    MemoryEffect::View => unreachable!(),
                }

                plan.steps.push(PlanStep::PerceptualHash(op.clone()));
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

/// Summary of costs for an entire pipeline.
#[derive(Debug)]
pub struct PipelineCostReport {
    /// Cost reports for each operation.
    pub operations: Vec<OpCostReport>,
    /// Total number of allocating operations.
    pub total_allocations: usize,
    /// List of (op_name, from_dtype, to_dtype) for dtype changes.
    pub dtype_changes: Vec<(String, DType, DType)>,
    /// Number of I/O operations.
    pub io_operations: usize,
    /// Number of zero-copy operations.
    pub zero_copy_operations: usize,
    /// Estimated total memory allocation in bytes (if shape known).
    pub estimated_memory_bytes: Option<usize>,
    /// Chain of dtypes through the pipeline.
    pub dtype_flow: Vec<DType>,
    /// Human-readable fusion descriptions.
    pub fusion_summary: Vec<String>,
}

// --- Helper for Fusion ---

fn try_fuse(outer: &ComputeOp, inner: &ComputeOp) -> Option<ComputeOp> {
    let mut ops = Vec::new();

    fn extract_ops(op: &ComputeOp, list: &mut Vec<ScalarOp>) -> bool {
        match op {
            ComputeOp::Scale(s) => {
                list.push(ScalarOp::Mul(*s));
                true
            }
            ComputeOp::Relu => {
                list.push(ScalarOp::Relu);
                true
            }
            ComputeOp::Fused(k) => {
                list.extend(k.ops.iter().cloned());
                true
            }
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
