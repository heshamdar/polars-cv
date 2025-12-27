use std::sync::Arc;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;
use crate::core::layout::Layout;
use crate::execution::{ExecutionPlan, PlanStep};
use crate::ops::affine::AffineParams;
use crate::ops::cost::{OpCost, OpCostReport};
use crate::ops::io::{PlaceholderMeta, SinkFormat, SourceFormat};
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
            },
            ViewDto::Materialize => {
                // Explicit materialization handled by Planner
                self.clone()
            }
        }
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
            if res.is_none() && op.memory_effect() == MemoryEffect::RequiresContiguous {
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
            ExprNode::Sink { .. } => "Sink",
        }
    }

    // --- Cost Reporting ---

    /// Generates a cost report for the entire pipeline.
    pub fn cost_report(&self) -> PipelineCostReport {
        let mut operations = Vec::new();
        self.collect_costs(&mut operations);

        let total_allocations = operations
            .iter()
            .filter(|r| r.intrinsic_cost == OpCost::Allocating)
            .count();

        let io_operations = operations
            .iter()
            .filter(|r| r.intrinsic_cost == OpCost::IO)
            .count();

        let dtype_changes: Vec<_> = operations
            .iter()
            .filter_map(|r| {
                r.dtype_change
                    .map(|(from, to)| (r.op_name.to_string(), from, to))
            })
            .collect();

        PipelineCostReport {
            operations,
            total_allocations,
            dtype_changes,
            io_operations,
        }
    }

    fn collect_costs(&self, ops: &mut Vec<OpCostReport>) {
        match &self.node {
            ExprNode::Source(_) => {}
            ExprNode::LazySource { format, .. } => {
                ops.push(OpCostReport::new(format.name(), format.cost()));
            }
            ExprNode::Placeholder(_) => {}
            ExprNode::View(op, child) => {
                child.collect_costs(ops);
                ops.push(OpCostReport::new(op.name(), op.intrinsic_cost()));
            }
            ExprNode::Compute(op, children) => {
                for child in children {
                    child.collect_costs(ops);
                }
                let input_dtype = children.first().map(|c| c.dtype).unwrap_or(DType::U8);
                let output_dtype = op.infer_dtype(&[input_dtype]);
                if input_dtype != output_dtype {
                    ops.push(OpCostReport::with_dtype_change(
                        op.name(),
                        op.intrinsic_cost(),
                        input_dtype,
                        output_dtype,
                    ));
                } else {
                    ops.push(OpCostReport::new(op.name(), op.intrinsic_cost()));
                }
            }
            ExprNode::Image(op, child) => {
                child.collect_costs(ops);
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
                    ops.push(OpCostReport::new(op.name(), op.intrinsic_cost()));
                }
            }
            ExprNode::Sink { format, input } => {
                input.collect_costs(ops);
                ops.push(OpCostReport::new(format.name(), format.cost()));
            }
        }
    }

    /// Returns a human-readable cost explanation.
    pub fn explain_costs(&self) -> String {
        let report = self.cost_report();
        let mut output = String::new();

        output.push_str("Pipeline Cost Summary:\n");
        output.push_str(&format!("  Operations: {}\n", report.operations.len()));
        output.push_str(&format!("  Allocations: {}\n", report.total_allocations));
        output.push_str(&format!(
            "  DType changes: {}\n",
            report.dtype_changes.len()
        ));
        output.push_str(&format!("  I/O operations: {}\n", report.io_operations));
        output.push_str("\nDetails:\n");

        for op in &report.operations {
            let dtype_info = if let Some((from, to)) = op.dtype_change {
                format!(" ({from:?} -> {to:?})")
            } else {
                String::new()
            };
            output.push_str(&format!(
                "  {} [{}]{}\n",
                op.op_name,
                op.intrinsic_cost.symbol(),
                dtype_info
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
