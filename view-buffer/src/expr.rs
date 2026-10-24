use std::sync::Arc;

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory};
use crate::core::layout::Layout;
use crate::execution::{ExecutionPlan, PlanStep};
use crate::ops::affine::AffineParams;
use crate::ops::scalar::FusedKernel;
use crate::ops::traits::{InputLayout, MemoryEffect};
use crate::ops::{
    ComputeOp, FilterType, ImageOp, ImageOpKind, NormalizeMethod, Op, ViewDto, ViewOp,
};

/// Declares [`OptConfig`] and [`ENGINE_PASSES`] from one list, so a toggle
/// cannot exist without its catalogue entry or the other way round.
macro_rules! engine_passes {
    ($($(#[doc = $doc:literal])* $name:ident: $summary:literal),+ $(,)?) => {
        /// Which engine-tier (Tier-2) optimizations [`ViewExpr::optimize_with`]
        /// applies.
        ///
        /// Each field toggles one output-preserving rewrite so it can be A/B
        /// differential tested (output-on == output-off). Every field defaults
        /// to `true`, and the struct is `#[serde(default)]`, so a direct
        /// view-buffer caller, an older graph spec, or a spec omitting
        /// individual keys gets the full set enabled — the historical behavior.
        /// An unknown key is refused: a toggle Python names and Rust does not
        /// have would otherwise be silently ignored. Mandatory correctness
        /// lowering (materialization, stride-preserving views, the f64 fusion
        /// exclusion) is *not* represented here: it is not optional, so it has
        /// no toggle.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
        pub struct OptConfig {
            $($(#[doc = $doc])* pub $name: bool,)+
        }

        impl Default for OptConfig {
            fn default() -> Self {
                Self { $($name: true,)+ }
            }
        }

        /// Every engine-tier pass: its [`OptConfig`] field name (the
        /// `OptFlags` field Python generates from it) and a one-line summary.
        pub const ENGINE_PASSES: &[(&str, &str)] = &[$((stringify!($name), $summary)),+];
    };
}

engine_passes! {
    /// Collapse `cast(inner) ∘ cast(target)` when `inner` losslessly contains the
    /// grandchild dtype and dropping it keeps the final cast on the same
    /// conversion path (a narrowing intermediate, or a float intermediate between
    /// integer input and integer target, is kept).
    cast_chain_collapse: "Drop a redundant intermediate cast from a cast chain when the intermediate dtype losslessly holds the input and dropping it keeps the final cast's conversion (a narrowing intermediate quantizes, and a float between an integer input and an integer target saturates, so both are kept).",
    /// Drop a `cast(T)` whose child is already dtype `T`.
    cast_identity: "Drop a cast whose target dtype already equals its input dtype.",
    /// Cancel `flip(a) ∘ flip(a)` (involution).
    view_flip_involution: "Cancel two adjacent flips over the same axes (flip∘flip = id).",
    /// Merge `transpose(p1) ∘ transpose(p2)` into one (or identity).
    view_transpose_merge: "Merge two adjacent transposes into one (or into the identity).",
    /// Fuse adjacent scalar/compute ops into a single kernel.
    scalar_fusion: "Fuse adjacent scalar/compute ops into one kernel (f64 chains stay unfused — a mandatory precision guard, not this toggle).",
}

/// A node in the expression graph.
#[derive(Debug, Clone)]
pub enum ExprNode {
    /// Concrete source data.
    Source(Arc<ViewBuffer>),

    /// An engine op of any family applied to its child. The family is the
    /// op's own business ([`ViewDto`]); a node never restates it.
    Op(ViewDto, Arc<ViewExpr>),
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

    /// Validate `op` against this expression's tracked shape and dtype, then
    /// apply it — the executor's only way to build a chain.
    ///
    /// The planner checks shapes it knows; a source that carries its shape per
    /// row (a blob, a list) reaches the engine with shapes the plan never saw.
    /// Checking here turns what used to be an index-out-of-bounds panic inside
    /// a kernel into an error the row policy can handle. [`Self::apply_op`]
    /// stays for callers that construct chains they know to be valid.
    pub fn try_apply_op(
        self: &Arc<Self>,
        op: ViewDto,
    ) -> Result<Arc<Self>, crate::ops::validation::ValidationError> {
        crate::ops::validation::validate_concrete(op.as_op(), &[&self.shape], &[self.dtype])?;
        Ok(self.apply_op(op))
    }

    /// Entry point for applying a serializable operation DTO (from JSON/Plugins).
    ///
    /// This is the single construction authority: it computes the new node's
    /// shape, strides and dtype from the op's own contract. The typed builder
    /// methods (`grayscale`, `resize`, `cast`, …) are sugar that name an op and
    /// delegate here, so none of them can track metadata a contract does not
    /// declare.
    pub fn apply_op(self: &Arc<Self>, op: ViewDto) -> Arc<Self> {
        // A compute op that is another op in disguise (a lattice rotation is
        // a view) is built as that op.
        let op = op.lowered();
        if let ViewDto::View(view) = &op {
            return self.apply_view_op(view);
        }
        // A cast's strides depend on its source dtype; every other op is one
        // node.
        if let ViewDto::Compute(ComputeOp::Cast { dtype }) = &op {
            return self.cast(*dtype);
        }
        self.op_node(op)
    }

    /// Build the nodes of a view op: most are one node, a few are a chain of
    /// views, and a padded crop ends in the `pad` op.
    fn apply_view_op(self: &Arc<Self>, view: &ViewOp) -> Arc<Self> {
        match view {
            ViewOp::Transpose { .. } => self.transpose(view.axes()),
            ViewOp::Reshape { shape } => self.reshape(shape.iter().map(|&d| d as usize).collect()),
            // A flip a row declines is no node: every flip node flips.
            ViewOp::Flip { when: false, .. } => self.clone(),
            ViewOp::Flip { .. } => self.flip(view.axes()),
            ViewOp::CropClip { .. } | ViewOp::CropPad { .. } => {
                // The window placed in this input: its in-bounds part,
                // then a padded crop's padding as the `pad` op.
                let placed = view.place(&self.shape).expect("a crop has a placement");
                let cropped = self.crop(placed.start, placed.end);
                match placed.pad {
                    None => cropped,
                    Some((top, bottom, left, right, value)) => cropped.apply_op(
                        ImageOp {
                            kind: ImageOpKind::Pad {
                                top,
                                bottom,
                                left,
                                right,
                                value,
                                mode: crate::ops::PadMode::Constant,
                            },
                        }
                        .into(),
                    ),
                }
            }
            ViewOp::Crop { .. } | ViewOp::Slice { .. } => {
                let (start, end) = view.window().expect("a crop or slice has a window");
                self.crop(start, end)
            }
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
            ViewOp::ChannelSelect { index, keepdims } => {
                self.channel_select(*index as usize, *keepdims)
            }
            ViewOp::Squeeze { .. } | ViewOp::ExpandDims { .. } => self.op_node(view.clone().into()),
        }
    }

    // Helper to calculate next strides
    fn calc_strides(&self, op: &dyn Op, new_shape: &[usize]) -> Option<Vec<isize>> {
        if let Some(current_strides) = &self.strides {
            // Try to infer strides from the operation
            let res = op.infer_strides(&self.shape, current_strides);

            // If Op returned None, it means the operation produces contiguous output
            // (either because it requires contiguous input, or because it allocates a new buffer).
            // Calculate contiguous strides for the new shape.
            if res.is_none() {
                let new_dtype = op.resolve_output_dtype(self.dtype);
                let l = Layout::new_contiguous(new_shape.to_vec(), new_dtype);
                return Some(l.strides.into_vec());
            }

            res
        } else {
            // If input strides are unknown, calculate contiguous strides for allocating ops
            // or ops that require contiguous input
            if op.memory_effect() != MemoryEffect::View {
                let new_dtype = op.resolve_output_dtype(self.dtype);
                let l = Layout::new_contiguous(new_shape.to_vec(), new_dtype);
                return Some(l.strides.into_vec());
            }
            None
        }
    }

    /// Build the node of one op, whose metadata comes entirely from the op's
    /// own contract: shape from `shape()`, strides from `calc_strides` (which
    /// honours the op's declared `MemoryEffect`), dtype from its
    /// `OutputDTypeRule`. The single construction authority every builder
    /// shares, so none of them can track metadata a contract does not declare.
    ///
    /// `cast` and `reshape` keep their own builders: a same-dtype cast passes
    /// the input strides through untouched, and a reshape's output strides are
    /// contiguous whatever its input's (the planner packs a strided input
    /// first); the generic path expresses neither.
    fn op_node(self: &Arc<Self>, op: ViewDto) -> Arc<Self> {
        let contract = op.as_op();
        let new_shape = contract.shape().concrete(&[&self.shape]);
        let new_strides = self.calc_strides(contract, &new_shape);
        let new_dtype = contract.resolve_output_dtype(self.dtype);
        Arc::new(Self {
            shape: new_shape,
            strides: new_strides,
            dtype: new_dtype,
            node: ExprNode::Op(op, self.clone()),
        })
    }

    // --- View Ops ---

    pub fn transpose(self: &Arc<Self>, perm: Vec<usize>) -> Arc<Self> {
        self.op_node(ViewOp::transpose(&perm).into())
    }

    pub fn reshape(self: &Arc<Self>, new_shape: Vec<usize>) -> Arc<Self> {
        let op = ViewOp::Reshape {
            shape: new_shape.iter().map(|&d| d as u32).collect(),
        };

        // The planner packs a strided input first (`ViewOfContiguous`), so
        // the output is always contiguous.
        let new_strides = if self.strides.is_some() {
            let l = Layout::new_contiguous(new_shape.clone(), self.dtype);
            Some(l.strides.into_vec())
        } else {
            None
        };

        Arc::new(Self {
            node: ExprNode::Op(op.into(), self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
        })
    }

    pub fn crop(self: &Arc<Self>, start: Vec<usize>, end: Vec<usize>) -> Arc<Self> {
        self.op_node(ViewOp::Slice { start, end }.into())
    }

    pub fn flip(self: &Arc<Self>, axes: Vec<usize>) -> Arc<Self> {
        self.op_node(ViewOp::flip(&axes).into())
    }

    // --- Compute Ops ---

    pub fn cast(self: &Arc<Self>, target: DType) -> Arc<Self> {
        let op = ComputeOp::Cast { dtype: target };
        let new_shape = op.shape().concrete(&[&self.shape]);

        // A same-dtype cast is an identity clone: the buffer (and its
        // strides) pass through untouched. Any real cast materializes a
        // fresh contiguous buffer in the target dtype, so input strides —
        // whatever their element size — never describe the output.
        let new_strides = if target == self.dtype {
            self.strides.clone()
        } else {
            self.strides.as_ref().map(|_| {
                Layout::new_contiguous(new_shape.clone(), target)
                    .strides
                    .into_vec()
            })
        };

        Arc::new(Self {
            node: ExprNode::Op(op.into(), self.clone()),
            shape: new_shape,
            strides: new_strides,
            dtype: target,
        })
    }

    pub fn affine(self: &Arc<Self>, params: AffineParams) -> Arc<Self> {
        self.op_node(ComputeOp::Affine(params).into())
    }

    pub fn scale(self: &Arc<Self>, factor: f32) -> Arc<Self> {
        self.op_node(ComputeOp::Scale { factor }.into())
    }

    pub fn relu(self: &Arc<Self>) -> Arc<Self> {
        self.op_node(ComputeOp::Relu.into())
    }

    pub fn fused(self: &Arc<Self>, kernel: FusedKernel) -> Arc<Self> {
        self.op_node(ComputeOp::Fused(kernel).into())
    }

    /// Normalize by the image's own statistics (`method`), emitting
    /// `out_dtype`: the `normalize` op. Computation is in f32; pass
    /// `DType::F32` for the default float output, or another dtype to have
    /// the normalized result cast to it (folded into the op's `Fixed` output
    /// rule).
    pub fn normalize(self: &Arc<Self>, method: NormalizeMethod, out_dtype: DType) -> Arc<Self> {
        self.op_node(
            ComputeOp::Normalize {
                method,
                out_dtype: Some(out_dtype),
            }
            .into(),
        )
    }

    /// Normalize each channel by given statistics, `(x - mean[c]) / std[c]`,
    /// emitting `out_dtype`: the `normalize_preset` op.
    pub fn normalize_preset(
        self: &Arc<Self>,
        mean: Vec<f32>,
        std: Vec<f32>,
        out_dtype: DType,
    ) -> Arc<Self> {
        self.op_node(
            ComputeOp::NormalizePreset {
                mean,
                std,
                out_dtype: Some(out_dtype),
            }
            .into(),
        )
    }

    /// Clamp values to [min, max] range.
    pub fn clamp(self: &Arc<Self>, min: f32, max: f32) -> Arc<Self> {
        self.op_node(ComputeOp::Clamp { min, max }.into())
    }

    /// Adjust contrast: `(pixel - mean) * factor + mean`.
    pub fn adjust_contrast(self: &Arc<Self>, factor: f32) -> Arc<Self> {
        self.op_node(ComputeOp::AdjustContrast { factor }.into())
    }

    /// Adjust gamma (power-law transformation).
    pub fn adjust_gamma(self: &Arc<Self>, gamma: f32) -> Arc<Self> {
        self.op_node(ComputeOp::AdjustGamma { gamma }.into())
    }

    /// Invert pixel values: `max_val - pixel`.
    pub fn invert(self: &Arc<Self>) -> Arc<Self> {
        self.op_node(ComputeOp::Invert.into())
    }

    /// Select a single channel from a [H, W, C] buffer, producing [H, W].
    pub fn channel_select(self: &Arc<Self>, index: usize, keepdims: bool) -> Arc<Self> {
        self.op_node(
            ViewOp::ChannelSelect {
                index: index as u32,
                keepdims,
            }
            .into(),
        )
    }

    // --- Image Ops ---

    // The image builders below are sugar over `apply_op` — they name the op and
    // delegate. `apply_op` (through `op_node`) is the single authority for
    // output shape/strides/dtype, so a builder cannot track metadata its
    // contract does not declare (the bug that `grayscale`/`threshold` carried
    // when they built the node by hand and hardcoded `DType::U8`).

    pub fn resize(self: &Arc<Self>, width: u32, height: u32, filter: FilterType) -> Arc<Self> {
        self.apply_op(
            ImageOp {
                kind: ImageOpKind::Resize {
                    width,
                    height,
                    filter,
                },
            }
            .into(),
        )
    }

    pub fn blur(self: &Arc<Self>, sigma: f32) -> Arc<Self> {
        self.apply_op(
            ImageOp {
                kind: ImageOpKind::Blur { sigma },
            }
            .into(),
        )
    }

    pub fn threshold(self: &Arc<Self>, value: f64) -> Arc<Self> {
        self.apply_op(
            ImageOp {
                kind: ImageOpKind::Threshold { value },
            }
            .into(),
        )
    }

    pub fn grayscale(self: &Arc<Self>) -> Arc<Self> {
        self.apply_op(
            ImageOp {
                kind: ImageOpKind::Grayscale { keepdims: true },
            }
            .into(),
        )
    }

    /// Morphological erosion: local minimum over the footprint.
    pub fn erode(
        self: &Arc<Self>,
        ksize: u32,
        iterations: u32,
        footprint: crate::ops::Footprint,
    ) -> Arc<Self> {
        self.apply_op(
            ImageOp {
                kind: ImageOpKind::Erode {
                    ksize,
                    iterations,
                    footprint,
                },
            }
            .into(),
        )
    }

    /// Morphological dilation: local maximum over the footprint.
    pub fn dilate(
        self: &Arc<Self>,
        ksize: u32,
        iterations: u32,
        footprint: crate::ops::Footprint,
    ) -> Arc<Self> {
        self.apply_op(
            ImageOp {
                kind: ImageOpKind::Dilate {
                    ksize,
                    iterations,
                    footprint,
                },
            }
            .into(),
        )
    }

    /// Morphological gradient: dilate − erode (edge outline).
    pub fn morph_gradient(
        self: &Arc<Self>,
        ksize: u32,
        footprint: crate::ops::Footprint,
    ) -> Arc<Self> {
        self.apply_op(
            ImageOp {
                kind: ImageOpKind::MorphGradient { ksize, footprint },
            }
            .into(),
        )
    }

    // --- Optimization ---

    /// Apply the engine-tier rewrites with every toggle enabled — the
    /// convenience entry point for direct view-buffer callers and tests. See
    /// [`optimize_with`](Self::optimize_with) for the toggled form.
    pub fn optimize(self: &Arc<Self>) -> Arc<Self> {
        self.optimize_with(&OptConfig::default())
    }

    /// Apply the engine-tier rewrites gated by `cfg`, each toggle guarding one
    /// output-preserving rewrite so it can be A/B differential tested. Recurses
    /// with the same `cfg`.
    pub fn optimize_with(self: &Arc<Self>, cfg: &OptConfig) -> Arc<Self> {
        let optimized_node = match &self.node {
            ExprNode::Source(_) => return self.clone(),
            ExprNode::Op(op, child) => ExprNode::Op(op.clone(), child.optimize_with(cfg)),
        };

        match optimized_node {
            ExprNode::Op(ViewDto::View(ViewOp::Flip { axes: axes1, when }), child) => {
                if cfg.view_flip_involution {
                    if let ExprNode::Op(
                        ViewDto::View(ViewOp::Flip {
                            axes: ref axes2, ..
                        }),
                        ref grandchild,
                    ) = &child.node
                    {
                        if axes1 == *axes2 {
                            return grandchild.clone();
                        }
                    }
                }
                self.rebuild(ExprNode::Op(
                    ViewDto::View(ViewOp::Flip { axes: axes1, when }),
                    child,
                ))
            }

            ExprNode::Op(ViewDto::View(ViewOp::Transpose { axes: p1 }), child) => {
                if cfg.view_transpose_merge {
                    if let ExprNode::Op(
                        ViewDto::View(ViewOp::Transpose { axes: ref p2 }),
                        ref grandchild,
                    ) = &child.node
                    {
                        // Compose the two permutations: applying `p1` after `p2` is a
                        // single transpose by `merged[i] = p2[p1[i]]`.
                        let merged: Vec<usize> =
                            p1.iter().map(|&i| p2[i as usize] as usize).collect();
                        let is_identity = merged.iter().enumerate().all(|(i, &x)| i == x);
                        if is_identity {
                            return grandchild.clone();
                        }
                        // Build the merged transpose through the canonical `transpose`
                        // builder so its shape/strides are recomputed from the
                        // grandchild's real layout, rather than copied from `self`
                        // (whose strides described the two-transpose chain and need
                        // not match the fused node's).
                        return grandchild.transpose(merged);
                    }
                }
                self.rebuild(ExprNode::Op(
                    ViewDto::View(ViewOp::Transpose { axes: p1 }),
                    child,
                ))
            }

            ExprNode::Op(ViewDto::Compute(op1), child) => {
                // Cast optimization: eliminate redundant casts
                if let ComputeOp::Cast {
                    dtype: target_dtype,
                } = &op1
                {
                    // Optimization 1: Identity cast (cast to same dtype as child)
                    // Example: u8 input -> cast(u8) -> output
                    // Result: eliminate the cast entirely
                    if cfg.cast_identity && child.dtype == *target_dtype {
                        return child.clone();
                    }

                    // Optimization 2: Consecutive casts. Dropping the intermediate
                    // `cast(inner)` is valid only when `inner` losslessly holds the
                    // grandchild's dtype — otherwise the intermediate quantizes and
                    // must run (e.g. `f32 -> cast(u8) -> cast(f32)` must keep the u8
                    // step: 0.5 -> 1 -> 1.0, not 0.5). A narrowing intermediate is
                    // therefore preserved; scalar fusion also refuses to fuse across
                    // a non-f32 mid-chain cast, so both casts execute.
                    //
                    // Lossless is not sufficient on its own: the final cast's
                    // conversion depends on its *source* kind — int -> int wraps
                    // (`as`), float -> int rounds and saturates. Dropping a float
                    // intermediate between an integer input and an integer target
                    // would switch the final cast from the saturating path to the
                    // wrapping one (u16 300 -> f32 -> u8 is 255, u16 300 -> u8 is
                    // 44), so that shape is kept too.
                    if cfg.cast_chain_collapse {
                        if let ExprNode::Op(
                            ViewDto::Compute(ComputeOp::Cast { dtype: inner }),
                            ref grandchild,
                        ) = &child.node
                        {
                            // Dropping a float intermediate replaces its
                            // float -> target conversion with the
                            // grandchild's own (`converts_like_float`).
                            let switches_int_conversion = DTypeCategory::Float.accepts(*inner)
                                && !grandchild.dtype.converts_like_float(*target_dtype);
                            if inner.losslessly_contains(grandchild.dtype)
                                && !switches_int_conversion
                            {
                                // Skip the (lossless) intermediate cast, cast directly
                                // from grandchild.
                                return Arc::new(Self {
                                    node: ExprNode::Op(
                                        ViewDto::Compute(ComputeOp::Cast {
                                            dtype: *target_dtype,
                                        }),
                                        grandchild.clone(),
                                    ),
                                    shape: self.shape.clone(),
                                    strides: self.strides.clone(),
                                    dtype: *target_dtype,
                                });
                            }
                        }
                    }
                }

                // Try fusing scalar operations
                if cfg.scalar_fusion {
                    if let ExprNode::Op(ViewDto::Compute(ref op2), ref grandchild) = &child.node {
                        // Each op's lowering may depend on the dtype it would have
                        // received unfused; the kernel's output is pinned to the
                        // chain's planned dtype.
                        let inner_input_dtype = grandchild.dtype;
                        let outer_input_dtype = child.dtype;
                        if let Some(fused) =
                            try_fuse(&op1, op2, inner_input_dtype, outer_input_dtype, self.dtype)
                        {
                            return Arc::new(Self {
                                node: ExprNode::Op(ViewDto::Compute(fused), grandchild.clone()),
                                shape: self.shape.clone(),
                                strides: self.strides.clone(),
                                dtype: self.dtype,
                            });
                        }
                    }
                }
                self.rebuild(ExprNode::Op(ViewDto::Compute(op1), child))
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

    // --- Execution Planning ---

    /// Builds and returns an execution plan from the expression graph, with all
    /// engine-tier optimizations enabled. See [`plan_with`](Self::plan_with).
    pub fn plan(self: &Arc<Self>) -> ExecutionPlan {
        self.plan_with(&OptConfig::default())
    }

    /// Builds an execution plan, applying only the engine-tier optimizations
    /// `cfg` enables — the entry point the plugin uses to honor per-query
    /// `opt_flags`.
    pub fn plan_with(self: &Arc<Self>, cfg: &OptConfig) -> ExecutionPlan {
        let optimized_expr = self.optimize_with(cfg);
        optimized_expr.build_plan()
    }

    fn build_plan(&self) -> ExecutionPlan {
        match &self.node {
            ExprNode::Source(buf) => ExecutionPlan {
                source: buf.as_ref().clone(),
                steps: Vec::new(),
            },
            ExprNode::Op(op, child) => {
                let mut plan = child.build_plan();
                materialize_if_needed(&mut plan, op.as_op().memory_effect());
                plan.steps.push(PlanStep::Op(op.clone()));
                plan
            }
        }
    }
}

/// Pack the plan's current buffer before an op that needs a layout it may not
/// have ([`MemoryEffect::input_layout`]): after a view step, or from a source
/// without it. The one place the planner decides it, for every kind of step.
/// Each step checks the buffer it is handed and packs only when the layout is
/// not there, so a planned step over a buffer that has it returns it as it is.
fn materialize_if_needed(plan: &mut ExecutionPlan, effect: MemoryEffect) {
    let source = plan.source.layout_facts();
    let step = match effect.input_layout() {
        InputLayout::Any => return,
        InputLayout::Contiguous if plan_ends_in_view(plan) || !source.is_contiguous() => {
            PlanStep::MaterializeContiguous
        }
        InputLayout::DenseRows if plan_ends_in_view(plan) || !source.is_dense_rows() => {
            PlanStep::MaterializeDenseRows
        }
        InputLayout::Contiguous | InputLayout::DenseRows => return,
    };
    plan.steps.push(step);
}

fn plan_ends_in_view(plan: &ExecutionPlan) -> bool {
    matches!(plan.steps.last(), Some(PlanStep::Op(ViewDto::View(_))))
}

// --- Helper for Fusion ---

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
    // The kernel stores its f32 result by the float -> target rule; a cast
    // the unfused chain would make from an integer intermediate may convert
    // differently (int -> int wraps), and then it must stay unfused.
    if let ComputeOp::Cast { dtype: target } = outer {
        if !outer_input_dtype.converts_like_float(*target) {
            return None;
        }
    }

    let mut ops = Vec::new();

    if !crate::ops::elementwise::lower_to_scalars(inner, inner_input_dtype, false, &mut ops) {
        return None;
    }

    if !crate::ops::elementwise::lower_to_scalars(outer, outer_input_dtype, true, &mut ops) {
        return None;
    }

    Some(ComputeOp::Fused(FusedKernel {
        ops,
        out_dtype: planned_out_dtype,
    }))
}

#[cfg(test)]
mod dtype_contract_tests {
    use super::*;
    use crate::ops::image::{ImageOp, ImageOpKind};

    /// Every image op's tracked dtype must be the one its contract declares.
    ///
    /// `apply_op` used to hardcode `DType::U8` for `Canny` and
    /// `HistogramEqualize` — a second copy of a rule `image.rs` already states
    /// as `OutputDTypeRule::Fixed(DType::U8)`, in the module the Python
    /// planner reads the first copy from over the FFI. Change either declared
    /// rule and the planner would publish one dtype while `ViewExpr` tracked
    /// another, which is the plan/exec divergence class this crate's contracts
    /// exist to prevent.
    ///
    /// Asserting the *property* rather than the two names means a new image op
    /// that reintroduces a hardcoded dtype fails here.
    #[test]
    fn image_ops_track_the_dtype_their_contract_declares() {
        let kinds = [
            ImageOpKind::Canny {
                low_threshold: 50.0,
                high_threshold: 150.0,
            },
            ImageOpKind::HistogramEqualize,
            ImageOpKind::Erode {
                ksize: 3,
                iterations: 1,
                footprint: crate::ops::Footprint::Square,
            },
            ImageOpKind::Dilate {
                ksize: 3,
                iterations: 1,
                footprint: crate::ops::Footprint::Disk,
            },
            ImageOpKind::ChannelSwap {
                order: vec![2, 1, 0],
            },
            // Grayscale (`PreserveInput`) and Threshold (`Fixed(U8)`) reached
            // `apply_op` through hand-written builders that stamped their own
            // dtype; `grayscale()` hardcoded `U8`, diverging from the contract
            // on non-u8 input. They now route through the same canonical block.
            ImageOpKind::Grayscale { keepdims: true },
            ImageOpKind::Threshold { value: 128.0 },
            ImageOpKind::Blur { sigma: 1.0 },
            ImageOpKind::Resize {
                width: 4,
                height: 4,
                filter: FilterType::Triangle,
            },
        ];

        for dtype in [DType::U8, DType::U16, DType::F32] {
            let base = ViewBuffer::from_vec_with_shape(vec![0u8; 8 * 8 * 3], vec![8, 8, 3]);
            let source = ViewExpr::new_source(base.cast(dtype));
            for kind in &kinds {
                let op = ImageOp { kind: kind.clone() };
                let expected = op.resolve_output_dtype(dtype);
                let applied = source.clone().apply_op(ViewDto::Image(op));
                assert_eq!(
                    applied.dtype, expected,
                    "{kind:?} on {dtype:?}: ViewExpr tracks {:?} but the op's \
                     OutputDTypeRule resolves to {expected:?}",
                    applied.dtype
                );
            }
        }
    }

    /// The typed builders are sugar over `apply_op`, not a second construction
    /// path: whatever dtype `apply_op` tracks for an op, the builder for that
    /// op must track the same. This is the direct regression for the
    /// `grayscale()`/`threshold()` hardcoded-`DType::U8` bug — a builder that
    /// stamps its own dtype instead of routing through `apply_op` fails here.
    #[test]
    fn typed_builders_agree_with_apply_op() {
        for dtype in [DType::U8, DType::U16, DType::F32] {
            let base = ViewBuffer::from_vec_with_shape(vec![0u8; 8 * 8 * 3], vec![8, 8, 3]);
            let source = ViewExpr::new_source(base.cast(dtype));

            let cases: [(Arc<ViewExpr>, ImageOpKind); 4] = [
                (
                    source.grayscale(),
                    ImageOpKind::Grayscale { keepdims: true },
                ),
                (
                    source.threshold(128.0),
                    ImageOpKind::Threshold { value: 128.0 },
                ),
                (
                    source.resize(4, 4, FilterType::Triangle),
                    ImageOpKind::Resize {
                        width: 4,
                        height: 4,
                        filter: FilterType::Triangle,
                    },
                ),
                (source.blur(1.0), ImageOpKind::Blur { sigma: 1.0 }),
            ];

            for (built, kind) in cases {
                let op = ImageOp { kind: kind.clone() };
                let expected = op.resolve_output_dtype(dtype);
                assert_eq!(
                    built.dtype, expected,
                    "the builder for {kind:?} on {dtype:?} tracks {:?} but the \
                     op's OutputDTypeRule resolves to {expected:?}",
                    built.dtype
                );
            }
        }
    }

    /// Fusion must not change executed values. A grayscale on f32 input feeds a
    /// downstream fused compute block; if grayscale mistracks its dtype as `U8`,
    /// the fused block reads `U8` as its input dtype and `invert` computes
    /// `255 - x` instead of `1 - x`. Comparing the fused chain against the
    /// unfused one pins the invariant without hardcoding the luminance formula.
    #[test]
    fn grayscale_f32_fused_chain_matches_unfused() {
        let data: Vec<u8> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let base = ViewBuffer::from_vec_with_shape(data, vec![2, 2, 3]);
        // Normalize to [0, 1] f32, then reduce to a single grayscale channel.
        let src = ViewExpr::new_source(base)
            .cast(DType::F32)
            .scale(1.0 / 255.0);

        // Unfused: `invert`'s child is an Image node, so it does not fuse.
        let unfused = src.grayscale().invert().plan().execute();
        // Fused: the trailing `scale` makes `invert` the inner op of a fusion,
        // whose input dtype is read from the grayscale node's tracked dtype.
        let fused = src.grayscale().invert().scale(1.0).plan().execute();

        assert_eq!(unfused.dtype(), DType::F32);
        assert_eq!(fused.dtype(), DType::F32);

        let u = unfused.as_slice::<f32>();
        let f = fused.as_slice::<f32>();
        assert_eq!(u.len(), f.len());
        for (a, b) in u.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "fusion changed the result: fused {b} != unfused {a}"
            );
        }
        // The correct inverted value of normalized data is `1 - gray` in [0, 1].
        for &a in u {
            assert!(
                (0.0..=1.0).contains(&a),
                "invert on normalized f32 produced {a}, expected [0, 1]"
            );
        }
    }
}

#[cfg(test)]
mod scalar_op_tests {
    use super::*;
    use crate::ops::scalar::signum_numpy;
    use crate::ops::scalar::ScalarOp;

    /// numpy-style comparison tolerant of NaN/inf equality.
    fn approx(got: f32, exp: f32) -> bool {
        if exp.is_nan() {
            got.is_nan()
        } else if exp.is_infinite() {
            got.is_infinite() && got.signum() == exp.signum()
        } else {
            (got - exp).abs() < 1e-6
        }
    }

    fn src_f32(data: &[f32]) -> Arc<ViewExpr> {
        ViewExpr::new_source(ViewBuffer::from_vec_with_shape(
            data.to_vec(),
            vec![data.len()],
        ))
    }

    /// A scalar op paired with its reference f32 formula.
    type ScalarCase = (ScalarOp, fn(f32) -> f32);

    /// The full core-math set paired with its reference f32 formula.
    fn cases() -> Vec<ScalarCase> {
        vec![
            (ScalarOp::Add(1.5), |x| x + 1.5),
            (ScalarOp::Sub(1.5), |x| x - 1.5),
            (ScalarOp::Mul(2.0), |x| x * 2.0),
            (ScalarOp::Div(4.0), |x| x / 4.0),
            (ScalarOp::Pow(2.0), |x| x.powf(2.0)),
            (ScalarOp::Neg, |x| -x),
            (ScalarOp::Abs, |x| x.abs()),
            (ScalarOp::Sqrt, |x| x.sqrt()),
            (ScalarOp::Square, |x| x * x),
            (ScalarOp::Recip, |x| 1.0 / x),
            (ScalarOp::Min(0.5), |x| crate::ops::util::minimum(x, 0.5)),
            (ScalarOp::Max(0.5), |x| crate::ops::util::maximum(x, 0.5)),
            (ScalarOp::Sign, signum_numpy::<f32>),
            (ScalarOp::Floor, |x| x.floor()),
            (ScalarOp::Ceil, |x| x.ceil()),
            (ScalarOp::Round, |x| x.round_ties_even()),
            (ScalarOp::Trunc, |x| x.trunc()),
            (ScalarOp::Relu, |x| crate::ops::util::maximum(x, 0.0)),
            (ScalarOp::Clamp(0.0, 1.0), |x| x.clamp(0.0, 1.0)),
        ]
    }

    /// Each scalar op, executed as a lone `ComputeOp::Scalar`, matches its
    /// reference formula. Pins every `apply_fused_op_passes` arm.
    #[test]
    fn scalar_op_matches_reference_f32() {
        let data: Vec<f32> = vec![-2.5, -1.0, -0.4, 0.0, 0.4, 1.0, 2.5, 4.0];
        for (op, reff) in cases() {
            let out = src_f32(&data)
                .apply_op(ViewDto::Compute(ComputeOp::Scalar(op.clone())))
                .plan()
                .execute();
            assert_eq!(out.dtype(), DType::F32, "{op:?} must promote to f32");
            let got = out.as_slice::<f32>();
            for (i, &x) in data.iter().enumerate() {
                assert!(
                    approx(got[i], reff(x)),
                    "{op:?} on {x}: got {}, expected {}",
                    got[i],
                    reff(x)
                );
            }
        }
    }

    /// Adjacent scalar ops collapse into a single fused kernel (one Compute
    /// step) and the fused result matches sequential reference application.
    #[test]
    fn scalar_chain_fuses_into_one_kernel() {
        let data: Vec<f32> = vec![-4.0, -1.0, 0.0, 2.0, 9.0];
        let expr = src_f32(&data)
            .apply_op(ViewDto::Compute(ComputeOp::Scalar(ScalarOp::Mul(2.0))))
            .apply_op(ViewDto::Compute(ComputeOp::Scalar(ScalarOp::Abs)))
            .apply_op(ViewDto::Compute(ComputeOp::Scalar(ScalarOp::Sqrt)));

        let compute_steps = expr
            .plan()
            .steps
            .iter()
            .filter(|s| matches!(s, PlanStep::Op(ViewDto::Compute(_))))
            .count();
        assert_eq!(
            compute_steps, 1,
            "three adjacent scalar ops must fuse into one kernel"
        );

        let got = expr.plan().execute();
        let got = got.as_slice::<f32>();
        for (i, &x) in data.iter().enumerate() {
            assert!(approx(got[i], (2.0 * x).abs().sqrt()));
        }
    }

    /// f64 stays unfused (the kernel is f32) and computes in f64 via
    /// `apply_f64`, preserving the `PromoteToFloat` f64 contract.
    #[test]
    fn scalar_f64_stays_unfused_and_computes_in_f64() {
        let data: Vec<f64> = vec![-2.5, -0.4, 0.0, 0.4, 2.5, 9.0];
        // A two-op f64 chain must NOT fuse.
        let chain = ViewExpr::new_source(ViewBuffer::from_vec_with_shape(
            data.clone(),
            vec![data.len()],
        ))
        .apply_op(ViewDto::Compute(ComputeOp::Scalar(ScalarOp::Abs)))
        .apply_op(ViewDto::Compute(ComputeOp::Scalar(ScalarOp::Sqrt)));
        let compute_steps = chain
            .plan()
            .steps
            .iter()
            .filter(|s| matches!(s, PlanStep::Op(ViewDto::Compute(_))))
            .count();
        assert_eq!(compute_steps, 2, "f64 scalar ops must not fuse");

        for (op, _) in cases() {
            let src = ViewExpr::new_source(ViewBuffer::from_vec_with_shape(
                data.clone(),
                vec![data.len()],
            ));
            let out = src
                .apply_op(ViewDto::Compute(ComputeOp::Scalar(op.clone())))
                .plan()
                .execute();
            assert_eq!(out.dtype(), DType::F64, "{op:?} must preserve f64");
            let got = out.as_slice::<f64>();
            for (i, &x) in data.iter().enumerate() {
                let exp = op.apply_f64(x);
                let ok = if exp.is_nan() {
                    got[i].is_nan()
                } else if exp.is_infinite() {
                    got[i].is_infinite() && got[i].signum() == exp.signum()
                } else {
                    (got[i] - exp).abs() < 1e-12
                };
                assert!(ok, "{op:?} on {x} (f64): got {}, expected {}", got[i], exp);
            }
        }
    }

    /// Integer input promotes to f32 (matching `.scale()`), not the input dtype.
    #[test]
    fn scalar_promotes_integer_to_f32() {
        let src = ViewExpr::new_source(ViewBuffer::from_vec_with_shape(
            vec![1u8, 4, 9, 16],
            vec![4],
        ));
        let out = src
            .apply_op(ViewDto::Compute(ComputeOp::Scalar(ScalarOp::Sqrt)))
            .plan()
            .execute();
        assert_eq!(out.dtype(), DType::F32);
        let got = out.as_slice::<f32>();
        for (g, e) in got.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((g - e).abs() < 1e-6);
        }
    }
}
