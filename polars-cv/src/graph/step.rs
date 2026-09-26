//! `GraphStep` — the executor's operation vocabulary, and the typed op.
//!
//! A step is a fusable single-buffer engine op (`Buffer(ViewDto)`, executed
//! through `ViewExpr`), an engine op that changes the data domain (geometry,
//! reduction, histogram, perceptual hash), or a **graph-level op**
//! ([`GraphOp`]): one that reads other nodes' buffers or an input column.
//!
//! Generic over the [`Mode`] like every family it holds: `GraphStep<Wire>` is
//! the typed op ([`TypedOp`](crate::ops::TypedOp)) — what a plan holds and
//! reads every rule below from, with a per-row value unknown rather than
//! stood in for — and resolving it for a row gives the `GraphStep<Exec>` that
//! runs. The step's math lives in view-buffer (`BinaryOp::execute`,
//! `apply_mask`, `ReductionOp::execute`, …); the executor's arms are wiring.

use polars_cv_macros::Resolve;
use view_buffer::core::dtype::OutputDTypeRule;
use view_buffer::mode::{Exec, Mode, NodeRef};
use view_buffer::ops::phash::PerceptualHashOp;
use view_buffer::ops::{Domain, OpShape, SpatialDependency};
use view_buffer::ops::{HistogramOp, ReductionOp};
use view_buffer::{GeometryOp, IdentityRule, Op, ViewDto};

use crate::ops::graph::GraphOp;

/// One operation of a graph node (see the module docs).
#[derive(Debug, Clone, PartialEq, Resolve)]
pub enum GraphStep<M: Mode = Exec> {
    /// A fusable single-buffer engine op, executed via `ViewExpr::apply_op`.
    Buffer(ViewDto<M>),
    /// Geometry op (extract_contours, rasterize, measures, transforms) —
    /// changes or consumes the contour domain.
    Geometry(GeometryOp<M>),
    /// Reduction: global → scalar, axis → smaller buffer.
    Reduction(ReductionOp<M>),
    /// Histogram: quantized → buffer, other modes → vector.
    Histogram(HistogramOp<M>),
    /// Perceptual hash: image buffer → 1-D u8 fingerprint (vector domain).
    PerceptualHash(PerceptualHashOp<M>),
    /// A graph-level op: reads other nodes or an input column.
    Graph(GraphOp<M>),
}

impl<M: Mode> GraphStep<M> {
    /// Whose contract this step's rules are: the engine op's, or the graph
    /// op's own.
    fn rules(&self) -> Rules<'_, M> {
        match self {
            GraphStep::Buffer(dto) => Rules::Engine(dto.as_op()),
            GraphStep::Geometry(op) => Rules::Engine(op),
            GraphStep::Reduction(op) => Rules::Engine(op),
            GraphStep::Histogram(op) => Rules::Engine(op),
            GraphStep::PerceptualHash(op) => Rules::Engine(op),
            GraphStep::Graph(op) => Rules::Graph(op),
        }
    }

    /// Refuse a parameter combination no row can run, from the values this
    /// step knows: all of them once resolved, the literals on the wire.
    pub fn check(&self) -> Result<(), String> {
        match self {
            GraphStep::Buffer(dto) => dto.check(),
            GraphStep::Geometry(op) => op.check(),
            GraphStep::Reduction(op) => op.check(),
            GraphStep::Histogram(op) => op.check(),
            GraphStep::PerceptualHash(op) => op.check(),
            GraphStep::Graph(op) => op.check(),
        }
    }

    /// Every domain this step can consume (see [`GraphOp::input_domains`]
    /// for the ones that take more than one).
    pub fn input_domains(&self) -> Vec<Domain> {
        match self {
            // Reductions consume any numeric container: a perceptual hash is
            // a 1-D u8 buffer encoded as a vector, and `hash_a ^ hash_b ->
            // reduce_popcount` is the library's own hamming distance.
            GraphStep::Reduction(_) => vec![Domain::Buffer, Domain::Vector],
            GraphStep::Graph(op) => op.input_domains(),
            GraphStep::Buffer(_) | GraphStep::Histogram(_) | GraphStep::PerceptualHash(_) => {
                vec![Domain::Buffer]
            }
            GraphStep::Geometry(op) => vec![op.input_domain()],
        }
    }

    /// The domain this step produces from an `input` in its accepted domains.
    pub fn output_domain(&self, input: Domain) -> Domain {
        match self {
            GraphStep::Buffer(dto) => dto.output_domain(),
            GraphStep::Geometry(op) => op.output_domain(),
            GraphStep::Reduction(op) => op.output_domain(),
            GraphStep::Histogram(op) => op.output_domain(),
            // Perceptual hash produces a fixed-length 1-D fingerprint.
            GraphStep::PerceptualHash(_) => Domain::Vector,
            GraphStep::Graph(op) => op.output_domain(input),
        }
    }

    /// The rule that determines this step's output element dtype.
    pub fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self.rules() {
            Rules::Engine(op) => op.output_dtype_rule(),
            Rules::Graph(op) => op.output_dtype_rule(),
        }
    }

    /// How this step's output depends on the spatial extent of its input — the
    /// plan-time authority for whether a spatial window may commute with it.
    pub fn spatial_dependency(&self) -> SpatialDependency {
        match self.rules() {
            Rules::Engine(op) => op.spatial_dependency(),
            Rules::Graph(op) => op.spatial_dependency(),
        }
    }

    /// Under what condition this step is a removable no-op — the plan-time
    /// authority an identity-elimination pass reads.
    pub fn identity_rule(&self) -> IdentityRule {
        match self.rules() {
            Rules::Engine(op) => op.identity_rule(),
            Rules::Graph(op) => op.identity_rule(),
        }
    }

    /// Whether this step reads another graph node (its [`operands`](Self::operands)),
    /// so a spatial window hoisted past it would crop only this operand.
    pub fn reads_other_nodes(&self) -> bool {
        !self.operands().is_empty()
    }

    /// Whether the step can run on inputs of these shapes and dtypes (its
    /// input, then any operand it reads by id), over what is known of them:
    /// every error is a verdict on a known fact, so the planner raises each
    /// one and execution calls it with everything known.
    pub fn validate(
        &self,
        inputs: &[&[view_buffer::ops::Dim]],
        dtypes: &[view_buffer::PlannedDType],
    ) -> Result<(), view_buffer::ops::validation::ValidationError> {
        match self.rules() {
            Rules::Engine(op) => op.validate(inputs, dtypes),
            Rules::Graph(op) => op.validate(inputs, dtypes),
        }
    }

    /// [`validate`](Self::validate) over inputs whose every size and dtype is
    /// known — execution's call, before the step runs on a row.
    pub fn validate_concrete(
        &self,
        shapes: &[&[usize]],
        dtypes: &[view_buffer::DType],
    ) -> Result<(), view_buffer::ops::validation::ValidationError> {
        let shapes: Vec<Vec<view_buffer::ops::Dim>> = shapes
            .iter()
            .map(|s| view_buffer::ops::shape_rule::known_dims(s))
            .collect();
        let shapes: Vec<&[view_buffer::ops::Dim]> = shapes.iter().map(Vec::as_slice).collect();
        let dtypes: Vec<view_buffer::PlannedDType> = dtypes
            .iter()
            .map(|&d| view_buffer::PlannedDType::Known(d))
            .collect();
        self.validate(&shapes, &dtypes)
    }

    /// The other graph nodes this step reads by id, in input order after its
    /// own input: a binary op's other operand, a mask, the channels a merge
    /// stacks, a canvas taken from another node. The planner passes each
    /// one's planned shape to [`shape`](Self::shape) and
    /// [`validate`](Self::validate) as further inputs.
    pub fn operands(&self) -> Vec<&NodeRef> {
        match self {
            GraphStep::Graph(op) => op.operands(),
            GraphStep::Geometry(view_buffer::GeometryOp::Rasterize {
                size: view_buffer::geometry::ops::RasterSize::FromNode(node),
                ..
            }) => vec![node],
            GraphStep::Buffer(_)
            | GraphStep::Geometry(_)
            | GraphStep::Reduction(_)
            | GraphStep::Histogram(_)
            | GraphStep::PerceptualHash(_) => Vec::new(),
        }
    }

    /// The domains a node this step reads (an [`operand`](Self::operands))
    /// may be in. The planner refuses any other at build; execution reads the
    /// same operands through [`input_domains`](Self::input_domains), which a
    /// graph op's operands share with its own input.
    pub fn operand_domains(&self) -> Vec<Domain> {
        match self {
            GraphStep::Graph(op) => op.input_domains(),
            // A canvas is read for its height and width.
            GraphStep::Geometry(_) => vec![Domain::Buffer],
            GraphStep::Buffer(_)
            | GraphStep::Reduction(_)
            | GraphStep::Histogram(_)
            | GraphStep::PerceptualHash(_) => Vec::new(),
        }
    }

    /// How the step's output shape follows from its inputs — every step has
    /// one, and the planner reads the output rank (its length), channel count
    /// (its axis 2) and sizes from it. A per-row value is `Sym::PerRow`.
    pub fn shape(&self) -> OpShape {
        match self.rules() {
            Rules::Engine(op) => op.shape(),
            Rules::Graph(op) => op.shape(),
        }
    }

    /// Whether this step is a hoistable H/W spatial window (a crop/ROI) — the
    /// plan-time authority the spatial-window pushdown reads.
    pub fn is_spatial_window(&self) -> bool {
        match self.rules() {
            Rules::Engine(op) => op.is_spatial_window(),
            // A graph-level op is never an H/W crop over a single buffer.
            Rules::Graph(_) => false,
        }
    }
}

/// A step's contract: an engine op's [`Op`] impl, or a graph op's own rules.
enum Rules<'a, M: Mode> {
    Engine(&'a dyn Op),
    Graph(&'a GraphOp<M>),
}
