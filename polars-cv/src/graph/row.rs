//! One row's node outputs, and how a step runs over them.
//!
//! Two halves of one rule: **a step reads another node only as one of its
//! [`operands`](GraphStep::operands)**, at one place.
//!
//! - [`RowOutputs`] holds the row's node outputs. Its storage is private to
//!   this module; the executor reads another node's output for a step only
//!   through [`RowOutputs::operands`], over positions computed once at compile
//!   time from the step's own `operands()`. A null operand nulls the reader
//!   there, for every step alike.
//! - [`run_step`] is every non-fusable step's body. It receives its input and
//!   the operands already gathered, so it cannot reach any other node; the
//!   buffers it reads are taken through the step's declared domains
//!   (`input_domains` for the input, `operand_domains` for the operands) and
//!   validated against their concrete shapes and dtypes in one call.
//!
//! The executor's arms used to repeat flush → buffer read → operand lookup by
//! id → validation → execute per step, each with its own error strings, and
//! `rasterize(shape=<node>)` had a resolver of its own (CR-117).

use std::borrow::Cow;
use std::sync::Arc;

use smallvec::SmallVec;
use view_buffer::geometry::label::score_contours_on_buffer;
use view_buffer::ops::{Domain, NodeOutput};
use view_buffer::ViewBuffer;

use super::encode::{execute_geometry_op, execute_region_op};
use super::step::GraphStep;
use crate::ops::graph::Role;
use crate::params::ParamCtx;

/// The operands one step reads, in `operands()` order.
pub(crate) type Operands<'o> = SmallVec<[&'o NodeOutput; 4]>;

/// One row's node outputs, by position in the compiled plan. `None` is a node
/// with no output this row (a null input, a null per-row parameter, a source
/// error under `on_error="null"`).
pub(crate) struct RowOutputs {
    outputs: Vec<Option<NodeOutput>>,
}

impl RowOutputs {
    pub(crate) fn new(nodes: usize) -> Self {
        RowOutputs {
            outputs: vec![None; nodes],
        }
    }

    /// Forget the previous row.
    pub(crate) fn clear(&mut self) {
        self.outputs.iter_mut().for_each(|o| *o = None);
    }

    /// The input of a node that reads node `upstream`: moved out when the
    /// reader is its only one (its buffer may then be written in place),
    /// shared otherwise.
    pub(crate) fn input(&mut self, upstream: usize, take: bool) -> Option<NodeOutput> {
        if take {
            self.outputs[upstream].take()
        } else {
            self.outputs[upstream].clone()
        }
    }

    /// The outputs a step reads, at the positions its operands compiled to;
    /// `None` when any of them has no output this row, which nulls the reader.
    pub(crate) fn operands(&self, positions: &[usize]) -> Option<Operands<'_>> {
        positions
            .iter()
            .map(|&i| self.outputs[i].as_ref())
            .collect()
    }

    /// Node `node`'s output this row.
    pub(crate) fn set(&mut self, node: usize, output: NodeOutput) {
        self.outputs[node] = Some(output);
    }

    /// A graph output's node, for encoding.
    pub(crate) fn output(&self, node: usize) -> Option<&NodeOutput> {
        self.outputs[node].as_ref()
    }
}

/// Run one non-fusable step over its input and its gathered operands.
///
/// `name` is the op's wire name, which prefixes every error the step raises.
/// Fusable buffer ops (`GraphStep::Buffer`) never reach here: the executor
/// accumulates them into one fused segment.
pub(crate) fn run_step(
    step: &GraphStep,
    name: &str,
    input: NodeOutput,
    operands: &[&NodeOutput],
    ctx: &ParamCtx<'_>,
    row_idx: usize,
) -> Result<NodeOutput, String> {
    let named = |e: String| format!("{name}: {e}");
    let buffers = || step_buffers(step, name, &input, operands);
    match step {
        GraphStep::Buffer(_) => Err(format!(
            "internal: {name} is a fusable op and runs in a fused segment"
        )),
        GraphStep::Geometry(op) => {
            // A canvas taken from another node: its height and width.
            let op = match operands {
                [] => Cow::Borrowed(op),
                [canvas] => {
                    let canvas = operand_buffer(step, name, canvas)?;
                    let &[height, width, ..] = canvas.shape() else {
                        return Err(named(format!(
                            "the canvas must be at least 2-D, got shape {:?}",
                            canvas.shape()
                        )));
                    };
                    Cow::Owned(op.clone().with_canvas(height as u32, width as u32))
                }
                _ => return Err(named("reads at most one canvas".into())),
            };
            execute_geometry_op(input, &op)
        }
        GraphStep::Reduction(op) => {
            let [buf] = exactly(name, buffers()?)?;
            let result = op.execute(&buf);
            // The op's declared output domain (the same authority the
            // planner reads) decides scalar vs buffer — not the result's
            // shape, which is also [1] for an axis reduction of a 1-D buffer.
            if op.output_domain() == Domain::Scalar {
                result.scalar_f64().map(NodeOutput::Scalar).ok_or_else(|| {
                    named(format!(
                        "a scalar reduction produced shape {:?}",
                        result.shape()
                    ))
                })
            } else {
                Ok(NodeOutput::from_buffer(result))
            }
        }
        // The fallible buffer steps: a data error (a value past a histogram's
        // range, past a lookup table) is the row's `Err`, which an engine op
        // cannot report.
        GraphStep::Histogram(op) => {
            let [buf] = exactly(name, buffers()?)?;
            op.execute(&buf).map(NodeOutput::from_buffer).map_err(named)
        }
        GraphStep::Lut(op) => {
            let [buf] = exactly(name, buffers()?)?;
            op.execute(&buf).map(NodeOutput::from_buffer).map_err(named)
        }
        GraphStep::PerceptualHash(op) => {
            let [buf] = exactly(name, buffers()?)?;
            // Converts to u8 image format and returns a 1-D u8 buffer: the
            // planned `vector` domain selects the List encoding at sink time.
            Ok(NodeOutput::from_buffer(
                view_buffer::execution::runner::apply_perceptual_hash((*buf).clone(), op.clone()),
            ))
        }
        GraphStep::Graph(graph) => match graph.role() {
            Role::Binary(op, _) => {
                let [a, b] = exactly(name, buffers()?)?;
                Ok(NodeOutput::from_buffer(op.execute(&a, &b)))
            }
            Role::ApplyMask { invert, .. } => {
                let [image, mask] = exactly(name, buffers()?)?;
                Ok(NodeOutput::from_buffer(view_buffer::apply_mask(
                    &image, &mask, *invert,
                )))
            }
            Role::ChannelMerge { .. } => {
                let all = buffers()?;
                let refs: Vec<&ViewBuffer> = all.iter().map(Arc::as_ref).collect();
                Ok(NodeOutput::from_buffer(view_buffer::apply_channel_merge(
                    &refs,
                )))
            }
            Role::Regions(op, _) => {
                let [other] = operands else {
                    return Err(format!("internal: {name} reads one other set"));
                };
                execute_region_op(op, &input, other)
            }
            Role::AssertShape { dims, exact } => {
                let shape: Vec<usize> = match &input {
                    NodeOutput::Buffer(buf) => buf.shape().to_vec(),
                    NodeOutput::Vector(vals) => vec![vals.len()],
                    other => {
                        return Err(format!(
                            "assert_shape() declares a shape, but the data here is {}",
                            other.domain().name()
                        ))
                    }
                };
                check_declared_shape(&shape, dims, *exact)?;
                Ok(input)
            }
            Role::ExtractShape => {
                let [buf] = exactly(name, buffers()?)?;
                Ok(NodeOutput::from_vector(
                    buf.shape().iter().map(|&d| d as f64).collect(),
                ))
            }
            Role::LabelReduce {
                contours,
                reduction,
                region_mode,
            } => {
                let [buf] = exactly(name, buffers()?)?;
                let column = ctx.col(contours.0).map_err(|e| e.to_string())?;
                let (series, row) = column.at(row_idx);
                let contours = crate::geom_columns::ContourColumn::new(series)
                    .row(row)
                    .map_err(|e| named(format!("contour parsing failed: {e}")))?;
                let Some(contours) = contours else {
                    return Ok(NodeOutput::from_vector(Vec::new()));
                };
                score_contours_on_buffer(&buf, &contours, *reduction, *region_mode)
                    .map(NodeOutput::from_vector)
            }
        },
    }
}

/// The buffers a buffer-reading step consumes — its input, then each operand —
/// read through its declared domains and validated, together, against their
/// concrete shapes and dtypes: the one place a step meets its own validation
/// at run time.
fn step_buffers(
    step: &GraphStep,
    name: &str,
    input: &NodeOutput,
    operands: &[&NodeOutput],
) -> Result<SmallVec<[Arc<ViewBuffer>; 4]>, String> {
    let mut buffers: SmallVec<[Arc<ViewBuffer>; 4]> = SmallVec::new();
    buffers.push(read_buffer(input, step, &step.input_domains(), name)?);
    for operand in operands {
        buffers.push(operand_buffer(step, name, operand)?);
    }
    {
        let shapes: SmallVec<[&[usize]; 4]> = buffers.iter().map(|b| b.shape()).collect();
        let dtypes: SmallVec<[view_buffer::DType; 4]> = buffers.iter().map(|b| b.dtype()).collect();
        step.validate_concrete(&shapes, &dtypes)
            .map_err(|e| format!("{name}: {e}"))?;
    }
    Ok(buffers)
}

/// The `N` buffers a step reads, or an internal error: the executor gathers
/// one operand per `operands()` entry, so a count that differs is a bug here.
fn exactly<const N: usize>(
    name: &str,
    buffers: SmallVec<[Arc<ViewBuffer>; 4]>,
) -> Result<[Arc<ViewBuffer>; N], String> {
    let got = buffers.len();
    buffers
        .into_iter()
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| format!("internal: {name} reads {N} buffer(s), got {got}"))
}

/// An operand's buffer, through the step's declared operand domains.
fn operand_buffer(
    step: &GraphStep,
    name: &str,
    operand: &NodeOutput,
) -> Result<Arc<ViewBuffer>, String> {
    read_buffer(
        operand,
        step,
        &step.operand_domains(),
        &format!("{name}'s operand"),
    )
}

/// Whether a runtime `NodeOutput::Vector` may be read as a 1-D buffer here.
///
/// The planner's `vector` *domain* and the runtime `Vector` *representation*
/// are not the same set (CR-116). `perceptual_hash` and the histogram vector
/// modes plan as `vector` but execute as `Buffer`, so the only steps that
/// ever receive a runtime `Vector` are those fed by `ExtractShape`,
/// `LabelReduce` or a contour measure.
///
/// Coercing one is sound only where the step's published output shape
/// derives from *this* operand alone.
///
/// - `Reduction` qualifies: rank and domain fold from its single input, so
///   `extract_shape().reduce_sum()` plans and executes the same scalar.
/// - `Binary` does **not**. Its output shape is both operands' one shape,
///   so it does not derive from this operand alone. (Coercing once let a
///   vector broadcast against a rank-3 image, executing at rank 3 after
///   the plan promised rank 1. Binary ops now broadcast nothing and the
///   planner refuses that pair where it is written; this keeps the
///   runtime representation from being read as what the plan did not
///   check.)
///
/// Exhaustive, so a new variant has to answer this question rather than
/// inherit an answer.
fn vector_reads_as_buffer(step: &GraphStep) -> bool {
    match step {
        GraphStep::Reduction(_) => true,
        GraphStep::Buffer(_)
        | GraphStep::Geometry(_)
        | GraphStep::Histogram(_)
        | GraphStep::Lut(_)
        | GraphStep::PerceptualHash(_) => false,
        GraphStep::Graph(graph) => match graph.role() {
            Role::Binary(..)
            | Role::Regions(..)
            | Role::ApplyMask { .. }
            | Role::ChannelMerge { .. }
            | Role::ExtractShape
            | Role::AssertShape { .. }
            | Role::LabelReduce { .. } => false,
        },
    }
}

/// The buffer `output` holds, honouring the step's *declared* domains.
///
/// **Read the contract; never restate it.** `accepted` is the step's own
/// declaration (`input_domains` or `operand_domains`), the single authority
/// the planner validates against through `Plan::push`. It decides
/// *rejection*; `vector_reads_as_buffer` decides the narrower question of
/// whether a runtime vector can stand in for a buffer, which the domain
/// vocabulary alone cannot answer.
fn read_buffer(
    output: &NodeOutput,
    step: &GraphStep,
    accepted: &[Domain],
    what: &str,
) -> Result<Arc<ViewBuffer>, String> {
    match output {
        NodeOutput::Buffer(buf) => Ok(Arc::clone(buf)),
        NodeOutput::Vector(vals) if vector_reads_as_buffer(step) => {
            Ok(Arc::new(ViewBuffer::from_vec(vals.as_ref().clone())))
        }
        NodeOutput::Vector(_) if accepted.contains(&Domain::Vector) => Err(format!(
            "{what} declares it accepts a vector, but cannot read one whose \
             rank the plan did not fix: its output shape depends on another \
             operand. Reduce or rasterize the vector first."
        )),
        _ => Err(format!(
            "{what} accepts {} input but received {:?}",
            accepted
                .iter()
                .map(|d| format!("{d:?}"))
                .collect::<Vec<_>>()
                .join(" or "),
            output.domain()
        )),
    }
}

/// Check one row's shape against an `assert_shape` declaration: the
/// declaration applied to the row's shape ([`crate::plan::apply_declaration`],
/// the reading the planner applies too).
///
/// The declaration is the user's statement about their data, so a mismatch is
/// reported as theirs: it names what they wrote and what arrived.
fn check_declared_shape(shape: &[usize], dims: &[Option<u32>], exact: bool) -> Result<(), String> {
    let declared: Vec<Option<usize>> = dims.iter().map(|d| d.map(|d| d as usize)).collect();
    let row = crate::plan::PlannedShape::Ranked(shape.iter().map(|&n| Some(n)).collect());
    if crate::plan::apply_declaration(&row, &declared, exact).is_ok() {
        return Ok(());
    }
    let size = |d: &Option<usize>| d.map_or("None".to_string(), |d| d.to_string());
    let written = if exact {
        format!(
            "dims=[{}]",
            declared.iter().map(size).collect::<Vec<_>>().join(", ")
        )
    } else {
        declared
            .iter()
            .enumerate()
            .filter_map(|(axis, d)| {
                d.map(|d| format!("{}={d}", crate::plan::declared_name(axis, false)))
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(format!(
        "assert_shape({written}) does not hold: the data is {shape:?}. An assertion states \
         what the data is; it does not change it. Correct the assertion, or drop it \
         and let the planner infer the shape."
    ))
}
