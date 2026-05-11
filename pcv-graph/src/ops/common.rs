//! Shared helpers used by op adapters.
//!
//! Most ops do the same dance: pull the input buffer out of a [`NodeOutput`],
//! build a [`ViewDto`], apply it via [`ViewExpr`], and wrap the result back as
//! a [`NodeOutput::Buffer`]. Centralizing it here keeps each op file focused
//! on the parameter parsing that actually differs between ops.

use view_buffer::expr::ViewExpr;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::op::{OpError, OpInputs};

/// Apply a single [`ViewDto`] to the single buffer input of an op.
///
/// Returns an [`OpError::DomainMismatch`] if the input isn't a `Buffer`,
/// or [`OpError::ArityMismatch`] if the op was somehow given a named-multi
/// input. Otherwise: clones the input buffer (cheap — `ViewBuffer` storage
/// is internally `Arc`-backed), wraps it in a `ViewExpr`, applies `dto`,
/// materializes via `plan().execute()`, and wraps the result.
pub fn apply_view_dto(
    op_name: &'static str,
    inputs: &OpInputs<'_>,
    dto: ViewDto,
) -> Result<NodeOutput, OpError> {
    let input = inputs.require_single()?;
    let buf_arc = input.as_buffer().ok_or_else(|| OpError::DomainMismatch {
        op: op_name,
        expected: Domain::Buffer,
        got: input.domain(),
    })?;
    let buf: view_buffer::ViewBuffer = (**buf_arc).clone();
    let expr = ViewExpr::new_source(buf);
    let result = expr.apply_op(dto).plan().execute();
    Ok(NodeOutput::from_buffer(result))
}
