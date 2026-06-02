//! Shared helpers used by op adapters.
//!
//! Most ops do the same dance: pull the input buffer out of a [`NodeOutput`],
//! build a [`ViewDto`], apply it via [`ViewExpr`], and wrap the result back as
//! a [`NodeOutput::Buffer`]. Centralizing it here keeps each op file focused
//! on the parameter parsing that actually differs between ops.

use view_buffer::core::dtype::DType;
use view_buffer::expr::ViewExpr;
use view_buffer::ops::reduction::ReductionOp;
use view_buffer::ops::{Domain, NodeOutput, ViewDto};

use crate::op::{OpError, OpInputs};

/// Apply a [`ReductionOp`] directly (reductions don't go through `ViewExpr`).
///
/// Mirrors the v1 dispatch at `polars-cv/src/graph/types.rs:746-775`: if the
/// reduction collapses to shape `[1]`, extract a `NodeOutput::Scalar` so
/// downstream sinks (e.g. the `numpy` sink for scalars) see the scalar
/// directly; otherwise return the reduced buffer.
pub fn apply_reduction(
    op_name: &'static str,
    inputs: &OpInputs<'_>,
    reduction: ReductionOp,
) -> Result<NodeOutput, OpError> {
    let input = inputs.require_single()?;
    let buf_arc = input.as_buffer().ok_or_else(|| OpError::DomainMismatch {
        op: op_name,
        expected: Domain::Buffer,
        got: input.domain(),
    })?;
    let result = reduction.execute(buf_arc.as_ref());

    if result.shape() == [1] {
        let scalar_val = match result.dtype() {
            DType::U8 => result.as_slice::<u8>()[0] as f64,
            DType::I8 => result.as_slice::<i8>()[0] as f64,
            DType::U16 => result.as_slice::<u16>()[0] as f64,
            DType::I16 => result.as_slice::<i16>()[0] as f64,
            DType::U32 => result.as_slice::<u32>()[0] as f64,
            DType::I32 => result.as_slice::<i32>()[0] as f64,
            DType::U64 => result.as_slice::<u64>()[0] as f64,
            DType::I64 => result.as_slice::<i64>()[0] as f64,
            DType::F32 => result.as_slice::<f32>()[0] as f64,
            DType::F64 => result.as_slice::<f64>()[0],
        };
        Ok(NodeOutput::Scalar(scalar_val))
    } else {
        Ok(NodeOutput::from_buffer(result))
    }
}

/// Parse the canonical dtype string. Matches `polars-cv/src/execute.rs:1348`.
pub fn parse_dtype(op_name: &'static str, s: &str) -> Result<DType, OpError> {
    Ok(match s {
        "u8" => DType::U8,
        "i8" => DType::I8,
        "u16" => DType::U16,
        "i16" => DType::I16,
        "u32" => DType::U32,
        "i32" => DType::I32,
        "u64" => DType::U64,
        "i64" => DType::I64,
        "f32" => DType::F32,
        "f64" => DType::F64,
        other => {
            return Err(OpError::Failed {
                op: op_name,
                message: format!("unknown dtype `{other}`"),
            })
        }
    })
}

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
