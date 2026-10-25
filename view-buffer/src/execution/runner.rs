//! Plan execution: run a sequence of [`ViewDto`] ops on one source buffer.
//!
//! Dispatch from one step to its op is [`ViewDto::execute`]; the kernels it
//! reaches are in [`crate::kernels`].

use crate::core::buffer::ViewBuffer;
use crate::expr::ViewExpr;
use crate::ops::dto::ViewDto;

/// High-level entry point to execute a plan described by a sequence of ViewDto operations.
/// This acts as the bridge between the serialized plan (e.g. from Python) and the
/// execution engine.
pub fn execute_plan(source: ViewBuffer, ops: Vec<ViewDto>) -> ViewBuffer {
    let mut expr = ViewExpr::new_source(source);
    for op in ops {
        expr = expr.apply_op(op);
    }
    // plan() performs optimization (fusion, etc) before execution
    expr.plan().execute()
}
