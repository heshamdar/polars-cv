//! `grayscale` op — converts RGB(A) → single-channel via BT.601.
//!
//! Wraps view-buffer's `ImageOp { kind: ImageOpKind::Grayscale }`, executed
//! through `ViewExpr::apply_op(...).plan().execute()`. This matches v1's
//! flow at `polars-cv/src/graph/types.rs:676-681` exactly — pcv-graph ops
//! are deliberately thin adapters so behavior is preserved bit-for-bit.

use std::sync::Arc;

use view_buffer::expr::ViewExpr;
use view_buffer::ops::{Domain, ImageOp, ImageOpKind, NodeOutput, ViewDto};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct GrayscaleOp;

impl Operation for GrayscaleOp {
    fn name(&self) -> &'static str {
        "grayscale"
    }

    fn input_arity(&self) -> InputArity {
        InputArity::Unary
    }

    fn input_domain(&self, _port: &str) -> Domain {
        Domain::Buffer
    }

    fn output_domain(&self) -> Domain {
        Domain::Buffer
    }

    fn contract(&self) -> &'static OpContract {
        &CONTRACT
    }

    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        let input = inputs.require_single()?;
        let buf_arc = input.as_buffer().ok_or_else(|| OpError::DomainMismatch {
            op: "grayscale",
            expected: Domain::Buffer,
            got: input.domain(),
        })?;
        // Clone the ViewBuffer (cheap — internal storage is Arc-backed).
        let buf: view_buffer::ViewBuffer = (**buf_arc).clone();
        let expr = ViewExpr::new_source(buf);
        let expr = expr.apply_op(ViewDto::Image(ImageOp {
            kind: ImageOpKind::Grayscale,
        }));
        let result = expr.plan().execute();
        Ok(NodeOutput::from_buffer(result))
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Drop,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "grayscale",
    doc: "Convert RGB(A) image to a single-channel intensity image using the \
          BT.601 luma weights. Alpha is discarded. Output shape is `[H, W, 1]` \
          for 3-D input or `[H, W]` for 2-D (already-single-channel) input.",
    params: &[],
};

fn factory(_params: &ParamMap) -> Result<OpHandle, OpError> {
    Ok(Arc::new(GrayscaleOp))
}

inventory::submit! {
    OpRegistration {
        name: "grayscale",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
