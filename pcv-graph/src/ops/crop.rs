//! `crop` op — take a sub-rectangle of an HWC buffer.
//!
//! The v1 parameter shape is `(top, left, height?, width?)` rather than
//! explicit `start`/`end` vectors; this op preserves that public shape and
//! translates internally to `ViewOp::Crop { start, end }`. Matches v1's
//! behavior at `polars-cv/src/execute.rs:359-401` exactly: negative top/left
//! are clamped to 0, missing height/width means "to end".

use std::sync::Arc;

use view_buffer::ops::{Domain, NodeOutput, ViewDto, ViewOp};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct CropOp {
    start: Vec<usize>,
    end: Vec<usize>,
}

impl Operation for CropOp {
    fn name(&self) -> &'static str {
        "crop"
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
        apply_view_dto(
            "crop",
            inputs,
            ViewDto::View(ViewOp::Crop {
                start: self.start.clone(),
                end: self.end.clone(),
            }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::Preserve,
    AlphaMode::Passthrough,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "crop",
    doc: "Crop an HWC buffer to `[top..top+height, left..left+width, :]`. \
          Negative `top`/`left` are clamped to 0. Missing `height` or `width` \
          extends the crop to the buffer's end on that axis. ViewBuffer's \
          slice will clamp out-of-range bounds.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let top = params
        .get("top")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .max(0) as usize;
    let left = params
        .get("left")
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
        .max(0) as usize;
    let height = params
        .get("height")
        .and_then(|v| v.as_i64())
        .map(|h| h.max(0) as usize);
    let width = params
        .get("width")
        .and_then(|v| v.as_i64())
        .map(|w| w.max(0) as usize);

    let start = vec![top, left, 0];
    let end = match (height, width) {
        (Some(h), Some(w)) => vec![top.saturating_add(h), left.saturating_add(w), usize::MAX],
        _ => vec![usize::MAX, usize::MAX, usize::MAX],
    };
    Ok(Arc::new(CropOp { start, end }))
}

inventory::submit! {
    OpRegistration {
        name: "crop",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
