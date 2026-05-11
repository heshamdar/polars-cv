//! `channel_select` op — extract a single channel from an HWC buffer.

use std::sync::Arc;

use view_buffer::ops::{Domain, NodeOutput, ViewDto, ViewOp};

use crate::contract::{AlphaMode, DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_view_dto;
use crate::params::{require_i64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ChannelSelectOp {
    index: usize,
}

impl Operation for ChannelSelectOp {
    fn name(&self) -> &'static str {
        "channel_select"
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
            "channel_select",
            inputs,
            ViewDto::View(ViewOp::ChannelSelect { index: self.index }),
        )
    }
}

const CONTRACT: OpContract = OpContract::new(
    DTypeEffect::Preserve,
    NdimEffect::ReduceOne,
    AlphaMode::Drop,
);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "channel_select",
    doc: "Extract a single channel from an HWC buffer. Output shape is \
          `[H, W]` (one fewer dim). Alpha is treated as a normal channel.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let index = require_i64(params, "channel_select", "index")?.max(0) as usize;
    Ok(Arc::new(ChannelSelectOp { index }))
}

inventory::submit! {
    OpRegistration {
        name: "channel_select",
        contract: &CONTRACT,
        schema: || &SCHEMA,
        factory,
    }
}
