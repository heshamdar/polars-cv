//! `reduce_std` op — standard deviation, global or along an axis. Optional `ddof`.

use std::sync::Arc;

use view_buffer::ops::reduction::ReductionOp;
use view_buffer::ops::{Domain, NodeOutput};

use crate::contract::{DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_reduction;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ReduceStdOp { axis: Option<usize>, ddof: u8 }

impl Operation for ReduceStdOp {
    fn name(&self) -> &'static str { "reduce_std" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Scalar }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_reduction(
            "reduce_std",
            inputs,
            ReductionOp::Std { axis: self.axis, ddof: self.ddof },
        )
    }
}

const CONTRACT: OpContract = OpContract::non_image(DTypeEffect::FixedF64, NdimEffect::ToZero);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "reduce_std",
    doc: "Standard deviation. `ddof` defaults to 0 (population); pass 1 for sample.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let axis = params.get("axis").and_then(|v| v.as_i64()).map(|i| i.max(0) as usize);
    let ddof = params.get("ddof").and_then(|v| v.as_u64()).unwrap_or(0).min(255) as u8;
    Ok(Arc::new(ReduceStdOp { axis, ddof }))
}

inventory::submit! {
    OpRegistration { name: "reduce_std", contract: &CONTRACT, schema: || &SCHEMA, factory }
}
