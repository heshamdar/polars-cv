//! `reduce_popcount` op — count set bits across the entire buffer.

use std::sync::Arc;

use view_buffer::ops::reduction::ReductionOp;
use view_buffer::ops::{Domain, NodeOutput};

use crate::contract::{DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_reduction;
use crate::params::ParamMap;
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ReducePopcountOp;

impl Operation for ReducePopcountOp {
    fn name(&self) -> &'static str { "reduce_popcount" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Scalar }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_reduction("reduce_popcount", inputs, ReductionOp::PopCount)
    }
}

const CONTRACT: OpContract = OpContract::non_image(DTypeEffect::FixedF64, NdimEffect::ToZero);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "reduce_popcount",
    doc: "Population count — total number of set bits in the buffer. Floats are cast to i64 first. \
          Useful for Hamming-distance and binary-mask pixel counts.",
    params: &[],
};

fn factory(_params: &ParamMap) -> Result<OpHandle, OpError> {
    Ok(Arc::new(ReducePopcountOp))
}

inventory::submit! {
    OpRegistration { name: "reduce_popcount", contract: &CONTRACT, schema: || &SCHEMA, factory }
}
