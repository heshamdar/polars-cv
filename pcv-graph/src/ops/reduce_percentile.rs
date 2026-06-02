//! `reduce_percentile` op — global q-th percentile.

use std::sync::Arc;

use view_buffer::ops::reduction::ReductionOp;
use view_buffer::ops::{Domain, NodeOutput};

use crate::contract::{DTypeEffect, NdimEffect, OpContract};
use crate::op::{ExecCtx, InputArity, OpError, OpHandle, OpInputs, Operation};
use crate::ops::common::apply_reduction;
use crate::params::{require_f64, ParamMap};
use crate::registry::{OpRegistration, OpSchemaDescriptor};

pub struct ReducePercentileOp { q: f64 }

impl Operation for ReducePercentileOp {
    fn name(&self) -> &'static str { "reduce_percentile" }
    fn input_arity(&self) -> InputArity { InputArity::Unary }
    fn input_domain(&self, _p: &str) -> Domain { Domain::Buffer }
    fn output_domain(&self) -> Domain { Domain::Scalar }
    fn contract(&self) -> &'static OpContract { &CONTRACT }
    fn execute(&self, _ctx: &ExecCtx, inputs: &OpInputs) -> Result<NodeOutput, OpError> {
        apply_reduction("reduce_percentile", inputs, ReductionOp::Percentile { q: self.q })
    }
}

const CONTRACT: OpContract = OpContract::non_image(DTypeEffect::FixedF64, NdimEffect::ToZero);

const SCHEMA: OpSchemaDescriptor = OpSchemaDescriptor {
    name: "reduce_percentile",
    doc: "Global q-th percentile, `q` in [0, 100]. Uses linear interpolation. Always F64.",
    params: &[],
};

fn factory(params: &ParamMap) -> Result<OpHandle, OpError> {
    let q = require_f64(params, "reduce_percentile", "q")?;
    Ok(Arc::new(ReducePercentileOp { q }))
}

inventory::submit! {
    OpRegistration { name: "reduce_percentile", contract: &CONTRACT, schema: || &SCHEMA, factory }
}
