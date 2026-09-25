//! Histogram: a buffer → vector op.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use view_buffer::ops::histogram::{HistogramClosed, HistogramOp, HistogramOutput};

use super::{FieldType, Literal, OpDef, Param, TypeDesc};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::OpShape;

/// Compute pixel value histogram.
///
/// Example:
///     >>> Pipeline().source("image_bytes").grayscale().histogram(bins=8)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Histogram {
    /// Number of bins (default 256), a Polars expression for per-row dynamic
    /// bin count, or an explicit list of bin edges.
    #[param(default = 256)]
    pub bins: Bins,
    /// (min, max) tuple. Auto-detected if None.
    pub range: Option<[Param<f64>; 2]>,
    /// "left" or "right" interval inclusiveness (default "left").
    #[param(default = "left")]
    pub closed: Literal<HistogramClosed>,
    /// "buckets" (list of structs), "counts" (bin counts), "normalized" (sum to
    /// 1.0), "quantized" (pixel indices), "edges" (bin edges).
    #[param(default = "buckets")]
    pub output: Literal<HistogramOutput>,
}

/// How the bins are given: a count, or the edges themselves.
///
/// Two variants rather than a count plus optional edges, so a spec cannot carry
/// both and have one ignored. On the wire a list is `Edges`, anything else
/// (a number or a slot) is `Count`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Bins {
    /// This many equal-width bins over the range; may be per-row (the output
    /// is a list, so its length may vary by row).
    Count(Param<u32>),
    /// Explicit, literal bin edges.
    Edges(Vec<Literal<f64>>),
}

impl<'de> Deserialize<'de> for Bins {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        if value.is_array() {
            serde_json::from_value(value).map(Bins::Edges)
        } else {
            serde_json::from_value(value).map(Bins::Count)
        }
        .map_err(D::Error::custom)
    }
}

impl FieldType for Bins {
    fn describe() -> TypeDesc {
        TypeDesc::OneOf {
            options: vec![
                <Param<u32> as FieldType>::describe(),
                <Vec<Literal<f64>> as FieldType>::describe(),
            ],
        }
    }
    fn visit_slots(&self, f: &mut dyn FnMut(usize)) {
        match self {
            Bins::Count(p) => p.visit_slots(f),
            Bins::Edges(e) => e.visit_slots(f),
        }
    }
}

impl OpDef for Histogram {
    fn shape(&self) -> Option<OpShape> {
        None
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Histogram {
            bins,
            range,
            closed,
            output,
        } = self;
        let mut op = match bins {
            Bins::Count(count) => HistogramOp::new(count.resolve(row, ctx)? as usize),
            Bins::Edges(edges) => {
                let edges: Vec<f64> = edges.iter().map(Literal::get).collect();
                HistogramOp::new(edges.len().saturating_sub(1)).with_edges(edges)
            }
        }
        .with_output(output.get())
        .with_closed(closed.get());
        if let Some([min, max]) = range {
            op = op.with_range(min.resolve(row, ctx)?, max.resolve(row, ctx)?);
        }
        Ok(GraphStep::Histogram(op))
    }
}
