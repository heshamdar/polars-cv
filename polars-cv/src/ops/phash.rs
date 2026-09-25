//! Perceptual hash: a buffer → vector fingerprint.

#[allow(unused_imports)]
use crate::ops::ParamExt as _;
use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::ops::phash::{HashAlgorithm, PerceptualHashOp};

use super::{Literal, OpDef};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::OpShape;

/// Compute a perceptual hash fingerprint.
///
/// Example:
///     >>> Pipeline().source("image_bytes").perceptual_hash()
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct PerceptualHash {
    /// "perceptual" (pHash), "average" (aHash), "difference" (dHash).
    #[param(default = "perceptual")]
    pub algorithm: Literal<HashAlgorithm>,
    /// Number of bits in the hash (must be power of 2). It fixes the output
    /// vector length, so it is literal-only.
    #[param(default = 64)]
    pub hash_size: Literal<u32>,
}

impl OpDef for PerceptualHash {
    fn shape(&self) -> Option<OpShape> {
        None
    }

    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let PerceptualHash {
            algorithm,
            hash_size,
        } = self;
        if hash_size.get() == 0 {
            polars_bail!(ComputeError: "hash_size must be a positive integer");
        }
        Ok(GraphStep::PerceptualHash(
            PerceptualHashOp::new(algorithm.get()).with_hash_size(hash_size.get()),
        ))
    }
}
