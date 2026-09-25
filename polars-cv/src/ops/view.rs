//! View ops: zero-copy reinterpretations of the buffer.

use polars::prelude::*;
use polars_cv_macros::Op;
use serde::{Deserialize, Serialize};
use view_buffer::{ViewDto, ViewOp};

use super::{Literal, OpDef, Param};
use crate::graph::step::GraphStep;
use crate::params::ParamCtx;
use view_buffer::ops::{OpShape, Sym};

/// Extract a rectangular region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Crop {
    /// Top offset.
    #[param(default = 0)]
    pub top: Param<u32>,
    /// Left offset.
    #[param(default = 0)]
    pub left: Param<u32>,
    /// Crop height (None = to end).
    pub height: Option<Param<u32>>,
    /// Crop width (None = to end).
    pub width: Option<Param<u32>>,
}

impl OpDef for Crop {
    fn shape(&self) -> Option<OpShape> {
        let size = |p: &Option<Param<u32>>| p.as_ref().map(Param::size);
        Some(OpShape::Crop {
            start: vec![self.top.size(), self.left.size(), Sym::Known(0)],
            len: vec![size(&self.height), size(&self.width), None],
        })
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Crop {
            top,
            left,
            height,
            width,
        } = self;
        // Bounds are `u32`, so a negative one never gets here. An absent extent
        // means "to the end of that axis" (`usize::MAX`); a window that runs
        // past the input is rejected by `ViewOp::Crop`'s `validate`, where the
        // input shape is known (CR-42).
        let top = top.resolve(row, ctx)? as usize;
        let left = left.resolve(row, ctx)? as usize;
        let end_of = |origin: usize, extent: &Option<Param<u32>>| -> PolarsResult<usize> {
            match extent {
                None => Ok(usize::MAX),
                Some(p) => Ok(origin + p.resolve(row, ctx)? as usize),
            }
        };
        let start = vec![top, left, 0];
        let end = vec![end_of(top, height)?, end_of(left, width)?, usize::MAX];
        Ok(GraphStep::Buffer(ViewDto::View(ViewOp::Crop {
            start,
            end,
        })))
    }
}

/// Transpose dimensions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Transpose {
    /// New order of axes: a permutation of every input axis.
    pub axes: Vec<Literal<u32>>,
}

impl OpDef for Transpose {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Transpose(axes_of(&self.axes)))
    }

    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Transpose { axes } = self;
        Ok(GraphStep::Buffer(ViewDto::View(ViewOp::Transpose(
            axes_of(axes),
        ))))
    }
}

/// Reshape array to new dimensions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Reshape {
    /// New shape. The number of entries fixes the output rank; each entry may
    /// be a Polars expression.
    pub shape: Vec<Param<u32>>,
}

impl OpDef for Reshape {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Fixed(self.shape.iter().map(Param::size).collect()))
    }

    fn resolve(&self, row: usize, ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Reshape { shape } = self;
        let shape = shape
            .iter()
            .map(|d| d.resolve(row, ctx).map(|d| d as usize))
            .collect::<PolarsResult<_>>()?;
        Ok(GraphStep::Buffer(ViewDto::View(ViewOp::Reshape(shape))))
    }
}

/// Flip along specified axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[serde(deny_unknown_fields)]
pub struct Flip {
    /// Axes to flip.
    pub axes: Vec<Literal<u32>>,
}

impl OpDef for Flip {
    fn shape(&self) -> Option<OpShape> {
        Some(OpShape::Preserve)
    }

    fn resolve(&self, _row: usize, _ctx: &ParamCtx) -> PolarsResult<GraphStep> {
        let Flip { axes } = self;
        Ok(GraphStep::Buffer(ViewDto::View(ViewOp::Flip(axes_of(
            axes,
        )))))
    }
}

fn axes_of(axes: &[Literal<u32>]) -> Vec<usize> {
    axes.iter().map(|a| a.get() as usize).collect()
}
