//! View operations that perform zero-copy transformations.

use crate::core::dtype::OutputDTypeRule;
use crate::mode::{known, size, Exec, Mode};
use crate::ops::shape_rule::{OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use polars_cv_macros::{Ops, Resolve};

/// The view ops: zero-copy layout changes. One variant per wire op (see
/// `crate::mode`), plus the engine-internal N-D `Slice` and lattice rotations.
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
pub enum ViewOp<M: Mode = Exec> {
    /// Extract a rectangular region.
    #[op(name = "crop", sample = {"top": 1, "left": 1, "height": 2, "width": 2})]
    Crop {
        /// Top offset.
        #[param(default = 0)]
        top: M::V<u32>,
        /// Left offset.
        #[param(default = 0)]
        left: M::V<u32>,
        /// Crop height (None = to end).
        height: Option<M::V<u32>>,
        /// Crop width (None = to end).
        width: Option<M::V<u32>>,
    },
    /// Transpose dimensions.
    #[op(name = "transpose", sample = {"axes": [1, 0, 2]})]
    Transpose {
        /// New order of axes: a permutation of every input axis.
        axes: Vec<M::L<u32>>,
    },
    /// Reshape array to new dimensions.
    #[op(name = "reshape", sample = {"shape": [2, 2, 1]})]
    Reshape {
        /// New shape. The number of entries fixes the output rank; each entry may
        /// be a Polars expression.
        shape: Vec<M::V<u32>>,
    },
    /// Flip along specified axes.
    #[op(name = "flip", sample = {"axes": [1]})]
    Flip {
        /// Axes to flip.
        axes: Vec<M::L<u32>>,
    },
    /// Extract a single channel from a multi-channel image: a 2D [H, W] buffer
    /// from a [H, W, C] input.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").channel_select(index=0)  # Red channel
    #[op(name = "channel_select", sample = {"index": 0})]
    ChannelSelect {
        /// Channel index to extract (0-based). Accepts a Polars expression for
        /// per-row dynamic selection.
        index: M::V<u32>,
    },
    // --- Engine-internal, never on the wire.
    /// A window on every axis, `start..end` (`usize::MAX`: to the end of
    /// the axis): the engine's N-D slice, which a wire `Crop` executes as.
    Slice { start: Vec<usize>, end: Vec<usize> },
    /// Rotates 90 degrees clockwise (zero-copy via transpose + flip).
    Rotate90,
    /// Rotates 180 degrees (zero-copy via double flip).
    Rotate180,
    /// Rotates 270 degrees clockwise / 90 degrees counter-clockwise (zero-copy via transpose + flip).
    Rotate270,
}

fn axes_of<M: Mode>(axes: &[M::L<u32>]) -> Vec<usize> {
    axes.iter().map(|a| M::lit(a) as usize).collect()
}

impl<M: Mode> ViewOp<M> {
    /// Refuse a parameter combination no row can execute. A view op's
    /// parameters are independent (a permutation is checked against the
    /// input's rank by `validate`).
    pub fn check(&self) -> Result<(), String> {
        Ok(())
    }

    /// How this op's output shape follows from its input — the one
    /// definition, read on the `Wire` op at plan time and on the `Exec` op at
    /// execution.
    pub fn shape(&self) -> OpShape {
        let known = |v: &[usize]| v.iter().map(|&n| Sym::Known(n)).collect();
        match self {
            ViewOp::Crop {
                top,
                left,
                height,
                width,
            } => OpShape::Crop {
                start: vec![size::<M>(top), size::<M>(left), Sym::Known(0)],
                len: vec![
                    height.as_ref().map(size::<M>),
                    width.as_ref().map(size::<M>),
                    None,
                ],
            },
            ViewOp::Transpose { axes } => OpShape::Transpose(axes_of::<M>(axes)),
            ViewOp::Reshape { shape } => OpShape::Fixed(shape.iter().map(size::<M>).collect()),
            ViewOp::Flip { .. } | ViewOp::Rotate180 => OpShape::Preserve,
            // One entry per input axis, as `ViewBuffer::slice` produces. An
            // `end` of `usize::MAX` is "to the end of this axis".
            ViewOp::Slice { start, end } => OpShape::Crop {
                start: known(start),
                len: start
                    .iter()
                    .zip(end)
                    .map(|(&s, &e)| (e != usize::MAX).then(|| Sym::Known(e.saturating_sub(s))))
                    .collect(),
            },
            ViewOp::Rotate90 | ViewOp::Rotate270 => OpShape::SwapHw,
            ViewOp::ChannelSelect { .. } => OpShape::DropChannelAxis,
        }
    }
}

/// A crop's `(start, end)` bounds, one per axis.
pub type Window = (Vec<Sym<usize>>, Vec<Sym<usize>>);

impl<M: Mode> ViewOp<M> {
    /// The window a crop or slice keeps: `start..end` per axis, `usize::MAX`
    /// running to the end of that axis; a bound a per-row value sets is
    /// `PerRow`.
    pub fn window_of(&self) -> Option<Window> {
        match self {
            ViewOp::Crop {
                top,
                left,
                height,
                width,
            } => {
                let (top, left) = (size::<M>(top), size::<M>(left));
                let end_of = |origin: Sym<usize>, extent: &Option<M::V<u32>>| match extent {
                    None => Sym::Known(usize::MAX),
                    Some(extent) => match (origin, size::<M>(extent)) {
                        (Sym::Known(o), Sym::Known(e)) => Sym::Known(o + e),
                        _ => Sym::PerRow,
                    },
                };
                Some((
                    vec![top, left, Sym::Known(0)],
                    vec![
                        end_of(top, height),
                        end_of(left, width),
                        Sym::Known(usize::MAX),
                    ],
                ))
            }
            ViewOp::Slice { start, end } => Some((
                start.iter().map(|&s| Sym::Known(s)).collect(),
                end.iter().map(|&e| Sym::Known(e)).collect(),
            )),
            _ => None,
        }
    }

    /// The axes of a transpose or flip, as the kernels index them.
    pub fn axes(&self) -> Vec<usize> {
        match self {
            ViewOp::Transpose { axes } | ViewOp::Flip { axes } => axes_of::<M>(axes),
            _ => Vec::new(),
        }
    }
}

impl ViewOp {
    /// The window a crop or slice keeps (every bound known at execution).
    pub fn window(&self) -> Option<(Vec<usize>, Vec<usize>)> {
        let (start, end) = self.window_of()?;
        let known = |v: Vec<Sym<usize>>| -> Vec<usize> {
            v.into_iter()
                .map(|s| s.known().expect("an executed op's values are known"))
                .collect()
        };
        Some((known(start), known(end)))
    }

    /// A transpose by `perm`.
    pub fn transpose(perm: &[usize]) -> Self {
        ViewOp::Transpose {
            axes: perm.iter().map(|&a| a as u32).collect(),
        }
    }

    /// A flip of `axes`.
    pub fn flip(axes: &[usize]) -> Self {
        ViewOp::Flip {
            axes: axes.iter().map(|&a| a as u32).collect(),
        }
    }
}

impl<M: Mode> Op for ViewOp<M> {
    fn validate(
        &self,
        input_shapes: &[&[usize]],
        _input_dtypes: &[crate::DType],
    ) -> Result<(), crate::ops::validation::ValidationError> {
        use crate::ops::validation::{require_axes, ValidationError};
        let shape = input_shapes[0];
        match self {
            ViewOp::Transpose { .. } => {
                let perm = self.axes();
                let mut seen = vec![false; shape.len()];
                let is_permutation = perm.len() == shape.len()
                    && perm
                        .iter()
                        .all(|&a| a < seen.len() && !std::mem::replace(&mut seen[a], true));
                if is_permutation {
                    Ok(())
                } else {
                    Err(ValidationError::NotAPermutation {
                        axes: perm,
                        ndim: shape.len(),
                    })
                }
            }
            // A reshape that changes the element count would describe memory
            // the buffer does not own.
            ViewOp::Reshape { shape: new } => {
                // A per-row dimension is checked per row.
                let Some(new) = new
                    .iter()
                    .map(|d| known::<M, u32>(d).map(|d| d as usize))
                    .collect::<Option<Vec<usize>>>()
                else {
                    return Ok(());
                };
                let (have, want) = (
                    shape.iter().product::<usize>(),
                    new.iter().product::<usize>(),
                );
                if have == want {
                    Ok(())
                } else {
                    Err(ValidationError::InvalidParameter {
                        param: "shape".to_string(),
                        reason: format!(
                            "cannot reshape {shape:?} ({have} elements) to {new:?} ({want} elements)"
                        ),
                    })
                }
            }
            ViewOp::Flip { .. } => require_axes(shape, &self.axes()),
            ViewOp::Crop { .. } | ViewOp::Slice { .. } => {
                let (start, end) = self.window_of().expect("a crop or slice has a window");
                if start.len() < shape.len() || end.len() < shape.len() {
                    return Err(ValidationError::ShapeRequirement {
                        requirement: "crop bounds for every axis of the input",
                        got: shape.to_vec(),
                    });
                }
                // An `end` of `usize::MAX` is "to the end of this axis". Any
                // other bound past the axis is a window outside the input:
                // rejected rather than clamped, since clamping returns a
                // smaller region than the caller asked for (CR-42).
                // A bound a per-row value sets is checked per row.
                for (axis, &dim) in shape.iter().enumerate() {
                    let (s, e) = (start[axis].known(), end[axis].known());
                    let past_end = |e: usize| e != usize::MAX && e > dim;
                    if s.is_some_and(|s| s > dim) || e.is_some_and(past_end) {
                        let text = |b: Option<usize>| match b {
                            Some(usize::MAX) => "end".to_string(),
                            Some(b) => b.to_string(),
                            None => "<per-row>".to_string(),
                        };
                        let (s, end_text) = (text(s), text(e));
                        return Err(ValidationError::InvalidParameter {
                            param: "window".to_string(),
                            reason: format!(
                                "crop window {s}..{end_text} on axis {axis} lies outside the \
                                 input of shape {shape:?}"
                            ),
                        });
                    }
                }
                Ok(())
            }
            // Image rotations: a [H, W] or [H, W, C] buffer. Anything else was
            // returned unchanged or rotated over the wrong axes.
            ViewOp::Rotate90 | ViewOp::Rotate180 | ViewOp::Rotate270 => {
                crate::ops::validation::require_hw_or_hwc(shape)
            }
            // A per-row index is checked per row; the rank is checked now.
            ViewOp::ChannelSelect { index } => match (known::<M, u32>(index), shape) {
                (Some(index), [_, _, c]) if (index as usize) < *c => Ok(()),
                (Some(0), [_, _]) | (None, [_, _] | [_, _, _]) => Ok(()),
                (index, _) => Err(ValidationError::InvalidParameter {
                    param: "index".to_string(),
                    reason: match index {
                        Some(index) => format!("channel {index} of a buffer of shape {shape:?}"),
                        None => format!("a channel of a buffer of shape {shape:?}"),
                    },
                }),
            },
        }
    }

    fn name(&self) -> &'static str {
        match self {
            ViewOp::Transpose { .. } => "Transpose",
            ViewOp::Reshape { .. } => "Reshape",
            ViewOp::Flip { .. } => "Flip",
            ViewOp::Crop { .. } | ViewOp::Slice { .. } => "Crop",
            ViewOp::Rotate90 => "Rotate90",
            ViewOp::Rotate180 => "Rotate180",
            ViewOp::Rotate270 => "Rotate270",
            ViewOp::ChannelSelect { .. } => "ChannelSelect",
        }
    }

    fn shape(&self) -> OpShape {
        ViewOp::shape(self)
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        // View ops (transpose, reshape, flip, crop, channel_select, rotate90)
        // only rearrange existing elements — the dtype is always preserved.
        OutputDTypeRule::PreserveInput
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::View
    }

    fn identity_rule(&self) -> IdentityRule {
        match self {
            // A full-frame crop at the origin, or a same-shape reshape (a
            // row-major no-op), moves no data; `OpShape::preserves` decides
            // whether this one is.
            ViewOp::Crop { .. } | ViewOp::Slice { .. } | ViewOp::Reshape { .. } => {
                IdentityRule::WhenShapePreserved
            }
            // Flip/transpose/rotate/channel-select move pixels or drop an axis
            // even when the shape is preserved (a square transpose, a 180°
            // rotate), so none is ever a no-op.
            ViewOp::Transpose { .. }
            | ViewOp::Flip { .. }
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270
            | ViewOp::ChannelSelect { .. } => IdentityRule::Never,
        }
    }

    fn is_spatial_window(&self) -> bool {
        match self {
            // A wire crop narrows only the H/W plane: a window.
            ViewOp::Crop { .. } => true,
            // A slice is a window only when it narrows the H/W plane and
            // leaves every further axis whole (`start == 0`, the `usize::MAX`
            // end); one that slices the channel axis would not commute with a
            // channel-changing pointwise op.
            ViewOp::Slice { start, end } => {
                start.iter().skip(2).all(|&s| s == 0)
                    && end.iter().skip(2).all(|&e| e == usize::MAX)
            }
            // Reshape/transpose/flip/rotate/channel-select are not H/W crops.
            ViewOp::Transpose { .. }
            | ViewOp::Reshape { .. }
            | ViewOp::Flip { .. }
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270
            | ViewOp::ChannelSelect { .. } => false,
        }
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        match self {
            // Picks a channel at the same (y, x) — spatially the identity.
            ViewOp::ChannelSelect { .. } => SpatialDependency::Pointwise,
            // Permute/reshape/flip/crop/rotate all remap coordinates (crop
            // offsets the origin; transpose/reshape/rotate move axes).
            ViewOp::Transpose { .. }
            | ViewOp::Reshape { .. }
            | ViewOp::Flip { .. }
            | ViewOp::Crop { .. }
            | ViewOp::Slice { .. }
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270 => SpatialDependency::geometric(),
        }
    }

    fn infer_strides(&self, _input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
        match self {
            ViewOp::Transpose { .. } => {
                Some(self.axes().iter().map(|&i| input_strides[i]).collect())
            }
            ViewOp::Reshape { .. } => {
                // Reshape as a view operation defers stride calculation to runtime/planner
                // since we need to verify contiguity with the actual DType.
                // Both contiguous and non-contiguous cases return None here.
                None
            }
            ViewOp::Flip { .. } => {
                let mut new_strides = input_strides.to_vec();
                for axis in self.axes() {
                    new_strides[axis] = -new_strides[axis];
                }
                Some(new_strides)
            }
            ViewOp::Crop { .. } | ViewOp::Slice { .. } => Some(input_strides.to_vec()),
            ViewOp::Rotate90 => {
                if input_strides.len() >= 2 {
                    let mut new_strides = input_strides.to_vec();
                    new_strides.swap(0, 1);
                    new_strides[1] = -new_strides[1];
                    Some(new_strides)
                } else {
                    Some(input_strides.to_vec())
                }
            }
            ViewOp::Rotate180 => {
                if input_strides.len() >= 2 {
                    let mut new_strides = input_strides.to_vec();
                    new_strides[0] = -new_strides[0];
                    new_strides[1] = -new_strides[1];
                    Some(new_strides)
                } else {
                    Some(input_strides.to_vec())
                }
            }
            ViewOp::Rotate270 => {
                if input_strides.len() >= 2 {
                    let mut new_strides = input_strides.to_vec();
                    new_strides.swap(0, 1);
                    new_strides[0] = -new_strides[0];
                    Some(new_strides)
                } else {
                    Some(input_strides.to_vec())
                }
            }
            ViewOp::ChannelSelect { .. } => {
                // Drop the last stride dimension (channel axis)
                if input_strides.len() >= 3 {
                    Some(input_strides[..2].to_vec())
                } else {
                    Some(input_strides.to_vec())
                }
            }
        }
    }
}
