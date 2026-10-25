//! View operations that perform zero-copy transformations.

use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule};
use crate::mode::{known, size, Exec, Mode};
use crate::ops::shape_rule::{place_axis, resolve_axis, OpShape, Sym};
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use polars_cv_macros::{Ops, Resolve};

/// The view ops: zero-copy layout changes. One variant per wire op (see
/// `crate::mode`), plus the engine-internal N-D `Slice` and lattice rotations.
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
pub enum ViewOp<M: Mode = Exec> {
    /// Extract a rectangular region inside the input.
    ///
    /// A window reaching past the input's edge is refused. ``crop_clip``
    /// keeps the part of such a window inside the input, and ``crop_pad``
    /// keeps its requested size, filled outside.
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
    /// Extract the part of a window that lies inside the input: strips, tiles
    /// and sliding windows near a border. Offsets may be negative, and the
    /// output is smaller than asked where the window reaches past the edge; a
    /// window with nothing inside the input is refused.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").crop_clip(left=8, width=24)
    #[op(name = "crop_clip", sample = {"top": -1, "left": 1, "height": 4, "width": 2})]
    CropClip {
        /// Top offset; may be negative.
        #[param(default = 0)]
        top: M::V<i64>,
        /// Left offset; may be negative.
        #[param(default = 0)]
        left: M::V<i64>,
        /// Window height (None = to end).
        height: Option<M::V<u32>>,
        /// Window width (None = to end).
        width: Option<M::V<u32>>,
    },
    /// Extract a ``height`` x ``width`` window at its requested size, filling
    /// what lies outside the input with ``value``: the in-bounds crop
    /// followed by a constant ``pad``. Offsets may be negative, and a window
    /// entirely outside the input is all fill.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").crop_pad(top=-8, left=-8, height=64, width=64)
    #[op(name = "crop_pad", sample = {"top": -2, "left": -2, "height": 68, "width": 68, "value": 7.0})]
    CropPad {
        /// Top offset; may be negative.
        #[param(default = 0)]
        top: M::V<i64>,
        /// Left offset; may be negative.
        #[param(default = 0)]
        left: M::V<i64>,
        /// Window height: the output's.
        height: M::V<u32>,
        /// Window width: the output's.
        width: M::V<u32>,
        /// The fill outside the input.
        #[param(default = 0.0)]
        value: M::V<f32>,
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
    /// Flip along specified axes, in every row or where ``when`` holds.
    ///
    /// Example:
    ///     >>> # Orient every view with the chest wall on the left
    ///     >>> pipe = Pipeline().source("image_bytes").flip_h(when=pl.col("view").str.starts_with("R"))
    #[op(name = "flip", sample = {"axes": [1], "when": false})]
    Flip {
        /// Axes to flip.
        axes: Vec<M::L<u32>>,
        /// Whether to flip (default True). Accepts a Polars expression for a
        /// per-row choice: the shape is the same either way.
        #[param(default = true)]
        when: M::V<bool>,
    },
    /// Extract one channel of a multi-channel image: ``[H, W, 1]`` from a
    /// ``[H, W, C]`` input (the shape ``grayscale`` and ``rasterize`` give),
    /// or ``[H, W]`` with ``keepdims=False``. A ``[H, W]`` input is returned
    /// as it is.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").channel_select(index=0)  # Red channel
    #[op(name = "channel_select", sample = {"index": 0, "keepdims": false})]
    ChannelSelect {
        /// Channel index to extract (0-based). Accepts a Polars expression for
        /// per-row dynamic selection.
        index: M::V<u32>,
        /// Keep the channel axis, as size 1 (default), or drop it.
        #[param(default = true)]
        keepdims: M::L<bool>,
    },
    /// Remove an axis of size 1: `[H, W, 1]` to `[H, W]` with `axis=-1`. The
    /// same elements in the same order, with no copy. An axis of another size
    /// is refused.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("array").squeeze(-1)
    #[op(name = "squeeze", sample = {"axis": -1})]
    Squeeze {
        /// The axis to remove, counted from the end when negative.
        axis: M::L<i32>,
    },
    /// Insert an axis of size 1: `[H, W]` to `[H, W, 1]` with `axis=-1`. The
    /// same elements in the same order, with no copy.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("array").expand_dims(-1)
    #[op(name = "expand_dims", sample = {"axis": -1})]
    ExpandDims {
        /// Where the new axis goes in the output, counted from the end when
        /// negative.
        axis: M::L<i32>,
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

/// Where a clipped or padded crop falls in a concrete input: the in-bounds
/// `start..end` per axis (as [`ViewOp::Slice`] takes it), and for
/// [`ViewOp::CropPad`] the padding around it.
#[derive(Debug, Clone, PartialEq)]
pub struct Placement {
    pub start: Vec<usize>,
    pub end: Vec<usize>,
    /// `(top, bottom, left, right, value)`.
    pub pad: Option<(u32, u32, u32, u32, f32)>,
}

impl ViewOp {
    /// Where this clipped or padded crop falls in an input of `shape`
    /// (validated to have an image plane), by [`place_axis`]; `None` for any
    /// other op.
    pub fn place(&self, shape: &[usize]) -> Option<Placement> {
        let (top, left, height, width, fill) = match self {
            ViewOp::CropClip {
                top,
                left,
                height,
                width,
            } => (*top, *left, *height, *width, None),
            ViewOp::CropPad {
                top,
                left,
                height,
                width,
                value,
            } => (*top, *left, Some(*height), Some(*width), Some(*value)),
            _ => return None,
        };
        let (h, w) = (shape.first().copied()?, shape.get(1).copied()?);
        let at = |origin, extent: Option<u32>, dim| {
            let (lo, hi, before, after) = place_axis(origin, extent.map(u64::from), dim);
            // Within the window's own extent, a `u32`.
            (lo, hi, before as u32, after as u32)
        };
        let (t0, t1, pt, pb) = at(top, height, h);
        let (l0, l1, pl, pr) = at(left, width, w);
        let mut start = vec![0; shape.len()];
        let mut end = vec![usize::MAX; shape.len()];
        (start[0], end[0], start[1], end[1]) = (t0, t1, l0, l1);
        let pad = fill
            .map(|value| (pt, pb, pl, pr, value))
            .filter(|&(t, b, l, r, _)| t + b + l + r > 0);
        Some(Placement { start, end, pad })
    }
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
            ViewOp::CropClip {
                top,
                left,
                height,
                width,
            } => OpShape::ClipWindow {
                start: [M::sym(top), M::sym(left)],
                len: [
                    height.as_ref().map(size::<M>),
                    width.as_ref().map(size::<M>),
                ],
            },
            ViewOp::CropPad { height, width, .. } => OpShape::SetHw {
                h: size::<M>(height),
                w: size::<M>(width),
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
            ViewOp::ChannelSelect { keepdims, .. } => match M::lit(keepdims) {
                true => OpShape::SingleChannel,
                false => OpShape::DropChannelAxis,
            },
            ViewOp::Squeeze { axis } => OpShape::RemoveAxis(M::lit(axis) as isize),
            ViewOp::ExpandDims { axis } => OpShape::InsertAxis(M::lit(axis) as isize),
        }
    }

    /// The position of a squeeze's or expand_dims' axis in an input of
    /// `rank`, its sign resolved (`None` outside the rank).
    pub fn axis_position(&self, rank: usize) -> Option<usize> {
        match self {
            ViewOp::Squeeze { axis } => resolve_axis(M::lit(axis) as isize, rank),
            ViewOp::ExpandDims { axis } => resolve_axis(M::lit(axis) as isize, rank + 1),
            _ => None,
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
            ViewOp::Transpose { axes } | ViewOp::Flip { axes, .. } => axes_of::<M>(axes),
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
            when: true,
        }
    }
}

impl<M: Mode> Op for ViewOp<M> {
    /// `Any`: a view only re-strides or re-labels the buffer, so no element is
    /// read and every dtype passes through.
    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Any
    }

    /// `None`: zero-copy, so there is no computation to carry out in another dtype.
    fn working_dtype(&self) -> Option<DType> {
        None
    }

    fn validate(
        &self,
        input_shapes: &[&[crate::ops::Dim]],
        _input_dtypes: &[crate::PlannedDType],
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
                // An unknown input size is checked per row.
                let Some(have) = shape.iter().map(|d| d.known()).product::<Option<usize>>() else {
                    return Ok(());
                };
                let want = new.iter().product::<usize>();
                if have == want {
                    Ok(())
                } else {
                    Err(ValidationError::InvalidParameter {
                        param: "shape".to_string(),
                        reason: format!(
                            "cannot reshape {} ({have} elements) to {new:?} ({want} elements)",
                            crate::ops::shape_rule::show_dims(shape)
                        ),
                    })
                }
            }
            ViewOp::Flip { .. } => require_axes(shape, &self.axes()),
            // Any window of the image plane, inside it or not.
            ViewOp::CropPad { .. } => crate::ops::validation::require_hw_or_hwc(shape),
            ViewOp::CropClip {
                top,
                left,
                height,
                width,
            } => {
                // A clip must keep something of the image plane.
                crate::ops::validation::require_hw_or_hwc(shape)?;
                let axes = [(top, height), (left, width)];
                for (axis, (origin, extent)) in axes.into_iter().enumerate() {
                    let dim = shape.get(axis).and_then(|d| d.known());
                    let extent = extent.as_ref().map(|e| known::<M, u32>(e));
                    let (Some(dim), Some(origin), Some(extent)) = (
                        dim,
                        known::<M, i64>(origin),
                        extent.map_or(Some(None), |e| e.map(Some)),
                    ) else {
                        continue;
                    };
                    let (lo, hi, _, _) = place_axis(origin, extent.map(u64::from), dim);
                    if lo == hi {
                        let end = extent.map_or("end".to_string(), |e| {
                            (i128::from(origin) + i128::from(e)).to_string()
                        });
                        return Err(ValidationError::InvalidParameter {
                            param: "window".to_string(),
                            reason: format!(
                                "crop window {origin}..{end} on axis {axis} lies entirely \
                                 outside the input of shape {}",
                                crate::ops::shape_rule::show_dims(shape)
                            ),
                        });
                    }
                }
                Ok(())
            }
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
                // smaller region than the caller asked for.
                // A bound a per-row value sets is checked per row.
                // Each axis whose size is known; an unknown one is checked
                // per row.
                for (axis, dim) in shape.iter().enumerate() {
                    let Some(dim) = dim.known() else { continue };
                    let (s, e) = (start[axis].known(), end[axis].known());
                    let past_end = |e: usize| e != usize::MAX && e > dim;
                    if s.is_some_and(|s| s > dim) || e.is_some_and(past_end) {
                        let text = |b: Option<usize>| match b {
                            Some(usize::MAX) => "end".to_string(),
                            Some(b) => b.to_string(),
                            None => "<per-row>".to_string(),
                        };
                        // Unsigned arithmetic in polars (`UInt32` - n) wraps
                        // below zero to ~2^32 before the value reaches here.
                        let wrapped = [s, e]
                            .into_iter()
                            .flatten()
                            .any(|b| b != usize::MAX && b >= 1 << 31);
                        let hint = if wrapped {
                            "; a bound near 2^32 is usually an unsigned subtraction that \
                             wrapped below zero in polars: compute offsets as Int64"
                        } else {
                            ""
                        };
                        let (s, end_text) = (text(s), text(e));
                        return Err(ValidationError::InvalidParameter {
                            param: "window".to_string(),
                            reason: format!(
                                "crop window {s}..{end_text} on axis {axis} lies outside the \
                                 input of shape {}{hint}",
                                crate::ops::shape_rule::show_dims(shape)
                            ),
                        });
                    }
                }
                Ok(())
            }
            ViewOp::Squeeze { axis } | ViewOp::ExpandDims { axis } => {
                let axis = M::lit(axis);
                let Some(at) = self.axis_position(shape.len()) else {
                    let (what, positions) = match self {
                        ViewOp::Squeeze { .. } => ("an axis", shape.len()),
                        _ => ("a position", shape.len() + 1),
                    };
                    return Err(ValidationError::InvalidParameter {
                        param: "axis".to_string(),
                        reason: format!(
                            "axis {axis} is not {what} of a buffer of shape {} (valid: {}..{})",
                            crate::ops::shape_rule::show_dims(shape),
                            -(positions as i64),
                            positions
                        ),
                    });
                };
                // An unknown size is checked per row.
                match (self, shape.get(at).and_then(|d| d.known())) {
                    (ViewOp::Squeeze { .. }, Some(n)) if n != 1 => {
                        Err(ValidationError::InvalidParameter {
                            param: "axis".to_string(),
                            reason: format!(
                                "only an axis of size 1 can be removed; axis {axis} of {} has \
                                 size {n}",
                                crate::ops::shape_rule::show_dims(shape)
                            ),
                        })
                    }
                    _ => Ok(()),
                }
            }
            // Image rotations: a [H, W] or [H, W, C] buffer. Anything else was
            // returned unchanged or rotated over the wrong axes.
            ViewOp::Rotate90 | ViewOp::Rotate180 | ViewOp::Rotate270 => {
                crate::ops::validation::require_hw_or_hwc(shape)
            }
            // A per-row index is checked per row; the rank is checked now.
            // A per-row index or an unknown channel count is checked per row;
            // the rank is checked now.
            ViewOp::ChannelSelect { index, .. } => match (known::<M, u32>(index), shape) {
                (Some(index), [_, _, c]) if c.known().is_none_or(|c| (index as usize) < c) => {
                    Ok(())
                }
                (Some(0), [_, _]) | (None, [_, _] | [_, _, _]) => Ok(()),
                (index, _) => {
                    let shape = crate::ops::shape_rule::show_dims(shape);
                    Err(ValidationError::InvalidParameter {
                        param: "index".to_string(),
                        reason: match index {
                            Some(index) => format!("channel {index} of a buffer of shape {shape}"),
                            None => format!("a channel of a buffer of shape {shape}"),
                        },
                    })
                }
            },
        }
    }

    fn name(&self) -> &'static str {
        match self {
            ViewOp::Transpose { .. } => "Transpose",
            ViewOp::Reshape { .. } => "Reshape",
            ViewOp::Flip { .. } => "Flip",
            ViewOp::Crop { .. } | ViewOp::Slice { .. } => "Crop",
            ViewOp::CropClip { .. } => "CropClip",
            ViewOp::CropPad { .. } => "CropPad",
            ViewOp::Rotate90 => "Rotate90",
            ViewOp::Rotate180 => "Rotate180",
            ViewOp::Rotate270 => "Rotate270",
            ViewOp::ChannelSelect { .. } => "ChannelSelect",
            ViewOp::Squeeze { .. } => "Squeeze",
            ViewOp::ExpandDims { .. } => "ExpandDims",
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
        match self {
            // A reshape reinterprets row-major order, so a flipped, transposed
            // or narrowed view is packed first (as NumPy's reshape copies).
            ViewOp::Reshape { .. } => MemoryEffect::ViewOfContiguous,
            // It executes as a slice and the `pad` op, whose packed rows it
            // reads (`ImageOpKind::Pad`'s effect).
            ViewOp::CropPad { .. } => MemoryEffect::RequiresDenseRows,
            _ => MemoryEffect::View,
        }
    }

    fn identity_rule(&self) -> IdentityRule {
        match self {
            // A padded crop's shape is its window's size wherever the window
            // sits, so a preserved shape does not say the data stayed put
            // (`top=-1` shifts the image down a row at the same size).
            ViewOp::CropPad { .. } => IdentityRule::Never,
            // A full-frame crop at the origin, or a same-shape reshape (a
            // row-major no-op), moves no data; `OpShape::preserves` decides
            // whether this one is. A clipped window keeps the shape only by
            // covering the whole input.
            ViewOp::Crop { .. }
            | ViewOp::CropClip { .. }
            | ViewOp::Slice { .. }
            | ViewOp::Reshape { .. } => IdentityRule::WhenShapePreserved,
            // Flip/transpose/rotate/channel-select move pixels or drop an axis
            // even when the shape is preserved (a square transpose, a 180°
            // rotate), so none is ever a no-op; a squeeze or expand_dims
            // always changes the rank.
            ViewOp::Squeeze { .. }
            | ViewOp::ExpandDims { .. }
            | ViewOp::Transpose { .. }
            | ViewOp::Flip { .. }
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270
            | ViewOp::ChannelSelect { .. } => IdentityRule::Never,
        }
    }

    fn is_spatial_window(&self) -> bool {
        match self {
            // A wire crop, clipped or not, narrows only the H/W plane: a
            // window. A padded one also fills outside the input, which a
            // pointwise op moved ahead of it would not see: not a window.
            ViewOp::Crop { .. } | ViewOp::CropClip { .. } => true,
            ViewOp::CropPad { .. } => false,
            // A slice is a window only when it narrows the H/W plane and
            // leaves every further axis whole (`start == 0`, the `usize::MAX`
            // end); one that slices the channel axis would not commute with a
            // channel-changing pointwise op.
            ViewOp::Slice { start, end } => {
                start.iter().skip(2).all(|&s| s == 0)
                    && end.iter().skip(2).all(|&e| e == usize::MAX)
            }
            // Reshape/transpose/flip/rotate/channel-select/squeeze/expand_dims
            // are not H/W crops.
            ViewOp::Squeeze { .. }
            | ViewOp::ExpandDims { .. }
            | ViewOp::Transpose { .. }
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
            // offsets the origin; transpose/reshape/rotate move axes), and
            // a squeeze or expand_dims at the front moves H and W.
            ViewOp::Squeeze { .. }
            | ViewOp::ExpandDims { .. }
            | ViewOp::Transpose { .. }
            | ViewOp::Reshape { .. }
            | ViewOp::Flip { .. }
            | ViewOp::Crop { .. }
            | ViewOp::CropClip { .. }
            | ViewOp::CropPad { .. }
            | ViewOp::Slice { .. }
            | ViewOp::Rotate90
            | ViewOp::Rotate180
            | ViewOp::Rotate270 => SpatialDependency::geometric(),
        }
    }

    fn infer_strides(&self, input_shape: &[usize], input_strides: &[isize]) -> Option<Vec<isize>> {
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
            ViewOp::Flip { when, .. } if M::sym(when).known() == Some(false) => {
                Some(input_strides.to_vec())
            }
            ViewOp::Flip { .. } => {
                let mut new_strides = input_strides.to_vec();
                for axis in self.axes() {
                    new_strides[axis] = -new_strides[axis];
                }
                Some(new_strides)
            }
            ViewOp::CropPad { .. } => None,
            ViewOp::Crop { .. } | ViewOp::CropClip { .. } | ViewOp::Slice { .. } => {
                Some(input_strides.to_vec())
            }
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
            // As `ViewBuffer::remove_axis`/`insert_axis` lay them out.
            ViewOp::Squeeze { .. } => {
                let at = self.axis_position(input_strides.len())?;
                let mut strides = input_strides.to_vec();
                strides.remove(at);
                Some(strides)
            }
            ViewOp::ExpandDims { .. } => {
                let at = self.axis_position(input_strides.len())?;
                // Past the last axis a packed layout steps one element, the
                // last axis's stride when it is packed; with no axis there is
                // no element size to read, and the output is packed anew.
                let stride = match input_shape.get(at) {
                    Some(&n) => input_strides[at] * n as isize,
                    None => *input_strides.last()?,
                };
                let mut strides = input_strides.to_vec();
                strides.insert(at, stride);
                Some(strides)
            }
            ViewOp::ChannelSelect { keepdims, .. } => {
                // The channel axis kept (size 1, any stride) or dropped.
                if input_strides.len() >= 3 && !M::lit(keepdims) {
                    Some(input_strides[..2].to_vec())
                } else {
                    Some(input_strides.to_vec())
                }
            }
        }
    }
}
