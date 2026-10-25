//! Reduction kernels: the global, per-axis, block and masked-select loops
//! behind `ReductionOp::execute`.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, ViewType};
use crate::ops::reduction::{percentile_of, Percent, Reduction, ReductionOp, Statistic};
use crate::ops::util::{coords_to_linear, displaces, linear_to_coords, maximum, minimum};
use std::cmp::Ordering::{Greater, Less};

impl ReductionOp {
    /// Execute popcount reduction - count all set bits in the buffer.
    pub(crate) fn execute_popcount(&self, buffer: &ViewBuffer) -> ViewBuffer {
        let count: u64 = match buffer.dtype() {
            DType::U8 => buffer
                .as_slice::<u8>()
                .iter()
                .map(|x| x.count_ones() as u64)
                .sum(),
            DType::I8 => buffer
                .as_slice::<i8>()
                .iter()
                .map(|x| (*x as u8).count_ones() as u64)
                .sum(),
            DType::U16 => buffer
                .as_slice::<u16>()
                .iter()
                .map(|x| x.count_ones() as u64)
                .sum(),
            DType::I16 => buffer
                .as_slice::<i16>()
                .iter()
                .map(|x| (*x as u16).count_ones() as u64)
                .sum(),
            DType::U32 => buffer
                .as_slice::<u32>()
                .iter()
                .map(|x| x.count_ones() as u64)
                .sum(),
            DType::I32 => buffer
                .as_slice::<i32>()
                .iter()
                .map(|x| (*x as u32).count_ones() as u64)
                .sum(),
            DType::U64 => buffer
                .as_slice::<u64>()
                .iter()
                .map(|x| x.count_ones() as u64)
                .sum(),
            DType::I64 => buffer
                .as_slice::<i64>()
                .iter()
                .map(|x| (*x as u64).count_ones() as u64)
                .sum(),
            // For floats, cast to i64 and count bits
            DType::F32 => buffer
                .as_slice::<f32>()
                .iter()
                .map(|x| (*x as i64 as u64).count_ones() as u64)
                .sum(),
            DType::F64 => buffer
                .as_slice::<f64>()
                .iter()
                .map(|x| (*x as i64 as u64).count_ones() as u64)
                .sum(),
        };

        ViewBuffer::from_scalar(count as f64)
    }

    /// Execute percentile reduction with linear interpolation (matches numpy default).
    pub(crate) fn execute_percentile(&self, buffer: &ViewBuffer, q: f64) -> ViewBuffer {
        let mut values: Vec<f64> = match buffer.dtype() {
            DType::U8 => buffer.as_slice::<u8>().iter().map(|&x| x as f64).collect(),
            DType::I8 => buffer.as_slice::<i8>().iter().map(|&x| x as f64).collect(),
            DType::U16 => buffer.as_slice::<u16>().iter().map(|&x| x as f64).collect(),
            DType::I16 => buffer.as_slice::<i16>().iter().map(|&x| x as f64).collect(),
            DType::U32 => buffer.as_slice::<u32>().iter().map(|&x| x as f64).collect(),
            DType::I32 => buffer.as_slice::<i32>().iter().map(|&x| x as f64).collect(),
            DType::U64 => buffer.as_slice::<u64>().iter().map(|&x| x as f64).collect(),
            DType::I64 => buffer.as_slice::<i64>().iter().map(|&x| x as f64).collect(),
            DType::F32 => buffer.as_slice::<f32>().iter().map(|&x| x as f64).collect(),
            DType::F64 => buffer.as_slice::<f64>().to_vec(),
        };

        ViewBuffer::from_scalar(percentile_of(&mut values, q))
    }

    pub(crate) fn execute_typed<T>(&self, buffer: &ViewBuffer) -> ViewBuffer
    where
        T: Copy + Default + PartialOrd + num_traits::Num + num_traits::NumCast + ViewType + 'static,
    {
        let data = buffer.as_slice::<T>();
        let _shape = buffer.shape();

        match self {
            ReductionOp::Max { axis: None } => {
                assert!(!data.is_empty(), "Cannot reduce Max on empty buffer");
                let max_val = data.iter().copied().fold(data[0], maximum);
                ViewBuffer::from_scalar(max_val)
            }
            ReductionOp::Min { axis: None } => {
                assert!(!data.is_empty(), "Cannot reduce Min on empty buffer");
                let min_val = data.iter().copied().fold(data[0], minimum);
                ViewBuffer::from_scalar(min_val)
            }
            ReductionOp::Mean { axis: None } => {
                let sum: f64 = data
                    .iter()
                    .copied()
                    .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                    .sum();
                let mean = sum / data.len() as f64;
                ViewBuffer::from_scalar(mean)
            }
            ReductionOp::Std { axis: None, ddof } => {
                let n = data.len() as f64;
                let denominator = n - *ddof as f64;
                if denominator <= 0.0 {
                    return ViewBuffer::from_scalar(f64::NAN);
                }
                let sum: f64 = data
                    .iter()
                    .copied()
                    .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                    .sum();
                let mean = sum / n;
                let variance: f64 = data
                    .iter()
                    .copied()
                    .map(|x| {
                        let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);
                        (xf - mean).powi(2)
                    })
                    .sum::<f64>()
                    / denominator;
                let std = variance.sqrt();
                ViewBuffer::from_scalar(std)
            }
            ReductionOp::Sum => ViewBuffer::from_scalar(sum_f64(data)),
            // Axis-based reductions
            ReductionOp::Max { axis: Some(ax) } => {
                self.reduce_axis::<T, _>(buffer, *ax as usize, |slice: &[T]| {
                    slice.iter().copied().fold(slice[0], maximum)
                })
            }
            ReductionOp::Min { axis: Some(ax) } => {
                self.reduce_axis::<T, _>(buffer, *ax as usize, |slice: &[T]| {
                    slice.iter().copied().fold(slice[0], minimum)
                })
            }
            ReductionOp::Mean { axis: Some(ax) } => {
                // For axis reduction, output is float
                reduce_axis_to_f64::<T, _>(buffer, *ax as usize, |slice: &[T]| {
                    let sum: f64 = slice
                        .iter()
                        .copied()
                        .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                        .sum();
                    sum / slice.len() as f64
                })
            }
            ReductionOp::Std {
                axis: Some(ax),
                ddof,
            } => {
                let ddof_val = *ddof;
                reduce_axis_to_f64::<T, _>(buffer, *ax as usize, move |slice: &[T]| {
                    let n = slice.len() as f64;
                    let denominator = n - ddof_val as f64;
                    if denominator <= 0.0 {
                        return f64::NAN;
                    }
                    let sum: f64 = slice
                        .iter()
                        .copied()
                        .map(|x| num_traits::NumCast::from(x).unwrap_or(0.0))
                        .sum();
                    let mean = sum / n;
                    let variance: f64 = slice
                        .iter()
                        .copied()
                        .map(|x| {
                            let xf: f64 = num_traits::NumCast::from(x).unwrap_or(0.0);
                            (xf - mean).powi(2)
                        })
                        .sum::<f64>()
                        / denominator;
                    variance.sqrt()
                })
            }
            ReductionOp::ArgMax { axis } => {
                self.reduce_axis_argmax::<T>(buffer, *axis as usize, true)
            }
            ReductionOp::ArgMin { axis } => {
                self.reduce_axis_argmax::<T>(buffer, *axis as usize, false)
            }
            // PopCount and Percentile are handled specially in execute() before calling execute_typed()
            ReductionOp::PopCount => unreachable!("PopCount is handled in execute()"),
            ReductionOp::Blocks { .. } => unreachable!("Blocks is handled in execute()"),
            ReductionOp::By { .. } => unreachable!("By is handled in execute()"),
            ReductionOp::Percentile { .. } => unreachable!("Percentile is handled in execute()"),
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn reduce_axis<T, F>(&self, buffer: &ViewBuffer, axis: usize, f: F) -> ViewBuffer
    where
        T: Copy + Default + ViewType + 'static,
        F: Fn(&[T]) -> T,
    {
        let shape = buffer.shape();
        let data = buffer.as_slice::<T>();

        // Calculate output shape (remove the reduced axis)
        let mut out_shape: Vec<usize> = shape.to_vec();
        let axis_size = out_shape.remove(axis);

        // Coordinate mapping must run over the REAL remaining dims (possibly
        // none, for a 1-D input); the [1] clamp below only pads the returned
        // buffer's shape.
        let coord_shape = out_shape.clone();
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let mut output = vec![T::default(); out_size];

        // Compute strides
        let strides = compute_strides(shape);

        // Allocate gather buffer once and reuse for each output element
        // to avoid O(out_size) heap allocations.
        let mut gather_buf = vec![T::default(); axis_size];
        let mut in_coords = vec![0usize; shape.len()];

        for (out_idx, out) in output.iter_mut().enumerate() {
            let out_coords = linear_to_coords(out_idx, &coord_shape);

            // Build full input coordinates from output coordinates with axis slot
            for (i, &c) in out_coords.iter().enumerate() {
                let target = if i < axis { i } else { i + 1 };
                in_coords[target] = c;
            }

            // Gather values along the reduction axis
            for a in 0..axis_size {
                in_coords[axis] = a;
                let in_idx = coords_to_linear(&in_coords, &strides);
                gather_buf[a] = data[in_idx];
            }

            *out = f(&gather_buf);
        }

        ViewBuffer::from_vec_with_shape(output, out_shape)
    }

    fn reduce_axis_argmax<T>(&self, buffer: &ViewBuffer, axis: usize, is_max: bool) -> ViewBuffer
    where
        T: Copy + Default + PartialOrd + ViewType + 'static,
    {
        let shape = buffer.shape();
        let data = buffer.as_slice::<T>();

        let mut out_shape: Vec<usize> = shape.to_vec();
        let axis_size = out_shape.remove(axis);

        // Coordinate mapping must run over the REAL remaining dims (possibly
        // none, for a 1-D input); the [1] clamp below only pads the returned
        // buffer's shape.
        let coord_shape = out_shape.clone();
        if out_shape.is_empty() {
            out_shape.push(1);
        }

        let out_size: usize = out_shape.iter().product();
        let mut output = vec![0i64; out_size];

        let strides = compute_strides(shape);

        // Reusable coordinate buffer (avoids per-iteration clone + insert)
        let mut in_coords = vec![0usize; shape.len()];

        for (out_idx, out) in output.iter_mut().enumerate() {
            let out_coords = linear_to_coords(out_idx, &coord_shape);

            // Build full coordinates from output coords
            for (i, &c) in out_coords.iter().enumerate() {
                let target = if i < axis { i } else { i + 1 };
                in_coords[target] = c;
            }

            in_coords[axis] = 0;
            let first_in_idx = coords_to_linear(&in_coords, &strides);
            let mut best_val = data[first_in_idx];
            let mut best_idx = 0usize;

            for a in 1..axis_size {
                in_coords[axis] = a;
                let in_idx = coords_to_linear(&in_coords, &strides);
                let val = data[in_idx];

                let want = if is_max { Greater } else { Less };
                if displaces(&best_val, &val, want) {
                    best_val = val;
                    best_idx = a;
                }
            }

            *out = best_idx as i64;
        }

        ViewBuffer::from_vec_with_shape(output, out_shape)
    }
}

/// [`masked_select`] over a `T` image and an `M` mask, both read where they
/// lie (a view is not packed first): the mask once to size the output, then
/// both in step, a block of pixels at a time.
pub(crate) fn gather<T: ViewType, M: ViewType + PartialOrd + Default>(
    buffer: &ViewBuffer,
    mask: &ViewBuffer,
    c: usize,
) -> ViewBuffer {
    use crate::core::strided::Walk;
    use std::mem::MaybeUninit;
    const SCRATCH: usize = 4096;
    let pixels: usize = mask.shape().iter().product();
    let block = (SCRATCH / c.max(1)).max(1);
    let (image, mask) = (Walk::of(buffer), Walk::of(mask));
    let mut mask_scratch = [MaybeUninit::<u64>::uninit(); SCRATCH];
    let mut kept = 0;
    let mut cursor = mask.cursor();
    for start in (0..pixels).step_by(block) {
        let n = block.min(pixels - start);
        let set = cursor.next::<M>(n, &mut mask_scratch);
        kept += set.iter().filter(|&&m| m > M::default()).count();
    }
    let mut out: Vec<T> = Vec::with_capacity(kept * c);
    // A pixel wider than the scratch (thousands of channels) reads one at a
    // time from the heap.
    let (mut stack, mut heap) = ([MaybeUninit::<u64>::uninit(); SCRATCH], Vec::new());
    let image_scratch: &mut [MaybeUninit<u64>] = if c <= SCRATCH {
        &mut stack
    } else {
        heap.resize(c, MaybeUninit::uninit());
        &mut heap
    };
    let (mut values_at, mut set_at) = (image.cursor(), mask.cursor());
    for start in (0..pixels).step_by(block) {
        let n = block.min(pixels - start);
        let set = set_at.next::<M>(n, &mut mask_scratch);
        let values = values_at.next::<T>(n * c, image_scratch);
        for (pixel, _) in set.iter().enumerate().filter(|(_, &m)| m > M::default()) {
            out.extend_from_slice(&values[pixel * c..(pixel + 1) * c]);
        }
    }
    let len = out.len();
    ViewBuffer::from_vec_with_shape(out, vec![len])
}

/// `f` over each line of `buffer` along `axis`, as an f64 buffer of the
/// remaining axes (`[1]` for a 1-D input).
#[allow(clippy::needless_range_loop)]
fn reduce_axis_to_f64<T, F>(buffer: &ViewBuffer, axis: usize, f: F) -> ViewBuffer
where
    T: Copy + Default + num_traits::NumCast + ViewType + 'static,
    F: Fn(&[T]) -> f64,
{
    let shape = buffer.shape();
    let data = buffer.as_slice::<T>();

    let mut out_shape: Vec<usize> = shape.to_vec();
    let axis_size = out_shape.remove(axis);

    // Coordinate mapping must run over the REAL remaining dims (possibly
    // none, for a 1-D input); the [1] clamp below only pads the returned
    // buffer's shape.
    let coord_shape = out_shape.clone();
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_size: usize = out_shape.iter().product();
    let mut output = vec![0.0f64; out_size];

    let strides = compute_strides(shape);

    // Allocate gather buffer once and reuse
    let mut gather_buf = vec![T::default(); axis_size];
    let mut in_coords = vec![0usize; shape.len()];

    for (out_idx, out) in output.iter_mut().enumerate() {
        let out_coords = linear_to_coords(out_idx, &coord_shape);

        for (i, &c) in out_coords.iter().enumerate() {
            let target = if i < axis { i } else { i + 1 };
            in_coords[target] = c;
        }

        for a in 0..axis_size {
            in_coords[axis] = a;
            let in_idx = coords_to_linear(&in_coords, &strides);
            gather_buf[a] = data[in_idx];
        }

        *out = f(&gather_buf);
    }

    ViewBuffer::from_vec_with_shape(output, out_shape)
}

/// `reduce(reduction, axis)`: the dedicated op's kernel for the statistic
/// (the one implementation of each over a whole buffer), its result in f64.
/// Along an axis, a sum and a percentile are `reduce`'s own (their
/// dedicated ops are global), over the same per-value kernels. A count is
/// the line length, and no values reduce to what [`Reduction::of`] gives over
/// none, which the dedicated ops refuse.
pub(crate) fn reduce_by(buf: &ViewBuffer, reduction: Reduction, axis: Option<u32>) -> ViewBuffer {
    let shape = buf.shape();
    let line = axis.map_or(shape.iter().product(), |a| shape[a as usize]);
    let count = reduction == Reduction::Named(Statistic::Count);
    if line == 0 || count {
        // A count, or lines of no values: one value fills the output.
        let value = if count {
            line as f64
        } else {
            reduction.of(&mut [])
        };
        return match axis {
            None => ViewBuffer::from_scalar(value),
            Some(a) => {
                let mut out: Vec<usize> = shape.to_vec();
                out.remove(a as usize);
                if out.is_empty() {
                    out.push(1);
                }
                let n = out.iter().product();
                ViewBuffer::from_vec_with_shape(vec![value; n], out)
            }
        };
    }
    let q = match reduction {
        Reduction::Percentile(q) => Some(q),
        Reduction::Named(Statistic::Median) => Some(Percent::MEDIAN),
        Reduction::Named(_) => None,
    };
    let out = match (reduction, q, axis) {
        (_, Some(q), None) => ReductionOp::Percentile { q }.execute(buf),
        (_, Some(q), Some(a)) => crate::core::dtype::with_dtype!(buf.dtype(), T => {
            reduce_axis_to_f64::<T, _>(buf, a as usize, |slice: &[T]| {
                let mut values: Vec<f64> = slice
                    .iter()
                    .map(|&x| num_traits::NumCast::from(x).unwrap_or(f64::NAN))
                    .collect();
                percentile_of(&mut values, q.q())
            })
        }),
        (Reduction::Named(Statistic::Sum), _, None) => ReductionOp::Sum.execute(buf),
        (Reduction::Named(Statistic::Sum), _, Some(a)) => {
            crate::core::dtype::with_dtype!(buf.dtype(), T => {
                reduce_axis_to_f64::<T, _>(buf, a as usize, sum_f64::<T>)
            })
        }
        (Reduction::Named(Statistic::Mean), ..) => ReductionOp::Mean { axis }.execute(buf),
        (Reduction::Named(Statistic::Min), ..) => ReductionOp::Min { axis }.execute(buf),
        (Reduction::Named(Statistic::Max), ..) => ReductionOp::Max { axis }.execute(buf),
        (Reduction::Named(Statistic::Std), ..) => ReductionOp::Std { axis, ddof: 0 }.execute(buf),
        (Reduction::Named(Statistic::Median | Statistic::Count), ..)
        | (Reduction::Percentile(_), ..) => {
            unreachable!("a count is filled above, and a median or percentile has a q")
        }
    };
    out.cast_to(DType::F64)
}

/// The f64 sum of `values`: the one sum `reduce_sum` and `reduce("sum")`
/// read. A sum of nothing is +0, not `f64`'s `Sum` identity, -0.
fn sum_f64<T: Copy + num_traits::NumCast>(values: &[T]) -> f64 {
    0.0 + values
        .iter()
        .map(|&x| num_traits::NumCast::from(x).unwrap_or(0.0))
        .sum::<f64>()
}

/// `[H, W(, C)]` pooled in `bh` x `bw` blocks (partial at the far edges),
/// each block and channel reduced by `reduction`: `[ceil(H/bh), ceil(W/bw)(, C)]`.
pub(crate) fn pool_blocks(
    buf: &ViewBuffer,
    bh: usize,
    bw: usize,
    reduction: Reduction,
) -> ViewBuffer {
    let shape = buf.shape();
    let (h, w) = (shape[0], shape[1]);
    let c = shape.get(2).copied().unwrap_or(1);
    let (oh, ow) = (h.div_ceil(bh), w.div_ceil(bw));
    let values = buf.cast_to(DType::F64).to_contiguous();
    let values = values.as_slice::<f64>();
    let mut out = Vec::with_capacity(oh * ow * c);
    let mut block = Vec::with_capacity(bh * bw);
    for by in 0..oh {
        for bx in 0..ow {
            for ch in 0..c {
                block.clear();
                for y in by * bh..((by + 1) * bh).min(h) {
                    for x in bx * bw..((bx + 1) * bw).min(w) {
                        block.push(values[(y * w + x) * c + ch]);
                    }
                }
                out.push(reduction.of(&mut block));
            }
        }
    }
    let mut out_shape = vec![oh, ow];
    out_shape.extend_from_slice(&shape[2..]);
    ViewBuffer::from_vec_with_shape(out, out_shape)
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
