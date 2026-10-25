//! Reduction operations for statistical aggregations.
//!
//! This module provides operations that reduce array dimensions
//! by computing statistics like max, min, mean, std, and sum.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{DType, DTypeCategory, OutputDTypeRule, ViewType};
use crate::ops::shape_rule::OpShape;
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::ValidationError;
use crate::ops::Domain;

use crate::mode::{known, size, Exec, Mode};
use polars_cv_macros::{Ops, Resolve};

/// A reduction that is not a percentile, by name. A [`Reduction`] is one of
/// these or a percentile, spelled `"pNN"` beside them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Statistic {
    Sum,
    Mean,
    Min,
    Max,
    /// Population standard deviation (NumPy's default, `ddof=0`).
    Std,
    Median,
    /// How many values there are.
    Count,
}

crate::naming::named_variants!(Statistic: "A named reduction of a set of pixel values to one (``reduce``, ``label_reduce``, ``reduce_masked``, ``block_reduce``). A percentile is spelled beside these as ``\"pNN\"``: ``\"p90\"``, ``\"p62.5\"`` (its digits only; NumPy's linear interpolation). An op that chooses a reduction takes one of these; an op that is a percentile (``reduce_percentile``, ``rank_filter``, ``normalize_percentile``) takes ``q`` instead.\n\n- SUM, MEAN, MIN, MAX\n- STD: population standard deviation\n- MEDIAN\n- COUNT: how many pixels\n\nOver no pixels each is NaN, except SUM and COUNT (0)." {
    "sum" => Sum,
    "mean" => Mean,
    "min" => Min,
    "max" => Max,
    "std" => Std,
    "median" => Median,
    "count" => Count,
});

/// A percentile: `q` in `[0, 100]`. The one percentile type: a reduction's
/// `"pNN"`, `reduce_percentile(q)`, `rank_filter(q)` and
/// `normalize_percentile`'s edges all hold one, so each is checked by the
/// same rule as it is read (literal or per row), and none can skip it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Percent(f64);

// Exact: `new` admits no NaN, and stores no negative zero.
impl Eq for Percent {}

impl Percent {
    /// The median.
    pub const MEDIAN: Percent = Percent(50.0);

    /// `q`, when it is in `[0, 100]`.
    pub fn new(q: f64) -> Result<Self, String> {
        if !(0.0..=100.0).contains(&q) {
            return Err(format!("q = {q} is outside [0, 100]"));
        }
        // `-0.0` is in range and is 0.
        Ok(Percent(q + 0.0))
    }

    /// The percentile, in `[0, 100]`.
    pub fn q(self) -> f64 {
        self.0
    }
}

impl crate::naming::WireScalar for Percent {
    const KIND: crate::naming::WireKind = crate::naming::WireKind::Float;
    const PY_TYPE: &'static str = "float";
    // A rank in the value distribution, not a size.
    const TAKES_THIS: bool = false;
    fn from_wire(value: crate::naming::WireValue<'_>) -> Result<Self, String> {
        Percent::new(f64::from_wire(value)?)
    }
    fn to_wire(self) -> crate::naming::WireValue<'static> {
        self.0.to_wire()
    }
    fn spellings() -> Vec<&'static str> {
        Vec::new()
    }
}

/// A reduction of a set of values to one: the one definition every reducer
/// over a selection of pixels reads (`label_reduce`'s contour regions,
/// `reduce_masked`'s mask, `block_reduce`'s blocks, `statistics()`), so a
/// "median" means the same wherever it is asked for.
///
/// A percentile carries its `q`, so a reduction is one value: a per-row
/// column may mix `"p90"` and `"mean"`, and no `q` sits beside a reduction
/// that does not read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    Named(Statistic),
    /// The `q`-th percentile, linearly interpolated as NumPy's default.
    Percentile(Percent),
}

impl Reduction {
    /// `values` reduced (reordered in place). Over no values: NaN, except a
    /// sum and a count (0).
    pub fn of(self, values: &mut [f64]) -> f64 {
        let n = values.len();
        let stat = match self {
            Reduction::Percentile(p) => return percentile_of(values, p.q()),
            Reduction::Named(stat) => stat,
        };
        match stat {
            Statistic::Count => return n as f64,
            // Not `sum()` of nothing, which is -0 (`f64`'s `Sum` identity).
            Statistic::Sum if n == 0 => return 0.0,
            Statistic::Sum => return values.iter().sum(),
            _ if n == 0 => return f64::NAN,
            _ => {}
        }
        match stat {
            Statistic::Mean => values.iter().sum::<f64>() / n as f64,
            Statistic::Min => values.iter().copied().fold(values[0], minimum),
            Statistic::Max => values.iter().copied().fold(values[0], maximum),
            Statistic::Std => {
                let mean = values.iter().sum::<f64>() / n as f64;
                (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt()
            }
            Statistic::Median => percentile_of(values, 50.0),
            Statistic::Count | Statistic::Sum => unreachable!("answered above"),
        }
    }

    /// The reduction `s` spells: a [`Statistic`]'s name, or `"pNN"`.
    pub fn parse(s: &str) -> Result<Self, String> {
        if let Some(stat) = crate::naming::lookup(Statistic::NAMED, s) {
            return Ok(Reduction::Named(stat));
        }
        // A percentile's digits only: no sign, exponent, space or bare dot,
        // which `f64::from_str` would otherwise take.
        let digits = s.strip_prefix('p').filter(|q| {
            let mut parts = q.split('.');
            let whole = parts.next().unwrap_or("");
            let frac = parts.next();
            parts.next().is_none()
                && !whole.is_empty()
                && whole.bytes().all(|b| b.is_ascii_digit())
                && frac.is_none_or(|f| !f.is_empty() && f.bytes().all(|b| b.is_ascii_digit()))
        });
        if let Some(q) = digits.and_then(|q| q.parse::<f64>().ok()) {
            return Percent::new(q)
                .map(Reduction::Percentile)
                .map_err(|e| format!("reduction '{s}': {e}"));
        }
        Err(format!(
            "unknown reduction '{s}', expected one of {:?} or a percentile \
             spelled \"pNN\" (\"p90\", \"p62.5\")",
            crate::naming::names(Statistic::NAMED)
        ))
    }

    /// Its spelling: the statistic's name, or `"p62.5"`.
    pub fn spelling(self) -> std::borrow::Cow<'static, str> {
        match self {
            Reduction::Named(stat) => stat.spelling().into(),
            Reduction::Percentile(p) => format!("p{}", p.q()).into(),
        }
    }
}

impl crate::naming::WireScalar for Reduction {
    const KIND: crate::naming::WireKind = crate::naming::WireKind::Name;
    const PY_TYPE: &'static str = "str";
    const TAKES_THIS: bool = false;
    fn from_wire(value: crate::naming::WireValue<'_>) -> Result<Self, String> {
        match value.as_str() {
            Some(s) => Reduction::parse(s),
            None => Err(format!("a reduction is a name or \"pNN\", got {value:?}")),
        }
    }
    fn to_wire(self) -> crate::naming::WireValue<'static> {
        crate::naming::WireValue::Str(self.spelling())
    }
    /// The named ones; a percentile is spelled `"pNN"` beside them.
    fn spellings() -> Vec<&'static str> {
        crate::naming::names(Statistic::NAMED)
    }
}

/// The `q`-th percentile of `values` (reordered in place) by NumPy's default
/// linear interpolation; NaN if any value is NaN or there are none. The one
/// interpolation `reduce_percentile` and every [`Reduction`] read.
pub fn percentile_of(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() || q.is_nan() || values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let n = values.len();
    debug_assert!(
        (0.0..=100.0).contains(&q),
        "a percentile's q is checked where it is given, got {q}"
    );
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    // Selection, not a sort: the `lo`-th value in order, then the least of
    // those after it is the next (O(n), where a sort is O(n log n) — per
    // window in `rank_filter`).
    let (_, &mut low, above) = values.select_nth_unstable_by(lo, f64::total_cmp);
    match above.iter().copied().min_by(f64::total_cmp) {
        Some(hi) => low * (1.0 - frac) + hi * frac,
        None => low,
    }
}

/// Refuse what an op selecting pixels by a mask (`reduce_masked`,
/// `masked_select`) cannot read, over what is known: a mask that is not u8
/// or float, an image that is not `[H, W]`/`[H, W, C]`, a mask that is not
/// `[H, W]`/`[H, W, 1]` of the image's height and width. The planner names
/// the op an error comes from.
pub fn validate_masked_reduce(
    buffer: &[crate::ops::Dim],
    mask: &[crate::ops::Dim],
    mask_dtype: crate::PlannedDType,
) -> Result<(), ValidationError> {
    if let crate::PlannedDType::Known(dtype) = mask_dtype {
        if !matches!(dtype, DType::U8 | DType::F32 | DType::F64) {
            return Err(ValidationError::Generic {
                message: format!(
                    "a mask is u8 or float (a pixel counts where it is > 0), got {}",
                    dtype.short_name()
                ),
            });
        }
    }
    if !matches!(buffer.len(), 2 | 3) {
        return Err(ValidationError::ShapeRequirement {
            requirement: "an [H, W] or [H, W, C] image to select from under a mask",
            got: buffer.to_vec(),
        });
    }
    let plane = match mask {
        [h, w] => [*h, *w],
        [h, w, c] if c.known().is_none_or(|c| c == 1) => [*h, *w],
        _ => {
            return Err(ValidationError::ShapeRequirement {
                requirement: "an [H, W] or [H, W, 1] mask",
                got: mask.to_vec(),
            })
        }
    };
    for (axis, (m, b)) in plane.iter().zip(buffer).enumerate() {
        if let (Some(m), Some(b)) = (m.known(), b.known()) {
            if m != b {
                return Err(ValidationError::Generic {
                    message: format!(
                        "the mask's axis {axis} is {m}, the image's {b}: a mask covers \
                         the image it selects from"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// What [`reduce_masked`] produces: one value, or a buffer along an axis.
pub enum Masked {
    Scalar(f64),
    Along(ViewBuffer),
}

/// `buffer`'s values where `mask` is set (`> 0`) reduced by `reduction`:
/// all of them (`axis: None`, every channel together), or along `axis` 0 or
/// 1 — one value per column or row (per channel, for `[H, W, C]`), as an
/// axis reduction's shape. A line with no masked pixel is NaN (0 for `sum`
/// and `count`). Shapes and the mask's dtype are checked by
/// [`validate_masked_reduce`] first.
pub fn reduce_masked(
    buffer: &ViewBuffer,
    mask: &ViewBuffer,
    reduction: Reduction,
    axis: Option<usize>,
) -> Masked {
    // The selection's own reduction: `By` is `Reduction::of` over any dtype
    // (`a_whole_buffer_reduces_as_its_values_do`).
    let Some(axis) = axis else {
        let value = ReductionOp::By {
            reduction,
            axis: None,
        }
        .execute(&masked_select(buffer, mask));
        return Masked::Scalar(value.to_contiguous().as_slice::<f64>()[0]);
    };
    let shape = buffer.shape();
    let (h, w) = (shape[0], shape[1]);
    let c = shape.get(2).copied().unwrap_or(1);
    let values = buffer.cast_to(DType::F64).to_contiguous();
    let values = values.as_slice::<f64>();
    let mask = mask.cast_to(DType::F64).to_contiguous();
    let keep = mask.as_slice::<f64>();
    let at = |y: usize, x: usize, ch: usize| values[(y * w + x) * c + ch];
    let set = |y: usize, x: usize| keep[y * w + x] > 0.0;
    let mut selected = Vec::new();
    // Axis 0 runs down each column, axis 1 along each row.
    let (lines, len) = if axis == 0 { (w, h) } else { (h, w) };
    let mut out = Vec::with_capacity(lines * c);
    for line in 0..lines {
        for ch in 0..c {
            selected.clear();
            for i in 0..len {
                let (y, x) = if axis == 0 { (i, line) } else { (line, i) };
                if set(y, x) {
                    selected.push(at(y, x, ch));
                }
            }
            out.push(reduction.of(&mut selected));
        }
    }
    let out_shape = if shape.len() == 3 {
        vec![lines, c]
    } else {
        vec![lines]
    };
    Masked::Along(ViewBuffer::from_vec_with_shape(out, out_shape))
}

/// `buffer`'s values where `mask` is set (`> 0`), as a 1-D buffer of its
/// dtype: row-major, every channel of a selected pixel together (NumPy's
/// `image[mask > 0].ravel()`). Shapes and the mask's dtype are checked by
/// [`validate_masked_reduce`] first.
pub fn masked_select(buffer: &ViewBuffer, mask: &ViewBuffer) -> ViewBuffer {
    let c = buffer.shape().get(2).copied().unwrap_or(1);
    crate::core::dtype::with_dtype!(buffer.dtype(), T => match mask.dtype() {
        DType::U8 => gather::<T, u8>(buffer, mask, c),
        DType::F32 => gather::<T, f32>(buffer, mask, c),
        DType::F64 => gather::<T, f64>(buffer, mask, c),
        other => unreachable!(
            "validate_masked_reduce admits u8, f32 and f64 masks, got {}",
            other.short_name()
        ),
    })
}

/// `normalize_percentile` of `buffer` with its window from the pixels `mask`
/// selects ([`masked_select`]): the `low` and `high` percentiles of those
/// values mapped to [0, 1] across the whole buffer, clipped outside, as
/// `out_dtype`. A constant selection is the empty window's zeros, and no
/// selected pixel an unknown (NaN) window, as `normalize_percentile`'s over
/// the same values.
pub fn normalize_percentile_masked(
    buffer: &ViewBuffer,
    mask: &ViewBuffer,
    low: Percent,
    high: Percent,
    out_dtype: DType,
) -> ViewBuffer {
    let selected = masked_select(buffer, mask);
    let (low, high) = crate::kernels::elementwise::percentile_window(&selected, low, high);
    // The window applied as `normalize_range` applies a known one: the same
    // kernels `normalize_percentile` lowers to.
    let apply = crate::ops::compute::ComputeOp::NormalizeRange {
        low,
        high,
        out_dtype: Some(out_dtype),
    };
    crate::kernels::elementwise::apply(buffer.clone(), &apply)
}

/// [`masked_select`] over a `T` image and an `M` mask, both read where they
/// lie (a view is not packed first): the mask once to size the output, then
/// both in step, a block of pixels at a time.
fn gather<T: ViewType, M: ViewType + PartialOrd + Default>(
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

/// The reductions: one variant per wire op (see `crate::mode`).
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
pub enum ReductionOp<M: Mode = Exec> {
    /// Reduce buffer by computing the maximum value.
    #[op(name = "reduce_max", sample = {"axis": 0})]
    Max {
        /// Axis to reduce along. None for global reduction. It fixes the
        /// output rank, so it is literal-only.
        axis: Option<M::L<u32>>,
    },
    /// Reduce buffer by computing the minimum value.
    #[op(name = "reduce_min", sample = {"axis": 0})]
    Min {
        /// Axis to reduce along. None for global reduction. It fixes the
        /// output rank, so it is literal-only.
        axis: Option<M::L<u32>>,
    },
    /// Compute arithmetic mean.
    #[op(name = "reduce_mean", sample = {"axis": 1})]
    Mean {
        /// Axis to reduce along. None for global reduction. It fixes the
        /// output rank, so it is literal-only.
        axis: Option<M::L<u32>>,
    },
    /// Reduce buffer by computing the standard deviation.
    ///
    /// Example:
    ///     >>> pipe = Pipeline().source("image_bytes").reduce_std(ddof=1)
    #[op(name = "reduce_std", sample = {"axis": 0, "ddof": 1})]
    Std {
        /// Axis to reduce along. None for global reduction.
        axis: Option<M::L<u32>>,
        /// Delta degrees of freedom. 0 for population std (default), 1 for sample
        /// std. Accepts a Polars expression for per-row dynamic values.
        #[param(default = 0)]
        ddof: M::V<u8>,
    },
    /// Sum all elements in the buffer.
    #[op(name = "reduce_sum", sample = {})]
    Sum,
    /// Index of the maximum value along an axis.
    ///
    /// Unlike other reductions it always requires an axis: a global index
    /// is ambiguous for a multi-dimensional array. The result has the reduced
    /// shape and an i64 dtype.
    #[op(name = "reduce_argmax", sample = {"axis": 0})]
    ArgMax {
        /// Axis along which to find the index.
        axis: M::L<u32>,
    },
    /// Index of the minimum value along an axis.
    ///
    /// Unlike other reductions it always requires an axis: a global index
    /// is ambiguous for a multi-dimensional array. The result has the reduced
    /// shape and an i64 dtype.
    #[op(name = "reduce_argmin", sample = {"axis": 0})]
    ArgMin {
        /// Axis along which to find the index.
        axis: M::L<u32>,
    },
    /// Count set bits (1s) in the buffer.
    #[op(name = "reduce_popcount", sample = {})]
    PopCount,
    /// Pool fixed blocks: each ``height`` x ``width`` block reduced to one
    /// value per channel, so ``[H, W, C]`` becomes
    /// ``[ceil(H / height), ceil(W / width), C]``. The last row and column of
    /// blocks are partial when the size does not divide, and reduce the
    /// pixels they hold.
    ///
    /// Example:
    ///     >>> Pipeline().source("array").block_reduce(height=20, width=20, reduction="mean")
    #[op(name = "block_reduce", sample = {"height": 2, "width": 3, "reduction": "max"})]
    Blocks {
        /// Block height in pixels. Accepts a Polars expression.
        height: M::V<u32>,
        /// Block width in pixels. Accepts a Polars expression.
        width: M::V<u32>,
        /// The reduction: "sum", "mean" (default), "min", "max", "std",
        /// "median", "count", or a percentile "pNN" ("p90", "p62.5").
        #[param(default = "mean")]
        reduction: M::V<Reduction>,
    },
    /// Reduce every value (every channel) to one, or along one axis, by any
    /// reduction: the statistic named, or a percentile "pNN". Each is
    /// computed by its dedicated op's kernel (``reduce("mean")`` is
    /// ``reduce_mean()``'s value, in f64). Over no values: NaN, or 0 for
    /// "sum" and "count".
    ///
    /// Example:
    ///     >>> Pipeline().source("image_bytes").reduce("p90")
    #[op(name = "reduce", sample = {"reduction": "p90", "axis": 0})]
    By {
        /// "sum", "mean", "min", "max", "std" (population), "median",
        /// "count", or a percentile "pNN" ("p90", "p62.5"). Accepts a Polars
        /// expression for a per-row choice.
        reduction: M::V<Reduction>,
        /// Axis to reduce along. None for global reduction. It fixes the
        /// output rank, so it is literal-only.
        axis: Option<M::L<u32>>,
    },
    /// Compute the q-th percentile of all values (linear interpolation, as
    /// numpy's default). Along an axis, ``reduce("pNN", axis=)``.
    #[op(name = "reduce_percentile", sample = {"q": 50.0})]
    Percentile {
        /// Percentile to compute, in [0, 100]. Accepts a Polars expression for
        /// per-row dynamic values.
        q: M::V<Percent>,
    },
}

impl<M: Mode> ReductionOp<M> {
    /// A block of no pixels is refused (every percentile is a [`Percent`],
    /// checked as it is read).
    pub fn check(&self) -> Result<(), String> {
        match self {
            ReductionOp::Blocks { height, width, .. } => {
                for (name, side) in [("height", height), ("width", width)] {
                    if known::<M, u32>(side) == Some(0) {
                        return Err(format!("block_reduce: a block {name} of 0 pixels"));
                    }
                }
                Ok(())
            }
            ReductionOp::Max { .. }
            | ReductionOp::Percentile { .. }
            | ReductionOp::Min { .. }
            | ReductionOp::Mean { .. }
            | ReductionOp::Sum
            | ReductionOp::Std { .. }
            | ReductionOp::ArgMax { .. }
            | ReductionOp::ArgMin { .. }
            | ReductionOp::PopCount
            | ReductionOp::By { .. } => Ok(()),
        }
    }

    /// The axis reduced, `None` for a global reduction.
    pub fn axis(&self) -> Option<usize> {
        match self {
            ReductionOp::Max { axis }
            | ReductionOp::Min { axis }
            | ReductionOp::Mean { axis }
            | ReductionOp::By { axis, .. }
            | ReductionOp::Std { axis, .. } => axis.as_ref().map(|a| M::lit(a) as usize),
            ReductionOp::ArgMax { axis } | ReductionOp::ArgMin { axis } => {
                Some(M::lit(axis) as usize)
            }
            ReductionOp::Sum
            | ReductionOp::PopCount
            | ReductionOp::Percentile { .. }
            | ReductionOp::Blocks { .. } => None,
        }
    }

    /// How this op's output shape follows from its input: the axis removed,
    /// or a single slot for a global reduction.
    pub fn shape(&self) -> OpShape {
        match self {
            ReductionOp::Blocks { height, width, .. } => OpShape::Blocks {
                h: size::<M>(height),
                w: size::<M>(width),
            },
            _ => OpShape::Reduce { axis: self.axis() },
        }
    }

    /// The domain this reduction produces: a scalar for a global one, a
    /// smaller buffer for an axis one or a pooling.
    pub fn output_domain(&self) -> Domain {
        if self.axis().is_some() || matches!(self, ReductionOp::Blocks { .. }) {
            Domain::Buffer
        } else {
            Domain::Scalar
        }
    }
}

impl ReductionOp {
    /// Execute the reduction on a buffer.
    pub fn execute(&self, buffer: &ViewBuffer) -> ViewBuffer {
        let contig = buffer.to_contiguous();

        if let ReductionOp::Blocks {
            height,
            width,
            reduction,
        } = self
        {
            return pool_blocks(&contig, *height as usize, *width as usize, *reduction);
        }

        if let ReductionOp::By { reduction, axis } = self {
            return reduce_by(&contig, *reduction, *axis);
        }

        // PopCount has special handling for each type
        if matches!(self, ReductionOp::PopCount) {
            return self.execute_popcount(&contig);
        }

        // Percentile has special handling (needs selection)
        if let ReductionOp::Percentile { q } = self {
            return self.execute_percentile(&contig, q.q());
        }

        match buffer.dtype() {
            DType::U8 => self.execute_typed::<u8>(&contig),
            DType::I8 => self.execute_typed::<i8>(&contig),
            DType::U16 => self.execute_typed::<u16>(&contig),
            DType::I16 => self.execute_typed::<i16>(&contig),
            DType::U32 => self.execute_typed::<u32>(&contig),
            DType::I32 => self.execute_typed::<i32>(&contig),
            DType::U64 => self.execute_typed::<u64>(&contig),
            DType::I64 => self.execute_typed::<i64>(&contig),
            DType::F32 => self.execute_typed::<f32>(&contig),
            DType::F64 => self.execute_typed::<f64>(&contig),
        }
    }

    /// Execute popcount reduction - count all set bits in the buffer.
    fn execute_popcount(&self, buffer: &ViewBuffer) -> ViewBuffer {
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
    fn execute_percentile(&self, buffer: &ViewBuffer, q: f64) -> ViewBuffer {
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

    fn execute_typed<T>(&self, buffer: &ViewBuffer) -> ViewBuffer
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

impl<M: Mode> Op for ReductionOp<M> {
    fn name(&self) -> &'static str {
        match self {
            ReductionOp::Max { .. } => "Max",
            ReductionOp::Min { .. } => "Min",
            ReductionOp::Mean { .. } => "Mean",
            ReductionOp::Std { .. } => "Std",
            ReductionOp::Sum => "Sum",
            ReductionOp::ArgMax { .. } => "ArgMax",
            ReductionOp::ArgMin { .. } => "ArgMin",
            ReductionOp::PopCount => "PopCount",
            ReductionOp::Blocks { .. } => "BlockReduce",
            ReductionOp::Percentile { .. } => "Percentile",
            ReductionOp::By { .. } => "Reduce",
        }
    }

    fn shape(&self) -> OpShape {
        ReductionOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        MemoryEffect::RequiresContiguous
    }

    fn identity_rule(&self) -> IdentityRule {
        // Computes / combines / reduces — never a removable no-op.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        false // A reduction is not an H/W crop window.
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // A reduction aggregates over the whole input (or a whole axis), so no
        // output pixel maps to a bounded input window: a hard reorder barrier.
        // Matched exhaustively (not a blanket) so a new variant must reconfirm
        // this rather than silently inherit Global.
        match self {
            ReductionOp::Max { .. }
            | ReductionOp::Min { .. }
            | ReductionOp::Mean { .. }
            | ReductionOp::Std { .. }
            | ReductionOp::Sum
            | ReductionOp::ArgMax { .. }
            | ReductionOp::ArgMin { .. }
            | ReductionOp::PopCount
            | ReductionOp::Blocks { .. }
            | ReductionOp::Percentile { .. }
            | ReductionOp::By { .. } => SpatialDependency::Global,
        }
    }

    fn infer_strides(
        &self,
        _input_shape: &[usize],
        _input_strides: &[isize],
    ) -> Option<Vec<isize>> {
        None
    }

    fn validate(
        &self,
        input_shapes: &[&[crate::ops::Dim]],
        _input_dtypes: &[crate::PlannedDType],
    ) -> Result<(), ValidationError> {
        let shape = input_shapes[0];
        // Blocks tile the image plane: `[H, W]` or `[H, W, C]` only.
        if matches!(self, ReductionOp::Blocks { .. }) {
            return crate::ops::validation::require_hw_or_hwc(shape);
        }
        if let Some(ax) = self.axis() {
            if ax >= shape.len() {
                return Err(ValidationError::InvalidAxis {
                    axis: ax,
                    ndim: shape.len(),
                });
            }
        }
        // An ordering reduction over nothing has no answer (numpy raises);
        // sum, mean and std are defined over nothing (0, NaN, NaN).
        let orders = matches!(
            self,
            ReductionOp::Max { .. }
                | ReductionOp::Min { .. }
                | ReductionOp::ArgMax { .. }
                | ReductionOp::ArgMin { .. }
                | ReductionOp::Percentile { .. }
        );
        let reduced: &[crate::ops::Dim] = match self.axis() {
            Some(ax) => &shape[ax..=ax],
            None => shape,
        };
        if orders && reduced.iter().any(|d| d.known() == Some(0)) {
            return Err(ValidationError::ShapeRequirement {
                requirement: "a non-empty extent to order (an empty one has no extreme)",
                got: shape.to_vec(),
            });
        }
        Ok(())
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Numeric
    }

    fn working_dtype(&self) -> Option<DType> {
        None
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        match self {
            ReductionOp::Mean { .. }
            | ReductionOp::Std { .. }
            | ReductionOp::Sum
            | ReductionOp::PopCount
            | ReductionOp::Blocks { .. }
            | ReductionOp::Percentile { .. }
            | ReductionOp::By { .. } => OutputDTypeRule::ForceF64,
            ReductionOp::ArgMax { .. } | ReductionOp::ArgMin { .. } => OutputDTypeRule::ForceI64,
            _ => OutputDTypeRule::PreserveInput,
        }
    }
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
fn reduce_by(buf: &ViewBuffer, reduction: Reduction, axis: Option<u32>) -> ViewBuffer {
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
fn pool_blocks(buf: &ViewBuffer, bh: usize, bw: usize, reduction: Reduction) -> ViewBuffer {
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

use super::util::{coords_to_linear, linear_to_coords};
use super::util::{displaces, maximum, minimum};
use std::cmp::Ordering::{Greater, Less};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_max() {
        let data = vec![1u8, 5, 3, 9, 2];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![5]);
        let op = ReductionOp::Max { axis: None };
        let result = op.execute(&buffer);
        assert_eq!(result.as_slice::<u8>()[0], 9);
    }

    #[test]
    fn test_axis_reduction_of_1d_input() {
        // Regression: reducing a 1-D buffer along axis 0 panicked — the [1]
        // shape clamp fed the coordinate mapping a phantom dimension. The
        // result is a rank-1 single-slot buffer, per ReduceByOne's clamp.
        let buffer = ViewBuffer::from_vec_with_shape(vec![1.0f32, 5.0, 3.0], vec![3]);
        for (op, expected) in [
            (ReductionOp::Max { axis: Some(0) }, 5.0f32),
            (ReductionOp::Min { axis: Some(0) }, 1.0),
        ] {
            let result = op.execute(&buffer);
            assert_eq!(result.shape(), &[1]);
            assert_eq!(result.as_slice::<f32>(), &[expected]);
        }

        // Mean computes in f64 along an axis.
        let result = ReductionOp::Mean { axis: Some(0) }.execute(&buffer);
        assert_eq!(result.shape(), &[1]);
        assert!((result.as_slice::<f64>()[0] - 3.0).abs() < 1e-10);

        let argmax = ReductionOp::ArgMax { axis: 0 }.execute(&buffer);
        assert_eq!(argmax.shape(), &[1]);
        assert_eq!(argmax.as_slice::<i64>(), &[1]);
    }

    /// An ordering reduction over nothing has no answer: numpy raises
    /// ("zero-size array to reduction operation maximum which has no
    /// identity", "attempt to get argmax of an empty sequence"). The
    /// contract refuses it; `argmax` over an empty axis panicked the engine.
    /// `sum` (0), `mean` and `std` (NaN) are defined over nothing.
    #[test]
    fn an_ordering_reduction_over_nothing_is_refused() {
        use crate::ops::{Dim, Op};
        use crate::PlannedDType;
        let f32_ = [PlannedDType::Known(DType::F32)];
        let check = |op: &ReductionOp, shape: &[usize]| {
            let dims: Vec<Dim> = shape.iter().map(|&n| Dim::Known(n)).collect();
            op.validate(&[&dims], &f32_)
        };
        for shape in [[0usize, 3, 1], [3, 0, 1]] {
            let empty_axis = if shape[0] == 0 { 0 } else { 1 };
            for op in [
                ReductionOp::ArgMax { axis: empty_axis },
                ReductionOp::ArgMin { axis: empty_axis },
                ReductionOp::Max {
                    axis: Some(empty_axis),
                },
                ReductionOp::Min {
                    axis: Some(empty_axis),
                },
                ReductionOp::Max { axis: None },
                ReductionOp::Min { axis: None },
                ReductionOp::Percentile { q: Percent::MEDIAN },
            ] {
                let err = check(&op, &shape).expect_err(&format!("{op:?} on {shape:?}"));
                assert!(err.to_string().contains("empty"), "{op:?}: {err}");
            }
            // Reducing a non-empty axis of an empty image is fine (empty out).
            let other = 2;
            assert!(check(&ReductionOp::ArgMax { axis: other }, &shape).is_ok());
            for op in [
                ReductionOp::Sum,
                ReductionOp::Mean { axis: None },
                ReductionOp::Std {
                    axis: None,
                    ddof: 0,
                },
            ] {
                assert!(check(&op, &shape).is_ok(), "{op:?}");
            }
        }
    }

    /// A NaN is the result of every ordering reduction, as in numpy: `max`
    /// and `min` propagate it, `argmax`/`argmin` give its first index, and
    /// `percentile` is NaN. Percentile used to sort with a comparator that is
    /// not a total order under NaN, which panics; max/min/arg* compared with
    /// `>`/`<`, so whether a NaN survived depended on where it sat.
    #[test]
    fn a_nan_is_the_result_of_every_ordering_reduction() {
        for data in [
            vec![f32::NAN, 1.0, 5.0, 3.0],
            vec![1.0, f32::NAN, 5.0, 3.0],
            vec![1.0, 5.0, 3.0, f32::NAN],
        ] {
            let nan_at = data.iter().position(|x| x.is_nan()).unwrap();
            let buffer = ViewBuffer::from_vec_with_shape(data.clone(), vec![4]);
            for op in [
                ReductionOp::Max { axis: None },
                ReductionOp::Min { axis: None },
                ReductionOp::Max { axis: Some(0) },
                ReductionOp::Min { axis: Some(0) },
            ] {
                let out = op.execute(&buffer);
                assert!(out.as_slice::<f32>()[0].is_nan(), "{op:?} of {data:?}");
            }
            for q in [0.0, 50.0, 100.0] {
                let q = Percent::new(q).unwrap();
                let out = ReductionOp::Percentile { q }.execute(&buffer);
                assert!(out.as_slice::<f64>()[0].is_nan(), "{q:?} of {data:?}");
            }
            for op in [
                ReductionOp::ArgMax { axis: 0 },
                ReductionOp::ArgMin { axis: 0 },
            ] {
                let out = op.execute(&buffer);
                assert_eq!(
                    out.as_slice::<i64>(),
                    &[nan_at as i64],
                    "{op:?} of {data:?}"
                );
            }
        }
        // Many NaNs, enough for the sort to notice an inconsistent order.
        let data: Vec<f64> = (0..64)
            .map(|i| if i % 3 == 0 { f64::NAN } else { i as f64 })
            .collect();
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![64]);
        let out = ReductionOp::Percentile {
            q: Percent::new(25.0).unwrap(),
        }
        .execute(&buffer);
        assert!(out.as_slice::<f64>()[0].is_nan());
    }

    #[test]
    fn test_global_mean() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![5]);
        let op = ReductionOp::Mean { axis: None };
        let result = op.execute(&buffer);
        assert!((result.as_slice::<f64>()[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_popcount_u8() {
        // 0xFF = 8 bits, 0x00 = 0 bits, 0x0F = 4 bits, 0xAA = 4 bits (10101010)
        let data = vec![0xFFu8, 0x00, 0x0F, 0xAA];
        let buffer = ViewBuffer::from_vec_with_shape(data, vec![4]);
        let op = ReductionOp::PopCount;
        let result = op.execute(&buffer);
        // 8 + 0 + 4 + 4 = 16 bits
        assert!((result.as_slice::<f64>()[0] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_popcount_for_hamming_distance() {
        // Simulate XOR of two hashes and count bits
        // hash1 = [0xFF, 0x00] (11111111 00000000)
        // hash2 = [0x0F, 0x0F] (00001111 00001111)
        // XOR   = [0xF0, 0x0F] (11110000 00001111)
        // popcount = 4 + 4 = 8 bits different
        let xor_result = vec![0xF0u8, 0x0F];
        let buffer = ViewBuffer::from_vec_with_shape(xor_result, vec![2]);
        let op = ReductionOp::PopCount;
        let result = op.execute(&buffer);
        assert!((result.as_slice::<f64>()[0] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_popcount_identical_hashes() {
        // XOR of identical hashes = all zeros = 0 bits
        let xor_result = vec![0x00u8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let buffer = ViewBuffer::from_vec_with_shape(xor_result, vec![8]);
        let op = ReductionOp::PopCount;
        let result = op.execute(&buffer);
        assert!((result.as_slice::<f64>()[0] - 0.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod reduction_value_tests {
    use super::*;
    use crate::naming::WireScalar;

    #[test]
    fn a_reduction_is_a_name_or_a_percentile_and_spells_itself_back() {
        // Any decimals: a percentile is the one `Percent`, as `q` is.
        for spelling in [
            "mean", "count", "p90", "p62.5", "p0", "p100", "p33.33", "p9.999",
        ] {
            let r = Reduction::parse(spelling).unwrap();
            assert_eq!(r.spelling(), spelling);
            assert_eq!(Reduction::from_wire(r.to_wire()).unwrap(), r);
        }
        assert_eq!(
            Reduction::parse("p62.5").unwrap(),
            Reduction::Percentile(Percent::new(62.5).unwrap())
        );
        // `"p50.0"` is p50, spelled back the shortest way.
        assert_eq!(Reduction::parse("p50.0").unwrap().spelling(), "p50");
    }

    #[test]
    fn a_malformed_percentile_is_refused_saying_why() {
        for (spelling, says) in [
            ("percentile", "unknown reduction 'percentile'"),
            ("p150", "outside [0, 100]"),
            ("p-1", "unknown reduction 'p-1'"),
            // The digits of a percentile only.
            ("pnan", "unknown reduction 'pnan'"),
            ("p+50", "unknown reduction 'p+50'"),
            ("p5e1", "unknown reduction 'p5e1'"),
            ("p50.", "unknown reduction 'p50.'"),
            ("p.5", "unknown reduction 'p.5'"),
            ("p", "unknown reduction 'p'"),
        ] {
            let err = Reduction::parse(spelling).unwrap_err();
            assert!(err.contains(says), "{spelling}: {err}");
        }
    }

    #[test]
    fn a_percentile_reduces_as_numpy_does() {
        let mut values = vec![4.0, 1.0, 3.0, 2.0];
        let p = Reduction::parse("p62.5").unwrap();
        // NumPy: np.percentile([1, 2, 3, 4], 62.5) == 2.875
        assert_eq!(p.of(&mut values), 2.875);
        assert!(p.of(&mut []).is_nan());
    }
}

#[cfg(test)]
mod one_implementation_tests {
    use super::*;

    /// Every reduction of a whole buffer (`reduce`, through the dedicated
    /// kernels) is [`Reduction::of`] over the same values (the masks',
    /// regions' and blocks' reduction): globally and per line, on integers,
    /// floats with a NaN, and no values. Swept over every `Statistic`, so a
    /// new one is held to it.
    #[test]
    fn a_whole_buffer_reduces_as_its_values_do() {
        let mut reductions: Vec<Reduction> = Statistic::NAMED
            .iter()
            .map(|&(_, s)| Reduction::Named(s))
            .collect();
        for q in [0.0, 37.5, 90.0, 100.0] {
            reductions.push(Reduction::Percentile(Percent::new(q).unwrap()));
        }
        let ints: Vec<i32> = vec![5, -3, 7, 7, 0, 12, -8, 1, 4, 4, 9, -2];
        let mut floats: Vec<f32> = ints.iter().map(|&v| v as f32 * 0.37).collect();
        floats[5] = f32::NAN;
        let buffers = [
            ViewBuffer::from_vec_with_shape(
                ints.iter().map(|&v| (v + 8) as u8).collect(),
                vec![3, 4],
            ),
            ViewBuffer::from_vec_with_shape(ints.clone(), vec![3, 4]),
            ViewBuffer::from_vec_with_shape(
                ints.iter().map(|&v| v as f32 * 0.37).collect(),
                vec![3, 4],
            ),
            ViewBuffer::from_vec_with_shape(floats, vec![3, 4]),
            ViewBuffer::from_vec_with_shape(Vec::<f32>::new(), vec![0, 4]),
        ];
        for buf in &buffers {
            let (h, w) = (buf.shape()[0], buf.shape()[1]);
            let values: Vec<f64> = buf
                .cast_to(DType::F64)
                .to_contiguous()
                .as_slice::<f64>()
                .to_vec();
            for &reduction in &reductions {
                for axis in [None, Some(0u32), Some(1)] {
                    let got = ReductionOp::By { reduction, axis }.execute(buf);
                    let got = got.to_contiguous();
                    let got = got.as_slice::<f64>();
                    let lines: Vec<Vec<f64>> = match axis {
                        None => vec![values.clone()],
                        Some(0) => (0..w)
                            .map(|x| (0..h).map(|y| values[y * w + x]).collect())
                            .collect(),
                        Some(_) => (0..h)
                            .map(|y| values[y * w..(y + 1) * w].to_vec())
                            .collect(),
                    };
                    let want: Vec<f64> = lines
                        .into_iter()
                        .map(|mut l| reduction.of(&mut l))
                        .collect();
                    assert_eq!(
                        got.len(),
                        want.len(),
                        "{reduction:?} {axis:?} {:?}",
                        buf.shape()
                    );
                    for (g, w) in got.iter().zip(&want) {
                        let same = (g.is_nan() && w.is_nan())
                            || g == w
                            || (g - w).abs() <= 1e-12 * w.abs().max(1.0);
                        assert!(
                            same,
                            "{reduction:?} {axis:?} {:?} {}: {g} vs {w}",
                            buf.shape(),
                            buf.dtype().short_name()
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod small_fix_tests {
    use super::*;

    /// A sum of nothing is +0, as NumPy's: Rust's `f64` `Sum` starts from
    /// -0, which showed through as `-0.0` for an empty region or line.
    #[test]
    fn a_sum_of_nothing_is_positive_zero() {
        let sum = Reduction::Named(Statistic::Sum);
        assert!(sum.of(&mut []).to_bits() == 0.0f64.to_bits());
        let empty = ViewBuffer::from_vec_with_shape(Vec::<f32>::new(), vec![0, 4]);
        for op in [
            ReductionOp::Sum,
            ReductionOp::By {
                reduction: sum,
                axis: None,
            },
        ] {
            let out = op.execute(&empty);
            let got = out.to_contiguous().as_slice::<f64>()[0];
            assert_eq!(got.to_bits(), 0.0f64.to_bits(), "{op:?}: {got}");
        }
        let lines = ReductionOp::By {
            reduction: sum,
            axis: Some(0),
        }
        .execute(&empty);
        for got in lines.to_contiguous().as_slice::<f64>() {
            assert_eq!(got.to_bits(), 0.0f64.to_bits(), "an empty column: {got}");
        }
    }

    /// The percentile by selection is the sorted reference's, exactly.
    #[test]
    fn a_percentile_by_selection_is_the_sorted_ones() {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for n in [1usize, 2, 3, 7, 64, 101] {
            for _ in 0..20 {
                let values: Vec<f64> = (0..n).map(|_| (next() % 50) as f64 - 20.0).collect();
                let mut sorted = values.clone();
                sorted.sort_by(f64::total_cmp);
                for q in [0.0, 12.5, 33.333, 50.0, 90.0, 99.9, 100.0] {
                    let pos = q / 100.0 * (n - 1) as f64;
                    let lo = pos.floor() as usize;
                    let frac = pos - lo as f64;
                    let want = match sorted.get(lo + 1) {
                        Some(&hi) => sorted[lo] * (1.0 - frac) + hi * frac,
                        None => sorted[lo],
                    };
                    let got = percentile_of(&mut values.clone(), q);
                    assert_eq!(got.to_bits(), want.to_bits(), "n={n} q={q}");
                }
            }
        }
    }
}

#[cfg(test)]
mod masked_select_tests {
    use super::*;

    /// NumPy's `image[mask > 0].ravel()`: row-major, a pixel's channels
    /// together, in the image's dtype, whatever the mask's dtype or layout.
    #[test]
    fn masked_select_is_a_boolean_index() {
        let rgb: Vec<u16> = (0..2 * 3 * 2).map(|v| v * 10).collect();
        let image = ViewBuffer::from_vec_with_shape(rgb, vec![2, 3, 2]);
        let keep = [true, false, true, false, false, true];
        let want: Vec<u16> = vec![0, 10, 40, 50, 100, 110];
        let u8_mask: Vec<u8> = keep.iter().map(|&k| if k { 255 } else { 0 }).collect();
        let f32_mask: Vec<f32> = keep.iter().map(|&k| if k { 0.5 } else { 0.0 }).collect();
        let masks = [
            ViewBuffer::from_vec_with_shape(u8_mask.clone(), vec![2, 3]),
            ViewBuffer::from_vec_with_shape(u8_mask.clone(), vec![2, 3, 1]),
            ViewBuffer::from_vec_with_shape(f32_mask, vec![2, 3]),
            // A strided view: the transpose of a [3, 2] mask.
            ViewBuffer::from_vec_with_shape(
                vec![
                    u8_mask[0], u8_mask[3], u8_mask[1], u8_mask[4], u8_mask[2], u8_mask[5],
                ],
                vec![3, 2],
            )
            .permute(&[1, 0]),
        ];
        for mask in &masks {
            let got = masked_select(&image, mask);
            assert_eq!(got.dtype(), DType::U16);
            assert_eq!(got.shape(), &[want.len()]);
            assert_eq!(got.to_contiguous().as_slice::<u16>(), want.as_slice());
        }
        let none = ViewBuffer::from_vec_with_shape(vec![0u8; 6], vec![2, 3]);
        assert_eq!(masked_select(&image, &none).shape(), &[0]);
    }

    /// Past one block of pixels, and through views of both operands, the
    /// gather is the boolean index of their packed copies.
    #[test]
    fn masked_select_reads_views_block_by_block() {
        let (h, w, c) = (70, 90, 3);
        let image: Vec<u8> = (0..h * w * c).map(|v| (v * 7 % 251) as u8).collect();
        let image = ViewBuffer::from_vec_with_shape(image, vec![h, w, c]);
        let mask: Vec<f32> = (0..h * w).map(|v| ((v * 13) % 5) as f32 - 2.0).collect();
        let mask = ViewBuffer::from_vec_with_shape(mask, vec![h, w]);
        let flipped = |b: &ViewBuffer| b.flip(&[0]).flip(&[1]);
        for (image, mask) in [
            (image.clone(), mask.clone()),
            (flipped(&image), flipped(&mask)),
        ] {
            let (packed, keep) = (image.to_contiguous(), mask.to_contiguous());
            let want: Vec<u8> = packed
                .as_slice::<u8>()
                .chunks(c)
                .zip(keep.as_slice::<f32>())
                .filter(|(_, &m)| m > 0.0)
                .flat_map(|(px, _)| px.to_vec())
                .collect();
            let got = masked_select(&image, &mask);
            assert_eq!(got.to_contiguous().as_slice::<u8>(), want.as_slice());
        }
    }
}
