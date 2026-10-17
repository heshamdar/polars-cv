//! `apply_lut`: each u8/u16 value read as an index into a table (CR-95).
//!
//! Its own family rather than an engine `ViewDto` op because a value past the
//! table is a *data* error, which an engine op cannot report: it surfaces as
//! the row's `Err` from [`LutOp::execute`], as the histogram's does. A table
//! of single entries reads through the element-wise engine's table map
//! (`ops::elementwise::lookup_column`), the one per-value table read; a
//! palette, one input to `K` outputs, writes its pixels in one pass.

use polars_cv_macros::{Ops, Resolve};

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::{with_dtype, DType, DTypeCategory, OutputDTypeRule};
use crate::mode::{known, Exec, Mode};
use crate::ops::shape_rule::OpShape;
use crate::ops::spatial_rule::SpatialDependency;
use crate::ops::traits::{IdentityRule, MemoryEffect, Op};
use crate::ops::validation::{require_single_channel, ValidationError};
use crate::ops::Domain;
use crate::PlannedDType;

/// The most rows a table can have: one per u16 value.
const MAX_ROWS: usize = 1 << 16;

/// Map each value through a lookup table.
///
/// Each u8 or u16 value is a row index into ``table``. Rows of one entry map
/// every element in place, on any channel count (as ``cv2.LUT`` with one
/// table); rows of ``K > 1`` entries take a single-channel image to ``K``
/// channels (a palette). The entries are stored as ``out_dtype``. A value with no
/// row is an error for its row: clamp first, keeping the dtype
/// (``clamp(0, len(table) - 1, out_dtype="u8")``), to send the excess to the
/// last row.
///
/// Example:
///     >>> heat = Pipeline().source("image_bytes").grayscale().apply_lut([[v, 0, 255 - v] for v in range(256)])
#[derive(Debug, Clone, PartialEq, Ops, Resolve)]
#[op(name = "apply_lut", sample = {"table": [[0, 0, 0], [255, 128, 0]], "out_dtype": "u8"})]
pub struct LutOp<M: Mode = Exec> {
    /// The rows, indexed by value: 1 to 65536 of them, each of the same
    /// length (1, or the palette's channel count). Each entry must fit
    /// ``out_dtype``, and may be a Polars expression; the lengths are literal.
    pub table: Vec<Vec<M::V<f64>>>,
    /// The output element dtype the entries are stored as.
    #[param(default = "u8")]
    pub out_dtype: M::L<DType>,
}

impl<M: Mode> LutOp<M> {
    /// The palette width: the length every row shares.
    fn width(&self) -> usize {
        self.table.first().map_or(1, Vec::len)
    }

    /// Refuse a table no row can use: none or too many rows, rows of
    /// different (or no) lengths, or a known entry that does not fit
    /// `out_dtype`. A per-row entry is checked when its row resolves.
    pub fn check(&self) -> Result<(), String> {
        let n = self.table.len();
        if n == 0 || n > MAX_ROWS {
            return Err(format!(
                "apply_lut: the table needs at least one row and at most {MAX_ROWS} \
                 (one per u16 value), got {n}"
            ));
        }
        let k = self.width();
        if k == 0 || self.table.iter().any(|row| row.len() != k) {
            return Err(
                "apply_lut: every row needs the same length, at least 1 (no ragged rows)"
                    .to_string(),
            );
        }
        let dtype = M::lit(&self.out_dtype);
        for (i, row) in self.table.iter().enumerate() {
            for value in row.iter().filter_map(|v| known::<M, f64>(v)) {
                if !fits(value, dtype) {
                    return Err(format!(
                        "apply_lut: entry {value} in row {i} does not fit {}",
                        dtype.short_name()
                    ));
                }
            }
        }
        Ok(())
    }

    /// One row per value; a palette (`K > 1`) adds a channel axis.
    pub fn shape(&self) -> OpShape {
        match self.width() {
            1 => OpShape::Preserve,
            k => OpShape::ExpandChannels(k),
        }
    }

    /// A table read maps a buffer to a buffer.
    pub fn output_domain(&self) -> Domain {
        Domain::Buffer
    }
}

/// Whether `value` is exactly an element of `dtype`: an integer in range for
/// an integer dtype, anything f32 does not overflow for f32 (it rounds to the
/// nearest), anything for f64.
fn fits(value: f64, dtype: DType) -> bool {
    let int_in = |lo: f64, hi: f64| value.fract() == 0.0 && value >= lo && value <= hi;
    match dtype {
        DType::F64 => true,
        DType::F32 => !value.is_finite() || value.abs() <= f64::from(f32::MAX),
        DType::U8 => int_in(0.0, u8::MAX.into()),
        DType::I8 => int_in(i8::MIN.into(), i8::MAX.into()),
        DType::U16 => int_in(0.0, u16::MAX.into()),
        DType::I16 => int_in(i16::MIN.into(), i16::MAX.into()),
        DType::U32 => int_in(0.0, u32::MAX.into()),
        DType::I32 => int_in(i32::MIN.into(), i32::MAX.into()),
        // An entry reaches the op as an f64, so only what f64 holds exactly.
        DType::U64 => int_in(0.0, 2f64.powi(53)),
        DType::I64 => int_in(-(2f64.powi(53)), 2f64.powi(53)),
    }
}

impl LutOp {
    /// `table[x]` for every element, stored as `out_dtype`; an `Err` naming the
    /// first value with no row.
    pub fn execute(&self, buf: &ViewBuffer) -> Result<ViewBuffer, String> {
        if !matches!(buf.dtype(), DType::U8 | DType::U16) {
            return Err(format!(
                "needs a u8 or u16 input, got {}",
                buf.dtype().short_name()
            ));
        }
        let n = self.table.len();
        if let Some(max) = largest(buf).filter(|&max| max >= n) {
            return Err(format!(
                "value {max} is past the table's {n} entries; clamp(0, {}, out_dtype=\"{}\") \
                 first sends larger values to the last row",
                n - 1,
                buf.dtype().short_name()
            ));
        }
        if self.width() == 1 {
            let entries: Vec<f64> = self.table.iter().map(|row| row[0]).collect();
            return Ok(crate::ops::elementwise::lookup_column(
                buf.clone(),
                &entries,
                self.out_dtype,
            ));
        }
        Ok(self.palette(buf))
    }

    /// A palette's read: each value's row of `K` entries, written as the
    /// output pixel's `K` channels in one pass over the input (a `[H, W]` or
    /// `[H, W, 1]` buffer, `validate` has checked), read where it lies. The
    /// element-wise table map writes one output per input, so it cannot.
    fn palette(&self, buf: &ViewBuffer) -> ViewBuffer {
        let k = self.width();
        let shape = vec![buf.shape()[0], buf.shape()[1], k];
        let flat: Vec<f64> = self.table.iter().flatten().copied().collect();
        with_dtype!(self.out_dtype, D => {
            let rows: Vec<D> = crate::core::convert::convert_slice(&flat);
            let mut out: Vec<D> = Vec::with_capacity(shape.iter().product());
            let mut read = |v: usize| out.extend_from_slice(&rows[v * k..(v + 1) * k]);
            match buf.dtype() {
                DType::U8 => crate::core::map::for_each_run::<u8>(buf, |run| {
                    run.iter().for_each(|&v| read(usize::from(v)))
                }),
                _ => crate::core::map::for_each_run::<u16>(buf, |run| {
                    run.iter().for_each(|&v| read(usize::from(v)))
                }),
            }
            ViewBuffer::from_vec_with_shape(out, shape)
        })
    }
}

/// The largest value of a u8/u16 buffer (`None` when it is empty), read in
/// place whatever its layout.
fn largest(buf: &ViewBuffer) -> Option<usize> {
    let mut max: Option<usize> = None;
    let mut fold = |m: Option<usize>| max = max.max(m);
    match buf.dtype() {
        DType::U8 => crate::core::map::for_each_run::<u8>(buf, |run| {
            fold(run.iter().max().map(|&v| usize::from(v)))
        }),
        DType::U16 => crate::core::map::for_each_run::<u16>(buf, |run| {
            fold(run.iter().max().map(|&v| usize::from(v)))
        }),
        _ => {}
    }
    max
}

impl<M: Mode> Op for LutOp<M> {
    fn name(&self) -> &'static str {
        "ApplyLut"
    }

    fn shape(&self) -> OpShape {
        LutOp::shape(self)
    }

    fn memory_effect(&self) -> MemoryEffect {
        // The range check and the table map read any view where it lies.
        MemoryEffect::StridePreserving
    }

    fn identity_rule(&self) -> IdentityRule {
        // Whether a table is the identity is a parameter value, which an
        // identity rule never depends on.
        IdentityRule::Never
    }

    fn is_spatial_window(&self) -> bool {
        false
    }

    fn spatial_dependency(&self) -> SpatialDependency {
        // Each output element reads the value at its own position.
        SpatialDependency::Pointwise
    }

    fn infer_strides(&self, _shape: &[usize], _strides: &[isize]) -> Option<Vec<isize>> {
        None
    }

    fn validate(
        &self,
        input_shapes: &[&[crate::ops::Dim]],
        input_dtypes: &[PlannedDType],
    ) -> Result<(), ValidationError> {
        let refuse = |got: DType| ValidationError::DTypeRequirement {
            expected: vec![DType::U8, DType::U16],
            got,
        };
        match input_dtypes.first() {
            Some(PlannedDType::Known(d)) if !matches!(d, DType::U8 | DType::U16) => {
                return Err(refuse(*d))
            }
            Some(PlannedDType::SomeFloat) => {
                return Err(ValidationError::Generic {
                    message: "apply_lut needs a u8 or u16 input, got a float; \
                              scale, round and cast it first"
                        .to_string(),
                })
            }
            _ => {}
        }
        if self.width() > 1 {
            if let Some(shape) = input_shapes.first() {
                require_single_channel(shape)?;
            }
        }
        Ok(())
    }

    fn accepted_input_dtypes(&self) -> DTypeCategory {
        DTypeCategory::Specific(vec![DType::U8, DType::U16])
    }

    fn working_dtype(&self) -> Option<DType> {
        None
    }

    fn output_dtype_rule(&self) -> OutputDTypeRule {
        OutputDTypeRule::Fixed(M::lit(&self.out_dtype))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op(table: &[&[f64]], dtype: DType) -> LutOp {
        LutOp {
            table: table.iter().map(|row| row.to_vec()).collect(),
            out_dtype: dtype,
        }
    }

    #[test]
    fn single_entry_rows_map_every_channel_in_place() {
        let buf = ViewBuffer::from_vec_with_shape(vec![0u8, 1, 2, 2, 1, 0], vec![1, 2, 3]);
        let out = op(&[&[10.0], &[20.0], &[-5.0]], DType::I16)
            .execute(&buf)
            .unwrap();
        assert_eq!(out.shape(), &[1, 2, 3]);
        assert_eq!(
            out.to_contiguous().as_slice::<i16>(),
            &[10, 20, -5, -5, 20, 10]
        );
    }

    #[test]
    fn a_palette_takes_one_channel_to_k() {
        let buf = ViewBuffer::from_vec_with_shape(vec![1u16, 0, 1], vec![1, 3, 1]);
        let out = op(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]], DType::F32)
            .execute(&buf)
            .unwrap();
        assert_eq!(out.shape(), &[1, 3, 3]);
        assert_eq!(
            out.to_contiguous().as_slice::<f32>(),
            &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn a_value_past_the_table_is_an_error_naming_it() {
        let buf = ViewBuffer::from_vec_with_shape(vec![0u8, 7, 2], vec![1, 3]);
        let err = op(&[&[1.0], &[2.0], &[3.0]], DType::U8)
            .execute(&buf)
            .unwrap_err();
        assert!(
            err.contains("value 7") && err.contains(r#"clamp(0, 2, out_dtype="u8")"#),
            "{err}"
        );
    }

    #[test]
    fn a_strided_view_reads_its_own_values() {
        let full = ViewBuffer::from_vec_with_shape((0u8..12).collect::<Vec<_>>(), vec![3, 4]);
        let flipped = full.flip(&[1]);
        let table: Vec<Vec<f64>> = (0..12).map(|v| vec![f64::from(v) * 2.0]).collect();
        let out = LutOp {
            table,
            out_dtype: DType::U8,
        }
        .execute(&flipped)
        .unwrap();
        assert_eq!(
            out.to_contiguous().as_slice::<u8>(),
            &[6, 4, 2, 0, 14, 12, 10, 8, 22, 20, 18, 16]
        );
        // The range check sees the flipped view's values too.
        let short = op(&[&[0.0][..]; 11], DType::U8);
        assert!(short.execute(&flipped).unwrap_err().contains("value 11"));
    }

    #[test]
    fn a_palette_reads_a_strided_view_in_logical_order() {
        // [2, 3, 1] flipped both ways: logical order 5, 4, 3, 2, 1, 0.
        let full = ViewBuffer::from_vec_with_shape((0u16..6).collect::<Vec<_>>(), vec![2, 3, 1]);
        let flipped = full.flip(&[0, 1]);
        let table: Vec<Vec<f64>> = (0..6).map(|v| vec![f64::from(v), -f64::from(v)]).collect();
        let out = LutOp {
            table,
            out_dtype: DType::I16,
        }
        .execute(&flipped)
        .unwrap();
        assert_eq!(out.shape(), &[2, 3, 2]);
        assert_eq!(
            out.to_contiguous().as_slice::<i16>(),
            &[5, -5, 4, -4, 3, -3, 2, -2, 1, -1, 0, 0]
        );
    }

    #[test]
    fn check_refuses_what_no_row_can_use() {
        let wire = |table: &[&[f64]], dtype: DType| op(table, dtype).check();
        assert!(wire(&[], DType::U8).is_err());
        assert!(wire(&[&[1.0, 2.0], &[3.0]], DType::U8).is_err());
        assert!(wire(&[&[]], DType::U8).is_err());
        assert!(wire(&[&[256.0]], DType::U8).is_err());
        assert!(wire(&[&[-1.0]], DType::U16).is_err());
        assert!(wire(&[&[0.5]], DType::I32).is_err());
        assert!(wire(&[&[f64::NAN]], DType::I32).is_err());
        assert!(wire(&[&[1e300]], DType::F32).is_err());
        assert!(wire(&[&[f64::NAN], &[f64::INFINITY]], DType::F32).is_ok());
        assert!(wire(&[&[-128.0, 127.0]], DType::I8).is_ok());
    }

    #[test]
    fn the_shape_is_preserved_or_gains_k_channels() {
        assert_eq!(op(&[&[0.0]], DType::U8).shape(), OpShape::Preserve);
        assert_eq!(
            op(&[&[0.0, 0.0, 0.0]], DType::U8).shape(),
            OpShape::ExpandChannels(3)
        );
    }
}
