//! The largest value of a u8/u16 buffer, for the lookup-table range check.

use crate::core::buffer::ViewBuffer;
use crate::core::dtype::DType;

/// The largest value of a u8/u16 buffer (`None` when it is empty), read in
/// place whatever its layout.
pub(crate) fn largest(buf: &ViewBuffer) -> Option<usize> {
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
