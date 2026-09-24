//! The one dtype name that is not an engine dtype.
//!
//! Kept in its own file because it is the deliberate exception to "dtype names
//! live in `dtype_table!`": `test_no_second_dtype_spelling_table` exempts
//! exactly this file, for this reason.

/// The element dtype a tensor sink may downcast to at encode time.
///
/// Only half precision: the engine has no f16 dtype, so it exists purely as an
/// encode-boundary downcast (halving the output bytes). Every other dtype is a
/// `.cast()` in the pipeline, which the planner tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkDType {
    F16,
}

view_buffer::naming::named_variants!(SinkDType {
    "f16" | "float16" => F16,
});
