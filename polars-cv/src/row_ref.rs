//! How an error names the row it is about: the one rendering.
//!
//! A plugin sees the rows polars passes it in one call: the whole column in
//! memory, one batch under the streaming engine. Nothing in the call says
//! where a batch starts in the frame, so an index is only ever the batch's.
//! Rendered bare (`(row 11)`), a streaming batch's index read as the frame's
//! row and named the wrong one. So every row reference states the batch it
//! counts in: `(row 11 of 12 in this batch)`; when the batch is the whole
//! frame (in memory) the index is the frame's row.
//!
//! Every per-row error is rendered through [`RowRef`] (or [`at_row`]), once,
//! by the code that owns the row loop; a message written inside the loop
//! names no row of its own. `tests/test_row_ref_guard.py::
//! test_rows_are_named_only_through_row_ref` rejects a hand-rolled one.

use core::fmt;

/// A row of a plugin call's batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowRef {
    /// The row's index within the batch.
    pub index: usize,
    /// How many rows the batch holds.
    pub batch: usize,
}

impl RowRef {
    pub(crate) fn new(index: usize, batch: usize) -> Self {
        RowRef { index, batch }
    }
}

impl fmt::Display for RowRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(row {} of {} in this batch)", self.index, self.batch)
    }
}

/// `message`, followed by the row it is about.
pub(crate) fn at_row(message: impl fmt::Display, index: usize, batch: usize) -> String {
    format!("{message} {}", RowRef::new(index, batch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_row_states_the_batch_it_counts_in() {
        assert_eq!(
            at_row("crop: window outside", 11, 12),
            "crop: window outside (row 11 of 12 in this batch)"
        );
    }
}
