//! Source trait — produces a [`NodeOutput`] for a single row.
//!
//! Sources are how a pipeline reads input data: image bytes, raw arrays,
//! existing list/array Polars columns, file paths, etc. Like ops, they're
//! registered via inventory so adding a built-in or extension source is a
//! single-file change.

use thiserror::Error;
use view_buffer::ops::{Domain, NodeOutput};

use crate::params::ParamMap;

/// Per-row inputs handed to a source's `produce` method.
///
/// The bridge layer materializes whatever the source declared as its
/// [`SourceSpec::input_columns`] into a slice of byte-or-blob references
/// before invoking the source. The source itself is Polars-free.
pub enum SourceInputs<'a> {
    /// No per-row input needed (e.g. constant generators, file_path with a
    /// literal path param).
    None,
    /// Bytes for the current row (e.g. image_bytes).
    Bytes(&'a [u8]),
    /// Raw byte buffer plus an optional path/uri context (e.g. blob, raw).
    Buffer { bytes: &'a [u8] },
}

/// Errors a source can raise while producing a row.
#[derive(Debug, Error)]
pub enum SourceError {
    #[error("source `{name}` got null input on row {row_idx}")]
    NullInput { name: &'static str, row_idx: usize },

    #[error("source `{name}` could not decode row {row_idx}: {message}")]
    DecodeFailed {
        name: &'static str,
        row_idx: usize,
        message: String,
    },

    #[error("source `{name}` config error: {message}")]
    Config {
        name: &'static str,
        message: String,
    },
}

/// A registered source — produces a [`NodeOutput`] for each row of a
/// pipeline batch.
pub trait Source: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// Domain of the produced output (`Buffer`, `Contour`, etc.).
    fn output_domain(&self) -> Domain;
    /// Produce one output for the current row.
    fn produce(&self, row_idx: usize, inputs: &SourceInputs)
        -> Result<NodeOutput, SourceError>;
}

/// Inventory entry for built-in / extension sources.
pub struct SourceRegistration {
    pub name: &'static str,
    pub factory: fn(&ParamMap) -> Result<std::sync::Arc<dyn Source>, SourceError>,
    pub output_domain: Domain,
}

inventory::collect!(SourceRegistration);

pub fn iter_sources() -> impl Iterator<Item = &'static SourceRegistration> {
    inventory::iter::<SourceRegistration>.into_iter()
}

pub fn find_source(name: &str) -> Option<&'static SourceRegistration> {
    iter_sources().find(|reg| reg.name == name)
}
