//! Sink trait — consumes a [`NodeOutput`] and emits a row of output bytes /
//! values for the bridge layer to assemble into a Polars Series.
//!
//! Sinks live behind a registry just like ops and sources so adding a built-in
//! (numpy, torch, png, jpeg, …) or an extension sink is a one-file change.

use thiserror::Error;
use view_buffer::ops::{Domain, NodeOutput};

use crate::params::ParamMap;

/// Output emitted by a sink for a single row.
///
/// The bridge layer in `polars-cv` knows how to assemble each variant into a
/// Polars Series of the appropriate dtype.
pub enum SinkRowOutput {
    /// Encoded bytes (e.g. PNG, JPEG, blob layout).
    Bytes(Vec<u8>),
    /// A single scalar value (e.g. a reduction result).
    Scalar(f64),
    /// A 1-D vector of f64 (e.g. histogram, centroid).
    Vector(Vec<f64>),
    /// Raw numpy-shaped output: `(buffer, dtype, shape, strides)`.
    ///
    /// The bridge wraps this as a Polars Struct using the existing
    /// `output.rs` zero-copy path. `strides` is in **elements**, not bytes.
    Numpy {
        data: Vec<u8>,
        dtype: &'static str,
        shape: Vec<u64>,
        strides: Vec<i64>,
    },
}

/// Errors a sink can raise.
#[derive(Debug, Error)]
pub enum SinkError {
    #[error("sink `{name}` got unexpected input domain {got:?}, expected {expected:?}")]
    DomainMismatch {
        name: &'static str,
        expected: Domain,
        got: Domain,
    },

    #[error("sink `{name}` failed on row {row_idx}: {message}")]
    Failed {
        name: &'static str,
        row_idx: usize,
        message: String,
    },

    #[error("sink `{name}` config error: {message}")]
    Config {
        name: &'static str,
        message: String,
    },
}

/// A registered sink.
pub trait Sink: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// Domain this sink expects on its input.
    fn input_domain(&self) -> Domain;
    /// Consume one row.
    fn consume(&self, row_idx: usize, input: &NodeOutput) -> Result<SinkRowOutput, SinkError>;
}

/// Inventory entry for built-in / extension sinks.
pub struct SinkRegistration {
    pub name: &'static str,
    pub factory: fn(&ParamMap) -> Result<std::sync::Arc<dyn Sink>, SinkError>,
    pub input_domain: Domain,
}

inventory::collect!(SinkRegistration);

pub fn iter_sinks() -> impl Iterator<Item = &'static SinkRegistration> {
    inventory::iter::<SinkRegistration>.into_iter()
}

pub fn find_sink(name: &str) -> Option<&'static SinkRegistration> {
    iter_sinks().find(|reg| reg.name == name)
}
