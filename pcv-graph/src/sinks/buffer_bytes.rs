//! Test sink: serialize a `Buffer` `NodeOutput` back to its raw bytes.
//!
//! Mirror of [`crate::sources::bytes_passthrough::BytesPassthroughSource`] —
//! keeps the integration test self-contained.

use std::sync::Arc;

use view_buffer::ops::{Domain, NodeOutput};

use crate::params::ParamMap;
use crate::sink::{Sink, SinkError, SinkRegistration, SinkRowOutput};

pub struct BufferBytesSink;

impl Sink for BufferBytesSink {
    fn name(&self) -> &'static str {
        "buffer_bytes"
    }

    fn input_domain(&self) -> Domain {
        Domain::Buffer
    }

    fn consume(&self, row_idx: usize, input: &NodeOutput) -> Result<SinkRowOutput, SinkError> {
        let buf_arc = input.as_buffer().ok_or_else(|| SinkError::DomainMismatch {
            name: "buffer_bytes",
            expected: Domain::Buffer,
            got: input.domain(),
        })?;
        let _ = row_idx;
        let buf: &view_buffer::ViewBuffer = buf_arc.as_ref();
        let (ptr, shape, _strides, dtype) = buf.as_raw_parts();
        let elem_count: usize = shape.iter().product();
        let total_bytes = elem_count * dtype.size_of();
        // Safety: as_raw_parts returns a pointer valid for `num_elements *
        // size_of_dtype` bytes for contiguous buffers; the buffer (held via
        // Arc) outlives this slice. Strided buffers would need to_contiguous()
        // first; the test source produces contiguous data.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, total_bytes) };
        Ok(SinkRowOutput::Bytes(bytes.to_vec()))
    }
}

fn factory(_params: &ParamMap) -> Result<Arc<dyn Sink>, SinkError> {
    Ok(Arc::new(BufferBytesSink))
}

inventory::submit! {
    SinkRegistration {
        name: "buffer_bytes",
        factory,
        input_domain: Domain::Buffer,
    }
}
