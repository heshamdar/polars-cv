//! Test source: wraps the per-row byte slice in a 1-D `u8` `ViewBuffer`.
//!
//! Used by the executor integration tests so the trait surface can be
//! exercised end-to-end before the real `image_bytes` source (which depends
//! on the `image` crate) lands. Real sources follow the same shape.

use std::sync::Arc;

use view_buffer::ops::{Domain, NodeOutput};
use view_buffer::ViewBuffer;

use crate::params::ParamMap;
use crate::source::{Source, SourceError, SourceInputs, SourceRegistration};

pub struct BytesPassthroughSource;

impl Source for BytesPassthroughSource {
    fn name(&self) -> &'static str {
        "bytes_passthrough"
    }

    fn output_domain(&self) -> Domain {
        Domain::Buffer
    }

    fn produce(
        &self,
        row_idx: usize,
        inputs: &SourceInputs,
    ) -> Result<NodeOutput, SourceError> {
        let bytes = match inputs {
            SourceInputs::Bytes(b) => *b,
            SourceInputs::Buffer { bytes } => *bytes,
            SourceInputs::None => {
                return Err(SourceError::NullInput {
                    name: "bytes_passthrough",
                    row_idx,
                })
            }
        };
        let len = bytes.len();
        let buf = ViewBuffer::from_vec_with_shape(bytes.to_vec(), vec![len]);
        Ok(NodeOutput::from_buffer(buf))
    }
}

fn factory(_params: &ParamMap) -> Result<Arc<dyn Source>, SourceError> {
    Ok(Arc::new(BytesPassthroughSource))
}

inventory::submit! {
    SourceRegistration {
        name: "bytes_passthrough",
        factory,
        output_domain: Domain::Buffer,
    }
}
