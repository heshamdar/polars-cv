//! `png` sink: encode a `Buffer` `NodeOutput` to PNG bytes via view-buffer's
//! [`ImageAdapter`]. Matches the v1 behavior at
//! `polars-cv/src/execute.rs:299`.

use std::sync::Arc;

use view_buffer::interop::image::ImageAdapter;
use view_buffer::ops::{Domain, NodeOutput};

use crate::params::ParamMap;
use crate::sink::{Sink, SinkError, SinkRegistration, SinkRowOutput};

pub struct PngSink;

impl Sink for PngSink {
    fn name(&self) -> &'static str {
        "png"
    }

    fn input_domain(&self) -> Domain {
        Domain::Buffer
    }

    fn consume(&self, row_idx: usize, input: &NodeOutput) -> Result<SinkRowOutput, SinkError> {
        let buf_arc = input.as_buffer().ok_or_else(|| SinkError::DomainMismatch {
            name: "png",
            expected: Domain::Buffer,
            got: input.domain(),
        })?;
        let bytes = ImageAdapter::encode(buf_arc.as_ref(), image::ImageFormat::Png).map_err(|e| {
            SinkError::Failed {
                name: "png",
                row_idx,
                message: format!("{e:?}"),
            }
        })?;
        Ok(SinkRowOutput::Bytes(bytes))
    }
}

fn factory(_params: &ParamMap) -> Result<Arc<dyn Sink>, SinkError> {
    Ok(Arc::new(PngSink))
}

inventory::submit! {
    SinkRegistration {
        name: "png",
        factory,
        input_domain: Domain::Buffer,
    }
}
