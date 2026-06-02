//! `image_bytes` source: decode encoded image bytes (PNG, JPEG, etc.) into a
//! `ViewBuffer` via view-buffer's [`ImageAdapter`].
//!
//! This is the same behavior as v1's `decode_source(... "image_bytes" ...)`
//! at `polars-cv/src/execute.rs:240`. The v2 path skips the optional
//! `pipeline.source.dtype` cast for now — it'll be added back as a `params`
//! key (`out_dtype`) once the bridge layer surfaces it.

use std::sync::Arc;

use view_buffer::interop::image::ImageAdapter;
use view_buffer::ops::{Domain, NodeOutput};

use crate::params::ParamMap;
use crate::source::{Source, SourceError, SourceInputs, SourceRegistration};

pub struct ImageBytesSource;

impl Source for ImageBytesSource {
    fn name(&self) -> &'static str {
        "image_bytes"
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
                    name: "image_bytes",
                    row_idx,
                })
            }
        };
        let buf = ImageAdapter::decode(bytes).map_err(|e| SourceError::DecodeFailed {
            name: "image_bytes",
            row_idx,
            message: format!("{e:?}"),
        })?;
        Ok(NodeOutput::from_buffer(buf))
    }
}

fn factory(_params: &ParamMap) -> Result<Arc<dyn Source>, SourceError> {
    Ok(Arc::new(ImageBytesSource))
}

inventory::submit! {
    SourceRegistration {
        name: "image_bytes",
        factory,
        output_domain: Domain::Buffer,
    }
}
