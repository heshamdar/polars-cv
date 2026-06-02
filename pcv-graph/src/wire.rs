//! Wire-format codec for [`Graph`].
//!
//! Default is JSON for the v2 bootstrap so the bridge can debug payloads
//! easily; bincode v2 will land in a follow-up commit. Both forms stamp
//! [`crate::WIRE_VERSION`] on the payload via the `Graph::wire_version`
//! field, so a mismatched plugin/Python combination fails loudly.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ir::Graph;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireFormat {
    Json,
}

impl WireFormat {
    pub fn from_env_or_default() -> Self {
        match std::env::var("PCV_WIRE_FORMAT").as_deref() {
            Ok("json") | Err(_) => WireFormat::Json,
            Ok(other) => panic!("unknown PCV_WIRE_FORMAT={other:?}; expected `json`"),
        }
    }
}

#[derive(Debug, Error)]
pub enum WireError {
    #[error("encode failed: {0}")]
    Encode(String),
    #[error("decode failed: {0}")]
    Decode(String),
}

pub fn encode(graph: &Graph, format: WireFormat) -> Result<Vec<u8>, WireError> {
    match format {
        WireFormat::Json => {
            serde_json::to_vec(graph).map_err(|e| WireError::Encode(e.to_string()))
        }
    }
}

pub fn decode(bytes: &[u8], format: WireFormat) -> Result<Graph, WireError> {
    match format {
        WireFormat::Json => {
            serde_json::from_slice(bytes).map_err(|e| WireError::Decode(e.to_string()))
        }
    }
}

/// Convenience wrapper: pick the format from `PCV_WIRE_FORMAT` (or default).
pub fn encode_default(graph: &Graph) -> Result<Vec<u8>, WireError> {
    encode(graph, WireFormat::from_env_or_default())
}

pub fn decode_default(bytes: &[u8]) -> Result<Graph, WireError> {
    decode(bytes, WireFormat::from_env_or_default())
}

/// Compact codec helper for embedded enums / params that need their own
/// stable string representation.
pub fn json_value_of<T: Serialize>(value: &T) -> Result<serde_json::Value, WireError> {
    serde_json::to_value(value).map_err(|e| WireError::Encode(e.to_string()))
}

pub fn json_into<T: for<'de> Deserialize<'de>>(value: serde_json::Value) -> Result<T, WireError> {
    serde_json::from_value(value).map_err(|e| WireError::Decode(e.to_string()))
}
