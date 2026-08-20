//! `read_file_bytes`: read the bytes a path column names, without decoding.
//!
//! This is [`crate::fetch`] exposed directly — the same stage the `file_path`
//! source runs before it decodes, with the decode omitted. The bytes are
//! returned verbatim, so an encoded file survives the round trip unchanged and
//! can be written back byte-for-byte. That is the one thing a decode cannot
//! offer: re-encoding a decoded JPEG never reproduces the original file, and
//! neither `ViewBuffer` nor the image sinks carry EXIF/ICC metadata.
//!
//! It also gives the header-only metadata family (`crate::image_metadata`) a
//! way to reach remote files: those functions take a `Binary` column, which
//! until now could only be produced outside the plugin.

use std::collections::HashMap;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::cloud::CloudOptions;
use crate::fetch;

/// Static kwargs for [`read_file_bytes`].
///
/// Closed for the same reason as [`GraphKwargs`]: this is a plugin-boundary
/// struct, so a kwarg Python emits and Rust does not declare is drift, not a
/// value to discard in silence.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReadBytesKwargs {
    /// Cloud credentials, keyed as `CloudOptions::from_map` expects. Same map
    /// the `file_path` source spec carries.
    #[serde(default)]
    pub cloud_options: Option<HashMap<String, String>>,
    /// `"raise"` (default) or `"null"`, matching `source(on_error=...)`.
    #[serde(default)]
    pub on_error: Option<String>,
    /// Locations this call may read from, matching
    /// `source(allowed_roots=...)`. Absent means unrestricted.
    #[serde(default)]
    pub allowed_roots: Option<Vec<String>>,
}

/// Read each path's bytes into a `Binary` column, without decoding.
#[polars_expr(output_type=Binary)]
fn read_file_bytes(inputs: &[Series], kwargs: ReadBytesKwargs) -> PolarsResult<Series> {
    let input = &inputs[0];
    let name = input.name().clone();

    let null_on_error = fetch::parse_on_error(
        kwargs.on_error.as_deref().unwrap_or("raise"),
        "read_bytes()",
    )?;
    let options = kwargs.cloud_options.as_ref().map(CloudOptions::from_map);

    // An all-null column has no paths to read; keep it null rather than
    // failing the dtype check on a column that carries no paths either way.
    if input.dtype() == &DataType::Null {
        return Ok(Series::full_null(name, input.len(), &DataType::Binary));
    }

    let ca = input.str().map_err(|_| {
        polars_err!(ComputeError:
            "read_bytes() expects a String column of paths, got {:?}",
            input.dtype()
        )
    })?;

    // Same batching as the `file_path` source: this call's remote paths are
    // deduped and fetched concurrently up front, local ones read per row.
    let policy = kwargs
        .allowed_roots
        .as_deref()
        .map(fetch::PathPolicy::new)
        .unwrap_or_default();
    let batch = fetch::prefetch(ca, options.as_ref(), &policy);

    let mut builder = BinaryChunkedBuilder::new(name, ca.len());
    for path in ca.iter() {
        let Some(path) = path else {
            builder.append_null();
            continue;
        };
        match fetch::row_bytes(&batch, path, options.as_ref(), &policy) {
            Ok(bytes) => builder.append_value(bytes.as_ref()),
            Err(_) if null_on_error => builder.append_null(),
            Err(e) => return Err(polars_err!(ComputeError: "{}", e)),
        }
    }
    Ok(builder.finish().into_series())
}
