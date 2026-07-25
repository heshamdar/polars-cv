//! Stage one of every path-based read: a column of paths → bytes.
//!
//! The `file_path` source is two stages welded together — fetch the bytes a path
//! names, then decode them as an image. This module is the first stage on its
//! own, so both consumers share one mechanism:
//!
//! - graph execution (`graph::compiled`) fetches, then decodes;
//! - the `read_file_bytes` expression (`crate::read_bytes`) fetches, and stops.
//!
//! Keeping them on one implementation means a change to fetching — a new scheme,
//! a credential fix, a retry policy — lands once and applies to both, and the
//! error text a user sees is the same either way.
//!
//! # Batching
//!
//! Fetching is **per plugin call**, which under the streaming engine is one
//! morsel: [`prefetch`] dedups the call's remote paths and fetches them
//! concurrently up front, converting per-row network latency into per-call
//! latency. Local paths are not touched there — [`row_bytes`] reads them inline
//! per row, so only one local file is resident at a time.
//!
//! # Security
//!
//! Neither entry point sanitizes paths: they read whatever the column names,
//! local or remote.
//!
//! TODO: Add path allowlisting/sandboxing here. This covers both the
//! `file_path` source and the `read_file_bytes` expression, which is why it
//! belongs in this module rather than at either call site. Safe when data comes
//! from trusted input only.

use std::borrow::Cow;
use std::collections::HashMap;

use polars::prelude::*;

use crate::cloud::{self, CloudOptions};

/// Maximum concurrent remote fetches within one plugin call.
///
/// Shared by both consumers deliberately: a knob on one and not the other would
/// let their behavior drift. If this ever needs tuning, expose it on both.
pub const DEFAULT_CONCURRENCY: usize = 16;

/// One call's fetched remote bytes for a single path column (path → result).
///
/// Errors are carried per path rather than raised, so each surfaces at the row
/// that asked for it and the caller's error policy applies as usual.
#[derive(Default)]
pub struct FetchedBatch {
    remote: HashMap<String, Result<Vec<u8>, String>>,
}

impl FetchedBatch {
    /// An empty batch — nothing prefetched, every path read inline.
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Dedup and concurrently fetch the remote paths in `ca`.
///
/// Local paths are deliberately skipped: [`row_bytes`] reads them inline, which
/// keeps at most one local file in memory instead of the whole call's worth.
/// A column with no remote paths yields an empty batch and costs nothing.
pub fn prefetch(
    ca: &StringChunked,
    options: Option<&CloudOptions>,
    max_concurrency: usize,
) -> FetchedBatch {
    let remote: Vec<String> = ca
        .iter()
        .flatten()
        .filter(|p| cloud::is_remote_path(p))
        .map(str::to_string)
        .collect();
    if remote.is_empty() {
        return FetchedBatch::empty();
    }
    FetchedBatch {
        // `read_files_concurrent` dedups internally, so repeated paths across
        // rows are fetched once.
        remote: cloud::read_files_concurrent(&remote, options, max_concurrency),
    }
}

/// Bytes for one row's path.
///
/// Remote paths come from `batch`; a miss falls back to an inline fetch rather
/// than failing, so a caller that prefetched a different column (or skipped
/// prefetching entirely) still works. Local paths are read here.
///
/// Borrows prefetched bytes and owns freshly-read ones, so the common remote
/// path stays copy-free.
pub fn row_bytes<'a>(
    batch: &'a FetchedBatch,
    path: &str,
    options: Option<&CloudOptions>,
) -> Result<Cow<'a, [u8]>, String> {
    if cloud::is_remote_path(path) {
        return match batch.remote.get(path) {
            Some(Ok(bytes)) => Ok(Cow::Borrowed(bytes.as_slice())),
            Some(Err(e)) => Err(format!("Failed to read remote file '{path}': {e}")),
            // Defensive: every remote path in the call is prefetched, but fall
            // back to an inline fetch rather than miss.
            None => cloud::read_file(path, options)
                .map(Cow::Owned)
                .map_err(|e| format!("Failed to read remote file '{path}': {e}")),
        };
    }
    // Already known non-remote: read the literal path, stripping only a
    // `file://` prefix. (Routing through the general `read_file` would re-parse
    // a bare colon-bearing filename as a bogus cloud URL.)
    cloud::read_local_path(path)
        .map(Cow::Owned)
        .map_err(|e| format!("Failed to read local file '{path}': {e}"))
}

/// Parse an `on_error` setting into "nulls the row on failure".
///
/// Shared by the `file_path` source and the `read_file_bytes` expression so the
/// accepted values and the rejection message cannot drift between them.
/// `context` names what is being configured, e.g. `node 'src'`.
pub fn parse_on_error(value: &str, context: &str) -> PolarsResult<bool> {
    match value {
        "raise" => Ok(false),
        "null" => Ok(true),
        other => Err(polars_err!(ComputeError:
            "Unknown on_error value '{}' for {} (expected 'raise' or 'null')",
            other, context
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn empty_string_ca() -> StringChunked {
        StringChunked::from_iter_options("paths".into(), std::iter::empty::<Option<&str>>())
    }

    #[test]
    fn prefetch_skips_columns_without_remote_paths() {
        let ca = StringChunked::from_iter_options(
            "paths".into(),
            [Some("/tmp/a.png"), None, Some("relative/b.png")].into_iter(),
        );
        let batch = prefetch(&ca, None, DEFAULT_CONCURRENCY);
        assert!(batch.remote.is_empty());
        assert!(prefetch(&empty_string_ca(), None, DEFAULT_CONCURRENCY)
            .remote
            .is_empty());
    }

    #[test]
    fn row_bytes_reads_local_files_verbatim() {
        let dir = std::env::temp_dir().join("polars_cv_fetch_local");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bytes.bin");
        // Deliberately not valid image data: fetching must not care.
        let payload: Vec<u8> = (0u8..=255).collect();
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&payload)
            .unwrap();

        let batch = FetchedBatch::empty();
        let got = row_bytes(&batch, path.to_str().unwrap(), None).unwrap();
        assert_eq!(got.as_ref(), payload.as_slice());

        // The `file://` form resolves to the same file.
        let uri = format!("file://{}", path.to_str().unwrap());
        assert_eq!(
            row_bytes(&batch, &uri, None).unwrap().as_ref(),
            &payload[..]
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn row_bytes_reports_missing_local_files() {
        let batch = FetchedBatch::empty();
        let err = row_bytes(&batch, "/nonexistent/polars-cv/missing.png", None).unwrap_err();
        assert!(err.contains("Failed to read local file"), "{err}");
        assert!(err.contains("missing.png"), "{err}");
    }

    #[test]
    fn row_bytes_surfaces_prefetched_errors() {
        let batch = FetchedBatch {
            remote: HashMap::from([(
                "s3://bucket/key.png".to_string(),
                Err("access denied".to_string()),
            )]),
        };
        let err = row_bytes(&batch, "s3://bucket/key.png", None).unwrap_err();
        assert!(err.contains("Failed to read remote file"), "{err}");
        assert!(err.contains("access denied"), "{err}");
    }

    #[test]
    fn row_bytes_borrows_prefetched_bytes() {
        let batch = FetchedBatch {
            remote: HashMap::from([("s3://bucket/key.png".to_string(), Ok(vec![1, 2, 3]))]),
        };
        let got = row_bytes(&batch, "s3://bucket/key.png", None).unwrap();
        assert!(matches!(got, Cow::Borrowed(_)));
        assert_eq!(got.as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn parse_on_error_accepts_exactly_raise_and_null() {
        assert!(!parse_on_error("raise", "node 'a'").unwrap());
        assert!(parse_on_error("null", "node 'a'").unwrap());
        let err = parse_on_error("skip", "node 'a'").unwrap_err().to_string();
        assert!(err.contains("Unknown on_error value 'skip'"), "{err}");
        assert!(err.contains("node 'a'"), "{err}");
    }
}
