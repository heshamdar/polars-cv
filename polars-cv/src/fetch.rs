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
//! A path column is data, and data can come from somewhere you do not control.
//! [`PathPolicy`] is the allowlist both entry points check against, and it lives
//! here for the same reason the rest of this module does: it is the one stage
//! both consumers share, so a restriction lands for both at once and cannot be
//! set on one and forgotten on the other.
//!
//! It is **opt-in**: the default policy allows everything, which is what every
//! existing pipeline gets. Pass `allowed_roots=` to `source("file_path", ...)`
//! or `.cv.read_bytes(...)` to restrict a query whose path column is not
//! trusted.
//!
//! Both functions take the policy as a required argument rather than reading it
//! from a field somewhere: a caller that forgets it does not silently get the
//! unrestricted behaviour, it fails to compile.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

use polars::prelude::*;

use crate::cloud::{self, CloudOptions};

/// Make `path` absolute and resolve `.` / `..` textually.
///
/// Used when a path cannot be canonicalized (it does not exist yet). Purely
/// lexical, so it cannot see through a symlink — which is why
/// [`PathPolicy::check`] canonicalizes first and only falls back to this.
fn lexical_absolute(path: &Path) -> PathBuf {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("/"))
            .join(path)
    };
    let mut out = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            // `..` above the root stays at the root, matching the kernel.
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Canonicalize `path` if it exists; otherwise resolve as much of it as does
/// exist and lexically reattach the missing suffix.
///
/// A plain `canonicalize` fails outright for a path that has not been created
/// yet, and falling all the way back to [`lexical_absolute`] leaves any
/// symlinked ancestor unresolved — on macOS, `std::env::temp_dir()` lives
/// under `/var`, itself a symlink to `/private/var`, so a not-yet-existing
/// file under an allowed temp-dir root would compare unequal to that root's
/// canonicalized form even though both denote the same location. Walking up
/// to the nearest existing ancestor and reattaching the missing suffix keeps
/// the two forms comparable.
fn resolve_best_effort(path: &Path) -> PathBuf {
    if let Ok(resolved) = std::fs::canonicalize(path) {
        return resolved;
    }
    let absolute = lexical_absolute(path);
    let mut suffix: Vec<&std::ffi::OsStr> = Vec::new();
    let mut ancestor = absolute.as_path();
    while let Some(parent) = ancestor.parent() {
        suffix.push(ancestor.file_name().unwrap_or_default());
        ancestor = parent;
        if let Ok(mut resolved) = std::fs::canonicalize(ancestor) {
            suffix.reverse();
            resolved.extend(suffix);
            return resolved;
        }
    }
    absolute
}

/// One allowed location: a local directory or a remote URI prefix.
#[derive(Debug, Clone)]
enum AllowedRoot {
    /// A local directory, resolved as far as it exists.
    Local(PathBuf),
    /// A remote URI prefix, normalized to end in `/` so that
    /// `s3://bucket/public` cannot also admit `s3://bucket/public-evil/...`.
    Remote(String),
}

/// Where a path column is permitted to read from.
///
/// `PathPolicy::default()` is the unrestricted policy — the only spelling of
/// it, so there is no second constructor to keep in step — and is what every
/// pipeline that does not ask for a sandbox gets. A non-empty policy is a *deny by default*
/// list: a path that matches no entry is refused, rather than being read and
/// hoped about.
///
/// Local and remote entries live in one list because they are one question —
/// "may this column read that?" — and splitting them is how a sandbox comes to
/// cover the filesystem while leaving `s3://` open. An entry is remote if it
/// parses as a remote URI (`cloud::is_remote_path`), local otherwise.
#[derive(Debug, Clone, Default)]
pub struct PathPolicy {
    roots: Vec<AllowedRoot>,
}

impl PathPolicy {
    /// Restrict reads to `roots`, or leave unrestricted if `roots` is empty.
    ///
    /// Local roots are canonicalized so the comparison in [`check`](Self::check)
    /// is between two resolved paths; a root that does not exist is kept in
    /// lexical form rather than dropped, so a typo'd root denies everything
    /// instead of silently widening the policy.
    pub fn new(roots: &[String]) -> Self {
        Self {
            roots: roots
                .iter()
                .map(|root| {
                    if cloud::is_remote_path(root) {
                        let mut prefix = root.clone();
                        if !prefix.ends_with('/') {
                            prefix.push('/');
                        }
                        AllowedRoot::Remote(prefix)
                    } else {
                        let path = Path::new(root);
                        AllowedRoot::Local(resolve_best_effort(path))
                    }
                })
                .collect(),
        }
    }

    /// Refuse `path` unless it falls inside an allowed root.
    ///
    /// Local paths are canonicalized before comparison, so `..` segments and
    /// symlinks are resolved rather than compared as text — a check against the
    /// literal string would be defeated by `/allowed/../etc/passwd`. When the
    /// file itself does not exist, [`resolve_best_effort`] resolves as much of
    /// its ancestry as does exist so a symlinked root still compares equal; the
    /// read then fails as not-found, which is the same answer.
    ///
    /// The comparison is component-wise ([`Path::starts_with`]), so an allowed
    /// root of `/data/images` does not also admit `/data/images-private`.
    pub fn check(&self, path: &str) -> Result<(), String> {
        if self.roots.is_empty() {
            return Ok(());
        }
        if cloud::is_remote_path(path) {
            // Object stores treat a key literally, but an HTTP server in front
            // of one may normalize `..` and walk out of the prefix. Under a
            // policy that is not a risk worth carrying for a key shape nobody
            // writes on purpose.
            if path.split('/').any(|segment| segment == "..") {
                return Err(self.denial(path, "it contains a '..' segment"));
            }
            let allowed = self.roots.iter().any(|root| match root {
                AllowedRoot::Remote(prefix) => {
                    path.starts_with(prefix.as_str())
                        // `s3://bucket/public/` also permits the exact prefix
                        // with no trailing slash.
                        || path.len() + 1 == prefix.len() && prefix.starts_with(path)
                }
                AllowedRoot::Local(_) => false,
            });
            return if allowed {
                Ok(())
            } else {
                Err(self.denial(path, "it is outside every allowed root"))
            };
        }
        let stripped = path.strip_prefix("file://").unwrap_or(path);
        let candidate = Path::new(stripped);
        let resolved = resolve_best_effort(candidate);
        let allowed = self.roots.iter().any(|root| match root {
            AllowedRoot::Local(dir) => resolved.starts_with(dir),
            AllowedRoot::Remote(_) => false,
        });
        if allowed {
            Ok(())
        } else {
            Err(self.denial(path, "it is outside every allowed root"))
        }
    }

    /// A denial that says what was refused and what would be accepted.
    fn denial(&self, path: &str, reason: &str) -> String {
        let roots: Vec<String> = self
            .roots
            .iter()
            .map(|root| match root {
                AllowedRoot::Local(dir) => dir.display().to_string(),
                AllowedRoot::Remote(prefix) => prefix.clone(),
            })
            .collect();
        format!(
            "path '{path}' is not permitted: {reason}. This column is restricted \
             by allowed_roots={roots:?}; a path is accepted only if it resolves \
             inside one of them."
        )
    }
}

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
/// `policy` is required rather than optional so a new caller cannot reach the
/// network by omitting it. Denied paths are filtered out here — the point of a
/// sandbox is that the request is never made — and [`row_bytes`] then produces
/// the refusal for the row that asked, so the message is written once.
pub fn prefetch(
    ca: &StringChunked,
    options: Option<&CloudOptions>,
    policy: &PathPolicy,
) -> FetchedBatch {
    let remote: Vec<String> = ca
        .iter()
        .flatten()
        .filter(|p| cloud::is_remote_path(p) && policy.check(p).is_ok())
        .map(str::to_string)
        .collect();
    if remote.is_empty() {
        return FetchedBatch::empty();
    }
    FetchedBatch {
        // `read_files_concurrent` dedups internally, so repeated paths across
        // rows are fetched once.
        remote: cloud::read_files_concurrent(&remote, options),
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
    policy: &PathPolicy,
) -> Result<Cow<'a, [u8]>, String> {
    // Every read in the plugin passes through here, so this is the check that
    // makes the policy total: the local branch below has no other gate, and the
    // remote branch can fall back to an inline fetch that `prefetch` never saw.
    policy.check(path)?;
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

/// What an unreadable path does to the query.
///
/// Distinct from the graph's [`RowErrorPolicy`](crate::graph::RowErrorPolicy):
/// this one is settled at *fetch* time, before any graph node runs, and it is
/// the only policy the `read_bytes` expression has — that path has no graph at
/// all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FetchErrorPolicy {
    /// An unreadable path fails the whole query (the default).
    #[default]
    Raise,
    /// An unreadable path yields null for that row only.
    Null,
}

view_buffer::naming::named_variants!(FetchErrorPolicy {
    "raise" => Raise,
    "null" => Null,
});

impl FetchErrorPolicy {
    /// Whether a failure nulls the row rather than failing the query.
    pub fn nulls_the_row(self) -> bool {
        self == FetchErrorPolicy::Null
    }
}

/// Parse an `on_error` setting into "nulls the row on failure".
///
/// Shared by the `file_path` source and the `read_file_bytes` expression so the
/// accepted values and the rejection message cannot drift between them.
/// `context` names what is being configured, e.g. `node 'src'`.
///
/// Reads [`FetchErrorPolicy::NAMED`] rather than matching on string literals,
/// so the values accepted here are exactly the ones `enum_variants` surfaces to
/// Python — the expected-values half of the message included. Spelling them by
/// hand is how the two Python call sites came to carry their own copies of the
/// list.
pub fn parse_on_error(value: &str, context: &str) -> PolarsResult<bool> {
    match view_buffer::naming::lookup(FetchErrorPolicy::NAMED, value) {
        Some(policy) => Ok(policy.nulls_the_row()),
        None => Err(polars_err!(ComputeError:
            "Unknown on_error value '{}' for {} (expected one of {:?})",
            value, context, view_buffer::naming::names(FetchErrorPolicy::NAMED)
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

    fn policy(roots: &[&str]) -> PathPolicy {
        PathPolicy::new(&roots.iter().map(|r| r.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn default_policy_allows_everything() {
        // The default must stay unrestricted: every pipeline that never asks
        // for a sandbox depends on it.
        let p = PathPolicy::default();
        for path in ["/etc/passwd", "relative.png", "s3://bucket/k", "http://h/x"] {
            assert!(p.check(path).is_ok(), "{path}");
        }
    }

    #[test]
    fn remote_prefix_matches_on_a_path_boundary() {
        let p = policy(&["s3://bucket/public"]);
        assert!(p.check("s3://bucket/public/a.png").is_ok());
        assert!(p.check("s3://bucket/public").is_ok(), "the prefix itself");
        // The classic prefix bug: a sibling key that merely starts with the
        // same characters must not be admitted.
        assert!(p.check("s3://bucket/public-evil/a.png").is_err());
        assert!(p.check("s3://bucket/private/a.png").is_err());
        assert!(p.check("s3://other/public/a.png").is_err());
    }

    #[test]
    fn remote_and_local_roots_do_not_cross_admit() {
        // One list, two kinds of entry: a local root must not admit a remote
        // URI that happens to share its text, or vice versa. Splitting these
        // into separate options is how a sandbox comes to cover the disk and
        // leave the network open.
        let p = policy(&["s3://bucket/public"]);
        assert!(p.check("/bucket/public/a.png").is_err());
        let p = policy(&["/srv/images"]);
        assert!(p.check("s3://srv/images/a.png").is_err());
    }

    #[test]
    fn remote_traversal_segments_are_refused() {
        let p = policy(&["https://host/pub"]);
        assert!(p.check("https://host/pub/../secret/x").is_err());
    }

    #[test]
    fn local_paths_are_resolved_before_comparison() {
        let root = std::env::temp_dir().join("polars_cv_policy_root");
        let inside = root.join("inside");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::write(inside.join("a.bin"), b"x").unwrap();

        let p = policy(&[root.to_str().unwrap()]);
        assert!(p.check(inside.join("a.bin").to_str().unwrap()).is_ok());
        // Textual containment is not enough: `..` must be resolved, or the
        // check is defeated by a path that literally contains the root.
        let escape = format!("{}/inside/../../etc/passwd", root.display());
        assert!(p.check(&escape).is_err(), "{escape}");
        // A sibling directory sharing the root's prefix is outside it.
        let sibling = format!("{}-other/a.bin", root.display());
        assert!(p.check(&sibling).is_err(), "{sibling}");
        // A file that does not exist cannot be canonicalized; it is still
        // judged, so a miss inside the root reads as not-found rather than as
        // a policy hole.
        assert!(p
            .check(inside.join("missing.bin").to_str().unwrap())
            .is_ok());
        assert!(p.check("/definitely/not/here.bin").is_err());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_nonexistent_root_denies_rather_than_widens() {
        // A typo'd root must not silently become "allow everything".
        let p = policy(&["/no/such/root/here"]);
        assert!(p.check("/etc/passwd").is_err());
        assert!(p.check("/no/such/root/here/a.png").is_ok());
    }

    #[test]
    fn prefetch_does_not_fetch_a_denied_remote_path() {
        // The point of a sandbox is that the request is never made. A denied
        // path must not reach `read_files_concurrent` at all, so it is absent
        // from the batch rather than present with an error.
        let ca = StringChunked::from_iter_options(
            "paths".into(),
            [Some("s3://blocked/a.png")].into_iter(),
        );
        let batch = prefetch(&ca, None, &policy(&["s3://allowed/"]));
        assert!(batch.remote.is_empty());
    }

    #[test]
    fn row_bytes_refuses_a_denied_path_before_reading() {
        let dir = std::env::temp_dir().join("polars_cv_policy_deny");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("readable.bin");
        std::fs::write(&path, b"payload").unwrap();

        // The file exists and is readable; only the policy stands in the way.
        let batch = FetchedBatch::empty();
        let err = row_bytes(
            &batch,
            path.to_str().unwrap(),
            None,
            &policy(&["/some/other/root"]),
        )
        .unwrap_err();
        assert!(err.contains("is not permitted"), "{err}");
        assert!(
            err.contains("allowed_roots"),
            "message must say what would be accepted: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prefetch_skips_columns_without_remote_paths() {
        let ca = StringChunked::from_iter_options(
            "paths".into(),
            [Some("/tmp/a.png"), None, Some("relative/b.png")].into_iter(),
        );
        let batch = prefetch(&ca, None, &PathPolicy::default());
        assert!(batch.remote.is_empty());
        assert!(prefetch(&empty_string_ca(), None, &PathPolicy::default())
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
        let got = row_bytes(&batch, path.to_str().unwrap(), None, &PathPolicy::default()).unwrap();
        assert_eq!(got.as_ref(), payload.as_slice());

        // The `file://` form resolves to the same file.
        let uri = format!("file://{}", path.to_str().unwrap());
        assert_eq!(
            row_bytes(&batch, &uri, None, &PathPolicy::default())
                .unwrap()
                .as_ref(),
            &payload[..]
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn row_bytes_reports_missing_local_files() {
        let batch = FetchedBatch::empty();
        let err = row_bytes(
            &batch,
            "/nonexistent/polars-cv/missing.png",
            None,
            &PathPolicy::default(),
        )
        .unwrap_err();
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
        let err =
            row_bytes(&batch, "s3://bucket/key.png", None, &PathPolicy::default()).unwrap_err();
        assert!(err.contains("Failed to read remote file"), "{err}");
        assert!(err.contains("access denied"), "{err}");
    }

    #[test]
    fn row_bytes_borrows_prefetched_bytes() {
        let batch = FetchedBatch {
            remote: HashMap::from([("s3://bucket/key.png".to_string(), Ok(vec![1, 2, 3]))]),
        };
        let got = row_bytes(&batch, "s3://bucket/key.png", None, &PathPolicy::default()).unwrap();
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
