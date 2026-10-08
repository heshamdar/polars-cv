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
//! # A window ahead of the rows
//!
//! Fetching is **per plugin call** — one morsel under the streaming engine,
//! which for a Parquet scan is a whole row group, and the whole column under
//! the in-memory engine. A [`Fetcher`] covers one call's path column. When a
//! row asks for its bytes ([`Fetcher::bytes`]), the fetcher starts fetching
//! that row's remote path and those of the next rows up to polars'
//! concurrency budget (`pl_async::get_concurrency_limit`, set by
//! `POLARS_CONCURRENCY_BUDGET`) on polars' `ASYNC` runtime, then waits for its
//! own. Each row thread therefore keeps a window of fetches in flight just
//! ahead of what it decodes. Network time overlaps decoding, and a call holds
//! a window's worth of encoded images rather than all of them. Each distinct
//! path is fetched once per call; its body is freed when the last row naming
//! it has read it. Local paths are read inline per row, one at a time.
//!
//! It used to fetch every remote path of the call before the first row
//! decoded (`tests/test_fetch_window.py`).
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

use std::collections::HashMap;
use std::ops::Deref;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use futures::future::AbortHandle;
use polars::prelude::*;
use pyo3_polars::export::polars_core::runtime::ASYNC;

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
        // The file the read will open ([`cloud::local_file_path`]), not a
        // second reading of the string: a URL it refuses is refused here too.
        let candidate =
            cloud::local_file_path(path).map_err(|e| self.denial(path, &e.to_string()))?;
        let resolved = resolve_best_effort(&candidate);
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

/// One call's fetches for one path column: a window ahead of its rows (see
/// the module docs).
///
/// Built per call; building reads only the column (which rows name which
/// distinct remote path), so it makes no request. Shared by every thread that
/// runs the call's rows. Dropping it aborts the fetches no row waited for.
pub struct Fetcher<'a> {
    ca: &'a StringChunked,
    policy: &'a PathPolicy,
    /// Per row, the index in `entries` of its remote path, when the policy
    /// admits it; `None` for a null, local or refused path. A refused path is
    /// never requested: the point of a sandbox is that the request is not
    /// made, and [`Fetcher::bytes`] reports the refusal for the row.
    slots: Vec<Option<usize>>,
    /// One per distinct remote path.
    entries: Vec<Arc<Entry>>,
    shared: Arc<Shared>,
    /// How many rows past the one being read a read starts fetching: polars'
    /// concurrency budget, which also bounds the requests in flight.
    window: usize,
}

/// What a call's in-flight fetches share with it.
struct Shared {
    options: Option<CloudOptions>,
    /// Fetched bodies held now, and the most held at once.
    resident: AtomicUsize,
    peak: AtomicUsize,
}

/// One distinct remote path of a call.
struct Entry {
    path: String,
    /// Rows that have yet to read this path. The body is freed when the last
    /// one has.
    uses_left: AtomicUsize,
    state: Mutex<State>,
    done: Condvar,
}

enum State {
    /// Not requested yet.
    Idle,
    /// Requested; the handle aborts it if the call ends first.
    Fetching(AbortHandle),
    /// Fetched (or failed), awaiting the rows that read it.
    Done(Result<Arc<Vec<u8>>, String>),
    /// Every row that names it has read it.
    Released,
}

impl Entry {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Start fetching, on polars' runtime, unless already started.
    fn start(self: &Arc<Self>, shared: &Arc<Shared>) {
        let mut state = self.lock();
        if !matches!(*state, State::Idle) {
            return;
        }
        let (entry, shared) = (Arc::clone(self), Arc::clone(shared));
        let (fetch, handle) = futures::future::abortable(async move {
            let result = cloud::read_remote_budgeted(&entry.path, shared.options.as_ref()).await;
            entry.finish(result, &shared);
        });
        *state = State::Fetching(handle);
        drop(state);
        drop(ASYNC.spawn(fetch));
    }

    fn finish(&self, result: Result<Vec<u8>, String>, shared: &Shared) {
        let mut state = self.lock();
        if !matches!(*state, State::Fetching(_)) {
            return;
        }
        if result.is_ok() {
            let now = shared.resident.fetch_add(1, Ordering::SeqCst) + 1;
            shared.peak.fetch_max(now, Ordering::SeqCst);
        }
        *state = State::Done(result.map(Arc::new));
        self.done.notify_all();
    }

    /// The fetched body, once fetched. The entry must have been started.
    fn wait(&self) -> Result<Arc<Vec<u8>>, String> {
        let mut state = self.lock();
        loop {
            match &*state {
                State::Done(result) => return result.clone(),
                State::Fetching(_) => {
                    state = self.done.wait(state).unwrap_or_else(|p| p.into_inner());
                }
                State::Idle | State::Released => {
                    return Err(format!(
                        "internal: '{}' read outside its fetch window",
                        self.path
                    ))
                }
            }
        }
    }

    /// One row has read this path; free the body after the last.
    fn release_one(&self, shared: &Shared) {
        if self.uses_left.fetch_sub(1, Ordering::SeqCst) != 1 {
            return;
        }
        let mut state = self.lock();
        if matches!(*state, State::Done(Ok(_))) {
            shared.resident.fetch_sub(1, Ordering::SeqCst);
        }
        *state = State::Released;
    }
}

impl<'a> Fetcher<'a> {
    /// The fetcher for `ca`, a call's path column.
    ///
    /// `policy` is required rather than optional so a new caller cannot reach
    /// the network by omitting it.
    pub fn new(
        ca: &'a StringChunked,
        options: Option<&CloudOptions>,
        policy: &'a PathPolicy,
    ) -> Self {
        let mut index: HashMap<&str, usize> = HashMap::new();
        let mut entries: Vec<Arc<Entry>> = Vec::new();
        let slots = ca
            .iter()
            .map(|path| {
                let path = path.filter(|p| cloud::is_remote_path(p) && policy.check(p).is_ok())?;
                let slot = *index.entry(path).or_insert_with(|| {
                    entries.push(Arc::new(Entry {
                        path: path.to_string(),
                        uses_left: AtomicUsize::new(0),
                        state: Mutex::new(State::Idle),
                        done: Condvar::new(),
                    }));
                    entries.len() - 1
                });
                entries[slot].uses_left.fetch_add(1, Ordering::Relaxed);
                Some(slot)
            })
            .collect();
        Fetcher {
            ca,
            policy,
            slots,
            entries,
            shared: Arc::new(Shared {
                options: options.cloned(),
                resident: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            }),
            window: polars::io::pl_async::get_concurrency_limit().max(1) as usize,
        }
    }

    /// Row `row`'s bytes: `None` for a null path.
    ///
    /// Every read in the plugin passes through here (or [`Fetcher::header`]),
    /// so the policy check here is what makes the sandbox total. A remote
    /// path's bytes are shared, not copied; a local file is read now.
    pub fn bytes(&self, row: usize) -> Result<Option<Bytes<'_>>, String> {
        let Some(path) = self.ca.get(row) else {
            return Ok(None);
        };
        self.policy.check(path)?;
        if !cloud::is_remote_path(path) {
            // Already known non-remote: read the path `cloud::local_file_path`
            // resolves, as the policy check above judged it. (Routing it
            // through a URL parser would read a bare colon-bearing filename as
            // a bogus cloud URL.)
            return cloud::read_local_path(path)
                .map(|b| Some(Bytes::Local(b)))
                .map_err(|e| format!("Failed to read local file '{path}': {e}"));
        }
        let entry = self.slots[row]
            .map(|slot| &self.entries[slot])
            .ok_or_else(|| format!("internal: remote path '{path}' has no fetch"))?;
        for ahead in row..(row + 1 + self.window).min(self.slots.len()) {
            if let Some(slot) = self.slots[ahead] {
                self.entries[slot].start(&self.shared);
            }
        }
        match entry.wait() {
            Ok(body) => Ok(Some(Bytes::Remote(RemoteBytes {
                body,
                entry,
                shared: &self.shared,
            }))),
            Err(e) => {
                entry.release_one(&self.shared);
                Err(format!("Failed to read remote file '{path}': {e}"))
            }
        }
    }

    /// Read row `row`'s path only as far as `parse` needs: `parse` over the
    /// file's leading bytes, `None` when it cannot tell from them or the path
    /// is null.
    ///
    /// A local file is read in a growing prefix (64 KiB, then four times as
    /// much each time `parse` cannot tell yet) until `parse` answers or the
    /// file ends — an image header is usually in the first few KiB, but a
    /// JPEG's frame header may sit behind large EXIF/comment segments. A
    /// remote file is the whole object ([`Fetcher::bytes`]): ranged reads
    /// would nest the store's concurrency permits (`cloud::read_object`).
    /// The [`PathPolicy`] check applies to both.
    pub fn header<T>(
        &self,
        row: usize,
        parse: impl Fn(&[u8]) -> Option<T>,
    ) -> Result<Option<T>, String> {
        let Some(path) = self.ca.get(row) else {
            return Ok(None);
        };
        if cloud::is_remote_path(path) {
            return self.bytes(row).map(|bytes| bytes.and_then(|b| parse(&b)));
        }
        self.policy.check(path)?;
        let mut limit = 64 * 1024;
        loop {
            let (bytes, eof) = cloud::read_local_prefix(path, limit)
                .map_err(|e| format!("Failed to read local file '{path}': {e}"))?;
            if let Some(found) = parse(&bytes) {
                return Ok(Some(found));
            }
            if eof {
                return Ok(None);
            }
            limit = limit.saturating_mul(4);
        }
    }
}

impl Drop for Fetcher<'_> {
    fn drop(&mut self) {
        for entry in &self.entries {
            if let State::Fetching(handle) = &*entry.lock() {
                handle.abort();
            }
        }
        if !self.entries.is_empty() {
            LAST_PEAK_RESIDENT.store(self.shared.peak.load(Ordering::SeqCst), Ordering::SeqCst);
        }
    }
}

/// One row's bytes: a fetched remote body (shared with the call's other rows
/// naming the same path) or a local file read for this row.
pub enum Bytes<'f> {
    Remote(RemoteBytes<'f>),
    Local(Vec<u8>),
}

/// A fetched body held for one row; dropping it counts the row as read.
pub struct RemoteBytes<'f> {
    body: Arc<Vec<u8>>,
    entry: &'f Arc<Entry>,
    shared: &'f Shared,
}

impl Drop for RemoteBytes<'_> {
    fn drop(&mut self) {
        self.entry.release_one(self.shared);
    }
}

impl Deref for Bytes<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Remote(remote) => remote.body.as_slice(),
            Bytes::Local(bytes) => bytes,
        }
    }
}

/// The most fetched remote bodies one call held at once, for the most recent
/// call that fetched any (`_lib._last_fetch_peak_resident`).
static LAST_PEAK_RESIDENT: AtomicUsize = AtomicUsize::new(0);

/// See [`LAST_PEAK_RESIDENT`].
pub(crate) fn last_peak_resident() -> usize {
    LAST_PEAK_RESIDENT.load(Ordering::SeqCst)
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

view_buffer::naming::named_variants!(FetchErrorPolicy: "What an unreadable path does to the query.\n\nSettled at fetch time, before any graph node runs, which is why it is not\n:class:`RowErrorPolicy`: ``.cv.read_bytes()`` has no graph at all, and\n``source(\"file_path\")`` resolves its bytes before the graph starts.\n- RAISE: an unreadable path fails the whole query.\n- NULL: an unreadable path yields null for that row only." {
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
/// so the values accepted here are exactly the generated Python enum's — the
/// expected-values half of the message included. Spelling them by
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

    fn column(paths: &[Option<&str>]) -> StringChunked {
        StringChunked::from_iter_options("paths".into(), paths.iter().copied())
    }

    /// One path read through a one-row call's fetcher, as every consumer
    /// reads it.
    fn read(path: &str, policy: &PathPolicy) -> Result<Option<Vec<u8>>, String> {
        let ca = column(&[Some(path)]);
        let fetcher = Fetcher::new(&ca, None, policy);
        fetcher.bytes(0).map(|b| b.map(|b| b.to_vec()))
    }

    /// A loopback HTTP server answering every GET with `body`, and how many
    /// requests it has answered.
    fn serve(body: &'static [u8]) -> (String, Arc<AtomicUsize>) {
        use std::io::{BufRead, BufReader};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let hits = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&hits);
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let hits = Arc::clone(&counted);
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut stream = stream;
                    loop {
                        // One request: lines up to the blank one.
                        let mut line = String::new();
                        let mut saw_request = false;
                        loop {
                            line.clear();
                            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                                return;
                            }
                            if line == "\r\n" {
                                break;
                            }
                            saw_request = true;
                        }
                        if !saw_request {
                            return;
                        }
                        hits.fetch_add(1, Ordering::SeqCst);
                        let head =
                            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len());
                        if stream.write_all(head.as_bytes()).is_err()
                            || stream.write_all(body).is_err()
                        {
                            return;
                        }
                    }
                });
            }
        });
        (base, hits)
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

    /// A `file://` URL is judged as the file it reads, by the one resolver
    /// both use: a host component (`file://host/etc/passwd`) once passed the
    /// check as the relative path `host/etc/passwd` — inside a root holding
    /// the working directory — while the read opened `/etc/passwd`.
    #[test]
    fn a_file_url_is_judged_as_the_file_it_reads() {
        let p = policy(&["."]);
        let outside = std::env::temp_dir().join("polars_cv_file_url_escape.bin");
        std::fs::write(&outside, b"secret").unwrap();
        let escape = format!("file://anyhost{}", outside.display());
        assert!(p.check(&escape).is_err(), "{escape}");
        let err = read(&escape, &p).unwrap_err();
        assert!(!err.contains("secret"), "{err}");
        // Unrestricted, a host still names no local file: refused, not guessed.
        assert!(read(&escape, &PathPolicy::default()).is_err());
        std::fs::remove_file(&outside).ok();

        // A percent-encoded file URL is the decoded file, for both.
        let root = std::env::temp_dir().join("polars_cv_file_url_root");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("a b.bin"), b"spaced").unwrap();
        let p = policy(&[root.to_str().unwrap()]);
        let url = format!("file://{}/a%20b.bin", root.display());
        assert!(p.check(&url).is_ok(), "{url}");
        assert_eq!(read(&url, &p).unwrap().unwrap(), b"spaced");
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
    fn a_denied_remote_path_is_never_fetched() {
        // The point of a sandbox is that the request is never made: a denied
        // path gets no fetch at all, and its row reports the refusal.
        let ca = column(&[Some("s3://blocked/a.png")]);
        let p = policy(&["s3://allowed/"]);
        let fetcher = Fetcher::new(&ca, None, &p);
        assert!(fetcher.entries.is_empty());
        let err = fetcher.bytes(0).err().unwrap();
        assert!(err.contains("is not permitted"), "{err}");
    }

    #[test]
    fn a_denied_local_path_is_refused_before_reading() {
        let dir = std::env::temp_dir().join("polars_cv_policy_deny");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("readable.bin");
        std::fs::write(&path, b"payload").unwrap();

        // The file exists and is readable; only the policy stands in the way.
        let err = read(path.to_str().unwrap(), &policy(&["/some/other/root"])).unwrap_err();
        assert!(err.contains("is not permitted"), "{err}");
        assert!(
            err.contains("allowed_roots"),
            "message must say what would be accepted: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn local_and_null_paths_make_no_fetch() {
        let ca = column(&[Some("/tmp/a.png"), None, Some("relative/b.png")]);
        let p = PathPolicy::default();
        assert!(Fetcher::new(&ca, None, &p).entries.is_empty());
        assert!(Fetcher::new(&column(&[]), None, &p).entries.is_empty());
        assert!(Fetcher::new(&ca, None, &p).bytes(1).unwrap().is_none());
    }

    #[test]
    fn local_files_are_read_verbatim() {
        let dir = std::env::temp_dir().join("polars_cv_fetch_local");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bytes.bin");
        // Deliberately not valid image data: fetching must not care.
        let payload: Vec<u8> = (0u8..=255).collect();
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&payload)
            .unwrap();

        let p = PathPolicy::default();
        assert_eq!(read(path.to_str().unwrap(), &p).unwrap().unwrap(), payload);
        // The `file://` form resolves to the same file.
        let uri = format!("file://{}", path.to_str().unwrap());
        assert_eq!(read(&uri, &p).unwrap().unwrap(), payload);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_missing_local_file_is_reported() {
        let err = read("/nonexistent/polars-cv/missing.png", &PathPolicy::default()).unwrap_err();
        assert!(err.contains("Failed to read local file"), "{err}");
        assert!(err.contains("missing.png"), "{err}");
    }

    #[test]
    fn a_failed_fetch_is_reported_at_its_row() {
        // An S3 store refuses a bearer-token command before any request.
        let ca = column(&[Some("s3://bucket/key.png")]);
        let options = CloudOptions::from_map(&HashMap::from([(
            "token_command".to_string(),
            "printf tok".to_string(),
        )]));
        let p = PathPolicy::default();
        let fetcher = Fetcher::new(&ca, Some(&options), &p);
        let err = fetcher.bytes(0).err().unwrap();
        assert!(
            err.contains("Failed to read remote file 's3://bucket/key.png'"),
            "{err}"
        );
        assert!(err.contains("S3 does not"), "{err}");
    }

    /// Rows naming one path share one fetch and one body, which is freed once
    /// the last of them has read it.
    #[test]
    fn rows_naming_one_path_share_one_fetch() {
        let (base, hits) = serve(b"body");
        let (a, b) = (format!("{base}/a.png"), format!("{base}/b.png"));
        let ca = column(&[Some(&a), Some(&b), Some(&a)]);
        let p = PathPolicy::default();
        let fetcher = Fetcher::new(&ca, None, &p);
        assert_eq!(fetcher.entries.len(), 2);
        let first = fetcher.bytes(0).unwrap().unwrap();
        let again = fetcher.bytes(2).unwrap().unwrap();
        assert_eq!(&*first, b"body");
        assert_eq!(
            first.as_ptr(),
            again.as_ptr(),
            "one body, not a copy per row"
        );
        drop((first, again));
        assert_eq!(&*fetcher.bytes(1).unwrap().unwrap(), b"body");
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        assert_eq!(
            fetcher.shared.resident.load(Ordering::SeqCst),
            0,
            "all freed"
        );
    }

    /// A read fetches its own row and the window after it: never more, and
    /// the window does fill.
    #[test]
    fn a_read_fetches_its_window_ahead() {
        let (base, hits) = serve(b"x");
        let window = polars::io::pl_async::get_concurrency_limit() as usize;
        let urls: Vec<String> = (0..window + 8).map(|i| format!("{base}/{i}.png")).collect();
        let ca = column(&urls.iter().map(|u| Some(u.as_str())).collect::<Vec<_>>());
        let p = PathPolicy::default();
        let fetcher = Fetcher::new(&ca, None, &p);
        drop(fetcher.bytes(0).unwrap());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while hits.load(Ordering::SeqCst) < window + 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the window never filled"
            );
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(hits.load(Ordering::SeqCst), window + 1);
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
