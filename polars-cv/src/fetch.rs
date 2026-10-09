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
use std::ops::{Deref, Range};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, OnceLock};

use futures::future::AbortHandle;
use polars::prelude::*;
use pyo3_polars::export::polars_core::runtime::ASYNC;

use crate::cloud::{self, CloudOptions, RangedReply};

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
    /// Per entry, the object read by range ([`Fetcher::open`]).
    remote: Vec<Arc<RemoteObject>>,
    /// The rows may read their files by range ([`Fetcher::ranged`]).
    ranged: bool,
    /// Per row, whether its read-ahead was planned ([`Fetcher::read_ahead`]).
    ahead_planned: Vec<std::sync::atomic::AtomicBool>,
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
        if let Ok(body) = &result {
            count_read(body.len());
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
        let remote = entries
            .iter()
            .map(|e| {
                Arc::new(RemoteObject::new(CloudRanges::new(
                    e.path.clone(),
                    options.cloned(),
                )))
            })
            .collect();
        Fetcher {
            ca,
            policy,
            slots,
            remote,
            ranged: false,
            ahead_planned: (0..ca.len()).map(|_| Default::default()).collect(),
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
                .map(|b| {
                    count_read(b.len());
                    Some(Bytes::Local(b))
                })
                .map_err(|e| format!("Failed to read local file '{path}': {e}"));
        }
        let entry = self.slots[row]
            .map(|slot| &self.entries[slot])
            .ok_or_else(|| format!("internal: remote path '{path}' has no fetch"))?;
        // Rows that may read by range fetch only their own object whole
        // (when it is not one to read by range): fetching ahead would pull
        // whole slides the next rows read only a window of.
        let window = if self.ranged { 0 } else { self.window };
        for ahead in row..(row + 1 + window).min(self.slots.len()) {
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

    /// This fetcher for rows that may read their files by range
    /// ([`Fetcher::open`]): a read of a whole object starts no fetch for the
    /// rows after it.
    pub fn ranged(mut self) -> Self {
        self.ranged = true;
        self
    }

    /// Row `row`'s file opened for reads by range — a local file, or a remote
    /// object through the call's shared [`RemoteObject`] — or `Ok(None)` for
    /// a null path. The policy check and its refusal are [`Fetcher::bytes`]'s.
    pub fn open(&self, row: usize) -> Result<Option<RangedFile>, String> {
        let Some(path) = self.ca.get(row) else {
            return Ok(None);
        };
        if !cloud::is_remote_path(path) {
            return Ok(self.local_file(row)?.map(RangedFile::Local));
        }
        self.policy.check(path)?;
        let slot = self.slots[row]
            .ok_or_else(|| format!("internal: remote path '{path}' has no fetch"))?;
        Ok(Some(RangedFile::Remote(RemoteReader::for_row(
            Arc::clone(&self.remote[slot]),
            row,
        ))))
    }

    /// The rows after `row` whose windows a row thread reads ahead: remote
    /// rows not yet planned, as far ahead as the window `bytes` keeps in
    /// flight (polars' concurrency budget, which also bounds the requests).
    pub fn rows_ahead(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        (row + 1..(row + 1 + self.window).min(self.slots.len()))
            .filter(|&r| self.slots[r].is_some() && !self.ahead_planned[r].load(Ordering::Relaxed))
    }

    /// Start reading row `row`'s window ahead of the row: its remote TIFF's
    /// header (from the call's shared structure blocks), then the chunks
    /// `crop` takes of level `level`, on a background task the row's own
    /// read then takes ([`RemoteReader::for_row`]). Once per row; nothing
    /// for a local or null path, a refused path, a file that is not a TIFF
    /// the chunk decoder carries, or a window the crop refuses — the row
    /// itself reads, and reports, those.
    pub fn read_ahead(&self, row: usize, level: u32, crop: &view_buffer::ViewOp) {
        if self.ahead_planned[row].swap(true, Ordering::Relaxed) {
            return;
        }
        let (Some(slot), Some(path)) = (self.slots[row], self.ca.get(row)) else {
            return;
        };
        if self.policy.check(path).is_err() {
            return;
        }
        let object = &self.remote[slot];
        let Some(cell) = object.claim_ahead(row) else {
            return;
        };
        let plan = || {
            let mut reader = RemoteReader::new(Arc::clone(object));
            let mut image = view_buffer::ImageAdapter::open_tiff(&mut reader, level).ok()?;
            let (shape, dtype) = image.shape()?;
            let window = view_buffer::ImageAdapter::tiff_window(crop, shape, dtype).ok()?;
            image.chunk_ranges(Some(window)).ok()?
        };
        object.read_ahead(cell, plan());
    }

    /// Row `row`'s file opened for reads by range, when its path is local:
    /// `Ok(None)` for a null or remote path (read those with
    /// [`Fetcher::bytes`]). The policy check and the error text are
    /// [`Fetcher::bytes`]'s.
    pub fn local_file(&self, row: usize) -> Result<Option<LocalFile>, String> {
        let Some(path) = self.ca.get(row) else {
            return Ok(None);
        };
        if cloud::is_remote_path(path) {
            return Ok(None);
        }
        self.policy.check(path)?;
        let file = cloud::local_file_path(path)
            .map_err(|e| e.to_string())
            .and_then(|p| std::fs::File::open(p).map_err(|e| e.to_string()))
            .map_err(|e| format!("Failed to read local file '{path}': {e}"))?;
        Ok(Some(LocalFile {
            reader: std::io::BufReader::with_capacity(LOCAL_BUFFER, file),
        }))
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
            count_read(bytes.len());
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

/// Every byte the plugin's path reads have taken from a file or a store, ever
/// (`_lib._fetch_bytes_read`): whole reads, header prefixes and ranged reads.
/// Cumulative, so a test reads it before and after a query and concurrent
/// calls cannot reset it under it.
static BYTES_READ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Count `n` bytes read from a file or store ([`BYTES_READ`]).
fn count_read(n: usize) {
    BYTES_READ.fetch_add(n as u64, Ordering::Relaxed);
}

/// See [`BYTES_READ`].
pub(crate) fn bytes_read() -> u64 {
    BYTES_READ.load(Ordering::Relaxed)
}

/// A file opened for reads by range, counting what it reads ([`BYTES_READ`]).
pub struct LocalFile {
    reader: std::io::BufReader<std::fs::File>,
}

/// Small buffers: a ranged read seeks between an IFD, its tag values and a
/// few chunks, and each seek refills the buffer.
const LOCAL_BUFFER: usize = 4 * 1024;

impl std::io::Read for LocalFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(buf)?;
        count_read(n);
        Ok(n)
    }
}

impl std::io::Seek for LocalFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        self.reader.seek(pos)
    }
}

impl view_buffer::interop::tiff_region::TiffSource for LocalFile {
    fn prefetch(&mut self, _: &[std::ops::Range<u64>]) -> std::io::Result<()> {
        Ok(())
    }
}

pub use crate::cloud::Head;

/// Where a [`RemoteObject`]'s bytes come from: a store answering head and
/// byte-range requests. [`CloudRanges`] in the plugin; a counting in-memory
/// store in the tests.
pub trait RangeStore: Send + Sync + 'static {
    /// Which object this is, across calls: its path and the options that
    /// reach it.
    fn identity(&self) -> String;
    /// The object's size and version.
    fn head(&self) -> Result<Head, String>;
    /// The bytes of each of `ranges`, in one operation.
    fn read(&self, ranges: &[Range<u64>]) -> Result<Vec<Vec<u8>>, String>;
}

/// A remote path's bytes by range, through `cloud` on polars' runtime.
pub struct CloudRanges {
    path: String,
    options: Option<CloudOptions>,
    /// The whole object, once a server ignoring ranges has sent it: every
    /// later size and range is answered from it, so such a server is read
    /// once per call rather than once per request.
    whole: OnceLock<Vec<u8>>,
}

impl CloudRanges {
    fn new(path: String, options: Option<CloudOptions>) -> Self {
        CloudRanges {
            path,
            options,
            whole: OnceLock::new(),
        }
    }

    /// The whole object a server sent, kept, and counted as read once.
    fn keep_whole(&self, body: Vec<u8>) -> &Vec<u8> {
        let n = body.len();
        let mut kept = false;
        let whole = self.whole.get_or_init(|| {
            kept = true;
            body
        });
        if kept {
            count_read(n);
        }
        whole
    }
}

/// Run `future` on polars' `ASYNC` runtime and wait for it on this (row)
/// thread, as the window's fetches are waited for.
fn block_on<T: Send + 'static>(future: impl std::future::Future<Output = T> + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    drop(ASYNC.spawn(async move {
        let _ = tx.send(future.await);
    }));
    rx.recv().expect("a remote read ended without answering")
}

impl RangeStore for CloudRanges {
    fn identity(&self) -> String {
        use std::hash::{Hash, Hasher};
        // Credentials enter only as a hash, and only in memory.
        let mut h = std::collections::hash_map::DefaultHasher::new();
        if let Some(o) = &self.options {
            let mut config: Vec<_> = o.config.iter().collect();
            config.sort();
            (config, &o.bearer_token, &o.token_command, o.anonymous).hash(&mut h);
        }
        format!("{}\u{0}{:016x}", self.path, h.finish())
    }

    fn head(&self) -> Result<Head, String> {
        if let Some(whole) = self.whole.get() {
            return Ok(Head {
                size: whole.len() as u64,
                version: None,
            });
        }
        let (path, options) = (self.path.clone(), self.options.clone());
        match block_on(async move { cloud::remote_head_budgeted(&path, options.as_ref()).await })? {
            RangedReply::Asked(head) => Ok(head),
            RangedReply::Whole(body) => Ok(Head {
                size: self.keep_whole(body).len() as u64,
                version: None,
            }),
        }
    }

    fn read(&self, ranges: &[Range<u64>]) -> Result<Vec<Vec<u8>>, String> {
        if let Some(whole) = self.whole.get() {
            return cloud::cut_ranges(whole, ranges, &self.path);
        }
        let (path, options, wanted) = (self.path.clone(), self.options.clone(), ranges.to_vec());
        let reply = block_on(async move {
            cloud::read_remote_ranges_budgeted(&path, options.as_ref(), &wanted).await
        })?;
        match reply {
            RangedReply::Asked(parts) => {
                count_read(parts.iter().map(Vec::len).sum());
                Ok(parts)
            }
            RangedReply::Whole(body) => {
                cloud::cut_ranges(self.keep_whole(body), ranges, &self.path)
            }
        }
    }
}

/// The block size a [`RemoteReader`] reads a file's structure in: one
/// request brings a TIFF's header, first IFD and small tag values together.
const REMOTE_BLOCK: u64 = 64 * 1024;

/// Ranges no further apart than this are fetched as one: a request costs
/// more than the gap's bytes.
const COALESCE_GAP: u64 = 16 * 1024;

/// Ranges `ranges`, sorted, with those no further apart than
/// [`COALESCE_GAP`] merged: what one request fetches.
fn coalesce(ranges: impl IntoIterator<Item = Range<u64>>) -> Vec<Range<u64>> {
    let mut wanted: Vec<Range<u64>> = ranges.into_iter().filter(|r| !r.is_empty()).collect();
    wanted.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<u64>> = Vec::new();
    for r in wanted {
        match merged.last_mut() {
            Some(last) if r.start <= last.end + COALESCE_GAP => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    merged
}

type BlockCell = OnceLock<Result<Arc<Vec<u8>>, String>>;

/// The structure blocks read of one version of an object: shared by every
/// row of a call naming it, and by later calls while the version holds.
#[derive(Default)]
struct Structure {
    /// Each block's one fetch: rows wanting a block another row is fetching
    /// wait for it rather than request it again.
    blocks: Mutex<HashMap<u64, Arc<BlockCell>>>,
    /// The bytes its blocks hold, for the cache's budget.
    bytes: AtomicUsize,
}

/// The most structure bytes kept across calls, over every object: a slide's
/// IFDs and the table blocks its patches touched, a few MiB at most.
const STRUCTURE_CACHE_BYTES: usize = 256 << 20;

/// Structure blocks kept across calls, by object identity, each with the
/// [`Head`] it was read under: a call reuses an object's blocks only when its
/// own `HEAD` gives the same size and version, and an object without a
/// version is never kept. Least recently used objects go first.
#[derive(Default)]
struct StructureCache {
    entries: HashMap<String, (Head, Arc<Structure>, u64)>,
    clock: u64,
}

static STRUCTURES: LazyLock<Mutex<StructureCache>> = LazyLock::new(Default::default);

/// The structure of `identity` at `head`: kept from an earlier call when the
/// version matches, else new (and kept, when it has a version).
fn structure_for(identity: String, head: &Head) -> Arc<Structure> {
    if head.version.is_none() {
        return Arc::default();
    }
    let mut cache = STRUCTURES.lock().unwrap_or_else(|p| p.into_inner());
    cache.clock += 1;
    let now = cache.clock;
    if let Some((kept, structure, used)) = cache.entries.get_mut(&identity) {
        if kept == head {
            *used = now;
            return Arc::clone(structure);
        }
    }
    let structure = Arc::<Structure>::default();
    cache.entries.insert(
        identity.clone(),
        (head.clone(), Arc::clone(&structure), now),
    );
    let held = |c: &StructureCache| -> usize {
        c.entries
            .values()
            .map(|(_, s, _)| s.bytes.load(Ordering::Relaxed))
            .sum()
    };
    while held(&cache) > STRUCTURE_CACHE_BYTES {
        let oldest = cache
            .entries
            .iter()
            .filter(|(k, _)| **k != identity)
            .min_by_key(|(_, (_, _, used))| *used)
            .map(|(k, _)| k.clone());
        match oldest {
            Some(k) => drop(cache.entries.remove(&k)),
            None => break,
        }
    }
    structure
}

/// Fetched ranges and their bytes, as one request returns them.
type Fetched = Vec<(Range<u64>, Vec<u8>)>;

/// A row's chunk fetch, started ahead of the row ([`RemoteObject::read_ahead`]).
#[derive(Default)]
struct Ahead {
    result: Mutex<Option<Result<Fetched, String>>>,
    done: Condvar,
}

impl Ahead {
    fn finish(&self, result: Result<Fetched, String>) {
        *self.result.lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
        self.done.notify_all();
    }

    /// The fetch's result, once finished.
    fn wait(&self) -> Result<Fetched, String> {
        let mut result = self.result.lock().unwrap_or_else(|p| p.into_inner());
        loop {
            if let Some(done) = result.take() {
                return done;
            }
            result = self.done.wait(result).unwrap_or_else(|p| p.into_inner());
        }
    }
}

/// Who reads a row's chunks: a planner that claimed them ahead of the row,
/// or the row itself, which began before any planner reached it.
enum AheadSlot {
    Claimed(Arc<Ahead>),
    Reading,
}

/// The most bytes one row's read-ahead fetches: a patch's chunks, not a
/// whole level.
const AHEAD_ROW_BYTES: u64 = 16 << 20;

/// One remote object of a call, read by range: its head, once asked; the
/// blocks structure reads have fetched ([`Structure`], shared by every row
/// naming it and kept across calls while its version holds); and the chunk
/// fetches started ahead of the rows that will read them.
pub struct RemoteObject<S: RangeStore = CloudRanges> {
    store: S,
    head: OnceLock<Result<Head, String>>,
    structure: OnceLock<Arc<Structure>>,
    /// Per row, who reads its chunks: decided once, under this lock, so a
    /// row is fetched by a planner or by itself, never both.
    ahead: Mutex<HashMap<usize, AheadSlot>>,
}

impl<S: RangeStore> RemoteObject<S> {
    pub fn new(store: S) -> Self {
        RemoteObject {
            store,
            head: OnceLock::new(),
            structure: OnceLock::new(),
            ahead: Mutex::new(HashMap::new()),
        }
    }

    fn head(&self) -> Result<Head, String> {
        self.head.get_or_init(|| self.store.head()).clone()
    }

    fn size(&self) -> Result<u64, String> {
        self.head().map(|h| h.size)
    }

    fn structure(&self) -> Result<&Arc<Structure>, String> {
        let head = self.head()?;
        Ok(self
            .structure
            .get_or_init(|| structure_for(self.store.identity(), &head)))
    }

    /// Block `index`, fetched once per version of the object.
    fn block(&self, index: u64) -> Result<Arc<Vec<u8>>, String> {
        let structure = self.structure()?;
        let cell = Arc::clone(
            structure
                .blocks
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .entry(index)
                .or_default(),
        );
        cell.get_or_init(|| {
            let start = index * REMOTE_BLOCK;
            let end = (start + REMOTE_BLOCK).min(self.size()?);
            let mut parts = self.store.read(std::slice::from_ref(&(start..end)))?;
            let block = parts.pop().unwrap_or_default();
            structure.bytes.fetch_add(block.len(), Ordering::Relaxed);
            Ok(Arc::new(block))
        })
        .clone()
    }

    /// Claim row `row`'s read-ahead: its cell, registered before the row's
    /// window is even planned, so the row waits for its planner rather than
    /// fetching the same chunks itself. `None` when the row was claimed or
    /// has begun reading for itself.
    fn claim_ahead(&self, row: usize) -> Option<Arc<Ahead>> {
        let mut ahead = self.ahead.lock().unwrap_or_else(|p| p.into_inner());
        if ahead.contains_key(&row) {
            return None;
        }
        let cell = Arc::<Ahead>::default();
        ahead.insert(row, AheadSlot::Claimed(Arc::clone(&cell)));
        Some(cell)
    }

    /// Row `row` begins reading: its read-ahead, when a planner claimed it
    /// first; else the row is marked as reading for itself, so no planner
    /// fetches it after.
    fn begin_row(&self, row: usize) -> Option<Arc<Ahead>> {
        let mut ahead = self.ahead.lock().unwrap_or_else(|p| p.into_inner());
        match ahead.insert(row, AheadSlot::Reading) {
            Some(AheadSlot::Claimed(cell)) => Some(cell),
            Some(AheadSlot::Reading) | None => None,
        }
    }

    /// Fetch `ranges` (a claimed row's chunks, coalesced as its own prefetch
    /// would) on a blocking task into `cell`; `None` (nothing planned) or a
    /// fetch over [`AHEAD_ROW_BYTES`] leaves the row to fetch for itself.
    fn read_ahead(self: &Arc<Self>, cell: Arc<Ahead>, ranges: Option<Vec<Range<u64>>>) {
        let merged = ranges.map(coalesce).unwrap_or_default();
        let total: u64 = merged.iter().map(|r| r.end - r.start).sum();
        if merged.is_empty() || total > AHEAD_ROW_BYTES {
            cell.finish(Ok(Vec::new()));
            return;
        }
        let object = Arc::clone(self);
        drop(ASYNC.spawn_blocking(move || {
            let result = object
                .store
                .read(&merged)
                .map(|parts| merged.into_iter().zip(parts).collect());
            cell.finish(result);
        }));
    }
}

/// One row's view of a [`RemoteObject`]: `Read + Seek` over it for the TIFF
/// decoder. Structure reads come from the object's shared blocks; the
/// chunks a decode prefetches come from the row's read-ahead when one was
/// started, else in one coalesced request, and are kept for this row only.
pub struct RemoteReader<S: RangeStore = CloudRanges> {
    object: Arc<RemoteObject<S>>,
    pos: u64,
    prefetched: Fetched,
    /// The row's read-ahead, which its first prefetch takes.
    ahead: Option<Arc<Ahead>>,
}

impl<S: RangeStore> RemoteReader<S> {
    pub fn new(object: Arc<RemoteObject<S>>) -> Self {
        RemoteReader {
            object,
            pos: 0,
            prefetched: Vec::new(),
            ahead: None,
        }
    }

    /// Row `row`'s reader: the row begins reading
    /// ([`RemoteObject::begin_row`]), and its first prefetch takes the row's
    /// read-ahead if a planner claimed it first.
    pub fn for_row(object: Arc<RemoteObject<S>>, row: usize) -> Self {
        let ahead = object.begin_row(row);
        RemoteReader {
            ahead,
            ..Self::new(object)
        }
    }
}

fn io_error(e: String) -> std::io::Error {
    std::io::Error::other(e)
}

impl<S: RangeStore> std::io::Read for RemoteReader<S> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let pos = self.pos;
        let served =
            if let Some((range, data)) = self.prefetched.iter().find(|(r, _)| r.contains(&pos)) {
                let from = (pos - range.start) as usize;
                let n = buf.len().min(data.len() - from);
                buf[..n].copy_from_slice(&data[from..from + n]);
                n
            } else {
                if pos >= self.object.size().map_err(io_error)? {
                    return Ok(0);
                }
                let block = self.object.block(pos / REMOTE_BLOCK).map_err(io_error)?;
                let from = (pos % REMOTE_BLOCK) as usize;
                let n = buf.len().min(block.len().saturating_sub(from));
                buf[..n].copy_from_slice(&block[from..from + n]);
                n
            };
        self.pos += served as u64;
        Ok(served)
    }
}

impl<S: RangeStore> std::io::Seek for RemoteReader<S> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        let target = match pos {
            std::io::SeekFrom::Start(n) => Some(n),
            std::io::SeekFrom::Current(d) => self.pos.checked_add_signed(d),
            std::io::SeekFrom::End(d) => {
                self.object.size().map_err(io_error)?.checked_add_signed(d)
            }
        };
        self.pos = target.ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "seek before the start")
        })?;
        Ok(self.pos)
    }
}

impl<S: RangeStore> view_buffer::interop::tiff_region::TiffSource for RemoteReader<S> {
    /// Bring in `ranges` (a decode's chunks): from the row's read-ahead
    /// when one was started, the rest in one request, ranges close together
    /// merged.
    fn prefetch(&mut self, ranges: &[Range<u64>]) -> std::io::Result<()> {
        if let Some(ahead) = self.ahead.take() {
            // A failed read-ahead is left for the row's own request to
            // report.
            if let Ok(parts) = ahead.wait() {
                self.prefetched.extend(parts);
            }
        }
        let merged = coalesce(
            ranges
                .iter()
                .filter(|r| {
                    !self
                        .prefetched
                        .iter()
                        .any(|(p, _)| p.start <= r.start && r.end <= p.end)
                })
                .cloned(),
        );
        if merged.is_empty() {
            return Ok(());
        }
        let parts = self.object.store.read(&merged).map_err(io_error)?;
        self.prefetched.extend(merged.into_iter().zip(parts));
        Ok(())
    }
}

/// A path's file opened for reads by range: local, or remote through a
/// call's shared [`RemoteObject`].
pub enum RangedFile {
    Local(LocalFile),
    Remote(RemoteReader),
}

impl std::io::Read for RangedFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            RangedFile::Local(f) => f.read(buf),
            RangedFile::Remote(r) => r.read(buf),
        }
    }
}

impl std::io::Seek for RangedFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match self {
            RangedFile::Local(f) => f.seek(pos),
            RangedFile::Remote(r) => r.seek(pos),
        }
    }
}

impl view_buffer::interop::tiff_region::TiffSource for RangedFile {
    fn prefetch(&mut self, ranges: &[Range<u64>]) -> std::io::Result<()> {
        match self {
            RangedFile::Local(f) => f.prefetch(ranges),
            RangedFile::Remote(r) => r.prefetch(ranges),
        }
    }
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

    /// An object in memory, recording every request a reader makes of it.
    struct MemStore {
        data: Vec<u8>,
        reads: Mutex<Vec<Vec<Range<u64>>>>,
        name: String,
        version: Option<String>,
    }

    impl MemStore {
        /// An object with no version: never kept across calls.
        fn new(data: Vec<u8>) -> Self {
            MemStore {
                data,
                reads: Mutex::new(Vec::new()),
                name: "mem".to_string(),
                version: None,
            }
        }

        /// Object `name` at `version`.
        fn versioned(data: Vec<u8>, name: &str, version: Option<&str>) -> Self {
            MemStore {
                name: name.to_string(),
                version: version.map(str::to_string),
                ..Self::new(data)
            }
        }

        fn reads(&self) -> Vec<Vec<Range<u64>>> {
            self.reads.lock().unwrap().clone()
        }

        fn bytes_read(&self) -> u64 {
            self.reads().iter().flatten().map(|r| r.end - r.start).sum()
        }
    }

    impl RangeStore for MemStore {
        fn identity(&self) -> String {
            self.name.clone()
        }

        fn head(&self) -> Result<Head, String> {
            Ok(Head {
                size: self.data.len() as u64,
                version: self.version.clone(),
            })
        }

        fn read(&self, ranges: &[Range<u64>]) -> Result<Vec<Vec<u8>>, String> {
            self.reads.lock().unwrap().push(ranges.to_vec());
            Ok(ranges
                .iter()
                .map(|r| self.data[r.start as usize..r.end as usize].to_vec())
                .collect())
        }
    }

    /// An uncompressed 8-bit gray tiled TIFF, `side`x`side` in `tile`-pixel
    /// tiles, built by hand (the `tiff` crate's encoder writes strips only),
    /// and its pixels.
    fn tiled_gray_tiff(side: u32, tile: u32) -> (Vec<u8>, Vec<u8>) {
        let pixel = |y: u32, x: u32| ((y * 7 + x * 3) % 251) as u8;
        let across = side.div_ceil(tile);
        let mut file = b"II*\0\0\0\0\0".to_vec();
        let mut offsets = Vec::new();
        for ty in 0..across {
            for tx in 0..across {
                offsets.push(file.len() as u32);
                for y in 0..tile {
                    for x in 0..tile {
                        file.push(pixel(ty * tile + y, tx * tile + x));
                    }
                }
            }
        }
        let array = |file: &mut Vec<u8>, values: &[u32]| {
            let at = file.len() as u32;
            for v in values {
                file.extend_from_slice(&v.to_le_bytes());
            }
            at
        };
        let counts = vec![tile * tile; offsets.len()];
        let (offsets_at, counts_at) = (array(&mut file, &offsets), array(&mut file, &counts));
        let ifd = file.len() as u32;
        file[4..8].copy_from_slice(&ifd.to_le_bytes());
        // (tag, type, count, value): type 3 SHORT, 4 LONG.
        let n = offsets.len() as u32;
        let entries: [(u16, u16, u32, u32); 10] = [
            (256, 4, 1, side),
            (257, 4, 1, side),
            (258, 3, 1, 8),
            (259, 3, 1, 1),
            (262, 3, 1, 1),
            (277, 3, 1, 1),
            (322, 4, 1, tile),
            (323, 4, 1, tile),
            (324, 4, n, offsets_at),
            (325, 4, n, counts_at),
        ];
        file.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (tag, kind, count, value) in entries {
            file.extend_from_slice(&tag.to_le_bytes());
            file.extend_from_slice(&kind.to_le_bytes());
            file.extend_from_slice(&count.to_le_bytes());
            file.extend_from_slice(&value.to_le_bytes());
        }
        file.extend_from_slice(&0u32.to_le_bytes());
        let pixels = (0..side)
            .flat_map(|y| (0..side).map(move |x| pixel(y, x)))
            .collect();
        (file, pixels)
    }

    #[test]
    fn a_remote_reader_reads_the_object_at_any_position() {
        use std::io::{Read, Seek, SeekFrom};
        let data: Vec<u8> = (0..300_000u32).map(|i| (i % 253) as u8).collect();
        let object = Arc::new(RemoteObject::new(MemStore::new(data.clone())));
        let mut reader = RemoteReader::new(Arc::clone(&object));
        for (at, len) in [
            (0, 10),
            (65_530, 20),
            (131_071, 1),
            (299_990, 50),
            (150_000, 70_000),
        ] {
            reader.seek(SeekFrom::Start(at)).unwrap();
            let mut got = Vec::new();
            reader.by_ref().take(len).read_to_end(&mut got).unwrap();
            let end = (at + len).min(data.len() as u64) as usize;
            assert_eq!(got, data[at as usize..end], "at {at}");
        }
        assert_eq!(reader.seek(SeekFrom::End(-1)).unwrap(), 299_999);
    }

    #[test]
    fn rows_share_the_blocks_of_an_objects_structure() {
        use std::io::Read;
        let object = Arc::new(RemoteObject::new(MemStore::new(vec![7u8; 200_000])));
        for _ in 0..3 {
            let mut header = [0u8; 100];
            RemoteReader::new(Arc::clone(&object))
                .read_exact(&mut header)
                .unwrap();
        }
        let first_block: Range<u64> = 0..REMOTE_BLOCK;
        assert_eq!(object.store.reads(), vec![vec![first_block]]);
    }

    #[test]
    fn a_prefetch_is_one_request_of_merged_ranges_and_serves_its_reads() {
        use std::io::{Read, Seek, SeekFrom};
        use view_buffer::interop::tiff_region::TiffSource;
        let data: Vec<u8> = (0..1_000_000u32).map(|i| (i % 249) as u8).collect();
        let object = Arc::new(RemoteObject::new(MemStore::new(data.clone())));
        let mut reader = RemoteReader::new(Arc::clone(&object));
        reader
            .prefetch(&[500_000..500_100, 0..10, 500_200..500_300, 900_000..900_010])
            .unwrap();
        assert_eq!(
            object.store.reads(),
            vec![vec![0..10, 500_000..500_300, 900_000..900_010]],
            "one request; ranges close together merged"
        );
        reader.seek(SeekFrom::Start(500_210)).unwrap();
        let mut got = [0u8; 20];
        reader.read_exact(&mut got).unwrap();
        assert_eq!(got, data[500_210..500_230]);
        assert_eq!(
            object.store.reads().len(),
            1,
            "a prefetched read makes no request"
        );
    }

    /// A later call's object reuses the structure blocks an earlier call
    /// read, only while the identity and version both match.
    #[test]
    fn a_later_call_reuses_structure_only_at_the_same_version() {
        use std::io::Read;
        let data = vec![9u8; 100_000];
        let name = "a_later_call_reuses_structure_only_at_the_same_version";
        let header = |store: MemStore| -> usize {
            let object = Arc::new(RemoteObject::new(store));
            let mut buf = [0u8; 16];
            RemoteReader::new(Arc::clone(&object))
                .read_exact(&mut buf)
                .unwrap();
            object.store.reads().len()
        };
        assert_eq!(
            header(MemStore::versioned(data.clone(), name, Some("v1"))),
            1
        );
        assert_eq!(
            header(MemStore::versioned(data.clone(), name, Some("v1"))),
            0,
            "same version: the block is kept"
        );
        assert_eq!(
            header(MemStore::versioned(data.clone(), name, Some("v2"))),
            1,
            "a new version is read afresh"
        );
        let unversioned = format!("{name}-unversioned");
        assert_eq!(
            header(MemStore::versioned(data.clone(), &unversioned, None)),
            1
        );
        assert_eq!(
            header(MemStore::versioned(data, &unversioned, None)),
            1,
            "no version: never kept"
        );
    }

    /// A row's read-ahead is the one request for its chunks: the row's own
    /// prefetch takes it, and reads the bytes it fetched.
    #[test]
    fn a_rows_prefetch_takes_its_read_ahead() {
        use std::io::{Read, Seek, SeekFrom};
        use view_buffer::interop::tiff_region::TiffSource;
        let data: Vec<u8> = (0..1_000_000u32).map(|i| (i % 241) as u8).collect();
        let object = Arc::new(RemoteObject::new(MemStore::new(data.clone())));
        let ranges = vec![500_000..500_100, 500_200..500_300, 900_000..900_010];
        let cell = object.claim_ahead(7).expect("unclaimed");
        assert!(object.claim_ahead(7).is_none(), "claimed once");
        object.read_ahead(cell, Some(ranges.clone()));
        let mut reader = RemoteReader::for_row(Arc::clone(&object), 7);
        reader.prefetch(&ranges).unwrap();
        assert_eq!(
            object.store.reads(),
            vec![vec![500_000..500_300, 900_000..900_010]],
            "one request, made ahead"
        );
        reader.seek(SeekFrom::Start(900_002)).unwrap();
        let mut got = [0u8; 8];
        reader.read_exact(&mut got).unwrap();
        assert_eq!(got, data[900_002..900_010]);
        // A row that began before any planner reached it reads for itself,
        // and is never claimed after.
        let mut other = RemoteReader::for_row(Arc::clone(&object), 8);
        assert!(object.claim_ahead(8).is_none(), "already reading");
        other.prefetch(std::slice::from_ref(&(10..20))).unwrap();
        assert_eq!(object.store.reads().len(), 2);
    }

    /// The point of it all: a window of a remote tiled TIFF decodes from its
    /// header and the tiles under it — a small fraction of the object.
    #[test]
    fn a_window_of_a_remote_tiff_reads_its_tiles_not_the_file() {
        use view_buffer::ops::ViewOp;
        // Large enough that the 64 KiB structure blocks are the small part,
        // as they are for a real slide.
        let (file, pixels) = tiled_gray_tiff(2048, 64);
        let size = file.len() as u64;
        let object = Arc::new(RemoteObject::new(MemStore::new(file)));
        let crop = ViewOp::Crop {
            top: 100,
            left: 200,
            height: Some(20),
            width: Some(30),
        };
        use view_buffer::interop::image::{RegionDecode, TiffRegion};
        let mut image =
            view_buffer::ImageAdapter::open_tiff(RemoteReader::new(Arc::clone(&object)), 0)
                .unwrap();
        let got = match view_buffer::ImageAdapter::decode_tiff_region(&mut image, Some(&crop)) {
            Ok(TiffRegion::Decoded(RegionDecode::Window(buffer))) => buffer.to_contiguous(),
            other => panic!("an uncompressed gray TIFF is the chunk decoder's: {other:?}"),
        };
        assert_eq!(got.shape(), &[20, 30, 1]);
        let want: Vec<u8> = (100..120)
            .flat_map(|y| pixels[y * 2048 + 200..y * 2048 + 230].iter().copied())
            .collect();
        assert_eq!(got.as_slice::<u8>(), want.as_slice());
        let read = object.store.bytes_read();
        assert!(read * 20 < size, "read {read} of {size} bytes");
    }

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
