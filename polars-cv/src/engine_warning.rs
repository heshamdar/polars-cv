//! One-time "running single-threaded" warning for the graph plugin.
//!
//! The plugin does not parallelize *within* a call — all multi-core execution
//! comes from the Polars **streaming** engine slicing the input into morsels and
//! invoking the plugin concurrently. Under the default in-memory engine a single
//! `collect` therefore runs the whole column on one thread, and nothing signals
//! it. This module emits a single, actionable warning when it observes a large
//! amount of work go through without the engine ever running two calls
//! concurrently.
//!
//! Detection is **concurrency-based**, not a fragile row-count or thread-name
//! heuristic: a [`CallGuard`] tracks how many plugin calls are in flight at once.
//! If the observed maximum ever reaches 2, the engine is parallelizing and the
//! warning is suppressed forever. Only when a configurable number of rows has
//! been processed with the observed concurrency still stuck at 1 (on a machine
//! that actually has spare cores) do we warn — once.
//!
//! Escape hatches:
//! - `POLARS_CV_SILENCE_ENGINE_WARNING=1` — never warn.
//! - `POLARS_CV_ENGINE_WARN_ROWS=<n>` — override the row threshold.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Plugin calls currently executing (RAII-tracked by [`CallGuard`]).
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
/// Maximum number of plugin calls ever seen executing at the same instant.
static MAX_CONCURRENCY: AtomicUsize = AtomicUsize::new(0);
/// Total rows processed across all calls so far.
static CUMULATIVE_ROWS: AtomicU64 = AtomicU64::new(0);
/// Whether the one-time warning has already fired.
static WARNED: AtomicBool = AtomicBool::new(false);

/// Rows processed single-threaded before the warning fires. Chosen well above a
/// typical streaming morsel so that, under streaming, several morsels run (and
/// bump the observed concurrency past 1) long before this is reached.
const DEFAULT_WARN_ROWS: u64 = 50_000;

/// RAII guard: bump the in-flight counter for the duration of one plugin call so
/// concurrent calls are actually observed, and check the warning condition on
/// entry.
pub struct CallGuard;

impl CallGuard {
    /// Enter a plugin call processing `n_rows` rows.
    pub fn enter(n_rows: usize) -> Self {
        let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
        // Record the high-water mark of concurrent calls.
        MAX_CONCURRENCY.fetch_max(now, Ordering::SeqCst);
        CUMULATIVE_ROWS.fetch_add(n_rows as u64, Ordering::SeqCst);
        maybe_warn();
        CallGuard
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Available parallelism, honoring `POLARS_MAX_THREADS` when set.
fn available_parallelism() -> usize {
    if let Ok(v) = std::env::var("POLARS_MAX_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            return n;
        }
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn warn_row_threshold() -> u64 {
    std::env::var("POLARS_CV_ENGINE_WARN_ROWS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_WARN_ROWS)
}

fn maybe_warn() {
    if WARNED.load(Ordering::Relaxed) {
        return;
    }
    if std::env::var("POLARS_CV_SILENCE_ENGINE_WARNING").is_ok() {
        return;
    }
    // The engine has parallelized across morsels at least once — no footgun.
    if MAX_CONCURRENCY.load(Ordering::SeqCst) >= 2 {
        return;
    }
    // Nothing to gain on a single-core machine (or an explicit 1-thread cap).
    if available_parallelism() <= 1 {
        return;
    }
    if CUMULATIVE_ROWS.load(Ordering::SeqCst) < warn_row_threshold() {
        return;
    }
    // Win the race to warn exactly once.
    if WARNED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        eprintln!(
            "polars-cv: cv.pipe has processed a large batch single-threaded. The \
             plugin only runs multi-core under the Polars streaming engine — the \
             default in-memory `collect` runs it on one thread. For multi-core \
             throughput use `.collect(engine=\"streaming\")` (or `scan_*` + \
             streaming). Silence this with POLARS_CV_SILENCE_ENGINE_WARNING=1."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_env_override_parses() {
        std::env::set_var("POLARS_CV_ENGINE_WARN_ROWS", "123");
        assert_eq!(warn_row_threshold(), 123);
        std::env::remove_var("POLARS_CV_ENGINE_WARN_ROWS");
        assert_eq!(warn_row_threshold(), DEFAULT_WARN_ROWS);
    }

    #[test]
    fn observed_concurrency_suppresses_warning() {
        // Two overlapping guards push the observed concurrency to >= 2, which
        // must latch the suppression path regardless of row volume.
        let a = CallGuard::enter(10);
        let b = CallGuard::enter(10);
        assert!(MAX_CONCURRENCY.load(Ordering::SeqCst) >= 2);
        drop(a);
        drop(b);
        // With concurrency observed, maybe_warn returns before warning even past
        // the threshold.
        std::env::set_var("POLARS_CV_ENGINE_WARN_ROWS", "1");
        let _c = CallGuard::enter(1_000_000);
        assert!(!WARNED.load(Ordering::SeqCst));
        std::env::remove_var("POLARS_CV_ENGINE_WARN_ROWS");
    }
}
