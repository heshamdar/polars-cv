//! One-time "this call ran on one thread" warning for the graph plugin.
//!
//! The plugin does not parallelize *within* a call. Its multi-core execution
//! comes from the engine invoking it several times at once: the streaming engine
//! once per morsel, the in-memory engine once per chunk. A single-chunk column
//! under the default in-memory engine is therefore one call on one core, and
//! nothing signals it. This module emits one actionable warning when that
//! happens *and it cost something*.
//!
//! The decision is made when a call **finishes**, from two facts about that
//! call alone (CR-32):
//! - **Elapsed time.** A call that ran longer than the threshold (default
//!   [`DEFAULT_WARN_SECONDS`]) was worth parallelizing. Time, not row count: an
//!   image row costs milliseconds, so a row threshold (the former 50 000)
//!   let a single-threaded run take tens of seconds without ever firing.
//! - **Overlap.** If any other plugin call ran at any point during this one,
//!   the engine was already running calls in parallel, and this call was not
//!   the bottleneck. This is per call: one overlap earlier in the process no
//!   longer silences the warning for good.
//!
//! Environment:
//! - `POLARS_CV_SILENCE_ENGINE_WARNING=1` — never warn.
//! - `POLARS_CV_ENGINE_WARN_SECONDS=<s>` — override the per-call time threshold.
//!   A value that is not a positive number is reported once and the default
//!   is used.
//! - `POLARS_CV_ENGINE_WARN_ROWS` — the former row threshold. It is no longer
//!   read, and setting it is reported once rather than silently ignored.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Plugin calls currently executing (RAII-tracked by [`CallGuard`]).
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);
/// Number of times a call started while another was already running. A call
/// that sees this change between its start and its end was overlapped.
static OVERLAPS: AtomicU64 = AtomicU64::new(0);
/// Whether the one-time warning has already fired.
static WARNED: AtomicBool = AtomicBool::new(false);

/// A single call running this long on one thread is worth telling the user
/// about. Chosen so interactive use (a handful of images) never trips it.
pub const DEFAULT_WARN_SECONDS: f64 = 2.0;

/// RAII guard: tracks one plugin call from start to finish, and on drop decides
/// whether that call is the one worth warning about.
pub struct CallGuard {
    start: Instant,
    overlaps_at_start: u64,
    overlapped_at_start: bool,
}

impl CallGuard {
    /// Enter a plugin call.
    pub fn enter() -> Self {
        let _ = threshold();
        let now = IN_FLIGHT.fetch_add(1, Ordering::SeqCst) + 1;
        let overlapped_at_start = now >= 2;
        if overlapped_at_start {
            OVERLAPS.fetch_add(1, Ordering::SeqCst);
        }
        CallGuard {
            start: Instant::now(),
            // Read after our own increment, so our own start is not an overlap.
            overlaps_at_start: OVERLAPS.load(Ordering::SeqCst),
            overlapped_at_start,
        }
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        let overlapped =
            self.overlapped_at_start || OVERLAPS.load(Ordering::SeqCst) != self.overlaps_at_start;
        IN_FLIGHT.fetch_sub(1, Ordering::SeqCst);
        maybe_warn(self.start.elapsed(), overlapped);
    }
}

/// The pure warning decision, factored out so it is unit-testable without
/// touching any process-global state or environment.
fn should_warn(
    already_warned: bool,
    silenced: bool,
    overlapped: bool,
    parallelism: usize,
    elapsed: Duration,
    threshold: Duration,
) -> bool {
    !already_warned
        && !silenced
        // Another call ran alongside this one: the engine was parallelizing.
        && !overlapped
        // Nothing to gain on a single-core machine (or an explicit 1-thread cap).
        && parallelism > 1
        && elapsed >= threshold
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

/// Parse a threshold setting: a positive, finite number of seconds.
fn parse_seconds(raw: &str) -> Option<Duration> {
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|s| s.is_finite() && *s > 0.0)
        .map(Duration::from_secs_f64)
}

/// The per-call time threshold, read once. Any configuration problem —
/// an unusable value, or the removed row setting — is reported here, once,
/// rather than silently ignored.
fn threshold() -> Duration {
    static THRESHOLD: OnceLock<Duration> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        if std::env::var_os("POLARS_CV_ENGINE_WARN_ROWS").is_some() {
            eprintln!(
                "polars-cv: POLARS_CV_ENGINE_WARN_ROWS is no longer read. The \
                 single-thread warning is based on how long a call runs; set \
                 POLARS_CV_ENGINE_WARN_SECONDS instead."
            );
        }
        let default = Duration::from_secs_f64(DEFAULT_WARN_SECONDS);
        match std::env::var("POLARS_CV_ENGINE_WARN_SECONDS") {
            Err(_) => default,
            Ok(raw) => parse_seconds(&raw).unwrap_or_else(|| {
                eprintln!(
                    "polars-cv: POLARS_CV_ENGINE_WARN_SECONDS={raw:?} is not a \
                     positive number of seconds; using {DEFAULT_WARN_SECONDS}."
                );
                default
            }),
        }
    })
}

fn maybe_warn(elapsed: Duration, overlapped: bool) {
    let decided = should_warn(
        WARNED.load(Ordering::Relaxed),
        std::env::var_os("POLARS_CV_SILENCE_ENGINE_WARNING").is_some(),
        overlapped,
        available_parallelism(),
        elapsed,
        threshold(),
    );
    if !decided {
        return;
    }
    // Win the race to warn exactly once.
    if WARNED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        eprintln!(
            "polars-cv: a cv.pipe call ran on one thread for {:.1}s with no other \
             call alongside it. The plugin runs multi-core when the engine calls \
             it for several batches at once. Use `.collect(engine=\"streaming\")` \
             (or `scan_*` + streaming) for multi-core throughput. Silence this with \
             POLARS_CV_SILENCE_ENGINE_WARNING=1.",
            elapsed.as_secs_f64()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_seconds, should_warn};
    use std::time::Duration;

    const T: Duration = Duration::from_secs(2);
    const LONG: Duration = Duration::from_secs(3);
    const SHORT: Duration = Duration::from_millis(100);

    #[test]
    fn warns_on_a_long_lone_call() {
        assert!(should_warn(false, false, false, 8, LONG, T));
        assert!(should_warn(false, false, false, 8, T, T));
    }

    #[test]
    fn suppressed_when_already_warned_or_silenced() {
        assert!(!should_warn(true, false, false, 8, LONG, T));
        assert!(!should_warn(false, true, false, 8, LONG, T));
    }

    #[test]
    fn suppressed_when_another_call_overlapped() {
        assert!(!should_warn(false, false, true, 8, 10 * LONG, T));
    }

    #[test]
    fn suppressed_on_single_core_or_short_call() {
        assert!(!should_warn(false, false, false, 1, 10 * LONG, T));
        assert!(!should_warn(false, false, false, 8, SHORT, T));
    }

    #[test]
    fn thresholds_must_be_positive_finite_seconds() {
        assert_eq!(parse_seconds("1.5"), Some(Duration::from_millis(1500)));
        assert_eq!(parse_seconds(" 3 "), Some(Duration::from_secs(3)));
        for bad in ["0", "-1", "abc", "inf", "NaN", ""] {
            assert_eq!(parse_seconds(bad), None, "{bad:?}");
        }
    }
}
