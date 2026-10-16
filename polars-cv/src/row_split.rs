//! A plugin call's rows, split over the plugin's thread pool.
//!
//! Rows are independent, so a call can be split into contiguous row ranges
//! that run on several threads and come back in row order. Without this a
//! call used one core however many rows it held, so the in-memory engine was
//! single-threaded on a single-chunk frame (CR-32). The pool is the plugin's
//! own copy of polars' `THREAD_POOL` (a plugin links its own polars-core, so
//! it cannot join the host's), sized by `POLARS_MAX_THREADS`.
//!
//! **How many threads a call gets is decided by what runs now, and nothing
//! else.** Every thread running a plugin call's rows counts against one
//! plugin-wide budget, the pool's size ([`BUSY`]): the thread that made the
//! call, and each pool thread helping it. A call always runs its rows on its
//! own thread, one range at a time, and before each range invites idle pool
//! threads to help while the count is under budget. A helper leaves after
//! any range that finds the count over budget. So:
//!
//! - a call running alone (the in-memory engine's one call per query) uses
//!   the whole pool;
//! - in the streaming engine's steady state, as many concurrent calls (one
//!   per morsel) as threads, each call keeps its rows on its own thread. The
//!   cores are already busy, and helping would only move every row's buffers
//!   between threads (measured: up to half a byte-heavy streaming query's
//!   throughput);
//! - a large call beside small ones (a big row group's morsel next to a
//!   small one's) takes up the threads the small ones free as they finish.
//!
//! No state outlives a call. The overlap *history* this replaced (whether
//! the last call of a graph had overlapped another) let one query decide how
//! the next ran: an eager query after a streaming run of the same pipeline
//! ran on one thread, 3.9x slower.
//!
//! **The one row splitter of the plugin**: the pipeline executor
//! (`graph::compiled`) and the geometry accessors (`geom_params`) both run
//! their rows through [`run_split`], or [`split`] when a call splits more
//! than one phase (running the rows, then filling the output column).

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use pyo3_polars::export::polars_core::runtime::THREAD_POOL;

/// Threads running a plugin call's rows now, callers and helpers alike: the
/// one count every call's split reads. Plugin-wide, because every call
/// shares the one pool.
static BUSY: AtomicUsize = AtomicUsize::new(0);

/// Whether calls split their rows at all (`polars_cv.set_row_parallelism`).
/// Off, every call runs its rows on the thread polars called it on, as one
/// range: the streaming engine's morsels are then the only parallelism, and
/// the plugin's pool is never woken.
static ENABLED: AtomicBool = AtomicBool::new(true);

/// Turn row splitting on or off for every later call; returns the previous
/// setting.
pub(crate) fn set_enabled(enabled: bool) -> bool {
    ENABLED.swap(enabled, Ordering::SeqCst)
}

/// See [`ENABLED`].
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

/// Exclusive use of the plugin's split budget, for a test whose assertion
/// depends on [`BUSY`]: how many threads a call runs on. `BUSY` is
/// plugin-wide, and the test binary runs tests on several threads, so
/// another test's call would take part of the budget. While the guard is
/// held, a call from any other thread waits in [`split`] until it is dropped
/// (threads the test starts itself are let through with
/// [`exclusive::admit_current_thread`]), and the guard is only returned once
/// the calls already running have finished.
#[cfg(test)]
pub(crate) fn exclusive_pool() -> exclusive::Guard {
    exclusive::acquire()
}

/// The test-only gate behind [`exclusive_pool`].
#[cfg(test)]
pub(crate) mod exclusive {
    use std::collections::HashSet;
    use std::sync::{Condvar, Mutex, MutexGuard};
    use std::thread::ThreadId;

    /// Serialises the exclusive tests themselves.
    static OWNER: Mutex<()> = Mutex::new(());
    /// The admitted threads while a test holds the pool.
    static ADMITTED: Mutex<Option<HashSet<ThreadId>>> = Mutex::new(None);
    static RELEASED: Condvar = Condvar::new();

    pub(crate) struct Guard {
        _owner: MutexGuard<'static, ()>,
    }

    fn admitted() -> MutexGuard<'static, Option<HashSet<ThreadId>>> {
        ADMITTED.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(super) fn acquire() -> Guard {
        let owner = OWNER.lock().unwrap_or_else(|p| p.into_inner());
        *admitted() = Some(HashSet::from([std::thread::current().id()]));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while super::BUSY.load(std::sync::atomic::Ordering::SeqCst) != 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "other tests' calls never finished"
            );
            std::thread::yield_now();
        }
        Guard { _owner: owner }
    }

    /// Let the current thread's calls through while a test holds the pool.
    pub(crate) fn admit_current_thread() {
        if let Some(set) = admitted().as_mut() {
            set.insert(std::thread::current().id());
        }
    }

    /// Wait while another test holds the pool and has not admitted this
    /// thread.
    pub(super) fn gate() {
        let me = std::thread::current().id();
        drop(
            RELEASED
                .wait_while(admitted(), |a| a.as_ref().is_some_and(|a| !a.contains(&me)))
                .unwrap_or_else(|p| p.into_inner()),
        );
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            *admitted() = None;
            RELEASED.notify_all();
        }
    }
}

/// Run `run(range_idx, rows)` over `0..len` in contiguous row ranges and
/// return each range's result in row order: [`split`] then [`Split::run`].
pub(crate) fn run_split<R: Send>(
    len: usize,
    run: impl Fn(usize, Range<usize>) -> R + Sync,
) -> Vec<R> {
    split().run(len, run)
}

/// Start a call: its thread counts as busy until the returned [`Split`] is
/// dropped. Hold it for every phase of the call that splits its rows.
pub(crate) fn split() -> Split {
    #[cfg(test)]
    exclusive::gate();
    BUSY.fetch_add(1, Ordering::SeqCst);
    Split { _caller: Slot }
}

/// One running call, whose own thread holds a slot of the budget.
pub(crate) struct Split {
    _caller: Slot,
}

/// One thread's place in [`BUSY`], given back on drop (a panic included).
struct Slot;

impl Slot {
    /// A place for a helper, if the pool has an idle thread.
    fn claim() -> Option<Slot> {
        let threads = THREAD_POOL.current_num_threads();
        let mut busy = BUSY.load(Ordering::SeqCst);
        while busy < threads {
            match BUSY.compare_exchange_weak(busy, busy + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return Some(Slot),
                Err(now) => busy = now,
            }
        }
        None
    }

    /// Whether more threads run rows than the pool has, so a helper should
    /// leave. A caller is never refused, so the count can exceed it.
    fn over_budget() -> bool {
        BUSY.load(Ordering::SeqCst) > THREAD_POOL.current_num_threads()
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        BUSY.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Split {
    /// Run `run(range_idx, rows)` over `0..len` in contiguous row ranges and
    /// return each range's result in row order.
    ///
    /// The ranges are taken in order from a shared counter: by this thread,
    /// and by the idle pool threads it invites before each range (see the
    /// module docs). A range that panics propagates the panic here once the
    /// helpers in flight have finished.
    pub(crate) fn run<R: Send>(
        &self,
        len: usize,
        run: impl Fn(usize, Range<usize>) -> R + Sync,
    ) -> Vec<R> {
        MAX_SPLIT_ROWS.fetch_max(len, Ordering::SeqCst);
        let threads = if enabled() {
            THREAD_POOL.current_num_threads()
        } else {
            1
        };
        let ranges = row_ranges(len, threads);
        if let [only] = ranges.as_slice() {
            let out = run(0, only.clone());
            LAST_SPLIT_WORKERS.store(1, Ordering::SeqCst);
            return vec![out];
        }
        let call = Call {
            ranges: &ranges,
            run: &run,
            started: Instant::now(),
            next: AtomicUsize::new(0),
            spent: AtomicU64::new(0),
            ran: AtomicUsize::new(0),
            outcomes: ranges.iter().map(|_| Mutex::new(None)).collect(),
            helpers: AtomicUsize::new(0),
            workers: AtomicUsize::new(1),
        };
        THREAD_POOL.in_place_scope(|scope| loop {
            call.invite(scope);
            if !call.take() {
                break;
            }
        });
        LAST_SPLIT_WORKERS.store(call.workers.into_inner(), Ordering::SeqCst);
        call.outcomes
            .into_iter()
            .map(|o| {
                o.into_inner()
                    .unwrap_or_else(|p| p.into_inner())
                    .expect("every range ran inside the scope")
            })
            .collect()
    }
}

/// One call's ranges and the state its workers share.
struct Call<'a, R, F> {
    ranges: &'a [Range<usize>],
    run: &'a F,
    /// When the call started: before any range has finished, how long the
    /// first one has been running so far, a lower bound on what a range costs.
    started: Instant,
    next: AtomicUsize,
    /// Time spent in, and count of, the ranges finished so far: what a range
    /// costs, so helpers are only invited for work worth a wake-up.
    spent: AtomicU64,
    ran: AtomicUsize,
    /// A `Mutex` per range, so ranges can be filled from any thread with
    /// results that are only `Send`; each is locked twice, uncontended.
    outcomes: Vec<Mutex<Option<R>>>,
    helpers: AtomicUsize,
    /// This thread, and each helper that took a range.
    workers: AtomicUsize,
}

impl<'a, R: Send, F: Fn(usize, Range<usize>) -> R + Sync> Call<'a, R, F> {
    /// Take the next range and run it; false once none is left.
    fn take(&self) -> bool {
        let idx = self.next.fetch_add(1, Ordering::SeqCst);
        let Some(rows) = self.ranges.get(idx) else {
            return false;
        };
        let start = Instant::now();
        let out = (self.run)(idx, rows.clone());
        self.spent
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.ran.fetch_add(1, Ordering::Relaxed);
        *self.outcomes[idx].lock().unwrap_or_else(|p| p.into_inner()) = Some(out);
        true
    }

    /// Invite idle threads, as many as the work left is worth
    /// ([`helpers_worth`]), no more than there are ranges left for them
    /// beside the inviter's next one. The caller does this before each of its
    /// ranges, and so does every helper: a helper woken before any range has
    /// finished is itself evidence that a range outlasts a wake-up, and
    /// invites one more, so a call of expensive rows reaches the whole pool
    /// within a few wake-ups instead of waiting for its first range to be
    /// measured.
    fn invite<'s>(&'s self, scope: &rayon::Scope<'s>) {
        let left = self
            .ranges
            .len()
            .saturating_sub(self.next.load(Ordering::SeqCst));
        let ran = self.ran.load(Ordering::Relaxed);
        let in_flight = self.started.elapsed().as_nanos() as u64;
        let invited = self.helpers.load(Ordering::SeqCst);
        let wanted = helpers_worth(
            left,
            self.spent.load(Ordering::Relaxed),
            ran,
            in_flight,
            invited,
        );
        while self.helpers.load(Ordering::SeqCst) < wanted {
            let Some(slot) = Slot::claim() else { break };
            self.helpers.fetch_add(1, Ordering::SeqCst);
            scope.spawn(move |scope| {
                let _slot = slot;
                self.invite(scope);
                if self.take() {
                    self.workers.fetch_add(1, Ordering::SeqCst);
                    while !Slot::over_budget() {
                        self.invite(scope);
                        if !self.take() {
                            break;
                        }
                    }
                }
                self.helpers.fetch_sub(1, Ordering::SeqCst);
            });
        }
    }
}

/// The least work, in nanoseconds of measured range time, worth waking one
/// more helper for. A wake-up and the helper's search of the pool for work
/// cost tens of microseconds of a core; an idle pool thread then keeps
/// searching before it sleeps, so a call that invites the whole pool for
/// microseconds of rows keeps every pool thread spinning (on 72 hyperthreads,
/// 50x the wall time and 300x the CPU of 16).
const HELPER_MIN_NS: u64 = 200_000;

/// How many helpers a call with `left` ranges still to take should have,
/// given `ran` finished ranges took `spent` ns, the call started `in_flight`
/// ns ago, and `invited` helpers have been invited so far: one per
/// [`HELPER_MIN_NS`] of the estimated remaining work, never more than the
/// ranges beside the inviter's next one.
///
/// Once a range has finished, a range costs their measured average. Before
/// that, the first range has been running for `in_flight` ns and is not
/// done, so a range costs at least that, and each invite asks for one helper
/// more than were invited: the caller's first wakes one, and every helper
/// that wakes before any range has finished wakes one more, so a call of
/// expensive rows reaches the pool one wake-up at a time instead of waiting
/// on its first range. That chain does not depend on how fast the host wakes
/// a thread (a few µs on macOS, where `in_flight` alone never justified a
/// second helper). Cheap rows finish their first range before the first
/// helper wakes, so the measured average is all they are ever judged by.
fn helpers_worth(left: usize, spent: u64, ran: usize, in_flight: u64, invited: usize) -> usize {
    let most = left.saturating_sub(1);
    let (per_range, floor) = match ran {
        0 => (in_flight, invited + 1),
        _ => (spent / ran as u64, 0),
    };
    let remaining = per_range.saturating_mul(left as u64);
    most.min(((remaining / HELPER_MIN_NS) as usize).max(floor))
}

/// Contiguous row ranges covering `0..len`, for `threads` workers.
///
/// A few ranges per thread so an expensive stretch of rows does not leave
/// the other threads idle, and so a thread freed part-way through a call
/// still finds ranges to take; one range when there is nothing to split.
pub(crate) fn row_ranges(len: usize, threads: usize) -> Vec<Range<usize>> {
    const RANGES_PER_THREAD: usize = 4;
    // One worker gains nothing from ranges: they would only hand the rows to
    // another thread while this one waits.
    if threads < 2 {
        return std::iter::once(0..len).collect();
    }
    let count = (threads * RANGES_PER_THREAD).clamp(1, len.max(1));
    let (base, extra) = (len / count, len % count);
    let mut start = 0;
    (0..count)
        .map(|i| {
            let end = start + base + usize::from(i < extra);
            let range = start..end;
            start = end;
            range
        })
        .collect()
}

/// How many workers ran the ranges of the most recent split: its calling
/// thread plus each pool thread that helped. Read through
/// `_lib._last_split_workers` by a Python test that checks a call's
/// parallelism at the user-facing entry point, where no Rust test hook
/// reaches. "Most recent" is racy under concurrent calls; it is meant for a
/// test that runs one query at a time.
static LAST_SPLIT_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// See [`LAST_SPLIT_WORKERS`].
pub(crate) fn last_split_workers() -> usize {
    LAST_SPLIT_WORKERS.load(Ordering::SeqCst)
}

/// The most rows one call ran since [`take_max_split_rows`] was last read:
/// under the streaming engine, the largest morsel a plugin call received.
/// Read through `_lib._take_max_split_rows` by the test that holds the
/// streaming guide's claims about morsel size to the plugin.
static MAX_SPLIT_ROWS: AtomicUsize = AtomicUsize::new(0);

/// See [`MAX_SPLIT_ROWS`]; reading it starts a new count.
pub(crate) fn take_max_split_rows() -> usize {
    MAX_SPLIT_ROWS.swap(0, Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::{helpers_worth, row_ranges, HELPER_MIN_NS};

    /// Cheap rows wake at most one helper; the rest come only for measured
    /// work worth a wake-up each, and never more than the ranges left.
    #[test]
    fn helpers_are_invited_for_measured_work_only() {
        assert_eq!(
            helpers_worth(288, 0, 0, 0, 0),
            1,
            "one helper before any range ran"
        );
        assert_eq!(
            helpers_worth(1, 0, 0, 0, 0),
            0,
            "no helper for the caller's last range"
        );
        // 1 µs ranges: 287 left is 287 µs of work, worth one helper.
        assert_eq!(helpers_worth(287, 1_000, 1, 0, 0), 1);
        // 1 ms ranges: each is worth a helper, capped by the ranges left.
        assert_eq!(helpers_worth(10, 1_000_000, 1, 0, 0), 9);
        assert_eq!(helpers_worth(100, 10 * HELPER_MIN_NS, 10, 0, 0), 99);
    }

    /// Before any range has finished, the time the first has been running is
    /// a lower bound on a range's cost; once one has, only the measurement
    /// counts, so cheap rows are never judged by the time since the call
    /// started.
    #[test]
    fn an_unfinished_first_range_bounds_the_cost_from_below() {
        // 15 ranges left, 50 µs in: at least 750 µs of work, three helpers.
        assert_eq!(helpers_worth(15, 0, 0, 50_000, 0), 3);
        // A first range running for a second is worth the whole pool.
        assert_eq!(helpers_worth(15, 0, 0, 1_000_000_000, 0), 14);
        // Measured 1 µs ranges: one helper, however long the call has run.
        assert_eq!(helpers_worth(287, 1_000, 1, 1_000_000_000, 0), 1);
    }

    /// A helper woken before any range has finished is itself evidence that
    /// a range outlasts a wake-up, so it invites one more, however short the
    /// call has run: on a host that wakes threads in a few µs (the 3-core
    /// macOS runner) the time since the call started never justified a
    /// second helper, and the call ran on two threads until its first range
    /// finished.
    #[test]
    fn a_helper_woken_before_any_range_finished_invites_one_more() {
        // Woken 5 µs in: 11 ranges at >= 5 µs is not worth a second helper
        // by itself, but the wake-up is.
        assert_eq!(helpers_worth(11, 0, 0, 5_000, 1), 2);
        assert_eq!(helpers_worth(11, 0, 0, 5_000, 2), 3);
        // Never more than the ranges beside the inviter's next one.
        assert_eq!(helpers_worth(3, 0, 0, 0, 2), 2);
        // Once a range has finished, only the measurement counts.
        assert_eq!(helpers_worth(287, 1_000, 1, 0, 5), 1);
    }

    /// Expensive rows reach the whole pool before the first range finishes:
    /// a call does not wait for its first range to be measured to share
    /// them. With one helper until then, a call of 3 ms image rows ran on
    /// two threads of four for its first ~second (w1 of `benchmarks/`:
    /// 10-15% slower at 4 threads than when it invited the pool at once).
    #[test]
    fn expensive_rows_reach_the_pool_before_any_range_finishes() {
        use std::collections::HashSet;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;
        use std::time::Duration;

        use pyo3_polars::export::polars_core::runtime::THREAD_POOL;

        let _pool = super::exclusive_pool();
        let threads = THREAD_POOL.current_num_threads();
        let finished = AtomicUsize::new(0);
        let early = Mutex::new(HashSet::new());
        super::run_split(threads * 4, |_, _| {
            if finished.load(Ordering::SeqCst) == 0 {
                early.lock().unwrap().insert(std::thread::current().id());
            }
            std::thread::sleep(Duration::from_millis(100));
            finished.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(
            early.into_inner().unwrap().len(),
            threads,
            "threads that started a range before any range finished"
        );
    }

    /// One worker has nothing to split across: the rows run on the caller.
    #[test]
    fn a_single_worker_takes_the_whole_call() {
        assert_eq!(
            row_ranges(300, 1),
            std::iter::once(0..300).collect::<Vec<_>>()
        );
        assert_eq!(row_ranges(0, 1), std::iter::once(0..0).collect::<Vec<_>>());
        assert_eq!(row_ranges(300, 2).len(), 8);
    }
}
