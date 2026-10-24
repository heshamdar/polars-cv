//! A plugin call's rows, split over the plugin's thread pool.
//!
//! Rows are independent, so a call can be split into contiguous row ranges
//! that run on several threads and come back in row order. Without this a
//! call used one core however many rows it held, so the in-memory engine was
//! single-threaded on a single-chunk frame. The pool is the plugin's
//! own copy of polars' `THREAD_POOL` (a plugin links its own polars-core, so
//! it cannot join the host's), sized by `POLARS_MAX_THREADS`.
//!
//! **How many threads a call gets is decided by what runs now, and nothing
//! else.** Every thread running a plugin call's rows counts against one
//! plugin-wide budget, the pool's size ([`BUSY`]): the thread that made the
//! call, and each pool thread helping it. A call always runs its rows on its
//! own thread, one range at a time, and before each range invites idle pool
//! threads to help while the count is under budget. A helper leaves after
//! any range that finds the count over budget.
//!
//! **How many helpers a call is worth is measured, never guessed.** A call
//! first runs a few rows on its own thread (the probe: one row, then 1, 2,
//! 4, ... until a range has run for [`PROBE_NS`]) and invites one helper per
//! [`HELPER_MIN_NS`] of the work that measurement says is left. Waking a
//! helper costs tens of microseconds of a core, and an idle pool thread
//! keeps searching for work before it sleeps, so inviting the pool for
//! microseconds of rows kept every thread of a many-core host spinning. A
//! call of cheap rows is done within its probe and wakes no one; a call of
//! expensive rows invites the whole pool after a row or two. The estimate
//! this replaced read a row's cost off how fast the host woke a thread
//! compared with how fast the first range finished, which differed between
//! Linux and macOS. So:
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
//! (`graph::compiled`) and the geometry accessors (`geometry::params`) both run
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

/// Test instrumentation for "a call spreads its rows": the threads that ran
/// a call's rows, and, when armed, a wait that a spreading call always meets
/// and a sequential one never does, so the assertion does not depend on how
/// fast the host wakes a thread.
///
/// The call's first row runs inside the probe, on the caller, before any
/// helper exists, so it cannot meet another thread. It only takes long
/// enough for the probe to measure it alone ([`FIRST_ROW_DECIDES_NS`]),
/// which invites helpers for the rest. The second row runs after that
/// invitation and waits (bounded) for a second thread. Waiting in the first
/// row instead always timed out, and the remaining rows, each microseconds,
/// then raced the helpers' wake-up: on a slow-waking host the caller ran
/// them all.
#[cfg(test)]
pub(crate) struct Rendezvous {
    seen: Mutex<std::collections::HashSet<std::thread::ThreadId>>,
    arrived: std::sync::Condvar,
    armed: AtomicBool,
    visits: AtomicUsize,
}

#[cfg(test)]
impl Rendezvous {
    /// See [`Rendezvous::reset`].
    pub(crate) fn new(armed: bool) -> Self {
        Rendezvous {
            seen: Mutex::default(),
            arrived: std::sync::Condvar::new(),
            armed: AtomicBool::new(armed),
            visits: AtomicUsize::new(0),
        }
    }

    /// Forget the threads seen; the next call waits for a second thread iff
    /// `armed`.
    pub(crate) fn reset(&self, armed: bool) {
        self.seen.lock().unwrap().clear();
        self.visits.store(0, Ordering::SeqCst);
        self.armed.store(armed, Ordering::SeqCst);
    }

    /// Record a row run on this thread; see the type's docs.
    pub(crate) fn visit(&self) {
        let mut seen = self.seen.lock().unwrap();
        seen.insert(std::thread::current().id());
        self.arrived.notify_all();
        if !self.armed.load(Ordering::SeqCst) {
            return;
        }
        match self.visits.fetch_add(1, Ordering::SeqCst) {
            0 => {
                drop(seen);
                std::thread::sleep(std::time::Duration::from_nanos(2 * FIRST_ROW_DECIDES_NS));
            }
            1 => {
                self.armed.store(false, Ordering::SeqCst);
                let _ = self
                    .arrived
                    .wait_timeout_while(seen, std::time::Duration::from_secs(5), |s| s.len() < 2)
                    .unwrap();
            }
            _ => {}
        }
    }

    pub(crate) fn threads(&self) -> usize {
        self.seen.lock().unwrap().len()
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
    /// This thread first runs the probe's ranges ([`Probe`]); the rest are
    /// taken in order from a shared counter, by this thread and by the idle
    /// pool threads the probe's measurement is worth (see the module docs).
    /// A range that panics propagates the panic here once the helpers in
    /// flight have finished.
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
        if threads < 2 || len < 2 {
            // One worker gains nothing from ranges: they would only hand the
            // rows to another thread while this one waits.
            let out = run(0, 0..len);
            LAST_SPLIT_WORKERS.store(1, Ordering::SeqCst);
            return vec![out];
        }
        // The probe: this thread's first ranges, timed.
        let mut outcomes = Vec::new();
        let mut probe = Probe::new();
        let mut start = 0;
        while let Some(n) = probe.next_len(len - start) {
            let began = Instant::now();
            outcomes.push(run(outcomes.len(), start..start + n));
            probe.record(n, began.elapsed().as_nanos() as u64);
            start += n;
        }
        let Some((spent, measured_rows)) = probe.measured() else {
            // Every row ran in the probe.
            LAST_SPLIT_WORKERS.store(1, Ordering::SeqCst);
            return outcomes;
        };
        let ranges: Vec<Range<usize>> = row_ranges(len - start, threads)
            .into_iter()
            .map(|r| r.start + start..r.end + start)
            .collect();
        let call = Call {
            ranges: &ranges,
            first_idx: outcomes.len(),
            end: len,
            run: &run,
            next: AtomicUsize::new(0),
            spent: AtomicU64::new(spent),
            rows_ran: AtomicUsize::new(measured_rows),
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
        outcomes.extend(call.outcomes.into_iter().map(|o| {
            o.into_inner()
                .unwrap_or_else(|p| p.into_inner())
                .expect("every range ran inside the scope")
        }));
        outcomes
    }
}

/// The least time, in nanoseconds, a probe range must run for its rows to
/// be measured: well above the timer's resolution and a range's fixed cost,
/// and short enough that a call of expensive rows waits on no more than a
/// few of them (the probe's ranges double, so its rows run alone for at most
/// about twice this, or for two rows when one already takes longer).
const PROBE_NS: u64 = 50_000;

/// A first row that takes this long is measured by itself. Otherwise the
/// first row is not measured at all: it pays a call's one-off costs (the
/// plan's first use of a source layout, a first allocation), and a first
/// row of 60 µs in a call of 1 µs rows would invite the pool for nothing.
/// No one-off cost is a millisecond of a core.
const FIRST_ROW_DECIDES_NS: u64 = 1_000_000;

/// A call's first ranges, run on its own thread to measure what a row costs
/// before any helper is woken: one row, then 1, 2, 4, ... rows, until a
/// range other than the first has run for [`PROBE_NS`] (or the first for
/// [`FIRST_ROW_DECIDES_NS`]). Its measurement is that range's rows and time.
struct Probe {
    ranges: usize,
    next: usize,
    measured: Option<(u64, usize)>,
}

impl Probe {
    fn new() -> Self {
        Probe {
            ranges: 0,
            next: 1,
            measured: None,
        }
    }

    /// The rows of the next probe range, given `rows_left` not yet run;
    /// `None` once the probe has measured them, or none are left.
    fn next_len(&self, rows_left: usize) -> Option<usize> {
        (self.measured.is_none() && rows_left > 0).then(|| self.next.min(rows_left))
    }

    /// The probe range just run took `ns` for its `rows`.
    fn record(&mut self, rows: usize, ns: u64) {
        let enough = match self.ranges {
            0 => FIRST_ROW_DECIDES_NS,
            _ => PROBE_NS,
        };
        if ns >= enough {
            self.measured = Some((ns, rows));
        }
        self.next = if self.ranges == 0 { 1 } else { rows * 2 };
        self.ranges += 1;
    }

    /// The deciding range's time and rows, once the probe has one.
    fn measured(&self) -> Option<(u64, usize)> {
        self.measured
    }
}

/// The ranges a call has left after its probe, and the state its workers
/// share.
struct Call<'a, R, F> {
    ranges: &'a [Range<usize>],
    /// The index the first of `ranges` runs as: the probe's ranges come
    /// first, and indices follow the rows.
    first_idx: usize,
    /// The end of the last range: the ranges are contiguous.
    end: usize,
    run: &'a F,
    next: AtomicUsize,
    /// Time spent in, and rows of, the measured ranges so far (the probe's
    /// deciding range, then every range taken here): what a row costs, so
    /// helpers are only invited for work worth a wake-up.
    spent: AtomicU64,
    rows_ran: AtomicUsize,
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
        let out = (self.run)(self.first_idx + idx, rows.clone());
        self.spent
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.rows_ran.fetch_add(rows.len(), Ordering::Relaxed);
        *self.outcomes[idx].lock().unwrap_or_else(|p| p.into_inner()) = Some(out);
        true
    }

    /// Invite idle threads, as many as the work left is worth
    /// ([`helpers_worth`]). The caller does this before each of its ranges,
    /// and so does every helper, so threads another call frees part-way
    /// through are taken up; the first, right after the probe, is the one
    /// that invites for the call's measured cost.
    fn invite<'s>(&'s self, scope: &rayon::Scope<'s>) {
        let taken = self.next.load(Ordering::SeqCst).min(self.ranges.len());
        let left = &self.ranges[taken..];
        let rows_left = left.first().map_or(0, |r| self.end - r.start);
        let wanted = helpers_worth(
            rows_left,
            left.len(),
            self.spent.load(Ordering::Relaxed),
            self.rows_ran.load(Ordering::Relaxed),
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

/// How many helpers a call with `rows_left` rows in `ranges_left` ranges
/// still to take should have, given its measured rows took `spent` ns over
/// `rows_ran` rows: one per [`HELPER_MIN_NS`] of the work left, never more
/// than the ranges beside the inviter's next one.
fn helpers_worth(rows_left: usize, ranges_left: usize, spent: u64, rows_ran: usize) -> usize {
    let per_row = spent / rows_ran.max(1) as u64;
    let remaining = per_row.saturating_mul(rows_left as u64);
    ranges_left
        .saturating_sub(1)
        .min((remaining / HELPER_MIN_NS) as usize)
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
    use super::{helpers_worth, row_ranges, Probe, FIRST_ROW_DECIDES_NS, HELPER_MIN_NS, PROBE_NS};

    /// Runs a probe over `len` rows of which row `i` takes `ns(i)`, and
    /// returns its range lengths and measurement.
    fn probe(len: usize, ns: impl Fn(usize) -> u64) -> (Vec<usize>, Option<(u64, usize)>) {
        let mut probe = Probe::new();
        let (mut lens, mut start) = (Vec::new(), 0);
        while let Some(n) = probe.next_len(len - start) {
            probe.record(n, (start..start + n).map(&ns).sum());
            lens.push(n);
            start += n;
        }
        (lens, probe.measured())
    }

    /// Cheap rows double the probe's ranges until one runs long enough to
    /// measure, and a call that ends first is never measured: it ran alone.
    #[test]
    fn the_probe_doubles_until_a_range_is_long_enough_to_measure() {
        assert_eq!(
            probe(1_000, |_| 1_000),
            (vec![1, 1, 2, 4, 8, 16, 32, 64], Some((64_000, 64)))
        );
        assert_eq!(probe(20, |_| 1_000), (vec![1, 1, 2, 4, 8, 4], None));
        assert_eq!(probe(1, |_| 1_000), (vec![1], None));
    }

    /// A first row is measured only when it is slow beyond any one-off cost;
    /// otherwise the probe reads the rows after it, so a call of cheap rows
    /// whose first row pays for planning is judged by its cheap rows.
    #[test]
    fn a_first_row_decides_only_when_it_is_slow_beyond_a_one_off_cost() {
        let slow = 3_000_000;
        assert_eq!(probe(100, |_| slow), (vec![1], Some((slow, 1))));
        assert_eq!(
            probe(1_000, |i| if i == 0 { 200_000 } else { 1_000 }),
            (vec![1, 1, 2, 4, 8, 16, 32, 64], Some((64_000, 64)))
        );
        // Below FIRST_ROW_DECIDES_NS but above PROBE_NS: the next row decides.
        let mid = PROBE_NS * 2;
        assert!(mid < FIRST_ROW_DECIDES_NS);
        assert_eq!(probe(100, |_| mid), (vec![1, 1], Some((mid, 1))));
    }

    /// One helper per HELPER_MIN_NS of measured work left, capped by the
    /// ranges beside the inviter's next one; none for work under that.
    #[test]
    fn helpers_are_invited_for_measured_work_only() {
        // 1 µs rows: 300 left is 300 µs of work, worth one helper.
        assert_eq!(helpers_worth(300, 16, 64_000, 64), 1);
        assert_eq!(helpers_worth(150, 16, 64_000, 64), 0);
        // 3 ms rows: each range is worth a helper, capped by the ranges.
        assert_eq!(helpers_worth(300, 16, 3_000_000, 1), 15);
        assert_eq!(helpers_worth(300, 1, 3_000_000, 1), 0);
        assert_eq!(helpers_worth(0, 0, 3_000_000, 1), 0);
        assert_eq!(helpers_worth(100, 100, 10 * HELPER_MIN_NS, 10), 99);
    }

    /// Expensive rows reach the whole pool as soon as the probe has measured
    /// them: every thread starts a range before the second range finishes.
    /// With one helper until the first range of ~300 rows finished, a call of
    /// 3 ms image rows ran on two threads of four for its first second (w1
    /// of `benchmarks/`: 10-15% slower at 4 threads).
    #[test]
    fn expensive_rows_reach_the_pool_once_the_probe_has_measured_them() {
        use std::collections::HashSet;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;
        use std::time::Duration;

        use pyo3_polars::export::polars_core::runtime::THREAD_POOL;

        let _pool = super::exclusive_pool();
        let threads = THREAD_POOL.current_num_threads();
        let finished = AtomicUsize::new(0);
        let early = Mutex::new(HashSet::new());
        super::run_split(threads * 4 + 1, |_, _| {
            if finished.load(Ordering::SeqCst) <= 1 {
                early.lock().unwrap().insert(std::thread::current().id());
            }
            std::thread::sleep(Duration::from_millis(100));
            finished.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(
            early.into_inner().unwrap().len(),
            threads,
            "threads that started a range before the second range finished"
        );
    }

    /// A call whose rows cost nothing runs on its own thread: it is done
    /// within its probe. Under the race-based estimate a call always woke a
    /// helper, and on a fast-waking host that helper took ranges. (That a
    /// measured cheap call invites no one is `helpers_worth`'s test.)
    #[test]
    fn a_call_of_cheap_rows_runs_on_its_own_thread() {
        use pyo3_polars::export::polars_core::runtime::THREAD_POOL;

        let _pool = super::exclusive_pool();
        let threads = THREAD_POOL.current_num_threads();
        if threads < 2 {
            return;
        }
        for _ in 0..20 {
            super::run_split(threads * 40, |_, rows| rows.len());
            assert_eq!(super::last_split_workers(), 1, "workers of a cheap call");
        }
    }

    /// Every row is run exactly once, in contiguous ranges whose indices
    /// increase with their rows (`graph::compiled` skips the ranges after a
    /// failing one by index), and the results come back in row order.
    #[test]
    fn every_row_runs_once_in_row_order() {
        use std::time::Duration;

        let _pool = super::exclusive_pool();
        for len in [0, 1, 2, 3, 7, 64, 100, 1_000] {
            for slow in [false, true] {
                let parts = super::run_split(len, |idx, rows| {
                    if slow {
                        std::thread::sleep(Duration::from_micros(300));
                    }
                    (idx, rows)
                });
                let mut next_row = 0;
                for (i, (idx, rows)) in parts.iter().enumerate() {
                    assert_eq!(*idx, i, "range index in row order (len {len})");
                    assert_eq!(rows.start, next_row, "contiguous (len {len})");
                    next_row = rows.end;
                }
                assert_eq!(next_row, len, "every row (len {len})");
            }
        }
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
