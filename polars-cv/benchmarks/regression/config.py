"""Fixed configuration for the regression suite — the single source of truth.

The matrix is intentionally small (one size, one count) so a full
single_ops + pipelines + e2e run finishes in a few minutes and stays stable.
The point of this suite is *comparability between two runs*, not absolute
coverage; the broader sweeps live in ``benchmarks.run_benchmarks``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Exact registration names from benchmarks.frameworks.get_adapter. The
# regression gate compares polars-cv against itself (before vs after), so we
# only ever run these two adapters — never the external frameworks.
POLARS_CV_ADAPTERS: list[str] = ["polars-cv-eager", "polars-cv-streaming"]

#: Results-file format version.
#:
#: v1 was a bare JSON array with a `.meta.json` sidecar that `compare.py` never
#: read. v2 is an envelope — `{"schema_version", "meta", "results"}` — so a
#: results file cannot exist without the provenance needed to decide whether it
#: may be compared to another. The committed `reports/**` baselines stay v1 and
#: stay readable as archives; `compare.py` refuses to compare across versions
#: rather than coercing, because runs from machines that no longer exist are
#: not comparable to new ones regardless of format.
SCHEMA_VERSION = 2

# All scenarios the suite knows how to run. "zero_copy" and "remote" are opt-in
# (each has its own matrix and its own result shape); the others share the
# (counts, sizes, warmup, iterations) signature.
#
# "remote" measures the `file_path` fetch path — the stage every `s3://`,
# `gs://`, `az://` and `http://` source goes through — against a loopback HTTP
# server. Nothing else in the suite touches it: every other scenario is handed
# bytes that are already in memory.
ALL_SCENARIOS: tuple[str, ...] = (
    "single_ops",
    "pipelines",
    "e2e",
    "zero_copy",
    "remote",
)
# Default to pipelines only: they exercise the full decode -> multi-op -> encode
# hot path across light/medium/heavy/imagenet/medical configs, run in ~3.5
# min/run, and were measured all-NEUTRAL on a same-binary self-check at the
# default count (see README). single_ops / e2e are opt-in via --scenarios for
# broader across-the-board coverage (slower).
DEFAULT_SCENARIOS: tuple[str, ...] = ("pipelines",)


#: Thread specification meaning "whatever the machine offers".
AUTO = "auto"


@dataclass(frozen=True)
class Cell:
    """One (engine, thread-count) point of the matrix.

    Each runs in its own process — `POLARS_MAX_THREADS` sizes the pool at the
    first polars import and is inert afterwards, so one interpreter cannot host
    two thread counts. See `benchmarks/regression/cell.py`.
    """

    engine: str  # "eager" | "streaming"
    threads: int | str  # an int, or AUTO

    @property
    def name(self) -> str:
        return f"{self.engine}@{self.threads}"


#: The matrix. Each cell catches something no other cell can.
#:
#: The plugin does not parallelise within a call — `CompiledGraph::execute` is
#: a sequential `for row_idx in 0..len` loop with no rayon anywhere — so all
#: multi-core execution comes from the streaming engine's morsel concurrency.
#: That fact is what selects these three, and what excludes a fourth:
#:
#: - eager@1      the row loop, per-row re-planning, decode/encode, and the
#:                buffering of every row result before the column is built.
#: - streaming@1  per-call fixed cost isolated from parallelism: the
#:                compiled-graph cache lookup and the per-morsel source
#:                resolution, which eager amortises over a whole column.
#: - streaming@N  morsel scaling, and the only cell that runs the concurrent
#:                path through the global compiled-graph cache Mutex.
#:
#: **eager@N is deliberately absent**: with a sequential row loop it is
#: byte-for-byte eager@1, so running it would buy a duplicate.
DEFAULT_CELLS: tuple[Cell, ...] = (
    Cell("eager", 1),
    Cell("streaming", 1),
    Cell("streaming", AUTO),
)


@dataclass(frozen=True)
class SuiteConfig:
    """A frozen, reproducible benchmark matrix.

    Defaults were chosen empirically: at image_count=300 a same-binary
    self-check is all-NEUTRAL (streaming noise <=3%, eager <=5%), whereas at
    count=50 streaming jitter reached ~12% (only ~5 morsels) and produced false
    regressions. Larger counts amortize fixed per-call overhead and stabilize
    the streaming engine — so 300 is the floor for a trustworthy gate.
    """

    image_counts: list[int] = field(default_factory=lambda: [300])
    image_sizes: list[tuple[int, int]] = field(default_factory=lambda: [(256, 256)])
    warmup_iterations: int = 3
    benchmark_iterations: int = 10
    # Whole-suite repeats, kept as the coarse noise-rejection lever for a local
    # before/after comparison. The paired A/B driver forces this to 1 and uses
    # interleaved rounds instead, which is a stronger design on a shared runner.
    suite_repeats: int = 3
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    cells: tuple[Cell, ...] = DEFAULT_CELLS


DEFAULT = SuiteConfig()


@dataclass(frozen=True)
class Thresholds:
    """Percent-change bands for classifying a result as improved/regressed.

    Throughput is the primary gate metric, latency is the cross-check. Memory
    is whole-process peak RSS (inherently noisy) so it gets a wider band and is
    advisory by default — see ``compare.py``.
    """

    # 7% sits above the measured same-binary noise floor (<=5% at count=300)
    # with margin, while still catching regressions/improvements of ~Phase-1
    # magnitude. Throughput is the only gated metric (latency is its reciprocal;
    # memory is advisory) — see compare.py.
    throughput_pct: float = 7.0
    latency_pct: float = 7.0
    memory_pct: float = 20.0


DEFAULT_THRESHOLDS = Thresholds()
